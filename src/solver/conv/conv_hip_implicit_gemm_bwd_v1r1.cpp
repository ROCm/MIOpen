/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <miopen/conv/invokers/impl_gemm.hpp>
#include <miopen/solver.hpp>
#include <miopen/handle.hpp>
#include <miopen/hip_build_utils.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/solver/implicitgemm_util.hpp>

#include <cstddef>

#define WORKAROUND_ISSUE_309 1

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

PerformanceImplicitGemmBwdDataV1R1::PerformanceImplicitGemmBwdDataV1R1(int BlockSize_,
                                                                       int GemmMPerBlock_,
                                                                       int GemmNPerBlock_,
                                                                       int GemmKPerBlock_,
                                                                       int GemmMPerThread_,
                                                                       int GemmNPerThread_,
                                                                       bool use_spare_set_)
    : BlockSize(BlockSize_),
      GemmMPerBlock(GemmMPerBlock_),
      GemmNPerBlock(GemmNPerBlock_),
      GemmKPerBlock(GemmKPerBlock_),
      GemmMPerThread(GemmMPerThread_),
      GemmNPerThread(GemmNPerThread_),
      use_spare_set(use_spare_set_)
{
}

PerformanceImplicitGemmBwdDataV1R1::PerformanceImplicitGemmBwdDataV1R1(bool spare)
{
    // always search full space, no matter if use_spare_set or not
    BlockSize = 64;

    GemmMPerBlock = 32;
    GemmNPerBlock = 32;
    GemmKPerBlock = 4;

    GemmMPerThread = 2;
    GemmNPerThread = 2;

    use_spare_set = spare;
}

bool PerformanceImplicitGemmBwdDataV1R1::operator==(
    const PerformanceImplicitGemmBwdDataV1R1& other) const
{
    // clang-format off
    return BlockSize == other.BlockSize
        && GemmMPerBlock == other.GemmMPerBlock
        && GemmNPerBlock == other.GemmNPerBlock
        && GemmKPerBlock == other.GemmKPerBlock
        && GemmMPerThread == other.GemmMPerThread
        && GemmNPerThread == other.GemmNPerThread
        && use_spare_set == other.use_spare_set;
    // clang-format on
}

std::tuple<int, bool>
PerformanceImplicitGemmBwdDataV1R1::CalculateGridSize(const ExecutionContext& ctx,
                                                      const ProblemDescription& problem) const
{
    int GridSize = 0;

    try
    {
        int gemm_m = 0;
        int gemm_n = 0;

        std::tie(gemm_m, gemm_n, std::ignore) =
            ConvHipImplicitGemmBwdDataV1R1::CalculateGemmSize(ctx, problem);

        if(!(gemm_m % GemmMPerBlock == 0 && gemm_n % GemmNPerBlock == 0))
            MIOPEN_THROW("invalid performance parameter");

        GridSize = (gemm_m / GemmMPerBlock) * (gemm_n / GemmNPerBlock);
    }
    catch(...)
    {
        return std::make_tuple(-1, false);
    }

    return std::make_tuple(GridSize, true);
}

std::tuple<int, int, int, int, bool>
PerformanceImplicitGemmBwdDataV1R1::CalculateBlockGemmPerformanceParameters() const
{
    int GemmMLevel0Cluster = 0;
    int GemmNLevel0Cluster = 0;
    int GemmMLevel1Cluster = 0;
    int GemmNLevel1Cluster = 0;

    try
    {
        if(BlockSize == 64)
        {
            GemmMLevel0Cluster = 4;
            GemmNLevel0Cluster = 4;
            GemmMLevel1Cluster = 2;
            GemmNLevel1Cluster = 2;
        }
        else if(BlockSize == 128)
        {
            GemmMLevel0Cluster = 4;
            GemmNLevel0Cluster = 4;
            GemmMLevel1Cluster = 4;
            GemmNLevel1Cluster = 2;
        }
        else if(BlockSize == 256)
        {
            GemmMLevel0Cluster = 4;
            GemmNLevel0Cluster = 4;
            GemmMLevel1Cluster = 4;
            GemmNLevel1Cluster = 4;
        }
        else
        {
            MIOPEN_LOG_E("BlockSize not supported");
            MIOPEN_THROW("invalid performance parameter");
        }

        if(!(GemmMPerBlock % GemmMPerThread == 0 && GemmNPerBlock % GemmNPerThread == 0))
            MIOPEN_THROW("invalid performance parameter");

        const auto thread_gemm_per_block_m = GemmMPerBlock / GemmMPerThread;
        const auto thread_gemm_per_block_n = GemmNPerBlock / GemmNPerThread;

        const auto thread_gemm_per_cluster_m = GemmMLevel0Cluster * GemmMLevel1Cluster;
        const auto thread_gemm_per_cluster_n = GemmNLevel0Cluster * GemmNLevel1Cluster;

        if(!(thread_gemm_per_block_m % thread_gemm_per_cluster_m == 0) &&
           (thread_gemm_per_block_n % thread_gemm_per_cluster_n == 0))
            MIOPEN_THROW("invalid performance parameter");

        const auto cluster_per_block_m = thread_gemm_per_block_m / thread_gemm_per_cluster_m;
        const auto cluster_per_block_n = thread_gemm_per_block_n / thread_gemm_per_cluster_n;

        // inline asm only support cluster_per_block_m = 2 andcluster_per_block_n = 2
        if(!(cluster_per_block_m == 2 && cluster_per_block_n == 2))
            MIOPEN_THROW("invalid performance parameter");
    }
    catch(...)
    {
        return std::make_tuple(-1, -1, -1, -1, false);
    }

    return std::make_tuple(
        GemmMLevel0Cluster, GemmNLevel0Cluster, GemmMLevel1Cluster, GemmNLevel1Cluster, true);
}

std::tuple<int, int, int, int, bool>
PerformanceImplicitGemmBwdDataV1R1::CalculateGemmABlockCopyPerformanceParameters(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    int ClusterLengths_GemmK      = 0;
    int ClusterLengths_GemmM      = 0;
    int SrcDataPerRead_GemmM      = amd_buffer_load_max_length<float>();
    int DstDataPerWrite_GemmM     = amd_lds_write_max_length<float>();
    int DstDataPerWrite_GemmKPACK = GetEPackLength(ctx, problem, false);

    try
    {
        // calculate vector length on gemmk dimension
        SrcDataPerRead_GemmM = gcd(SrcDataPerRead_GemmM, GemmMPerBlock);

        // calculate threadwise copy size
        const auto a_data_per_thread_copy = (GemmKPerBlock * GemmMPerBlock) / BlockSize;

        if(!(a_data_per_thread_copy > 0))
            MIOPEN_THROW("invalid performance parameter");

        // GemmABlockCopySrcDataPerRead_GemmK also bounded by size of threadwise copy
        SrcDataPerRead_GemmM = gcd(SrcDataPerRead_GemmM, a_data_per_thread_copy);

        // decide threadwise copy lengths
        const auto a_data_per_thread_copy_gemmm = SrcDataPerRead_GemmM;
        const auto a_data_per_thread_copy_gemmk =
            a_data_per_thread_copy / a_data_per_thread_copy_gemmm;

        // GemmABlockCopyDstDataPerWrite_GemmM also bounded by size of threadwise copy
        if(problem.IsFp32())
        {
            DstDataPerWrite_GemmM = gcd(DstDataPerWrite_GemmM, a_data_per_thread_copy_gemmm);
        }

        // calculate blockwise copy thread cluster lengths
        ClusterLengths_GemmK = GemmKPerBlock / a_data_per_thread_copy_gemmk;
        ClusterLengths_GemmM = GemmMPerBlock / a_data_per_thread_copy_gemmm;

        if(!(ClusterLengths_GemmK > 0 && ClusterLengths_GemmM > 0))
            MIOPEN_THROW("invalid performance parameter");
    }
    catch(...)
    {
        return std::make_tuple(-1, -1, -1, -1, false);
    }

    if(problem.IsFp32())
    {
        return std::make_tuple(ClusterLengths_GemmK,
                               ClusterLengths_GemmM,
                               SrcDataPerRead_GemmM,
                               DstDataPerWrite_GemmM,
                               true);
    }
    else
    {
        return std::make_tuple(ClusterLengths_GemmK,
                               ClusterLengths_GemmM,
                               SrcDataPerRead_GemmM,
                               DstDataPerWrite_GemmKPACK,
                               true);
    }
}

std::tuple<int, int, int, int, bool>
PerformanceImplicitGemmBwdDataV1R1::CalculateGemmBBlockCopyPerformanceParameters(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    int ClusterLengths_GemmK      = 0;
    int ClusterLengths_GemmN      = 0;
    int SrcDataPerRead_GemmN      = amd_buffer_load_max_length<float>();
    int DstDataPerWrite_GemmN     = amd_lds_write_max_length<float>();
    int DstDataPerWrite_GemmKPACK = GetEPackLength(ctx, problem, false);

    try
    {
        SrcDataPerRead_GemmN = gcd(SrcDataPerRead_GemmN, GemmNPerBlock);

        // calculate vector length on gemmn dimension
        const auto ho = ProblemInterpreter::GetOutputHeightHo(problem);
        const auto wo = ProblemInterpreter::GetOutputWidthWo(problem);

        SrcDataPerRead_GemmN = gcd(SrcDataPerRead_GemmN, ho * wo);

        // calculate threadwise copy size
        const auto b_data_per_thread_copy = (GemmKPerBlock * GemmNPerBlock) / BlockSize;

        if(!(b_data_per_thread_copy > 0))
            MIOPEN_THROW("invalid performance parameter");

        // GemmBBlockCopySrcDataPerRead_GemmN also bounded by size of threadwise copy
        SrcDataPerRead_GemmN = gcd(SrcDataPerRead_GemmN, b_data_per_thread_copy);

        const auto b_data_per_thread_copy_gemmn = SrcDataPerRead_GemmN;
        const auto b_data_per_thread_copy_gemmk =
            b_data_per_thread_copy / b_data_per_thread_copy_gemmn;

        // GemmBBlockCopyDstDataPerWrite_GemmN also bounded by size of threadwise copy
        if(problem.IsFp32())
        {
            DstDataPerWrite_GemmN = gcd(DstDataPerWrite_GemmN, b_data_per_thread_copy_gemmn);
        }

        // calculate blockwise copy thread cluster lengths
        ClusterLengths_GemmK = GemmKPerBlock / b_data_per_thread_copy_gemmk;
        ClusterLengths_GemmN = GemmNPerBlock / b_data_per_thread_copy_gemmn;

        if(!(ClusterLengths_GemmK > 0 && ClusterLengths_GemmN > 0))
            MIOPEN_THROW("invalid performance parameter");
    }
    catch(...)
    {
        MIOPEN_LOG_I("catch");
        return std::make_tuple(-1, -1, -1, -1, false);
    }

    if(problem.IsFp32())
    {
        return std::make_tuple(ClusterLengths_GemmK,
                               ClusterLengths_GemmN,
                               SrcDataPerRead_GemmN,
                               DstDataPerWrite_GemmN,
                               true);
    }
    else
    {
        return std::make_tuple(ClusterLengths_GemmK,
                               ClusterLengths_GemmN,
                               SrcDataPerRead_GemmN,
                               DstDataPerWrite_GemmKPACK,
                               true);
    }
}

std::tuple<int, bool>
PerformanceImplicitGemmBwdDataV1R1::CalculateGemmCThreadCopyPerformanceParameters(
    const ProblemDescription& problem) const
{
    int DstDataPerWrite_GemmN1 = amd_buffer_store_max_length<float>();

    try
    {
        // GemmCThreadCopyDstDataPerWrite_GemmN1 bounded by size of threadwise GEMM
        DstDataPerWrite_GemmN1 = gcd(DstDataPerWrite_GemmN1, GemmNPerThread);

        // GemmCThreadCopyDstDataPerWrite_GemmN1 limited by global memory layout of input tensor
        // calculate vector length on gemmn dimension
        const auto y               = ProblemInterpreter::GetFilterHeightY(problem);
        const auto x               = ProblemInterpreter::GetFilterWidthX(problem);
        const auto hi              = ProblemInterpreter::GetInputHeightHi(problem);
        const auto wi              = ProblemInterpreter::GetInputWidthWi(problem);
        const auto conv_stride_h   = ProblemInterpreter::GetAdjustedConvolutionStrideH(problem);
        const auto conv_stride_w   = ProblemInterpreter::GetAdjustedConvolutionStrideW(problem);
        const auto conv_dilation_w = ProblemInterpreter::GetAdjustedConvolutionDilationW(problem);
        const auto in_left_pad_h   = ProblemInterpreter::GetInputLeftPadH(problem);
        const auto in_left_pad_w   = ProblemInterpreter::GetInputLeftPadW(problem);
        const auto in_right_pad_h  = ProblemInterpreter::GetAdjustedInputRightPadH(problem);
        const auto in_right_pad_w  = ProblemInterpreter::GetAdjustedInputRightPadW(problem);

        if(problem.Is3d())
        {
            const auto di             = ProblemInterpreter::GetInputDepthDi(problem);
            const auto z              = ProblemInterpreter::GetFilterDepthZ(problem);
            const auto conv_stride_d  = ProblemInterpreter::GetAdjustedConvolutionStrideD(problem);
            const auto in_right_pad_d = ProblemInterpreter::GetAdjustedInputRightPadD(problem);
            const auto in_left_pad_d  = ProblemInterpreter::GetInputLeftPadD(problem);

            if(z == 1 && y == 1 && x == 1 && conv_stride_d == 1 && conv_stride_h == 1 &&
               conv_stride_w == 1 && in_left_pad_d == 0 && in_left_pad_h == 0 &&
               in_left_pad_w == 0 && in_right_pad_d == 0 && in_right_pad_h == 0 &&
               in_right_pad_w == 0)
            {
                // \todo there are more configs that can go through this if branch
                DstDataPerWrite_GemmN1 = gcd(DstDataPerWrite_GemmN1, di * hi * wi);
            }
            else if(conv_stride_w == 1)
            {
                DstDataPerWrite_GemmN1 =
                    gcd(DstDataPerWrite_GemmN1, in_left_pad_w, wi, in_right_pad_w, conv_dilation_w);
            }
            else
            {
                DstDataPerWrite_GemmN1 = 1;
            }
        }
        else
        {
            if(y == 1 && x == 1 && conv_stride_h == 1 && conv_stride_w == 1 && in_left_pad_h == 0 &&
               in_left_pad_w == 0 && in_right_pad_h == 0 && in_right_pad_w == 0)
            {
                // \todo there are more configs that can go through this if branch
                DstDataPerWrite_GemmN1 = gcd(DstDataPerWrite_GemmN1, hi * wi);
            }
            else if(conv_stride_w == 1)
            {
                DstDataPerWrite_GemmN1 =
                    gcd(DstDataPerWrite_GemmN1, in_left_pad_w, wi, in_right_pad_w, conv_dilation_w);
            }
            else
            {
                DstDataPerWrite_GemmN1 = 1;
            }
        }
    }
    catch(...)
    {
        return std::make_tuple(-1, false);
    }

    return std::make_tuple(DstDataPerWrite_GemmN1, true);
}

std::tuple<std::size_t, bool> PerformanceImplicitGemmBwdDataV1R1::CalculateLdsNumberOfByte(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    std::size_t lds_size = 0;

    try
    {
        bool valid = false;

        int GemmABlockCopyDescDataPerWriteGemm = 0;
        std::tie(std::ignore, std::ignore, std::ignore, GemmABlockCopyDescDataPerWriteGemm, valid) =
            CalculateGemmABlockCopyPerformanceParameters(ctx, problem);

        if(!valid)
            MIOPEN_THROW("invalid performance parameter");

        int GemmBBlockCopyDescDataPerWriteGemm = 0;
        std::tie(std::ignore, std::ignore, std::ignore, GemmBBlockCopyDescDataPerWriteGemm, valid) =
            CalculateGemmBBlockCopyPerformanceParameters(ctx, problem);

        if(!valid)
            MIOPEN_THROW("invalid performance parameter");

        const int epack                        = GetEPackLength(ctx, problem, false);
        const auto ThreadGemmDataPerRead_GemmM = problem.IsFp32() ? GemmMPerThread : epack;
        const auto ThreadGemmDataPerRead_GemmN = problem.IsFp32() ? GemmNPerThread : epack;

        const auto max_lds_align = lcm(GemmABlockCopyDescDataPerWriteGemm,
                                       GemmBBlockCopyDescDataPerWriteGemm,
                                       ThreadGemmDataPerRead_GemmM,
                                       ThreadGemmDataPerRead_GemmN);

        const auto a_block_space =
            GemmKPerBlock * integer_least_multiple(GemmMPerBlock, max_lds_align);
        const auto b_block_space =
            GemmKPerBlock * integer_least_multiple(GemmNPerBlock, max_lds_align);

        lds_size = 2 * (static_cast<std::size_t>(a_block_space) + b_block_space) * sizeof(float);
    }
    catch(...)
    {
        return std::make_tuple(0, false);
    }

    return std::make_tuple(lds_size, true);
}

bool PerformanceImplicitGemmBwdDataV1R1::IsValidValue() const
{
    // clang-format off
    return IsTwoPower<64, 256>(BlockSize) &&
           IsTwoPower<32, 128>(GemmMPerBlock) &&
           IsTwoPower<32, 128>(GemmNPerBlock) &&
           IsTwoPower<4, 16>(GemmKPerBlock) &&
           IsTwoPower<2, 4>(GemmMPerThread) &&
           IsTwoPower<2, 4>(GemmNPerThread);
    // clang-format on
}

bool PerformanceImplicitGemmBwdDataV1R1::IsValid(const ExecutionContext& ctx,
                                                 const ProblemDescription& problem) const
{
    if(!IsValidValue())
        return false;

    bool valid = false;

    // check blockwise GEMM size
    int gemm_m = 0;
    int gemm_n = 0;
    int gemm_k = 0;

    std::tie(gemm_m, gemm_n, gemm_k) =
        ConvHipImplicitGemmBwdDataV1R1::CalculateGemmSize(ctx, problem);

    if(!(gemm_m % GemmMPerBlock == 0 && gemm_n % GemmNPerBlock == 0 && gemm_k % GemmKPerBlock == 0))
        return false;

    if(!(GemmMPerBlock % GemmMPerThread == 0 && GemmNPerBlock % GemmNPerThread == 0))
        return false;

    // check thread cluster in blockwise GEMM
    std::tie(std::ignore, std::ignore, std::ignore, std::ignore, valid) =
        CalculateBlockGemmPerformanceParameters();

    if(!valid)
        return false;

    // check blockwise copy of A matrix
    std::tie(std::ignore, std::ignore, std::ignore, std::ignore, valid) =
        CalculateGemmABlockCopyPerformanceParameters(ctx, problem);

    if(!valid)
        return false;

    // check blockwise copy of B matrix
    std::tie(std::ignore, std::ignore, std::ignore, std::ignore, valid) =
        CalculateGemmBBlockCopyPerformanceParameters(ctx, problem);

    if(!valid)
        return false;

    // check threadwise copy of C matrix
    std::tie(std::ignore, valid) = CalculateGemmCThreadCopyPerformanceParameters(problem);

    if(!valid)
        return false;

    // check LDS allocation
    std::size_t lds_size      = 0;
    std::tie(lds_size, valid) = CalculateLdsNumberOfByte(ctx, problem);

    return (valid and lds_size <= get_lds_max_number_of_byte());
}

void PerformanceImplicitGemmBwdDataV1R1::HeuristicInit(const ExecutionContext& ctx,
                                                       const ProblemDescription& problem)
{
    PerformanceImplicitGemmBwdDataV1R1 config;

    config = {256, 128, 128, 16, 4, 4};
    if(!config.IsValid(ctx, problem))
        config = {256, 128, 128, 8, 4, 4};
    if(!config.IsValid(ctx, problem))
        config = {256, 128, 128, 4, 4, 4};
    if(!config.IsValid(ctx, problem))
        config = {128, 128, 64, 16, 4, 4};
    if(!config.IsValid(ctx, problem))
        config = {128, 128, 64, 8, 4, 4};
    if(!config.IsValid(ctx, problem))
        config = {128, 128, 64, 4, 4, 4};
    if(!config.IsValid(ctx, problem))
        config = {128, 64, 128, 16, 4, 4};
    if(!config.IsValid(ctx, problem))
        config = {128, 64, 128, 8, 4, 4};
    if(!config.IsValid(ctx, problem))
        config = {128, 64, 128, 4, 4, 4};
    if(!config.IsValid(ctx, problem))
        config = {64, 64, 64, 16, 4, 4};
    if(!config.IsValid(ctx, problem))
        config = {64, 64, 64, 8, 4, 4};
    if(!config.IsValid(ctx, problem))
        config = {64, 64, 64, 4, 4, 4};
    if(!config.IsValid(ctx, problem))
        config = {64, 64, 32, 16, 4, 2};
    if(!config.IsValid(ctx, problem))
        config = {64, 64, 32, 8, 4, 2};
    if(!config.IsValid(ctx, problem))
        config = {64, 64, 32, 4, 4, 2};
    if(!config.IsValid(ctx, problem))
        config = {64, 32, 64, 16, 2, 4};
    if(!config.IsValid(ctx, problem))
        config = {64, 32, 64, 8, 2, 4};
    if(!config.IsValid(ctx, problem))
        config = {64, 32, 64, 4, 2, 4};
    if(!config.IsValid(ctx, problem))
        config = {64, 32, 32, 16, 2, 2};
    if(!config.IsValid(ctx, problem))
        config = {64, 32, 32, 8, 2, 2};
    if(!config.IsValid(ctx, problem))
        config = {64, 32, 32, 4, 2, 2};
    if(!config.IsValid(ctx, problem))
    {
        MIOPEN_LOG_E("All attempts failed: ");
        assert(false);
    }

    *this = config;
    MIOPEN_LOG_I(ToString());
}

bool PerformanceImplicitGemmBwdDataV1R1::SetNextValue(const ProblemDescription&)
{
    // always search full space, no matter if use_spare_set or not
    do
    {
        if(!NextTwoPower<64, 256>(BlockSize))
            break;
        if(!NextTwoPower<32, 128>(GemmMPerBlock))
            break;
        if(!NextTwoPower<32, 128>(GemmNPerBlock))
            break;
        if(!NextTwoPower<4, 16>(GemmKPerBlock))
            break;
        if(!NextTwoPower<2, 4>(GemmMPerThread))
            break;
        if(!NextTwoPower<2, 4>(GemmNPerThread))
            break;

        return false;
    } while(false);

    return true;
}

std::tuple<int, int, int>
ConvHipImplicitGemmBwdDataV1R1::CalculateGemmSize(const ExecutionContext& ctx,
                                                  const ProblemDescription& problem)
{
    const auto n  = ProblemInterpreter::GetBatchN(problem);
    const auto k  = ProblemInterpreter::GetOutputChannelK(problem);
    const auto c  = ProblemInterpreter::GetInputChannelC(problem);
    const auto ho = ProblemInterpreter::GetOutputHeightHo(problem);
    const auto wo = ProblemInterpreter::GetOutputWidthWo(problem);
    const auto y  = ProblemInterpreter::GetFilterHeightY(problem);
    const auto x  = ProblemInterpreter::GetFilterWidthX(problem);

    const auto gemm_m =
        c * y * x * (problem.Is3d() ? ProblemInterpreter::GetFilterDepthZ(problem) : 1);
    const auto gemm_n =
        n * ho * wo * (problem.Is3d() ? ProblemInterpreter::GetOutputDepthDo(problem) : 1);
    const auto gemm_k = k / GetEPackLength(ctx, problem, false);

    return std::make_tuple(gemm_m, gemm_n, gemm_k);
}

size_t ConvHipImplicitGemmBwdDataV1R1::GetWorkspaceSize(const ExecutionContext&,
                                                        const ProblemDescription& problem) const
{
    if(problem.IsFp32())
    {
        return 0;
    }
    else
    {
        // In case of fp16/bfp16, because there is no atomic add ISA,
        // reduction via atomic add ISA is done via fp32. As a result,
        // workspace is computed with miopenFloat data type.
        // Later, a separate kernel is invoked that casts from fp32 to fp16/bfp16
        std::size_t n  = ProblemInterpreter::GetBatchN(problem);
        std::size_t c  = ProblemInterpreter::GetInputChannelC(problem);
        std::size_t hi = ProblemInterpreter::GetInputHeightHi(problem);
        std::size_t wi = ProblemInterpreter::GetInputWidthWi(problem);
        return n * c * (problem.Is3d() ? ProblemInterpreter::GetInputDepthDi(problem) : 1) * hi *
               wi * miopen::GetTypeSize(miopenFloat);
    }
}

bool ConvHipImplicitGemmBwdDataV1R1::IsApplicable(const ExecutionContext& ctx,
                                                  const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1))
        return false;
    if(ThisSolverIsDeprecatedStatic::IsDisabled(ctx))
        return false;
    if(problem.GetConv().attribute.deterministic)
        return false;
    if(!ctx.use_hip_kernels)
        return false;
    if(problem.HasNonPackedTensors())
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(!problem.IsLayoutDefault())
        return false;
    if(!IsComposableKernelSupportedHardware(ctx))
        return false;
    if(!problem.IsDirectionBackwardData())
        return false;
    if(!problem.Is2d() && !(problem.Is3d() && problem.IsFp32()))
        return false;

    if(!(problem.IsFp32() || problem.IsBfp16()))
        return false;

    if(problem.IsTensorsCasted())
        return false;
    if(problem.GetGroupCount() != 1)
        return false;
    if(!IsIndexRangeLargeEnough(problem))
        return false;
#if WORKAROUND_ISSUE_309
    if(problem.IsBfp16())
    {
        if(!env::enabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1))
            return false;
    }
#endif

    const auto k = ProblemInterpreter::GetOutputChannelK(problem);
    if(k % GetEPackLength(ctx, problem, false) != 0)
        return false;

    int gemm_m = 0;
    int gemm_n = 0;
    int gemm_k = 0;

    std::tie(gemm_m, gemm_n, gemm_k) = CalculateGemmSize(ctx, problem);

    return gemm_m % 32 == 0 && gemm_n % 32 == 0 && gemm_k % 4 == 0;
}

PerformanceImplicitGemmBwdDataV1R1
ConvHipImplicitGemmBwdDataV1R1::GetDefaultPerformanceConfig(const ExecutionContext& ctx,
                                                            const ProblemDescription& problem) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemmBwdDataV1R1>(ctx, problem);
}

bool ConvHipImplicitGemmBwdDataV1R1::IsValidPerformanceConfig(
    const ExecutionContext& ctx,
    const ProblemDescription& problem,
    const PerformanceImplicitGemmBwdDataV1R1& config) const
{
    MIOPEN_LOG_I("");
    return config.IsValidValue() && config.IsValid(ctx, problem);
}

PerformanceImplicitGemmBwdDataV1R1
ConvHipImplicitGemmBwdDataV1R1::Search(const ExecutionContext& ctx,
                                       const ProblemDescription& problem,
                                       const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

ConvSolution
ConvHipImplicitGemmBwdDataV1R1::GetSolution(const ExecutionContext& ctx,
                                            const ProblemDescription& problem,
                                            const PerformanceImplicitGemmBwdDataV1R1& config) const
{
    ConvSolution result;
    KernelInfo construction_parameters;

    assert(config.IsValid(ctx, problem));

    int grid_size = 0;

    std::tie(grid_size, std::ignore) = config.CalculateGridSize(ctx, problem);

    construction_parameters.l_wk.push_back(config.BlockSize);
    construction_parameters.l_wk.push_back(1);
    construction_parameters.l_wk.push_back(1);

    construction_parameters.g_wk.push_back(static_cast<std::size_t>(config.BlockSize) * grid_size);
    construction_parameters.g_wk.push_back(1);
    construction_parameters.g_wk.push_back(1);

    // clang-format off
    construction_parameters.kernel_file = problem.Is3d()
                                              ? "static_kernel_gridwise_convolution_backward_data_implicit_gemm_v1r1_ncdhw_kczyx_nkdhw.cpp"
                                              : "static_kernel_gridwise_convolution_backward_data_implicit_gemm_v1r1_nchw_kcyx_nkhw.cpp";
    // clang-format on

    construction_parameters.kernel_name =
        problem.Is3d() ? "gridwise_convolution_backward_data_implicit_gemm_v1r1_ncdhw_kczyx_nkdhw"
                       : "gridwise_convolution_backward_data_implicit_gemm_v1r1_nchw_kcyx_nkhw";

    int GemmMLevel0Cluster                      = 0;
    int GemmNLevel0Cluster                      = 0;
    int GemmMLevel1Cluster                      = 0;
    int GemmNLevel1Cluster                      = 0;
    int GemmABlockCopyClusterLengths_GemmK      = 0;
    int GemmABlockCopyClusterLengths_GemmM      = 0;
    int GemmABlockCopySrcDataPerRead_GemmM      = 0;
    int GemmABlockCopyDstDataPerWrite_GemmM     = 0;
    int GemmABlockCopyDstDataPerWrite_GemmKPACK = 0;
    int GemmBBlockCopyClusterLengths_GemmK      = 0;
    int GemmBBlockCopyClusterLengths_GemmN      = 0;
    int GemmBBlockCopySrcDataPerRead_GemmN      = 0;
    int GemmBBlockCopyDstDataPerWrite_GemmN     = 0;
    int GemmBBlockCopyDstDataPerWrite_GemmKPACK = 0;
    int GemmCThreadCopyDstDataPerWrite_GemmN1   = 0;

    std::tie(GemmMLevel0Cluster,
             GemmNLevel0Cluster,
             GemmMLevel1Cluster,
             GemmNLevel1Cluster,
             std::ignore) = config.CalculateBlockGemmPerformanceParameters();

    if(problem.IsFp32())
    {
        std::tie(GemmABlockCopyClusterLengths_GemmK,
                 GemmABlockCopyClusterLengths_GemmM,
                 GemmABlockCopySrcDataPerRead_GemmM,
                 GemmABlockCopyDstDataPerWrite_GemmM,
                 std::ignore) = config.CalculateGemmABlockCopyPerformanceParameters(ctx, problem);

        std::tie(GemmBBlockCopyClusterLengths_GemmK,
                 GemmBBlockCopyClusterLengths_GemmN,
                 GemmBBlockCopySrcDataPerRead_GemmN,
                 GemmBBlockCopyDstDataPerWrite_GemmN,
                 std::ignore) = config.CalculateGemmBBlockCopyPerformanceParameters(ctx, problem);
    }
    else
    {
        std::tie(GemmABlockCopyClusterLengths_GemmK,
                 GemmABlockCopyClusterLengths_GemmM,
                 GemmABlockCopySrcDataPerRead_GemmM,
                 GemmABlockCopyDstDataPerWrite_GemmKPACK,
                 std::ignore) = config.CalculateGemmABlockCopyPerformanceParameters(ctx, problem);

        std::tie(GemmBBlockCopyClusterLengths_GemmK,
                 GemmBBlockCopyClusterLengths_GemmN,
                 GemmBBlockCopySrcDataPerRead_GemmN,
                 GemmBBlockCopyDstDataPerWrite_GemmKPACK,
                 std::ignore) = config.CalculateGemmBBlockCopyPerformanceParameters(ctx, problem);
    }

    std::tie(GemmCThreadCopyDstDataPerWrite_GemmN1, std::ignore) =
        config.CalculateGemmCThreadCopyPerformanceParameters(problem);

    result.workspace_sz = GetWorkspaceSize(ctx, problem);

    // clang-format off
    construction_parameters.comp_options =
        std::string(" -DCK_PARAM_PROBLEM_N=") + std::to_string(ProblemInterpreter::GetBatchN(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_K=") + std::to_string(ProblemInterpreter::GetOutputChannelK(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_C=") + std::to_string(ProblemInterpreter::GetInputChannelC(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_HI=") + std::to_string(ProblemInterpreter::GetInputHeightHi(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_WI=") + std::to_string(ProblemInterpreter::GetInputWidthWi(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_HO=") + std::to_string(ProblemInterpreter::GetOutputHeightHo(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_WO=") + std::to_string(ProblemInterpreter::GetOutputWidthWo(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_Y=") + std::to_string(ProblemInterpreter::GetFilterHeightY(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_X=") + std::to_string(ProblemInterpreter::GetFilterWidthX(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_STRIDE_H=") + std::to_string(ProblemInterpreter::GetAdjustedConvolutionStrideH(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_STRIDE_W=") + std::to_string(ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DILATION_H=") + std::to_string(ProblemInterpreter::GetAdjustedConvolutionDilationH(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DILATION_W=") + std::to_string(ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_IN_LEFT_PAD_H=") + std::to_string(ProblemInterpreter::GetInputLeftPadH(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_IN_LEFT_PAD_W=") + std::to_string(ProblemInterpreter::GetInputLeftPadW(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_IN_RIGHT_PAD_H=") + std::to_string(ProblemInterpreter::GetAdjustedInputRightPadH(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_IN_RIGHT_PAD_W=") + std::to_string(ProblemInterpreter::GetAdjustedInputRightPadW(problem)) +
        std::string(" -DCK_PARAM_TUNABLE_BLOCK_SIZE=") + std::to_string(config.BlockSize) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_M_PER_BLOCK=") + std::to_string(config.GemmMPerBlock) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_N_PER_BLOCK=") + std::to_string(config.GemmNPerBlock) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_K_PER_BLOCK=") + std::to_string(config.GemmKPerBlock) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_M_PER_THREAD=") + std::to_string(config.GemmMPerThread) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_N_PER_THREAD=") + std::to_string(config.GemmNPerThread) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_M_LEVEL0_CLUSTER=") + std::to_string(GemmMLevel0Cluster) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_N_LEVEL0_CLUSTER=") + std::to_string(GemmNLevel0Cluster) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_M_LEVEL1_CLUSTER=") + std::to_string(GemmMLevel1Cluster) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_N_LEVEL1_CLUSTER=") + std::to_string(GemmNLevel1Cluster) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K=") + std::to_string(GemmABlockCopyClusterLengths_GemmK) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M=") + std::to_string(GemmABlockCopyClusterLengths_GemmM) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_M=") + std::to_string(GemmABlockCopySrcDataPerRead_GemmM) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K=") + std::to_string(GemmBBlockCopyClusterLengths_GemmK) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N=") + std::to_string(GemmBBlockCopyClusterLengths_GemmN) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_N=") + std::to_string(GemmBBlockCopySrcDataPerRead_GemmN) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_C_THREAD_COPY_DST_DATA_PER_WRITE_GEMM_N1=") + std::to_string(GemmCThreadCopyDstDataPerWrite_GemmN1) +
        std::string(" -DCK_PARAM_DEPENDENT_GRID_SIZE=") + std::to_string(grid_size) +
        std::string(" -DCK_THREADWISE_GEMM_USE_AMD_INLINE_ASM=") + (use_amd_inline_asm(ctx, problem) ? '1' : '0') +
        std::string(" -DCK_USE_AMD_INLINE_ASM=") + (use_amd_inline_asm(ctx, problem) ? '1' : '0') +
        get_static_ck_common_compiler_flag(ctx) +
        ctx.general_compile_options;
    // clang-format on
    if(problem.Is3d())
    {
        construction_parameters.comp_options +=
            std::string(" -DCK_PARAM_PROBLEM_DI=") +
            std::to_string(ProblemInterpreter::GetInputDepthDi(problem)) +
            std::string(" -DCK_PARAM_PROBLEM_DO=") +
            std::to_string(ProblemInterpreter::GetOutputDepthDo(problem)) +
            std::string(" -DCK_PARAM_PROBLEM_Z=") +
            std::to_string(ProblemInterpreter::GetFilterDepthZ(problem)) +
            std::string(" -DCK_PARAM_PROBLEM_CONV_STRIDE_D=") +
            std::to_string(ProblemInterpreter::GetAdjustedConvolutionStrideD(problem)) +
            std::string(" -DCK_PARAM_PROBLEM_CONV_DILATION_D=") +
            std::to_string(ProblemInterpreter::GetAdjustedConvolutionDilationD(problem)) +
            std::string(" -DCK_PARAM_PROBLEM_IN_LEFT_PAD_D=") +
            std::to_string(ProblemInterpreter::GetInputLeftPadD(problem)) +
            std::string(" -DCK_PARAM_PROBLEM_IN_RIGHT_PAD_D=") +
            std::to_string(ProblemInterpreter::GetAdjustedInputRightPadD(problem));
    }

    if(problem.IsFp32())
    {
        construction_parameters.comp_options +=
            std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_M=") +
            std::to_string(GemmABlockCopyDstDataPerWrite_GemmM) +
            std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_N=") +
            std::to_string(GemmBBlockCopyDstDataPerWrite_GemmN);
    }
    else
    {
        construction_parameters.comp_options +=
            std::string(" -DCK_PARAM_KPACK_LENGTH=") +
            std::to_string(GetEPackLength(ctx, problem, false)) +
            std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_KPACK=") +
            std::to_string(GemmABlockCopyDstDataPerWrite_GemmKPACK) +
            std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_KPACK=") +
            std::to_string(GemmBBlockCopyDstDataPerWrite_GemmKPACK);
    }

    result.invoker_factory = miopen::conv::MakeImplGemmDataInvokerFactory(problem);
    result.construction_params.push_back(construction_parameters);
    return result;
}

} // namespace conv
} // namespace solver
} // namespace miopen
