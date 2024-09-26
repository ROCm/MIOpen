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
#include <miopen/conv/solvers.hpp>
#include <miopen/handle.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/solver/implicitgemm_util.hpp>

#include <cstddef>
#include <numeric>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

PerformanceImplicitGemmBwdDataV4R1::PerformanceImplicitGemmBwdDataV4R1(int BlockSize_,
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

PerformanceImplicitGemmBwdDataV4R1::PerformanceImplicitGemmBwdDataV4R1(bool spare)
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

bool PerformanceImplicitGemmBwdDataV4R1::operator==(
    const PerformanceImplicitGemmBwdDataV4R1& other) const
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
PerformanceImplicitGemmBwdDataV4R1::CalculateGridSize(const ProblemDescription& problem) const
{
    int GridSize = 0;

    try
    {
        int gemm_m = 0;
        int gemm_n = 0;

        std::tie(gemm_m, gemm_n, std::ignore) =
            ConvHipImplicitGemmBwdDataV4R1::CalculateGemmSize(problem, 0);

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
PerformanceImplicitGemmBwdDataV4R1::CalculateBlockGemmPerformanceParameters() const
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
PerformanceImplicitGemmBwdDataV4R1::CalculateGemmABlockCopyPerformanceParameters(
    const ProblemDescription& problem) const
{
    int ClusterLengths_GemmK  = 0;
    int ClusterLengths_GemmM  = 0;
    int SrcDataPerRead_GemmM  = amd_buffer_load_max_length<float>();
    int DstDataPerWrite_GemmM = amd_lds_write_max_length<float>();

    try
    {
        // calculate vector length on gemmk dimension
        SrcDataPerRead_GemmM = gcd(SrcDataPerRead_GemmM, GemmMPerBlock);

        const auto y = ProblemInterpreter::GetFilterHeightY(problem);
        const auto x = ProblemInterpreter::GetFilterWidthX(problem);

        // \todo too conservative
        if(problem.Is3d())
        {
            const auto z = ProblemInterpreter::GetFilterDepthZ(problem);
            if(!(z == 1 && y == 1 && x == 1))
                SrcDataPerRead_GemmM = 1;
        }
        else
        {
            if(!(y == 1 && x == 1))
                SrcDataPerRead_GemmM = 1;
        }

        // calculate threadwise copy size
        const auto a_data_per_thread_copy = (GemmKPerBlock * GemmMPerBlock) / BlockSize;

        if(!(a_data_per_thread_copy > 0))
            MIOPEN_THROW("invalid performance parameter");

        // GemmABlockCopySrcDataPerRead_GemmK also bounded by size of threadwise copy
        SrcDataPerRead_GemmM = gcd(SrcDataPerRead_GemmM, a_data_per_thread_copy);

        // decide threadwise copy lengths
        const auto a_data_per_thread_copy_gemmm = SrcDataPerRead_GemmM;
        if(a_data_per_thread_copy_gemmm == 0)
            MIOPEN_THROW("DIV/0 with a_data_per_thread_copy_gemmm");
        const auto a_data_per_thread_copy_gemmk =
            a_data_per_thread_copy / a_data_per_thread_copy_gemmm;

        // GemmABlockCopyDstDataPerWrite_GemmM also bounded by size of threadwise copy
        DstDataPerWrite_GemmM = gcd(DstDataPerWrite_GemmM, a_data_per_thread_copy_gemmm);

        // calculate blockwise copy thread cluster lengths
        ClusterLengths_GemmK = GemmKPerBlock / a_data_per_thread_copy_gemmk; // NOLINT
        ClusterLengths_GemmM = GemmMPerBlock / a_data_per_thread_copy_gemmm; // NOLINT

        if(!(ClusterLengths_GemmK > 0 && ClusterLengths_GemmM > 0))
            MIOPEN_THROW("invalid performance parameter");
    }
    catch(...)
    {
        return std::make_tuple(-1, -1, -1, -1, false);
    }

    return std::make_tuple(ClusterLengths_GemmK,
                           ClusterLengths_GemmM,
                           SrcDataPerRead_GemmM,
                           DstDataPerWrite_GemmM,
                           true);
}

std::tuple<int, int, int, int, bool>
PerformanceImplicitGemmBwdDataV4R1::CalculateGemmBBlockCopyPerformanceParameters(
    const ProblemDescription& problem) const
{
    int ClusterLengths_GemmK  = 0;
    int ClusterLengths_GemmN  = 0;
    int SrcDataPerRead_GemmN  = amd_buffer_load_max_length<float>();
    int DstDataPerWrite_GemmN = amd_lds_write_max_length<float>();

    try
    {
        SrcDataPerRead_GemmN = gcd(SrcDataPerRead_GemmN, GemmNPerBlock);

        // calculate vector length on gemmn dimension
        const auto y           = ProblemInterpreter::GetFilterHeightY(problem);
        const auto x           = ProblemInterpreter::GetFilterWidthX(problem);
        const auto left_pad_h  = ProblemInterpreter::GetInputLeftPadH(problem);
        const auto left_pad_w  = ProblemInterpreter::GetInputLeftPadW(problem);
        const auto right_pad_h = ProblemInterpreter::GetAdjustedInputRightPadH(problem);
        const auto right_pad_w = ProblemInterpreter::GetAdjustedInputRightPadW(problem);

        // \todo too conversative
        if(problem.Is3d())
        {
            const auto z           = ProblemInterpreter::GetFilterDepthZ(problem);
            const auto left_pad_d  = ProblemInterpreter::GetInputLeftPadD(problem);
            const auto right_pad_d = ProblemInterpreter::GetAdjustedInputRightPadD(problem);
            if(z == 1 && y == 1 && x == 1 && left_pad_h == 0 && left_pad_w == 0 &&
               left_pad_d == 0 && right_pad_h == 0 && right_pad_w == 0 && right_pad_d == 0)
            {
                const auto dout = ProblemInterpreter::GetOutputDepthDo(problem);
                const auto ho   = ProblemInterpreter::GetOutputHeightHo(problem);
                const auto wo   = ProblemInterpreter::GetOutputWidthWo(problem);

                SrcDataPerRead_GemmN = gcd(SrcDataPerRead_GemmN, dout * ho * wo);
            }
            else
            {
                SrcDataPerRead_GemmN = 1;
            }
        }
        else
        {
            if(y == 1 && x == 1 && left_pad_h == 0 && left_pad_w == 0 && right_pad_h == 0 &&
               right_pad_w == 0)
            {
                const auto ho = ProblemInterpreter::GetOutputHeightHo(problem);
                const auto wo = ProblemInterpreter::GetOutputWidthWo(problem);

                SrcDataPerRead_GemmN = gcd(SrcDataPerRead_GemmN, ho * wo);
            }
            else
            {
                SrcDataPerRead_GemmN = 1;
            }
        }

        // calculate threadwise copy size
        int b_data_per_thread_copy = (GemmKPerBlock * GemmNPerBlock) / BlockSize;

        if(!(b_data_per_thread_copy > 0))
            MIOPEN_THROW("invalid performance parameter");

        // GemmBBlockCopySrcDataPerRead_GemmN also bounded by size of threadwise copy
        SrcDataPerRead_GemmN = gcd(SrcDataPerRead_GemmN, b_data_per_thread_copy);

        const auto b_data_per_thread_copy_gemmn = SrcDataPerRead_GemmN;
        const auto b_data_per_thread_copy_gemmk =
            b_data_per_thread_copy / b_data_per_thread_copy_gemmn;

        // GemmBBlockCopyDstDataPerWrite_GemmN also bounded by size of threadwise copy
        DstDataPerWrite_GemmN = gcd(DstDataPerWrite_GemmN, b_data_per_thread_copy_gemmn);

        // calculate blockwise copy thread cluster lengths
        ClusterLengths_GemmK = GemmKPerBlock / b_data_per_thread_copy_gemmk; // NOLINT
        ClusterLengths_GemmN = GemmNPerBlock / b_data_per_thread_copy_gemmn; // NOLINT

        if(!(ClusterLengths_GemmK > 0 && ClusterLengths_GemmN > 0))
            MIOPEN_THROW("invalid performance parameter");
    }
    catch(...)
    {
        MIOPEN_LOG_I("catch");
        return std::make_tuple(-1, -1, -1, -1, false);
    }

    return std::make_tuple(ClusterLengths_GemmK,
                           ClusterLengths_GemmN,
                           SrcDataPerRead_GemmN,
                           DstDataPerWrite_GemmN,
                           true);
}

std::tuple<int, bool>
PerformanceImplicitGemmBwdDataV4R1::CalculateGemmCThreadCopyPerformanceParameters(
    const ProblemDescription& problem) const
{
    int DstDataPerWrite_GemmN1 = amd_buffer_store_max_length<float>();

    try
    {
        // GemmCThreadCopyDstDataPerWrite_GemmN1 bounded by size of threadwise GEMM
        DstDataPerWrite_GemmN1 = gcd(DstDataPerWrite_GemmN1, GemmNPerThread);

        // GemmCThreadCopyDstDataPerWrite_GemmN1 limited by global memory layout of input tensor
        // calculate vector length on gemmn dimension
        const auto y              = ProblemInterpreter::GetFilterHeightY(problem);
        const auto x              = ProblemInterpreter::GetFilterWidthX(problem);
        const auto hi             = ProblemInterpreter::GetInputHeightHi(problem);
        const auto wi             = ProblemInterpreter::GetInputWidthWi(problem);
        const auto conv_stride_h  = ProblemInterpreter::GetAdjustedConvolutionStrideH(problem);
        const auto conv_stride_w  = ProblemInterpreter::GetAdjustedConvolutionStrideW(problem);
        const auto in_left_pad_h  = ProblemInterpreter::GetInputLeftPadH(problem);
        const auto in_left_pad_w  = ProblemInterpreter::GetInputLeftPadW(problem);
        const auto in_right_pad_h = ProblemInterpreter::GetAdjustedInputRightPadH(problem);
        const auto in_right_pad_w = ProblemInterpreter::GetAdjustedInputRightPadW(problem);

        if(problem.Is3d())
        {
            const auto z              = ProblemInterpreter::GetFilterDepthZ(problem);
            const auto conv_stride_d  = ProblemInterpreter::GetAdjustedConvolutionStrideD(problem);
            const auto in_left_pad_d  = ProblemInterpreter::GetInputLeftPadD(problem);
            const auto in_right_pad_d = ProblemInterpreter::GetAdjustedInputRightPadD(problem);

            if(z == 1 && y == 1 && x == 1 && conv_stride_d == 1 && conv_stride_h == 1 &&
               conv_stride_w == 1 && in_left_pad_d == 0 && in_left_pad_h == 0 &&
               in_left_pad_w == 0 && in_right_pad_d == 0 && in_right_pad_h == 0 &&
               in_right_pad_w == 0)
            {
                // \todo too conservative, there are more configs that can go through this if branch
                const auto di          = ProblemInterpreter::GetInputDepthDi(problem);
                DstDataPerWrite_GemmN1 = gcd(DstDataPerWrite_GemmN1, di * hi * wi);
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
                // \todo too conservative, there are more configs that can go through this if branch
                DstDataPerWrite_GemmN1 = gcd(DstDataPerWrite_GemmN1, hi * wi);
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

std::tuple<std::size_t, bool> PerformanceImplicitGemmBwdDataV4R1::CalculateLdsNumberOfByte(
    const ProblemDescription& problem) const
{
    std::size_t lds_size = 0;

    try
    {
        bool valid = false;

        int GemmABlockCopyDescDataPerWriteGemmM = 0;
        std::tie(
            std::ignore, std::ignore, std::ignore, GemmABlockCopyDescDataPerWriteGemmM, valid) =
            CalculateGemmABlockCopyPerformanceParameters(problem);

        if(!valid)
            MIOPEN_THROW("invalid performance parameter");

        int GemmBBlockCopyDescDataPerWriteGemmN = 0;
        std::tie(
            std::ignore, std::ignore, std::ignore, GemmBBlockCopyDescDataPerWriteGemmN, valid) =
            CalculateGemmBBlockCopyPerformanceParameters(problem);

        if(!valid)
            MIOPEN_THROW("invalid performance parameter");

        const auto ThreadGemmDataPerRead_GemmM = GemmMPerThread;
        const auto ThreadGemmDataPerRead_GemmN = GemmNPerThread;

        const auto max_lds_align = lcm(GemmABlockCopyDescDataPerWriteGemmM,
                                       GemmBBlockCopyDescDataPerWriteGemmN,
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

bool PerformanceImplicitGemmBwdDataV4R1::IsValidValue() const
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

bool PerformanceImplicitGemmBwdDataV4R1::IsValid(const ProblemDescription& problem) const
{
    if(!IsValidValue())
        return false;

    bool valid = false;

    // check blockwise GEMM size
    for(int gemm_id = 0; gemm_id < ConvHipImplicitGemmBwdDataV4R1::CalculateNumberOfGemm(problem);
        ++gemm_id)
    {
        int gemm_m = 0;
        int gemm_n = 0;
        int gemm_k = 0;

        std::tie(gemm_m, gemm_n, gemm_k) =
            ConvHipImplicitGemmBwdDataV4R1::CalculateGemmSize(problem, gemm_id);

        if(!(gemm_m % GemmMPerBlock == 0 && gemm_n % GemmNPerBlock == 0 &&
             gemm_k % GemmKPerBlock == 0))
            return false;
    }

    if(!(GemmMPerBlock % GemmMPerThread == 0 && GemmNPerBlock % GemmNPerThread == 0))
        return false;

    // check thread cluster in blockwise GEMM
    std::tie(std::ignore, std::ignore, std::ignore, std::ignore, valid) =
        CalculateBlockGemmPerformanceParameters();

    if(!valid)
        return false;

    // check blockwise copy of A matrix
    std::tie(std::ignore, std::ignore, std::ignore, std::ignore, valid) =
        CalculateGemmABlockCopyPerformanceParameters(problem);

    if(!valid)
        return false;

    // check blockwise copy of B matrix
    std::tie(std::ignore, std::ignore, std::ignore, std::ignore, valid) =
        CalculateGemmBBlockCopyPerformanceParameters(problem);

    if(!valid)
        return false;

    // check threadwise copy of C matrix
    std::tie(std::ignore, valid) = CalculateGemmCThreadCopyPerformanceParameters(problem);

    if(!valid)
        return false;

    // check LDS allocation
    std::size_t lds_size      = 0;
    std::tie(lds_size, valid) = CalculateLdsNumberOfByte(problem);

    return (valid and lds_size <= get_lds_max_number_of_byte());
}

void PerformanceImplicitGemmBwdDataV4R1::HeuristicInit(const ExecutionContext& ctx,
                                                       const ProblemDescription& problem)
{
    std::ignore = ctx;
    PerformanceImplicitGemmBwdDataV4R1 config;

    config = {256, 128, 128, 16, 4, 4};
    if(!config.IsValid(problem))
        config = {256, 128, 128, 8, 4, 4};
    if(!config.IsValid(problem))
        config = {256, 128, 128, 4, 4, 4};
    if(!config.IsValid(problem))
        config = {128, 128, 64, 16, 4, 4};
    if(!config.IsValid(problem))
        config = {128, 128, 64, 8, 4, 4};
    if(!config.IsValid(problem))
        config = {128, 128, 64, 4, 4, 4};
    if(!config.IsValid(problem))
        config = {128, 64, 128, 16, 4, 4};
    if(!config.IsValid(problem))
        config = {128, 64, 128, 8, 4, 4};
    if(!config.IsValid(problem))
        config = {128, 64, 128, 4, 4, 4};
    if(!config.IsValid(problem))
        config = {64, 64, 64, 16, 4, 4};
    if(!config.IsValid(problem))
        config = {64, 64, 64, 8, 4, 4};
    if(!config.IsValid(problem))
        config = {64, 64, 64, 4, 4, 4};
    if(!config.IsValid(problem))
        config = {64, 64, 32, 16, 4, 2};
    if(!config.IsValid(problem))
        config = {64, 64, 32, 8, 4, 2};
    if(!config.IsValid(problem))
        config = {64, 64, 32, 4, 4, 2};
    if(!config.IsValid(problem))
        config = {64, 32, 64, 16, 2, 4};
    if(!config.IsValid(problem))
        config = {64, 32, 64, 8, 2, 4};
    if(!config.IsValid(problem))
        config = {64, 32, 64, 4, 2, 4};
    if(!config.IsValid(problem))
        config = {64, 32, 32, 16, 2, 2};
    if(!config.IsValid(problem))
        config = {64, 32, 32, 8, 2, 2};
    if(!config.IsValid(problem))
        config = {64, 32, 32, 4, 2, 2};
    if(!config.IsValid(problem))
    {
        MIOPEN_LOG_E("All attempts failed: ");
        assert(false);
    }

    *this = config;
    MIOPEN_LOG_I(ToString());
}

bool PerformanceImplicitGemmBwdDataV4R1::SetNextValue(const ProblemDescription&)
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

int ConvHipImplicitGemmBwdDataV4R1::CalculateNumberOfGemm(const ProblemDescription& problem)
{
    const auto conv_stride_h   = ProblemInterpreter::GetAdjustedConvolutionStrideH(problem);
    const auto conv_stride_w   = ProblemInterpreter::GetAdjustedConvolutionStrideW(problem);
    const auto conv_dilation_h = ProblemInterpreter::GetAdjustedConvolutionDilationH(problem);
    const auto conv_dilation_w = ProblemInterpreter::GetAdjustedConvolutionDilationW(problem);

    const auto gcd_stride_dilation_h = gcd(conv_stride_h, conv_dilation_h);
    const auto gcd_stride_dilation_w = gcd(conv_stride_w, conv_dilation_w);

    const auto ytilda = conv_stride_h / gcd_stride_dilation_h;
    const auto xtilda = conv_stride_w / gcd_stride_dilation_w;

    if(problem.Is3d())
    {
        const auto conv_stride_d   = ProblemInterpreter::GetAdjustedConvolutionStrideD(problem);
        const auto conv_dilation_d = ProblemInterpreter::GetAdjustedConvolutionDilationD(problem);
        const auto gcd_stride_dilation_d = gcd(conv_stride_d, conv_dilation_d);
        const auto ztilda                = conv_stride_d / gcd_stride_dilation_d;

        return ztilda * ytilda * xtilda;
    }

    return ytilda * xtilda;
}

std::tuple<int, int, int>
ConvHipImplicitGemmBwdDataV4R1::CalculateGemmSize(const ProblemDescription& problem, int gemm_id)
{
    const auto n               = ProblemInterpreter::GetBatchN(problem);
    const auto k               = ProblemInterpreter::GetOutputChannelK(problem);
    const auto c               = ProblemInterpreter::GetInputChannelC(problem);
    const auto hi              = ProblemInterpreter::GetInputHeightHi(problem);
    const auto wi              = ProblemInterpreter::GetInputWidthWi(problem);
    const auto ho              = ProblemInterpreter::GetOutputHeightHo(problem);
    const auto wo              = ProblemInterpreter::GetOutputWidthWo(problem);
    const auto y               = ProblemInterpreter::GetFilterHeightY(problem);
    const auto x               = ProblemInterpreter::GetFilterWidthX(problem);
    const auto conv_stride_h   = ProblemInterpreter::GetAdjustedConvolutionStrideH(problem);
    const auto conv_stride_w   = ProblemInterpreter::GetAdjustedConvolutionStrideW(problem);
    const auto conv_dilation_h = ProblemInterpreter::GetAdjustedConvolutionDilationH(problem);
    const auto conv_dilation_w = ProblemInterpreter::GetAdjustedConvolutionDilationW(problem);
    const auto in_left_pad_h   = ProblemInterpreter::GetInputLeftPadH(problem);
    const auto in_left_pad_w   = ProblemInterpreter::GetInputLeftPadW(problem);

    const auto gcd_stride_dilation_h = gcd(conv_stride_h, conv_dilation_h);
    const auto gcd_stride_dilation_w = gcd(conv_stride_w, conv_dilation_w);

    const auto ytilda = conv_stride_h / gcd_stride_dilation_h;
    const auto xtilda = conv_stride_w / gcd_stride_dilation_w;

    const auto ydot = integer_divide_ceil(y, ytilda);
    const auto xdot = integer_divide_ceil(x, xtilda);

    const auto htilda = ho + integer_divide_ceil(conv_dilation_h * (y - 1), conv_stride_h);
    const auto wtilda = wo + integer_divide_ceil(conv_dilation_w * (x - 1), conv_stride_w);

    // intermediate result could be negative, use int instead of size_t
    const auto htilda_left =
        std::max(0, in_left_pad_h - conv_dilation_h * (ytilda - 1)) / conv_stride_h;
    const auto wtilda_left =
        std::max(0, in_left_pad_w - conv_dilation_w * (xtilda - 1)) / conv_stride_w;

    const auto htilda_right =
        std::min(htilda, integer_divide_ceil(in_left_pad_h + hi - 1, conv_stride_h) + 1);
    const auto wtilda_right =
        std::min(wtilda, integer_divide_ceil(in_left_pad_w + wi - 1, conv_stride_w) + 1);

    const auto htilda_slice = htilda_right - htilda_left;
    const auto wtilda_slice = wtilda_right - wtilda_left;

    if(problem.Is3d())
    {
        const auto i_ytilda   = (gemm_id % (xtilda * ytilda)) / xtilda;
        const auto i_xtilda   = (gemm_id % (xtilda * ytilda)) % xtilda;
        const auto ydot_slice = (i_ytilda + 1) * ydot <= y ? ydot : y % ydot;
        const auto xdot_slice = (i_xtilda + 1) * xdot <= x ? xdot : x % xdot;

        const auto di              = ProblemInterpreter::GetInputDepthDi(problem);
        const auto dout            = ProblemInterpreter::GetOutputDepthDo(problem);
        const auto z               = ProblemInterpreter::GetFilterDepthZ(problem);
        const auto conv_stride_d   = ProblemInterpreter::GetAdjustedConvolutionStrideD(problem);
        const auto conv_dilation_d = ProblemInterpreter::GetAdjustedConvolutionDilationD(problem);
        const auto in_left_pad_d   = ProblemInterpreter::GetInputLeftPadD(problem);
        const auto gcd_stride_dilation_z = gcd(conv_stride_d, conv_dilation_d);
        const auto ztilda                = conv_stride_d / gcd_stride_dilation_z;
        const auto zdot                  = integer_divide_ceil(z, ztilda);
        const auto dtilda = dout + integer_divide_ceil(conv_dilation_d * (z - 1), conv_stride_d);

        const auto dtilda_left =
            std::max(0, in_left_pad_d - conv_dilation_d * (ztilda - 1)) / conv_stride_d;
        const auto dtilda_right =
            std::min(dtilda, integer_divide_ceil(in_left_pad_d + di - 1, conv_stride_d) + 1);
        const auto dtilda_slice = dtilda_right - dtilda_left;
        const auto i_ztilda     = gemm_id / (xtilda * ytilda);
        const auto zdot_slice   = (i_ztilda + 1) * zdot <= z ? zdot : z % zdot;

        const auto gemm_m_3d = c;
        const auto gemm_n_3d = n * dtilda_slice * htilda_slice * wtilda_slice;
        const auto gemm_k_3d = k * zdot_slice * ydot_slice * xdot_slice;

        return std::make_tuple(gemm_m_3d, gemm_n_3d, gemm_k_3d);
    }
    else
    {
        // gemm_k size is different for each GEMM
        const auto i_ytilda = gemm_id / xtilda;
        const auto i_xtilda = gemm_id % xtilda;

        const auto ydot_slice = (i_ytilda + 1) * ydot <= y ? ydot : y % ydot;
        const auto xdot_slice = (i_xtilda + 1) * xdot <= x ? xdot : x % xdot;

        const auto gemm_m = c;
        const auto gemm_n = n * htilda_slice * wtilda_slice;
        const auto gemm_k = k * ydot_slice * xdot_slice;

        return std::make_tuple(gemm_m, gemm_n, gemm_k);
    }
}

bool ConvHipImplicitGemmBwdDataV4R1::IsApplicable(const ExecutionContext& ctx,
                                                  const ProblemDescription& problem) const
{
#if WORKAROUND_SWDEV_229277_227616_229195
    if(!env::enabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1))
        return false;
#endif
    if(ThisSolverIsDeprecatedStatic::IsDisabled(ctx))
        return false;

    if(env::disabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1))
        return false;
    if(problem.GetConv().attribute.deterministic)
        return false;

    if(!IsComposableKernelSupportedHardware(ctx))
        return false;

    if(!problem.IsDirectionBackwardData())
        return false;

    if(!ctx.use_hip_kernels)
        return false;

    if(!problem.Is2d() && !problem.Is3d())
        return false;

    if(!problem.IsFp32())
        return false;

    if(problem.HasNonPackedTensors())
        return false;

    if(!problem.AllTensorsDimsFitIntoInt())
        return false;

    if(problem.IsTensorsCasted())
        return false;

    if(problem.GetGroupCount() != 1)
        return false;

    if(!problem.IsLayoutDefault())
        return false;

    if(!IsIndexRangeLargeEnough(problem))
        return false;

    int gemm_m = 0;
    int gemm_n = 0;

    std::tie(gemm_m, gemm_n, std::ignore) = CalculateGemmSize(problem, 0);

    for(int gemm_id = 0; gemm_id < CalculateNumberOfGemm(problem); ++gemm_id)
    {
        int gemm_k = 0;

        std::tie(std::ignore, std::ignore, gemm_k) = CalculateGemmSize(problem, gemm_id);

        if(gemm_k % 4 != 0)
            return false;
    }

    return (gemm_m % 32 == 0 && gemm_n % 32 == 0);
}

PerformanceImplicitGemmBwdDataV4R1
ConvHipImplicitGemmBwdDataV4R1::GetDefaultPerformanceConfig(const ExecutionContext& ctx,
                                                            const ProblemDescription& problem) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemmBwdDataV4R1>(ctx, problem);
}

bool ConvHipImplicitGemmBwdDataV4R1::IsValidPerformanceConfig(
    const ExecutionContext&,
    const ProblemDescription& problem,
    const PerformanceImplicitGemmBwdDataV4R1& config) const
{
    MIOPEN_LOG_I("");
    return config.IsValidValue() && config.IsValid(problem);
}

PerformanceImplicitGemmBwdDataV4R1
ConvHipImplicitGemmBwdDataV4R1::Search(const ExecutionContext& ctx,
                                       const ProblemDescription& problem,
                                       const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

ConvSolution
ConvHipImplicitGemmBwdDataV4R1::GetSolution(const ExecutionContext& ctx,
                                            const ProblemDescription& problem,
                                            const PerformanceImplicitGemmBwdDataV4R1& config) const
{
    ConvSolution result;

    assert(config.IsValid(problem));

    // a series of kernels
    for(std::size_t gemm_id = 0; gemm_id < CalculateNumberOfGemm(problem); ++gemm_id)
    {
        KernelInfo construction_parameters;

        int gemm_m = 0;
        int gemm_n = 0;
        int gemm_k = 0;

        std::tie(gemm_m, gemm_n, gemm_k) = CalculateGemmSize(problem, gemm_id);

        // don't compile or launch an empty gridwise GEMM
        if(gemm_k > 0)
        {
            int grid_size = 0;

            std::tie(grid_size, std::ignore) = config.CalculateGridSize(problem);

            construction_parameters.l_wk.push_back(config.BlockSize);
            construction_parameters.l_wk.push_back(1);
            construction_parameters.l_wk.push_back(1);

            construction_parameters.g_wk.push_back(static_cast<std::size_t>(config.BlockSize) *
                                                   grid_size);
            construction_parameters.g_wk.push_back(1);
            construction_parameters.g_wk.push_back(1);

            if(problem.Is3d())
            {
                // clang-format off
                construction_parameters.kernel_file =
                    "static_kernel_gridwise_convolution_backward_data_implicit_gemm_v4r1_ncdhw_kczyx_nkdhw.cpp";

                construction_parameters.kernel_name =
                    "gridwise_convolution_backward_data_implicit_gemm_v4r1_ncdhw_kczyx_nkdhw";
                // clang-format on
            }
            else
            {
                // clang-format off
                construction_parameters.kernel_file =
                    "static_kernel_gridwise_convolution_backward_data_implicit_gemm_v4r1_nchw_kcyx_nkhw.cpp";

                construction_parameters.kernel_name =
                    "gridwise_convolution_backward_data_implicit_gemm_v4r1_nchw_kcyx_nkhw";
                // clang-format on
            }

            int GemmMLevel0Cluster                    = 0;
            int GemmNLevel0Cluster                    = 0;
            int GemmMLevel1Cluster                    = 0;
            int GemmNLevel1Cluster                    = 0;
            int GemmABlockCopyClusterLengths_GemmK    = 0;
            int GemmABlockCopyClusterLengths_GemmM    = 0;
            int GemmABlockCopySrcDataPerRead_GemmM    = 0;
            int GemmABlockCopyDstDataPerWrite_GemmM   = 0;
            int GemmBBlockCopyClusterLengths_GemmK    = 0;
            int GemmBBlockCopyClusterLengths_GemmN    = 0;
            int GemmBBlockCopySrcDataPerRead_GemmN    = 0;
            int GemmBBlockCopyDstDataPerWrite_GemmN   = 0;
            int GemmCThreadCopyDstDataPerWrite_GemmN1 = 0;

            std::tie(GemmMLevel0Cluster,
                     GemmNLevel0Cluster,
                     GemmMLevel1Cluster,
                     GemmNLevel1Cluster,
                     std::ignore) = config.CalculateBlockGemmPerformanceParameters();

            std::tie(GemmABlockCopyClusterLengths_GemmK,
                     GemmABlockCopyClusterLengths_GemmM,
                     GemmABlockCopySrcDataPerRead_GemmM,
                     GemmABlockCopyDstDataPerWrite_GemmM,
                     std::ignore) = config.CalculateGemmABlockCopyPerformanceParameters(problem);

            std::tie(GemmBBlockCopyClusterLengths_GemmK,
                     GemmBBlockCopyClusterLengths_GemmN,
                     GemmBBlockCopySrcDataPerRead_GemmN,
                     GemmBBlockCopyDstDataPerWrite_GemmN,
                     std::ignore) = config.CalculateGemmBBlockCopyPerformanceParameters(problem);

            std::tie(GemmCThreadCopyDstDataPerWrite_GemmN1, std::ignore) =
                config.CalculateGemmCThreadCopyPerformanceParameters(problem);

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
                std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_M=") + std::to_string(GemmABlockCopyDstDataPerWrite_GemmM) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K=") + std::to_string(GemmBBlockCopyClusterLengths_GemmK) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N=") + std::to_string(GemmBBlockCopyClusterLengths_GemmN) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_N=") + std::to_string(GemmBBlockCopySrcDataPerRead_GemmN) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_N=") + std::to_string(GemmBBlockCopyDstDataPerWrite_GemmN) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_C_THREAD_COPY_DST_DATA_PER_WRITE_GEMM_N1=") + std::to_string(GemmCThreadCopyDstDataPerWrite_GemmN1) +
                std::string(" -DCK_PARAM_DEPENDENT_GRID_SIZE=") + std::to_string(grid_size) +
                std::string(" -DCK_THREADWISE_GEMM_USE_AMD_INLINE_ASM=") + (use_amd_inline_asm(ctx, problem) ? '1' : '0') +
                std::string(" -DCK_USE_AMD_INLINE_ASM=") + (use_amd_inline_asm(ctx, problem) ? '1' : '0') +
                std::string(" -DCK_PARAM_GEMM_ID=") + std::to_string(gemm_id) +
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

            result.construction_params.push_back(construction_parameters);
        }
    }

    result.invoker_factory = miopen::conv::MakeImplGemmDataInvokerFactory(problem);
    return result;
}

} // namespace conv
} // namespace solver
} // namespace miopen
