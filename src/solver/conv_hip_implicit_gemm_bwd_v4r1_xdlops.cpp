/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#include <miopen/generic_search.hpp>
#include <miopen/solver/implicitgemm_util.hpp>
#include <cstddef>

/// Disable ConvHipImplicitGemmBwdDataV4R1Xdlops for FP32 by default.
/// \ref https://github.com/ROCmSoftwarePlatform/MIOpen/issues/1206.
#define WORKAROUND_ISSUE_1206 1

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1_XDLOPS)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1_XDLOPS_PERF_VALS)

namespace miopen {
namespace solver {

std::tuple<int, bool>
PerformanceImplicitGemmBwdDataV4R1Xdlops::CalculateGridSize(const ConvolutionContext& ctx) const
{
    int GridSize = 0;

    try
    {
        int gemm_g = 0;
        int gemm_m = 0;
        int gemm_n = 0;

        std::tie(gemm_g, gemm_m, gemm_n, std::ignore) =
            ConvHipImplicitGemmBwdDataV4R1Xdlops::CalculateGemmSize(ctx, 0);

        if(!(gemm_m % GemmMPerBlock == 0 && gemm_n % GemmNPerBlock == 0))
            MIOPEN_THROW("invalid performance parameter");

        GridSize = gemm_g * (gemm_m / GemmMPerBlock) * (gemm_n / GemmNPerBlock);
    }
    catch(...)
    {
        return std::make_tuple(-1, false);
    }

    return std::make_tuple(GridSize, true);
}

std::tuple<int, int, int, int, int, bool>
PerformanceImplicitGemmBwdDataV4R1Xdlops::CalculateGemmABlockCopyPerformanceParameters(
    const ConvolutionContext& ctx) const
{
    int ClusterLengths_GemmK     = 0;
    int ClusterLengths_GemmM     = 0;
    int ClusterLengths_GemmKPack = 0;
    int SrcDataPerRead_GemmM     = ctx.IsFp32() ? amd_buffer_load_max_length<float>()
                                            : amd_buffer_load_max_length<half_float::half>();

    int DstDataPerWrite_GemmKPack = ctx.IsFp32() ? amd_buffer_load_max_length<float>()
                                                 : amd_buffer_load_max_length<half_float::half>();

    try
    {
        const auto WaveSize = 64;
        const auto BlockSize =
            GemmNPerBlock * GemmMPerBlock / (GemmMPerWave * GemmNPerWave) * WaveSize;

        // calculate vector length on gemmk dimension
        SrcDataPerRead_GemmM = gcd(SrcDataPerRead_GemmM, GemmMPerBlock);

        const auto y = ConvolutionContextInterpreter::GetFilterHeightY(ctx);
        const auto x = ConvolutionContextInterpreter::GetFilterWidthX(ctx);

        // \todo too conservative
        if(!(y == 1 && x == 1))
            SrcDataPerRead_GemmM = 1;

        // calculate threadwise copy size
        auto a_data_per_thread_copy =
            std::max(1, (GemmKPerBlock * GemmMPerBlock * GemmKPACKSize) / BlockSize);

        a_data_per_thread_copy = lcm(a_data_per_thread_copy, SrcDataPerRead_GemmM);
        // decide threadwise copy lengths
        const auto a_data_per_thread_copy_gemmm = SrcDataPerRead_GemmM;
        if(!(a_data_per_thread_copy_gemmm > 0))
            MIOPEN_THROW("invalid performance parameter");
        const auto tmp = a_data_per_thread_copy / a_data_per_thread_copy_gemmm;

        int data_per_thread_copy_gemmk     = -1;
        int data_per_thread_copy_gemmkpack = -1;

        if(GemmAThreadCopyMoreGemmK)
        {
            data_per_thread_copy_gemmk = gcd(GemmKPerBlock, tmp);
            if(!(data_per_thread_copy_gemmk > 0))
                MIOPEN_THROW("invalid performance parameter");
            data_per_thread_copy_gemmkpack = tmp / data_per_thread_copy_gemmk;
            if(!(data_per_thread_copy_gemmkpack > 0))
                MIOPEN_THROW("invalid performance parameter");
        }
        else
        {
            data_per_thread_copy_gemmkpack = gcd(GemmKPACKSize, tmp);
            if(!(data_per_thread_copy_gemmkpack > 0))
                MIOPEN_THROW("invalid performance parameter");
            data_per_thread_copy_gemmk = tmp / data_per_thread_copy_gemmkpack;
            if(!(data_per_thread_copy_gemmk > 0))
                MIOPEN_THROW("invalid performance parameter");
        }

        if(DstDataPerWrite_GemmKPack > data_per_thread_copy_gemmkpack)
            DstDataPerWrite_GemmKPack = data_per_thread_copy_gemmkpack;
        DstDataPerWrite_GemmKPack = gcd(DstDataPerWrite_GemmKPack, data_per_thread_copy_gemmkpack);

        if(!(GemmKPerBlock % data_per_thread_copy_gemmk == 0 &&
             GemmMPerBlock % a_data_per_thread_copy_gemmm == 0 &&
             GemmKPACKSize % data_per_thread_copy_gemmkpack == 0))
            MIOPEN_THROW("invalid performance parameter");

        ClusterLengths_GemmK     = GemmKPerBlock / data_per_thread_copy_gemmk;
        ClusterLengths_GemmM     = GemmMPerBlock / a_data_per_thread_copy_gemmm;
        ClusterLengths_GemmKPack = GemmKPACKSize / data_per_thread_copy_gemmkpack;
        // blockwise-copy support that block_size is larger than thread cluster size, which means
        // some threads may not do threadwise copy
        if(BlockSize < ClusterLengths_GemmK * ClusterLengths_GemmM * ClusterLengths_GemmKPack)
            MIOPEN_THROW("invalid performance parameter");
    }
    catch(...)
    {
        return std::make_tuple(-1, -1, -1, -1, -1, false);
    }

    return std::make_tuple(ClusterLengths_GemmK,
                           ClusterLengths_GemmM,
                           ClusterLengths_GemmKPack,
                           SrcDataPerRead_GemmM,
                           DstDataPerWrite_GemmKPack,
                           true);
}

std::tuple<int, int, int, int, int, bool>
PerformanceImplicitGemmBwdDataV4R1Xdlops::CalculateGemmBBlockCopyPerformanceParameters(
    const ConvolutionContext& ctx) const
{
    int ClusterLengths_GemmK     = 0;
    int ClusterLengths_GemmN     = 0;
    int ClusterLengths_GemmKPack = 0;
    int SrcDataPerRead_GemmN     = ctx.IsFp32() ? amd_buffer_load_max_length<float>()
                                            : amd_buffer_load_max_length<half_float::half>();

    int DstDataPerWrite_GemmKPack = ctx.IsFp32() ? amd_lds_write_max_length<float>()
                                                 : amd_lds_write_max_length<half_float::half>();

    try
    {
        const auto WaveSize = 64;
        const auto BlockSize =
            GemmNPerBlock * GemmMPerBlock / (GemmMPerWave * GemmNPerWave) * WaveSize;

        SrcDataPerRead_GemmN = gcd(SrcDataPerRead_GemmN, GemmNPerBlock);

        // calculate vector length on gemmn dimension
        const auto y           = ConvolutionContextInterpreter::GetFilterHeightY(ctx);
        const auto x           = ConvolutionContextInterpreter::GetFilterWidthX(ctx);
        const auto left_pad_h  = ConvolutionContextInterpreter::GetInputLeftPadH(ctx);
        const auto left_pad_w  = ConvolutionContextInterpreter::GetInputLeftPadW(ctx);
        const auto right_pad_h = ConvolutionContextInterpreter::GetAdjustedInputRightPadH(ctx);
        const auto right_pad_w = ConvolutionContextInterpreter::GetAdjustedInputRightPadW(ctx);

        // \todo too conversative
        if(y == 1 && x == 1 && left_pad_h == 0 && left_pad_w == 0 && right_pad_h == 0 &&
           right_pad_w == 0)
        {
            const auto ho        = ConvolutionContextInterpreter::GetOutputHeightHo(ctx);
            const auto wo        = ConvolutionContextInterpreter::GetOutputWidthWo(ctx);
            SrcDataPerRead_GemmN = gcd(SrcDataPerRead_GemmN, ho * wo);
        }
        else
        {
            SrcDataPerRead_GemmN = 1;
        }

        // calculate threadwise copy size
        int b_data_per_thread_copy =
            std::max(1, (GemmKPerBlock * GemmNPerBlock * GemmKPACKSize) / BlockSize);

        if(!(b_data_per_thread_copy > 0))
            MIOPEN_THROW("invalid performance parameter");

        b_data_per_thread_copy = lcm(SrcDataPerRead_GemmN, b_data_per_thread_copy);
        if(BlockSize > GemmNPerBlock && GemmKPACKSize > BlockSize / GemmNPerBlock)
            MIOPEN_THROW("invalid performance parameter");

        const auto data_per_thread_copy_gemmn = SrcDataPerRead_GemmN;
        if(!(data_per_thread_copy_gemmn > 0))
            MIOPEN_THROW("invalid performance parameter");

        const auto tmp = b_data_per_thread_copy / data_per_thread_copy_gemmn;
        if(!(tmp > 0))
            MIOPEN_THROW("invalid performance parameter");
        int data_per_thread_copy_gemmkpack = -1;
        int data_per_thread_copy_gemmk     = -1;

        if(GemmBThreadCopyMoreGemmKPack)
        {
            data_per_thread_copy_gemmkpack = gcd(GemmKPACKSize, tmp);
            if(!(data_per_thread_copy_gemmkpack > 0))
                MIOPEN_THROW("invalid performance parameter");

            data_per_thread_copy_gemmk = tmp / data_per_thread_copy_gemmkpack;
            if(!(data_per_thread_copy_gemmk > 0))
                MIOPEN_THROW("invalid performance parameter");
        }
        else
        {
            data_per_thread_copy_gemmk = gcd(GemmKPerBlock, tmp);
            if(!(data_per_thread_copy_gemmk > 0))
                MIOPEN_THROW("invalid performance parameter");
            data_per_thread_copy_gemmkpack = tmp / data_per_thread_copy_gemmk;
            if(!(data_per_thread_copy_gemmkpack > 0))
                MIOPEN_THROW("invalid performance parameter");
        }

        // vector write into LDS
        if(DstDataPerWrite_GemmKPack > data_per_thread_copy_gemmkpack)
            DstDataPerWrite_GemmKPack = data_per_thread_copy_gemmkpack;

        DstDataPerWrite_GemmKPack = gcd(DstDataPerWrite_GemmKPack, data_per_thread_copy_gemmkpack);

        if(!(GemmKPerBlock % data_per_thread_copy_gemmk == 0 &&
             GemmNPerBlock % data_per_thread_copy_gemmn == 0 &&
             GemmKPACKSize % data_per_thread_copy_gemmkpack == 0))
            MIOPEN_THROW("invalid performance parameter");

        ClusterLengths_GemmK     = GemmKPerBlock / data_per_thread_copy_gemmk;
        ClusterLengths_GemmN     = GemmNPerBlock / data_per_thread_copy_gemmn;
        ClusterLengths_GemmKPack = GemmKPACKSize / data_per_thread_copy_gemmkpack;

        if(BlockSize < ClusterLengths_GemmK * ClusterLengths_GemmN * ClusterLengths_GemmKPack)
            MIOPEN_THROW("invalid performance parameter");
    }
    catch(...)
    {
        MIOPEN_LOG_I("catch");
        return std::make_tuple(-1, -1, -1, -1, -1, false);
    }

    return std::make_tuple(ClusterLengths_GemmK,
                           ClusterLengths_GemmN,
                           ClusterLengths_GemmKPack,
                           SrcDataPerRead_GemmN,
                           DstDataPerWrite_GemmKPack,
                           true);
}

std::tuple<std::size_t, bool> PerformanceImplicitGemmBwdDataV4R1Xdlops::CalculateLdsNumberOfByte(
    const ConvolutionContext& ctx) const
{
    std::size_t lds_size = 0;

    try
    {
        bool valid = false;

        int GemmABlockCopyClusterLengths_GemmM      = 0;
        int GemmABlockCopyDescDataPerWriteGemmKPACK = 0;
        int GemmKPack                               = GemmKPACKSize;

        std::tie(std::ignore,
                 GemmABlockCopyClusterLengths_GemmM,
                 std::ignore,
                 std::ignore,
                 GemmABlockCopyDescDataPerWriteGemmKPACK,
                 valid) = CalculateGemmABlockCopyPerformanceParameters(ctx);

        if(!valid)
            MIOPEN_THROW("invalid performance parameter");

        int GemmBBlockCopyClusterLengths_GemmN      = 0;
        int GemmBBlockCopyDescDataPerWriteGemmKPACK = 0;
        std::tie(std::ignore,
                 GemmBBlockCopyClusterLengths_GemmN,
                 std::ignore,
                 std::ignore,
                 GemmBBlockCopyDescDataPerWriteGemmKPACK,
                 valid) = CalculateGemmBBlockCopyPerformanceParameters(ctx);

        if(!valid)
            MIOPEN_THROW("invalid performance parameter");

        if(GemmABlockCopyClusterLengths_GemmM == 0 || GemmBBlockCopyClusterLengths_GemmN == 0)
            MIOPEN_THROW("invalid performance parameter");

        const auto ThreadGemmDataPerRead_GemmM = GemmMPerBlock / GemmABlockCopyClusterLengths_GemmM;
        const auto ThreadGemmDataPerRead_GemmN = GemmNPerBlock / GemmBBlockCopyClusterLengths_GemmN;

        const auto max_lds_align = lcm(GemmABlockCopyDescDataPerWriteGemmKPACK,
                                       GemmBBlockCopyDescDataPerWriteGemmKPACK,
                                       ThreadGemmDataPerRead_GemmM,
                                       ThreadGemmDataPerRead_GemmN);

        const auto a_block_space =
            GemmKPerBlock * integer_least_multiple(GemmMPerBlock, max_lds_align);
        const auto b_block_space =
            GemmKPerBlock * integer_least_multiple(GemmNPerBlock, max_lds_align);
        lds_size = (a_block_space + b_block_space) * GetTypeSize(ctx.in_data_type) * GemmKPack;
    }
    catch(...)
    {
        return std::make_tuple(0, false);
    }

    return std::make_tuple(lds_size, true);
}

bool PerformanceImplicitGemmBwdDataV4R1Xdlops::IsReallyValid(const ConvolutionContext& ctx) const
{
    if(!IsValidValue())
        return false;

    int GemmM = 0, GemmN = 0, GemmK = 0, gemm_k_total = 0;

    // GemmKPACKSize = 4 for fp16
    if(ctx.IsFp16() && GemmKPACKSize % 4 != 0)
        return false;

    if(ctx.IsBfp16() && GemmKPACKSize % 2 != 0)
        return false;
    // check blockwise GEMM size
    for(int gemm_id = 0; gemm_id < ConvHipImplicitGemmBwdDataV4R1Xdlops::CalculateNumberOfGemm(ctx);
        ++gemm_id)
    {

        std::tie(std::ignore, GemmM, GemmN, gemm_k_total) =
            ConvHipImplicitGemmBwdDataV4R1Xdlops::CalculateGemmSize(ctx, gemm_id);

        if(gemm_k_total % GemmKPACKSize != 0)
            return false;

        GemmK = gemm_k_total / GemmKPACKSize;

        if(!(GemmM % GemmMPerBlock == 0 && GemmN % GemmNPerBlock == 0 &&
             GemmK % GemmKPerBlock == 0))
            return false; // wrong! cannot divice N evenly among thread
    }
    // heuristic to reduce search space
    {
        // use largest XdlopsGemm
        if(GemmMPerBlock % GemmMPerWave != 0)
            return false;
        if(GemmNPerBlock % GemmNPerWave != 0)
            return false;
    }

    if(!(GemmM % GemmMPerBlock == 0 && GemmN % GemmNPerBlock == 0 && GemmK % GemmKPerBlock == 0))
        return false; // wrong! cannot divice N evenly among thread

    if(!IsValidBlockwiseGemmXdlops(ctx,
                                   GemmMPerBlock,
                                   GemmNPerBlock,
                                   GemmKPerBlock,
                                   GemmMPerWave,
                                   GemmNPerWave,
                                   GemmKPACKSize))
        return false;

    bool valid = false;

    // check blockwise copy of A matrix
    std::tie(std::ignore, std::ignore, std::ignore, std::ignore, std::ignore, valid) =
        CalculateGemmABlockCopyPerformanceParameters(ctx);

    if(!valid)
        return false;

    // check blockwise copy of B matrix
    std::tie(std::ignore, std::ignore, std::ignore, std::ignore, std::ignore, valid) =
        CalculateGemmBBlockCopyPerformanceParameters(ctx);

    if(!valid)
        return false;

    std::size_t lds_size      = 0;
    std::tie(lds_size, valid) = CalculateLdsNumberOfByte(ctx);

    return (valid and lds_size <= 64 * 1024);
}

bool PerformanceImplicitGemmBwdDataV4R1Xdlops::IsFastToBeUsedForTuning(
    const ConvolutionContext& ctx) const
{
    if(use_spare_set)
        return true;

    // somehow, 128x128 wave-wise GEMM tend to spill register
    // TODO revisit this when 128x128 wave-wise GEMM become efficient
    {
        if(GemmMPerWave * GemmNPerWave > 64 * 128)
            return false;
    }

    // don't need too many blocks
    {
        int gemm_m = 0;
        int gemm_n = 0;

        std::tie(std::ignore, gemm_m, gemm_n, std::ignore) =
            ConvHipImplicitGemmBwdDataV4R1Xdlops::CalculateGemmSize(ctx, 0);

        // this is grid size using current blockwise-GEMM
        const int grid_size = (gemm_m * gemm_n) / (GemmMPerBlock * GemmNPerBlock);

        // this the the biggest blockwise-GEMM you can do
        int max_blockwise_gemm_size =
            std::max(gcd(256, gemm_m) * gcd(128, gemm_n), gcd(128, gemm_m) * gcd(256, gemm_n));

        // this is the grid size using the biggest blockwise-GEMM
        auto grid_size_max_blockwise_gemm =
            (std::size_t(gemm_m) * gemm_n) / max_blockwise_gemm_size;

        const float ratio = float(grid_size) / grid_size_max_blockwise_gemm;

        const auto num_cu = ctx.GetStream().GetMaxComputeUnits();

        // heuristic to exclude performance paramater that result in very large number of blocks
        if(grid_size_max_blockwise_gemm > 5 * num_cu)
        {
            if(ratio > 2.81)
                return false;
        }
        else if(grid_size_max_blockwise_gemm > 4 * num_cu)
        {
            if(ratio > 3.61)
                return false;
        }
        else if(grid_size_max_blockwise_gemm > 3 * num_cu)
        {
            if(ratio > 4.41)
                return false;
        }
        else if(grid_size_max_blockwise_gemm > 2 * num_cu)
        {
            if(ratio > 6.41)
                return false;
        }
        else if(grid_size_max_blockwise_gemm > num_cu)
        {
            if(ratio > 12.41)
                return false;
        }
    }

    // don't need too many waves per block
    {
        const int wave_per_block = (GemmMPerBlock / GemmMPerWave) * (GemmNPerBlock / GemmNPerWave);

        if(!(wave_per_block > 1 && wave_per_block <= 4))
        {
            return false;
        }
    }

    // each thread should not too much data
    {
        const int block_size = (GemmMPerBlock / GemmMPerWave) * (GemmNPerBlock / GemmNPerWave) * 64;

        const int a_data_per_thread_copy =
            (GemmKPerBlock * GemmMPerBlock * GemmKPACKSize) / block_size;
        const int b_data_per_thread_copy =
            (GemmKPerBlock * GemmNPerBlock * GemmKPACKSize) / block_size;

        if(ctx.IsFp32())
        {
            if(a_data_per_thread_copy > 16 || b_data_per_thread_copy > 16)
                return false;
        }
        else if(ctx.IsFp16() || ctx.IsBfp16())
        {
            if(a_data_per_thread_copy > 32 || b_data_per_thread_copy > 32)
                return false;
        }
    }

    return true;
}

bool PerformanceImplicitGemmBwdDataV4R1Xdlops::IsValid(const ConvolutionContext& ctx) const
{

    return IsReallyValid(ctx) && IsFastToBeUsedForTuning(ctx);
}

PerformanceImplicitGemmBwdDataV4R1Xdlops::PerformanceImplicitGemmBwdDataV4R1Xdlops()
    : PerformanceImplicitGemmBwdDataV4R1Xdlops::PerformanceImplicitGemmBwdDataV4R1Xdlops(
          16, 4, 1, 1, 4, 16, false, false)
{
}

PerformanceImplicitGemmBwdDataV4R1Xdlops::PerformanceImplicitGemmBwdDataV4R1Xdlops(bool spare)
    : PerformanceImplicitGemmBwdDataV4R1Xdlops::PerformanceImplicitGemmBwdDataV4R1Xdlops(
          16, 4, 1, 1, 4, 16, false, false, spare)
{
}

PerformanceImplicitGemmBwdDataV4R1Xdlops::PerformanceImplicitGemmBwdDataV4R1Xdlops(
    int GemmNPerBlock_,
    int GemmMPerBlock_,
    int GemmKPerBlock_,
    int GemmKPACKSize_,
    int GemmMPerWave_,
    int GemmNPerWave_,
    bool GemmAThreadCopyMoreGemmK_,
    bool GemmBThreadCopyMoreGemmKPack_,
    bool use_spare_set_)
    : GemmNPerBlock(GemmNPerBlock_),
      GemmMPerBlock(GemmMPerBlock_),
      GemmKPerBlock(GemmKPerBlock_),
      GemmKPACKSize(GemmKPACKSize_),
      GemmMPerWave(GemmMPerWave_),
      GemmNPerWave(GemmNPerWave_),
      GemmAThreadCopyMoreGemmK(GemmAThreadCopyMoreGemmK_),
      GemmBThreadCopyMoreGemmKPack(GemmBThreadCopyMoreGemmKPack_),
      use_spare_set(use_spare_set_)
{
}

bool PerformanceImplicitGemmBwdDataV4R1Xdlops::operator==(
    const PerformanceImplicitGemmBwdDataV4R1Xdlops& other) const
{
    // clang-format off
    return GemmNPerBlock == other.GemmNPerBlock
        && GemmMPerBlock == other.GemmMPerBlock
        && GemmKPerBlock == other.GemmKPerBlock
        && GemmKPACKSize == other.GemmKPACKSize
        && GemmMPerWave == other.GemmMPerWave
        && GemmNPerWave == other.GemmNPerWave
        && GemmAThreadCopyMoreGemmK  == other.GemmAThreadCopyMoreGemmK
        && GemmBThreadCopyMoreGemmKPack  == other.GemmBThreadCopyMoreGemmKPack
        && use_spare_set == other.use_spare_set;
    // clang-format on
}

bool PerformanceImplicitGemmBwdDataV4R1Xdlops::IsValidValue() const
{
    // clang-format off
    return IsTwoPower<16,256>(GemmNPerBlock)
        && IsTwoPower<4,256>(GemmMPerBlock)
        && IsTwoPower<1,8>(GemmKPerBlock)
        && IsTwoPower<1,8>(GemmKPACKSize)
        && IsTwoPower<4,128>(GemmMPerWave)
        && IsTwoPower<16,128>(GemmNPerWave); // clang-format on
}

bool PerformanceImplicitGemmBwdDataV4R1Xdlops::SetNextValue(const ConvolutionContext& /*config*/)
{
    GemmBThreadCopyMoreGemmKPack = true;
    GemmAThreadCopyMoreGemmK     = true;
    do
    {
        if(!NextTwoPower<16, 256>(GemmNPerBlock))
            break;
        if(!NextTwoPower<4, 256>(GemmMPerBlock))
            break;
        if(!NextTwoPower<1, 8>(GemmKPerBlock))
            break;
        if(!NextTwoPower<1, 8>(GemmKPACKSize))
            break;
        if(!NextTwoPower<16, 128>(GemmNPerWave))
            break;
        if(!NextTwoPower<4, 128>(GemmMPerWave))
            break;

        return false;
    } while(false);

    return true;
}

void PerformanceImplicitGemmBwdDataV4R1Xdlops::HeuristicInit(const ConvolutionContext& ctx)
{
    PerformanceImplicitGemmBwdDataV4R1Xdlops tmp;

    auto get_euristic_config = [&](auto is_valid_func) {
        if(ctx.IsFp32())
        {
            tmp              = {256, 256, 8, 4, 128, 128, true, true};
            bool all_visited = false;
            do
            {
                do
                {
                    // list in reverse order of importance,
                    // and favor large GEMM
                    if(!PreviousTwoPower<1, 8>(tmp.GemmKPerBlock))
                        break;
                    if(!PreviousTwoPower<1, 4>(tmp.GemmKPACKSize))
                        break;
                    if(!PreviousTwoPower<16, 128>(tmp.GemmNPerWave))
                        break;
                    if(!PreviousTwoPower<4, 128>(tmp.GemmMPerWave))
                        break;
                    if(!PreviousTwoPower<16, 256>(tmp.GemmNPerBlock))
                        break;
                    if(!PreviousTwoPower<4, 256>(tmp.GemmMPerBlock))
                        break;

                    all_visited = true;
                } while(false);

                if(is_valid_func(tmp, ctx))
                    break;
            } while(!all_visited);
        }
        else if(ctx.IsFp16())
        {
            tmp              = {256, 256, 8, 8, 128, 128, true, true};
            bool all_visited = false;
            do
            {
                do
                {
                    // list in reverse order of importance,
                    // and favor large GEMM
                    if(!PreviousTwoPower<1, 8>(tmp.GemmKPerBlock))
                        break;
                    if(!PreviousTwoPower<4, 8>(tmp.GemmKPACKSize))
                        break;
                    if(!PreviousTwoPower<16, 128>(tmp.GemmNPerWave))
                        break;
                    if(!PreviousTwoPower<4, 128>(tmp.GemmMPerWave))
                        break;
                    if(!PreviousTwoPower<16, 256>(tmp.GemmNPerBlock))
                        break;
                    if(!PreviousTwoPower<4, 256>(tmp.GemmMPerBlock))
                        break;

                    all_visited = true;
                } while(false);

                if(is_valid_func(tmp, ctx))
                    break;
            } while(!all_visited);
        }
        else if(ctx.IsBfp16())
        {
            tmp              = {256, 256, 8, 8, 128, 128, true, true};
            bool all_visited = false;
            do
            {
                do
                {
                    // list in reverse order of importance,
                    // and favor large GEMM
                    if(!PreviousTwoPower<1, 8>(tmp.GemmKPerBlock))
                        break;
                    if(!PreviousTwoPower<2, 8>(tmp.GemmKPACKSize))
                        break;
                    if(!PreviousTwoPower<16, 128>(tmp.GemmNPerWave))
                        break;
                    if(!PreviousTwoPower<4, 128>(tmp.GemmMPerWave))
                        break;
                    if(!PreviousTwoPower<16, 256>(tmp.GemmNPerBlock))
                        break;
                    if(!PreviousTwoPower<4, 256>(tmp.GemmMPerBlock))
                        break;

                    all_visited = true;
                } while(false);

                if(is_valid_func(tmp, ctx))
                    break;
            } while(!all_visited);
        }
        else
        {
            MIOPEN_LOG_E("Only fp32, fp16, and bfp16 are supported");
            assert(false);
        }
    };

    // first round: really valid and fast
    get_euristic_config([](auto config, auto conv_context) {
        return config.IsReallyValid(conv_context) && config.IsFastToBeUsedForTuning(conv_context);
    });

    // second round: really valid
    if(!tmp.IsReallyValid(ctx))
    {
        get_euristic_config(
            [](auto config, auto conv_context) { return config.IsReallyValid(conv_context); });
    }

    // final check
    if(!tmp.IsReallyValid(ctx))
    {
        MIOPEN_LOG_I("All attempts unsuccessful");
    }
    *this = tmp;
    MIOPEN_LOG_I(ToString());
}

std::string PerformanceImplicitGemmBwdDataV4R1Xdlops::ToString() const
{
    std::ostringstream ss;
    Serialize(ss);
    return ss.str();
}

int ConvHipImplicitGemmBwdDataV4R1Xdlops::CalculateNumberOfGemm(const ConvolutionContext& ctx)
{
    const auto conv_stride_h = ConvolutionContextInterpreter::GetAdjustedConvolutionStrideH(ctx);
    const auto conv_stride_w = ConvolutionContextInterpreter::GetAdjustedConvolutionStrideW(ctx);
    const auto conv_dilation_h =
        ConvolutionContextInterpreter::GetAdjustedConvolutionDilationH(ctx);
    const auto conv_dilation_w =
        ConvolutionContextInterpreter::GetAdjustedConvolutionDilationW(ctx);

    const auto gcd_stride_dilation_h = gcd(conv_stride_h, conv_dilation_h);
    const auto gcd_stride_dilation_w = gcd(conv_stride_w, conv_dilation_w);

    const auto ytilda = conv_stride_h / gcd_stride_dilation_h;
    const auto xtilda = conv_stride_w / gcd_stride_dilation_w;

    return ytilda * xtilda;
}

std::tuple<int, int, int, int>
ConvHipImplicitGemmBwdDataV4R1Xdlops::CalculateGemmSize(const ConvolutionContext& ctx, int gemm_id)
{
    const auto g             = ConvolutionContextInterpreter::GetGroupCountG(ctx);
    const auto n             = ConvolutionContextInterpreter::GetBatchN(ctx);
    const auto k             = ConvolutionContextInterpreter::GetOutputChannelK(ctx);
    const auto c             = ConvolutionContextInterpreter::GetInputChannelC(ctx);
    const auto hi            = ConvolutionContextInterpreter::GetInputHeightHi(ctx);
    const auto wi            = ConvolutionContextInterpreter::GetInputWidthWi(ctx);
    const auto ho            = ConvolutionContextInterpreter::GetOutputHeightHo(ctx);
    const auto wo            = ConvolutionContextInterpreter::GetOutputWidthWo(ctx);
    const auto y             = ConvolutionContextInterpreter::GetFilterHeightY(ctx);
    const auto x             = ConvolutionContextInterpreter::GetFilterWidthX(ctx);
    const auto conv_stride_h = ConvolutionContextInterpreter::GetAdjustedConvolutionStrideH(ctx);
    const auto conv_stride_w = ConvolutionContextInterpreter::GetAdjustedConvolutionStrideW(ctx);
    const auto conv_dilation_h =
        ConvolutionContextInterpreter::GetAdjustedConvolutionDilationH(ctx);
    const auto conv_dilation_w =
        ConvolutionContextInterpreter::GetAdjustedConvolutionDilationW(ctx);
    const auto in_left_pad_h = ConvolutionContextInterpreter::GetInputLeftPadH(ctx);
    const auto in_left_pad_w = ConvolutionContextInterpreter::GetInputLeftPadW(ctx);

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

    // gemm_k size is different for each GEMM
    const auto i_ytilda = gemm_id / xtilda;
    const auto i_xtilda = gemm_id % xtilda;

    const auto ydot_slice = (i_ytilda + 1) * ydot <= y ? ydot : y % ydot;
    const auto xdot_slice = (i_xtilda + 1) * xdot <= x ? xdot : x % xdot;

    const auto gemm_m = c / g;
    const auto gemm_n = n * htilda_slice * wtilda_slice;
    const auto gemm_k = (k / g) * ydot_slice * xdot_slice;

    return std::make_tuple(g, gemm_m, gemm_n, gemm_k);
}

bool ConvHipImplicitGemmBwdDataV4R1Xdlops::IsApplicable(const ConvolutionContext& ctx) const
{
#if WORKAROUND_ISSUE_1206
    if(ctx.IsFp32())
    {
        if(!miopen::IsEnabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1_XDLOPS{}))
            return false;
    }
    else
    {
        if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1_XDLOPS{}))
            return false;
    }
#else
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1_XDLOPS{}))
        return false;
#endif
    if(!IsComposableKernelSupportedHardware(ctx))
        return false;
    if(!ctx.direction.IsBackwardData())
        return false;
    if(!ctx.use_hip_kernels)
        return false;
    if(!ctx.Is2d())
        return false;
    if(!(ctx.IsFp32() || ctx.IsFp16() || ctx.IsBfp16()))
        return false;
    if(!IsApplicableXdlops(ctx))
        return false;
    if(!IsIndexRangeLargeEnough(ctx))
        return false;
    if(!ctx.IsLayoutDefault())
        return false;
    if(ctx.GetStream().GetDeviceName() == "gfx90a" && ctx.conv_problem.IsGfx90aFp16altRequired())
        return false;

    bool is_applicable = true;
    int gemm_g         = 0;
    int gemm_m         = 0;
    int gemm_n         = 0;
    int gemm_k_total   = 0;

    for(int gemm_id = 0; gemm_id < CalculateNumberOfGemm(ctx); ++gemm_id)
    {
        std::tie(gemm_g, gemm_m, gemm_n, gemm_k_total) = CalculateGemmSize(ctx, gemm_id);
        if(!IsValidGridGemmXdlops(gemm_m, gemm_n, gemm_k_total))
            return false;
    }
    return is_applicable;
}

PerformanceImplicitGemmBwdDataV4R1Xdlops
ConvHipImplicitGemmBwdDataV4R1Xdlops::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemmBwdDataV4R1Xdlops>(ctx);
}

bool ConvHipImplicitGemmBwdDataV4R1Xdlops::IsValidPerformanceConfig(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmBwdDataV4R1Xdlops& c) const
{
    MIOPEN_LOG_I("");
    return c.IsReallyValid(ctx);
}
PerformanceImplicitGemmBwdDataV4R1Xdlops
ConvHipImplicitGemmBwdDataV4R1Xdlops::Search(const ConvolutionContext& ctx,
                                             const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, invoke_ctx);
}

ConvSolution ConvHipImplicitGemmBwdDataV4R1Xdlops::GetSolution(
    const ConvolutionContext& ctx,
    const PerformanceImplicitGemmBwdDataV4R1Xdlops& config,
    const bool disableConfigOverrideFromEnv) const
{
    ConvSolution result;

    if(!config.IsReallyValid(ctx))
    {
        MIOPEN_LOG_E("invalid performance parameter");
        assert(false);
    }

    const PerformanceImplicitGemmBwdDataV4R1Xdlops* pcfg = &config;
    PerformanceImplicitGemmBwdDataV4R1Xdlops fromEnv;
    if(!disableConfigOverrideFromEnv)
    {
        std::string s;
        const auto p_asciz =
            miopen::GetStringEnv(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1_XDLOPS_PERF_VALS{});
        if(p_asciz != nullptr)
        {
            s = std::string(p_asciz);
            if(!s.empty()) // else nothing to parse.
            {
                if(!fromEnv.Deserialize(s) || !fromEnv.IsReallyValid(ctx))
                {
                    MIOPEN_LOG_E("MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1_XDLOPS_PERF_VALS: "
                                 "Bad format or invalid for the problem config: "
                                 << s);
                }
                else
                {
                    MIOPEN_LOG_I("Overridden from env: " << fromEnv.ToString());
                    pcfg = &fromEnv;
                }
            }
        }
    }

    // a series of kernels
    for(std::size_t gemm_id = 0; gemm_id < CalculateNumberOfGemm(ctx); ++gemm_id)
    {
        KernelInfo construction_parameters;

        int gemm_g = 0;
        int gemm_m = 0;
        int gemm_n = 0;
        int gemm_k = 0;

        std::tie(gemm_g, gemm_m, gemm_n, gemm_k) = CalculateGemmSize(ctx, gemm_id);

        // don't compile or launch an empty gridwise GEMM
        if(gemm_k > 0)
        {
            int grid_size = 0;

            const std::size_t GemmMPerBlock = pcfg->GemmMPerBlock;
            const std::size_t GemmNPerBlock = pcfg->GemmNPerBlock;
            const std::size_t GemmKPerBlock = pcfg->GemmKPerBlock;
            const std::size_t GemmMPerWave  = pcfg->GemmMPerWave;
            const std::size_t GemmNPerWave  = pcfg->GemmNPerWave;

            const std::size_t block_size =
                GemmNPerBlock * GemmMPerBlock / (GemmMPerWave * GemmNPerWave) * wave_size;

            std::tie(grid_size, std::ignore) = pcfg->CalculateGridSize(ctx);

            construction_parameters.l_wk.push_back(block_size);
            construction_parameters.l_wk.push_back(1);
            construction_parameters.l_wk.push_back(1);

            construction_parameters.g_wk.push_back(block_size * grid_size);
            construction_parameters.g_wk.push_back(1);
            construction_parameters.g_wk.push_back(1);

            // clang-format off
            construction_parameters.kernel_file =
                "static_kernel_gridwise_convolution_backward_data_implicit_gemm_v4r1_xdlops_nchw_kcyx_nkhw.cpp";

            construction_parameters.kernel_name =
                "gridwise_convolution_backward_data_implicit_gemm_v4r1_xdlops_nchw_kcyx_nkhw";
            // clang-format on

            // TODO: add fp16 calculation by GetWorkspaceSize(ctx);
            result.workspce_sz = 0;

            int GemmABlockCopySrcDataPerRead_GemmM = 1;
            int GemmBBlockCopySrcDataPerRead_GemmN = 1;
            int GemmABlockCopyClusterLengths_GemmK = 0;
            int GemmABlockCopyClusterLengths_GemmM = 0;
            int GemmBBlockCopyClusterLengths_GemmK = 0;
            int GemmBBlockCopyClusterLengths_GemmN = 0;

            int GemmABlockCopyClusterLengths_GemmKPack  = 1;
            int GemmABlockCopyDstDataPerWrite_GemmKPack = 1;

            std::tie(GemmABlockCopyClusterLengths_GemmK,
                     GemmABlockCopyClusterLengths_GemmM,
                     GemmABlockCopyClusterLengths_GemmKPack,
                     GemmABlockCopySrcDataPerRead_GemmM,
                     GemmABlockCopyDstDataPerWrite_GemmKPack,
                     std::ignore) = pcfg->CalculateGemmABlockCopyPerformanceParameters(ctx);

            int GemmBBlockCopyClusterLengths_GemmKPack  = 1;
            int GemmBBlockCopyDstDataPerWrite_GemmKPack = 1;

            std::tie(GemmBBlockCopyClusterLengths_GemmK,
                     GemmBBlockCopyClusterLengths_GemmN,
                     GemmBBlockCopyClusterLengths_GemmKPack,
                     GemmBBlockCopySrcDataPerRead_GemmN,
                     GemmBBlockCopyDstDataPerWrite_GemmKPack,
                     std::ignore) = pcfg->CalculateGemmBBlockCopyPerformanceParameters(ctx);

            const auto GemmABlockCopyDstDataPerWrite_GemmKPACK =
                GemmABlockCopyDstDataPerWrite_GemmKPack;
            const auto GemmBBlockCopyDstDataPerWrite_GemmKPACK =
                GemmBBlockCopyDstDataPerWrite_GemmKPack;

            // clang-format off
            construction_parameters.comp_options =
                std::string(" -DCK_PARAM_PROBLEM_N=") + std::to_string(ConvolutionContextInterpreter::GetBatchN(ctx)) +
                std::string(" -DCK_PARAM_PROBLEM_K=") + std::to_string(ConvolutionContextInterpreter::GetOutputChannelK(ctx)) +
                std::string(" -DCK_PARAM_PROBLEM_C=") + std::to_string(ConvolutionContextInterpreter::GetInputChannelC(ctx)) +
                std::string(" -DCK_PARAM_PROBLEM_HI=") + std::to_string(ConvolutionContextInterpreter::GetInputHeightHi(ctx)) +
                std::string(" -DCK_PARAM_PROBLEM_WI=") + std::to_string(ConvolutionContextInterpreter::GetInputWidthWi(ctx)) +
                std::string(" -DCK_PARAM_PROBLEM_HO=") + std::to_string(ConvolutionContextInterpreter::GetOutputHeightHo(ctx)) +
                std::string(" -DCK_PARAM_PROBLEM_WO=") + std::to_string(ConvolutionContextInterpreter::GetOutputWidthWo(ctx)) +
                std::string(" -DCK_PARAM_PROBLEM_Y=") + std::to_string(ConvolutionContextInterpreter::GetFilterHeightY(ctx)) +
                std::string(" -DCK_PARAM_PROBLEM_X=") + std::to_string(ConvolutionContextInterpreter::GetFilterWidthX(ctx)) +
                std::string(" -DCK_PARAM_PROBLEM_CONV_STRIDE_H=") + std::to_string(ConvolutionContextInterpreter::GetAdjustedConvolutionStrideH(ctx)) +
                std::string(" -DCK_PARAM_PROBLEM_CONV_STRIDE_W=") + std::to_string(ConvolutionContextInterpreter::GetAdjustedConvolutionStrideW(ctx)) +
                std::string(" -DCK_PARAM_PROBLEM_CONV_DILATION_H=") + std::to_string(ConvolutionContextInterpreter::GetAdjustedConvolutionDilationH(ctx)) +
                std::string(" -DCK_PARAM_PROBLEM_CONV_DILATION_W=") + std::to_string(ConvolutionContextInterpreter::GetAdjustedConvolutionDilationW(ctx)) +
                std::string(" -DCK_PARAM_PROBLEM_IN_LEFT_PAD_H=") + std::to_string(ConvolutionContextInterpreter::GetInputLeftPadH(ctx)) +
                std::string(" -DCK_PARAM_PROBLEM_IN_LEFT_PAD_W=") + std::to_string(ConvolutionContextInterpreter::GetInputLeftPadW(ctx)) +
                std::string(" -DCK_PARAM_PROBLEM_IN_RIGHT_PAD_H=") + std::to_string(ConvolutionContextInterpreter::GetAdjustedInputRightPadH(ctx)) +
                std::string(" -DCK_PARAM_PROBLEM_IN_RIGHT_PAD_W=") + std::to_string(ConvolutionContextInterpreter::GetAdjustedInputRightPadW(ctx)) +
                std::string(" -DCK_PARAM_PROBLEM_CONV_GROUP_COUNTS=") + std::to_string(ctx.group_counts) +
                std::string(" -DCK_PARAM_TUNABLE_BLOCK_SIZE=") + std::to_string(block_size) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_M_PER_BLOCK=") + std::to_string(GemmMPerBlock) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_N_PER_BLOCK=") + std::to_string(GemmNPerBlock) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_K_PER_BLOCK=") + std::to_string(GemmKPerBlock) +
                std::string(" -DCK_PARAM_GEMM_M_PER_WAVE=") + std::to_string(GemmMPerWave) +
                std::string(" -DCK_PARAM_GEMM_N_PER_WAVE=") + std::to_string(GemmNPerWave) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K=") + std::to_string(GemmABlockCopyClusterLengths_GemmK) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M=") + std::to_string(GemmABlockCopyClusterLengths_GemmM) +
                std::string(" -DCK_PARAM_DEPENDENT_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_KPACK=") + std::to_string(GemmABlockCopyClusterLengths_GemmKPack) +

                std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_M=") + std::to_string(GemmABlockCopySrcDataPerRead_GemmM) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K=") + std::to_string(GemmBBlockCopyClusterLengths_GemmK) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N=") + std::to_string(GemmBBlockCopyClusterLengths_GemmN) +
                std::string(" -DCK_PARAM_DEPENDENT_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_KPACK=") + std::to_string(GemmBBlockCopyClusterLengths_GemmKPack) +

                std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_N=") + std::to_string(GemmBBlockCopySrcDataPerRead_GemmN) +
                std::string(" -DCK_PARAM_DEPENDENT_GRID_SIZE=") + std::to_string(grid_size) +
                std::string(" -DCK_USE_AMD_XDLOPS=") + std::to_string(IsXdlopsSupport(ctx) ? 1 : 0) +
                std::string(" -DCK_USE_AMD_XDLOPS_INLINE_ASM=") + std::to_string(miopen::IsEnabled(MIOPEN_DEBUG_IMPLICIT_GEMM_XDLOPS_INLINE_ASM{}) ? 1 : 0) +
                std::string(" -DCK_USE_AMD_XDLOPS_EMULATE=") + (miopen::IsEnabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS_EMULATE{}) ? '1' : '0') +
                std::string(" -DCK_PARAM_GEMM_ID=") + std::to_string(gemm_id) +
                get_static_ck_common_compiler_flag(ctx) +
                ctx.general_compile_options;

                construction_parameters.comp_options +=
                    std::string(" -DCK_PARAM_KPACK_LENGTH=") + std::to_string(pcfg->GemmKPACKSize) +
                    std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_KPACK=") + std::to_string(GemmABlockCopyDstDataPerWrite_GemmKPACK) +
                    std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_KPACK=") + std::to_string(GemmBBlockCopyDstDataPerWrite_GemmKPACK);

            result.construction_params.push_back(construction_parameters);

        }
    }
    result.invoker_factory = conv::MakeImplGemmDataInvokerFactory(ctx);
    return result;
}

} // namespace solver
} // namespace miopen
