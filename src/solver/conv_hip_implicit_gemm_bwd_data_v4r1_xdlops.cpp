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
#include <cstddef>
#include "miopen/solver.hpp"
#include "miopen/handle.hpp"
#include <miopen/generic_search.hpp>
#include "implicitgemm_util.hpp"

namespace miopen {
namespace solver {

std::tuple<int, bool>
PerformanceImplicitGemmBwdDataV4R1Xdlops::CalculateGridSize(const ConvolutionContext& ctx) const
{
    int GridSize = 0;

    try
    {
        int gemm_m = 0;
        int gemm_n = 0;

        std::tie(gemm_m, gemm_n, std::ignore) =
            ConvHipImplicitGemmBwdDataV4R1Xdlops::CalculateGemmSize(ctx, 0);

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
PerformanceImplicitGemmBwdDataV4R1Xdlops::CalculateGemmABlockCopyPerformanceParameters(
    const ConvolutionContext& ctx) const
{
    int ClusterLengths_GemmK  = 0;
    int ClusterLengths_GemmM  = 0;
    int SrcDataPerRead_GemmM  = amd_buffer_load_max_length<float>();
    int DstDataPerWrite_GemmM = amd_lds_write_max_length<float>();

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
        DstDataPerWrite_GemmM = gcd(DstDataPerWrite_GemmM, a_data_per_thread_copy_gemmm);

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

    return std::make_tuple(ClusterLengths_GemmK,
                           ClusterLengths_GemmM,
                           SrcDataPerRead_GemmM,
                           DstDataPerWrite_GemmM,
                           true);
}

std::tuple<int, int, int, int, bool>
PerformanceImplicitGemmBwdDataV4R1Xdlops::CalculateGemmBBlockCopyPerformanceParameters(
    const ConvolutionContext& ctx) const
{
    int ClusterLengths_GemmK  = 0;
    int ClusterLengths_GemmN  = 0;
    int SrcDataPerRead_GemmN  = amd_buffer_load_max_length<float>();
    int DstDataPerWrite_GemmN = amd_lds_write_max_length<float>();

    try
    {
        const auto WaveSize = 64;
        const auto BlockSize =
            GemmNPerBlock * GemmMPerBlock / (GemmMPerWave * GemmNPerWave) * WaveSize;

        SrcDataPerRead_GemmN = gcd(SrcDataPerRead_GemmN, GemmNPerBlock);

        // calculate vector length on gemmn dimension
        const auto y = ConvolutionContextInterpreter::GetFilterHeightY(ctx);
        const auto x = ConvolutionContextInterpreter::GetFilterWidthX(ctx);

        // \todo too conversative
        if(y == 1 && x == 1)
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

    return std::make_tuple(ClusterLengths_GemmK,
                           ClusterLengths_GemmN,
                           SrcDataPerRead_GemmN,
                           DstDataPerWrite_GemmN,
                           true);
}

std::tuple<std::size_t, bool> PerformanceImplicitGemmBwdDataV4R1Xdlops::CalculateLdsNumberOfByte(
    const ConvolutionContext& ctx) const
{
    std::size_t lds_size = 0;

    try
    {
        bool valid = false;

        int GemmABlockCopyClusterLengths_GemmM  = 0;
        int GemmABlockCopyDescDataPerWriteGemmM = 0;
        int GemmKPack                           = GemmKPACKSize;
        std::tie(std::ignore,
                 GemmABlockCopyClusterLengths_GemmM,
                 std::ignore,
                 GemmABlockCopyDescDataPerWriteGemmM,
                 valid) = CalculateGemmABlockCopyPerformanceParameters(ctx);

        if(!valid)
            MIOPEN_THROW("invalid performance parameter");

        int GemmBBlockCopyClusterLengths_GemmN  = 0;
        int GemmBBlockCopyDescDataPerWriteGemmN = 0;
        std::tie(std::ignore,
                 GemmBBlockCopyClusterLengths_GemmN,
                 std::ignore,
                 GemmBBlockCopyDescDataPerWriteGemmN,
                 valid) = CalculateGemmBBlockCopyPerformanceParameters(ctx);

        if(!valid)
            MIOPEN_THROW("invalid performance parameter");

        const auto ThreadGemmDataPerRead_GemmM = GemmMPerBlock / GemmABlockCopyClusterLengths_GemmM;
        const auto ThreadGemmDataPerRead_GemmN = GemmNPerBlock / GemmBBlockCopyClusterLengths_GemmN;

        const auto max_lds_align = lcm(GemmABlockCopyDescDataPerWriteGemmM,
                                       GemmBBlockCopyDescDataPerWriteGemmN,
                                       ThreadGemmDataPerRead_GemmM,
                                       ThreadGemmDataPerRead_GemmN);

        const auto a_block_space =
            GemmKPerBlock * integer_least_multiple(GemmMPerBlock, max_lds_align);
        const auto b_block_space =
            GemmKPerBlock * integer_least_multiple(GemmNPerBlock, max_lds_align);

        lds_size = 2 * (a_block_space + b_block_space) * GetTypeSize(ctx.in_data_type) * GemmKPack;
    }
    catch(...)
    {
        return std::make_tuple(0, false);
    }

    return std::make_tuple(lds_size, true);
}

bool PerformanceImplicitGemmBwdDataV4R1Xdlops::IsValid(const ConvolutionContext& ctx) const
{
    int GemmM = 0, GemmN = 0, GemmK = 0;

    const auto& GemmKBlocks = 1;

    // GemmKPACKSize = 1 for fp32
    if(ctx.IsFp32() && GemmKPACKSize != 1)
        return false;

    // GemmKPACKSize = 4 for fp16
    if(ctx.IsFp16() && GemmKPACKSize != 4)
        return false;

    // GemmKPACKSize = 2, 4 for bfp16 bwd non-group
    if(ctx.IsBfp16() && ctx.group_counts == 1 && GemmKPACKSize != 2 && GemmKPACKSize != 4)
        return false;

    // GemmKPACKSize = 2 for bfp16 bwd group
    if(ctx.IsBfp16() && ctx.group_counts > 1 && GemmKPACKSize != 2)
        return false;

    // check blockwise GEMM size
    for(int gemm_id = 0; gemm_id < ConvHipImplicitGemmBwdDataV4R1Xdlops::CalculateNumberOfGemm(ctx);
        ++gemm_id)
    {

        std::tie(GemmM, GemmN, GemmK) =
            ConvHipImplicitGemmBwdDataV4R1Xdlops::CalculateGemmSize(ctx, gemm_id);

        if(ctx.group_counts > 1)
        {
            GemmM = GemmM / ctx.group_counts;
            GemmK = GemmK / ctx.group_counts;
        }

        if(!(GemmM % GemmMPerBlock == 0 && GemmN % GemmNPerBlock == 0 &&
             GemmK % (GemmKPerBlock * GemmKBlocks) == 0))
            return false; // wrong! cannot divice N evenly among thread

        // check if (gemmk / kpack) % kPerBlock is 0
        if(ctx.IsFp16() || ctx.IsBfp16())
        {
            if(!((GemmK / GemmKPACKSize) % (GemmKPerBlock * GemmKBlocks) == 0))
                return false;
        }
    }
    // heuristic to reduce search space
    {
        // use largest XdlopsGemm
        if(GemmMPerBlock >= 64 && GemmMPerWave != 64)
            return false;
        if(GemmNPerBlock >= 64 && GemmNPerWave != 64)
            return false;
        if((GemmMPerBlock == 32 || GemmMPerBlock == 16) && GemmMPerWave != GemmMPerBlock)
            return false;
        if((GemmNPerBlock == 32 || GemmNPerBlock == 16) && GemmNPerWave != GemmNPerBlock)
            return false;
    }

    if(!(GemmM % GemmMPerBlock == 0 && GemmN % GemmNPerBlock == 0 && GemmK % GemmKPerBlock == 0))
        return false; // wrong! cannot divice N evenly among thread

    if(!IsValidXdlopsGemm(GemmMPerBlock, GemmNPerBlock, GemmKPerBlock, GemmMPerWave, GemmNPerWave))
        return false;

    bool valid = false;

    // check blockwise copy of A matrix
    std::tie(std::ignore, std::ignore, std::ignore, std::ignore, valid) =
        CalculateGemmABlockCopyPerformanceParameters(ctx);

    if(!valid)
        return false;

    // check blockwise copy of B matrix
    std::tie(std::ignore, std::ignore, std::ignore, std::ignore, valid) =
        CalculateGemmBBlockCopyPerformanceParameters(ctx);

    if(!valid)
        return false;

    std::size_t lds_size = 0;
    std::tie(lds_size, valid) = CalculateLdsNumberOfByte(ctx);

    return (valid and lds_size <= 64 * 1024);
}

PerformanceImplicitGemmBwdDataV4R1Xdlops::PerformanceImplicitGemmBwdDataV4R1Xdlops(bool spare)
{
    GemmNPerBlock = spare ? 16 : 64;
    GemmMPerBlock = spare ? 4 : 64;
    GemmKPerBlock = spare ? 4 : 8;

    GemmKPACKSize = 1;

    GemmMPerWave = spare ? 4 : 64;
    GemmNPerWave = spare ? 16 : 64;

    use_spare_set = spare;
}

PerformanceImplicitGemmBwdDataV4R1Xdlops::PerformanceImplicitGemmBwdDataV4R1Xdlops(
    int GemmNPerBlock_,
    int GemmMPerBlock_,
    int GemmKPerBlock_,
    int GemmKPACKSize_,
    int GemmMPerWave_,
    int GemmNPerWave_,
    bool use_spare_set_)
    : GemmNPerBlock(GemmNPerBlock_),
      GemmMPerBlock(GemmMPerBlock_),
      GemmKPerBlock(GemmKPerBlock_),
      GemmKPACKSize(GemmKPACKSize_),
      GemmMPerWave(GemmMPerWave_),
      GemmNPerWave(GemmNPerWave_),
      use_spare_set(use_spare_set_)
{
}

bool PerformanceImplicitGemmBwdDataV4R1Xdlops::
operator==(const PerformanceImplicitGemmBwdDataV4R1Xdlops& other) const
{
    // clang-format off
    return GemmNPerBlock == other.GemmNPerBlock
        && GemmMPerBlock == other.GemmMPerBlock
        && GemmKPerBlock == other.GemmKPerBlock
        && GemmKPACKSize == other.GemmKPACKSize
        && GemmMPerWave == other.GemmMPerWave
        && GemmNPerWave == other.GemmNPerWave
        && use_spare_set == other.use_spare_set;
    // clang-format on
}

bool PerformanceImplicitGemmBwdDataV4R1Xdlops::IsValidValue() const
{
    // clang-format off
    return IsTwoPower<16,128>(GemmNPerBlock)
        && IsTwoPower<4,128>(GemmMPerBlock)
        && IsTwoPower<4,32>(GemmKPerBlock)
        && IsTwoPower<1,4>(GemmKPACKSize)
        && IsTwoPower<4,64>(GemmMPerWave)
        && IsTwoPower<16,64>(GemmNPerWave); // clang-format on
}

bool PerformanceImplicitGemmBwdDataV4R1Xdlops::SetNextValue()
{
    do
    {
        if(!use_spare_set)
        {
            if(!NextTwoPower<64, 128>(GemmNPerBlock))
                break;
            if(!NextTwoPower<64, 128>(GemmMPerBlock))
                break;
            if(!NextTwoPower<8, 32>(GemmKPerBlock))
                break;
            if(!NextTwoPower<1, 4>(GemmKPACKSize))
                break;
        }
        else
        {
            if(!NextTwoPower<16, 128>(GemmNPerBlock))
                break;
            if(!NextTwoPower<4, 128>(GemmMPerBlock))
                break;
            if(!NextTwoPower<4, 32>(GemmKPerBlock))
                break;
            if(!NextTwoPower<1, 4>(GemmKPACKSize))
                break;
            if(!NextTwoPower<4, 64>(GemmMPerWave))
                break;
            if(!NextTwoPower<16, 64>(GemmNPerWave))
                break;
        }
        return false;
    } while(false);

    return true;
}

void PerformanceImplicitGemmBwdDataV4R1Xdlops::EuristicInit(const ConvolutionContext& ctx)
{
    PerformanceImplicitGemmBwdDataV4R1Xdlops tmp;
    if(ctx.IsFp32())
    {
        tmp = {128, 128, 8, 1, 64, 64, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {64, 32, 4, 1, 32, 64, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {32, 64, 4, 1, 64, 32, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {32, 32, 4, 1, 32, 32, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {64, 16, 4, 1, 16, 64, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {16, 64, 4, 1, 64, 16, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {16, 16, 4, 1, 16, 16, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {64, 4, 16, 1, 4, 64, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {64, 8, 8, 1, 8, 64, use_spare_set};
    }
    else if(ctx.IsBfp16())
    {
        tmp = {128, 128, 8, 2, 64, 64, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {64, 32, 4, 2, 32, 64, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {32, 64, 4, 2, 64, 32, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {32, 32, 4, 2, 32, 32, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {64, 16, 4, 2, 16, 64, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {16, 64, 4, 2, 64, 16, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {16, 16, 4, 2, 16, 16, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {64, 4, 16, 2, 4, 64, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {64, 8, 8, 2, 8, 64, use_spare_set};
    }
    else if(ctx.IsFp16())
    {
        tmp = {128, 128, 8, 4, 64, 64, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {64, 32, 4, 4, 32, 64, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {32, 64, 4, 4, 64, 32, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {32, 32, 4, 4, 32, 32, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {64, 16, 4, 4, 16, 64, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {16, 64, 4, 4, 64, 16, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {16, 16, 4, 4, 16, 16, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {64, 4, 16, 4, 4, 64, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {64, 8, 8, 4, 8, 64, use_spare_set};
    }
    else
    {
        MIOPEN_LOG_E("Only fp32, fp16, and bfp16 are supported");
        assert(false);
    }
    if(!tmp.IsValid(ctx))
    {
        MIOPEN_LOG_E("All attempts failed");
        assert(false);
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

std::tuple<int, int, int>
ConvHipImplicitGemmBwdDataV4R1Xdlops::CalculateGemmSize(const ConvolutionContext& ctx, int gemm_id)
{
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

    const auto gemm_m = c;
    const auto gemm_n = n * htilda_slice * wtilda_slice;
    const auto gemm_k = k * ydot_slice * xdot_slice;

    return std::make_tuple(gemm_m, gemm_n, gemm_k);
}

bool ConvHipImplicitGemmBwdDataV4R1Xdlops::IsApplicable(const ConvolutionContext& ctx) const
{
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

    bool is_applicable = true;
    int gemm_m         = 0;
    int gemm_n         = 0;
    std::tie(gemm_m, gemm_n, std::ignore) = CalculateGemmSize(ctx, 0);
    is_applicable = is_applicable && gemm_m % 32 == 0 && gemm_n % 32 == 0;
    for(int gemm_id = 0; gemm_id < CalculateNumberOfGemm(ctx); ++gemm_id)
    {
        int gemm_k = 0;
        std::tie(std::ignore, std::ignore, gemm_k) = CalculateGemmSize(ctx, gemm_id);
        is_applicable = is_applicable && gemm_k % 4 == 0;
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
    return c.IsValidValue() && c.IsValid(ctx);
}
PerformanceImplicitGemmBwdDataV4R1Xdlops
ConvHipImplicitGemmBwdDataV4R1Xdlops::Search(const ConvolutionContext& ctx) const
{
    return GenericSearchBwd(*this, ctx);
}

int ConvHipImplicitGemmBwdDataV4R1Xdlops::RunAndMeasureSolution(const miopen::Handle& profile_h,
                                                                ConstData_t bot_buf,
                                                                Data_t top_buf,
                                                                ConstData_t wei_buf,
                                                                ConstData_t bias_buf,
                                                                const ConvolutionContext&,
                                                                const ConvSolution& solution,
                                                                float& elapsed_time) const
{
    assert(bias_buf == nullptr);
    (void)bias_buf;

#ifdef NDEBUG
    try
#endif
    {

        elapsed_time = float(0);

        for(auto& k_info : solution.construction_params)
        {

            auto kernel = profile_h.AddKernel("",
                                              "",
                                              k_info.kernel_file,
                                              k_info.kernel_name,
                                              k_info.l_wk,
                                              k_info.g_wk,
                                              k_info.comp_options);

            kernel(bot_buf, wei_buf, top_buf);

            elapsed_time += profile_h.GetKernelTime();
        }
    }

#ifdef NDEBUG
    catch(miopen::Exception& ex)
    {
        MIOPEN_LOG_WE(ex.what());
        return -1;
    }
#endif
    return 0;
}

ConvSolution ConvHipImplicitGemmBwdDataV4R1Xdlops::GetSolution(
    const ConvolutionContext& ctx,
    const PerformanceImplicitGemmBwdDataV4R1Xdlops& config,
    bool) const
{
    ConvSolution result;

    assert(config.IsValid(ctx));

    // a series of kernels
    for(std::size_t gemm_id = 0; gemm_id < CalculateNumberOfGemm(ctx); ++gemm_id)
    {
        KernelInfo construction_parameters;

        int gemm_m = 0;
        int gemm_n = 0;
        int gemm_k = 0;

        std::tie(gemm_m, gemm_n, gemm_k) = CalculateGemmSize(ctx, gemm_id);

        // don't compile or launch an empty gridwise GEMM
        if(gemm_k > 0)
        {
            int grid_size = 0;

            const std::size_t GemmMPerBlock = config.GemmMPerBlock;
            const std::size_t GemmNPerBlock = config.GemmNPerBlock;
            const std::size_t GemmKPerBlock = config.GemmKPerBlock;
            const std::size_t GemmMPerWave  = config.GemmMPerWave;
            const std::size_t GemmNPerWave  = config.GemmNPerWave;

            const std::size_t block_size =
                GemmNPerBlock * GemmMPerBlock / (GemmMPerWave * GemmNPerWave) * wave_size;

            std::tie(grid_size, std::ignore) = config.CalculateGridSize(ctx);

            construction_parameters.l_wk.push_back(block_size);
            construction_parameters.l_wk.push_back(1);
            construction_parameters.l_wk.push_back(1);

            construction_parameters.g_wk.push_back(block_size * grid_size);
            construction_parameters.g_wk.push_back(1);
            construction_parameters.g_wk.push_back(1);

            if(ctx.group_counts > 1)
            {
                construction_parameters.kernel_file = "gridwise_convolution_backward_data_implicit_"
                                                      "gemm_v4r1_xdlops_gnchw_gkcyx_gnkhw.cpp";

                construction_parameters.kernel_name = "gridwise_convolution_backward_data_implicit_"
                                                      "gemm_v4r1_xdlops_gnchw_gkcyx_gnkhw";
            }
            else
            {
                construction_parameters.kernel_file = "gridwise_convolution_backward_data_implicit_"
                                                      "gemm_v4r1_xdlops_nchw_kcyx_nkhw.cpp";

                construction_parameters.kernel_name =
                    "gridwise_convolution_backward_data_implicit_gemm_v4r1_xdlops_nchw_kcyx_nkhw";
            }

            // TODO: add fp16 calculation by GetWorkspaceSize(ctx);
            result.workspce_sz = 0;

            int GemmABlockCopySrcDataPerRead_GemmM  = 1;
            int GemmABlockCopyDstDataPerWrite_GemmM = 1;
            int GemmBBlockCopySrcDataPerRead_GemmN  = 1;
            int GemmBBlockCopyDstDataPerWrite_GemmN = 1;
            int GemmABlockCopyClusterLengths_GemmK  = 0;
            int GemmABlockCopyClusterLengths_GemmM  = 0;
            int GemmBBlockCopyClusterLengths_GemmK  = 0;
            int GemmBBlockCopyClusterLengths_GemmN  = 0;

            std::tie(GemmABlockCopyClusterLengths_GemmK,
                     GemmABlockCopyClusterLengths_GemmM,
                     GemmABlockCopySrcDataPerRead_GemmM,
                     GemmABlockCopyDstDataPerWrite_GemmM,
                     std::ignore) = config.CalculateGemmABlockCopyPerformanceParameters(ctx);

            std::tie(GemmBBlockCopyClusterLengths_GemmK,
                     GemmBBlockCopyClusterLengths_GemmN,
                     GemmBBlockCopySrcDataPerRead_GemmN,
                     GemmBBlockCopyDstDataPerWrite_GemmN,
                     std::ignore) = config.CalculateGemmBBlockCopyPerformanceParameters(ctx);

            GemmABlockCopyDstDataPerWrite_GemmM =
                ctx.IsFp32() ? GemmABlockCopyDstDataPerWrite_GemmM : 1;
            GemmBBlockCopyDstDataPerWrite_GemmN =
                ctx.IsFp32() ? GemmBBlockCopyDstDataPerWrite_GemmN : 1;

            const auto GemmABlockCopyDstDataPerWrite_GemmKPACK =
                !ctx.IsFp32() ? config.GemmKPACKSize : 1;
            const auto GemmBBlockCopyDstDataPerWrite_GemmKPACK =
                !ctx.IsFp32() ? config.GemmKPACKSize : 1;

            // clang-format off
            construction_parameters.comp_options =
                std::string(" -std=c++14 ") +
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
                std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_M=") + std::to_string(GemmABlockCopySrcDataPerRead_GemmM) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K=") + std::to_string(GemmBBlockCopyClusterLengths_GemmK) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N=") + std::to_string(GemmBBlockCopyClusterLengths_GemmN) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_N=") + std::to_string(GemmBBlockCopySrcDataPerRead_GemmN) +
                std::string(" -DCK_PARAM_DEPENDENT_GRID_SIZE=") + std::to_string(grid_size) +
                std::string(" -DCK_USE_AMD_BUFFER_ATOMIC_ADD=") + (support_amd_buffer_atomic_add(ctx) ? '1' : '0') +
                std::string(" -DCK_USE_AMD_XDLOPS=") + std::to_string(IsXdlopsSupport(ctx) ? 1 : 0) +
                std::string(" -DCK_USE_AMD_XDLOPS_INLINE_ASM=") + std::to_string(miopen::IsEnabled(MIOPEN_DEBUG_IMPLICIT_GEMM_XDLOPS_INLINE_ASM{}) ? 1 : 0) +
                std::string(" -DCK_USE_AMD_XDLOPS_EMULATE=") + (miopen::IsEnabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS_EMULATE{}) ? '1' : '0') +
                std::string(" -DCK_PARAM_GEMM_ID=") + std::to_string(gemm_id) +
                ctx.general_compile_options;

            if(ctx.IsFp32())
            {
                construction_parameters.comp_options +=
                    std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_M=") + std::to_string(GemmABlockCopyDstDataPerWrite_GemmM) +
                    std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_N=") + std::to_string(GemmBBlockCopyDstDataPerWrite_GemmN);
            }
            else
            {
                construction_parameters.comp_options +=
                    std::string(" -DCK_PARAM_KPACK_LENGTH=") + std::to_string(config.GemmKPACKSize) +
                    std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_KPACK=") + std::to_string(GemmABlockCopyDstDataPerWrite_GemmKPACK) +
                    std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_KPACK=") + std::to_string(GemmBBlockCopyDstDataPerWrite_GemmKPACK);
            }

            result.construction_params.push_back(construction_parameters);

        }
    }
    result.invoker_factory = conv::MakeImplGemmDataInvokerFactory(ctx);
    return result;
}

} // namespace solver
} // namespace miopen
