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

#include "miopen/solver.hpp"
#include "miopen/handle.hpp"
#include <miopen/generic_search.hpp>
#include "miopen/stringutils.hpp"
#include "implicitgemm_util.hpp"
#include "miopen/implicitgemm_params.hpp"
#include <miopen/env.hpp>

namespace miopen {
namespace solver {

std::tuple<int, int, int, int, bool>
PerformanceImplicitGemmV4R4GenXdlopsWrWFp32::CalculateGemmABlockCopyPerformanceParameters(
    const ConvolutionContext& ctx) const
{
    int ClusterLengths_GemmK  = 0;
    int ClusterLengths_GemmM  = 0;
    int SrcDataPerRead_GemmK  = amd_buffer_load_max_length<float>();
    int DstDataPerWrite_GemmM = amd_lds_write_max_length<float>();

    try
    {
        const auto WaveSize = 64;
        const auto BlockSize =
            GemmNPerBlock * GemmMPerBlock / (GemmMPerWave * GemmNPerWave) * WaveSize;

        // calculate vector length on gemmk dimension
        SrcDataPerRead_GemmK = gcd(SrcDataPerRead_GemmK, GemmKPerBlock);

        // calculate threadwise copy size
        const auto a_data_per_thread_copy = (GemmKPerBlock * GemmMPerBlock) / BlockSize;

        if(!(a_data_per_thread_copy > 0))
            MIOPEN_THROW("invalid performance parameter");

        // GemmABlockCopySrcDataPerRead_GemmK also bounded by size of threadwise copy
        SrcDataPerRead_GemmK = gcd(SrcDataPerRead_GemmK, a_data_per_thread_copy);

        const auto ho = ConvolutionContextInterpreter::GetOutputHeightHo(ctx);
        const auto wo = ConvolutionContextInterpreter::GetOutputWidthWo(ctx);

        SrcDataPerRead_GemmK = gcd(SrcDataPerRead_GemmK, ho * wo);

        // decide threadwise copy lengths
        const auto a_data_per_thread_copy_gemmk = SrcDataPerRead_GemmK;
        const auto a_data_per_thread_copy_gemmm =
            a_data_per_thread_copy / a_data_per_thread_copy_gemmk;

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
                           SrcDataPerRead_GemmK,
                           DstDataPerWrite_GemmM,
                           true);
}

std::tuple<int, int, int, int, bool>
PerformanceImplicitGemmV4R4GenXdlopsWrWFp32::CalculateGemmBBlockCopyPerformanceParameters(
    const ConvolutionContext& ctx) const
{
    int ClusterLengths_GemmK  = 0;
    int ClusterLengths_GemmN  = 0;
    int SrcDataPerRead_GemmK  = amd_buffer_load_max_length<float>();
    int DstDataPerWrite_GemmN = amd_lds_write_max_length<float>();

    try
    {
        const auto WaveSize = 64;
        const auto BlockSize =
            GemmNPerBlock * GemmMPerBlock / (GemmMPerWave * GemmNPerWave) * WaveSize;

        SrcDataPerRead_GemmK = gcd(SrcDataPerRead_GemmK, GemmKPerBlock);

        const auto y = ConvolutionContextInterpreter::GetFilterHeightY(ctx);
        const auto x = ConvolutionContextInterpreter::GetFilterWidthX(ctx);

        const auto hi = ConvolutionContextInterpreter::GetInputHeightHi(ctx);
        const auto wi = ConvolutionContextInterpreter::GetInputWidthWi(ctx);
        const auto wo = ConvolutionContextInterpreter::GetOutputWidthWo(ctx);
        // calculate vector length on gemmn dimension
        const auto conv_stride_h =
            ConvolutionContextInterpreter::GetAdjustedConvolutionStrideH(ctx);
        const auto conv_stride_w =
            ConvolutionContextInterpreter::GetAdjustedConvolutionStrideW(ctx);
        const auto conv_dilation_w =
            ConvolutionContextInterpreter::GetAdjustedConvolutionDilationW(ctx);
        const auto in_left_pad_h  = ConvolutionContextInterpreter::GetInputLeftPadH(ctx);
        const auto in_left_pad_w  = ConvolutionContextInterpreter::GetInputLeftPadW(ctx);
        const auto in_right_pad_h = ConvolutionContextInterpreter::GetAdjustedInputRightPadH(ctx);
        const auto in_right_pad_w = ConvolutionContextInterpreter::GetAdjustedInputRightPadW(ctx);

        if(y == 1 && x == 1 && conv_stride_h == 1 && conv_stride_w == 1 && in_left_pad_h == 0 &&
           in_left_pad_w == 0 && in_right_pad_h == 0 && in_right_pad_w == 0)
        {
            // \todo there are more configs that can go through this if branch
            SrcDataPerRead_GemmK = gcd(SrcDataPerRead_GemmK, hi * wi);
        }
        else if(in_left_pad_w == 0 && in_right_pad_w == 0)
        {
            SrcDataPerRead_GemmK = gcd(SrcDataPerRead_GemmK, wo);
        }
        else if(conv_stride_w == 1)
        {
            SrcDataPerRead_GemmK =
                gcd(SrcDataPerRead_GemmK, in_left_pad_w, wi, in_right_pad_w, conv_dilation_w);
        }
        else
        {
            SrcDataPerRead_GemmK = 1;
        }

        // calculate threadwise copy size
        const auto b_data_per_thread_copy = (GemmKPerBlock * GemmNPerBlock) / BlockSize;

        if(!(b_data_per_thread_copy > 0))
            MIOPEN_THROW("invalid performance parameter");

        // GemmBBlockCopySrcDataPerRead_GemmK also bounded by size of threadwise copy
        SrcDataPerRead_GemmK = gcd(SrcDataPerRead_GemmK, b_data_per_thread_copy);

        const auto b_data_per_thread_copy_gemmk = SrcDataPerRead_GemmK;
        const auto b_data_per_thread_copy_gemmn =
            b_data_per_thread_copy / b_data_per_thread_copy_gemmk;

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
                           SrcDataPerRead_GemmK,
                           DstDataPerWrite_GemmN,
                           true);
}

std::tuple<std::size_t, bool> PerformanceImplicitGemmV4R4GenXdlopsWrWFp32::CalculateLdsNumberOfByte(
    const ConvolutionContext& ctx) const
{
    std::size_t lds_size = 0;

    try
    {
        bool valid = false;

        int GemmABlockCopyDescDataPerWriteGemmM = 0;
        int GemmABlockCopyClusterLengths_GemmM  = 0;
        std::tie(std::ignore,
                 GemmABlockCopyClusterLengths_GemmM,
                 std::ignore,
                 GemmABlockCopyDescDataPerWriteGemmM,
                 valid) = CalculateGemmABlockCopyPerformanceParameters(ctx);

        if(!valid)
            MIOPEN_THROW("invalid performance parameter");

        int GemmBBlockCopyDescDataPerWriteGemmN = 0;
        int GemmBBlockCopyClusterLengths_GemmN  = 0;
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

        lds_size = 2 * (a_block_space + b_block_space) * sizeof(float);
    }
    catch(...)
    {
        return std::make_tuple(0, false);
    }

    return std::make_tuple(lds_size, true);
}

bool PerformanceImplicitGemmV4R4GenXdlopsWrWFp32::IsValid(const ConvolutionContext& ctx) const
{
    const std::size_t n  = ConvolutionContextInterpreter::GetBatchN(ctx);
    const std::size_t k  = ConvolutionContextInterpreter::GetOutputChannelK(ctx) / ctx.group_counts;
    const std::size_t c  = ConvolutionContextInterpreter::GetInputChannelC(ctx) / ctx.group_counts;
    const std::size_t ho = ConvolutionContextInterpreter::GetOutputHeightHo(ctx);
    const std::size_t wo = ConvolutionContextInterpreter::GetOutputWidthWo(ctx);
    const std::size_t y  = ConvolutionContextInterpreter::GetFilterHeightY(ctx);
    const std::size_t x  = ConvolutionContextInterpreter::GetFilterWidthX(ctx);

    const std::size_t GemmM = k;
    const std::size_t GemmN = c * y * x;
    const std::size_t GemmK = n * ho * wo;

    // heuristic to reduce search space
    {
        // do not split GemmK if grid_size is large enough
        const auto GridSize = (GemmN / GemmNPerBlock) * (GemmM / GemmMPerBlock);
        if(GridSize > 256 && GemmKBlocks > 1)
            return false;

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

    if(!(GemmM % GemmMPerBlock == 0 && GemmN % GemmNPerBlock == 0 &&
         GemmK % (GemmKPerBlock * GemmKBlocks) == 0))
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

PerformanceImplicitGemmV4R4GenXdlopsWrWFp32::PerformanceImplicitGemmV4R4GenXdlopsWrWFp32(bool spare)
{
    GemmMPerBlock = 4;
    GemmNPerBlock = 16;
    GemmKPerBlock = 4;
    GemmKBlocks   = 1;

    GemmMPerWave = 4;
    GemmNPerWave = 16;

    use_spare_set = spare;
}

PerformanceImplicitGemmV4R4GenXdlopsWrWFp32::PerformanceImplicitGemmV4R4GenXdlopsWrWFp32(
    int GemmMPerBlock_,
    int GemmNPerBlock_,
    int GemmKPerBlock_,
    int GemmKBlocks_,
    int GemmMPerWave_,
    int GemmNPerWave_,
    bool use_spare_set_)
    : GemmMPerBlock(GemmMPerBlock_),
      GemmNPerBlock(GemmNPerBlock_),
      GemmKPerBlock(GemmKPerBlock_),
      GemmKBlocks(GemmKBlocks_),
      GemmMPerWave(GemmMPerWave_),
      GemmNPerWave(GemmNPerWave_),
      use_spare_set(use_spare_set_)
{
}

bool PerformanceImplicitGemmV4R4GenXdlopsWrWFp32::
operator==(const PerformanceImplicitGemmV4R4GenXdlopsWrWFp32& other) const
{
    // clang-format off
    return  GemmMPerBlock == other.GemmMPerBlock
        && GemmNPerBlock == other.GemmNPerBlock
        && GemmKPerBlock == other.GemmKPerBlock
        && GemmKBlocks == other.GemmKBlocks
        && GemmMPerWave == other.GemmMPerWave
        && GemmNPerWave == other.GemmNPerWave
        && use_spare_set == other.use_spare_set;
    // clang-format on
}

bool PerformanceImplicitGemmV4R4GenXdlopsWrWFp32::IsValidValue() const
{
    // clang-format off
    return
        IsTwoPower<4,128>(GemmMPerBlock)
        && IsTwoPower<16,128>(GemmNPerBlock)
        && IsTwoPower<4,16>(GemmKPerBlock)
        && IsTwoPower<1,64>(GemmKBlocks)
        && IsTwoPower<4,64>(GemmMPerWave)
        && IsTwoPower<16,64>(GemmNPerWave);
    // clang-format on
}

bool PerformanceImplicitGemmV4R4GenXdlopsWrWFp32::SetNextValue()
{
    do
    {
        if(!NextTwoPower<4, 128>(GemmMPerBlock))
            break;
        if(!NextTwoPower<16, 128>(GemmNPerBlock))
            break;
        if(!NextTwoPower<4, 16>(GemmKPerBlock))
            break;
        if(!NextTwoPower<1, 64>(GemmKBlocks))
            break;
        if(!NextTwoPower<4, 64>(GemmMPerWave))
            break;
        if(!NextTwoPower<16, 64>(GemmNPerWave))
            break;
        return false;
    } while(false);

    return true;
}

void PerformanceImplicitGemmV4R4GenXdlopsWrWFp32::EuristicInit(const ConvolutionContext& ctx)
{
    PerformanceImplicitGemmV4R4GenXdlopsWrWFp32 tmp;
    tmp = {128, 128, 16, 1, 64, 64, use_spare_set};

    if(!tmp.IsValid(ctx))
        tmp = {4, 64, 16, 1, 4, 64, use_spare_set};
    if(!tmp.IsValid(ctx))
        tmp = {8, 64, 8, 1, 8, 64, use_spare_set};
    if(!tmp.IsValid(ctx))
        tmp = {16, 16, 4, 1, 16, 16, use_spare_set};

    if(!tmp.IsValid(ctx))
    {
        MIOPEN_LOG_I("All attempts failed");
        assert(false);
    }
    *this = tmp;
    MIOPEN_LOG_I(ToString());
}

std::string PerformanceImplicitGemmV4R4GenXdlopsWrWFp32::ToString() const
{
    std::ostringstream ss;
    Serialize(ss);
    return ss.str();
}

PerformanceImplicitGemmV4R4GenXdlopsWrWFp32
ConvHipImplicitGemmV4R4GenXdlopsWrWFp32::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemmV4R4GenXdlopsWrWFp32>(ctx);
}

ConvSolution ConvHipImplicitGemmV4R4GenXdlopsWrWFp32::GetSolution(
    const ConvolutionContext& ctx,
    const PerformanceImplicitGemmV4R4GenXdlopsWrWFp32& config,
    bool) const
{
    ConvSolution result;
    KernelInfo construction_parameters;

    const std::size_t n  = ConvolutionContextInterpreter::GetBatchN(ctx);
    const std::size_t k  = ConvolutionContextInterpreter::GetOutputChannelK(ctx);
    const std::size_t c  = ConvolutionContextInterpreter::GetInputChannelC(ctx);
    const std::size_t ho = ConvolutionContextInterpreter::GetOutputHeightHo(ctx);
    const std::size_t wo = ConvolutionContextInterpreter::GetOutputWidthWo(ctx);
    const std::size_t y  = ConvolutionContextInterpreter::GetFilterHeightY(ctx);
    const std::size_t x  = ConvolutionContextInterpreter::GetFilterWidthX(ctx);
    const std::size_t hi = ConvolutionContextInterpreter::GetInputHeightHi(ctx);
    const std::size_t wi = ConvolutionContextInterpreter::GetInputWidthWi(ctx);
    const std::size_t conv_stride_h =
        ConvolutionContextInterpreter::GetAdjustedConvolutionStrideH(ctx);
    const std::size_t conv_stride_w =
        ConvolutionContextInterpreter::GetAdjustedConvolutionStrideW(ctx);
    const std::size_t conv_dilation_h =
        ConvolutionContextInterpreter::GetAdjustedConvolutionDilationH(ctx);
    const std::size_t conv_dilation_w =
        ConvolutionContextInterpreter::GetAdjustedConvolutionDilationW(ctx);
    const std::size_t in_left_pad_h = ConvolutionContextInterpreter::GetInputLeftPadH(ctx);
    const std::size_t in_left_pad_w = ConvolutionContextInterpreter::GetInputLeftPadW(ctx);
    const std::size_t in_right_pad_h =
        ConvolutionContextInterpreter::GetAdjustedInputRightPadH(ctx);
    const std::size_t in_right_pad_w =
        ConvolutionContextInterpreter::GetAdjustedInputRightPadW(ctx);

    const std::size_t GemmM = k;
    const std::size_t GemmN = c * y * x;

    const std::size_t block_size = config.GemmNPerBlock * config.GemmMPerBlock /
                                   (config.GemmMPerWave * config.GemmNPerWave) * wave_size;
    const std::size_t grid_size =
        (GemmN / config.GemmNPerBlock) * (GemmM / config.GemmMPerBlock) * config.GemmKBlocks;

    std::size_t lkl_wk0 = block_size;
    std::size_t lkl_wk1 = 1;
    std::size_t lkl_wk2 = 1;

    construction_parameters.l_wk.push_back(lkl_wk0);
    construction_parameters.l_wk.push_back(lkl_wk1);
    construction_parameters.l_wk.push_back(lkl_wk2);

    std::size_t gbl_wk0 = lkl_wk0 * grid_size;
    std::size_t gbl_wk1 = 1;
    std::size_t gbl_wk2 = 1;

    construction_parameters.g_wk.push_back(gbl_wk0);
    construction_parameters.g_wk.push_back(gbl_wk1);
    construction_parameters.g_wk.push_back(gbl_wk2);

    int GemmABlockCopyClusterLengths_GemmK  = 0;
    int GemmABlockCopyClusterLengths_GemmM  = 0;
    int GemmABlockCopySrcDataPerRead_GemmK  = 0;
    int GemmABlockCopyDstDataPerWrite_GemmM = 0;
    int GemmBBlockCopyClusterLengths_GemmK  = 0;
    int GemmBBlockCopyClusterLengths_GemmN  = 0;
    int GemmBBlockCopySrcDataPerRead_GemmK  = 0;
    int GemmBBlockCopyDstDataPerWrite_GemmN = 0;

    std::tie(GemmABlockCopyClusterLengths_GemmK,
             GemmABlockCopyClusterLengths_GemmM,
             GemmABlockCopySrcDataPerRead_GemmK,
             GemmABlockCopyDstDataPerWrite_GemmM,
             std::ignore) = config.CalculateGemmABlockCopyPerformanceParameters(ctx);

    std::tie(GemmBBlockCopyClusterLengths_GemmK,
             GemmBBlockCopyClusterLengths_GemmN,
             GemmBBlockCopySrcDataPerRead_GemmK,
             GemmBBlockCopyDstDataPerWrite_GemmN,
             std::ignore) = config.CalculateGemmBBlockCopyPerformanceParameters(ctx);

    // clang-format off
    construction_parameters.kernel_file = "gridwise_convolution_implicit_gemm_v4r4_gen_xdlops_wrw_fp32_nchw_kcyx_nkhw_lds_double_buffer.cpp";
    construction_parameters.kernel_name = "gridwise_convolution_implicit_gemm_v4r4_gen_xdlops_wrw_fp32_nchw_kcyx_nkhw_lds_double_buffer";

    construction_parameters.comp_options =
        std::string(" -std=c++14 ") +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_FORWARD=") + std::to_string(0) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_DATA=") + std::to_string(0) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_WEIGHT=") + std::to_string(1) +
        std::string(" -DCK_PARAM_PROBLEM_N=") + std::to_string(n) +
        std::string(" -DCK_PARAM_PROBLEM_C=") + std::to_string(c) +
        std::string(" -DCK_PARAM_PROBLEM_K=") + std::to_string(k) +
        std::string(" -DCK_PARAM_PROBLEM_Y=") + std::to_string(y) +
        std::string(" -DCK_PARAM_PROBLEM_X=") + std::to_string(x) +
        std::string(" -DCK_PARAM_PROBLEM_HI=") + std::to_string(hi) +
        std::string(" -DCK_PARAM_PROBLEM_WI=") + std::to_string(wi) +
        std::string(" -DCK_PARAM_PROBLEM_HO=") + std::to_string(ho) +
        std::string(" -DCK_PARAM_PROBLEM_WO=") + std::to_string(wo) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_STRIDE_H=") + std::to_string(conv_stride_h) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_STRIDE_W=") + std::to_string(conv_stride_w) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DILATION_H=") + std::to_string(conv_dilation_h) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DILATION_W=") + std::to_string(conv_dilation_w) +
        std::string(" -DCK_PARAM_PROBLEM_LEFT_PAD_H=") + std::to_string(in_left_pad_h) +
        std::string(" -DCK_PARAM_PROBLEM_LEFT_PAD_W=") + std::to_string(in_left_pad_w) +
        std::string(" -DCK_PARAM_PROBLEM_RIGHT_PAD_H=") + std::to_string(in_right_pad_h) +
        std::string(" -DCK_PARAM_PROBLEM_RIGHT_PAD_W=") + std::to_string(in_right_pad_w) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_GROUP_COUNTS=") + std::to_string(ctx.group_counts) +
        std::string(" -DCK_PARAM_TUNABLE_BLOCK_SIZE=") + std::to_string(block_size) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_M_PER_BLOCK=") + std::to_string(config.GemmMPerBlock) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_N_PER_BLOCK=") + std::to_string(config.GemmNPerBlock) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_K_PER_BLOCK=") + std::to_string(config.GemmKPerBlock) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_K_BLOCKS=") + std::to_string(config.GemmKBlocks) +
        std::string(" -DCK_PARAM_DEPENDENT_GRID_SIZE=") + std::to_string(grid_size) +
        std::string(" -DCK_PARAM_GEMM_M_PER_WAVE=") + std::to_string(config.GemmMPerWave) +
        std::string(" -DCK_PARAM_GEMM_N_PER_WAVE=") + std::to_string(config.GemmNPerWave) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K=") + std::to_string(GemmBBlockCopyClusterLengths_GemmK) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N=") + std::to_string(GemmBBlockCopyClusterLengths_GemmN) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K=") + std::to_string(GemmABlockCopyClusterLengths_GemmK) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M=") + std::to_string(GemmABlockCopyClusterLengths_GemmM) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM=") + std::to_string(GemmBBlockCopySrcDataPerRead_GemmK) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM=") + std::to_string(GemmABlockCopySrcDataPerRead_GemmK) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_N=") + std::to_string(GemmBBlockCopyDstDataPerWrite_GemmN) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_M=") + std::to_string(GemmABlockCopyDstDataPerWrite_GemmM) +
        std::string(" -DCK_USE_AMD_XDLOPS=") + (IsXdlopsSupport(ctx) ? '1' : '0') +
        std::string(" -DCK_USE_AMD_XDLOPS_INLINE_ASM=") + (miopen::IsEnabled(MIOPEN_DEBUG_IMPLICIT_GEMM_XDLOPS_INLINE_ASM{}) ? '1' : '0') +
        std::string(" -DCK_USE_AMD_XDLOPS_EMULATE=") + (miopen::IsEnabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS_EMULATE{}) ? '1' : '0') +
        ctx.general_compile_options;
    // clang-format on

    result.construction_params.push_back(construction_parameters);
    return result;
}

int ConvHipImplicitGemmV4R4GenXdlopsWrWFp32::RunAndMeasureSolution(const miopen::Handle& profile_h,
                                                                   ConstData_t bot_buf,
                                                                   ConstData_t top_buf,
                                                                   Data_t wei_buf,
                                                                   ConstData_t bias_buf,
                                                                   const ConvolutionContext& ctx,
                                                                   const ConvSolution& solution,
                                                                   float& elapsed_time) const
{
    assert(bias_buf == nullptr);
    (void)bias_buf;

    return RunAndMeasureSolutionBase(
        profile_h, bot_buf, top_buf, wei_buf, ctx, solution, elapsed_time);
}

bool ConvHipImplicitGemmV4R4GenXdlopsWrWFp32::IsApplicable(const ConvolutionContext& ctx) const
{
/// \todo Fix and remove this workaround.
/// There are random failures with certain configs,
/// see https://github.com/ROCmSoftwarePlatform/MIOpen/pull/228
/// We can't trust this solver until the reason is found and fixed.
#if 1
    (void)ctx;
    return false;
#else
    if(!(ctx.IsFp32()))
        return false;
    if(!ctx.use_hip_kernels)
        return false;
    if(!ctx.direction.IsBackwardWrW())
        return false;
    if(!ctx.Is2d())
        return false;
    if(ctx.group_counts > 1)
        return false;

    const std::size_t n  = ConvolutionContextInterpreter::GetBatchN(ctx);
    const std::size_t k  = ConvolutionContextInterpreter::GetOutputChannelK(ctx) / ctx.group_counts;
    const std::size_t c  = ConvolutionContextInterpreter::GetInputChannelC(ctx) / ctx.group_counts;
    const std::size_t y  = ConvolutionContextInterpreter::GetFilterHeightY(ctx);
    const std::size_t x  = ConvolutionContextInterpreter::GetFilterWidthX(ctx);
    const std::size_t ho = ConvolutionContextInterpreter::GetOutputHeightHo(ctx);
    const std::size_t wo = ConvolutionContextInterpreter::GetOutputWidthWo(ctx);

    const std::size_t GemmM = k;
    const std::size_t GemmN = c * y * x;
    const std::size_t GemmK = n * ho * wo;

    return IsValidGridGemmXdlops(GemmM, GemmN, GemmK) && IsXdlopsSupport(ctx);
#endif
}

bool ConvHipImplicitGemmV4R4GenXdlopsWrWFp32::IsValidPerformanceConfig(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmV4R4GenXdlopsWrWFp32& c) const
{
    MIOPEN_LOG_I("");
    return c.IsValidValue() && c.IsValid(ctx);
}

PerformanceImplicitGemmV4R4GenXdlopsWrWFp32
ConvHipImplicitGemmV4R4GenXdlopsWrWFp32::Search(const ConvolutionContext& ctx) const
{
    return GenericSearchWrW(*this, ctx);
}

} // namespace solver
} // namespace miopen
