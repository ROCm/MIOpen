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

#include <miopen/solver.hpp>

#include <miopen/conv/invokers/impl_gemm.hpp>
#include <miopen/handle.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/implicitgemm_params.hpp>

#include "implicitgemm_util.hpp"

namespace miopen {
namespace solver {

bool PerformanceImplicitGemmForwardV4R4XdlopsFp16Bfp16::IsValid(const ConvolutionContext& ctx) const
{
    const auto n  = ConvolutionContextInterpreter::GetBatchN(ctx);
    const auto k  = ConvolutionContextInterpreter::GetOutputChannelK(ctx) / ctx.group_counts;
    const auto c  = ConvolutionContextInterpreter::GetInputChannelC(ctx) / ctx.group_counts;
    const auto ho = ConvolutionContextInterpreter::GetOutputHeightHo(ctx);
    const auto wo = ConvolutionContextInterpreter::GetOutputWidthWo(ctx);
    const auto y  = ConvolutionContextInterpreter::GetFilterHeightY(ctx);
    const auto x  = ConvolutionContextInterpreter::GetFilterWidthX(ctx);

    std::size_t GemmM, GemmN, GemmK;

    // GemmKPack = 4 for fp16
    if(ctx.IsFp16() && GemmKPack != 4)
        return false;

    // GemmKPack = 2, 4 for bfp16 fwd non-group
    if(ctx.direction.IsForward() && ctx.IsBfp16() && ctx.group_counts == 1 && GemmKPack != 4 &&
       GemmKPack != 2)
        return false;

    // GemmKPack = 2 for bfp16 fwd group
    if(ctx.direction.IsForward() && ctx.IsBfp16() && ctx.group_counts > 1 && GemmKPack != 2)
        return false;

    // GemmKPack = 2 for bfp16 bwd, wrw
    if(!ctx.direction.IsForward() && ctx.IsBfp16() && GemmKPack != 2)
        return false;

    if(c % GemmKPack != 0)
        return false;

    const auto nonVectorizedC = c / GemmKPack;
    GemmM                     = k;
    GemmN                     = static_cast<std::size_t>(n) * ho * wo;
    GemmK                     = static_cast<std::size_t>(nonVectorizedC) * y * x;

    if(!(GemmKPerBlock % GemmBBlockCopyClusterLengths_GemmK == 0 &&
         GemmKPerBlock % GemmABlockCopyClusterLengths_GemmK == 0 &&
         GemmNPerBlock % GemmBBlockCopyClusterLengths_GemmN == 0 &&
         GemmMPerBlock % GemmABlockCopyClusterLengths_GemmM == 0))
        return false;

    if(!(GemmM % GemmMPerBlock == 0 && GemmN % GemmNPerBlock == 0 &&
         GemmK % (GemmKPerBlock * GemmKSegment) == 0))
        return false; // wrong! cannot divice N evenly among thread

    // unsupported xdlops-gemm
    if(GemmMPerWave == 16 && GemmNPerWave == 32)
        return false;
    if(GemmMPerWave == 32 && GemmNPerWave == 16)
        return false;
    if(GemmMPerWave == 8 && GemmNPerWave != 64)
        return false;
    if(GemmMPerWave == 4 && GemmNPerWave != 64)
        return false;

    const auto WaveSize  = 64;
    const auto BlockSize = GemmNPerBlock * GemmMPerBlock / (GemmMPerWave * GemmNPerWave) * WaveSize;

    // fail with blockSize >= 512
    /// \todo fix the issue with blockSize >= 512
    if(BlockSize < 64 || BlockSize > 256)
        return false;

    if(BlockSize != GemmBBlockCopyClusterLengths_GemmK * GemmBBlockCopyClusterLengths_GemmN)
        return false;

    if(BlockSize != GemmABlockCopyClusterLengths_GemmM * GemmABlockCopyClusterLengths_GemmK)
        return false;

    if((GemmMPerBlock % GemmMPerWave) != 0 || (GemmNPerBlock % GemmNPerWave) != 0)
        return false;

    const auto GemmBBlockCopyThreadSliceLengths_GemmN =
        GemmNPerBlock / GemmBBlockCopyClusterLengths_GemmN;
    const auto GemmABlockCopyThreadSliceLengths_GemmM =
        GemmMPerBlock / GemmABlockCopyClusterLengths_GemmM;
    const auto lds_size = ComputeLDSRequiredSize(ctx,
                                                 GemmNPerBlock,
                                                 GemmMPerBlock,
                                                 GemmKPerBlock,
                                                 1,
                                                 1,
                                                 GemmBBlockCopyThreadSliceLengths_GemmN,
                                                 GemmABlockCopyThreadSliceLengths_GemmM,
                                                 GemmKPack);
    return lds_size <= 64 * 1024;
}

PerformanceImplicitGemmForwardV4R4XdlopsFp16Bfp16::
    PerformanceImplicitGemmForwardV4R4XdlopsFp16Bfp16(bool spare)
{
    GemmNPerBlock = spare ? 16 : 64;
    GemmMPerBlock = spare ? 4 : 64;
    GemmKPerBlock = spare ? 4 : 8;
    GemmKSegment  = 1;
    GemmKPack     = 1;

    GemmMPerWave = spare ? 4 : 64;
    GemmNPerWave = spare ? 16 : 64;

    GemmBBlockCopyClusterLengths_GemmK = 4;
    GemmBBlockCopyClusterLengths_GemmN = 4;

    GemmABlockCopyClusterLengths_GemmK = 2;
    GemmABlockCopyClusterLengths_GemmM = 4;

    use_spare_set = spare;
}

PerformanceImplicitGemmForwardV4R4XdlopsFp16Bfp16::
    PerformanceImplicitGemmForwardV4R4XdlopsFp16Bfp16(int GemmMPerBlock_,
                                                      int GemmNPerBlock_,
                                                      int GemmKPerBlock_,
                                                      int GemmKSegment_,
                                                      int GemmKPack_,
                                                      int GemmMPerWave_,
                                                      int GemmNPerWave_,
                                                      int GemmABlockCopyClusterLengths_GemmK_,
                                                      int GemmABlockCopyClusterLengths_GemmM_,
                                                      int GemmBBlockCopyClusterLengths_GemmK_,
                                                      int GemmBBlockCopyClusterLengths_GemmN_,
                                                      bool use_spare_set_)
    : GemmMPerBlock(GemmMPerBlock_),
      GemmNPerBlock(GemmNPerBlock_),
      GemmKPerBlock(GemmKPerBlock_),
      GemmKSegment(GemmKSegment_),
      GemmKPack(GemmKPack_),
      GemmMPerWave(GemmMPerWave_),
      GemmNPerWave(GemmNPerWave_),
      GemmABlockCopyClusterLengths_GemmK(GemmABlockCopyClusterLengths_GemmK_),
      GemmABlockCopyClusterLengths_GemmM(GemmABlockCopyClusterLengths_GemmM_),
      GemmBBlockCopyClusterLengths_GemmK(GemmBBlockCopyClusterLengths_GemmK_),
      GemmBBlockCopyClusterLengths_GemmN(GemmBBlockCopyClusterLengths_GemmN_),
      use_spare_set(use_spare_set_)
{
}

bool PerformanceImplicitGemmForwardV4R4XdlopsFp16Bfp16::
operator==(const PerformanceImplicitGemmForwardV4R4XdlopsFp16Bfp16& other) const
{
    // clang-format off
    return GemmMPerBlock == other.GemmMPerBlock
        && GemmNPerBlock == other.GemmNPerBlock
        && GemmKPerBlock == other.GemmKPerBlock
        && GemmKSegment == other.GemmKSegment
        && GemmKPack == other.GemmKPack 
        && GemmMPerWave == other.GemmMPerWave
        && GemmNPerWave == other.GemmNPerWave
        && GemmABlockCopyClusterLengths_GemmK == other.GemmABlockCopyClusterLengths_GemmK
        && GemmABlockCopyClusterLengths_GemmM == other.GemmABlockCopyClusterLengths_GemmM
        && GemmBBlockCopyClusterLengths_GemmK == other.GemmBBlockCopyClusterLengths_GemmK
        && GemmBBlockCopyClusterLengths_GemmN == other.GemmBBlockCopyClusterLengths_GemmN
        && use_spare_set == other.use_spare_set;
    // clang-format on
}

bool PerformanceImplicitGemmForwardV4R4XdlopsFp16Bfp16::IsValidValue() const
{
    // clang-format off
    return IsTwoPower<32,128>(GemmMPerBlock)
        && IsTwoPower<32,128>(GemmNPerBlock)
        && IsTwoPower<4,32>(GemmKPerBlock)
        && IsTwoPower<1,64>(GemmKSegment)
        && IsTwoPower<1,4>(GemmKPack)
        && IsTwoPower<4,64>(GemmMPerWave)
        && IsTwoPower<16,64>(GemmNPerWave)
        && IsTwoPower<2,16>(GemmABlockCopyClusterLengths_GemmK)
        && IsTwoPower<4,128>(GemmABlockCopyClusterLengths_GemmM)
        && IsTwoPower<4,16>(GemmBBlockCopyClusterLengths_GemmK)
        && IsTwoPower<4,32>(GemmBBlockCopyClusterLengths_GemmN);
}

bool PerformanceImplicitGemmForwardV4R4XdlopsFp16Bfp16::SetNextValue()
{
    do
    {
        if(!use_spare_set)
        {
            if(!NextTwoPower<64, 128>(GemmMPerBlock))
                break;
            if(!NextTwoPower<64, 128>(GemmNPerBlock))
                break;
            if(!NextTwoPower<4, 32>(GemmKPerBlock))
                break;
            if(!NextTwoPower<1, 4>(GemmKPack))
                break;
        }
        else
        {
            if(!NextTwoPower<4, 128>(GemmMPerBlock))
                break;
            if(!NextTwoPower<16, 128>(GemmNPerBlock))
                break;
            if(!NextTwoPower<4, 32>(GemmKPerBlock))
                break;
            if(!NextTwoPower<1, 4>(GemmKPack))
                break;
            if(!NextTwoPower<4, 64>(GemmMPerWave))
                break;
            if(!NextTwoPower<16, 64>(GemmNPerWave))
                break;
        }
        if(!NextTwoPower<1, 64>(GemmKSegment))
            break;
        if(!NextTwoPower<2, 16>(GemmABlockCopyClusterLengths_GemmK))
            break;
        if(!NextTwoPower<4, 128>(GemmABlockCopyClusterLengths_GemmM))
            break;
        if(!NextTwoPower<4, 16>(GemmBBlockCopyClusterLengths_GemmK))
            break;
        if(!NextTwoPower<4, 32>(GemmBBlockCopyClusterLengths_GemmN))
            break;
        return false;
    } while(false);

    return true;
}

void PerformanceImplicitGemmForwardV4R4XdlopsFp16Bfp16::EuristicInit(const ConvolutionContext& ctx)
{
    PerformanceImplicitGemmForwardV4R4XdlopsFp16Bfp16 tmp;
    if(ctx.IsBfp16())
    {
        tmp = {128, 128, 16, 1, 2, 64, 64, 8, 32, 4, 64, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {64, 32, 4, 1, 2, 32, 64, 4, 16, 2, 32, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {64, 32, 4, 1, 2, 32, 64, 4, 16, 4, 16, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {32, 64, 4, 1, 2, 64, 32, 4, 16, 4, 16, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {32, 32, 4, 1, 2, 32, 32, 4, 16, 2, 32, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {64, 16, 4, 1, 2, 16, 64, 4, 16, 4, 16, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {16, 64, 4, 1, 2, 64, 16, 4, 16, 4, 16, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {16, 16, 4, 1, 2, 16, 16, 4, 16, 4, 16, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {64, 4, 16, 1, 2, 4, 64, 16, 4, 16, 4, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {64, 8, 8, 1, 2, 8, 64, 4, 16, 8, 8, use_spare_set};
    }
    else if(ctx.IsFp16())
    {
        tmp = {128, 128, 16, 1, 4, 64, 64, 8, 32, 4, 64, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {64, 32, 4, 1, 4, 32, 64, 4, 16, 2, 32, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {64, 32, 4, 1, 4, 32, 64, 4, 16, 4, 16, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {32, 64, 4, 1, 4, 64, 32, 4, 16, 4, 16, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {32, 32, 4, 1, 4, 32, 32, 4, 16, 2, 32, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {64, 16, 4, 1, 4, 16, 64, 4, 16, 4, 16, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {16, 64, 4, 1, 4, 64, 16, 4, 16, 4, 16, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {16, 16, 4, 1, 4, 16, 16, 4, 16, 4, 16, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {64, 4, 16, 1, 4, 4, 64, 16, 4, 16, 4, use_spare_set};
        if(!tmp.IsValid(ctx))
            tmp = {64, 8, 8, 1, 4, 8, 64, 4, 16, 8, 8, use_spare_set};
    }
    else
    {
        MIOPEN_LOG_E("Only fp16, and bfp16 are supported");
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

std::string PerformanceImplicitGemmForwardV4R4XdlopsFp16Bfp16::ToString() const
{
    std::ostringstream ss;
    Serialize(ss);
    return ss.str();
}

PerformanceImplicitGemmForwardV4R4XdlopsFp16Bfp16
ConvHipImplicitGemmForwardV4R4XdlopsFp16Bfp16::GetPerformanceConfig(
    const ConvolutionContext& ctx) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemmForwardV4R4XdlopsFp16Bfp16>(ctx);
}

ConvSolution ConvHipImplicitGemmForwardV4R4XdlopsFp16Bfp16::GetSolution(
    const ConvolutionContext& ctx,
    const PerformanceImplicitGemmForwardV4R4XdlopsFp16Bfp16& config,
    bool) const
{
    ConvSolution result;
    KernelInfo construction_parameters;

    const int n  = KernelBatchN(ctx);
    const int k  = KernelOutputChannelK(ctx);
    const int ho = KernelOutputHeightHo(ctx);
    const int wo = KernelOutputWidthWo(ctx);

    auto gemm_m = k;
    auto gemm_n = (n * ho * wo);

    std::size_t block_size =
        (config.GemmNPerBlock * config.GemmMPerBlock) / (config.GemmMPerWave * config.GemmNPerWave) * wave_size;
    std::size_t grid_size = (gemm_m / config.GemmMPerBlock) * (gemm_n / config.GemmNPerBlock) * config.GemmKSegment;

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

    construction_parameters.kernel_file =
        "gridwise_convolution_forward_implicit_gemm_v4r4_xdlops_fp16_bfp16_nchw_kcyx_nkhw.cpp";

    construction_parameters.kernel_name =
        "gridwise_convolution_forward_implicit_gemm_v4r4_xdlops_fp16_bfp16_nchw_kcyx_nkhw";

    std::size_t BBlockCopySubLengths_GemmN =
        config.GemmNPerBlock / config.GemmBBlockCopyClusterLengths_GemmN;

    // Ensure that vectorized b is read via right alignment from global memory
    // Consider slicing window's stride and dilation to ensure global memory reads of B are aligned
    // with vector length.
    int BBlockCopySrcDataPerRead_GemmN = GetReadWriteVectorSize(BBlockCopySubLengths_GemmN);

    const int y              = KernelFilterHeightY(ctx);
    const int x              = KernelFilterWidthX(ctx);
    const int c              = KernelInputChannelC(ctx);
    const auto hi            = ConvolutionContextInterpreter::GetInputHeightHi(ctx);
    const auto wi            = ConvolutionContextInterpreter::GetInputWidthWi(ctx);
    const auto conv_stride_h = ConvolutionContextInterpreter::GetAdjustedConvolutionStrideH(ctx);
    const auto conv_stride_w = ConvolutionContextInterpreter::GetAdjustedConvolutionStrideW(ctx);
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
        BBlockCopySrcDataPerRead_GemmN = gcd(BBlockCopySrcDataPerRead_GemmN, hi * wi);
    }
    else if(conv_stride_w == 1)
    {
        BBlockCopySrcDataPerRead_GemmN =
            gcd(BBlockCopySrcDataPerRead_GemmN, in_left_pad_w, wi, in_right_pad_w, conv_dilation_w);
    }
    else
    {
        BBlockCopySrcDataPerRead_GemmN = 1;
    }

    std::size_t ABlockCopySrcDataPerRead_GemmKPACK = 1;

    // For fp16 non-group fwd cases, E = (C * Y * X)/EPack
    // Since C*Y*X are in contiguous memory, EPack extracted from it, could be vectorized.
    ABlockCopySrcDataPerRead_GemmKPACK = (c * y * x) % config.GemmKPack != 0 ? 1 : config.GemmKPack;

    const auto ABlockCopyDstDataPerWrite_GemmKPACK = config.GemmKPack;
    const auto BBlockCopyDstDataPerWrite_GemmKPACK = config.GemmKPack;

    // clang-format off
    construction_parameters.comp_options =
        std::string(" -std=c++14 ") +
        std::string(" -DCK_PARAM_PROBLEM_N=") + std::to_string(ctx.batch_sz) +
        std::string(" -DCK_PARAM_PROBLEM_K=") + std::to_string(ctx.n_outputs) +
        std::string(" -DCK_PARAM_PROBLEM_C=") + std::to_string(ctx.n_inputs) +
        std::string(" -DCK_PARAM_PROBLEM_Y=") + std::to_string(ctx.kernel_size_h) +
        std::string(" -DCK_PARAM_PROBLEM_X=") + std::to_string(ctx.kernel_size_w) +
        std::string(" -DCK_PARAM_PROBLEM_HI=") + std::to_string(ctx.in_height) +
        std::string(" -DCK_PARAM_PROBLEM_WI=") + std::to_string(ctx.in_width) +
        std::string(" -DCK_PARAM_PROBLEM_HO=") + std::to_string(ctx.out_height) +
        std::string(" -DCK_PARAM_PROBLEM_WO=") + std::to_string(ctx.out_width) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_STRIDE_H=") + std::to_string(ctx.kernel_stride_h) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_STRIDE_W=") + std::to_string(ctx.kernel_stride_w) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DILATION_H=") + std::to_string(ctx.kernel_dilation_h) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DILATION_W=") + std::to_string(ctx.kernel_dilation_w) +
        std::string(" -DCK_PARAM_PROBLEM_LEFT_PAD_H=") + std::to_string(in_left_pad_h) +
        std::string(" -DCK_PARAM_PROBLEM_LEFT_PAD_W=") + std::to_string(in_left_pad_w) +
        std::string(" -DCK_PARAM_PROBLEM_RIGHT_PAD_H=") + std::to_string(in_right_pad_h) +
        std::string(" -DCK_PARAM_PROBLEM_RIGHT_PAD_W=") + std::to_string(in_right_pad_w) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_K_BLOCKS=") + std::to_string(config.GemmKSegment) +
        std::string(" -DCK_PARAM_GEMM_KPACK_LENGTH=") + std::to_string(config.GemmKPack) +
        std::string(" -DCK_PARAM_TUNABLE_BLOCK_SIZE=") + std::to_string(block_size) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_N_PER_BLOCK=") + std::to_string(config.GemmNPerBlock) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_M_PER_BLOCK=") + std::to_string(config.GemmMPerBlock) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_K_PER_BLOCK=") + std::to_string(config.GemmKPerBlock) +
        std::string(" -DCK_PARAM_GEMM_M_PER_WAVE=") + std::to_string(config.GemmMPerWave) +
        std::string(" -DCK_PARAM_GEMM_N_PER_WAVE=") + std::to_string(config.GemmNPerWave) +
        std::string(" -DCK_PARAM_DEPENDENT_GRID_SIZE=") + std::to_string(grid_size) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K=") + std::to_string(config.GemmABlockCopyClusterLengths_GemmK) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M=") + std::to_string(config.GemmABlockCopyClusterLengths_GemmM) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_KPACK=") + std::to_string(ABlockCopySrcDataPerRead_GemmKPACK) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_KPACK=") + std::to_string(ABlockCopyDstDataPerWrite_GemmKPACK) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K=") + std::to_string(config.GemmBBlockCopyClusterLengths_GemmK) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N=") + std::to_string(config.GemmBBlockCopyClusterLengths_GemmN) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_N=") + std::to_string(BBlockCopySrcDataPerRead_GemmN) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_KPACK=") + std::to_string(BBlockCopyDstDataPerWrite_GemmKPACK) +
        std::string(" -DCK_USE_AMD_XDLOPS=") + std::to_string(IsXdlopsSupport(ctx) ? 1 : 0) +
        std::string(" -DCK_USE_AMD_XDLOPS_INLINE_ASM=") + std::to_string(miopen::IsEnabled(MIOPEN_DEBUG_IMPLICIT_GEMM_XDLOPS_INLINE_ASM{}) ? 1 : 0) +
        std::string(" -DCK_USE_AMD_XDLOPS_EMULATE=") + (miopen::IsEnabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS_EMULATE{}) ? '1' : '0') +
        ctx.general_compile_options;
    // clang-format on

    result.invoker_factory = conv::MakeImplGemmDataInvokerFactory(ctx);

    result.construction_params.push_back(construction_parameters);
    return result;
}

int ConvHipImplicitGemmForwardV4R4XdlopsFp16Bfp16::RunAndMeasureSolution(
    miopen::Handle& profile_h,
    ConstData_t bot_buf,
    Data_t top_buf,
    ConstData_t wei_buf,
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

bool ConvHipImplicitGemmForwardV4R4XdlopsFp16Bfp16::IsApplicable(
    const ConvolutionContext& ctx) const
{
    if(!(ctx.IsFp16() || ctx.IsBfp16()))
        return false;

    if(!ctx.direction.IsForward())
        return false;

    if(!ctx.Is2d())
        return false;

    return IsApplicableXdlops(ctx);
}

bool ConvHipImplicitGemmForwardV4R4XdlopsFp16Bfp16::IsValidPerformanceConfig(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmForwardV4R4XdlopsFp16Bfp16& c) const
{
    MIOPEN_LOG_I("");
    return c.IsValidValue() && c.IsValid(ctx);
}

PerformanceImplicitGemmForwardV4R4XdlopsFp16Bfp16
ConvHipImplicitGemmForwardV4R4XdlopsFp16Bfp16::Search(const ConvolutionContext& ctx) const
{
    return GenericSearchFwd(*this, ctx);
}

} // namespace solver
} // namespace miopen
