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

PerformanceImplicitGemmXdlops
ConvHipImplicitGemmV4R4GenFwdXdlops::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemmXdlops>(ctx);
}

static inline ConvSolution GetSolutionBase(const ConvolutionContext& ctx,
                                           const PerformanceImplicitGemmXdlops& config,
                                           const ImplicitGemmXdlopsKernel kernel,
                                           const int n,
                                           const int k,
                                           const int ho,
                                           const int wo)
{
    ConvSolution result;
    KernelInfo construction_parameters;

    // In bwd wrw, kernel's output height ho = y(filter's height) and output width wo=x(filter's
    // width)
    std::size_t b = (n * ho * wo);

    std::size_t GemmNPerBlock = config.BPerBlock;
    std::size_t GemmMPerBlock = config.KPerBlock;
    std::size_t GemmKPerBlock = config.EPerBlock;
    std::size_t GemmKBlocks   = config.EBlocks;

    std::size_t block_size =
        GemmNPerBlock * GemmMPerBlock / (config.GemmMPerWave * config.GemmNPerWave) * wave_size;
    std::size_t grid_size = (b / GemmNPerBlock) * (k / GemmMPerBlock) * GemmKBlocks;

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

    if(kernel == ImplicitGemmXdlopsKernel::KernelFwdWrw)
    {
        // clang-format off
        if(ctx.group_counts > 1)
        {

            construction_parameters.kernel_file =
                "gridwise_convolution_implicit_gemm_v4r4_gen_xdlops_gnchw_gkcyx_gnkhw_lds_double_buffer.cpp";

            construction_parameters.kernel_name =
		"gridwise_convolution_implicit_gemm_v4r4_gen_xdlops_gnchw_gkcyx_gnkhw_lds_double_buffer";
        }
        else
        {

            construction_parameters.kernel_file =
                "gridwise_convolution_implicit_gemm_v4r4_gen_xdlops_nchw_kcyx_nkhw_lds_double_buffer.cpp";

            construction_parameters.kernel_name =
		"gridwise_convolution_implicit_gemm_v4r4_gen_xdlops_nchw_kcyx_nkhw_lds_double_buffer";
        }
        // clang-format on
    }
    else
    {
        MIOPEN_THROW("invalid value of 'kernel'");
    }

    std::size_t ABlockCopySubLengths_GemmK = GemmKPerBlock / config.WeiBlockCopyClusterLengths_E;
    std::size_t ABlockCopySubLengths_GemmM = GemmMPerBlock / config.WeiBlockCopyClusterLengths_K;
    std::size_t BBlockCopySubLengths_GemmN = GemmNPerBlock / config.InBlockCopyClusterLengths_B;

    // Ensure that vectorized b is read via right alignment from global memory
    // Consider slicing window's stride and dilation to ensure global memory reads of B are aligned
    // with vector length.
    int BBlockCopySrcDataPerRead_GemmN = GetReadWriteVectorSize(BBlockCopySubLengths_GemmN);

    const int Y              = KernelFilterHeightY(ctx);
    const int X              = KernelFilterWidthX(ctx);
    const int C              = KernelInputChannelC(ctx);
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

    if(Y == 1 && X == 1 && conv_stride_h == 1 && conv_stride_w == 1 && in_left_pad_h == 0 &&
       in_left_pad_w == 0 && in_right_pad_h == 0 && in_right_pad_w == 0)
    {
        // \todo there are more configs that can go through this if branch
        BBlockCopySrcDataPerRead_GemmN = gcd(BBlockCopySrcDataPerRead_GemmN, hi * wi);
    }
    else if(conv_stride_w == 1 && conv_dilation_w == 1)
    {
        BBlockCopySrcDataPerRead_GemmN =
            gcd(BBlockCopySrcDataPerRead_GemmN, in_left_pad_w, wi, in_right_pad_w);
    }
    else
    {
        BBlockCopySrcDataPerRead_GemmN = 1;
    }

    std::size_t ABlockCopySrcDataPerRead_GemmK     = 1;
    std::size_t ABlockCopySrcDataPerRead_GemmKPACK = 1;
    if(ctx.IsFp32())
    {
        ABlockCopySrcDataPerRead_GemmK = GetReadWriteVectorSize(ABlockCopySubLengths_GemmK);
    }
    else
    {
        if(ctx.group_counts > 1)
        {
            // For bfp16/fp16 group fwd cases, E = C/EPack * Y * X, where C/EPack are in
            // continuous memory with Y*X because EPack is extracted from
            ABlockCopySrcDataPerRead_GemmK =
                (((C / config.EPACKSize) * Y * X) % ABlockCopySubLengths_GemmK) != 0
                    ? 1
                    : GetReadWriteVectorSize(ABlockCopySubLengths_GemmK);
        }
        else
        {
            // For fp16 non-group fwd cases, E = (C * Y * X)/EPack
            // Since C*Y*X are in contiguous memory, EPack extracted from it, could be vectorized.
            ABlockCopySrcDataPerRead_GemmKPACK =
                (C * Y * X) % config.EPACKSize != 0 ? 1 : config.EPACKSize;
        }
    }

    // For wrw cases in fp16/bfp16,
    // Since C/EPack are not in contiguous memory along with Y*X, vector length
    // needs to be in multiple of Y*X
    if(ctx.direction.IsBackwardWrW())
        ABlockCopySrcDataPerRead_GemmK = (Y * X) % ABlockCopySubLengths_GemmK != 0
                                             ? 1
                                             : GetReadWriteVectorSize(ABlockCopySubLengths_GemmK);

    ABlockCopySrcDataPerRead_GemmK =
        ctx.direction.IsBackwardData() ? 1 : ABlockCopySrcDataPerRead_GemmK;

    const auto ABlockCopyDstDataPerWrite_GemmM =
        ctx.IsFp32() ? GetReadWriteVectorSize(ABlockCopySubLengths_GemmM) : 1;
    const auto BBlockCopyDstDataPerWrite_GemmN =
        ctx.IsFp32() ? GetReadWriteVectorSize(BBlockCopySubLengths_GemmN) : 1;
    const auto ABlockCopyDstDataPerWrite_GemmKPACK = !ctx.IsFp32() ? config.EPACKSize : 1;
    const auto BBlockCopyDstDataPerWrite_GemmKPACK = !ctx.IsFp32() ? config.EPACKSize : 1;

    const ImplicitGemmDirection direction =
        ctx.direction.IsForward()
            ? ImplicitGemmDirection::ForwardData
            : (ctx.direction.IsBackwardData() ? ImplicitGemmDirection::BackwardData
                                              : ImplicitGemmDirection::BackwardWeight);

    if(ctx.direction.IsBackwardWrW())
    {
        // clang-format off
        construction_parameters.comp_options =
        std::string(" -DCK_PARAM_PROBLEM_K=") + std::to_string(ctx.n_inputs) + // swapped
        std::string(" -DCK_PARAM_PROBLEM_C=") + std::to_string(ctx.n_outputs) + // swapped
        std::string(" -DCK_PARAM_PROBLEM_HI=") + std::to_string(ctx.out_height) + // swapped
        std::string(" -DCK_PARAM_PROBLEM_WI=") + std::to_string(ctx.out_width) + // swapped
        std::string(" -DCK_PARAM_PROBLEM_HO=") + std::to_string(ctx.in_height) +  // swapped
        std::string(" -DCK_PARAM_PROBLEM_WO=") + std::to_string(ctx.in_width) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_FORWARD=") + std::to_string(0) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_DATA=") + std::to_string(0) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_WEIGHT=") + std::to_string(1);
        // clang-format on
    }
    else
    {
        // clang-format off
        construction_parameters.comp_options =
        std::string(" -DCK_PARAM_PROBLEM_K=") + std::to_string(ctx.n_outputs) +
        std::string(" -DCK_PARAM_PROBLEM_C=") + std::to_string(ctx.n_inputs) +
        std::string(" -DCK_PARAM_PROBLEM_HI=") + std::to_string(ctx.in_height) +
        std::string(" -DCK_PARAM_PROBLEM_WI=") + std::to_string(ctx.in_width) +
        std::string(" -DCK_PARAM_PROBLEM_HO=") + std::to_string(ctx.out_height) +
        std::string(" -DCK_PARAM_PROBLEM_WO=") + std::to_string(ctx.out_width) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_FORWARD=") + std::to_string(1) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_DATA=") + std::to_string(0) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_WEIGHT=") + std::to_string(0);
        // clang-format on
    }

    // Compute correct padding values
    std::size_t in_height  = ctx.in_height;
    std::size_t in_width   = ctx.in_width;
    std::size_t out_height = ctx.out_height;
    std::size_t out_width  = ctx.out_width;

    if(ctx.direction.IsBackwardWrW())
    {
        in_height  = ctx.out_height; // unswap
        in_width   = ctx.out_width;  // unswap
        out_height = ctx.in_height;  // unswap
        out_width  = ctx.in_width;   // unswap
    }

    // adjust padding size to align with the way MIOpen deal with padding
    std::size_t left_pad_h = ctx.pad_h;
    std::size_t left_pad_w = ctx.pad_w;

    std::size_t hi_padded = 1 + (ctx.kernel_size_h - 1) * ctx.kernel_dilation_h +
                            (out_height - 1) * ctx.kernel_stride_h;
    std::size_t wi_padded =
        1 + (ctx.kernel_size_w - 1) * ctx.kernel_dilation_w + (out_width - 1) * ctx.kernel_stride_w;

    std::size_t right_pad_h =
        hi_padded > (left_pad_h + in_height) ? hi_padded - (left_pad_h + in_height) : 0;
    std::size_t right_pad_w =
        wi_padded > (left_pad_w + in_width) ? wi_padded - (left_pad_w + in_width) : 0;

    // clang-format off
    construction_parameters.comp_options +=
        std::string(" -std=c++14 ") +
        std::string(" -DCK_PARAM_PROBLEM_DIRECTION=") + std::to_string(static_cast<int>(direction)) +
        std::string(" -DCK_PARAM_PROBLEM_N=") + std::to_string(ctx.batch_sz) +
        std::string(" -DCK_PARAM_PROBLEM_Y=") + std::to_string(ctx.kernel_size_h) +
        std::string(" -DCK_PARAM_PROBLEM_X=") + std::to_string(ctx.kernel_size_w) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_STRIDE_H=") + std::to_string(ctx.kernel_stride_h) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_STRIDE_W=") + std::to_string(ctx.kernel_stride_w) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DILATION_H=") + std::to_string(ctx.kernel_dilation_h) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DILATION_W=") + std::to_string(ctx.kernel_dilation_w) +
        std::string(" -DCK_PARAM_PROBLEM_LEFT_PAD_H=") + std::to_string(left_pad_h) +
        std::string(" -DCK_PARAM_PROBLEM_LEFT_PAD_W=") + std::to_string(left_pad_w) +
        std::string(" -DCK_PARAM_PROBLEM_RIGHT_PAD_H=") + std::to_string(right_pad_h) +
        std::string(" -DCK_PARAM_PROBLEM_RIGHT_PAD_W=") + std::to_string(right_pad_w) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_GROUP_COUNTS=") + std::to_string(ctx.group_counts) +
        std::string(" -DCK_PARAM_TUNABLE_BLOCK_SIZE=") + std::to_string(block_size) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_N_PER_BLOCK=") + std::to_string(GemmNPerBlock) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_M_PER_BLOCK=") + std::to_string(GemmMPerBlock) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_K_PER_BLOCK=") + std::to_string(GemmKPerBlock) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_K_BLOCKS=") + std::to_string(GemmKBlocks) +
        std::string(" -DCK_PARAM_DEPENDENT_GRID_SIZE=") + std::to_string(grid_size) +
        std::string(" -DCK_PARAM_GEMM_M_PER_WAVE=") + std::to_string(config.GemmMPerWave) +
        std::string(" -DCK_PARAM_GEMM_N_PER_WAVE=") + std::to_string(config.GemmNPerWave) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K=") + std::to_string(config.InBlockCopyClusterLengths_E) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N=") + std::to_string(config.InBlockCopyClusterLengths_B) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K=") + std::to_string(config.WeiBlockCopyClusterLengths_E) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M=") + std::to_string(config.WeiBlockCopyClusterLengths_K) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_N=") + std::to_string(BBlockCopySrcDataPerRead_GemmN) +
        std::string(" -DCK_PARAM_GEMM_KPACK_LENGTH=") + std::to_string(config.EPACKSize) +
        std::string(" -DCK_USE_AMD_XDLOPS=") + std::to_string(IsXdlopsSupport(ctx) ? 1 : 0) +
        std::string(" -DCK_USE_AMD_XDLOPS_INLINE_ASM=") + std::to_string(miopen::IsEnabled(MIOPEN_DEBUG_IMPLICIT_GEMM_XDLOPS_INLINE_ASM{}) ? 1 : 0) +
        std::string(" -DCK_USE_AMD_XDLOPS_EMULATE=") + (miopen::IsEnabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS_EMULATE{}) ? '1' : '0') +
        ctx.general_compile_options;
    // clang-format on

    if(ctx.IsFp32())
    {
        construction_parameters.comp_options +=
            std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_N=") +
            std::to_string(BBlockCopyDstDataPerWrite_GemmN) +
            std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_M=") +
            std::to_string(ABlockCopyDstDataPerWrite_GemmM) +
            std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_K=") +
            std::to_string(ABlockCopySrcDataPerRead_GemmK);
    }
    else
    {
        construction_parameters.comp_options +=
            std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_KPACK=") +
            std::to_string(BBlockCopyDstDataPerWrite_GemmKPACK) +
            std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_KPACK=") +
            std::to_string(ABlockCopyDstDataPerWrite_GemmKPACK);

        if(ctx.direction.IsBackwardWrW() || ctx.group_counts > 1)
        {
            construction_parameters.comp_options +=
                std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_K=") +
                std::to_string(ABlockCopySrcDataPerRead_GemmK);
        }
        else // only fwd non-group case
        {
            construction_parameters.comp_options +=
                std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_KPACK=") +
                std::to_string(ABlockCopySrcDataPerRead_GemmKPACK);
        }
    }

    if(ctx.direction.IsForward() || ctx.direction.IsBackwardData())
        result.invoker_factory = conv::MakeImplGemmDataInvokerFactory(ctx);

    result.construction_params.push_back(construction_parameters);
    return result;
}

ConvSolution ConvHipImplicitGemmV4R4GenFwdXdlops::GetSolution(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmXdlops& config, bool) const
{
    return GetSolutionBase(ctx,
                           config,
                           ImplicitGemmXdlopsKernel::KernelFwdWrw,
                           KernelBatchN(ctx),
                           KernelOutputChannelK(ctx),
                           KernelOutputHeightHo(ctx),
                           KernelOutputWidthWo(ctx));
}

ConvSolution ConvHipImplicitGemmV4R4GenWrWXdlops::GetSolution(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmXdlops& config, bool) const
{
    ConvSolution result = GetSolutionBase(ctx,
                                          config,
                                          ImplicitGemmXdlopsKernel::KernelFwdWrw,
                                          KernelBatchN(ctx),
                                          KernelOutputChannelK(ctx),
                                          KernelOutputHeightHo(ctx),
                                          KernelOutputWidthWo(ctx));

    result.workspce_sz = GetWorkspaceSize(ctx);
    return result;
}

int ConvHipImplicitGemmV4R4GenFwdXdlops::RunAndMeasureSolution(const miopen::Handle& profile_h,
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

int ConvHipImplicitGemmV4R4GenWrWXdlops::RunAndMeasureSolution(const miopen::Handle& profile_h,
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

bool ConvHipImplicitGemmV4R4GenFwdXdlops::IsApplicable(const ConvolutionContext& ctx) const
{
    if(!(ctx.IsFp16() || ctx.IsBfp16()))
        return false;
    if(!ctx.use_hip_kernels)
        return false;
    if(!ctx.direction.IsForward())
        return false;
    if(!ctx.Is2d())
        return false;
    return IsApplicableXdlops(ctx);
}

bool ConvHipImplicitGemmV4R4GenWrWXdlops::IsApplicable(const ConvolutionContext& ctx) const
{
    if(!(ctx.IsFp32() || ctx.IsFp16() || ctx.IsBfp16()))
        return false;
    if(!ctx.use_hip_kernels)
        return false;
    if(!ctx.direction.IsBackwardWrW())
        return false;
    if(!ctx.Is2d())
        return false;

    if(ConvHipImplicitGemmV4R4GenXdlopsWrWFp32{}.IsApplicable(ctx))
        return false;

    return IsApplicableXdlops(ctx);
}

PerformanceImplicitGemmXdlops
ConvHipImplicitGemmV4R4GenWrWXdlops::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemmXdlops>(ctx);
}

bool ConvHipImplicitGemmV4R4GenFwdXdlops::IsValidPerformanceConfig(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmXdlops& c) const
{
    MIOPEN_LOG_I("");
    return c.IsValidValue() && c.IsValid(ctx);
}

bool ConvHipImplicitGemmV4R4GenWrWXdlops::IsValidPerformanceConfig(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmXdlops& c) const
{
    MIOPEN_LOG_I("");
    return c.IsValidValue() && c.IsValid(ctx);
}

PerformanceImplicitGemmXdlops
ConvHipImplicitGemmV4R4GenFwdXdlops::Search(const ConvolutionContext& ctx) const
{
    return GenericSearchFwd(*this, ctx);
}

PerformanceImplicitGemmXdlops
ConvHipImplicitGemmV4R4GenWrWXdlops::Search(const ConvolutionContext& ctx) const
{
    // fp16/bfp16 uses fp32 workspace to leverage fp32 atomic add
    if(ctx.IsFp16() || ctx.IsBfp16())
        return GenericSearchWrW(*this, ctx, SearchTweak::WorkspaceInsteadOfWeightsBuffer);
    else
        return GenericSearchWrW(*this, ctx);
}

size_t ConvHipImplicitGemmV4R4GenWrWXdlops::GetWorkspaceSize(const ConvolutionContext& ctx) const
{
    if(ctx.IsFp32())
        return 0;
    else
    {
        // In case of fp16/bfp16, because there is no atomic add ISA,
        // reduction via atomic add ISA is done via fp32. As a result,
        // workspace is computed with miopenFloat data type.
        // Later, a separate kernel is invoked that casts from fp32 to fp16/bfp16
        std::size_t k = KernelBatchN(ctx);
        std::size_t c = KernelOutputChannelK(ctx);
        std::size_t y = ConvolutionContextInterpreter::GetFilterHeightY(ctx);
        std::size_t x = ConvolutionContextInterpreter::GetFilterWidthX(ctx);
        return k * c * y * x * miopen::GetTypeSize(miopenFloat);
    }
}

} // namespace solver
} // namespace miopen
