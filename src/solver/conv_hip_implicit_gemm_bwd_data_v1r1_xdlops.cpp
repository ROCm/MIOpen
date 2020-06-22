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
#include <miopen/generic_search.hpp>

#include "implicitgemm_util.hpp"

#include <cstddef>

namespace miopen {
namespace solver {

size_t ConvHipImplicitGemmBwdDataV1R1Xdlops::GetWorkspaceSize(const ConvolutionContext& ctx) const
{
    if(ctx.IsFp32())
        return 0;
    else
    {
        // In case of fp16/bfp16, because there is no atomic add ISA,
        // reduction via atomic add ISA is done via fp32. As a result,
        // workspace is computed with miopenFloat data type.
        // Later, a separate kernel is invoked that casts from fp32 to fp16/bfp16
        std::size_t n  = ConvolutionContextInterpreter::GetBatchN(ctx);
        std::size_t c  = ConvolutionContextInterpreter::GetInputChannelC(ctx);
        std::size_t hi = ConvolutionContextInterpreter::GetInputHeightHi(ctx);
        std::size_t wi = ConvolutionContextInterpreter::GetInputWidthWi(ctx);
        return n * c * hi * wi * miopen::GetTypeSize(miopenFloat);
    }
}

bool ConvHipImplicitGemmBwdDataV1R1Xdlops::IsApplicable(const ConvolutionContext& ctx) const
{
    if(!ctx.direction.IsBackwardData())
        return false;
    if(!ctx.use_hip_kernels)
        return false;
    if(!ctx.Is2d())
        return false;
    if(!(ctx.IsFp32() || ctx.IsFp16() || ctx.IsBfp16()))
        return false;
    return IsApplicableXdlops(ctx);
}

PerformanceImplicitGemmXdlops
ConvHipImplicitGemmBwdDataV1R1Xdlops::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemmXdlops>(ctx);
}

int ConvHipImplicitGemmBwdDataV1R1Xdlops::RunAndMeasureSolution(const miopen::Handle& profile_h,
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

    KernelInfo k_info = solution.construction_params[0];

#ifdef NDEBUG
    try
#endif
    {
        elapsed_time = std::numeric_limits<float>::max();
        auto kernel  = profile_h.AddKernel("",
                                          "",
                                          k_info.kernel_file,
                                          k_info.kernel_name,
                                          k_info.l_wk,
                                          k_info.g_wk,
                                          k_info.comp_options);

        kernel(bot_buf, wei_buf, top_buf);

        elapsed_time = profile_h.GetKernelTime();
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

PerformanceImplicitGemmXdlops
ConvHipImplicitGemmBwdDataV1R1Xdlops::Search(const ConvolutionContext& ctx) const
{
    // fp16/bfp16 uses fp32 workspace to leverage fp32 atomic add
    if(ctx.IsFp16() || ctx.IsBfp16())
        return GenericSearchBwd(*this, ctx, SearchTweak::WorkspaceInsteadOfXBuffer);
    else
        return GenericSearchBwd(*this, ctx);
}

bool ConvHipImplicitGemmBwdDataV1R1Xdlops::IsValidPerformanceConfig(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmXdlops& c) const
{
    MIOPEN_LOG_I("");
    return c.IsValidValue() && c.IsValid(ctx);
}

ConvSolution ConvHipImplicitGemmBwdDataV1R1Xdlops::GetSolution(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmXdlops& config, bool) const
{
    ConvSolution result;
    KernelInfo construction_parameters;

    const std::size_t n  = ConvolutionContextInterpreter::GetBatchN(ctx);
    const std::size_t k  = ConvolutionContextInterpreter::GetOutputChannelK(ctx);
    const std::size_t c  = ConvolutionContextInterpreter::GetInputChannelC(ctx);
    const std::size_t hi = ConvolutionContextInterpreter::GetInputHeightHi(ctx);
    const std::size_t wi = ConvolutionContextInterpreter::GetInputWidthWi(ctx);
    const std::size_t ho = ConvolutionContextInterpreter::GetOutputHeightHo(ctx);
    const std::size_t wo = ConvolutionContextInterpreter::GetOutputWidthWo(ctx);
    const std::size_t y  = ConvolutionContextInterpreter::GetFilterHeightY(ctx);
    const std::size_t x  = ConvolutionContextInterpreter::GetFilterWidthX(ctx);
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

    const std::size_t GemmM = (static_cast<std::size_t>(c) * y * x);
    const std::size_t GemmN = (static_cast<std::size_t>(n) * ho * wo);

    const std::size_t GemmMPerBlock = config.KPerBlock;
    const std::size_t GemmNPerBlock = config.BPerBlock;
    const std::size_t GemmKPerBlock = config.EPerBlock;
    const std::size_t GemmMPerWave  = config.GemmMPerWave;
    const std::size_t GemmNPerWave  = config.GemmNPerWave;

    const std::size_t block_size =
        GemmNPerBlock * GemmMPerBlock / (GemmMPerWave * GemmNPerWave) * wave_size;

    const std::size_t grid_size = (GemmM / GemmMPerBlock) * (GemmN / GemmNPerBlock);

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

    if(ctx.group_counts > 1)
    {
        construction_parameters.kernel_file =
            "gridwise_convolution_backward_data_implicit_gemm_v1r1_xdlops_gnchw_gkcyx_gnkhw.cpp";

        construction_parameters.kernel_name =
            "gridwise_convolution_backward_data_implicit_gemm_v1r1_xdlops_gnchw_gkcyx_gnkhw";
    }
    else
    {
        construction_parameters.kernel_file =
            "gridwise_convolution_backward_data_implicit_gemm_v1r1_xdlops_nchw_kcyx_nkhw.cpp";

        construction_parameters.kernel_name =
            "gridwise_convolution_backward_data_implicit_gemm_v1r1_xdlops_nchw_kcyx_nkhw";
    }

    result.workspce_sz = GetWorkspaceSize(ctx);

    const auto GemmABlockCopyClusterLengths_GemmM = config.WeiBlockCopyClusterLengths_K;
    const auto GemmBBlockCopyClusterLengths_GemmN = config.InBlockCopyClusterLengths_B;

    const auto ABlockCopySubLengths_GemmM = GemmMPerBlock / GemmABlockCopyClusterLengths_GemmM;
    const auto BBlockCopySubLengths_GemmN = GemmNPerBlock / GemmBBlockCopyClusterLengths_GemmN;

    const auto GemmABlockCopySrcDataPerRead_GemmM =
        GetReadWriteVectorSize(ABlockCopySubLengths_GemmM);
    const auto GemmBBlockCopySrcDataPerRead_GemmN =
        GetReadWriteVectorSize(gcd(GemmNPerBlock, ho * wo, BBlockCopySubLengths_GemmN));

    const auto GemmABlockCopyDstDataPerWrite_GemmM =
        ctx.IsFp32() ? GetReadWriteVectorSize(ABlockCopySubLengths_GemmM) : 1;
    const auto GemmBBlockCopyDstDataPerWrite_GemmN =
        ctx.IsFp32() ? GetReadWriteVectorSize(BBlockCopySubLengths_GemmN) : 1;
    const auto GemmABlockCopyDstDataPerWrite_GemmKPACK =
        !ctx.IsFp32() ? GetEPackLength(ctx, true) : 1;
    const auto GemmBBlockCopyDstDataPerWrite_GemmKPACK =
        !ctx.IsFp32() ? GetEPackLength(ctx, true) : 1;

    // clang-format off
    construction_parameters.comp_options =
        std::string(" -std=c++14 ") +
        std::string(" -DCK_PARAM_PROBLEM_N=") + std::to_string(n) +
        std::string(" -DCK_PARAM_PROBLEM_K=") + std::to_string(k) +
        std::string(" -DCK_PARAM_PROBLEM_C=") + std::to_string(c) +
        std::string(" -DCK_PARAM_PROBLEM_HI=") + std::to_string(hi) +
        std::string(" -DCK_PARAM_PROBLEM_WI=") + std::to_string(wi) +
        std::string(" -DCK_PARAM_PROBLEM_HO=") + std::to_string(ho) +
        std::string(" -DCK_PARAM_PROBLEM_WO=") + std::to_string(wo) +
        std::string(" -DCK_PARAM_PROBLEM_Y=") + std::to_string(y) +
        std::string(" -DCK_PARAM_PROBLEM_X=") + std::to_string(x) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_STRIDE_H=") + std::to_string(conv_stride_h) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_STRIDE_W=") + std::to_string(conv_stride_w) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DILATION_H=") + std::to_string(conv_dilation_h) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DILATION_W=") + std::to_string(conv_dilation_w) +
        std::string(" -DCK_PARAM_PROBLEM_IN_LEFT_PAD_H=") + std::to_string(in_left_pad_h) +
        std::string(" -DCK_PARAM_PROBLEM_IN_LEFT_PAD_W=") + std::to_string(in_left_pad_w) +
        std::string(" -DCK_PARAM_PROBLEM_IN_RIGHT_PAD_H=") + std::to_string(in_right_pad_h) +
        std::string(" -DCK_PARAM_PROBLEM_IN_RIGHT_PAD_W=") + std::to_string(in_right_pad_w) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_GROUP_COUNTS=") + std::to_string(ctx.group_counts) +
        std::string(" -DCK_PARAM_TUNABLE_BLOCK_SIZE=") + std::to_string(block_size) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_M_PER_BLOCK=") + std::to_string(GemmMPerBlock) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_N_PER_BLOCK=") + std::to_string(GemmNPerBlock) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_K_PER_BLOCK=") + std::to_string(GemmKPerBlock) +
        std::string(" -DCK_PARAM_GEMM_M_PER_WAVE=") + std::to_string(GemmMPerWave) +
        std::string(" -DCK_PARAM_GEMM_N_PER_WAVE=") + std::to_string(GemmNPerWave) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K=") + std::to_string(config.WeiBlockCopyClusterLengths_E) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M=") + std::to_string(config.WeiBlockCopyClusterLengths_K) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_M=") + std::to_string(GemmABlockCopySrcDataPerRead_GemmM) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K=") + std::to_string(config.InBlockCopyClusterLengths_E) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N=") + std::to_string(config.InBlockCopyClusterLengths_B) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_N=") + std::to_string(GemmBBlockCopySrcDataPerRead_GemmN) +
        std::string(" -DCK_PARAM_DEPENDENT_GRID_SIZE=") + std::to_string(grid_size) +
        std::string(" -DCK_THREADWISE_GEMM_USE_AMD_INLINE_ASM=") + (use_amd_inline_asm(ctx) ? '1' : '0') +
        std::string(" -DCK_USE_AMD_INLINE_ASM=") + (use_amd_inline_asm(ctx) ? '1' : '0') +
        std::string(" -DCK_USE_AMD_BUFFER_ATOMIC_ADD=") + (support_amd_buffer_atomic_add(ctx) ? '1' : '0') +
        std::string(" -DCK_USE_AMD_XDLOPS=") + std::to_string(IsXdlopsSupport(ctx) ? 1 : 0) +
        std::string(" -DCK_USE_AMD_XDLOPS_INLINE_ASM=") + std::to_string(miopen::IsEnabled(MIOPEN_DEBUG_IMPLICIT_GEMM_XDLOPS_INLINE_ASM{}) ? 1 : 0) +
        std::string(" -DCK_USE_AMD_XDLOPS_EMULATE=") + (miopen::IsEnabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS_EMULATE{}) ? '1' : '0') +
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
            std::string(" -DCK_PARAM_KPACK_LENGTH=") + std::to_string(GetEPackLength(ctx, true)) +
            std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_KPACK=") + std::to_string(GemmABlockCopyDstDataPerWrite_GemmKPACK) +
            std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_KPACK=") + std::to_string(GemmBBlockCopyDstDataPerWrite_GemmKPACK);
    }

    result.invoker_factory = conv::MakeImplGemmDataInvokerFactory(ctx);
    result.construction_params.push_back(construction_parameters);
    return result;
}

} // namespace solver
} // namespace miopen
