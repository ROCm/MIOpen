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
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver.hpp>
#include <miopen/handle.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/solver/implicitgemm_util.hpp>

#include <cstddef>

#include "../composable_kernel/host/driver_online/include/compile_param_conv_igemm_fwd_v6r1_dlops_nchw_kcyx_nkhw.hpp"

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW)

namespace miopen {
namespace solver {

static inline auto get_compile_param_conv_igemm_fwd_v6r1_dlops_nchw_kcyx_nkhw(
    const PerformanceConvCkIgemmFwdV6r1DlopsNchw& config)
{
    return ck::kernel_compile_parameter::compile_param_list_conv_igemm_fwd_v6r1_dlops_nchw_kcyx_nkhw
        [config.compile_param_list_id];
}

bool PerformanceConvCkIgemmFwdV6r1DlopsNchw::SetNextValue(const ConvolutionContext& ctx)
{
    if(compile_param_list_id <
       ck::kernel_compile_parameter::compile_param_list_conv_igemm_fwd_v6r1_dlops_nchw_kcyx_nkhw
           .size())
    {
        compile_param_list_id++;
        return true;
    }
    else
    {
        return false;
    }
}

bool PerformanceConvCkIgemmFwdV6r1DlopsNchw::IsValid(const ConvolutionContext& ctx) const
{
    // TODO
    return true;
}

bool ConvCkIgemmFwdV6r1DlopsNchw::IsApplicable(const ConvolutionContext& ctx) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW{}))
        return false;
    if(!ctx.use_hip_kernels)
        return false;
    if(!ctx.IsLayoutDefault())
        return false;
    if(!IsComposableKernelSupportedHardware(ctx))
        return false;
    if(!ctx.direction.IsForward())
        return false;
    if(!ctx.Is2d())
        return false;
    if(!(ctx.IsFp32() or ctx.IsFp16()))
        return false;
    if(ctx.group_counts != 1)
        return false;

    return true;
}

PerformanceConvCkIgemmFwdV6r1DlopsNchw
ConvCkIgemmFwdV6r1DlopsNchw::GetPerformanceConfig(ConvolutionContext&) const
{
    // TODO
    return PerformanceConvCkIgemmFwdV6r1DlopsNchw(0);
}

bool ConvCkIgemmFwdV6r1DlopsNchw::IsValidPerformanceConfig(
    ConvolutionContext&, PerformanceConvCkIgemmFwdV6r1DlopsNchw&) const
{
    // TODO
    return true;
}

#if 0
ConvSolution
ConvCkIgemmFwdV6r1DlopsNchw::GetSolution(const ConvolutionContext& ctx,
                                         PerformanceConvCkIgemmFwdV6r1DlopsNchw config,
                                         bool disableConfigOverrideFromEnv) const;
#else
ConvSolution
ConvCkIgemmFwdV6r1DlopsNchw::GetSolution(const ConvolutionContext& ctx) const
#endif
{
    ConvSolution sol;
    KernelInfo kernel0_info, kernel1_info;

    const int N             = ConvolutionContextInterpreter::GetBatchN(ctx);
    const int K             = ConvolutionContextInterpreter::GetOutputChannelK(ctx);
    const int C             = ConvolutionContextInterpreter::GetInputChannelC(ctx);
    const int Y             = ConvolutionContextInterpreter::GetFilterHeightY(ctx);
    const int X             = ConvolutionContextInterpreter::GetFilterWidthX(ctx);
    const int Hi            = ConvolutionContextInterpreter::GetInputHeightHi(ctx);
    const int Wi            = ConvolutionContextInterpreter::GetInputWidthWi(ctx);
    const int Ho            = ConvolutionContextInterpreter::GetOutputHeightHo(ctx);
    const int Wo            = ConvolutionContextInterpreter::GetOutputWidthWo(ctx);
    const int ConvStrideH   = ConvolutionContextInterpreter::GetAdjustedConvolutionStrideH(ctx);
    const int ConvStrideW   = ConvolutionContextInterpreter::GetAdjustedConvolutionStrideW(ctx);
    const int ConvDilationH = ConvolutionContextInterpreter::GetAdjustedConvolutionDilationH(ctx);
    const int ConvDilationW = ConvolutionContextInterpreter::GetAdjustedConvolutionDilationW(ctx);
    const int InLeftPadH    = ConvolutionContextInterpreter::GetInputLeftPadH(ctx);
    const int InLeftPadW    = ConvolutionContextInterpreter::GetInputLeftPadW(ctx);
    const int InRightPadH   = ConvolutionContextInterpreter::GetAdjustedInputRightPadH(ctx);
    const int InRightPadW   = ConvolutionContextInterpreter::GetAdjustedInputRightPadW(ctx);

#if 1
    auto config = PerformanceConvCkIgemmFwdV6r1DlopsNchw(0);
#endif

    const auto compile_param = get_compile_param_conv_igemm_fwd_v6r1_dlops_nchw_kcyx_nkhw(config);

    const int N0 = compile_param.GN0;
    const int N1 = N / N0;

    const int C0 = compile_param.GK1;
    const int C1 = C / C0;

    const int GM0 = 1;
    const int GM1 = K;

    // GN0 is tunable
    const int GN1 = N1 * Ho * Wo;

    // GK1 is tuanble
    const int GK0 = C1 * Y * X;

    const int GM11 = compile_param.GM1PerBlockGM11;
    const int GN11 = compile_param.GN1PerBlockGN11;

    const int GM10 = GM1 / GM11;
    const int GN10 = GN1 / GN11;

    const bool hasMainKBlockLoop =
        ((GK0 + compile_param.GK0PerBlock) / (2 * compile_param.GK0PerBlock) > 1);
    const bool hasDoubleTailKBlockLoop = ((GK0 / compile_param.GK0PerBlock) % 2 == 0);

    // kernel0: prepare
    {
        kernel0_info.kernel_file =
            "dynamic_convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw.cpp";

        kernel0_info.kernel_name =
            "dynamic_convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw_prepare";

        kernel0_info.l_wk.push_back(1);
        kernel0_info.l_wk.push_back(1);
        kernel0_info.l_wk.push_back(1);

        kernel0_info.g_wk.push_back(1);
        kernel0_info.g_wk.push_back(1);
        kernel0_info.g_wk.push_back(1);

        // clang-format off
        kernel0_info.comp_options =
            compile_param.GetCompileParameterString() +
            " -DCK_PARAM_IN_WEI_DATATYPE=" + 
                std::string("70") + 
            " -DCK_PARAM_ACC_DATATYPE=" + 
                std::string("70") +
            " -DCK_PARAM_OUT_DATATYPE=" + 
                std::string("70") + 
            " -DCK_PARAM_HAS_MAIN_KBLOCK_LOOP=" +
                std::to_string(hasMainKBlockLoop) + 
            " -DCK_PARAM_HAS_DOUBLE_TAIL_KBLOCK_LOOP=" +
                std::to_string(hasDoubleTailKBlockLoop) + 
            get_ck_common_compiler_flag(ctx) +
            ctx.general_compile_options;
        // clang-format on
    }

    // kernel1: compute
    {
        kernel1_info.kernel_file =
            "dynamic_convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw.cpp";

        kernel1_info.kernel_name =
            "dynamic_convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw";

        const int grid_size = GM10 * GN10;

        kernel1_info.l_wk.push_back(compile_param.BlockSize);
        kernel1_info.l_wk.push_back(1);
        kernel1_info.l_wk.push_back(1);

        kernel1_info.g_wk.push_back(compile_param.BlockSize * grid_size);
        kernel1_info.g_wk.push_back(1);
        kernel1_info.g_wk.push_back(1);

        // clang-format off
        kernel1_info.comp_options =
            compile_param.GetCompileParameterString() +
            " -DCK_PARAM_IN_WEI_DATATYPE=" + 
                std::string("70") + 
            " -DCK_PARAM_ACC_DATATYPE=" + 
                std::string("70") +
            " -DCK_PARAM_OUT_DATATYPE=" + 
                std::string("70") + 
            " -DCK_PARAM_HAS_MAIN_KBLOCK_LOOP=" +
                std::to_string(hasMainKBlockLoop) + 
            " -DCK_PARAM_HAS_DOUBLE_TAIL_KBLOCK_LOOP=" +
                std::to_string(hasDoubleTailKBlockLoop) + 
            get_ck_common_compiler_flag(ctx) +
            ctx.general_compile_options;
        // clang-format on
    }

    sol.construction_params.push_back(kernel0_info);
    sol.construction_params.push_back(kernel1_info);

    // workspace is used to save transformed tensor descriptors
    sol.workspce_sz = GetWorkspaceSize(ctx);

    sol.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            const auto& data_ctx = primitive_params.CastTo<conv::DataInvokeParams>();
            const auto& tensors  = data_ctx.tensors;
            auto kernel0         = handle.Run(kernels[0]);
            auto kernel1         = handle.Run(kernels[1]);

            float elapsed = 0;

            // kernel for transforming tensor descriptors
            kernel0(N,
                    C,
                    Hi,
                    Wi,
                    K,
                    Y,
                    X,
                    ConvStrideH,
                    ConvStrideW,
                    ConvDilationH,
                    ConvDilationW,
                    InLeftPadH,
                    InLeftPadW,
                    InRightPadH,
                    InRightPadW,
                    data_ctx.workSpace);

            if(handle.IsProfilingEnabled())
            {
                elapsed += handle.GetKernelTime();
            }

            // kernel for computatition
            kernel1(tensors.w, tensors.in, tensors.out, data_ctx.workSpace);

            if(handle.IsProfilingEnabled())
            {
                elapsed += handle.GetKernelTime();
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };

    return sol;
}

std::size_t ConvCkIgemmFwdV6r1DlopsNchw::GetWorkspaceSize(const ConvolutionContext& ctx) const
{
    return 4096L;
}

} // namespace solver
} // namespace miopen
