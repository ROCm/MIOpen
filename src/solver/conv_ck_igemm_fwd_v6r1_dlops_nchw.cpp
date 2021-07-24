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

#include "../composable_kernel/host/driver_online/include/convolution_problem_descriptor.hpp"
#include "../composable_kernel/host/driver_online/include/compile_param_conv_igemm_fwd_v6r1_dlops_nchw_kcyx_nkhw.hpp"

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW)

namespace miopen {
namespace solver {

// TODO: move this to common header
static inline auto get_ck_convolution_problem_descriptor(const ConvolutionContext& ctx)
{
    auto tmp = ck::ConvolutionProblemDescriptor{};

    tmp.N             = ConvolutionContextInterpreter::GetBatchN(ctx);
    tmp.K             = ConvolutionContextInterpreter::GetOutputChannelK(ctx);
    tmp.C             = ConvolutionContextInterpreter::GetInputChannelC(ctx);
    tmp.Y             = ConvolutionContextInterpreter::GetFilterHeightY(ctx);
    tmp.X             = ConvolutionContextInterpreter::GetFilterWidthX(ctx);
    tmp.Hi            = ConvolutionContextInterpreter::GetInputHeightHi(ctx);
    tmp.Wi            = ConvolutionContextInterpreter::GetInputWidthWi(ctx);
    tmp.Ho            = ConvolutionContextInterpreter::GetOutputHeightHo(ctx);
    tmp.Wo            = ConvolutionContextInterpreter::GetOutputWidthWo(ctx);
    tmp.ConvStrideH   = ConvolutionContextInterpreter::GetAdjustedConvolutionStrideH(ctx);
    tmp.ConvStrideW   = ConvolutionContextInterpreter::GetAdjustedConvolutionStrideW(ctx);
    tmp.ConvDilationH = ConvolutionContextInterpreter::GetAdjustedConvolutionDilationH(ctx);
    tmp.ConvDilationW = ConvolutionContextInterpreter::GetAdjustedConvolutionDilationW(ctx);
    tmp.InLeftPadH    = ConvolutionContextInterpreter::GetInputLeftPadH(ctx);
    tmp.InLeftPadW    = ConvolutionContextInterpreter::GetInputLeftPadW(ctx);
    tmp.InRightPadH   = ConvolutionContextInterpreter::GetAdjustedInputRightPadH(ctx);
    tmp.InRightPadW   = ConvolutionContextInterpreter::GetAdjustedInputRightPadW(ctx);

    return tmp;
}

static inline auto get_ck_compile_param_conv_igemm_fwd_v6r1_dlops_nchw_kcyx_nkhw(
    const PerformanceConvCkIgemmFwdV6r1DlopsNchw& config)
{
    return ck::compile_param_list_conv_igemm_fwd_v6r1_dlops_nchw_kcyx_nkhw
        [config.ck_compile_param_list_id];
}

bool PerformanceConvCkIgemmFwdV6r1DlopsNchw::SetNextValue(const ConvolutionContext& ctx)
{
    if(ck_compile_param_list_id <
       ck::compile_param_list_conv_igemm_fwd_v6r1_dlops_nchw_kcyx_nkhw.size() - 1)
    {
        ck_compile_param_list_id++;
        return true;
    }
    else
    {
        return false;
    }
}

bool PerformanceConvCkIgemmFwdV6r1DlopsNchw::IsValid(const ConvolutionContext& ctx) const
{
    return ck::ConvIgemmFwdV6r1DlopsNchwKcyxNkhw::IsValidCompileParameter(
        get_ck_convolution_problem_descriptor(ctx),
        get_ck_compile_param_conv_igemm_fwd_v6r1_dlops_nchw_kcyx_nkhw(*this));
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

    return ck::ConvIgemmFwdV6r1DlopsNchwKcyxNkhw::IsApplicable(
        get_ck_convolution_problem_descriptor(ctx));
}

PerformanceConvCkIgemmFwdV6r1DlopsNchw
ConvCkIgemmFwdV6r1DlopsNchw::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    for(int i = 0; i < ck::compile_param_list_conv_igemm_fwd_v6r1_dlops_nchw_kcyx_nkhw.size(); ++i)
    {
        if(IsValidPerformanceConfig(ctx, i))
        {
            return PerformanceConvCkIgemmFwdV6r1DlopsNchw(i);
        }
    }

    MIOPEN_LOG_E("cannot find a valid tuning parameter");
}

bool ConvCkIgemmFwdV6r1DlopsNchw::IsValidPerformanceConfig(
    const ConvolutionContext& ctx, const PerformanceConvCkIgemmFwdV6r1DlopsNchw& config) const
{
    return ck::ConvIgemmFwdV6r1DlopsNchwKcyxNkhw::IsValidCompileParameter(
        get_ck_convolution_problem_descriptor(ctx),
        get_ck_compile_param_conv_igemm_fwd_v6r1_dlops_nchw_kcyx_nkhw(config));
}

ConvSolution
ConvCkIgemmFwdV6r1DlopsNchw::GetSolution(const ConvolutionContext& ctx,
                                         const PerformanceConvCkIgemmFwdV6r1DlopsNchw config,
                                         bool disableConfigOverrideFromEnv) const
{
    ConvSolution sol;
    KernelInfo kernel0_info, kernel1_info;

    const auto ck_conv_problem_desc = get_ck_convolution_problem_descriptor(ctx);

    const auto ck_compile_param =
        get_ck_compile_param_conv_igemm_fwd_v6r1_dlops_nchw_kcyx_nkhw(config);

    // kernel0: prepare
    {
        kernel0_info.kernel_file =
            "dynamic_convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw.cpp";

        kernel0_info.kernel_name =
            "dynamic_convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw_prepare";

        kernel0_info.l_wk = {1, 1, 1};
        kernel0_info.g_wk = {1, 1, 1};

        kernel0_info.comp_options = ck_compile_param.GetCompileParameterString() +
                                    get_ck_common_compiler_flag(ctx) + ctx.general_compile_options;
    }

    // kernel1: compute
    {
        kernel1_info.kernel_file =
            "dynamic_convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw.cpp";

        kernel1_info.kernel_name =
            "dynamic_convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw";

        const auto block_size = std::size_t(ck::ConvIgemmFwdV6r1DlopsNchwKcyxNkhw::GetBlockSize(
            ck_conv_problem_desc, ck_compile_param));

        const auto grid_size = std::size_t(ck::ConvIgemmFwdV6r1DlopsNchwKcyxNkhw::GetGridSize(
            ck_conv_problem_desc, ck_compile_param));

        kernel1_info.l_wk = {block_size, 1, 1};
        kernel1_info.g_wk = {block_size * grid_size, 1, 1};

        kernel1_info.comp_options = ck_compile_param.GetCompileParameterString() +
                                    get_ck_common_compiler_flag(ctx) + ctx.general_compile_options;
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
            kernel0(ConvolutionContextInterpreter::GetBatchN(ctx),
                    ConvolutionContextInterpreter::GetInputChannelC(ctx),
                    ConvolutionContextInterpreter::GetInputHeightHi(ctx),
                    ConvolutionContextInterpreter::GetInputWidthWi(ctx),
                    ConvolutionContextInterpreter::GetOutputChannelK(ctx),
                    ConvolutionContextInterpreter::GetFilterHeightY(ctx),
                    ConvolutionContextInterpreter::GetFilterWidthX(ctx),
                    ConvolutionContextInterpreter::GetAdjustedConvolutionStrideH(ctx),
                    ConvolutionContextInterpreter::GetAdjustedConvolutionStrideW(ctx),
                    ConvolutionContextInterpreter::GetAdjustedConvolutionDilationH(ctx),
                    ConvolutionContextInterpreter::GetAdjustedConvolutionDilationW(ctx),
                    ConvolutionContextInterpreter::GetInputLeftPadH(ctx),
                    ConvolutionContextInterpreter::GetInputLeftPadW(ctx),
                    ConvolutionContextInterpreter::GetAdjustedInputRightPadH(ctx),
                    ConvolutionContextInterpreter::GetAdjustedInputRightPadW(ctx),
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
    // workspace is used for save transformed tensor descritpors created by prepare kernel
    return 4096L;
}

PerformanceConvCkIgemmFwdV6r1DlopsNchw
ConvCkIgemmFwdV6r1DlopsNchw::Search(const ConvolutionContext& ctx,
                                    const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, invoke_ctx);
}

} // namespace solver
} // namespace miopen
