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
#include <miopen/conv/solvers.hpp>
#include <miopen/handle.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/solver/ck_utility_common.hpp>
#include <miopen/solver/implicitgemm_util.hpp>
#include <cstddef>

#include "../composable_kernel/host/solver/include/conv_igemm_fwd_v6r1_dlops_nchw_kcyx_nkhw.hpp"

#define WORKAROUND_SWDEV_411729 1

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW)

namespace miopen {
namespace solver {
namespace ck_utility {

static inline auto get_ck_tunable_conv_igemm_fwd_v6r1_dlops_nchw_kcyx_nkhw(
    const conv::PerformanceConvCkIgemmFwdV6r1DlopsNchw& config)
{
    return ck::driver::ConvIgemmFwdV6r1DlopsNchwKcyxNkhw::GetTunableList()[config
                                                                               .ck_tunable_list_id];
}

} // namespace ck_utility

namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

bool PerformanceConvCkIgemmFwdV6r1DlopsNchw::SetNextValue(const ProblemDescription&)
{
    if(ck_tunable_list_id <
       ck::driver::ConvIgemmFwdV6r1DlopsNchwKcyxNkhw::GetTunableList().size() - 1)
    {
        ck_tunable_list_id++;
        return true;
    }
    else
    {
        return false;
    }
}

bool PerformanceConvCkIgemmFwdV6r1DlopsNchw::IsValid(const ProblemDescription& problem) const
{
    auto compile_param = ck::driver::CompileParameterConvIgemmFwdV6r1DlopsNchwKcyxNkhw{};
    bool found         = false;

    std::tie(compile_param, found) =
        ck::driver::ConvIgemmFwdV6r1DlopsNchwKcyxNkhw::CalculateCompileParameterBasedOnTunable(
            ck_utility::get_ck_convolution_problem_descriptor(problem),
            ck_utility::get_ck_tunable_conv_igemm_fwd_v6r1_dlops_nchw_kcyx_nkhw(*this));

    if(!found)
        return false;

    return ck::driver::ConvIgemmFwdV6r1DlopsNchwKcyxNkhw::IsValidCompileParameter(
        ck_utility::get_ck_convolution_problem_descriptor(problem), compile_param);
}

bool ConvCkIgemmFwdV6r1DlopsNchw::IsApplicable(const ExecutionContext& ctx,
                                               const ProblemDescription& problem) const
{
#if WORKAROUND_SWDEV_411729
    if(!env::enabled(MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW))
#else
    if(env::disabled(MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW))
#endif
    {
        return false;
    }
    if(ThisSolverIsDeprecatedStatic::IsDisabled(ctx))
        return false;
    if(!ctx.use_hip_kernels)
        return false;
    if(!ck_utility::is_ck_supported_hardware(ctx.GetStream()))
        return false;
    if(!problem.IsLayoutDefault())
        return false;
    if(!problem.IsDirectionForward())
        return false;
    if(!problem.Is2d())
        return false;
    if(!(problem.IsFp32() or problem.IsFp16()))
        return false;
    if(problem.HasNonPackedTensors())
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(problem.IsTensorsCasted())
        return false;
    if(problem.GetGroupCount() != 1)
        return false;
    if(ctx.GetStream().GetTargetProperties().Name() == "gfx90a" &&
       problem.IsGfx90aFp16altRequired())
        return false;
    if(!IsIndexRangeLargeEnough(problem))
        return false;

    return ck::driver::ConvIgemmFwdV6r1DlopsNchwKcyxNkhw::IsApplicable(
        ck_utility::get_ck_convolution_problem_descriptor(problem));
}

PerformanceConvCkIgemmFwdV6r1DlopsNchw
ConvCkIgemmFwdV6r1DlopsNchw::GetDefaultPerformanceConfig(const ExecutionContext& ctx,
                                                         const ProblemDescription& problem) const
{
    for(int i = 0; i < ck::driver::ConvIgemmFwdV6r1DlopsNchwKcyxNkhw::GetTunableList().size(); ++i)
    {
        if(IsValidPerformanceConfig(ctx, problem, i))
        {
            return {i};
        }
    }

    MIOPEN_LOG_E("cannot find a valid performance config");

    return {-1};
}

bool ConvCkIgemmFwdV6r1DlopsNchw::IsValidPerformanceConfig(
    const ExecutionContext&,
    const ProblemDescription& problem,
    const PerformanceConvCkIgemmFwdV6r1DlopsNchw& config) const
{
    return config.IsValid(problem);
}

ConvSolution
ConvCkIgemmFwdV6r1DlopsNchw::GetSolution(const ExecutionContext& ctx,
                                         const ProblemDescription& problem,
                                         const PerformanceConvCkIgemmFwdV6r1DlopsNchw& config) const
{
    ConvSolution sol;
    KernelInfo kernel0_info, kernel1_info;

    const auto ck_conv_problem_desc = ck_utility::get_ck_convolution_problem_descriptor(problem);

    auto ck_compile_param = ck::driver::CompileParameterConvIgemmFwdV6r1DlopsNchwKcyxNkhw{};

    std::tie(ck_compile_param, std::ignore) =
        ck::driver::ConvIgemmFwdV6r1DlopsNchwKcyxNkhw::CalculateCompileParameterBasedOnTunable(
            ck_conv_problem_desc,
            ck_utility::get_ck_tunable_conv_igemm_fwd_v6r1_dlops_nchw_kcyx_nkhw(config));

    // kernel0: prepare
    {
        kernel0_info.kernel_file =
            "convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw.cpp";

        kernel0_info.kernel_name =
            "convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw_prepare";

        kernel0_info.l_wk = {1, 1, 1};
        kernel0_info.g_wk = {1, 1, 1};

        kernel0_info.comp_options = ck_compile_param.GetCompileParameterString() +
                                    ck_utility::get_ck_common_compiler_flag(ctx.GetStream());
    }

    // kernel1: compute
    {
        kernel1_info.kernel_file =
            "convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw.cpp";

        kernel1_info.kernel_name = "convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw";

        const auto block_size =
            std::size_t(ck::driver::ConvIgemmFwdV6r1DlopsNchwKcyxNkhw::GetBlockSize(
                ck_conv_problem_desc, ck_compile_param));

        const auto grid_size =
            std::size_t(ck::driver::ConvIgemmFwdV6r1DlopsNchwKcyxNkhw::GetGridSize(
                ck_conv_problem_desc, ck_compile_param));

        kernel1_info.l_wk = {block_size, 1, 1};
        kernel1_info.g_wk = {block_size * grid_size, 1, 1};

        kernel1_info.comp_options = ck_compile_param.GetCompileParameterString() +
                                    ck_utility::get_ck_common_compiler_flag(ctx.GetStream());
    }

    sol.construction_params.push_back(kernel0_info);
    sol.construction_params.push_back(kernel1_info);

    // workspace is used to save transformed tensor descriptors
    sol.workspace_sz = GetWorkspaceSize(ctx, problem);

    sol.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            const auto& data_ctx = primitive_params.CastTo<miopen::conv::DataInvokeParams>();
            const auto& tensors  = data_ctx.tensors;
            auto kernel0         = handle.Run(kernels[0]);
            auto kernel1         = handle.Run(kernels[1]);

            float elapsed = 0;

            // kernel for transforming tensor descriptors
            kernel0(ck_conv_problem_desc.N,
                    ck_conv_problem_desc.C,
                    ck_conv_problem_desc.Hi,
                    ck_conv_problem_desc.Wi,
                    ck_conv_problem_desc.K,
                    ck_conv_problem_desc.Y,
                    ck_conv_problem_desc.X,
                    ck_conv_problem_desc.ConvStrideH,
                    ck_conv_problem_desc.ConvStrideW,
                    ck_conv_problem_desc.ConvDilationH,
                    ck_conv_problem_desc.ConvDilationW,
                    ck_conv_problem_desc.InLeftPadH,
                    ck_conv_problem_desc.InLeftPadW,
                    ck_conv_problem_desc.InRightPadH,
                    ck_conv_problem_desc.InRightPadW,
                    data_ctx.workSpace);

            if(handle.IsProfilingEnabled())
            {
                elapsed += handle.GetKernelTime();
            }

            // kernel for computation
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

std::size_t ConvCkIgemmFwdV6r1DlopsNchw::GetWorkspaceSize(const ExecutionContext&,
                                                          const ProblemDescription& problem) const
{
    return ck::driver::ConvIgemmFwdV6r1DlopsNchwKcyxNkhw::GetMaxWorkSpaceSize(
        ck_utility::get_ck_convolution_problem_descriptor(problem));
}

PerformanceConvCkIgemmFwdV6r1DlopsNchw
ConvCkIgemmFwdV6r1DlopsNchw::Search(const ExecutionContext& ctx,
                                    const ProblemDescription& problem,
                                    const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

} // namespace conv
} // namespace solver
} // namespace miopen
