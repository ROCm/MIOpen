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
#include <miopen/solver/ck_utility_common.hpp>
#include <cstddef>

#include "../composable_kernel/host/solver/include/conv_igemm_fwd_v4r4r4_xdlops_nhwc_kyxc_nhwk.hpp"

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V4R4R4_XDLOPS_NHWC)

namespace miopen {
namespace solver {
namespace ck_utility {

static inline auto get_ck_tunable_conv_igemm_fwd_v4r4r4_xdlops_nhwc_kyxc_nhwk(
    const PerformanceConvCkIgemmFwdV4r4r4XdlopsNhwc& config)
{
    return ck::driver::ConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk::GetTunableList()
        [config.ck_tunable_list_id];
}

} // namespace ck_utility

bool PerformanceConvCkIgemmFwdV4r4r4XdlopsNhwc::SetNextValue(const ConvolutionContext&)
{
    if(ck_tunable_list_id <
       ck::driver::ConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk::GetTunableList().size() - 1)
    {
        ck_tunable_list_id++;
        return true;
    }
    else
    {
        return false;
    }
}

bool PerformanceConvCkIgemmFwdV4r4r4XdlopsNhwc::IsValid(const ConvolutionContext& ctx) const
{
    auto compile_param = ck::driver::CompileParameterConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk{};
    bool found         = false;

    std::tie(compile_param, found) =
        ck::driver::ConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk::CalculateCompileParameterBasedOnTunable(
            ck_utility::get_ck_convolution_problem_descriptor(ctx),
            ck_utility::get_ck_tunable_conv_igemm_fwd_v4r4r4_xdlops_nhwc_kyxc_nhwk(*this));

    if(!found)
        return false;

    return ck::driver::ConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk::IsValidCompileParameter(
        ck_utility::get_ck_convolution_problem_descriptor(ctx), compile_param);
}

bool ConvCkIgemmFwdV4r4r4XdlopsNhwc::IsApplicable(const ConvolutionContext& ctx) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V4R4R4_XDLOPS_NHWC{}))
        return false;
    if(!ctx.use_hip_kernels)
        return false;
    if(!ck_utility::is_supported_xdlops(ctx.GetStream()))
        return false;
    if(!ctx.IsLayoutNHWC())
        return false;
    if(!ctx.direction.IsForward())
        return false;
    if(!ctx.Is2d())
        return false;
    if(!(ctx.IsFp32() or ctx.IsFp16()))
        return false;
    if(ctx.group_counts != 1)
        return false;

    {
        // this kernel use int32_t for memory offset, which covers 2GB of memory maximum
        constexpr auto max_index_range = static_cast<std::size_t>(INT32_MAX) + 1;

        if(!(ctx.bot_sz < max_index_range && ctx.weights_sz < max_index_range &&
             ctx.top_sz < max_index_range))
            return false;
    }

    return ck::driver::ConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk::IsApplicable(
        ck_utility::get_ck_convolution_problem_descriptor(ctx));
}

PerformanceConvCkIgemmFwdV4r4r4XdlopsNhwc
ConvCkIgemmFwdV4r4r4XdlopsNhwc::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    for(int i = 0; i < ck::driver::ConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk::GetTunableList().size();
        ++i)
    {
        if(IsValidPerformanceConfig(ctx, i))
        {
            return {i};
        }
    }

    MIOPEN_LOG_E("cannot find a valid performance config");

    return {-1};
}

bool ConvCkIgemmFwdV4r4r4XdlopsNhwc::IsValidPerformanceConfig(
    const ConvolutionContext& ctx, const PerformanceConvCkIgemmFwdV4r4r4XdlopsNhwc& config) const
{
    return config.IsValid(ctx);
}

ConvSolution
ConvCkIgemmFwdV4r4r4XdlopsNhwc::GetSolution(const ConvolutionContext& ctx,
                                            const PerformanceConvCkIgemmFwdV4r4r4XdlopsNhwc& config,
                                            bool) const
{
    ConvSolution sol;
    KernelInfo kernel0_info, kernel1_info;

    const auto ck_conv_problem_desc = ck_utility::get_ck_convolution_problem_descriptor(ctx);

    auto ck_compile_param = ck::driver::CompileParameterConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk{};

    std::tie(ck_compile_param, std::ignore) =
        ck::driver::ConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk::CalculateCompileParameterBasedOnTunable(
            ck_conv_problem_desc,
            ck_utility::get_ck_tunable_conv_igemm_fwd_v4r4r4_xdlops_nhwc_kyxc_nhwk(config));

    // kernel0: prepare
    {
        kernel0_info.kernel_file =
            "convolution_forward_implicit_gemm_v4r4r4_xdlops_nhwc_kyxc_nhwk.cpp";

        kernel0_info.kernel_name =
            "convolution_forward_implicit_gemm_v4r4r4_xdlops_nhwc_kyxc_nhwk_prepare";

        kernel0_info.l_wk = {1, 1, 1};
        kernel0_info.g_wk = {1, 1, 1};

        kernel0_info.comp_options = ck_compile_param.GetCompileParameterString() +
                                    ck_utility::get_ck_common_compiler_flag(ctx.GetStream());
    }

    // kernel1: compute
    {
        kernel1_info.kernel_file =
            "convolution_forward_implicit_gemm_v4r4r4_xdlops_nhwc_kyxc_nhwk.cpp";

        kernel1_info.kernel_name = "convolution_forward_implicit_gemm_v4r4r4_xdlops_nhwc_kyxc_nhwk";

        const auto block_size =
            std::size_t(ck::driver::ConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk::GetBlockSize(
                ck_conv_problem_desc, ck_compile_param));

        const auto grid_size =
            std::size_t(ck::driver::ConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk::GetGridSize(
                ck_conv_problem_desc, ck_compile_param));

        kernel1_info.l_wk = {block_size, 1, 1};
        kernel1_info.g_wk = {block_size * grid_size, 1, 1};

        kernel1_info.comp_options = ck_compile_param.GetCompileParameterString() +
                                    ck_utility::get_ck_common_compiler_flag(ctx.GetStream());
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
            kernel1(tensors.in, tensors.w, tensors.out, data_ctx.workSpace);

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

std::size_t ConvCkIgemmFwdV4r4r4XdlopsNhwc::GetWorkspaceSize(const ConvolutionContext& ctx) const
{
    return ck::driver::ConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk::GetMaxWorkSpaceSize(
        ck_utility::get_ck_convolution_problem_descriptor(ctx));
}

PerformanceConvCkIgemmFwdV4r4r4XdlopsNhwc
ConvCkIgemmFwdV4r4r4XdlopsNhwc::Search(const ConvolutionContext& ctx,
                                       const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, invoke_ctx);
}

} // namespace solver
} // namespace miopen
