/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#include <miopen/env.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/conv/compiled_in_parameters.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/tensors.hpp>

#include <boost/any.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_3X3)

namespace miopen {
namespace solver {

bool ConvBinWinograd3x3U::IsApplicable(const ExecutionContext& ctx,
                                       const ProblemDescription& problem) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_3X3{}))
        return false;
    if(!problem.Is2d())
        return false;
    if(!(problem.direction.IsForward() || problem.direction.IsBackwardData()))
        return false;
    if(!(ctx.rmv.IsV2orV3() && ctx.use_asm_kernels))
        return false;

    const auto target = ctx.GetStream().GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false;

    const auto name = ctx.GetStream().GetDeviceName();
    if(!(name == "gfx803" || name == "gfx900" || name == "gfx906" || name == "gfx908"))
        return false;

    // Check if kernel is suitable for the problem description
    // and able to correctly run with given parameters.
    const auto device_is_gfx8         = StartsWith(name, "gfx8");
    const auto grid_workgroup_count_x = ctx.GetStream().GetMaxComputeUnits();
    if(!problem.IsLayoutDefault())
    {
        return false;
    }

    // clang-format off
    return problem.pad_w == 1
        && problem.pad_h == 1
        && problem.kernel_size_w == 3
        && problem.kernel_size_h == 3
        && problem.kernel_stride_w == 1
        && problem.kernel_stride_h == 1
        && problem.kernel_dilation_w == 1
        && problem.kernel_dilation_h == 1
        && problem.batch_sz < std::pow(2, 16)
        && problem.n_inputs < std::pow(2, 16)
        && problem.n_outputs < std::pow(2, 16)
        && problem.in_height < std::pow(2, 16)
        && problem.in_width < std::pow(2, 16)
        && grid_workgroup_count_x < std::pow(2, 16)
        && (problem.n_inputs * problem.in_height * problem.in_width) <= std::pow(2, 28)
        && (problem.n_outputs * problem.in_height * problem.in_width) <= std::pow(2, 28)
        && (problem.n_inputs * problem.kernel_size_w * problem.kernel_size_h) <= std::pow(2, 28)
        && (problem.n_outputs * problem.kernel_size_w * problem.kernel_size_h) <= std::pow(2, 28)
        && problem.n_inputs % 2 == 0
        && problem.n_inputs >= (device_is_gfx8 ? 16 : 18)
        && problem.IsFp32()
        && problem.group_counts == 1
        && problem.in_layout == "NCHW";
        /// && (isForwardDirection() ? _weights_layout == "KCHW" : _weights_layout == "CKHW" )
        /// Actually, K<->C flpping is controlled by separate flag, so we can support either
        /// layout in both directions.
    // clang-format on
}

ConvSolution ConvBinWinograd3x3U::GetSolution(const ExecutionContext& ctx,
                                              const ProblemDescription& problem) const
{
    ConvSolution result;
    const auto n_groups = ctx.GetStream().GetMaxComputeUnits();
    const auto name     = ctx.GetStream().GetDeviceName();

    KernelInfo kernel;

    kernel.g_wk.clear();
    kernel.g_wk.push_back(512 * n_groups);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    kernel.l_wk.clear();
    kernel.l_wk.push_back(512);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    kernel.kernel_name = "miopenSp3AsmConv3x3F";

    KernelBuildParameters options{
        {"ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4},
    };
    kernel.comp_options = options.GenerateFor(kbp::GcnAsm{});

    if(StartsWith(name, "gfx8"))
        kernel.kernel_file = "conv_3x3_wheel_alpha_v3_0b.s";
    else if(StartsWith(name, "gfx9"))
        kernel.kernel_file = "conv_3x3_wheel_alpha_v7_0_3b.s";
    else
        MIOPEN_THROW("Unsupported device.");

    result.construction_params.push_back(kernel);

    const auto is_forward = problem.direction.IsForward();

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        constexpr int F_REVERSE_R = 1 << 0;
        constexpr int F_REVERSE_S = 1 << 1;
        constexpr int F_FLIP_K_C  = 1 << 2;
        // These are not used yet. Nevertheless let's keep as a shader documentation.
        // constexpr int F_FLIP_DATA_N_C = 1 << 3; // Unsupported in f3x2.
        // constexpr int F_FLIP_OUT_N_K = 1 << 4; // Unsupported in f3x2.
        // constexpr int L_F_ADDR_INDIRECT  = 1 << 6;
        // constexpr int L_F_BIAS  = 1 << 7;
        // constexpr int L_F_LEAKY_RELU  = 1 << 8;

        // not used in this particular kernel
        // constexpr int L_F_NKC_STRIDES = 1 << 9;

        int flags         = is_forward ? 0 : F_REVERSE_R + F_REVERSE_S + F_FLIP_K_C;
        int reserved      = 0;
        int* reserved_ptr = nullptr;
        int N, C, H, W, K, n_groups_, out_H, out_W, R, S, pad_H, pad_W;
        GetCompiledInParameters(
            ctx, problem, &N, &C, &H, &W, &K, &n_groups_, &out_H, &out_W, &R, &S, &pad_H, &pad_W);
        MIOPEN_LOG_I2(" N=" << N << " C=" << C << " H=" << H << " W=" << W << " K=" << K
                            << " n_groups=" << n_groups_ << " flags=" << flags << " R=" << R
                            << " S=" << S << " pad_H=" << pad_H << " pad_W=" << pad_W
                            << " out_H=" << out_H << " out_W=" << out_W);

        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            const auto k        = handle.Run(kernels[0]);
            const auto& fwd_ctx = primitive_params.CastTo<conv::DataInvokeParams>();
            const auto& tensors = fwd_ctx.tensors;

            k(N,
              C,
              H,
              W,
              K,
              n_groups_,
              flags,
              reserved,
              tensors.in,
              tensors.w,
              tensors.out,
              reserved_ptr);
        };
    };

    return result;
}
} // namespace solver
} // namespace miopen
