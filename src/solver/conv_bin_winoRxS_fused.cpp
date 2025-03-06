/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include <miopen/conv/compiled_in_parameters.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/env.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/tensors.hpp>
#include <miopen/fusion_plan.hpp>
#include <miopen/fusion/solvers.hpp>
#include <miopen/fusion/utils.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_AMD_FUSED_WINOGRAD)

/// \return v rounded up (towards +inf) to the nearest multiple of m.
/// Defined for positive values only.
static inline int Ceiling(const int v, const int m)
{
    assert(m > 0 && v >= 0);
    if(v % m != 0)
    {
        return (v / m + 1) * m;
    }
    return v;
}

namespace miopen {
namespace solver {
namespace fusion {

bool ConvBinWinogradRxSFused::IsApplicable(const FusionContext& context,
                                           const FusionDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_AMD_FUSED_WINOGRAD))
        return false;
    if(!context.use_asm_kernels)
        return false;
    if(!WinoCommonIsApplicable(context, problem))
        return false;

    const auto conv_problem = problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    const auto conv_ctx     = context.GetConvContext(conv_problem);

    const std::string name = conv_ctx.GetStream().GetDeviceName();
    if(name != "gfx803")
        return false;

    const auto W           = conv_problem.GetInWidth();
    const auto H           = conv_problem.GetInHeight();
    const auto C           = conv_problem.GetInChannels();
    const auto N           = conv_problem.GetInBatchSize();
    const auto K           = conv_problem.GetOutChannels();
    const auto y           = conv_problem.GetWeightsHeight();
    const auto x           = conv_problem.GetWeightsWidth();
    const auto OH          = conv_problem.GetOutHeight();
    const auto OW          = conv_problem.GetOutWidth();
    const auto pad_h       = conv_problem.GetPadH();
    const auto pad_w       = conv_problem.GetPadW();
    const auto group_count = conv_problem.GetGroupCount();

    size_t padded_y = 0;
    size_t padded_x = 0;

    if(conv_problem.IsTensorsCasted())
        return false;
    if(conv_problem.GetKernelStrideH() == 1)
    {
        if(y <= 3)
        {
            if(!(C % 2 == 0))
                return false;
            padded_y = 3;
            padded_x = Ceiling(x, 3);
        }
        else
        {
            padded_y = Ceiling(y, 6);
            padded_x = Ceiling(x, 3);
        }
    }
    else if(conv_problem.GetKernelStrideH() == 2)
    {
        padded_y = Ceiling(y, 6);
        if(x % 6 == 1)
            padded_x = Ceiling(x, 3);
        else
            padded_x = Ceiling(x, 6);
    }
    else
        return false;

    if(!(((padded_x / 3) * (padded_y * 3) * C) >= 18))
        return false;

    // clang-format off
    return conv_problem.GetKernelStrideH() == conv_problem.GetKernelStrideW()
        && conv_problem.GetDilationH() == 1
        && conv_problem.GetDilationW() == 1
        && (C * x * y) <= std::pow(2, 28)
        && (K * x * y) <= std::pow(2, 28)
        && (K * OH * OW) <= std::pow(2, 28)
        && (C *  H *  W) <= std::pow(2, 28)
        && y <= std::pow(2, 16)
        && x <= std::pow(2, 16)
        && pad_h <= std::pow(2, 16)
        && pad_w <= std::pow(2, 16)
        && OH <= std::pow(2, 16)
        && OW <= std::pow(2, 16)
        && H <= std::pow(2, 16)
        && W <= std::pow(2, 16)
        && C <= std::pow(2, 16)
        && K <= std::pow(2, 16)
        && N <= std::pow(2, 16)
        && group_count == 1;
    // clang-format on
}

ConvSolution ConvBinWinogradRxSFused::GetSolution(const FusionContext& context,
                                                  const FusionDescription& problem) const
{
    const auto conv_problem = problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    const auto conv_ctx     = context.GetConvContext(conv_problem);
    ConvSolution result;
    KernelInfo kernel;

    const auto n_groups = conv_ctx.GetStream().GetMaxComputeUnits();
    kernel.g_wk.push_back(512 * n_groups);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    kernel.l_wk.push_back(512);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    KernelBuildParameters options{
        {"ROCM_METADATA_VERSION", conv_ctx.rmv.UseV3() ? 5 : 4},
    };
    kernel.comp_options = options.GenerateFor(kbp::GcnAsm{});

    kernel.kernel_name = "miopenSp3AsmConvRxSU_CBA";
    if(conv_problem.GetKernelStrideH() == 1)
    {
        kernel.kernel_file = "conv_3x3_wheel_alpha_v9_2_7.s";
    }
    else if(conv_problem.GetKernelStrideH() == 2)
    {
        kernel.kernel_file = "conv_3x3_wheel_alpha_v9_2_7_stride_2.s";
    }
    result.construction_params.push_back(kernel);
    const auto& desc    = *problem.fusion_plan_desc;
    const int bias_idx  = GetOpIdx(desc.op_map, miopenFusionOpBiasForward);
    const int activ_idx = GetOpIdx(desc.op_map, miopenFusionOpActivForward);
    int N, C, H, W, K, n_groups_, out_H, out_W, R, S, pad_H, pad_W;
    GetCompiledInParameters(conv_ctx,
                            conv_problem,
                            &N,
                            &C,
                            &H,
                            &W,
                            &K,
                            &n_groups_,
                            &out_H,
                            &out_W,
                            &R,
                            &S,
                            &pad_H,
                            &pad_W);
    const int zero = 0;
    int flags      = [&]() {
        if(bias_idx != -1 && activ_idx != -1)
        {
            return (1 << 7) + (1 << 8);
        }
        else if(bias_idx != -1)
        {
            return (1 << 7);
        }
        else
        {
            return zero;
        }
    }();
    const miopenActivationMode_t activ_mode = [&]() {
        if(activ_idx != -1)
        {
            const auto& activ_op =
                dynamic_cast<ActivFwdFusionOpDescriptor&>(*desc.op_map[activ_idx]);
            return activ_op.activMode;
        }
        return miopenActivationPASTHRU;
    }();

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            const auto& launch_kernel = handle.Run(kernels[0]);
            const auto& invoke_ctx =
                primitive_parameters.CastTo<miopen::fusion::FusionInvokeParams>();
            const auto& bot_buf = invoke_ctx.in;
            const auto& wei_buf = dynamic_cast<miopen::fusion::ConvolutionOpInvokeParam&>(
                                      *invoke_ctx.op_args.params[0])
                                      .weights;
            const auto& top_buf = invoke_ctx.out;
            const auto bias_ptr = [&]() {
                if(bias_idx != -1)
                {
                    return dynamic_cast<miopen::fusion::BiasOpInvokeParam&>(
                               *invoke_ctx.op_args.params[1])
                        .bdata;
                }
                else
                    return static_cast<ConstData_t>(nullptr);
            }();
            float activ_alpha = [&]() {
                if(activ_idx != -1)
                {
                    const auto& activ_args = dynamic_cast<miopen::fusion::ActivationOpInvokeParam&>(
                        *invoke_ctx.op_args.params[activ_idx]);
                    if(activ_mode == miopenActivationLEAKYRELU)
                        return (static_cast<float>(activ_args.activAlpha));
                }
                return static_cast<float>(0.0);
            }();
            launch_kernel(N,
                          C,
                          H,
                          W,
                          K,
                          n_groups_, // Not related to group convolutions
                          flags,     // flags
                          zero,      // reserved
                          bot_buf,
                          wei_buf,
                          top_buf,
                          nullptr, // return_addr
                          R,
                          S,
                          pad_H,
                          pad_W,
                          out_H,
                          out_W,
                          bias_ptr,
                          activ_alpha // leaky relu alpha
            );
        };
    };
    return result;
}
float ConvBinWinogradRxSFused::GetWti(const FusionContext&, const FusionDescription& problem) const
{
    /// \ref Negative WTI values

    const auto conv_problem = problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    const auto x            = conv_problem.GetWeightsWidth();
    const auto y            = conv_problem.GetWeightsHeight();
    if(x == 3 && y == 3)
        return wti_approximate_worst * .005f;
    else
        return wti_approximate_worst * .475f;
}
} // namespace fusion
} // namespace solver
} // namespace miopen
