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

#include <miopen/solver.hpp>

#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/compiled_in_parameters.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/sequences.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/fusion/solvers.hpp>
#include <miopen/fusion/utils.hpp>

#include <boost/any.hpp>
#include <boost/optional.hpp>

#include <tuple>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_G1)

namespace miopen {
namespace solver {
namespace fusion {

bool ConvBinWinogradRxSf2x3g1Fused::IsApplicable(const FusionContext& context,
                                                 const FusionDescription& problem) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_G1{}))
        return false;
    if(!WinoCommonIsApplicable(context))
        return false;
    const miopen::ConvolutionContext conv_ctx =
        context.GetConvContext(0, miopen::conv::Direction::Forward, problem);
    const std::string name = conv_ctx.GetStream().GetDeviceName();
    if(name == "gfx900" || name == "gfx906" || name == "gfx908" || name == "gfx90a" ||
       name == "gfx1011" || name == "gfx1012" || name == "gfx1030" || name == "gfx1031")
    {
        const auto oH = conv_ctx.problem.conv_problem.GetOutHeight();
        const auto oW = conv_ctx.problem.conv_problem.GetOutWidth();
        if(oH * oW > std::pow(2, 23))
            return false;
    }
    else
        return false;
    if(conv_ctx.problem.kernel_stride_h > 2)
        return false;
    return true;
}

ConvSolution ConvBinWinogradRxSf2x3g1Fused::GetSolution(const FusionContext& context,
                                                        const FusionDescription& problem) const
{
    ConvSolution result;
    KernelInfo kernel;

    const auto conv_ctx = context.GetConvContext(0, miopen::conv::Direction::Forward, problem);

    const auto n_groups = conv_ctx.GetStream().GetMaxHardwareComputeUnits();
    const auto name     = conv_ctx.GetStream().GetDeviceName();
    const auto is_gfx9  = StartsWith(name, "gfx9");
    const auto is_gfx10 = StartsWith(name, "gfx10");
    size_t wg_size      = is_gfx9 ? 512 : 256;
    kernel.g_wk.push_back(wg_size * n_groups);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    kernel.l_wk.push_back(wg_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    KernelBuildParameters options{
        {"ROCM_METADATA_VERSION", 5},
    };
    kernel.comp_options = options.GenerateFor(kbp::GcnAsm{});
    if(!is_gfx9)
        kernel.comp_options += std::string(" -mcumode -mwavefrontsize64");
    const auto kernel_postfix = "_fp32_stride" + std::to_string(conv_ctx.problem.kernel_stride_h);
    kernel.kernel_file        = "Conv_Winograd_v21_1_3" + kernel_postfix + ".s";
    const std::string family  = [&]() {
        if(is_gfx9)
            return "gfx9";
        else if(is_gfx10)
            return "gfx10";
        return "";
    }();
    kernel.kernel_name = "miopenSp3AsmConv_v21_1_3_" + family + kernel_postfix;
    result.construction_params.push_back(kernel);
    const auto x = conv_ctx.problem.conv_problem.GetWeightsWidth();
    const auto y = conv_ctx.problem.conv_problem.GetWeightsHeight();
    if(x == 3 && y == 3)
        result.weight = 100;
    else
        result.weight = 5;
    const auto& desc    = *problem.fusion_plan_desc;
    const int bias_idx  = GetOpIdx(desc.op_map, miopenFusionOpBiasForward);
    const int activ_idx = GetOpIdx(desc.op_map, miopenFusionOpActivForward);
    int N, C, H, W, K, n_groups_, out_H, out_W, R, S, pad_H, pad_W;
    GetCompiledInParameters(context,
                            conv_ctx.problem,
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
            return (1 << 7) + (1 << 8);
        else if(bias_idx != -1)
            return (1 << 7);
        else
            return zero;
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
            auto zero_u64 = static_cast<uint64_t>(0);
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
                          static_cast<void*>(nullptr), // return_addr
                          R,
                          S,
                          pad_H,
                          pad_W,
                          out_H,
                          out_W,
                          bias_ptr,
                          activ_alpha, // leaky relu alpha
                          zero,        // reserved2", Other, zero_int),
                          zero_u64,    // d_offset", Other, zero_uint64),
                          zero_u64,    // f_offset", Other, zero_uint64),
                          zero_u64,    // o_offset", Other, zero_uint64),
                          zero_u64,    // b_offset", Other, zero_uint64),
                          zero,        // d_byte_stride_nk", InputTensorDesc, zero_int),
                          zero,        // d_byte_stride_c", InputTensorDesc, zero_int),
                          zero,        // d_byte_stride_h", InputTensorDesc, zero_int),
                          zero,        // d_byte_stride_w", InputTensorDesc, zero_int),
                          zero,        // f_byte_stride_nk", OpAttr, zero_int),
                          zero,        // f_byte_stride_c", OpAttr, zero_int),
                          zero,        // f_byte_stride_h", OpAttr, zero_int),
                          zero,        // f_byte_stride_w", OpAttr, zero_int),
                          zero,        // o_byte_stride_nk", OutputTensorDesc, zero_int),
                          zero,        // o_byte_stride_c", OutputTensorDesc, zero_int),
                          zero,        // o_byte_stride_h", OutputTensorDesc, zero_int),
                          zero,        // o_byte_stride_w", OutputTensorDesc, zero_int),
                          zero,        // group_count", OpAttr, zero_int),
                          zero,        // d_byte_stride_g", Other, zero_int),
                          zero,        // f_byte_stride_g", Other, zero_int),
                          zero         // o_byte_stride_g", Other, zero_int),
            );
        };
    };
    return result;
}

} // namespace fusion
} // namespace solver
} // namespace miopen
