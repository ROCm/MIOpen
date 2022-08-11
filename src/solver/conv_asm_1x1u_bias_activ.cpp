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

#include <sstream>
#include <limits>
#include <cassert>

#include <miopen/conv/fused_data_invoke_params.hpp>
#include <miopen/conv/tensors.hpp>
#include <miopen/env.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/fusion_plan.hpp>
#include <miopen/fusion/solvers.hpp>
#include <miopen/fusion/fusion_invoke_params.hpp>

#include "half.hpp"

using half_float::half;

namespace miopen {
namespace solver {

namespace fusion {

PerformanceConfigConvBiasActivAsm1x1U
ConvBiasActivAsm1x1U::GetDefaultPerformanceConfig(const ConvolutionContext& params) const
{
    std::ignore = params;
#if 0
    PerformanceConfigConvBiasActivAsm1x1U pp;
    pp.HeuristicInit(params);
    MIOPEN_LOG_I(pp.ToString());
#endif
    return {};
}

bool ConvBiasActivAsm1x1U::IsValidPerformanceConfig(
    const ConvolutionContext& problem, const PerformanceConfigConvBiasActivAsm1x1U& c) const
{
    return c.IsValidValue() && c.IsValid(problem);
}

PerformanceConfigConvBiasActivAsm1x1U
ConvBiasActivAsm1x1U::Search(const ConvolutionContext& context, const AnyInvokeParams&) const
{
    std::ignore = context;
    return {};
#if 0
    auto cba_context    = context;
    cba_context.bias    = 1;
    cba_context.bias_sz = cba_context.n_outputs * ((context.out_data_type == miopenHalf) ? 2 : 4);
    if(!context.direction.IsForward())
        MIOPEN_THROW("Only inference supported.");

    /// Workaround: Fused conv API does not pass user-allocated buffers here,
    /// but we need these buffers for search.
    auto& handle        = cba_context.GetStream();
    const auto bias_buf = handle.Create(cba_context.bias_sz);
    const auto in_buf   = handle.Create(cba_context.bot_sz);
    const auto wei_buf  = handle.Create(cba_context.weights_sz);
    const auto out_buf  = handle.Create(cba_context.top_sz);

    auto tensors    = FusedConvDataTensors{};
    tensors.in      = in_buf.get();
    tensors.w       = wei_buf.get();
    tensors.out     = out_buf.get();
    tensors.inDesc  = context.conv_problem.GetIn();
    tensors.wDesc   = context.conv_problem.GetWeights();
    tensors.outDesc = context.conv_problem.GetOut();
    tensors.bias    = bias_buf.get();
    PerformanceConfigConvBiasActivAsm1x1U pp;
    pp.HeuristicInit(context);
    return pp;
    // return GenericSearch(*this, cba_context, fused_invoke_ctx);
#endif
}

inline ConvolutionContext ConvOp2Ctx(const ExecutionContext& context,
                                     const ConvForwardOpDescriptor& conv_op)
{
    const auto dir = conv::Direction::Forward;
    TensorDescriptor out_desc;
    conv_op.GetOutputDesc(out_desc);
    auto ctx = ConvolutionContext{
        conv_op.input_desc, conv_op.filter_desc, out_desc, conv_op.base_desc /* conv desc */, dir};
    ctx.do_search                  = context.do_search;
    ctx.save_srch_req              = context.save_srch_req;
    ctx.use_asm_kernels            = context.use_asm_kernels;
    ctx.use_hip_kernels            = context.use_hip_kernels;
    ctx.use_opencl_convolutions    = context.use_opencl_convolutions;
    ctx.use_binaries               = context.use_binaries;
    ctx.disable_search_enforce     = context.disable_search_enforce;
    ctx.disable_perfdb_access      = context.disable_perfdb_access;
    ctx.use_dynamic_solutions_only = context.use_dynamic_solutions_only;
    ctx.general_compile_options    = "";

    ctx.SetStream(&context.GetStream());
    ctx.DetectRocm();
    ctx.SetupFloats();
    return ctx;
}

ConvSolution ConvBiasActivAsm1x1U::GetSolution(const ExecutionContext& context,
                                               const miopen::FusionPlanDescriptor& desc) const
/*
                                  const PerformanceConfigConvAsm1x1U& config,
                                  bool disableConfigOverrideFromEnv) const
*/
{
    const auto& conv_op = dynamic_cast<ConvForwardOpDescriptor&>(*desc.op_map[0]);
    auto ctx            = ConvOp2Ctx(context, conv_op);
#if 0
    std::ignore = config;
    std::ignore = disableConfigOverrideFromEnv;
#endif

    auto config = PerformanceConfigConvAsm1x1U{};
    config.HeuristicInit(ctx);

    auto sol = ConvAsm1x1U::GetSolution(ctx, config);

    if(sol.construction_params.size() != 1)
        MIOPEN_THROW("ConvBiasActivAsm1x1U expects only one kernel");

    auto& kernel_info       = sol.construction_params[0];
    kernel_info.kernel_file = "conv1x1u_bias_activ.s";

    const bool has_bias = [&]() {
        if(desc.op_map.size() == 3)
            return true;
        else if(desc.op_map[1]->kind() == miopenFusionOpBiasForward)
            return true;
        return false;
    }();
    const int activ_idx = [&]() {
        if(desc.op_map.size() == 3)
            return 2;
        else if(desc.op_map[1]->kind() == miopenFusionOpActivForward)
            return 1;
        return -1;
    }();

    std::ostringstream cba_options;
    GenerateClangDefsym(cba_options, "fusion_mode", 1);
    if(has_bias)
        GenerateClangDefsym(cba_options, "bias_mode", 1);
    if(activ_idx != -1)
    {
        GenerateClangDefsym(cba_options, "enable_activ", 1);
        const auto& activ_op = dynamic_cast<ActivFwdFusionOpDescriptor&>(*desc.op_map[activ_idx]);
        GenerateClangDefsym(cba_options, "activ_mode", static_cast<int>(activ_op.activMode));
    }
    kernel_info.comp_options += cba_options.str();

    const auto out_data_type = ctx.conv_problem.GetOutDataType();

    sol.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            const auto& kernel = handle.Run(kernels[0]);
            const auto& invoke_ctx =
                primitive_parameters.CastTo<miopen::fusion::FusionInvokeParams>();
            const auto& bot_ocl_buf = invoke_ctx.in;
            const auto& wei_ocl_buf =
                std::dynamic_pointer_cast<miopen::fusion::ConvolutionOpInvokeParam>(
                    invoke_ctx.op_invokers[0])
                    ->workspace;
            const auto& top_ocl_buf  = invoke_ctx.out;
            const auto& bias_ocl_buf = [&]() -> ConstData_t {
                if(has_bias)
                    return std::dynamic_pointer_cast<miopen::fusion::BiasOpInvokeParam>(
                               invoke_ctx.op_invokers[1])
                        ->bdata;
                else
                    return nullptr;
            }();

            if(activ_idx == -1) // skip the activation args
            {
                kernel(bot_ocl_buf, top_ocl_buf, wei_ocl_buf, bias_ocl_buf);
            }
            else
            {
                const auto& activ_invoker =
                    std::dynamic_pointer_cast<miopen::fusion::ActivationOpInvokeParam>(
                        invoke_ctx.op_invokers[activ_idx]);
                const auto activ_alpha = activ_invoker->activAlpha;
                const auto activ_beta  = activ_invoker->activBeta;
                const auto activ_gamma = activ_invoker->activGamma;
                if(out_data_type == miopenHalf)
                {
                    short unused = 0;
                    auto alpha   = half(activ_alpha);
                    auto beta    = half(activ_beta);
                    auto gamma   = half(activ_gamma);
                    kernel(alpha,
                           beta,
                           gamma,
                           unused,
                           bot_ocl_buf,
                           top_ocl_buf,
                           wei_ocl_buf,
                           bias_ocl_buf);
                }
                else
                {
                    int unused  = 0;
                    float alpha = activ_alpha;
                    float beta  = activ_beta;
                    float gamma = activ_gamma;
                    kernel(alpha,
                           beta,
                           gamma,
                           unused,
                           bot_ocl_buf,
                           top_ocl_buf,
                           wei_ocl_buf,
                           bias_ocl_buf);
                }
            }
        };
    };
    return sol;
}

bool ConvBiasActivAsm1x1U::IsApplicable(const ExecutionContext& context,
                                        const miopen::FusionPlanDescriptor& desc) const
{
    std::ignore = context;
    std::ignore = desc;
    if(desc.op_map.empty())
    {
        MIOPEN_THROW("");
    }
    // check the sequence of prims
    if(desc.op_map.size() > 3)
        return false;
    if(desc.op_map[0]->kind() != miopenFusionOpConvForward)
        return false;
    if(desc.op_map.size() >= 2)
    {
        const auto prim = desc.op_map[1]->kind();
        if(!(prim == miopenFusionOpBiasForward || prim == miopenFusionOpActivForward))
            return false;
    }
    if(desc.op_map.size() == 3)
    {
        const auto prim = desc.op_map[2]->kind();
        if(prim != miopenFusionOpActivForward)
            return false;
    }
    // Get the conv problem descriptor from the ops and pass it to the base class IsApplicable
    const auto& conv_op = dynamic_cast<ConvForwardOpDescriptor&>(*desc.op_map[0]);
    auto ctx            = ConvOp2Ctx(context, conv_op);

    // Check if the conovlution part is applicable
    if(!ConvAsm1x1U::IsApplicable(ctx))
        return false;
#if 0
    ctx.SetBufs(bufs);
#endif
    return true;
}

} // namespace fusion
} // namespace solver
} // namespace miopen
