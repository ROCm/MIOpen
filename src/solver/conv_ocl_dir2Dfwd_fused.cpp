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

#include <miopen/handle.hpp>
#include <miopen/legacy_exhaustive_search.hpp>
#include <miopen/env.hpp>
#include <miopen/conv/tensors.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/fusion/solvers.hpp>
#include <miopen/conv/invokers/gen_x_w_y_pad.hpp>
#include <miopen/fusion/fusion_invoke_params.hpp>
#include <miopen/fusion_plan.hpp>
#include <miopen/fusion/utils.hpp>
#include <miopen/kernel_build_params.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD)

namespace miopen {
namespace solver {
namespace fusion {

PerformanceConfigConvOclDirectFwdFused
ConvOclDirectFwdFused::Search(const FusionContext& context,
                              const FusionDescription& problem,
                              const AnyInvokeParams& invoke_params) const
{
    const auto conv_problem          = problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    const auto conv_ctx              = context.GetConvContext(conv_problem);
    const auto legacy                = conv::ConvOclDirectFwd{};
    const auto& fusion_invoke_params = invoke_params.CastTo<miopen::fusion::FusionInvokeParams>();
    const auto wei_ocl_ptr           = dynamic_cast<miopen::fusion::ConvolutionOpInvokeParam&>(
                                 *fusion_invoke_params.op_args.params[0])
                                 .weights;
    const auto& tensors = miopen::ConvFwdTensors{fusion_invoke_params.inDesc,
                                                 fusion_invoke_params.in,
                                                 conv_problem.GetWeights(),
                                                 wei_ocl_ptr,
                                                 fusion_invoke_params.outDesc,
                                                 fusion_invoke_params.out};
    const auto data_invoke_params =
        miopen::conv::DataInvokeParams{tensors, nullptr, 0, fusion_invoke_params.gfx90aFp16alt};
    return legacy.Search(conv_ctx, conv_problem, data_invoke_params);
}

bool ConvOclDirectFwdFused::IsApplicable(const FusionContext& context,
                                         const FusionDescription& problem) const
{
    const auto& desc = *problem.fusion_plan_desc;
    if(desc.op_map.empty())
    {
        MIOPEN_THROW("No operators added to fusion plan");
    }
    // check the sequence of prims
    if(desc.op_map.size() > 4)
        return false;
    if(desc.op_map[0]->kind() != miopenFusionOpConvForward)
        return false;
    if(desc.op_map.size() >= 2)
    {
        const auto prim = desc.op_map[1]->kind();
        if(!(prim == miopenFusionOpBatchNormInference || prim == miopenFusionOpBiasForward ||
             prim == miopenFusionOpActivForward))
            return false;
    }
    if(desc.op_map.size() >= 3)
    {
        const auto prim = desc.op_map[2]->kind();
        if(!(prim == miopenFusionOpActivForward || prim == miopenFusionOpBatchNormInference))
            return false;
    }
    if(desc.op_map.size() == 4)
    {
        const auto prim = desc.op_map[3]->kind();
        if(!(prim == miopenFusionOpActivForward))
            return false;
    }
    const auto conv_problem = problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    if(!conv_problem.IsFp32())
        return false;
    const auto base     = conv::ConvOclDirectFwd{};
    const auto conv_ctx = context.GetConvContext(conv_problem);
    return base.IsApplicable(conv_ctx, conv_problem);
}

ConvSolution
ConvOclDirectFwdFused::GetSolution(const FusionContext& context,
                                   const FusionDescription& problem,
                                   const PerformanceConfigConvOclDirectFwdFused& config) const
{
    const auto conv_problem = problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    const auto conv_ctx     = context.GetConvContext(conv_problem);
    ConvSolution result = conv::ConvOclDirectFwd::BaseGetSolution(conv_ctx, conv_problem, config);

    if(result.construction_params.size() != 1)
        MIOPEN_THROW("ConvOclDirectFwdFused expects only one kernel");

    auto& kernel_info = result.construction_params[0];
    KernelBuildParameters build_params;
    kernel_info.kernel_file = "MIOpenConvDirBatchNormActiv.cl";
    kernel_info.kernel_name = "MIOpenConvUniBatchNormActiv";
    const auto& desc        = *problem.fusion_plan_desc;

    const int bias_idx  = GetOpIdx(desc.op_map, miopenFusionOpBiasForward);
    const int activ_idx = GetOpIdx(desc.op_map, miopenFusionOpActivForward);
    const int bn_idx    = GetOpIdx(desc.op_map, miopenFusionOpBatchNormInference);

    if(bias_idx != -1)
        build_params.Define("MLO_CONV_BIAS", 1);
    if(activ_idx != -1)
    {
        const auto& activ_op = dynamic_cast<ActivFwdFusionOpDescriptor&>(*desc.op_map[activ_idx]);
        build_params.Define("MIOPEN_YES_ACTIV", 1);
        build_params.Define("MIOPEN_NRN_OP_ID", static_cast<int>(activ_op.activMode));
    }
    if(bn_idx != -1)
    {
        const auto& bn_op =
            dynamic_cast<BatchNormInferenceFusionOpDescriptor&>(*desc.op_map[bn_idx]);

        std::vector<size_t> vld{256, 1, 1};
        if(bn_op.mode == miopenBNSpatial)
            build_params.Define("SPATIAL_BN");
        else if(bn_op.mode == miopenBNPerActivation)
            build_params.Define("PERACT_BN");
        int n, c, h, w;
        std::tie(n, c, h, w)   = tien<4>(bn_op.input_desc.GetLengths());
        size_t read_len        = (bn_op.mode == miopenBNSpatial) ? h * w : c * h * w;
        const size_t read_unit = [&]() {
            if(bn_op.mode == miopenBNSpatial && bn_op.input_desc.GetType() != miopenHalf)
                return (read_len % 4 == 0) ? 4 : (read_len % 2 == 0u) ? 2 : 1;
            else
                return 1;
        }();
        if(bn_op.input_desc.GetType() ==
           miopenHalf) // impossible path from the fusion metadata graph
            build_params.Define("MIOPEN_USE_FPMIX", 1);
        build_params.Define("MIO_BN_CHW", c * h * w);
        build_params.Define("MIO_BN_HW", h * w);
        build_params.Define("MIO_BN_N", n);
        build_params.Define("MIO_BN_GRP0", vld.at(0));
        build_params.Define("MIO_BN_GRP1", 1);
        build_params.Define("MIO_BN_GRP2", 1);

        const std::string READ_TYPE =
            (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string(read_unit);
        build_params.Define("MIOPEN_READ_UNIT", read_unit);
        build_params.Define("MIOPEN_READ_TYPE", READ_TYPE);
    }
    kernel_info.comp_options += " " + build_params.GenerateFor(kbp::OpenCL{});

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            const auto& kernel = handle.Run(kernels[0]);
            const auto& invoke_ctx =
                primitive_parameters.CastTo<miopen::fusion::FusionInvokeParams>();
            const auto& bot_buf = invoke_ctx.in;
            const auto& wei_buf = dynamic_cast<miopen::fusion::ConvolutionOpInvokeParam&>(
                                      *invoke_ctx.op_args.params[0])
                                      .weights;
            const auto& top_buf = invoke_ctx.out;
            std::vector<OpKernelArg> opArgs; // The kernel signature has a max of  12 arguments
            if(activ_idx != -1)
            {
                const auto& activ_args = dynamic_cast<miopen::fusion::ActivationOpInvokeParam&>(
                    *invoke_ctx.op_args.params[activ_idx]);
                opArgs.emplace_back(static_cast<float>(activ_args.activAlpha));
                opArgs.emplace_back(static_cast<float>(activ_args.activBeta));
                opArgs.emplace_back(static_cast<float>(activ_args.activGamma));
            }
            if(bn_idx != -1)
            {
                const auto& bn_args =
                    dynamic_cast<miopen::fusion::BatchNormInferenceOpInvokeParam&>(
                        *invoke_ctx.op_args.params[bn_idx]);
                opArgs.emplace_back(static_cast<double>(bn_args.epsilon));
            }
            opArgs.emplace_back(bot_buf);
            opArgs.emplace_back(top_buf);
            opArgs.emplace_back(wei_buf);
            if(bias_idx != -1)
            {
                opArgs.emplace_back(
                    dynamic_cast<miopen::fusion::BiasOpInvokeParam&>(*invoke_ctx.op_args.params[1])
                        .bdata);
            }
            if(bn_idx != -1)
            {
                const auto& bn_args =
                    dynamic_cast<miopen::fusion::BatchNormInferenceOpInvokeParam&>(
                        *invoke_ctx.op_args.params[bn_idx]);
                opArgs.emplace_back(bn_args.bnBias);
                opArgs.emplace_back(bn_args.bnScale);
                opArgs.emplace_back(bn_args.estimatedMean);
                opArgs.emplace_back(bn_args.estimatedVariance);
            }
            kernel(opArgs);
        };
    };

    return result;
}

float ConvOclDirectFwdFused::GetWti(const FusionContext&, const FusionDescription& problem) const
{
    /// \ref Negative WTI values

    const auto& desc = *problem.fusion_plan_desc;
    const int bn_idx = GetOpIdx(desc.op_map, miopenFusionOpBatchNormInference);
    if(bn_idx != -1)
        return wti_approximate_worst;
    else
        return wti_approximate_worst * .45f;
}

PerformanceConfigConvOclDirectFwdFused
ConvOclDirectFwdFused::GetDefaultPerformanceConfig(const FusionContext& context,
                                                   const FusionDescription& problem) const
{
    const auto base = conv::ConvOclDirectFwd{};
    MIOPEN_LOG_I("Using Unfused class to initialize performance config");
    const auto conv_problem = problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    const auto conv_ctx     = context.GetConvContext(conv_problem);
    return base.GetDefaultPerformanceConfig(conv_ctx, conv_problem);
}

bool ConvOclDirectFwdFused::IsValidPerformanceConfig(
    const FusionContext& context,
    const FusionDescription& problem,
    const PerformanceConfigConvOclDirectFwdFused& c) const
{
    const auto base         = conv::ConvOclDirectFwd{};
    const auto conv_problem = problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    const auto conv_ctx     = context.GetConvContext(conv_problem);
    return base.IsValidPerformanceConfig(conv_ctx, conv_problem, c);
}

} // namespace fusion
} // namespace solver
} // namespace miopen
