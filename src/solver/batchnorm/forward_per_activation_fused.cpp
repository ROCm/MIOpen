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

#include <miopen/batchnorm/solvers.hpp>

#include <miopen/batchnorm/invoke_params.hpp>
#include <miopen/batch_norm.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/fusion/solvers.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_BN_FWDTRG_ACTIV_FUSED)

namespace miopen {

namespace solver {
namespace fusion {

bool BnFwdTrgActivationFused::IsApplicable(const FusionContext& /*context*/,
                                           const FusionDescription& problem) const
{
    const auto& desc = *problem.fusion_plan_desc;
    if(desc.op_map.empty())
        MIOPEN_THROW("");
    if(env::disabled(MIOPEN_DEBUG_BN_FWDTRG_ACTIV_FUSED))
        return false;
    if(desc.op_map.size() != 2)
        return false;
    if(desc.op_map.at(0)->kind() != miopenFusionOpBatchNormFwdTrain)
        return false;
    if(desc.op_map.at(1)->kind() != miopenFusionOpActivForward)
        return false;
    return true;
}

ConvSolution BnFwdTrgActivationFused::GetSolution(const FusionContext& context,
                                                  const FusionDescription& problem) const
{
    const auto bn_problem = problem.GetBnProblem(0, miopen::batchnorm::Direction::ForwardTraining);

    int n, c, h, w;
    auto result = ConvSolution{miopenStatusSuccess};
    miopenDataType_t input_type;
    bool savePopStats         = true;
    const bool saveBatchStats = true;
    {
        const auto& handle = context.GetStream();
        auto kernel        = KernelInfo{};

        kernel.kernel_file = "MIOpenBatchNormActivFwdTrain";
        kernel.kernel_name = "MIOpenBatchNormActivFwdTrain";
        const auto mode    = bn_problem.GetMode();
        if(mode == miopenBNSpatial)
        { // SPATIAL kernels
            kernel.kernel_file += "Spatial.cl";
            kernel.kernel_name += "Spatial";
        }
        else
        { // PER ACTIVATION
            kernel.kernel_file += "PerAct.cl";
            kernel.kernel_name += "PerActivation";
        }
        size_t xlocalsize, ylocalsize, zlocalsize;
        const auto& input_desc = bn_problem.GetXDesc();
        input_type             = input_desc.GetType();
        std::tie(n, c, h, w)   = tien<4>(input_desc.GetLengths());
        size_t in_cstride      = static_cast<size_t>(h) * w;

        xlocalsize = 1024;
        ylocalsize = 1;
        zlocalsize = 1;

        if(mode == miopenBNSpatial)
        {
            if((in_cstride <= 1024) && (in_cstride > 512))
            {
                xlocalsize = 64 * ((in_cstride + 63) / 64);
            }
        }
        else
        {
            xlocalsize = 1;
            ylocalsize = 256;
        }
        kernel.l_wk = {xlocalsize, ylocalsize, zlocalsize};

        size_t xgridsize = c * xlocalsize;
        size_t zgridsize = 1;
        size_t ygridsize = 1;

        if(mode != miopenBNSpatial)
        {
            auto segment = int(std::ceil(double(in_cstride) / double(ylocalsize)));
            xgridsize    = c;
            ygridsize    = segment * ylocalsize;
        }

        kernel.g_wk       = {xgridsize, ygridsize, zgridsize};
        size_t in_nstride = c * in_cstride;
        size_t in_nchw    = n * in_nstride;

        unsigned int ldsgcn   = xlocalsize / 64;
        unsigned int ldsnogcn = xlocalsize;

        int variant = 0;

        if(mode == miopenBNSpatial)
        {
            if(in_cstride > 1024)
            {
                variant = 1;
            }
            else if(in_cstride > 512)
            {
                variant = 3;
            }
        }

        size_t read_unit = 1;
        size_t read_len  = (mode == miopenBNSpatial) ? in_cstride : in_nstride;

        if(mode == miopenBNSpatial)
        {
            read_unit = (read_len % 4 == 0) ? 4 : (read_len % 2 == 0) ? 2 : 1;
        }
        savePopStats          = bn_problem.GetResultSave(); // TODO: double check this
        std::string READ_TYPE = (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string(read_unit);
        const auto& activ_op =
            dynamic_cast<ActivFwdFusionOpDescriptor&>(*problem.fusion_plan_desc->op_map[1]);
        const auto build_params = KernelBuildParameters{
            {"MIO_BN_N", static_cast<int>(n)},
            {"MIO_BN_C", static_cast<int>(c)},
            {"MIO_BN_HW", static_cast<int>(in_cstride)},
            {"MIO_BN_NHW", static_cast<int>(n * h * w)},
            {"MIO_BN_CHW", static_cast<int>(in_nstride)},
            {"MIO_BN_NCHW", static_cast<int>(in_nchw)},
            {"MIO_BN_GRP0", static_cast<int>(xlocalsize)},
            {"MIO_BN_GRP1", static_cast<int>(ylocalsize)},
            {"MIO_BN_GRP2", static_cast<int>(zlocalsize)},
            {"MIO_BN_LDS_SIZE", static_cast<int>(ldsnogcn)},
            {"MIO_BN_LDSGCN_SIZE", static_cast<int>(ldsgcn)},
            {"MIOPEN_READ_UNIT", static_cast<int>(read_unit)},
            {"MIOPEN_READ_TYPE", READ_TYPE},
            {"MIO_SAVE_MEAN_VARIANCE", static_cast<int>(saveBatchStats)},
            {"MIO_RUNNING_RESULT", static_cast<int>(savePopStats)},
            {"MIO_BN_VARIANT", static_cast<int>(variant)},
            {"MIO_BN_GFX103X", static_cast<int>(StartsWith(handle.GetDeviceName(), "gfx103"))},
            {"MIO_BN_GFX110X", static_cast<int>(StartsWith(handle.GetDeviceName(), "gfx110"))},
            {"MIO_BN_GFX120X", static_cast<int>(StartsWith(handle.GetDeviceName(), "gfx120"))},
            {"MIOPEN_YES_ACTIV", static_cast<int>(1)},
            {"MIOPEN_NRN_OP_ID", static_cast<int>(activ_op.activMode)},
            {"MIOPEN_USE_FP16", static_cast<int>(input_desc.GetType() == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(input_desc.GetType() == miopenFloat)}};
        kernel.comp_options = build_params.GenerateFor(kbp::OpenCL{});
        if(bn_problem.GetMode() == miopenBNSpatial)
            kernel.comp_options += " -DSPATIAL_BN";
        else
            kernel.comp_options += " -DPERACT_BN";
        if(input_desc.GetType() == miopenHalf)
            kernel.comp_options += " -DMIOPEN_USE_FPMIX=1";

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel   = handle_.Run(kernels.front());
            const auto& invoke_ctx  = raw_params.CastTo<miopen::fusion::FusionInvokeParams>();
            const auto& bot_ocl_buf = invoke_ctx.in;
            const auto& top_ocl_buf = invoke_ctx.out;
            const auto& bn_invoke =
                dynamic_cast<miopen::fusion::BatchNormFwdTrainingOpInvokeParam&>(
                    *invoke_ctx.op_args.params[0]);
            const auto& activ_invoker = dynamic_cast<miopen::fusion::ActivationOpInvokeParam&>(
                *invoke_ctx.op_args.params[1]);
            const auto activ_alpha = activ_invoker.activAlpha;
            const auto activ_beta  = activ_invoker.activBeta;
            const auto activ_gamma = activ_invoker.activGamma;
            const auto mode        = bn_problem.GetMode();
            std::vector<OpKernelArg> kern_args;
            if(mode == miopenBNSpatial)
                kern_args.push_back({static_cast<float>(1.0f / (n * h * w))});

            if(input_type == miopenFloat)
            {
                kern_args.push_back({static_cast<float>(activ_alpha)});
                kern_args.push_back({static_cast<float>(activ_beta)});
                kern_args.push_back({static_cast<float>(activ_gamma)});
            }
            else if(input_type == miopenHalf)
            {
                kern_args.push_back({static_cast<half_float::half>(activ_alpha)});
                kern_args.push_back({static_cast<half_float::half>(activ_beta)});
                kern_args.push_back({static_cast<half_float::half>(activ_gamma)});
            }
            else
                MIOPEN_THROW("Unsupported Precision");
            kern_args.push_back({bn_invoke.epsilon});
            if(savePopStats)
                kern_args.push_back({bn_invoke.expAvgFactor});
            kern_args.push_back({bot_ocl_buf});
            kern_args.push_back({top_ocl_buf});
            kern_args.push_back({bn_invoke.bnBias});
            kern_args.push_back({bn_invoke.bnScale});
            if(savePopStats)
            {
                kern_args.push_back({bn_invoke.runningMean});
                kern_args.push_back({bn_invoke.runningVariance});
            }
            if(saveBatchStats)
            {
                kern_args.push_back({bn_invoke.savedInvVariance});
                kern_args.push_back({bn_invoke.savedMean});
            }
            kernel(kern_args);
        };
    };

    return result;
}

} // namespace fusion

} // namespace solver

} // namespace miopen
