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
#include <array>
#include <cassert>
#include <miopen/batch_norm.hpp>
#include <miopen/fusion.hpp>
#include <miopen/fusion_plan.hpp>
#include <miopen/logger.hpp>
#include <miopen/handle.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/fusion/solvers.hpp>
#include <miopen/fusion/fusion_invoke_params.hpp>
#include <miopen/fusion/utils.hpp>
#include <miopen/find_db.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/conv/solver_finders.hpp>
#include <miopen/driver_arguments.hpp>
#include <miopen/config.hpp>

#include <ostream>
#include <ios>
#include <algorithm>
#include <string>
#include <half/half.hpp>

#define MIOPEN_CHECK(x)          \
    if(x != miopenStatusSuccess) \
        return x;

namespace miopen {

miopenStatus_t ConvBiasActivFusion(Handle& handle,
                                   const void* alpha1,
                                   const TensorDescriptor& xDesc,
                                   ConstData_t x,
                                   const TensorDescriptor& wDesc,
                                   ConstData_t w,
                                   const ConvolutionDescriptor& conv_desc,
                                   miopenConvFwdAlgorithm_t algo,
                                   void* workspace,
                                   size_t workspaceSizeInBytes,
                                   const void* alpha2,
                                   const TensorDescriptor& zDesc,
                                   ConstData_t z,
                                   const TensorDescriptor& biasDesc,
                                   ConstData_t bias,
                                   const ActivationDescriptor& activationDesc,
                                   const TensorDescriptor& yDesc,
                                   Data_t y)
{
    assert(workspace == nullptr);
    assert(workspaceSizeInBytes == 0);
    std::ignore = workspace;
    std::ignore = workspaceSizeInBytes;
    /// \todo: add workspace support in fusion

    /*
    if(alpha1 != nullptr)
    {
        const auto falpha1 = *(static_cast<const float*>(alpha1));
        if(falpha1 != 1.0f)
            MIOPEN_THROW(miopenStatusNotImplemented, "alpha1 can only be 1.0");
    }
    if(alpha2 != nullptr)
    {
        const auto falpha2 = *(static_cast<const float*>(alpha2));
        if(falpha2 != 1.0f)
            MIOPEN_THROW(miopenStatusNotImplemented, "alpha2 can only be 1.0");
    }
    */

    // TODO: The type of these pointers depends on the ConvolutionDescriptor's data
    // type
    float falpha1 = alpha1 != nullptr ? *(static_cast<const float*>(alpha1)) : 1.0f;
    float falpha2 = alpha2 != nullptr ? *(static_cast<const float*>(alpha2)) : 1.0f;

    // if(z != nullptr || zDesc.GetNumDims() != 0)
    // MIOPEN_THROW(miopenStatusNotImplemented, "The addition of z vector is not yet supported");
    FusionPlanDescriptor fusePlanDesc{miopenVerticalFusion, xDesc};
    OperatorArgs fusionArgs;
    auto convOp  = std::make_shared<ConvForwardOpDescriptor>(conv_desc, wDesc);
    auto zOp     = std::make_shared<TensorScaleAddOpDescriptor>(zDesc);
    auto biasOp  = std::make_shared<BiasFusionOpDescriptor>(biasDesc);
    auto activOp = std::make_shared<ActivFwdFusionOpDescriptor>(activationDesc.GetMode());

    if(activationDesc.GetMode() != miopenActivationRELU)
    {
        MIOPEN_THROW(miopenStatusNotImplemented,
                     "only Activation Mode == miopenActivationRELU is supported");
    }

    MIOPEN_CHECK(fusePlanDesc.AddOp(convOp));
    MIOPEN_CHECK(fusePlanDesc.SetConvAlgo(algo));
    MIOPEN_CHECK(fusePlanDesc.AddOp(zOp));
    MIOPEN_CHECK(fusePlanDesc.AddOp(biasOp));
    MIOPEN_CHECK(fusePlanDesc.AddOp(activOp));

    MIOPEN_CHECK(fusePlanDesc.Compile(handle));
    float alpha       = 1.0f;
    float beta        = 0.0f;
    float activ_alpha = activationDesc.GetAlpha();
    float activ_beta  = activationDesc.GetBeta();
    float activ_gamma = activationDesc.GetGamma();

    // Set the Args
    MIOPEN_CHECK(convOp->SetArgs(fusionArgs, &falpha1, &beta, w));
    MIOPEN_CHECK(zOp->SetArgs(fusionArgs, falpha2, z));
    MIOPEN_CHECK(biasOp->SetArgs(fusionArgs, &alpha, &beta, bias));
    MIOPEN_CHECK(activOp->SetArgs(fusionArgs, &alpha, &beta, activ_alpha, activ_beta, activ_gamma));
    MIOPEN_CHECK(fusePlanDesc.Execute(handle, xDesc, x, yDesc, y, fusionArgs));
    return miopenStatusSuccess;
}

static auto
AllocateBuffersAndMakeFusionInvokeParams(Handle& handle,
                                         const FusionDescription& problem,
                                         std::vector<Allocator::ManageDataPtr>& invoke_bufs,
                                         miopen::OperatorArgs& params,
                                         const FusionPlanDescriptor& plan)
{
    const auto allocate_buffer = [&](std::size_t size) {
        auto ptr = handle.Create(size);
        auto ret = ptr.get();
        invoke_bufs.push_back(std::move(ptr));
        return ret;
    };

    const auto conv_id      = solver::fusion::GetOpIdx(plan.op_map, miopenFusionOpConvForward);
    const auto bias_id      = solver::fusion::GetOpIdx(plan.op_map, miopenFusionOpBiasForward);
    const auto activ_fwd_id = solver::fusion::GetOpIdx(plan.op_map, miopenFusionOpActivForward);
    const auto activ_bwd_id = solver::fusion::GetOpIdx(plan.op_map, miopenFusionOpActivBackward);
    const auto bn_inf_id = solver::fusion::GetOpIdx(plan.op_map, miopenFusionOpBatchNormInference);
    const auto bn_fwd_id = solver::fusion::GetOpIdx(plan.op_map, miopenFusionOpBatchNormFwdTrain);
    const auto bn_bwd_id = solver::fusion::GetOpIdx(plan.op_map, miopenFusionOpBatchNormBwdTrain);
    const auto tensor_add_op_id =
        solver::fusion::GetOpIdx(plan.op_map, miopenFusionOpTensorScaleAdd);

    const auto any_activ = activ_fwd_id != -1 || activ_bwd_id != -1;
    const auto any_bn    = bn_inf_id != -1 || bn_fwd_id != -1 || bn_bwd_id != -1;

    Data_t bias_ptr = nullptr;
    TensorDescriptor in_desc, out_desc;
    bool gfx90aaltimpl = false;

    if(conv_id != -1)
    {
        const auto conv_problem =
            problem.GetConvProblem(conv_id, conv::Direction::Forward, bias_id != -1 ? 1 : 0);
        gfx90aaltimpl = conv_problem.GetConv().attribute.gfx90aFp16alt.GetFwd();

        in_desc  = conv_problem.GetIn();
        out_desc = conv_problem.GetOut();

        if(bias_id != -1)
        {
            bias_ptr = allocate_buffer(conv_problem.GetBiasSize());

            MIOPEN_LOG_I("bias addr: " << bias_ptr << ", size: " << conv_problem.GetBiasSize());
            params.SetArg(bias_id, std::make_unique<miopen::fusion::BiasOpInvokeParam>(bias_ptr));
        }

        auto wei_ptr = allocate_buffer(conv_problem.GetWeightsSize());
        params.SetArg(conv_id, std::make_unique<miopen::fusion::ConvolutionOpInvokeParam>(wei_ptr));

        MIOPEN_LOG_I("weight addr: " << wei_ptr << ", size: " << conv_problem.GetWeightsSize());
    }

    if(any_activ)
    {
        const float alpha = 0.5f;
        const float beta  = 0.5f;
        const float gamma = 0.5f;

        if(activ_fwd_id != -1)
        {
            params.SetArg(
                activ_fwd_id,
                std::make_unique<miopen::fusion::ActivationOpInvokeParam>(alpha, beta, gamma));
        }
        else if(activ_bwd_id != -1)
        {
            const auto& activ_op =
                dynamic_cast<ActivBwdFusionOpDescriptor&>(*plan.op_map[activ_bwd_id]);

            const auto space = activ_op.input_desc.GetNumBytes();
            auto x           = allocate_buffer(space);
            auto y           = allocate_buffer(space);

            params.SetArg(activ_bwd_id,
                          std::make_unique<miopen::fusion::ActivationBwdOpInvokeParam>(
                              y, x, alpha, beta, gamma));
        }
    }

    if(tensor_add_op_id != -1)
    {
        const auto& tensor_add_op =
            dynamic_cast<const TensorScaleAddOpDescriptor&>(*plan.op_map[tensor_add_op_id]);
        assert(&tensor_add_op);

        float alpha      = 1.0f;
        const auto space = tensor_add_op.tensor_desc.GetNumBytes();
        auto ptr         = allocate_buffer(space);

        params.SetArg(tensor_add_op_id,
                      std::make_unique<miopen::fusion::TensorScaleAddOpInvokeParam>(alpha, ptr));
    }

    if(any_bn)
    {
        const auto epsilon = 0.00001;
        const auto expAvg  = 0.99;
        const auto alpha   = 1.0;
        const auto beta    = 0.0;

        if(bn_inf_id != -1)
        {
            const auto& bn_op =
                dynamic_cast<BatchNormInferenceFusionOpDescriptor&>(*plan.op_map[bn_inf_id]);

            out_desc = in_desc = bn_op.input_desc;

            const auto size   = bn_op.base_desc.GetNumBytes();
            auto scale_ptr    = allocate_buffer(size);
            auto mean_ptr     = allocate_buffer(size);
            auto variance_ptr = allocate_buffer(size);
            if(bias_ptr == nullptr)
                bias_ptr = allocate_buffer(size);

            if(bias_ptr == nullptr)
                allocate_buffer(bn_op.base_desc.GetNumBytes());

            bn_op.SetArgs(
                params, &alpha, &beta, scale_ptr, bias_ptr, mean_ptr, variance_ptr, epsilon);
        }
        else if(bn_fwd_id != -1)
        {
            const auto& bn_op =
                dynamic_cast<BatchNormFwdTrainFusionOpDescriptor&>(*plan.op_map[bn_fwd_id]);

            out_desc = in_desc = bn_op.input_desc;

            // We don't have descriptor here
            miopen::TensorDescriptor derivedBnDesc{};
            miopen::DeriveBNTensorDescriptor(derivedBnDesc, in_desc, bn_op.mode);

            const auto size              = derivedBnDesc.GetNumBytes();
            Data_t scale_ptr             = allocate_buffer(size);
            Data_t mean_ptr              = allocate_buffer(size);
            Data_t variance_ptr          = allocate_buffer(size);
            Data_t save_mean_ptr         = allocate_buffer(size);
            Data_t save_inv_variance_ptr = allocate_buffer(size);
            if(bias_ptr == nullptr)
                bias_ptr = allocate_buffer(size);

            bn_op.SetArgs(params,
                          &alpha,
                          &beta,
                          mean_ptr,
                          variance_ptr,
                          save_mean_ptr,
                          save_inv_variance_ptr,
                          scale_ptr,
                          bias_ptr,
                          expAvg,
                          epsilon);
        }
        else if(bn_bwd_id != -1)
        {
            const auto& bn_op =
                dynamic_cast<BatchNormBwdTrainFusionOpDescriptor&>(*plan.op_map[bn_bwd_id]);

            out_desc = in_desc = bn_op.input_desc;

            Data_t x_ptr = allocate_buffer(in_desc.GetNumBytes());

            // We don't have descriptor here
            miopen::TensorDescriptor derivedBnDesc{};
            miopen::DeriveBNTensorDescriptor(derivedBnDesc, in_desc, bn_op.mode);

            const auto size               = derivedBnDesc.GetNumBytes();
            Data_t scale_ptr              = allocate_buffer(size);
            Data_t res_bn_scale_diff_ptr  = allocate_buffer(size);
            Data_t res_bn_bias_diff_ptr   = allocate_buffer(size);
            Data_t saved_mean_ptr         = allocate_buffer(size);
            Data_t saved_inv_variance_ptr = allocate_buffer(size);
            if(bias_ptr == nullptr)
                bias_ptr = allocate_buffer(size);

            bn_op.SetArgs(params,
                          &alpha,
                          &beta,
                          x_ptr,
                          scale_ptr,
                          bias_ptr,
                          res_bn_scale_diff_ptr,
                          res_bn_bias_diff_ptr,
                          saved_mean_ptr,
                          saved_inv_variance_ptr);
        }
    }

    const auto in_ptr = allocate_buffer(in_desc.GetNumBytes());
    MIOPEN_LOG_I("in addr: " << in_ptr << ", size: " << in_desc.GetNumBytes());
    const auto out_ptr = allocate_buffer(out_desc.GetNumBytes());
    MIOPEN_LOG_I("out addr: " << out_ptr << ", size: " << in_desc.GetNumBytes());

    return miopen::fusion::FusionInvokeParams(
        params, in_desc, in_ptr, out_desc, out_ptr, gfx90aaltimpl);
}

namespace debug {

std::string LogCmdConvolutionFusion(const miopenFusionPlanDescriptor_t fusePlanDesc,
                                    int fusion_mode)
{
    const auto& conv_op =
        dynamic_cast<ConvForwardOpDescriptor*>(deref(fusePlanDesc).op_map[0].get());

    const miopenTensorDescriptor_t& xDesc         = &deref(fusePlanDesc).input_desc;
    const miopenTensorDescriptor_t& wDesc         = &conv_op->filter_desc;
    const miopenConvolutionDescriptor_t& convDesc = &conv_op->base_desc;
    const miopenTensorDescriptor_t& yDesc         = &deref(fusePlanDesc).output_desc;
    std::string str;

    if(deref(fusePlanDesc).data_type == miopenBFloat16)
    {
        str = "CBAInferfp16";
    }
    else
    {
        str = "CBAInfer";
    }

    str += " -F " + std::to_string(fusion_mode);
    str += ConvArgsForMIOpenDriver(miopen::deref(xDesc),
                                   miopen::deref(wDesc),
                                   miopen::deref(convDesc),
                                   miopen::deref(yDesc),
                                   miopenProblemDirection_t::miopenProblemDirectionForward,
                                   std::nullopt,
                                   false);

    return str;
}

std::string LogCmdBnormFusion(const miopenFusionPlanDescriptor_t fusePlanDesc, int fusion_mode)
{
    assert(!deref(fusePlanDesc).op_map.empty());

    std::string str;
    if(deref(fusePlanDesc).data_type == miopenBFloat16)
    {
        str = "CBAInferfp16";
    }
    else
    {
        str = "CBAInfer";
    }
    str += " -F " + std::to_string(fusion_mode);

    const auto& bn_op =
        dynamic_cast<BatchNormInferenceFusionOpDescriptor*>(deref(fusePlanDesc).op_map[0].get());

    if(bn_op != nullptr)
    {
        str += BnormArgsForMIOpenDriver(&bn_op->input_desc,
                                        bn_op->mode,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        miopen::debug::BatchNormDirection_t::ForwardInference,
                                        false);
    }
    else
    {
        MIOPEN_LOG_E("Dereferencing nullptr when logging batch norm");
    }
    return str;
}

MIOPEN_INTERNALS_EXPORT
void LogCmdFusion(const miopenFusionPlanDescriptor_t fusePlanDesc)
{
    if(miopen::IsLoggingCmd())
    {
        int fusion_mode = GetFusionMode(fusePlanDesc);
        switch(fusion_mode)
        {
        case 0:
        case 1:
        case 3:
        case 4:
        case 5:
        case 6: MIOPEN_LOG_DRIVER_CMD(LogCmdConvolutionFusion(fusePlanDesc, fusion_mode)); break;
        case 2: MIOPEN_LOG_DRIVER_CMD(LogCmdBnormFusion(fusePlanDesc, fusion_mode)); break;
        default: MIOPEN_LOG_E("Unknown fusion plan : " << fusion_mode);
        }
    }
}
} // namespace debug

FusionPlanDescriptor::FusionPlanDescriptor(const miopenFusionDirection_t dir,
                                           const TensorDescriptor& inDesc)
    : fusion_dir(dir),
      input_desc(inDesc),
      is_valid(false),
      kernel_source_type(OpenclText),
      fp_contains_bn(false),
      data_type(inDesc.GetType())
{
}

miopenStatus_t FusionPlanDescriptor::AddOp(std::shared_ptr<FusionOpDescriptor> desc)
{
    desc->SetIdx(op_count);
    if(op_map.empty())
        desc->SetInputDesc(input_desc);
    else
        desc->SetInputDesc(output_desc);
    desc->GetOutputDesc(output_desc);
    op_map.emplace_back(desc);
    op_count++;
    return miopenStatusSuccess;
}

miopenStatus_t FusionPlanDescriptor::GetOp(int op_idx, std::shared_ptr<FusionOpDescriptor>& desc)
{
    auto err = miopenStatusSuccess;

    if(op_idx >= op_map.size())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Operator index out of bounds");
    }

    desc = op_map.at(op_idx);
    return err;
}

TensorDescriptor FusionPlanDescriptor::FusionPlanDescriptor::DeriveOutputDescriptor()
{
    TensorDescriptor i_desc = input_desc;
    TensorDescriptor o_desc;
    if(fusion_dir == miopenVerticalFusion)
    {
        // All the ops should have the same output descriptor otherwise
        // fusion would not be feasible, thus we need to call GetOutputDesc on all
        // the ops and make sure it returns the same value
        for(auto&& op : op_map)
        {
            op->SetInputDesc(i_desc);
            op->GetOutputDesc(o_desc);
            i_desc = o_desc;
        }
    }
    else
    {
        MIOPEN_THROW(miopenStatusNotImplemented, "Unsupported fusion direction");
    }
    return o_desc;
}

miopenStatus_t FusionPlanDescriptor::GetWorkspaceSizeImmed(Handle& handle,
                                                           size_t& workSpaceSize,
                                                           miopenConvFwdAlgorithm_t /*algo*/)
{
    workSpaceSize = 0;
    for(auto&& op : op_map)
    {
        if(op->kind() == miopenFusionOpConvForward)
        {
            auto& conv_op = dynamic_cast<ConvForwardOpDescriptor&>(*op);
            TensorDescriptor opd;
            conv_op.GetOutputDesc(opd);
            const auto ctx     = ExecutionContext{&handle};
            const auto problem = conv::ProblemDescription{conv_op.input_desc,
                                                          conv_op.filter_desc,
                                                          opd,
                                                          conv_op.base_desc,
                                                          conv::Direction::Forward};
            const auto tmp_sz  = conv_op.base_desc.GetWorkSpaceSize(ctx, problem);
            if(tmp_sz > workSpaceSize)
                workSpaceSize = tmp_sz;
        }
    }
    return miopenStatusSuccess;
}

miopenStatus_t FusionPlanDescriptor::GetConvAlgos(int reqAlgoCount,
                                                  int& retAlgoCount,
                                                  miopenConvFwdAlgorithm_t* ptrAlgos)
{
    const std::vector<miopenConvFwdAlgorithm_t> algos = {miopenConvolutionFwdAlgoDirect,
                                                         miopenConvolutionFwdAlgoWinograd};
    retAlgoCount = std::min(reqAlgoCount, static_cast<int>(algos.size()));
    for(auto idx = 0; idx < retAlgoCount; idx++)
    {
        ptrAlgos[idx] = algos[idx];
    }
    return miopenStatusSuccess;
}

miopenStatus_t FusionPlanDescriptor::SetConvAlgo(miopenConvFwdAlgorithm_t algo)
{
    conv_fwd_algo = algo;
    return miopenStatusSuccess;
}

std::ostream& operator<<(std::ostream& stream, const FusionPlanDescriptor& /*fpd*/)
{
    // stream << "kernel_name: " << fpd.kernel_name;
    return stream;
}

// Fusion operator descriptors
// Conv Forward
miopenStatus_t ConvForwardOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc) const
{
    return miopen::try_(
        [&]() { output_desc = base_desc.GetForwardOutputTensor(input_desc, filter_desc); });
}

/*
miopenStatus_t
ConvForwardOpDescriptor::SetArgs(OperatorArgs& args, float alpha, float beta, ConstData_t w)
{
    auto op_args = std::make_unique<fusion::ConvolutionOpInvokeParam>(alpha, beta, w);
    args.SetArg(GetIdx(), std::move(op_args));
    return miopenStatusSuccess;
}
*/

miopenStatus_t ConvForwardOpDescriptor::SetArgs(OperatorArgs& args,
                                                const void* alpha,
                                                const void* beta,
                                                ConstData_t w)
{
    float falpha = alpha != nullptr ? *reinterpret_cast<const float*>(alpha) : 1.0f;
    float fbeta  = beta != nullptr ? *reinterpret_cast<const float*>(beta) : 0.0f;
    auto op_args = std::make_unique<fusion::ConvolutionOpInvokeParam>(falpha, fbeta, w);
    args.SetArg(GetIdx(), std::move(op_args));
    return miopenStatusSuccess;
}

// Activ Forward ------------------------------------

miopenStatus_t ActivFwdFusionOpDescriptor::SetArgs(OperatorArgs& args,
                                                   const void* /*alpha*/,
                                                   const void* /*beta*/,
                                                   double activAlpha,
                                                   double activBeta,
                                                   double activGamma)
{
    auto op_args =
        std::make_unique<fusion::ActivationOpInvokeParam>(activAlpha, activBeta, activGamma);
    args.SetArg(GetIdx(), std::move(op_args));
    return miopenStatusSuccess;
}

miopenStatus_t ActivFwdFusionOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc) const
{
    // activation does not change the size
    output_desc = input_desc;
    return miopenStatusSuccess;
}
// Activ Backwards-----------------------------------------
miopenStatus_t ActivBwdFusionOpDescriptor::SetArgs(OperatorArgs& args,
                                                   const void* /*alpha*/,
                                                   const void* /*beta*/,
                                                   ConstData_t y,
                                                   ConstData_t x,
                                                   double activAlpha,
                                                   double activBeta,
                                                   double activGamma)
{
    auto op_args = std::make_unique<fusion::ActivationBwdOpInvokeParam>(
        y, x, activAlpha, activBeta, activGamma);
    args.SetArg(GetIdx(), std::move(op_args));
    return miopenStatusSuccess;
}

miopenStatus_t ActivBwdFusionOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc) const
{
    // activation does not change the size
    output_desc = input_desc;
    return miopenStatusSuccess;
}
//==============================

miopenStatus_t BatchNormInferenceFusionOpDescriptor::SetArgs(OperatorArgs& args,
                                                             const void*,
                                                             const void*,
                                                             ConstData_t bnScale,
                                                             ConstData_t bnBias,
                                                             ConstData_t estimatedMean,
                                                             ConstData_t estimatedVariance,
                                                             double epsilon) const
{
    auto op_args = std::make_unique<fusion::BatchNormInferenceOpInvokeParam>(
        bnScale, bnBias, estimatedMean, estimatedVariance, epsilon);
    args.SetArg(GetIdx(), std::move(op_args));
    return miopenStatusSuccess;
}

miopenStatus_t
BatchNormInferenceFusionOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc) const
{
    output_desc = input_desc;
    return miopenStatusSuccess;
}

// Batch Normalization Forward Training --------------
miopenStatus_t BatchNormFwdTrainFusionOpDescriptor::SetArgs(OperatorArgs& args,
                                                            const void* /*alpha*/,
                                                            const void* /*beta*/,
                                                            Data_t runningMean,
                                                            Data_t runningVariance,
                                                            Data_t savedMean,
                                                            Data_t savedInvVariance,
                                                            ConstData_t bnScale,
                                                            ConstData_t bnBias,
                                                            double expAvgFactor,
                                                            double epsilon) const
{
    if(runningMeanVar && (runningMean == nullptr || runningVariance == nullptr))
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "Save batch statistics was turned on at op creation time "
                     "but runningMean or runningVariance is set to nullptr");
    }
    auto op_args = std::make_unique<fusion::BatchNormFwdTrainingOpInvokeParam>(runningMean,
                                                                               runningVariance,
                                                                               savedMean,
                                                                               savedInvVariance,
                                                                               bnScale,
                                                                               bnBias,
                                                                               expAvgFactor,
                                                                               epsilon);
    args.SetArg(GetIdx(), std::move(op_args));
    return miopenStatusSuccess;
}

miopenStatus_t
BatchNormFwdTrainFusionOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc) const
{
    output_desc = input_desc;
    return miopenStatusSuccess;
}

// end BN forward training -----------------------------

// Batch Normalization Backward Training --------------
miopenStatus_t BatchNormBwdTrainFusionOpDescriptor::SetArgs(OperatorArgs& args,
                                                            const void* /*alpha*/,
                                                            const void* /*beta*/,
                                                            ConstData_t x,
                                                            ConstData_t bnScale,
                                                            ConstData_t bnBias,
                                                            Data_t resBnScaleDiff,
                                                            Data_t resBnBiasDiff,
                                                            ConstData_t savedMean,
                                                            ConstData_t savedInvVariance) const
{
    auto op_args = std::make_unique<fusion::BatchNormBwdTrainingOpInvokeParam>(
        x, bnScale, bnBias, resBnScaleDiff, resBnBiasDiff, savedMean, savedInvVariance);
    args.SetArg(GetIdx(), std::move(op_args));
    return miopenStatusSuccess;
}
miopenStatus_t
BatchNormBwdTrainFusionOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc) const
{
    output_desc = input_desc;
    return miopenStatusSuccess;
}

// end BN backwards training ---------------------------

// Bias forward
miopenStatus_t BiasFusionOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc) const
{
    output_desc = input_desc;
    return miopenStatusSuccess;
}

miopenStatus_t BiasFusionOpDescriptor::SetArgs(OperatorArgs& args,
                                               const void* /*alpha*/,
                                               const void* /*beta*/,
                                               ConstData_t bdata)
{
    auto op_args = std::make_unique<fusion::BiasOpInvokeParam>(bdata);
    args.SetArg(GetIdx(), std::move(op_args));
    return miopenStatusSuccess;
}

miopenStatus_t TensorScaleAddOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc) const
{
    output_desc = this->tensor_desc;
    return miopenStatusSuccess;
}

miopenStatus_t
TensorScaleAddOpDescriptor::SetArgs(OperatorArgs& args, float alpha, ConstData_t tensor_ptr)
{
    auto op_args = std::make_unique<fusion::TensorScaleAddOpInvokeParam>(alpha, tensor_ptr);
    args.SetArg(GetIdx(), std::move(op_args));
    return miopenStatusSuccess;
}

std::string FusionPlanDescriptor::GetAlgorithmName(const Handle& /*handle*/)
{
    if(conv_fwd_algo)
    {
        return miopen::ConvolutionAlgoToDirectionalString(
            static_cast<miopenConvAlgorithm_t>(*conv_fwd_algo), miopen::conv::Direction::Forward);
    }
    MIOPEN_THROW(miopenStatusBadParm,
                 "GetAlgorithmName was called, but Algorithm has not been set");
}

static auto GetFusedNonConvSolvers()
{
    return solver::SolverContainer<solver::fusion::BnFwdInferActivationFused,
                                   solver::fusion::BnFwdTrgActivationFused,
                                   solver::fusion::BnBwdTrgActivationFused>{};
}

static auto GetFusedDirectSolvers()
{
    return solver::SolverContainer<solver::fusion::ConvBiasActivAsm1x1U,
                                   solver::fusion::ConvOclDirectFwdFused>{};
}

static auto GetFusedIGemmSolvers()
{
    return solver::SolverContainer<solver::fusion::ConvCKIgemmFwdBiasActivFused,
                                   solver::fusion::ConvCKIgemmFwdBiasResAddActivFused>{};
}

static auto GetFusedWinogradSolvers()
{
    return solver::SolverContainer<solver::fusion::ConvBinWinogradRxSFused,
                                   solver::fusion::ConvBinWinogradRxSf2x3g1Fused,
                                   solver::fusion::ConvWinoFuryRxSFused<2, 3>>{};
}

static auto GetAllFusionSolvers()
{
    return GetFusedNonConvSolvers() + GetFusedDirectSolvers() + GetFusedIGemmSolvers() +
           GetFusedWinogradSolvers();
}

solver::ConvSolution MakeFusedSolution(const FusionContext& ctx,
                                       solver::Id id,
                                       const std::optional<std::string>& perf_cfg_override,
                                       const FusionDescription& problem,
                                       const AnyInvokeParams& invoke_params)
{
    decltype(auto) db = GetDb(ctx);
    solver::ConvSolution solution{miopenStatusInternalError};

    GetAllFusionSolvers().FindById(id, [&](auto solver) {
        solution = miopen::solver::FindSolution(
            solver, ctx, problem, db, invoke_params, perf_cfg_override.value_or(""));
    });

    return solution;
}

struct FusionFindParameters : PrimitiveFindParameters
{
};

template <class SolverContainer>
class FusionSolverFinder : public SolversFinderMixin<FusionDescription, FusionFindParameters>
{
public:
    explicit FusionSolverFinder(SolverContainer solvers_, const std::string& algo_name)
        : solvers(solvers_), algo(algo_name)
    {
    }

protected:
    AlgorithmName GetAlgorithmName(const FusionDescription&) const override { return algo; }

    bool IsEnabled(const ExecutionContext&,
                   const FusionDescription&,
                   const FusionFindParameters&) const override
    {
        return true;
    }

    std::vector<solver::ConvSolution>
    FindImpl(const ExecutionContext& ctx,
             const FusionDescription& problem,
             const AnyInvokeParams& invoke_ctx,
             const FusionFindParameters&,
             const std::optional<FindOptions>& options) const override
    {
        const auto fusion_ctx = FusionContext(ctx);
        return solvers.SearchForAllSolutions(fusion_ctx,
                                             problem,
                                             miopen::GetDb(ctx),
                                             invoke_ctx,
                                             std::numeric_limits<std::size_t>::max(),
                                             options);
    }

private:
    SolverContainer solvers;
    AlgorithmName algo;
};

static const std::vector<std::unique_ptr<ISolversFinder>>& GetFusionSolverFinders()
{
    static const std::vector<std::unique_ptr<ISolversFinder>> finders = [] {
        constexpr const auto add = [](auto& to, auto solvers, const std::string& algo) {
            to.emplace_back(std::make_unique<FusionSolverFinder<decltype(solvers)>>(solvers, algo));
        };

        auto tmp = std::vector<std::unique_ptr<ISolversFinder>>{};
        add(tmp, GetFusedNonConvSolvers(), "fusion");
        add(tmp, GetFusedDirectSolvers(), "miopenConvolutionFwdAlgoDirect");
        add(tmp, GetFusedIGemmSolvers(), "miopenConvolutionFwdAlgoImplicitGEMM");
        add(tmp, GetFusedWinogradSolvers(), "miopenConvolutionFwdAlgoWinograd");
        return tmp;
    }();
    return finders;
}

static std::vector<Solution>
FindFusion(const ExecutionContext& ctx,
           const FusionDescription& fusion_problem,
           const std::function<fusion::FusionInvokeParams()>& invoke_params,
           const std::optional<FindOptions>& options = std::nullopt)
{
    return UserFindDbRecord::TryLoad(
        ctx.GetStream(),
        fusion_problem,
        [&]() {
            // fusion_ctx.use_dynamic_solutions_only = findMode.IsDynamicHybrid(fusion_ctx);

            // We need buffers for find, thus we lazily get them, possibly allocating.
            return FindCore(invoke_params(),
                            ctx,
                            fusion_problem,
                            FusionFindParameters{},
                            GetFusionSolverFinders(),
                            options);
        },
        "fusion");
}

namespace {

// Copy from convolutionocl.cpp
struct SolutionTimeComparator
{
    inline bool operator()(const miopenConvSolution_t& lhs, const miopenConvSolution_t& rhs) const
    {
        // Negative values are very coarse estimations.
        // The more modulus, the "worse" (slower) is solution.
        if(lhs.time < 0 && rhs.time < 0)
            return !(lhs.time < rhs.time);
        // Positive values are always "better" than negative (coarse) estimations.
        if(lhs.time > 0 && rhs.time < 0)
            return true;
        if(lhs.time < 0 && rhs.time > 0)
            return false;
        // Both values are positive. The less is the better.
        return (lhs.time < rhs.time);
    }
};

std::ostream& operator<<(std::ostream& os, const miopenConvSolution_t& s)
{
    return os << "id: " << s.solution_id                              //
              << ", algo: " << s.algorithm                            //
              << ", time: " << s.time << ", ws: " << s.workspace_size //
              << ", name: " << miopen::solver::Id(s.solution_id).ToString();
}

// Modified copy from convolutionocl.cpp
std::vector<miopenConvSolution_t> GetSolutions(const FusionContext& ctx,
                                               const FusionDescription& problem,
                                               const size_t maxSolutionCount)
{
    const FindDbRecord fdb_record{ctx.GetStream(), problem, "fusion"};

    if(fdb_record.empty())
        return {};

    auto interim = std::vector<miopenConvSolution_t>{};
    interim.reserve(20); // Heuristic for speed.

    for(const auto& pair : fdb_record)
    {
        const auto solver_id = solver::Id{pair.first};

        // Wrong IDs can't be used to call IsApplicable(), so let's
        // ignore obsolete or invalid IDs read from find-db first.
        if(!solver_id.IsValid())
        {
            // Do not disturb users with warnings unless detailed log is enabled.
            MIOPEN_LOG_I("[Warning] incorrect solver_id: " << pair.first);
            continue;
        }

        // algorithm doesn't matter for our purpose here, so we stub it out
        interim.emplace_back(miopenConvSolution_t{pair.second.time,
                                                  pair.second.workspace,
                                                  solver_id.Value(),
                                                  miopenConvolutionAlgoDirect});
    }

    std::sort(begin(interim), end(interim), SolutionTimeComparator{});
    auto out = std::vector<miopenConvSolution_t>{};
    out.reserve(maxSolutionCount);
    auto n_copied = 0;
    for(const auto& s : interim)
    {
        const auto solver_id = solver::Id{s.solution_id};
        bool is_applicable   = false;

        GetAllFusionSolvers().FindById(
            solver_id, [&](auto solver) { is_applicable = solver.IsApplicable(ctx, problem); });

        if(!is_applicable)
            continue;
        out.push_back(s);
        if(++n_copied >= maxSolutionCount)
            break;
    }

    for(const auto& s : out)
        MIOPEN_LOG_I2(s);

    return out;
}

} // namespace

miopenStatus_t FusionPlanDescriptor::Compile(Handle& handle)
{
    std::vector<Allocator::ManageDataPtr> invoke_bufs;
    miopen::OperatorArgs params;

    const auto& fusion_problem = FusionDescription{this};
    std::vector<Solution> find_results;

    const auto network_config = fusion_problem.MakeNetworkConfig();
    auto invoker = handle.GetInvoker(network_config, std::nullopt, AlgorithmName{"fusion"});

    if(invoker)
    {
        invokers.push_back(*invoker);
        return miopenStatusSuccess;
    }

    {
        FindMode findMode(solver::Primitive::Fusion);
        auto sol = boost::optional<miopenConvSolution_t>{};

        if(findMode.IsFast(fusion_problem) || findMode.IsHybrid(fusion_problem))
        {
            const auto ctx      = FusionContext{handle};
            auto sols           = GetSolutions(ctx, fusion_problem, 1);
            const auto fallback = sols.empty();

            if(fallback)
            {
                auto fallback_failed = true;
                bool found           = false;

                GetAllFusionSolvers().Foreach([&](auto solver) {
                    if(found || !solver.IsApplicable(ctx, fusion_problem))
                        return;
                    const auto id  = solver::Id(solver.SolverDbId());
                    const auto wti = solver.GetWti(ctx, fusion_problem);
                    // Assume WTI == 1.0 (100%) is 10 ms.
                    // Return negative values as is, avoid DIV/0.
                    const auto time = wti <= 0.0f ? wti : (10.f / wti);
                    sols.push_back({time, 0, id.Value(), miopenConvolutionAlgoDirect});
                    fallback_failed = false;
                });

                if(fallback_failed)
                {
                    MIOPEN_LOG_I("No supported fusion solvers found");
                    return miopenStatusUnsupportedOp;
                }
            }

            // override the normal find with immed mode with env var
            if(!sols.empty() && (!(findMode.IsHybrid(fusion_problem) && fallback)))
            // || env::enabled(MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK)
            {
                std::sort(sols.begin(), sols.end(), SolutionTimeComparator());
                sol = sols.front();
            }
            // In Hybrid Find mode, we use Normal Find instead of Immediate fallback kernels.
        }

        if(sol.has_value())
        {
            // We need to create an invoker

            const auto id = solver::Id{sol->solution_id};

            GetAllFusionSolvers().FindById(id, [&](auto solver) {
                const auto ctx      = FusionContext{handle};
                auto db             = GetDb(ctx);
                const auto solution = solver::FindSolution(
                    solver, ctx, fusion_problem, db, {}); // auto tune is not expected here
                auto invoker =
                    handle.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
                // We register the invoker below

                auto ret = Solution{id, sol->time, solver.GetWorkspaceSize(ctx, fusion_problem)};
                ret.SetInvoker(std::move(invoker));
                find_results.push_back(std::move(ret));
            });
        }
        else
        {
            find_results = Find(handle, [&]() {
                return AllocateBuffersAndMakeFusionInvokeParams(
                    handle, fusion_problem, invoke_bufs, params, *this);
            });
        }
    }

    for(const auto& result : find_results)
    {
        const auto primitive = result.GetSolver().GetPrimitive();
        const auto algorithm = result.GetSolver().GetAlgo();

        if(conv_fwd_algo && primitive != solver::Primitive::Fusion &&
           algorithm != static_cast<miopenConvAlgorithm_t>(*conv_fwd_algo))
            continue;

        const auto id = result.GetSolver();
        invoker       = result.GetInvoker();

        if(!invoker)
            invoker = handle.GetInvoker(network_config, id);

        if(!invoker)
        {
            MIOPEN_LOG_E("Find-db has not produced an invoker");
            continue;
        }

        handle.RegisterInvoker(*invoker, network_config, id.ToString());
        invokers.push_back(std::move(*invoker));
        MIOPEN_LOG_I2(miopen::ConvolutionAlgoToString(algorithm));
    }

    if(invokers.empty())
    {
        MIOPEN_LOG_I("No supported fusion solvers found");
        return miopenStatusUnsupportedOp;
    }

    handle.SetAsFound1_0(
        network_config, AlgorithmName{"fusion"}, find_results.front().GetSolver().ToString());
    return miopenStatusSuccess;
}

std::vector<Solution>
FusionPlanDescriptor::Find(Handle& handle,
                           const std::function<fusion::FusionInvokeParams()>& invoke_params,
                           const std::optional<FindOptions>& options) const
{
    auto ctx = ExecutionContext(&handle);
    if(options)
        ctx.do_search = options->exhaustive_search;
    return FindFusion(ctx, this, invoke_params, options);
}

miopenStatus_t FusionPlanDescriptor::Execute(const Handle& handle,
                                             const TensorDescriptor& inputDesc,
                                             ConstData_t input,
                                             const TensorDescriptor& outputDesc,
                                             Data_t output,
                                             const OperatorArgs& op_args)
{
    miopen::debug::LogCmdFusion(this);

    if(output_desc != outputDesc)
    {
        MIOPEN_THROW(miopenStatusBadParm, "The output descriptors dont match.");
    }
    if(input_desc != inputDesc)
    {
        MIOPEN_THROW(miopenStatusBadParm, "The input descriptors dont match.");
    }
    if(invokers.empty())
    {
        MIOPEN_THROW(miopenStatusBadParm, "The Fusion Plan was not compiled successfully");
    }

    const auto plan_params =
        fusion::FusionInvokeParams{op_args, inputDesc, input, outputDesc, output, false};
    invokers[0](handle, plan_params);

    return miopenStatusSuccess;
}

} // namespace miopen
