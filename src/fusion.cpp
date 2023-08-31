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
#include <cassert>
#include <miopen/fusion.hpp>
#include <miopen/fusion_plan.hpp>
#include <miopen/logger.hpp>
#include <miopen/handle.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/fusion/solvers.hpp>
#include <miopen/fusion/fusion_invoke_params.hpp>
#include <miopen/find_solution.hpp>

#include <ostream>
#include <ios>
#include <algorithm>
#include <string>
#if !defined(_WIN32) && (HIP_PACKAGE_VERSION_FLAT >= 5006000000ULL)
#include <half/half.hpp>
#else
#include <half.hpp>
#endif

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
    if(z != nullptr || zDesc.GetSize() != 0)
        MIOPEN_THROW(miopenStatusNotImplemented, "The addition of z vector is not yet supported");
    FusionPlanDescriptor fusePlanDesc{miopenVerticalFusion, xDesc};
    OperatorArgs fusionArgs;
    auto convoOp = std::make_shared<ConvForwardOpDescriptor>(conv_desc, wDesc);
    auto biasOp  = std::make_shared<BiasFusionOpDescriptor>(biasDesc);
    auto activOp = std::make_shared<ActivFwdFusionOpDescriptor>(activationDesc.GetMode());
    MIOPEN_CHECK(fusePlanDesc.AddOp(convoOp));
    MIOPEN_CHECK(fusePlanDesc.SetConvAlgo(algo));
    MIOPEN_CHECK(fusePlanDesc.AddOp(biasOp));
    MIOPEN_CHECK(fusePlanDesc.AddOp(activOp));

    MIOPEN_CHECK(fusePlanDesc.Compile(handle));
    float alpha       = static_cast<float>(1.0);
    float beta        = static_cast<float>(0);
    float activ_alpha = activationDesc.GetAlpha();
    float activ_beta  = activationDesc.GetBeta();
    float activ_gamma = activationDesc.GetGamma();

    // Set the Args
    MIOPEN_CHECK(convoOp->SetArgs(fusionArgs, &alpha, &beta, w));
    MIOPEN_CHECK(activOp->SetArgs(fusionArgs, &alpha, &beta, activ_alpha, activ_beta, activ_gamma));
    MIOPEN_CHECK(biasOp->SetArgs(fusionArgs, &alpha, &beta, bias));
    MIOPEN_CHECK(fusePlanDesc.Execute(handle, xDesc, x, yDesc, y, fusionArgs));
    return miopenStatusSuccess;
}

static auto AllocateBuffersAndMakeConvBiasActivFusionInvokeParams(
    const FusionContext& context,
    const FusionDescription& problem,
    std::vector<Allocator::ManageDataPtr>& invoke_bufs,
    miopen::OperatorArgs& params)
{
    const int bias          = 1;
    const auto conv_problem = problem.GetConvProblem(0, conv::Direction::Forward, bias);
    const auto conv_ctx     = context.GetConvContext(conv_problem);

    auto& handle = conv_ctx.GetStream();

    invoke_bufs.push_back(handle.Create(conv_problem.GetBiasSize()));
    invoke_bufs.push_back(handle.Create(conv_problem.GetInSize()));
    invoke_bufs.push_back(handle.Create(conv_problem.GetWeightsSize()));
    invoke_bufs.push_back(handle.Create(conv_problem.GetOutSize()));

    MIOPEN_LOG_I("bias addr: " << invoke_bufs[0].get() << " , size: " << conv_problem.GetBiasSize()
                               << " , in addr: " << invoke_bufs[1].get()
                               << " , size: " << conv_problem.GetInSize()
                               << " , weigth addr: " << invoke_bufs[2].get()
                               << " , size: " << conv_problem.GetWeightsSize() << " , out addr: "
                               << invoke_bufs[3].get() << " , size: " << conv_problem.GetOutSize());

    const auto gfx90aaltimpl = conv_problem.GetConv().attribute.gfx90aFp16alt.GetFwd();

    auto conv_data =
        std::make_unique<miopen::fusion::ConvolutionOpInvokeParam>(invoke_bufs[2].get());
    auto bias_data = std::make_unique<miopen::fusion::BiasOpInvokeParam>(invoke_bufs[0].get());

    const float activ_alpha = 0.5f;
    const float activ_beta  = 0.5f;
    const float activ_gamma = 0.5f;
    auto activ_data         = std::make_unique<miopen::fusion::ActivationOpInvokeParam>(
        activ_alpha, activ_beta, activ_gamma);

    params.SetArg(0, std::move(conv_data));
    params.SetArg(1, std::move(bias_data));
    params.SetArg(2, std::move(activ_data));

    return miopen::fusion::FusionInvokeParams(params,
                                              conv_problem.GetIn(),
                                              invoke_bufs[1].get(),
                                              conv_problem.GetOut(),
                                              invoke_bufs[3].get(),
                                              gfx90aaltimpl);
}

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

miopenStatus_t ConvForwardOpDescriptor::SetArgs(OperatorArgs& args,
                                                const void* /*alpha*/,
                                                const void* /*beta*/,
                                                ConstData_t w)
{
    auto op_args = std::make_unique<fusion::ConvolutionOpInvokeParam>(w);
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
                                                             double epsilon)
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
                                                            double epsilon)
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
                                                            ConstData_t savedInvVariance)
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

std::string FusionPlanDescriptor::GetAlgorithmName(const Handle& /*handle*/)
{
    if(conv_fwd_algo)
        return miopen::ConvolutionAlgoToDirectionalString(
            static_cast<miopenConvAlgorithm_t>(*conv_fwd_algo), miopen::conv::Direction::Forward);
    MIOPEN_THROW(miopenStatusBadParm,
                 "GetAlgorithmName was called, but Algorithm has not been set");
}

static auto GetFusedSolvers()
{
    return solver::SolverContainer<solver::fusion::ConvBiasActivAsm1x1U,
                                   solver::fusion::ConvOclDirectFwdFused,
                                   solver::fusion::ConvBinWinogradRxSFused,
                                   solver::fusion::ConvBinWinogradRxSf2x3g1Fused,
                                   solver::fusion::BnFwdInferActivationFused,
                                   solver::fusion::BnFwdTrgActivationFused,
                                   solver::fusion::BnBwdTrgActivationFused,
                                   solver::fusion::ConvCKIgemmFwdBiasActivFused>{};
}

static NetworkConfig GetPlanConfig(const FusionContext& fusion_ctx,
                                   const FusionDescription& problem)
{
    std::ostringstream ss;
    const auto& input_desc  = problem.fusion_plan_desc->input_desc;
    const auto& output_desc = problem.fusion_plan_desc->output_desc;
    ss << input_desc.ToString() << ((input_desc.GetType() == miopenHalf) ? "FP16" : "FP32");
    ss << output_desc.ToString() << ((output_desc.GetType() == miopenHalf) ? "FP16" : "FP32");
    std::stringstream op_config;
    problem.GetNetworkConfig(op_config, fusion_ctx.GetStream());
    ss << op_config.str();
    return NetworkConfig{ss.str()};
}

static auto MakeFusionInvokeParams(const FusionContext& fusion_ctx,
                                   const FusionDescription& fusion_problem,
                                   std::vector<Allocator::ManageDataPtr>& invoke_bufs,
                                   miopen::OperatorArgs& params)
{
    if(fusion_problem.fusion_plan_desc->op_map.size() == 3 &&
       (fusion_problem.fusion_plan_desc->op_map[0]->kind() == miopenFusionOpConvForward) &&
       (fusion_problem.fusion_plan_desc->op_map[1]->kind() == miopenFusionOpBiasForward) &&
       (fusion_problem.fusion_plan_desc->op_map[2]->kind() == miopenFusionOpActivForward))
    {
        // Workaround: Fused API does not pass user-allocated buffers,
        // but we need these buffers during SearchForAllSolutions.
        // Since, SearchForAllSolutions invokes kernel launch and kernel launch needs these buffers.
        MIOPEN_LOG_I2("Allocating buffers for conv+bias+activ fusion");
        return AllocateBuffersAndMakeConvBiasActivFusionInvokeParams(
            fusion_ctx, fusion_problem, invoke_bufs, params);
    }
    else
    {
        // handle the rest of the fusion operators cases
        // eg: Convolution + Bias + BatchNorm + Activation,
        //     Convolution + BatchNorm + Activation
        //     Convolution + BatchNorm
        //     Convolution + Activation
        //     GEMM + Activation
        //
        MIOPEN_LOG_W("Allocating buffers for given fusion operators is not supported yet.");
        return miopen::fusion::FusionInvokeParams(OperatorArgs(),
                                                  miopen::TensorDescriptor(),
                                                  nullptr,
                                                  miopen::TensorDescriptor(),
                                                  nullptr,
                                                  false);
    }
}

miopenStatus_t FusionPlanDescriptor::Compile(Handle& handle)
{
    miopenStatus_t status = miopenStatusUnknownError;
    const auto solvers    = GetFusedSolvers();
    auto fusion_ctx       = FusionContext{handle};
    auto fusion_problem   = FusionDescription{this};
    AnyInvokeParams invoke_params;
    miopen::OperatorArgs params;
    std::vector<Allocator::ManageDataPtr> invoke_bufs;
    const FindEnforce enforce;
    // If we are tuning, then we need to allocate buffers.
    if(enforce.IsSearch(fusion_ctx))
        invoke_params = MakeFusionInvokeParams(fusion_ctx, fusion_problem, invoke_bufs, params);
    // During search mode, miopen invokes kernel to find the best config.
    // If memory allocation of the invoke params for the given fusion plan
    // is not supported we return early.
    if(enforce.IsSearch(fusion_ctx) && invoke_bufs.empty())
    {
        MIOPEN_LOG_I("No supported fusion solvers found during Search Mode.");
        return miopenStatusUnsupportedOp;
    }
    // tmp_sols is a collection of ConvSolutions that isApplicable for the fusion_problem.
    // These ConvSolutions stores instructions on how to build. It also stores invoker.
    const auto tmp_sols = solvers.SearchForAllSolutions(
        fusion_ctx, fusion_problem, miopen::GetDb(fusion_ctx), invoke_params);
    std::vector<miopen::solver::ConvSolution> sols;
    // Filter for Solvers
    if(conv_fwd_algo)
    {
        for(const auto& sol : tmp_sols)
        {
            const auto id      = miopen::solver::Id{sol.solver_id};
            const auto strAlgo = id.GetAlgo(miopen::conv::Direction::Forward);
            MIOPEN_LOG_I2(id.ToString());
            MIOPEN_LOG_I2(strAlgo);
            const auto algo = miopen::StringToConvolutionFwdAlgo(strAlgo);
            MIOPEN_LOG_I2(algo);
            if(algo == *conv_fwd_algo)
                sols.push_back(sol);
        }
    }
    else
        sols = tmp_sols;
    if(sols.empty())
    {
        MIOPEN_LOG_I("No supported fusion solvers found");
        return miopenStatusUnsupportedOp;
    }
    else
    {
        network_config = GetPlanConfig(fusion_ctx, fusion_problem);
        for(const auto& sol : sols)
        {
            if(!sol.invoker_factory)
                MIOPEN_THROW(miopenStatusInternalError,
                             "Invoker missing from solver " + sol.solver_id);
            const auto invoker =
                handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
            handle.RegisterInvoker(invoker, network_config, sol.solver_id, {});
            solutions.push_back(sol);
        }
        std::sort(solutions.begin(),
                  solutions.end(),
                  [](const solver::ConvSolution& a, const solver::ConvSolution& b) -> bool {
                      return a.weight > b.weight;
                  });
        status = miopenStatusSuccess;
    }
    return status;
}

miopenStatus_t FusionPlanDescriptor::Execute(const Handle& handle,
                                             const TensorDescriptor& inputDesc,
                                             ConstData_t input,
                                             const TensorDescriptor& outputDesc,
                                             Data_t output,
                                             const OperatorArgs& op_args)
{
    if(output_desc != outputDesc)
    {
        MIOPEN_THROW(miopenStatusBadParm, "The output descriptors dont match.");
    }
    if(input_desc != inputDesc)
    {
        MIOPEN_THROW(miopenStatusBadParm, "The input descriptors dont match.");
    }
    if(solutions.empty())
    {
        MIOPEN_THROW(miopenStatusBadParm, "The Fusion Plan was not compiled successfully");
    }
    const auto& solution = solutions[0];
    if(!solution.Succeeded())
    {
        MIOPEN_THROW(miopenStatusBadParm, "The Fusion Plan was not compiled");
    }

    const auto invoker = handle.GetInvoker(network_config, solver::Id{solution.solver_id}, {});
    const auto plan_params =
        fusion::FusionInvokeParams{op_args, inputDesc, input, outputDesc, output, false};
    (*invoker)(handle, plan_params);

    return miopenStatusSuccess;
}

} // namespace miopen
