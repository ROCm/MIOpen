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
#include <initializer_list>
#include <memory>
#include <miopen/convolution.hpp>
#include <miopen/batch_norm.hpp>
#include <miopen/activ.hpp>
#include <miopen/fusion.hpp>
#include <miopen/fusion_plan.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor.hpp>

// Return an error code that is "NotImplemented", if it exists then return success
// This function should:
//		set up the place descriptor with expected input and ouput edges.
// 		Set up the internal datastructures for the fused kernel.
extern "C" miopenStatus_t miopenCreateFusionPlan(miopenFusionPlanDescriptor_t* fusePlanDesc,
                                                 const miopenFusionDirection_t fuseDirection,
                                                 const miopenTensorDescriptor_t inputDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, fuseDirection, inputDesc);
    return miopen::try_([&] {
        auto& desc = miopen::deref(fusePlanDesc);
        desc       = new miopen::FusionPlanDescriptor(fuseDirection, miopen::deref(inputDesc));
    });
}

extern "C" miopenStatus_t miopenDestroyFusionPlan(miopenFusionPlanDescriptor_t fusePlanDesc)
{

    MIOPEN_LOG_FUNCTION(fusePlanDesc);
    return miopen::try_([&] { miopen_destroy_object(fusePlanDesc); });
}

extern "C" miopenStatus_t miopenFusionPlanGetOp(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                const int op_idx,
                                                miopenFusionOpDescriptor_t* op)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, op_idx);
    miopenStatus_t res = miopenStatusBadParm;
    miopen::try_([&] {
        std::shared_ptr<miopen::FusionOpDescriptor> desc;
        res               = miopen::deref(fusePlanDesc).GetOp(op_idx, desc);
        miopen::deref(op) = desc.get();
    });
    return res;
}

// Return an error code that is "NotImplemented", if it exists then return success
extern "C" miopenStatus_t miopenCompileFusionPlan(miopenHandle_t handle,
                                                  miopenFusionPlanDescriptor_t fusePlanDesc)
{
    MIOPEN_LOG_FUNCTION(handle, fusePlanDesc);
    miopenStatus_t res = miopenStatusUnknownError;
    miopen::try_([&] { res = miopen::deref(fusePlanDesc).Compile(miopen::deref(handle)); });
    return res;
}

extern "C" miopenStatus_t
miopenFusionPlanGetWorkSpaceSize(miopenHandle_t handle,
                                 miopenFusionPlanDescriptor_t fusePlanDesc,
                                 size_t* workSpaceSize,
                                 miopenConvFwdAlgorithm_t algo)
{
    MIOPEN_LOG_FUNCTION(handle, fusePlanDesc, algo);
    miopenStatus_t res = miopenStatusUnknownError;
    miopen::try_([&] {
        size_t sz;
        res = miopen::deref(fusePlanDesc).GetWorkspaceSizeImmed(miopen::deref(handle), sz, algo);
        miopen::deref(workSpaceSize) = sz;
    });
    return res;
}

extern "C" miopenStatus_t
miopenFusionPlanConvolutionGetAlgo(miopenFusionPlanDescriptor_t fusePlanDesc,
                                   const int requestAlgoCount,
                                   int* returnedAlgoCount,
                                   miopenConvFwdAlgorithm_t* returnedAlgos)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, requestAlgoCount);
    miopenStatus_t res = miopenStatusUnknownError;
    miopen::try_([&] {
        int cnt = 0;
        res     = miopen::deref(fusePlanDesc).GetConvAlgos(requestAlgoCount, cnt, returnedAlgos);
        miopen::deref(returnedAlgoCount) = cnt;
    });
    return res;
}

extern "C" miopenStatus_t
miopenFusionPlanConvolutionSetAlgo(miopenFusionPlanDescriptor_t fusePlanDesc,
                                   miopenConvFwdAlgorithm_t algo)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, algo);
    miopenStatus_t res = miopenStatusUnknownError;
    miopen::try_([&] { res = miopen::deref(fusePlanDesc).SetConvAlgo(algo); });
    return res;
}

// Create convolution ops with unknown algorithms
extern "C" miopenStatus_t miopenCreateOpConvForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                    miopenFusionOpDescriptor_t* convOp,
                                                    miopenConvolutionDescriptor_t convDesc,
                                                    const miopenTensorDescriptor_t wDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, convOp, convDesc, wDesc);
    miopenStatus_t res = miopenStatusUnknownError;
    miopen::try_([&] {
        auto fod = std::make_shared<miopen::ConvForwardOpDescriptor>(miopen::deref(convDesc),
                                                                     miopen::deref(wDesc));
        miopen::deref(convOp) = fod.get();
        res                   = miopen::deref(fusePlanDesc).AddOp(fod);
    });
    return res;
}
// Activation create ops
extern "C" miopenStatus_t miopenCreateOpActivationForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                          miopenFusionOpDescriptor_t* activOp,
                                                          miopenActivationMode_t mode)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, activOp, mode);
    miopenStatus_t res = miopenStatusUnknownError;
    miopen::try_([&] {
        auto fod               = std::make_shared<miopen::ActivFwdFusionOpDescriptor>(mode);
        miopen::deref(activOp) = fod.get();
        res                    = miopen::deref(fusePlanDesc).AddOp(fod);
    });
    return res;
}

extern "C" miopenStatus_t
miopenCreateOpActivationBackward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                 miopenFusionOpDescriptor_t* activOp,
                                 miopenActivationMode_t mode)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, activOp, mode);
    miopenStatus_t res = miopenStatusUnknownError;
    miopen::try_([&] {
        auto fod               = std::make_shared<miopen::ActivBwdFusionOpDescriptor>(mode);
        miopen::deref(activOp) = fod.get();
        res                    = miopen::deref(fusePlanDesc).AddOp(fod);
    });
    return res;
}
//---

extern "C" miopenStatus_t miopenCreateOpBiasForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                    miopenFusionOpDescriptor_t* biasOp,
                                                    const miopenTensorDescriptor_t bDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, biasOp, bDesc);
    miopenStatus_t res = miopenStatusUnknownError;
    miopen::try_([&] {
        auto bod = std::make_shared<miopen::BiasFusionOpDescriptor>(miopen::deref(bDesc));
        miopen::deref(biasOp) = bod.get();
        res                   = miopen::deref(fusePlanDesc).AddOp(bod);
    });
    return res;
}

// Batch normalization create op
extern "C" miopenStatus_t
miopenCreateOpBatchNormInference(miopenFusionPlanDescriptor_t fusePlanDesc,
                                 miopenFusionOpDescriptor_t* bnOp,
                                 const miopenBatchNormMode_t bn_mode,
                                 const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, bnOp, bn_mode, bnScaleBiasMeanVarDesc);
    miopenStatus_t res = miopenStatusUnknownError;
    miopen::try_([&] {
        auto bod = std::make_shared<miopen::BatchNormInferenceFusionOpDescriptor>(
            bn_mode, miopen::deref(bnScaleBiasMeanVarDesc));
        miopen::deref(bnOp) = bod.get();
        res                 = miopen::deref(fusePlanDesc).AddOp(bod);
    });
    return res;
}

extern "C" miopenStatus_t miopenCreateOpBatchNormForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                         miopenFusionOpDescriptor_t* bnOp,
                                                         const miopenBatchNormMode_t bn_mode,
                                                         bool runningMeanVariance)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, bnOp, bn_mode, runningMeanVariance);
    miopenStatus_t res = miopenStatusUnknownError;
    miopen::try_([&] {
        auto bod = std::make_shared<miopen::BatchNormFwdTrainFusionOpDescriptor>(
            bn_mode, runningMeanVariance);
        miopen::deref(bnOp) = bod.get();
        res                 = miopen::deref(fusePlanDesc).AddOp(bod);
    });
    return res;
}

extern "C" miopenStatus_t miopenCreateOpBatchNormBackward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                          miopenFusionOpDescriptor_t* bnOp,
                                                          const miopenBatchNormMode_t bn_mode)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, bnOp, bn_mode);
    miopenStatus_t res = miopenStatusUnknownError;
    miopen::try_([&] {
        auto bod = std::make_shared<miopen::BatchNormBwdTrainFusionOpDescriptor>(bn_mode);
        miopen::deref(bnOp) = bod.get();
        res                 = miopen::deref(fusePlanDesc).AddOp(bod);
    });
    return res;
}
//---

extern "C" miopenStatus_t miopenCreateOperatorArgs(miopenOperatorArgs_t* args)
{
    MIOPEN_LOG_FUNCTION(args);
    return miopen::try_([&] {
        auto& theArgs = miopen::deref(args);
        theArgs       = new miopen::OperatorArgs();
    });
}

extern "C" miopenStatus_t miopenDestroyOperatorArgs(miopenOperatorArgs_t args)
{
    MIOPEN_LOG_FUNCTION(args);
    return miopen::try_([&] { miopen_destroy_object(args); });
}
extern "C" miopenStatus_t miopenSetOpArgsConvForward(miopenOperatorArgs_t args,
                                                     const miopenFusionOpDescriptor_t convOp,
                                                     const void* alpha,
                                                     const void* beta,
                                                     const void* w)
{
    MIOPEN_LOG_FUNCTION(args, alpha, beta, convOp, w);
    return miopen::try_([&] {
        auto&& op = dynamic_cast<miopen::ConvForwardOpDescriptor&>(miopen::deref(convOp));
        auto tmp  = DataCast(w);
        op.SetArgs(miopen::deref(args), alpha, beta, tmp);
    });
}

extern "C" miopenStatus_t miopenSetOpArgsBiasForward(miopenOperatorArgs_t args,
                                                     const miopenFusionOpDescriptor_t biasOp,
                                                     const void* alpha,
                                                     const void* beta,
                                                     const void* bias)
{

    MIOPEN_LOG_FUNCTION(args, biasOp, alpha, beta, bias);
    return miopen::try_([&] {
        auto&& op = dynamic_cast<miopen::BiasFusionOpDescriptor&>(miopen::deref(biasOp));
        op.SetArgs(miopen::deref(args), alpha, beta, DataCast(bias));
    });
}

extern "C" miopenStatus_t miopenSetOpArgsActivForward(miopenOperatorArgs_t args,
                                                      const miopenFusionOpDescriptor_t activFwdOp,
                                                      const void* alpha,
                                                      const void* beta,
                                                      double activAlpha,
                                                      double activBeta,
                                                      double activGamma)
{

    MIOPEN_LOG_FUNCTION(args, activFwdOp, alpha, beta, activAlpha, activBeta, activGamma);
    return miopen::try_([&] {
        auto&& op = dynamic_cast<miopen::ActivFwdFusionOpDescriptor&>(miopen::deref(activFwdOp));
        op.SetArgs(miopen::deref(args), alpha, beta, activAlpha, activBeta, activGamma);
    });
}

extern "C" miopenStatus_t miopenSetOpArgsActivBackward(miopenOperatorArgs_t args,
                                                       const miopenFusionOpDescriptor_t activBwdOp,
                                                       const void* alpha,
                                                       const void* beta,
                                                       const void* y,
                                                       const void* /*reserved*/,
                                                       double activAlpha,
                                                       double activBeta,
                                                       double activGamma)
{
    MIOPEN_LOG_FUNCTION(args, activBwdOp, alpha, beta, y, activAlpha, activBeta, activGamma);
    return miopen::try_([&] {
        auto&& op = dynamic_cast<miopen::ActivBwdFusionOpDescriptor&>(miopen::deref(activBwdOp));
        op.SetArgs(miopen::deref(args),
                   alpha,
                   beta,
                   DataCast(y),
                   nullptr,
                   activAlpha,
                   activBeta,
                   activGamma);
    });
}

// Fusion op args for Batch Normalization
extern "C" miopenStatus_t miopenSetOpArgsBatchNormInference(miopenOperatorArgs_t args,
                                                            const miopenFusionOpDescriptor_t bnOp,
                                                            const void* alpha,
                                                            const void* beta,
                                                            const void* bnScale,
                                                            const void* bnBias,
                                                            const void* estimatedMean,
                                                            const void* estimatedVariance,
                                                            double epsilon)
{
    MIOPEN_LOG_FUNCTION(
        args, bnOp, alpha, beta, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon);
    return miopen::try_([&] {
        auto&& op =
            dynamic_cast<miopen::BatchNormInferenceFusionOpDescriptor&>(miopen::deref(bnOp));
        op.SetArgs(miopen::deref(args),
                   alpha,
                   beta,
                   DataCast(bnScale),
                   DataCast(bnBias),
                   DataCast(estimatedMean),
                   DataCast(estimatedVariance),
                   epsilon);
    });
}

extern "C" miopenStatus_t miopenSetOpArgsBatchNormForward(miopenOperatorArgs_t args,
                                                          const miopenFusionOpDescriptor_t bnFwdOp,
                                                          const void* alpha,
                                                          const void* beta,
                                                          const void* bnScale,
                                                          const void* bnBias,
                                                          void* savedMean,
                                                          void* savedInvVariance,
                                                          void* runningMean,
                                                          void* runningVariance,
                                                          double expAvgFactor,
                                                          double epsilon)
{
    MIOPEN_LOG_FUNCTION(args,
                        bnFwdOp,
                        alpha,
                        beta,
                        bnScale,
                        bnBias,
                        savedMean,
                        savedInvVariance,
                        runningMean,
                        runningVariance,
                        expAvgFactor,
                        epsilon);
    return miopen::try_([&] {
        auto&& op =
            dynamic_cast<miopen::BatchNormFwdTrainFusionOpDescriptor&>(miopen::deref(bnFwdOp));
        op.SetArgs(miopen::deref(args),
                   alpha,
                   beta,
                   DataCast(runningMean),
                   DataCast(runningVariance),
                   DataCast(savedMean),
                   DataCast(savedInvVariance),
                   DataCast(bnScale),
                   DataCast(bnBias),
                   expAvgFactor,
                   epsilon);
    });
}

extern "C" miopenStatus_t miopenSetOpArgsBatchNormBackward(miopenOperatorArgs_t args,
                                                           const miopenFusionOpDescriptor_t bnBwdOp,
                                                           const void* alpha,
                                                           const void* beta,
                                                           const void* x,
                                                           const void* bnScale,
                                                           const void* bnBias,
                                                           void* resultBnScaleDiff,
                                                           void* resultBnBiasDiff,
                                                           const void* savedMean,
                                                           const void* savedInvVariance)
{
    MIOPEN_LOG_FUNCTION(args,
                        bnBwdOp,
                        alpha,
                        beta,
                        x,
                        bnScale,
                        bnBias,
                        resultBnScaleDiff,
                        resultBnBiasDiff,
                        savedMean,
                        savedInvVariance);
    return miopen::try_([&] {
        auto&& op =
            dynamic_cast<miopen::BatchNormBwdTrainFusionOpDescriptor&>(miopen::deref(bnBwdOp));
        op.SetArgs(miopen::deref(args),
                   alpha,
                   beta,
                   DataCast(x),
                   DataCast(bnScale),
                   DataCast(bnBias),
                   DataCast(resultBnScaleDiff),
                   DataCast(resultBnBiasDiff),
                   DataCast(savedMean),
                   DataCast(savedInvVariance));
    });
}
//---

// Return an error code that is "NotImplemented", if it exists then return success
extern "C" miopenStatus_t miopenExecuteFusionPlan(const miopenHandle_t handle,
                                                  const miopenFusionPlanDescriptor_t fusePlanDesc,
                                                  const miopenTensorDescriptor_t inputDesc,
                                                  const void* input,
                                                  const miopenTensorDescriptor_t outputDesc,
                                                  void* output,
                                                  miopenOperatorArgs_t args)
{
    MIOPEN_LOG_FUNCTION(handle, fusePlanDesc, inputDesc, input, outputDesc, output, args);
    return miopen::try_([&] {
        miopen::deref(fusePlanDesc)
            .Execute(miopen::deref(handle),
                     miopen::deref(inputDesc),
                     DataCast(input),
                     miopen::deref(outputDesc),
                     DataCast(output),
                     miopen::deref(args));
    });
}

extern "C" miopenStatus_t
miopenConvolutionBiasActivationForward(miopenHandle_t handle,
                                       const void* alpha1,
                                       const miopenTensorDescriptor_t xDesc,
                                       const void* x,
                                       const miopenTensorDescriptor_t wDesc,
                                       const void* w,
                                       const miopenConvolutionDescriptor_t conv_desc,
                                       miopenConvFwdAlgorithm_t algo,
                                       void* workspace,
                                       size_t workspaceSizeInBytes,
                                       const void* alpha2,
                                       const miopenTensorDescriptor_t zDesc,
                                       const void* z,
                                       const miopenTensorDescriptor_t biasDesc,
                                       const void* bias,
                                       const miopenActivationDescriptor_t activationDesc,
                                       const miopenTensorDescriptor_t yDesc,
                                       void* y)
{

    MIOPEN_LOG_FUNCTION(handle,
                        alpha1,
                        xDesc,
                        x,
                        wDesc,
                        w,
                        conv_desc,
                        algo,
                        workspace,
                        workspaceSizeInBytes,
                        alpha2,
                        zDesc,
                        z,
                        biasDesc,
                        bias,
                        activationDesc,
                        ydesc,
                        y);
    miopenStatus_t res = miopenStatusUnknownError;
    const auto try_res = miopen::try_([&] {
        res = ConvBiasActivFusion(miopen::deref(handle),
                                  alpha1,
                                  miopen::deref(xDesc),
                                  DataCast(x),
                                  miopen::deref(wDesc),
                                  DataCast(w),
                                  miopen::deref(conv_desc),
                                  algo,
                                  DataCast(workspace),
                                  workspaceSizeInBytes,
                                  alpha2,
                                  miopen::deref(zDesc),
                                  DataCast(z),
                                  miopen::deref(biasDesc),
                                  DataCast(bias),
                                  miopen::deref(activationDesc),
                                  miopen::deref(yDesc),
                                  DataCast(y));
    });
    if(try_res == miopenStatusSuccess)
        return res;
    return try_res;
}
