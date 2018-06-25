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
#include <array>
#include <initializer_list>
#include <memory>
#include <miopen/pooling.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/convolution.hpp>
#include <miopen/batch_norm.hpp>
#include <miopen/activ.hpp>
#include <miopen/fusion.hpp>
#include <miopen/errors.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>

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
        miopen::deref(fusePlanDesc) =
            new miopen::FusionPlanDescriptor(fuseDirection, miopen::deref(inputDesc));
    });
}

extern "C" miopenStatus_t
miopenDestroyFusionPlanDescriptor(miopenFusionPlanDescriptor_t fusePlanDesc)
{

    MIOPEN_LOG_FUNCTION(fusePlanDesc)
    return miopen::try_([&] { miopen_destroy_object(fusePlanDesc); });
}

// Return an error code that is "NotImplemented", if it exists then return success
extern "C" miopenStatus_t miopenCompileFusionPlan(miopenHandle_t handle,
                                                  miopenFusionPlanDescriptor_t fusePlanDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc);
    return miopen::try_([&] { miopen::deref(fusePlanDesc).Compile(miopen::deref(handle)); });
}

// Create convolution ops with known algorithm
extern "C" miopenStatus_t miopenCreateOpConvForwardAlgo(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                        miopenFusionOpDescriptor_t* convOp,
                                                        miopenConvolutionDescriptor_t convDesc,
                                                        miopenConvFwdAlgorithm_t fwdAlgo,
                                                        const miopenTensorDescriptor_t wDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, convOp, convDesc, fwdAlgo, wDesc);
    miopenStatus_t res = miopenStatusSuccess;
    miopen::try_([&] {
        auto fod = std::make_shared<miopen::ConvForwardOpDescriptor>(
            miopen::deref(convDesc), miopen::deref(wDesc), fwdAlgo);
        miopen::deref(convOp) = fod.get();
        res                   = miopen::deref(fusePlanDesc).AddOp(fod);
    });
    return res;
}

extern "C" miopenStatus_t
miopenFusionPlanGetWorkSpaceSize(miopenHandle_t handle,
                                 miopenFusionPlanDescriptor_t fusePlanDesc,
                                 size_t* workSpaceSize,
                                 miopenConvFwdAlgorithm_t algo)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, workSpaceSize);
    miopenStatus_t res = miopenStatusSuccess;
    miopen::try_([&] {
        size_t sz;
        res = miopen::deref(fusePlanDesc).GetWorkspaceSizeImmed(miopen::deref(handle), sz, algo);
        miopen::deref(workSpaceSize) = sz;
    });
    return res;
}

extern "C" miopenStatus_t
miopenCreateOpConvBackwardDataAlgo(miopenFusionPlanDescriptor_t fusePlanDesc,
                                   miopenFusionOpDescriptor_t* convOp,
                                   miopenConvolutionDescriptor_t convDesc,
                                   miopenConvBwdDataAlgorithm_t bwdDataAlgo,
                                   const miopenTensorDescriptor_t wDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, convOp, convDesc, bwdDataAlgo, wDesc);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t
miopenCreateOpConvBackwardWeightsAlgo(miopenFusionPlanDescriptor_t fusePlanDesc,
                                      miopenFusionOpDescriptor_t* convOp,
                                      miopenConvolutionDescriptor_t convDesc,
                                      miopenConvBwdWeightsAlgorithm_t bwdWeightsAlgo,
                                      const miopenTensorDescriptor_t wDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, convOp, convDesc, bwdWeightsAlgo, wDesc);
    return (miopenStatusSuccess);
}

// Create convolution ops with unknown algorithms
extern "C" miopenStatus_t miopenCreateOpConvForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                    miopenFusionOpDescriptor_t* convOp,
                                                    miopenConvolutionDescriptor_t convDesc,
                                                    const miopenTensorDescriptor_t wDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, convOp, convDesc, wDesc);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenCreateOpConvBackwardData(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                         miopenFusionOpDescriptor_t* convOp,
                                                         miopenConvolutionDescriptor_t convDesc,
                                                         const miopenTensorDescriptor_t wDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, convOp, convDesc, wDesc);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t
miopenCreateOpConvBackwardWeights(miopenFusionPlanDescriptor_t fusePlanDesc,
                                  miopenFusionOpDescriptor_t* convOp,
                                  miopenConvolutionDescriptor_t convDesc,
                                  const miopenTensorDescriptor_t wDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, convOp, convDesc, wDesc);
    return (miopenStatusSuccess);
}

//---

// Activation create ops
extern "C" miopenStatus_t miopenCreateOpActivationForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                          miopenFusionOpDescriptor_t* activOp,
                                                          miopenActivationMode_t mode)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, activOp, mode);
    miopenStatus_t res = miopenStatusSuccess;
    miopen::try_([&] {
        auto fod               = std::make_shared<miopen::ActivFusionOpDescriptor>(mode);
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
    return (miopenStatusSuccess);
}
//---

extern "C" miopenStatus_t miopenCreateOpBiasForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                    miopenFusionOpDescriptor_t* biasOp,
                                                    const miopenTensorDescriptor_t bDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, biasOp, bDesc);
    miopenStatus_t res = miopenStatusSuccess;
    miopen::try_([&] {
        auto bod = std::make_shared<miopen::BiasFusionOpDescriptor>(miopen::deref(bDesc));
        miopen::deref(biasOp) = bod.get();
        res                   = miopen::deref(fusePlanDesc).AddOp(bod);
    });
    return res;
}

extern "C" miopenStatus_t miopenCreateOpBiasBackward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                     miopenFusionOpDescriptor_t* biasOp,
                                                     const miopenTensorDescriptor_t dbDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, biasOp, dbDesc);
    return (miopenStatusSuccess);
}

// Batch normalization create op
extern "C" miopenStatus_t
miopenCreateOpBatchNormInference(miopenFusionPlanDescriptor_t fusePlanDesc,
                                 miopenFusionOpDescriptor_t* bnOp,
                                 const miopenBatchNormMode_t bn_mode,
                                 const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, bnOp, bn_mode, bnScaleBiasMeanVarDesc);
    miopenStatus_t res = miopenStatusSuccess;
    miopen::try_([&] {
        auto bod = std::make_shared<miopen::BatchNormInferenceFusionOpDescriptor>(
            bn_mode, miopen::deref(bnScaleBiasMeanVarDesc));
        miopen::deref(bnOp) = bod.get();
        res                 = miopen::deref(fusePlanDesc).AddOp(bod);
    });
    return res;
}

extern "C" miopenStatus_t
miopenCreateOpBatchNormForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                               miopenFusionOpDescriptor_t* bnOp,
                               const miopenBatchNormMode_t bn_mode,
                               const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, bnOp, bn_mode, bnScaleBiasMeanVarDesc);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t
miopenCreateOpBatchNormBackward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                miopenFusionOpDescriptor_t* bnOp,
                                const miopenBatchNormMode_t bn_mode,
                                const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, bnOp, bn_mode, bnScaleBiasMeanVarDesc);
    return (miopenStatusSuccess);
}
//---

// Create TensorOps op
extern "C" miopenStatus_t miopenCreateOpTensorOp(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                 miopenFusionOpDescriptor_t* tOp,
                                                 miopenTensorOp_t tensorOp,
                                                 const miopenTensorDescriptor_t bDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, tOp, tensorOp, bDesc);
    return (miopenStatusSuccess);
}
//---

// Create pooling ops
extern "C" miopenStatus_t miopenCreateOpPoolingForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                       miopenFusionOpDescriptor_t* poolOp,
                                                       const miopenPoolingDescriptor_t poolDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, poolOp, poolDesc);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenCreateOpPoolingBackward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                        miopenFusionOpDescriptor_t* poolOp,
                                                        const miopenPoolingDescriptor_t poolDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, poolOp, poolDesc);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenCreateOperatorArgs(miopenOperatorArgs_t* args)
{
    MIOPEN_LOG_FUNCTION(args);
    return miopen::try_([&] { miopen::deref(args) = new miopen::OperatorArgs(); });
}

extern "C" miopenStatus_t miopenDestroyOperatorArgs(miopenOperatorArgs_t args)
{
    MIOPEN_LOG_FUNCTION(args);
    return miopen::try_([&] { miopen_destroy_object(args); });
}

// Fusion op args for Convolution
extern "C" miopenStatus_t miopenSetOpArgsConvForward(miopenOperatorArgs_t args,
                                                     const miopenFusionOpDescriptor_t convOp,
                                                     const void* alpha,
                                                     const void* beta,
                                                     const void* w)
{
    MIOPEN_LOG_FUNCTION(args, alpha, beta, convOp, w);
    return miopen::try_([&] {
        auto op  = dynamic_cast<miopen::ConvForwardOpDescriptor&>(miopen::deref(convOp));
        auto tmp = DataCast(w);
        op.SetArgs(miopen::deref(args), alpha, beta, tmp);
    });
}

extern "C" miopenStatus_t miopenSetOpArgsConvBackwardData(miopenOperatorArgs_t args,
                                                          const miopenFusionOpDescriptor_t convOp,
                                                          const void* alpha,
                                                          const void* beta,
                                                          const void* w,
                                                          void* workSpace,
                                                          size_t workSpaceSize)
{
    MIOPEN_LOG_FUNCTION(args, convOp, alpha, beta, w, workSpace, workSpaceSize);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t
miopenSetOpArgsConvBackwardWeights(miopenOperatorArgs_t args,
                                   const miopenFusionOpDescriptor_t convOp,
                                   const void* alpha,
                                   const void* beta,
                                   const void* x,
                                   void* dw,
                                   void* workSpace,
                                   size_t workSpaceSize)
{
    MIOPEN_LOG_FUNCTION(args, convOp, alpha, beta, x, dw, workSpace, workSpaceSize);
    return (miopenStatusSuccess);
}
//----

// Fusion op args for bias
extern "C" miopenStatus_t miopenSetOpArgsBiasForward(miopenOperatorArgs_t args,
                                                     const miopenFusionOpDescriptor_t biasOp,
                                                     const void* alpha,
                                                     const void* beta,
                                                     const void* bias)
{

    MIOPEN_LOG_FUNCTION(args, biasOp, alpha, beta, bias);
    return miopen::try_([&] {
        auto op = dynamic_cast<miopen::BiasFusionOpDescriptor&>(miopen::deref(biasOp));
        op.SetArgs(miopen::deref(args), alpha, beta, DataCast(bias));
    });
}

extern "C" miopenStatus_t miopenSetOpArgsBiasBackward(miopenOperatorArgs_t args,
                                                      const miopenFusionOpDescriptor_t dbiasOp,
                                                      const void* alpha,
                                                      const void* beta,
                                                      const void* dbias)
{

    MIOPEN_LOG_FUNCTION(args, dbiasOp, alpha, beta, dbias);
    return (miopenStatusSuccess);
}
//---

extern "C" miopenStatus_t miopenSetOpArgsActivForward(miopenOperatorArgs_t args,
                                                      const miopenFusionOpDescriptor_t activOp,
                                                      const void* alpha,
                                                      const void* beta,
                                                      double activAlpha,
                                                      double activBeta,
                                                      double activGamma)
{

    MIOPEN_LOG_FUNCTION(args, activOp, alpha, beta, activAlpha, activBeta, activGamma);
    return miopen::try_([&] {
        auto op = dynamic_cast<miopen::ActivFusionOpDescriptor&>(miopen::deref(activOp));
        op.SetArgs(miopen::deref(args), alpha, beta, activAlpha, activBeta, activGamma);
    });
}

extern "C" miopenStatus_t miopenSetOpArgsActivBackward(miopenOperatorArgs_t args,
                                                       const miopenFusionOpDescriptor_t activOp,
                                                       const void* alpha,
                                                       const void* beta,
                                                       double activAlpha,
                                                       double activBeta,
                                                       double activGamma)
{
    MIOPEN_LOG_FUNCTION(args, activOp, alpha, beta, activAlpha, activBeta, activGamma);
    return (miopenStatusSuccess);
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
        auto op = dynamic_cast<miopen::BatchNormInferenceFusionOpDescriptor&>(miopen::deref(bnOp));
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
                                                          const miopenFusionOpDescriptor_t bnOp,
                                                          const void* alpha,
                                                          const void* beta,
                                                          const void* bnScale,
                                                          const void* bnBias,
                                                          void* savedMean,
                                                          void* savedInvVariance,
                                                          void* runningMean,
                                                          void* runningVariance,
                                                          double epsilon)
{
    MIOPEN_LOG_FUNCTION(args,
                        bnOp,
                        alpha,
                        beta,
                        bnScale,
                        bnBias,
                        savedMean,
                        savedInvVariance,
                        runningMean,
                        runningVariance,
                        epsilon);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenSetOpArgsBatchNormBackward(miopenOperatorArgs_t args,
                                                           const miopenFusionOpDescriptor_t bnOp,
                                                           const void* alpha,
                                                           const void* beta,
                                                           const void* x,
                                                           const void* bnScale,
                                                           void* resultBnScaleDiff,
                                                           void* resultBnBiasDiff,
                                                           const void* savedMean,
                                                           const void* savedInvVariance)
{
    MIOPEN_LOG_FUNCTION(args,
                        bnOp,
                        alpha,
                        beta,
                        x,
                        bnScale,
                        resultBnScaleDiff,
                        resultBnBiasDiff,
                        savedMean,
                        savedInvVariance);
    return (miopenStatusSuccess);
}
//---

// Pooling arg ops
extern "C" miopenStatus_t miopenSetOpArgsPoolingForward(miopenOperatorArgs_t args,
                                                        const miopenFusionOpDescriptor_t poolingOp,
                                                        const void* alpha,
                                                        const void* beta,
                                                        bool do_backward,
                                                        void* workSpace,
                                                        size_t workSpaceSize)
{

    MIOPEN_LOG_FUNCTION(args, poolingOp, alpha, beta, do_backward, workSpace, workSpaceSize);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenSetOpArgsPoolingBackward(miopenOperatorArgs_t args,
                                                         const miopenFusionOpDescriptor_t poolingOp,
                                                         const void* alpha,
                                                         const void* beta,
                                                         const void* y,
                                                         const void* x,
                                                         const void* workSpace,
                                                         size_t workSpaceSize)
{

    MIOPEN_LOG_FUNCTION(args, poolingOp, alpha, beta, y, x, workSpace, workSpaceSize);
    return (miopenStatusSuccess);
}
//----

extern "C" miopenStatus_t miopenSetOpArgsTensorOp(miopenOperatorArgs_t args,
                                                  const miopenFusionOpDescriptor_t tOp,
                                                  const void* alpha1,
                                                  const void* alpha2,
                                                  const void* B,
                                                  const void* beta)
{
    MIOPEN_LOG_FUNCTION(args, tOp, alpha1, alpha2, B, beta);
    return (miopenStatusSuccess);
}

// Return an error code that is "NotImplemented", if it exists then return success
extern "C" miopenStatus_t miopenExecuteFusionPlan(const miopenFusionPlanDescriptor_t fusePlanDesc,
                                                  const miopenTensorDescriptor_t inputDesc,
                                                  const void* input,
                                                  const miopenTensorDescriptor_t outputDesc,
                                                  void* output,
                                                  miopenOperatorArgs_t args)
{
    // MIOPEN_LOG_FUNCTION(handle, fusePlanDesc, inputDesc, input, outputDesc, output, args);
    return miopen::try_([&] {

        miopen::deref(fusePlanDesc)
            .Execute(miopen::deref(inputDesc),
                     DataCast(input),
                     miopen::deref(outputDesc),
                     DataCast(output),
                     miopen::deref(args));
    });
}

// Heurtistic based benchmarking.
extern "C" miopenStatus_t miopenGetFusionPlanCostEstimate(
    miopenOpCost_t* opCost, miopenHandle_t handle, const miopenFusionPlanDescriptor_t fusePlanDesc)
{
    // MIOPEN_LOG_FUNCTION(opCost, handle, fusePlanDesc);
    (void)(opCost);
    (void)(handle);
    (void)(fusePlanDesc);
    return (miopenStatusSuccess);
}

// Empirical benchmarking, aka we actually run the fusion plan.
extern "C" miopenStatus_t
miopenGetFusionPlanCostEmpirical(miopenOpCost_t* opCost,
                                 miopenHandle_t handle,
                                 const miopenFusionPlanDescriptor_t fusePlanDesc,
                                 const miopenTensorDescriptor_t inputDesc,
                                 const void* input,
                                 miopenOperatorArgs_t args)
{
    // MIOPEN_LOG_FUNCTION(opCost, handle, fusePlanDesc, inputDesc, input, args);
    (void)(handle);
    (void)(fusePlanDesc);
    (void)(inputDesc);
    (void)(input);
    (void)(args);
    (void)(opCost);
    return (miopenStatusSuccess);
}
