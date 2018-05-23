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
                                                 miopenFusionDirection_t fuseDirection,
                                                 const miopenTensorDescriptor_t inputDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, fuseDirection, inputDesc);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t
miopenDestroyFusionPlanDescriptor(miopenFusionPlanDescriptor_t fusePlanDesc)
{

    MIOPEN_LOG_FUNCTION(fusePlanDesc)
    return (miopenStatusSuccess);
}

// Return an error code that is "NotImplemented", if it exists then return success
extern "C" miopenStatus_t miopenIsFusionPlanValid(miopenFusionPlanDescriptor_t fusePlanDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc);
    return (miopenStatusSuccess);
}

// Create convolution ops
extern "C" miopenStatus_t miopenCreateOpConvForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                    miopenOperatorDescriptor_t* convOp,
                                                    miopenConvolutionDescriptor_t convDesc,
                                                    miopenConvFwdAlgorithm_t fwdAlgo,
                                                    const miopenTensorDescriptor_t wDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, convOp, convDesc, fwdAlgo, wDesc);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenCreateOpConvBackwardData(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                         miopenOperatorDescriptor_t* convOp,
                                                         miopenConvolutionDescriptor_t convDesc,
                                                         miopenConvBwdDataAlgorithm_t bwdDataAlgo,
                                                         const miopenTensorDescriptor_t wDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, convOp, convDesc, bwdDataAlgo, wDesc);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t
miopenCreateOpConvBackwardWeights(miopenFusionPlanDescriptor_t fusePlanDesc,
                                  miopenOperatorDescriptor_t* convOp,
                                  miopenConvolutionDescriptor_t convDesc,
                                  miopenConvBwdWeightsAlgorithm_t bwdWeightsAlgo,
                                  const miopenTensorDescriptor_t wDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, convOp, convDesc, bwdWeightsAlgo, wDesc);
    return (miopenStatusSuccess);
}
//---

// Activation create ops
extern "C" miopenStatus_t
miopenCreateOpActivationForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                miopenOperatorDescriptor_t* activOp,
                                const miopenActivationDescriptor_t activDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, activOp, activDesc);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t
miopenCreateOpActivationBackward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                 miopenOperatorDescriptor_t* activOp,
                                 const miopenActivationDescriptor_t activDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, activOp, activDesc);
    return (miopenStatusSuccess);
}
//---

// Batch normalization create op
extern "C" miopenStatus_t
miopenCreateOpBatchNormInference(miopenFusionPlanDescriptor_t fusePlanDesc,
                                 miopenOperatorDescriptor_t* bnOp,
                                 const miopenBatchNormMode_t bn_mode)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, bnOp, bn_mode);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenCreateOpBatchNormForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                         miopenOperatorDescriptor_t* bnOp,
                                                         const miopenBatchNormMode_t bn_mode)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, bnOp, bn_mode);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenCreateOpBatchNormBackward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                          miopenOperatorDescriptor_t* bnOp,
                                                          const miopenBatchNormMode_t bn_mode)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, bnOp, bn_mode);
    return (miopenStatusSuccess);
}
//---

// Create TensorOps op
extern "C" miopenStatus_t miopenCreateOpTensorOp(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                 miopenOperatorDescriptor_t* tOp,
                                                 miopenTensorOp_t tensorOp,
                                                 const miopenTensorDescriptor_t bDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, tOp, tensorOp, bDesc);
    return (miopenStatusSuccess);
}
//---

// Create pooling ops
extern "C" miopenStatus_t miopenCreateOpPoolingForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                       miopenOperatorDescriptor_t* poolOp,
                                                       const miopenPoolingDescriptor_t poolDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, poolOp, poolDesc);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenCreateOpPoolingBackward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                        miopenOperatorDescriptor_t* poolOp,
                                                        const miopenPoolingDescriptor_t poolDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, poolOp, poolDesc);
    return (miopenStatusSuccess);
}
//---

extern "C" miopenStatus_t miopenCreateOperatorArgs(miopenOperatorArgs_t* args)
{
    MIOPEN_LOG_FUNCTION(args);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenDestroyOperatorArgs(miopenOperatorArgs_t args)
{

    MIOPEN_LOG_FUNCTION(args);
    return (miopenStatusSuccess);
}

// Fusion op args for Convolution
extern "C" miopenStatus_t miopenSetOpArgsConvForward(miopenOperatorArgs_t args,
                                                     const miopenOperatorDescriptor_t convOp,
                                                     const void* alpha,
                                                     const void* beta,
                                                     const void* w)
{
    MIOPEN_LOG_FUNCTION(args, alpha, beta, convOp, w);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenSetOpArgsConvBackwardData(miopenOperatorArgs_t args,
                                                          const miopenOperatorDescriptor_t convOp,
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
                                   const miopenOperatorDescriptor_t convOp,
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

// Fusion op args for Batch Normalization
extern "C" miopenStatus_t miopenSetOpArgsBatchNormInference(miopenOperatorArgs_t args,
                                                            const miopenOperatorDescriptor_t bnOp,
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
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenSetOpArgsBatchNormForward(miopenOperatorArgs_t args,
                                                          const miopenOperatorDescriptor_t bnOp,
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
                                                           const miopenOperatorDescriptor_t bnOp,
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
                                                        const miopenOperatorDescriptor_t poolingOp,
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
                                                         const miopenOperatorDescriptor_t poolingOp,
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
                                                  const miopenOperatorDescriptor_t tOp,
                                                  const void* alpha1,
                                                  const void* alpha2,
                                                  const void* B,
                                                  const void* beta)
{
    MIOPEN_LOG_FUNCTION(args, tOp, alpha1, alpha2, B, beta);
    return (miopenStatusSuccess);
}

// Return an error code that is "NotImplemented", if it exists then return success
extern "C" miopenStatus_t miopenExecuteFusionPlan(miopenHandle_t handle,
                                                  const miopenFusionPlanDescriptor_t fusePlanDesc,
                                                  const miopenTensorDescriptor_t inputDesc,
                                                  const void* input,
                                                  const miopenTensorDescriptor_t outputDesc,
                                                  void* output,
                                                  miopenOperatorArgs_t args)
{
    // MIOPEN_LOG_FUNCTION(handle, fusePlanDesc, inputDesc, input, outputDesc, output, args);
    return (miopenStatusSuccess);
}

// Heurtistic based benchmarking.
extern "C" miopenStatus_t miopenGetFusionPlanCostEstimate(
    miopenOpCost_t* opCost, miopenHandle_t handle, const miopenFusionPlanDescriptor_t fusePlanDesc)
{
    // MIOPEN_LOG_FUNCTION(opCost, handle, fusePlanDesc);
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
    return (miopenStatusSuccess);
}
