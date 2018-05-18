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
#include <miopen/errors.hpp>
#include <miopen/logger.hpp>


// Return an error code that is "NotImplemented", if it exists then return success
// This function should:
//		set up the place descriptor with expected input and ouput edges.
// 		Set up the internal datastructures for the fused kernel.
extern "C" miopenStatus_t miopenCreateFusionPlan(miopenFusionPlanDescriptor_t* fusePlanDesc,
                                               miopenFusionDirection_t fuseDirection,
                                               const TensorDescriptor& xDesc,
                                               const TensorDescriptor& yDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, fuseDirection, xDesc, yDesc);
    return (miopenStatusSuccess);
}


// Return an error code that is "NotImplemented", if it exists then return success
extern "C" miopenStatus_t miopenIsFusionPlanValid(miopenFusionPlanDescriptor_t fusePlanDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc);
    return (miopenStatusSuccess);
}



// IF the algo is not present
extern "C" miopenStatus_t miopenCreateOpConvolutionForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                      miopenOperatorDescriptor_t* convOp,
                                                      miopenConvolutionDescriptor_t convDesc,
                                                      miopenConvFwdAlgorithm_t fwdAlgo,
                                                      const TensorDescriptor& wDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, convOp, convDesc, fwdAlgo, wDesc);
    return (miopenStatusSuccess);
}



extern "C" miopenStatus_t miopenCreateOpConvolutionForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                      miopenOperatorDescriptor_t* convOp,
                                                      miopenConvolutionDescriptor_t convDesc,
                                                      miopenConvFwdAlgorithm_t fwdAlgo,
                                                      const TensorDescriptor& wDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, convOp, convDesc, fwdAlgo, wDesc);
    return (miopenStatusSuccess);
}




extern "C" miopenStatus_t
miopenCreateOpActivationForward(miopenFusionPlanDescriptor_t fusePlanDesc, miopenOperatorDescriptor_t* activOp,
                                  const miopenActivationDescriptor_t activDesc)
{
    MIOPEN_LOG_FUNCTION(activOp, activDesc);
    return (miopenStatusSuccess);
}



extern "C" miopenStatus_t
miopenCreateOpBatchNormInference(miopenFusionPlanDescriptor_t fusePlanDesc, miopenOperatorDescriptor_t *bnOp,
                                 const miopenBatchNormMode_t bn_mode)
{
    MIOPEN_LOG_FUNCTION(bnOp, bn_mode);
    return (miopenStatusSuccess);
}



extern "C" miopenStatus_t
miopenCreateOpBatchNormForward(miopenFusionPlanDescriptor_t fusePlanDesc, miopenOperatorDescriptor_t *bnOp,
                                 const miopenBatchNormMode_t bn_mode)
{
    MIOPEN_LOG_FUNCTION(bnOp, bn_mode);
    return (miopenStatusSuccess);
}



extern "C" miopenStatus_t miopenCreateOpTensorOp(miopenOperatorDescriptor_t tOp,
                                                  miopenTensorOp_t tensorOp,
                                                  const TensorDescriptor& aDesc,
                                                  const TensorDescriptor& bDesc,
                                                  const TensorDescriptor& cDesc)
{
    MIOPEN_LOG_FUNCTION(tOp, tensorOp, aDesc, bDesc, cDesc);
    return (miopenStatusSuccess);
}


extern "C" miopenStatus_t miopenCreateOpPoolingForward(miopenOperatorDescriptor_t poolOp, 
                                                        const miopenPoolingDescriptor_t poolDesc)
{
    MIOPEN_LOG_FUNCTION(poolOp, poolDesc);
    return (miopenStatusSuccess);
}



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

extern "C" miopenStatus_t miopenSetOpArgsConvForward(miopenOperatorArgs_t args,
                                                      const miopenOperatorDescriptor_t convOp,
                                                      const void* w)
{
    MIOPEN_LOG_FUNCTION(args, convOp, w);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenSetOpArgsBatchNormInference(miopenOperatorArgs_t args,
                                                            const miopenOperatorDescriptor_t bnOp,
                                                            const void* bnScale,
                                                            const void* bnBias,
                                                            const void* estimatedMean,
                                                            const void* estimatedVariance,
                                                            double epsilon)
{
  MIOPEN_LOG_FUNCTION(args, bnOp, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon);
  return (miopenStatusSuccess);
}


extern "C" miopenStatus_t miopenSetOpArgsBatchNormForward(miopenOperatorArgs_t args,
                                                            const miopenOperatorDescriptor_t bnOp,
                                                            const void* bnScale,
                                                            const void* bnBias,
                                                            void* savedMean,
                                                            void* savedInvVariance,
                                                            void* runningMean,
                                                            void* runningVariance,
                                                            double epsilon)
{
  MIOPEN_LOG_FUNCTION(args, bnOp, bnScale, bnBias, savedMean, savedInvVariance, runningMean, runningVariance, epsilon);
  return (miopenStatusSuccess);
}

// This is essentially a noop, but it forces the users to have a matching length arg array in
// execute
// Potentially should be removed.
extern "C" miopenStatus_t
miopenSetOpArgsPoolingForward(miopenOperatorArgs_t args,
                                const miopenOperatorDescriptor_t poolingOp, 
                                bool do_backward, 
                                void* workSpace, 
                                size_t workSpaceSize)
{

  MIOPEN_LOG_FUNCTION(args, poolingOp);
  return (miopenStatusSuccess);
}

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
// This function should:
//	   Associate the data structure pointers to the user input and do checks
//     This function allows the user to reuse fusion plan.
// The number of operator descriptors must equal the number of args and both should match
// the value stored in the fused plan descriptor.
extern "C" miopenStatus_t miopenExecuteFusionPlan(const miopenHandle_t handle,
                                                  const miopenFusionPlanDescriptor_t fusePlanDesc,
                                                  const miopenTensorDescriptor_t xDesc,
                                                  const void* x,
                                                  const miopenTensorDescriptor_t yDesc,
                                                  void* y,
                                                  miopenOperatorArgs_t args)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, xDesc, x, yDesc, y, args);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenGetOperatorType(const miopenOperatorDescriptor miopenOp,
                                                miopenOperator_t* opType)
{
    MIOPEN_LOG_FUNCTION(miopenOp, opType);
    return (miopenStatusSuccess);
}

// Heurtistic based benchmarking.
extern "C" miopenStatus_t
miopenGetFusionPlanCostEstimate(const miopenHandle_t handle,
                                const miopenFusionPlanDescriptor_t fusePlanDesc,
                                miopenOpCost_t* opCost)
{
    MIOPEN_LOG_FUNCTION(miopenOp, fusePlanDesc, opReal, opCost);
    return (miopenStatusSuccess);
}

// Empirical benchmarking, aka we actually run the fusion plan.
extern "C" miopenStatus_t
miopenGetFusionPlanCostEmpirical(const miopenHandle_t handle,
                                 const miopenFusionPlanDescriptor_t fusePlanDesc,
                                 const size_t workSpaceSize,
                                 void* workSpace,
                                 openOperatorArgs_t args miopenOpCost_t* opCost)
{
    MIOPEN_LOG_FUNCTION(handle, fusePlanDesc, workSpaceSize, workSpace, args, opCost);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t
miopenDestroyFusionPlanDescriptor(miopenFusionPlanDescriptor_t fusePlanDesc)
{

    MIOPEN_LOG_FUNCTION(fusePlanDesc)
    return (miopenStatusSuccess);
}

