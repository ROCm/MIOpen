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
extern "C" miopenStatus_t
miopenCreateFusionPlanDescriptor(miopenFusionPlanDescriptor_t* fusePlanDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc);
    return (miopenStatusSuccess);
}

// Return an error code that is "NotImplemented", if it exists then return success
// This function should:
//		set up the place descriptor with expected input and ouput edges.
// 		Set up the internal datastructures for the fused kernel.
extern "C" miopenStatus_t miopenInitFusionPlan(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                 const miopenPipelineMode_t pipelineMode,
                                                 const miopenFusionDirection_t fuseDirection)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, pipelineMode, fuseDirection, xDesc);
    return (miopenStatusSuccess);
}


// Return an error code that is "NotImplemented", if it exists then return success
// This function should:
//    set up the place descriptor with expected input and ouput edges.
//    Set up the internal datastructures for the fused kernel.
extern "C" miopenStatus_t
miopenAddOpToFusionPlan(miopenFusionPlanDescriptor_t fusePlanDesc,
                              miopenOperatorDescriptor_t op)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, op);
    return (miopenStatusSuccess);
}




extern "C" miopenStatus_t
miopenResetFusionPlan(miopenFusionPlanDescriptor_t fusePlanDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc);
    return (miopenStatusSuccess);
}


// Return an error code that is "NotImplemented", if it exists then return success
extern "C" miopenStatus_t
miopenIsFusionPlanValid(miopenFusionPlanDescriptor_t fusePlanDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc);
    return (miopenStatusSuccess);
}




extern "C" miopenStatus_t
miopenCreateOp(miopenOperatorDescriptor_t* Op, const miopenOperator_t operator)
{

    MIOPEN_LOG_FUNCTION(Op, dataType);
    return (miopenStatusSuccess);
}


// IF the algo is not present 
extern "C" miopenStatus_t miopenConfigConvInferenceOp(miopenOperatorDescriptor_t convOp,
                                                      miopenConvolutionDescriptor_t convDesc,
                                                      miopenConvFwdAlgorithm_t algo,
                                                      const TensorDescriptor& xDesc,
                                                      const TensorDescriptor& wDesc,
                                                      const TensorDescriptor& yDesc)
{
    MIOPEN_LOG_FUNCTION(convOp, convDesc, algo, xDesc, wDesc, yDesc);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t
miopenConfigActivationInferenceOp(miopenOperatorDescriptor_t activOp,
                                  const miopenActivationDescriptor_t activDesc,
                                  const TensorDescriptor& xDesc,
                                  const TensorDescriptor& yDesc)
{
    MIOPEN_LOG_FUNCTION(activOp, activDesc, xDesc, yDesc);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t
miopenConfigBatchNormInferenceOp(miopenOperatorDescriptor_t bnOp,
                                 const miopenBatchNormMode_t bn_mode,
                                 const TensorDescriptor& xDesc,
                                 const TensorDescriptor& yDesc,
                                 const TensorDescriptor& bnScaleBiasMeanVarDesc)
{

    MIOPEN_LOG_FUNCTION(bnOp, bn_mode, xDesc, yDesc, bnScaleBiasMeanVarDesc);
    return (miopenStatusSuccess);
}


extern "C" miopenStatus_t
miopenConfigTensorOpInferenceOp(miopenOperatorDescriptor_t tOp,
                                 miopenTensorOp_t tensorOp,
                                 const TensorDescriptor& aDesc,
                                 const TensorDescriptor& bDesc,
                                 const TensorDescriptor& cDesc)
{

    MIOPEN_LOG_FUNCTION(tOp, tensorOp, aDesc, bDesc, cDesc);
    return (miopenStatusSuccess);
}


extern "C" miopenStatus_t
miopenConfigPoolingInferenceOp(miopenOperatorDescriptor_t poolOp,
                                 const miopenPoolingDescriptor_t poolDesc,
                                 const TensorDescriptor& xDesc,
                                 const TensorDescriptor& yDesc)
{

    MIOPEN_LOG_FUNCTION(poolOp, poolDesc, xDesc, yDesc);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenCreateOperatorArgs(miopenOperatorArgs_t *args){
    
    MIOPEN_LOG_FUNCTION(args);
    return (miopenStatusSuccess);
}


extern "C" miopenStatus_t miopenDestroyOperatorArgs(miopenOperatorArgs_t args){
    
    MIOPEN_LOG_FUNCTION(args);
    return (miopenStatusSuccess);
}


extern "C" miopenStatus_t miopenSetOpArgsConvInference(miopenOperatorArgs_t args,
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
                                                            const double epsilon)
{

    MIOPEN_LOG_FUNCTION(args, 
                        bnOp,
                        bnScale,
                        bnBias,
                        estimatedMean,
                        estimatedVariance,
                        epsilon);
    return (miopenStatusSuccess);
}


// This is essentially a noop, but it forces the users to have a matching length arg array in
// execute
// Potentially should be removed.
extern "C" miopenStatus_t
miopenSetOpArgsActivationInference(miopenOperatorArgs_t args,
                                   const miopenOperatorDescriptor_t activOp)
{

    MIOPEN_LOG_FUNCTION(args, activOp);
    return (miopenStatusSuccess);
}

// This is essentially a noop, but it forces the users to have a matching length arg array in
// execute
// Potentially should be removed.
extern "C" miopenStatus_t miopenSetOpArgsPoolingInference(miopenOperatorArgs_t args,
                                                          const miopenOperatorDescriptor_t poolingOp
                                                          )
{

    MIOPEN_LOG_FUNCTION(args, poolingOp);
    return (miopenStatusSuccess);
}


extern "C" miopenStatus_t miopenSetOpArgsTensorOp(miopenOperatorArgs_t args,
                                                    const miopenOperatorDescriptor_t tOp,
                                                    const void* alpha1,
                                                    const void* alpha2,
                                                    const void* beta,
                                                    const void* B)
{

    MIOPEN_LOG_FUNCTION(args, 
                        tOp,
                        alpha1, 
                        alpha2,
                        beta,
                        B);
    return (miopenStatusSuccess);
}


// Return an error code that is "NotImplemented", if it exists then return success
// This function should:
//	   Associate the data structure pointers to the user input and do checks
//     This function allows the user to reuse fusion plan.
// The number of operator descriptors must equal the number of args and both should match
// the value stored in the fused plan descriptor.
extern "C" miopenStatus_t
miopenExecuteFusionPlan(const miopenHandle_t handle,
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
                                  openOperatorArgs_t args
                                  miopenOpCost_t* opCost)
{
    MIOPEN_LOG_FUNCTION(handle, fusePlanDesc, workSpaceSize, workSpace, args, opCost);
    return (miopenStatusSuccess);
}



extern "C" miopenStatus_t miopenDestroyFusionPlanDescriptor(miopenFusionPlanDescriptor_t fusePlanDesc)
{

    MIOPEN_LOG_FUNCTION(fusePlanDesc)
    return (miopenStatusSuccess);
}



extern "C" miopenStatus_t miopenDestroyOperator(miopenOperatorDescriptor_t miopenOp)
{

    MIOPEN_LOG_FUNCTION(miopenOp)
    return (miopenStatusSuccess);
}


extern "C" miopenStatus_t miopenConvBatchNormActivationInference(miopenHandle_t handle, 
                                                        const miopenConvolutionDescriptor_t convDesc,
                                                        const miopenTensorDescriptor_t xDesc,
                                                        const void* x,
                                                        const miopenTensorDescriptor_t wDesc,
                                                        const void* w,
                                                        miopenBatchNormMode_t bn_mode,
                                                        const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                                        void* bnScale,
                                                        void* bnBias,
                                                        void* estimatedMean,
                                                        void* estimatedVariance,
                                                        double epsilon
                                                        const miopenActivationDescriptor_t activDesc,
                                                        const miopenTensorDescriptor_t yDesc,
                                                        void* y);
{
      return (miopenStatusSuccess);
}


extern "C" miopenStatus_t miopenConvTensorOpActivationInference(miopenHandle_t handle, 
                                                        const miopenConvolutionDescriptor_t convDesc,
                                                        const miopenTensorDescriptor_t xDesc,
                                                        const void* x,
                                                        const miopenTensorDescriptor_t wDesc,
                                                        const void* w,
                                                        const void* alpha1,
                                                        const void* alpha2,
                                                        const miopenTensorDescriptor_t bDesc,
                                                        const void* B,
                                                        const void* beta,
                                                        const miopenActivationDescriptor_t activDesc,
                                                        const miopenTensorDescriptor_t yDesc,
                                                        void* y)
{
        return (miopenStatusSuccess);
}


extern "C" miopenStatus_t miopenConvActivationPoolingInference(miopenHandle_t handle, 
                                                        const miopenConvolutionDescriptor_t convDesc,
                                                        const miopenTensorDescriptor_t xDesc,
                                                        const void* x,
                                                        const miopenTensorDescriptor_t wDesc,
                                                        const void* w,
                                                        const miopenActivationDescriptor_t activDesc,
                                                        const miopenPoolingDescriptor_t poolDesc,
                                                        const miopenTensorDescriptor_t yDesc,
                                                        void* y)
{
        return (miopenStatusSuccess);
}
