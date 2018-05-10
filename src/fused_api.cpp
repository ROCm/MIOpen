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
extern "C" miopenStatus_t miopenCheckFusionPlans(miopenFusionPlanDescriptor_t* fusePlanDesc,
                                                 miopenOperator_t* arrayOfOperators,
                                                 const size_t numOps,
                                                 const miopenPipelineMode_t pipelineMode,
                                                 const miopenFusionDirection_t fuseDirection);
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc);
    return (miopenStatusSuccess);
}

// Return an error code that is "NotImplemented", if it exists then return success
// This function should:
//		set up the place descriptor with expected input and ouput edges.
// 		Set up the internal datastructures for the fused kernel.
extern "C" miopenStatus_t
miopenSetFusionPlanDescriptor(miopenFusionPlanDescriptor_t* fusePlanDesc,
                              miopenOperatorDescriptor_t* arrayOfOperators,
                              const size_t numOps,
                              const miopenPipelineMode_t pipelineMode,
                              const miopenFusionDirection_t fuseDirection);
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc);
    return (miopenStatusSuccess);
}

// Datatype is set in operator, in the future we may want to have
extern "C" miopenStatus_t
miopenCreateOp(miopenOperatorDescriptor_t* Op, const miopenOperator_t operator)
{

    MIOPEN_LOG_FUNCTION(Op, dataType);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenConfigConvInferenceOp(miopenOperatorDescriptor_t* convOp,
                                                      miopenConvolutionDescriptor_t convDesc,
                                                      miopenConvFwdAlgorithm_t algo,
                                                      const TensorDescriptor& xDesc,
                                                      const TensorDescriptor& wDesc,
                                                      const TensorDescriptor& yDesc, )
{
    MIOPEN_LOG_FUNCTION(convOp, convDesc, xDesc, wDesc, yDesc);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t
miopenConfigActivationInferenceOp(miopenOperatorDescriptor_t* activOp,
                                  const miopenActivationDescriptor_t activDesc,
                                  const TensorDescriptor& xDesc,
                                  const TensorDescriptor& yDesc)
{
    MIOPEN_LOG_FUNCTION(activOp, activDesc, xDesc, yDesc);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t
miopenConfigBatchNormInferenceOp(miopenOperatorDescriptor_t* bnOp,
                                 const TensorDescriptor& xDesc,
                                 const TensorDescriptor& yDesc,
                                 const TensorDescriptor& bnScaleBiasMeanVarDesc)
{

    MIOPEN_LOG_FUNCTION(bnOp, bn_mode, xDesc, yDesc, bnScaleBiasMeanVarDesc);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenSetArgsConvInferenceOp(const miopenOperatorDescriptor_t convOp,
                                                       const void* w)
{
    MIOPEN_LOG_FUNCTION(convOp, w);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenSetArgsBatchNormInferenceOp(const miopenOperatorDescriptor_t bnOp,
                                                            miopenOperatorArgs_t* bnArgs,
                                                            const void* bnScale,
                                                            const void* bnBias,
                                                            const void* estimatedMean,
                                                            const void* estimatedVariance,
                                                            const double epsilon)
{

    MIOPEN_LOG_FUNCTION(bnOp,
                        dataType,
                        bn_mode,
                        alpha,
                        beta,
                        bnScale,
                        bnBias,
                        estimatedMean,
                        estimatedVariance,
                        epsilon);
    return (miopenStatusSuccess);
}

// This is essentially a noop, but it forces the users to have a matching length arg array in
// execute
extern "C" miopenStatus_t
miopenSetArgsActivationInferenceOp(const miopenOperatorDescriptor_t activOp,
                                   miopenOperatorArgs_t* actArgs);
{

    MIOPEN_LOG_FUNCTION(bnOp,
                        dataType,
                        bn_mode,
                        alpha,
                        beta,
                        bnScale,
                        bnBias,
                        estimatedMean,
                        estimatedVariance,
                        epsilon);
    return (miopenStatusSuccess);
}

// This is essentially a noop, but it forces the users to have a matching length arg array in
// execute
extern "C" miopenStatus_t miopenSetArgsPoolingInferenceOp(miopenOperatorDescriptor_t* poolingOp,
                                                          const miopenPoolingDescriptor_t poolDesc)
{

    MIOPEN_LOG_FUNCTION(poolingOp, poolDesc, alpha, beta);
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
                        const miopenOperatorDescriptor_t* arrayOfOperatorDesc,
                        const miopenOperatorArgs_t* arrayOfArgs)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc, arrayOfOperatorDesc, arrayOfArgs);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenGetOperatorType(const miopenOperatorDescriptor miopenOp,
                                                miopenOperator_t* opType)
{
    MIOPEN_LOG_FUNCTION(miopenOp, opType);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t
miopenGetFusionPlanCostEstimate(const miopenHandle_t handle,
                                const miopenFusionPlanDescriptor_t fusePlanDesc,
                                const miopenOperatorDescriptor_t* arrayOfOperatorDesc,
                                miopenOpCost_t* opCost)
{
    MIOPEN_LOG_FUNCTION(miopenOp, nOps, opReal, opCost);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenCreateFusionPlan(const miopenHandle_t handle,
                                                 const miopenFusionPlanDescriptor fusePlanDescr,
                                                 const size_t nOps,
                                                 const miopenOpRealization_t* opReal)
{

    MIOPEN_LOG_FUNCTION(handle, fusePlanDescr, nOps, opReal);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenExecuteFusionPlan(const miopenHandle_t handle,
                                                  const miopenFusionPlanDescriptor fusePlanDescr,
                                                  size_t n_src,
                                                  const void** src,
                                                  size_t n_dst,
                                                  const void** dst,
                                                  size_t n_weights,
                                                  const void** weights,
                                                  const void* workSpace,
                                                  size_t workSpaceSize)
{

    MIOPEN_LOG_FUNCTION(handle,
                        fusePlanDescr,
                        n_src,
                        src,
                        n_dst,
                        dst,
                        n_weights,
                        weights,
                        workSpace,
                        workSpaceSize);
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenDestroyFusionPlanDescriptor(miopenFusionPlanDescriptor fusePlanDesc)
{

    MIOPEN_LOG_FUNCTION(fusePlanDesc)
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenDestroyOperator(const miopenOperatorDescriptor miopenOp)
{

    MIOPEN_LOG_FUNCTION(miopenOp)
    //    return miopen::try_([&] { miopen_destroy_object(activDesc); });
    return (miopenStatusSuccess);
}
