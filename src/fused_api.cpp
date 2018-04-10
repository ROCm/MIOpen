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

extern "C" miopenStatus_t miopenCreateFusionPlanDescriptor(miopenFusionPlanDescriptor* fusePlanDesc)
{
    MIOPEN_LOG_FUNCTION(fusePlanDesc);
//    return miopen::try_([&] { miopen::deref(activDesc) = new miopen::ActivationDescriptor(); });
	return (miopenStatusSuccess);
}


extern "C" miopenStatus_t miopenCreateFusionDescriptor(miopenFusionDescriptor* fuseDesc)
{
	MIOPEN_LOG_FUNCTION(fuseDesc);
	return (miopenStatusSuccess);
}


extern "C" miopenStatus_t miopenGetOperatorType(const miopenOperatorDescriptor miopenOp,
	                                                    miopenOperator_t* opType)
{

    MIOPEN_LOG_FUNCTION(miopenOp, opType);
	return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenCreateConvForwardOp(miopenOperatorDescriptor* convOp,
	                                                const miopenTensorDescriptor_t wDesc,
	                                                bool immutableWeights,
	                                                const miopenTensorDescriptor_t bDesc,
	                                                const miopenConvolutionDescriptor_t convDesc,
	                                                bool exhaustiveSearch)
{

    MIOPEN_LOG_FUNCTION(convOp, wDesc, immutableWeights, bDesc, convDesc, exhaustiveSearch);
	return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenCreateActivationForwardOp(miopenOperatorDescriptor* activOp,
	                                                      const miopenActivationDescriptor_t activDesc,
	                                                      const void* alpha,
	                                                      const void* beta)
{

	MIOPEN_LOG_FUNCTION(activOp, activDesc, alpha, beta);
	return (miopenStatusSuccess);
}


extern "C" miopenStatus_t miopenCreateBatchNormalizationForwardInferenceOp(miopenOperatorDescriptor* bnOp,
	                                                                       miopenBatchNormMode_t bn_mode,
	                                                                       void* alpha,
	                                                                       void* beta,
	                                                                       const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
	                                                                       void* bnScale,
	                                                                       void* bnBias,
	                                                                       void* estimatedMean,
	                                                                       void* estimatedVariance,
	                                                                       double epsilon)
{

	MIOPEN_LOG_FUNCTION(bnOp, bn_mode, alpha, beta, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon);
	return (miopenStatusSuccess);
}

extern "C"  miopenStatus_t miopenCreatePoolingForwardInferenceOp(miopenOperatorDescriptor* poolingOp,
	                                                             const miopenPoolingDescriptor_t poolDesc,
	                                                             const void* alpha,
	                                                             const void* beta
                                                                 )
{

	MIOPEN_LOG_FUNCTION(poolingOp, poolDesc, alpha, beta);
	return (miopenStatusSuccess);
}

extern "C"  miopenStatus_t miopenCreateEltWizeOp(miopenOperatorDescriptor* eltWiseOp,
	                                             const size_t n,
	                                             const char* op,
	                                             double ALPHA,
	                                             double BETA,
	                                             double GAMMA)
{

	MIOPEN_LOG_FUNCTION(eltWiseOp, n, op, ALPHA, BETA, GAMMA);
	return (miopenStatusSuccess);
}


extern "C"  miopenStatus_t miopenSetFusionDescriptor(miopenFusionDescriptor fuseDescr,
	                                                 const miopenFuseMode_t mode,
	                                                 const size_t n,
	                                                 const miopenFusionDescriptor *vertFuse,
	                                                 const miopenOperatorDescriptor * ops)

{

	MIOPEN_LOG_FUNCTION(fuseDescr, mode, n, vertFuse, ops);
	return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenCreateFusionPlanForwardInference(const miopenHandle_t handle,
	                                                             miopenFusionPlanDescriptor fusePlanDescr,
	                                                             const miopenFusionDescriptor fuseDescr,
	                                                             const miopenTensorDescriptor_t srcDesc,
	                                                             const miopenTensorDescriptor_t dstDesc,
	                                                            size_t* workSpaceSize)
{

	MIOPEN_LOG_FUNCTION(handle, fusePlanDescr, fuseDescr, srcDesc, dstDesc, workSpaceSize);
	return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenExecuteFusionPlanForwardInference(const miopenHandle_t handle,
	                                                              const miopenFusionPlanDescriptor fusePlanDescr,
	                                                              const void* src,
	                                                              const void* dst,
	                                                              void* workSpace,
	                                                              size_t workSpaceSize,
	                                                              size_t n_weights,
	                                                              const void ** weights
                                                                  )
{

	MIOPEN_LOG_FUNCTION(handle, fusePlanDescr, src, dst, workSpace, workSpaceSize, n_weights, const void ** weights);
	return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenDestroyFusionPlanDescriptor(miopenFusionPlanDescriptor fusePlanDesc)
{

    MIOPEN_LOG_FUNCTION(fusePlanDesc)
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenDestroyFusionDescriptor(miopenFusionDescriptor fuseDesc)
{

	MIOPEN_LOG_FUNCTION(fuseDesc)
    return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenDestroyOperator(const miopenOperatorDescriptor miopenOp)
{

	MIOPEN_LOG_FUNCTION(miopenOp)
		//    return miopen::try_([&] { miopen_destroy_object(activDesc); });
	return (miopenStatusSuccess);
}
