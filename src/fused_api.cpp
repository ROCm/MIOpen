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


extern "C" miopenStatus_t miopenCreateConvOp(miopenOperatorDescriptor* convOp,
	                                        const miopenPipelineMode_t pipelineMode,
	                                        const miopenOp_t  opDesc,
	                                        const miopenConvolutionDescriptor_t convDesc,
	                                        bool exhaustiveSearch
)
{

    MIOPEN_LOG_FUNCTION(convOp, pipelineMode, opDesc, convDesc, exhaustiveSearch);
	return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenCreateActivationOp(miopenOperatorDescriptor* activOp,
	                                              const miopenPipelineMode_t pipelineMode,
	                                              const miopenOp_t  opDesc,
	                                              const miopenActivationDescriptor_t activDesc,
	                                              const void* alpha,
	                                              const void* beta
)
{

	MIOPEN_LOG_FUNCTION(activOp, pipelineMode, opDesc, activDesc, alpha, beta);
	return (miopenStatusSuccess);
}


extern "C" miopenStatus_t miopenCreateBatchNormalizationOp(miopenOperatorDescriptor* bnOp,
	                                                       const miopenPipelineMode_t pipelineMode,
	                                                       const miopenOp_t  opDesc,
	                                                       const miopenBatchNormMode_t bn_mode,
	                                                       const void* alpha,
	                                                       const void* beta,
	                                                       const void* bnScale,
	                                                       const void* bnBias,
	                                                       const void* estimatedMean,
	                                                       const void* estimatedVariance,
	                                                       double epsilon)
{

	MIOPEN_LOG_FUNCTION(bnOp, pipelineMode, opDesc, bn_mode, alpha, beta, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon);
	return (miopenStatusSuccess);
}

extern "C"  miopenStatus_t miopenCreatePoolingOp(miopenOperatorDescriptor* poolingOp,
	                                             const miopenPipelineMode_t pipelineMode,
	                                             const miopenOp_t  opDesc,
	                                             const miopenPoolingDescriptor_t poolDesc,
	                                             const void* alpha,
	                                             const void* beta
)
{

	MIOPEN_LOG_FUNCTION(poolingOp, pipelineMode, opDesc, poolDesc, alpha, beta);
	return (miopenStatusSuccess);
}

extern "C"  miopenStatus_t miopenCreateEltWizeOp(miopenOperatorDescriptor* eltWiseOp,
	                                             const miopenPipelineMode_t pipelineMode,
	                                             const miopenOp_t  opDesc,
	                                             const size_t n,
	                                             const char* op,
	                                             double ALPHA,
	                                             double BETA,
	                                             double GAMMA
)
{

	MIOPEN_LOG_FUNCTION(eltWiseOp, pipelineMode, opDesc, n, op, ALPHA, BETA, GAMMA);
	return (miopenStatusSuccess);
}


extern "C" miopenStatus_t miopenGetOperatorType(const miopenOperatorDescriptor miopenOp,
	miopenOperator_t* opType
)
{
	MIOPEN_LOG_FUNCTION(miopenOp, opType);
	return (miopenStatusSuccess);
}


extern "C" miopenStatus_t miopenGetNRealizations(const miopenOperatorDescriptor miopenOp,
	size_t* nReal
)
{
	MIOPEN_LOG_FUNCTION(miopenOp, nReal);
	return (miopenStatusSuccess);
}

extern "C"  miopenStatus_t miopenGetRealizations(const miopenOperatorDescriptor miopenOp,
	                                             const size_t nReal,
	                                             miopenOpRealization_t * opReal,
	                                             size_t * nRet
)
{
	MIOPEN_LOG_FUNCTION(miopenOp, nReal, opReal, nRet);
	return (miopenStatusSuccess);
}

extern "C"   miopenStatus_t miopenGetFusionPlanCost(const miopenHandle_t handle,
	                                                const size_t nOps,
	                                                const miopenOpRealization_t * opReal,
	                                                miopenOpCost_t * opCost
)
{
	MIOPEN_LOG_FUNCTION(miopenOp, nOps, opReal, opCost);
	return (miopenStatusSuccess);
}

extern "C" miopenStatus_t miopenCreateFusionPlan(const miopenHandle_t handle,
	                                             const miopenFusionPlanDescriptor fusePlanDescr,
	                                             const size_t nOps,
	                                             const miopenOpRealization_t * opReal
)
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
	                                              const void ** weights,
	                                              const void* workSpace,
	                                              size_t workSpaceSize
)
{

	MIOPEN_LOG_FUNCTION(handle, fusePlanDescr, n_src, src, n_dst, dst, n_weights, weights, workSpace, workSpaceSize);
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
