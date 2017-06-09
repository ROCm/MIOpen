#ifndef GUARD_MIOPEN_BATCHNORMALIZATION_HPP_
#define GUARD_MIOPEN_BATCHNORMALIZATION_HPP_

#include <chrono>
#include <cmath>
#include <miopen/miopen.h>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>
#include <miopen/common.hpp>
#include <miopen/mlo_internal.hpp>

#define MIO_BN_CPP_PROF         0
#define MIOPEN_BN_CPP_DEBUG     1
#define MIO_BN_STATIC_WGSIZE    256

#define MIO_BN_TIME_EVERYTHING  0

namespace miopen {

    
void DeriveBNTensorDescriptor(
                TensorDescriptor& derivedBnDesc, 
                const TensorDescriptor& xDesc, 
                miopenBatchNormMode_t bn_mode);

void BatchNormForwardInference(
		Handle&			handle,
		miopenBatchNormMode_t	bn_mode,
		const void		*alpha,
		const void              *beta,
		const TensorDescriptor&	xDesc,
		ConstData_t		x,
		const TensorDescriptor& yDesc,
		Data_t                  y,
		const TensorDescriptor& bnScaleBiasMeanVarDesc,
		ConstData_t		bnScale,
		ConstData_t		bnBias,
		ConstData_t     	estimatedMean,
		ConstData_t		estimatedVariance,
		double			epsilon);



void BatchNormForwardTraining(
		Handle& 		handle,
		miopenBatchNormMode_t	bn_mode,
		const void 		*alpha, /* these don't seem to be used in conv */
		const void 		*beta,
		const TensorDescriptor& xDesc,
		ConstData_t		x,
		const TensorDescriptor& yDesc,
		Data_t			y,
		const TensorDescriptor& bnScaleBiasMeanVarDesc,
		ConstData_t		bnScale,
		ConstData_t		bnBias,
		double 			expAvgFactor,
		Data_t			resultRunningMean,
		Data_t			resultRunningVariance,
		double 			epsilon,
		Data_t 			resultSaveMean,
		Data_t			resultSaveInvVariance);



void BatchNormBackward(
		Handle&			handle,
		miopenBatchNormMode_t	bn_mode,
		const void		*alphaDataDiff,
		const void		*betaDataDiff,
		const void		*alphaParamDiff,
		const void		*betaParamDiff,
		const TensorDescriptor&	xDesc,
		ConstData_t		x,
		const TensorDescriptor&	dyDesc,
		ConstData_t		dy,
		const TensorDescriptor&	dxDesc,
		Data_t			dx,
		const TensorDescriptor&	bnScaleBiasDiffDesc,
		ConstData_t		bnScale,
		Data_t			resultBnScaleDiff,
		Data_t			resultBnBiasDiff,
		double			epsilon,
		ConstData_t		savedMean,
		ConstData_t		savedInvVariance);

}  // namespace miopen


#endif // GUARD_MIOPEN_BATCHNORMALIZATION_HPP_
