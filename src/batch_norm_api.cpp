#include <miopen/batch_norm.hpp>
#include <miopen/errors.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/logger.hpp>
#include <initializer_list>
#include <array>


extern "C"
miopenStatus_t miopenDeriveBNTensorDescriptor(miopenTensorDescriptor_t derivedBnDesc, const miopenTensorDescriptor_t xDesc, miopenBatchNormMode_t bn_mode){

    MIOPEN_LOG_FUNCTION(derivedBnDesc, xDesc, bn_mode);
    return miopen::try_([&] {    
        DeriveBNTensorDescriptor(miopen::deref(derivedBnDesc),miopen::deref(xDesc),bn_mode);
    });
}
    

extern "C"
miopenStatus_t miopenBatchNormalizationForwardInference(
					miopenHandle_t			handle,
					miopenBatchNormMode_t		bn_mode,
					void				*alpha,
					void				*beta,
					const miopenTensorDescriptor_t	xDesc,
					const void			*x,
					const miopenTensorDescriptor_t	yDesc,
					void				*y,
					const miopenTensorDescriptor_t	bnScaleBiasMeanVarDesc,
					void				*bnScale,
					void				*bnBias,
					void				*estimatedMean,
					void				*estimatedVariance,
					double				epsilon){
    MIOPEN_LOG_FUNCTION(bn_mode, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon);
    return miopen::try_([&] {
                        miopen::BatchNormForwardInference(
                                        miopen::deref(handle),
                                        bn_mode,
                                        alpha,
                                        beta,
                                        miopen::deref(xDesc),
                                        DataCast(x),
                                        miopen::deref(yDesc),
                                        DataCast(y),
                                        miopen::deref(bnScaleBiasMeanVarDesc),
                                        DataCast(bnScale),
                                        DataCast(bnBias),
                                        DataCast(estimatedMean),
                                        DataCast(estimatedVariance),
                                        epsilon);
    });
}



extern "C"
miopenStatus_t miopenBatchNormalizationForwardTraining(
					miopenHandle_t			handle,
					miopenBatchNormMode_t		bn_mode,
					void				*alpha,
					void				*beta,
					const miopenTensorDescriptor_t	xDesc,
					const void			*x,
					const miopenTensorDescriptor_t	yDesc,
					void				*y,
					const miopenTensorDescriptor_t	bnScaleBiasMeanVarDesc,
					void				*bnScale,
					void				*bnBias,
					double				expAvgFactor,
					void				*resultRunningMean,
					void				*resultRunningVariance,
					double				epsilon,
					void				*resultSaveMean,
					void				*resultSaveInvVariance){

    MIOPEN_LOG_FUNCTION(bn_mode, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, expAvgFactor, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance);
    return miopen::try_([&] {
                    miopen::BatchNormForwardTraining(
                                    miopen::deref(handle),
                                    bn_mode,
                                    alpha,
                                    beta,
                                    miopen::deref(xDesc),
                                    DataCast(x),
                                    miopen::deref(yDesc),
                                    DataCast(y),
                                    miopen::deref(bnScaleBiasMeanVarDesc),
                                    DataCast(bnScale),
                                    DataCast(bnBias),
                                    expAvgFactor,
                                    DataCast(resultRunningMean),
                                    DataCast(resultRunningVariance),
                                    epsilon,
                                    DataCast(resultSaveMean),
                                    DataCast(resultSaveInvVariance));
    });
}


extern "C"
miopenStatus_t miopenBatchNormalizationBackward(
			miopenHandle_t			handle,
			miopenBatchNormMode_t           bn_mode,
			const void                      *alphaDataDiff,
			const void                      *betaDataDiff,
			const void                      *alphaParamDiff,
			const void                      *betaParamDiff,
			const miopenTensorDescriptor_t	xDesc,
			const void			*x,
			const miopenTensorDescriptor_t	dyDesc,
			const void			*dy,
			const miopenTensorDescriptor_t	dxDesc,
			void				*dx,
			const miopenTensorDescriptor_t	bnScaleBiasDiffDesc,
			const void			*bnScale,
			void                            *resultBnScaleDiff,
			void                            *resultBnBiasDiff,
			double                          epsilon,
			const void                      *savedMean,
			const void                      *savedInvVariance){

    MIOPEN_LOG_FUNCTION(bn_mode, xDesc, x, dyDesc, dy, dxDesc, dx, bnScaleBiasDiffDesc, bnScale, resultBnScaleDiff, resultBnBiasDiff, epsilon, savedMean, savedInvVariance);
    return miopen::try_([&] {
            miopen::BatchNormBackward(
                                    miopen::deref(handle),
                                    bn_mode,
                                    alphaDataDiff,
                                    betaDataDiff,
                                    alphaParamDiff,
                                    betaParamDiff,
                                    miopen::deref(xDesc),
                                    DataCast(x),
                                    miopen::deref(dyDesc),
                                    DataCast(dy),
                                    miopen::deref(dxDesc),
                                    DataCast(dx),
                                    miopen::deref(bnScaleBiasDiffDesc),
                                    DataCast(bnScale),
                                    DataCast(resultBnScaleDiff),
                                    DataCast(resultBnBiasDiff),
                                    epsilon,
                                    DataCast(savedMean),
                                    DataCast(savedInvVariance));
    });
}








