#ifndef MIOPEN_GUARD_MIOPEN_H_
#define MIOPEN_GUARD_MIOPEN_H_

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wextern-c-compat"
#endif

#include <stddef.h>

#include "miopen_export.h"

#if MIOPEN_BACKEND_OPENCL
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#elif MIOPEN_BACKEND_HIP
#include <hip/hip_runtime_api.h>
#endif

#define MIOPEN_DECLARE_OBJECT(name) \
struct name {}; \
typedef struct name * name ## _t;

#ifdef __cplusplus
extern "C" {
#endif

#if MIOPEN_BACKEND_OPENCL
typedef cl_command_queue miopenAcceleratorQueue_t;
#elif MIOPEN_BACKEND_HIP
typedef hipStream_t miopenAcceleratorQueue_t;
#endif

MIOPEN_DECLARE_OBJECT(miopenHandle);

typedef enum {
	miopenStatusSuccess = 0,
	miopenStatusNotInitialized = 1,
	miopenStatusInvalidValue = 2,
	miopenStatusBadParm = 3,
	miopenStatusAllocFailed = 4,
	miopenStatusInternalError = 5,
	miopenStatusNotImplemented = 6,
	miopenStatusUnknownError = 7,
} miopenStatus_t;

MIOPEN_EXPORT miopenStatus_t miopenCreate(miopenHandle_t *handle);

MIOPEN_EXPORT miopenStatus_t miopenCreateWithStream(miopenHandle_t *handle,
		miopenAcceleratorQueue_t				stream);

MIOPEN_EXPORT miopenStatus_t miopenDestroy(miopenHandle_t handle);

MIOPEN_EXPORT miopenStatus_t miopenSetStream(miopenHandle_t handle,
        miopenAcceleratorQueue_t            streamID);

MIOPEN_EXPORT miopenStatus_t miopenGetStream(miopenHandle_t handle,
		miopenAcceleratorQueue_t				*streamID);

// Get time for last kernel launched
MIOPEN_EXPORT miopenStatus_t miopenGetKernelTime(miopenHandle_t handle, float* time);
// Enable profiling to retrieve time
MIOPEN_EXPORT miopenStatus_t miopenEnableProfiling(miopenHandle_t handle, bool enable);


MIOPEN_DECLARE_OBJECT(miopenTensorDescriptor);
MIOPEN_DECLARE_OBJECT(miopenConvolutionDescriptor);
MIOPEN_DECLARE_OBJECT(miopenPoolingDescriptor);
MIOPEN_DECLARE_OBJECT(miopenLRNDescriptor);
MIOPEN_DECLARE_OBJECT(miopenActivationDescriptor);

//typedef struct miopenPoolingDescriptor *miopenPoolingDescriptor_t;

//typedef struct miopenLRNDescriptor *miopenLRNDescriptor_t;

typedef enum {
	miopenHalf = 0,
	miopenFloat = 1,
} miopenDataType_t;

typedef enum {
	miopenOpTensorAdd = 0,
	miopenOpTensorMul = 1,
	miopenTensorMin = 2,
	miopenTensorMax = 3,
} miopenTensorOp_t;

typedef enum {
	miopenConvolution = 0,
	miopenCrossCorrelation = 1,
} miopenConvolutionMode_t;

typedef enum {
	miopenPoolingMax = 0,
	miopenPoolingAverage = 1,
} miopenPoolingMode_t;

typedef enum {
	miopenLRNWithinChannel = 0,
	miopenLRNCrossChannel = 1,
} miopenLRNMode_t;

typedef enum {
        miopenBNPerActivation = 0,
        miopenBNSpatial       = 1,
}miopenBatchNormMode_t;


typedef enum {
	miopenActivationPATHTRU		= 0,
	miopenActivationLOGISTIC	= 1,	//	1 / (1 + e^-x)	//Sigmoid
	miopenActivationTANH		= 2,	//	a * tanh( b * x)
	miopenActivationRELU		= 3,	//	max(0, x)
	miopenActivationSOFTRELU	= 4,	//	log(1 + e^x)   // bonomial normal log likelihood
	miopenActivationABS			= 5, //	abs(x)
	miopenActivationPOWER		= 6, // (a + b * x ) ^power
//	miopenActivationBRELU		= 7, //	min(a, max(0, x))
//	miopenActivationSQUARE		= 8,//	x^2
//	miopenActivationSQR			= 9,//	sqr(x)
//	miopenActivationLINEAR		= 10,//	a + b * x
} miopenActivationMode_t;
// Create a Tensor Descriptor
MIOPEN_EXPORT miopenStatus_t miopenCreateTensorDescriptor(miopenTensorDescriptor_t *tensorDesc);

// Only supporting NCHW for now and merging both expert and regular cuDNN APIs
MIOPEN_EXPORT miopenStatus_t miopenSet4dTensorDescriptor(
		miopenTensorDescriptor_t	tensorDesc,
		miopenDataType_t			dataType, // half/float/double
		int							n,
		int							c,
		int							h,
		int							w);

// Get the details of the tensor desciptor
MIOPEN_EXPORT miopenStatus_t miopenGet4dTensorDescriptor(
		miopenTensorDescriptor_t	tensorDesc,
		miopenDataType_t			*dataType,
		int							*n,
		int							*c, 
		int							*h,
		int							*w,
		int							*nStride,
		int							*cStride,
		int							*hStride,
		int							*wStride);

// Not sure if the following two APIs are required right now
MIOPEN_EXPORT miopenStatus_t miopenSetTensorDescriptor(
		miopenTensorDescriptor_t	tensorDesc,
		miopenDataType_t			dataType,
		int							nbDims,
		int							*dimsA,
		int							*stridesA);

MIOPEN_EXPORT miopenStatus_t miopenGetTensorDescriptorSize(miopenTensorDescriptor_t tensorDesc, int* size);

// Get the details of the n-dimensional tensor desciptor
MIOPEN_EXPORT miopenStatus_t miopenGetTensorDescriptor(
		miopenTensorDescriptor_t	tensorDesc,
		miopenDataType_t			*dataType,
		int							*dimsA,
		int							*stridesA);
		
MIOPEN_EXPORT miopenStatus_t miopenDestroyTensorDescriptor(miopenTensorDescriptor_t tensorDesc);

/* This function implements the equation C = op ( alpha1[0] * A, alpha2[0] * B
 * ) + beta[0] * C, given tensors A, B, and C and scaling factors alpha1,
 * alpha2, and beta. The op to use is indicated by the descriptor opTensorDesc.
 * Currently-supported ops are listed by the miopenOpTensorDescriptor_t enum.
 */
MIOPEN_EXPORT miopenStatus_t miopenOpTensor(miopenHandle_t handle,
		miopenTensorOp_t				tensorOp,
		const void						*alpha1,
		const miopenTensorDescriptor_t	aDesc,
		const void						*A,
		const void						*alpha2,
		const miopenTensorDescriptor_t	bDesc,
		const void						*B,
		const void						*beta,
		const miopenTensorDescriptor_t	cDesc,
		void							*C);

/* Pointer in Host memory to a single value. All elements of the y tensor will
 * be set to value[0]. The data type of the element in value[0] has to match
 * the data type of tensor y.
 */
MIOPEN_EXPORT miopenStatus_t miopenSetTensor(miopenHandle_t                 handle,
		const miopenTensorDescriptor_t yDesc,
		void                          *y,
		const void                    *alpha );

MIOPEN_EXPORT miopenStatus_t miopenScaleTensor(miopenHandle_t                 handle,
		const miopenTensorDescriptor_t yDesc,
		void                          *y,
		const void                    *alpha );

MIOPEN_EXPORT miopenStatus_t miopenCreateConvolutionDescriptor(
		miopenConvolutionDescriptor_t *convDesc);

MIOPEN_EXPORT miopenStatus_t miopenInitConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc,
		miopenConvolutionMode_t mode,
		int pad_h,
		int pad_w,
		int u,
		int v,
		int upscalex,
		int upscaley);

MIOPEN_EXPORT miopenStatus_t miopenGetConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc,
		miopenConvolutionMode_t *mode,
		int *pad_h,
		int *pad_w,
		int *u,
		int *v,
		int *upscalex,
		int *upscaley);

/* This function returns the dimensions of the resulting 4D tensor of a 2D
 * convolution, given the convolution descriptor, the input tensor descriptor
 * and the filter descriptor This function can help to setup the output tensor
 * and allocate the proper amount of memory prior to launch the actual
 * convolution.
 */
MIOPEN_EXPORT miopenStatus_t miopenGetConvolutionForwardOutputDim(miopenConvolutionDescriptor_t convDesc,
		const miopenTensorDescriptor_t		inputTensorDesc,
		const miopenTensorDescriptor_t		filterDesc,
		int									*n,
		int 								*c,
		int 								*h,
		int 								*w);
		

// TODO: Add APIs for N-dimensional filter descriptors. Tensorflow uses them.
//

MIOPEN_EXPORT miopenStatus_t miopenDestroyConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc);

typedef enum {
	miopenConvolutionFwdAlgoGEMM = 0,
	miopenConvolutionFwdAlgoDirect = 1,
	miopenConvolutionFwdAlgoFFT = 2,
	miopenConvolutionFwdAlgoWinograd = 3,
} miopenConvFwdAlgorithm_t;

typedef enum {
	miopenConvolutionBwdWeightsAlgoGEMM = 0,
	miopenConvolutionBwdWeightsAlgoDirect = 1,
} miopenConvBwdWeightsAlgorithm_t;

typedef enum {
	miopenConvolutionBwdDataAlgoDirect = 0,
	miopenConvolutionBwdDataAlgoWinograd = 1,
} miopenConvBwdDataAlgorithm_t;

// Same perf struct for forward, backward filter and backward
// data algorthms
typedef struct{
	union {
		miopenConvFwdAlgorithm_t fwd_algo;
		miopenConvBwdWeightsAlgorithm_t bwd_weights_algo;
		miopenConvBwdDataAlgorithm_t bwd_data_algo;
	};
	float time;
	size_t memory;
} miopenConvAlgoPerf_t;
/* This function attempts all MIOpen algorithms for miopenConvolutionForward(),
 * and outputs performance metrics to a user- allocated array of
 * miopenConvolutionFwdAlgoPerf_t. These metrics are written in sorted fashion
 * where the first element has the lowest compute time.
 */

MIOPEN_EXPORT miopenStatus_t miopenConvolutionForwardGetWorkSpaceSize(
        miopenHandle_t                      handle,
		const miopenTensorDescriptor_t		wDesc,
		const miopenTensorDescriptor_t		xDesc,
		const miopenConvolutionDescriptor_t convDesc,
		const miopenTensorDescriptor_t		yDesc,
		size_t								*workSpaceSize);

MIOPEN_EXPORT miopenStatus_t miopenFindConvolutionForwardAlgorithm(miopenHandle_t handle,
		const miopenTensorDescriptor_t		xDesc,
		const void							*x,
		const miopenTensorDescriptor_t		wDesc,
		const void							*w,
		const miopenConvolutionDescriptor_t	convDesc,
		const miopenTensorDescriptor_t		yDesc,
		void							*y,
		const int							requestAlgoCount,
		int									*returnedAlgoCount,
		miopenConvAlgoPerf_t				*perfResults,
		void								*workSpace,
		size_t								workSpaceSize,
		bool								exhaustiveSearch);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionForward(miopenHandle_t handle,
		const void							*alpha,
		const miopenTensorDescriptor_t		xDesc,
		const void							*x,
		const miopenTensorDescriptor_t		wDesc,
		const void							*w,
		const miopenConvolutionDescriptor_t convDesc,
		miopenConvFwdAlgorithm_t			algo,
		const void							*beta,
		const miopenTensorDescriptor_t		yDesc,
		void								*y,
		void								*workSpace,
		size_t								workSpaceSize);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionForwardBias(miopenHandle_t handle,
		const void						*alpha,
		const miopenTensorDescriptor_t	bDesc,
		const void						*b,
		const void						*beta,
		const miopenTensorDescriptor_t	yDesc,
		void							*y);

MIOPEN_EXPORT miopenStatus_t miopenFindConvolutionBackwardDataAlgorithm(miopenHandle_t handle,
		const miopenTensorDescriptor_t		dyDesc,
		const void							*dy,
		const miopenTensorDescriptor_t		wDesc,
		const void							*w,
		const miopenConvolutionDescriptor_t	convDesc,
		const miopenTensorDescriptor_t		dxDesc,
		const void							*dx,
		const int							requestAlgoCount,
		int									*returnedAlgoCount,
		miopenConvAlgoPerf_t				*perfResults,
		void								*workSpace,
		size_t								workSpaceSize,
		bool								exhaustiveSearch);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardData(miopenHandle_t handle,
		const void							*alpha,
		const miopenTensorDescriptor_t		dyDesc,
		const void							*dy,
		const miopenTensorDescriptor_t		wDesc,
		const void							*w,
		const miopenConvolutionDescriptor_t convDesc,
		miopenConvBwdDataAlgorithm_t		algo,
		const void							*beta,
		const miopenTensorDescriptor_t		dxDesc,
		void								*dx,
		void								*workSpace,
		size_t								workSpaceSize);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardDataGetWorkSpaceSize(
        miopenHandle_t                      handle,
		const miopenTensorDescriptor_t		dyDesc,
		const miopenTensorDescriptor_t		wDesc,
		const miopenConvolutionDescriptor_t convDesc,
		const miopenTensorDescriptor_t		dxDesc,
		size_t								*workSpaceSize);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardWeightsGetWorkSpaceSize(
        miopenHandle_t                      handle,
		const miopenTensorDescriptor_t		dyDesc,
		const miopenTensorDescriptor_t		xDesc,
		const miopenConvolutionDescriptor_t	convDesc,
		const miopenTensorDescriptor_t		dwDesc,
		size_t								*workSpaceSize);

MIOPEN_EXPORT miopenStatus_t miopenFindConvolutionBackwardWeightsAlgorithm(miopenHandle_t handle,
		const miopenTensorDescriptor_t		dyDesc,
		const void							*dy,
		const miopenTensorDescriptor_t		xDesc,
		const void							*x,
		const miopenConvolutionDescriptor_t	convDesc,
		const miopenTensorDescriptor_t		dwDesc,
		void							*dw,
		const int							requestAlgoCount,
		int									*returnedAlgoCount,
		miopenConvAlgoPerf_t				*perfResults,
		void								*workSpace,
		size_t								workSpaceSize,
		bool								exhaustiveSearch);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardWeights(miopenHandle_t handle,
		const void							*alpha,
		const miopenTensorDescriptor_t		dyDesc,
		const void							*dy,
		const miopenTensorDescriptor_t		xDesc,
		const void							*x,
		const miopenConvolutionDescriptor_t convDesc,
		miopenConvBwdWeightsAlgorithm_t		algo,
		const void							*beta,
		const miopenTensorDescriptor_t		dwDesc,
		void								*dw,
		void								*workSpace,
		size_t								workSpaceSize);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardBias(miopenHandle_t handle,
		const void						*alpha,
		const miopenTensorDescriptor_t		dyDesc,
		const void						*dy,
		const void						*beta,
		const miopenTensorDescriptor_t		dbDesc,
		void							*db);

// Pooling APIs

MIOPEN_EXPORT miopenStatus_t miopenCreatePoolingDescriptor(miopenPoolingDescriptor_t *poolDesc);

MIOPEN_EXPORT miopenStatus_t miopenSet2dPoolingDescriptor(
		miopenPoolingDescriptor_t			poolDesc,
		miopenPoolingMode_t					mode,
		int									windowHeight,
		int									windowWidth,
		int									pad_h,
		int									pad_w,
		int									u,
		int									v);
	
MIOPEN_EXPORT miopenStatus_t miopenGet2dPoolingDescriptor(
		const miopenPoolingDescriptor_t		poolDesc,
		miopenPoolingMode_t					*mode,
		int									*windowHeight,
		int									*windowWidth,
		int									*pad_h,
		int									*pad_w,
		int									*u,
		int									*v);

MIOPEN_EXPORT miopenStatus_t miopenSetNdPoolingDescriptor(
		miopenPoolingDescriptor_t			poolDesc,
		miopenPoolingMode_t					mode,
		int									nbDims,
		int									*windowDimA,
		int									*padA,
		int									*stridesA);

MIOPEN_EXPORT miopenStatus_t miopenGetNdPoolingDescriptor(
		const miopenPoolingDescriptor_t		poolDesc,
		miopenPoolingMode_t					*mode,
		int									*nbDims,
		int									*windowDimA,
		int									*padA,
		int									*stridesA);

MIOPEN_EXPORT miopenStatus_t miopenGetPoolingForwardOutputDim(
		const miopenPoolingDescriptor_t		poolDesc,
		const miopenTensorDescriptor_t		tensorDesc,
		int									*n,
		int									*c,
		int									*h,
		int									*w);

MIOPEN_EXPORT miopenStatus_t miopenPoolingGetWorkSpaceSize(
		const miopenTensorDescriptor_t		yDesc,
		size_t								*workSpaceSize);

MIOPEN_EXPORT miopenStatus_t miopenPoolingForward(
		miopenHandle_t						handle,
		const miopenPoolingDescriptor_t		poolDesc,
		const void							*alpha,
		const miopenTensorDescriptor_t		xDesc,
		const void							*x,
		const void							*beta,
		const miopenTensorDescriptor_t		yDesc,
		void								*y,
		bool                                do_backward,
		void								*workSpace,
		size_t								workSpaceSize);

MIOPEN_EXPORT miopenStatus_t miopenPoolingBackward(
		miopenHandle_t						handle,
		const miopenPoolingDescriptor_t		poolDesc,
		const void							*alpha,
		const miopenTensorDescriptor_t		yDesc,
		const void							*y,
		const miopenTensorDescriptor_t		dyDesc,
		const void							*dy,
		const miopenTensorDescriptor_t		xDesc,
		const void							*x,
		const void							*beta,
		const miopenTensorDescriptor_t		dxDesc,
		void								*dx,
		const void							*workSpace);


MIOPEN_EXPORT miopenStatus_t miopenDestroyPoolingDescriptor(miopenPoolingDescriptor_t poolDesc);

// LRN APIs

MIOPEN_EXPORT miopenStatus_t miopenCreateLRNDescriptor(miopenLRNDescriptor_t *lrnDesc);

MIOPEN_EXPORT miopenStatus_t miopenSetLRNDescriptor(
	const miopenLRNDescriptor_t			lrnDesc,
	miopenLRNMode_t						mode,
	unsigned int						lrnN,
	double								lrnAlpha,
	double								lrnBeta,
	double								lrnK);


MIOPEN_EXPORT miopenStatus_t miopenGetLRNDescriptor(
		const miopenLRNDescriptor_t			lrnDesc,
		miopenLRNMode_t						*mode,
		unsigned int						*lrnN,
		double								*lrnAlpha,
		double								*lrnBeta,
		double								*lrnK);

MIOPEN_EXPORT miopenStatus_t miopenLRNGetWorkSpaceSize(
		const miopenTensorDescriptor_t		yDesc,
		size_t								*workSpaceSize);

MIOPEN_EXPORT miopenStatus_t miopenLRNForward(
		miopenHandle_t						handle,
		const miopenLRNDescriptor_t			lrnDesc,
		const void							*alpha,
		const miopenTensorDescriptor_t		xDesc,
		const void							*x,
		const void							*beta,
		const miopenTensorDescriptor_t		yDesc,
		void								*y,
		bool                                do_backward,
		void								*workSpace);


MIOPEN_EXPORT miopenStatus_t miopenLRNBackward(
		miopenHandle_t						handle,
		const miopenLRNDescriptor_t			lrnDesc,
		const void							*alpha,
		const miopenTensorDescriptor_t		yDesc,
		const void							*y,
		const miopenTensorDescriptor_t		dyDesc,
		const void							*dy,
		const miopenTensorDescriptor_t		xDesc,
		const void							*x,
		const void							*beta,
		const miopenTensorDescriptor_t		dxDesc,
		void								*dx,
		const void							*workSpace);

MIOPEN_EXPORT miopenStatus_t miopenDestroyLRNDescriptor(miopenLRNDescriptor_t lrnDesc);



//BATCH NORMALIZATION APIs


MIOPEN_EXPORT miopenStatus_t miopenDeriveBNTensorDescriptor(miopenTensorDescriptor_t & derivedBnDesc,
                                        const miopenTensorDescriptor_t & xDesc,
                                        miopenBatchNormMode_t bn_mode);



MIOPEN_EXPORT miopenStatus_t miopenBatchNormalizationForwardTraining(
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
					double				exponentialAverageFactor,
					void				*resultRunningMean,
					void				*resultRunningVariance,
					double				epsilon,
					void				*resultSaveMean,
					void				*resultSaveInvVariance);



MIOPEN_EXPORT miopenStatus_t miopenBatchNormalizationForwardInference(
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
					double				epsilon);



MIOPEN_EXPORT miopenStatus_t miopenBatchNormalizationBackward(
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
					const void                      *savedInvVariance);




// Activation APIs

MIOPEN_EXPORT miopenStatus_t miopenCreateActivationDescriptor(miopenActivationDescriptor_t *activDesc);

MIOPEN_EXPORT miopenStatus_t miopenSetActivationDescriptor(
	const miopenActivationDescriptor_t	activDesc,
	miopenActivationMode_t				mode,
	double								activAlpha,
	double								activBeta,
	double								activPower);


MIOPEN_EXPORT miopenStatus_t miopenGetActivationDescriptor(
	const miopenActivationDescriptor_t	activDesc,
	miopenActivationMode_t				*mode,
	double								*activAlpha,
	double								*activBeta,
	double								*activPower);



MIOPEN_EXPORT miopenStatus_t miopenActivationForward(
	miopenHandle_t						handle,
	const miopenActivationDescriptor_t	activDesc,
	const void							*alpha,
	const miopenTensorDescriptor_t		xDesc,
	const void							*x,
	const void							*beta,
	const miopenTensorDescriptor_t		yDesc,
	void								*y,
	bool                                do_backward,
	void								*workSpace);


MIOPEN_EXPORT miopenStatus_t miopenActivationBackward(
	miopenHandle_t						handle,
	const miopenActivationDescriptor_t	activDesc,
	const void							*alpha,
	const miopenTensorDescriptor_t		yDesc,
	const void							*y,
	const miopenTensorDescriptor_t		dyDesc,
	const void							*dy,
	const miopenTensorDescriptor_t		xDesc,
	const void							*x,
	const void							*beta,
	const miopenTensorDescriptor_t		dxDesc,
	void								*dx,
	const void							*workSpace);

MIOPEN_EXPORT miopenStatus_t miopenDestroyActivationDescriptor(miopenActivationDescriptor_t activDesc);

// Softmax APIs

MIOPEN_EXPORT miopenStatus_t miopenSoftmaxForward(
	miopenHandle_t						handle,
	const void							*alpha,
	const miopenTensorDescriptor_t		xDesc,
	const void							*x,
	const void							*beta,
	const miopenTensorDescriptor_t		yDesc,
	void								*y);

MIOPEN_EXPORT miopenStatus_t miopenSoftmaxBackward(
	miopenHandle_t						handle,
	const void							*alpha,
	const miopenTensorDescriptor_t		yDesc,
	const void							*y,
	const miopenTensorDescriptor_t		dyDesc,
	const void							*dy,
	const void							*beta,
	const miopenTensorDescriptor_t		dxDesc,
	void								*dx);

// GEMM API

MIOPEN_EXPORT miopenStatus_t miopenGemm(
		miopenHandle_t			handle,
		bool					isDataColMajor,
		bool					transA, 
		bool					transB, 
		int M, int N, int K, 
		const void *alpha, 
		const void *A, int lda, 
		const void *B, int ldb, 
		const void *beta, 
		void *C, int ldc );

#ifdef __cplusplus
}
#endif

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif // _MIOPEN_H_

