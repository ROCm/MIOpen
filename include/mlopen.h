#ifndef _MLOPEN_H_
#define _MLOPEN_H_

#include <stddef.h>

#include "mlopen_export.h"

#if MLOPEN_BACKEND_OPENCL
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#elif MLOPEN_BACKEND_HIP
#include <hip_runtime.h>
#endif

#define MLOPEN_DECLARE_OBJECT(name) \
struct name {}; \
typedef struct name * name ## _t;

#ifdef __cplusplus
extern "C" {
#endif

#if MLOPEN_BACKEND_OPENCL
typedef cl_command_queue mlopenAcceleratorQueue_t;
#elif MLOPEN_BACKEND_HIP
typedef hipStream_t mlopenAcceleratorQueue_t;
#endif

MLOPEN_DECLARE_OBJECT(mlopenHandle);

typedef enum {
	mlopenStatusSuccess = 0,
	mlopenStatusNotInitialized = 1,
	mlopenStatusInvalidValue = 2,
	mlopenStatusBadParm = 3,
	mlopenStatusAllocFailed = 4,
	mlopenStatusInternalError = 5,
	mlopenStatusNotImplemented = 6,
	mlopenStatusUnknownError = 7,
} mlopenStatus_t;

// TODO: C does not really have default function arguments. Need to modify this
// later or is it OK to leave it like this?
MLOPEN_EXPORT mlopenStatus_t mlopenCreate(mlopenHandle_t *handle,
		int							numStreams = 0,
		mlopenAcceleratorQueue_t				*stream = NULL);

MLOPEN_EXPORT mlopenStatus_t mlopenDestroy(mlopenHandle_t handle);

// Returns numStream'th stream for that particular handle
MLOPEN_EXPORT mlopenStatus_t mlopenGetStream(mlopenHandle_t handle,
		mlopenAcceleratorQueue_t				*streamId,
		int							numStream = 0);

// Get time for last kernel launched
MLOPEN_EXPORT mlopenStatus_t mlopenGetKernelTime(mlopenHandle_t handle, float* time);
// Enable profiling to retrieve time
MLOPEN_EXPORT mlopenStatus_t mlopenEnableProfiling(mlopenHandle_t handle, bool enable);


MLOPEN_DECLARE_OBJECT(mlopenTensorDescriptor);
MLOPEN_DECLARE_OBJECT(mlopenConvolutionDescriptor);
MLOPEN_DECLARE_OBJECT(mlopenPoolingDescriptor);
MLOPEN_DECLARE_OBJECT(mlopenLRNDescriptor);

//typedef struct mlopenPoolingDescriptor *mlopenPoolingDescriptor_t;

typedef struct mlopenLRNDescriptor *mlopenLRNDescriptor_t;

typedef enum {
	mlopenHalf = 0,
	mlopenFloat = 1,
	mlopenDouble = 2,
} mlopenDataType_t;

typedef enum {
	mlopenOpTensorAdd = 0,
	mlopenOpTensorMul = 1,
	mlopenTensorMin = 2,
	mlopenTensorMax = 3,
} mlopenTensorOp_t;

typedef enum {
	mlopenConvolution = 0,
	mlopenCrossCorrelation = 1,
} mlopenConvolutionMode_t;

typedef enum {
	mlopenPoolingMax = 0,
	mlopenPoolingAverage = 1,
} mlopenPoolingMode_t;

typedef enum {
	mlopenLRNWithinChannel = 0,
	mlopenLRNCrossChannel = 1,
} mlopenLRNMode_t;

// Create a Tensor Descriptor
MLOPEN_EXPORT mlopenStatus_t mlopenCreateTensorDescriptor(mlopenTensorDescriptor_t *tensorDesc);

// Only supporting NCHW for now and merging both expert and regular cuDNN APIs
MLOPEN_EXPORT mlopenStatus_t mlopenSet4dTensorDescriptor(
		mlopenTensorDescriptor_t	tensorDesc,
		mlopenDataType_t			datatype, // half/float/double
		int							n,
		int							c,
		int							h,
		int							w);

// Get the details of the tensor desciptor
MLOPEN_EXPORT mlopenStatus_t mlopenGet4dTensorDescriptor(
		mlopenTensorDescriptor_t	tensorDesc,
		mlopenDataType_t			*datatype,
		int							*n,
		int							*c, 
		int							*h,
		int							*w,
		int							*nStride,
		int							*cStride,
		int							*hStride,
		int							*wStride);

// Not sure if the following two APIs are required right now
MLOPEN_EXPORT mlopenStatus_t mlopenSetTensorDescriptor(
		mlopenTensorDescriptor_t	tensorDesc,
		mlopenDataType_t			datatype,
		int							nbDims,
		int							*dimA,
		int							*strideA);

MLOPEN_EXPORT mlopenStatus_t mlopenGetTensorDescriptorSize(mlopenTensorDescriptor_t tensorDesc, int* size);

// Get the details of the n-dimensional tensor desciptor
MLOPEN_EXPORT mlopenStatus_t mlopenGetTensorDescriptor(
		mlopenTensorDescriptor_t	tensorDesc,
		mlopenDataType_t			*datatype,
		int							*dimA,
		int							*strideA);
		
MLOPEN_EXPORT mlopenStatus_t mlopenDestroyTensorDescriptor(mlopenTensorDescriptor_t tensorDesc);

/* This function copies the scaled data from one tensor to another
 * tensor with a different layout. Those descriptors need to have the
 * same dimensions but not necessarily the same strides. The input
 * and output tensors must not overlap in any way (i.e., tensors
 * cannot be transformed in place). This function can be used
 * to convert a tensor with an unsupported format to a supported one.
 *
 * [MD]: Can this routine also suffice the requirements of AddTensor() routine? Both are doing the same thing -
 * dstValue = alpha*srcValue + beta*dstValue;
 */
MLOPEN_EXPORT mlopenStatus_t mlopenTransformTensor(mlopenHandle_t handle,
		const void						*alpha,
		const mlopenTensorDescriptor_t	xDesc,
		const void						*x,
		const void						*beta,
		const mlopenTensorDescriptor_t	 yDesc,
		void							*y);

#if 0
/* This function adds the scaled values of a bias tensor to another tensor.
 * Each dimension of the bias tensor A must match the corresponding dimension
 * of the destination tensor C or must be equal to 1. In the latter case, the
 * same value from the bias tensor for those dimensions will be used to blend
 * into the C tensor.
 *
 * [MD]: See above; may be just TransformTensor is sufficient
 */
mlopenStatus_t mlopenAddTensor(mlopenHandle_t handle,
		const void						*alpha,
		const mlopenTensorDescriptor_t	aDesc,
		const void						*A,
		const void						*beta,
		const mlopenTensorDescriptor_t	 cDesc,
		void							*C);
#endif // AddTensor API

/* This function implements the equation C = op ( alpha1[0] * A, alpha2[0] * B
 * ) + beta[0] * C, given tensors A, B, and C and scaling factors alpha1,
 * alpha2, and beta. The op to use is indicated by the descriptor opTensorDesc.
 * Currently-supported ops are listed by the mlopenOpTensorDescriptor_t enum.
 *
 * [MD]: Not sure if OpTensorDescriptor_t is required?
 */
MLOPEN_EXPORT mlopenStatus_t mlopenOpTensor(mlopenHandle_t handle,
		//const mlopenOpTensorDescriptor_t opTensorDesc,
		mlopenTensorOp_t				tensorOp,
		const void						*alpha1,
		const mlopenTensorDescriptor_t	aDesc,
		const void						*A,
		const void						*alpha2,
		const mlopenTensorDescriptor_t	bDesc,
		const void						*B,
		const void						*beta,
		const mlopenTensorDescriptor_t	cDesc,
		void							*C);

/* Pointer in Host memory to a single value. All elements of the y tensor will
 * be set to value[0]. The data type of the element in value[0] has to match
 * the data type of tensor y.
 */
MLOPEN_EXPORT mlopenStatus_t mlopenSetTensor(mlopenHandle_t                 handle,
		const mlopenTensorDescriptor_t yDesc,
		void                          *y,
		const void                    *valuePtr );

MLOPEN_EXPORT mlopenStatus_t mlopenScaleTensor(mlopenHandle_t                 handle,
		const mlopenTensorDescriptor_t yDesc,
		void                          *y,
		const void                    *alpha );

#if 0 
/* [MD]: I do not think there is any need to create separate filter
 * descriptor, just using the tensor descriptor should be fine.  mlopenStatus_t
 * mlopenCreateFilterDescriptor(mlopenFilterDescriptor_t *filterDesc);
 */

mlopenStatus_t mlopenInitFilterDescriptor(mlopenFilterDescriptor_t filterDesc,
		mlopenDataType_t datatype,
		int k,
		int c,
		int h,
		int w);

mlopenStatus_t mlopenGetFilterDescriptor(mlopenFilterDescriptor_t filterDesc,
		mlopenDataType_t datatype,
		int *k,
		int *c,
		int *h,
		int *w);

// TODO: Add APIs for N-dimensional filter descriptors. Tensorflow uses them.
//

mlopenStatus_t mlopenDestroyFilterDescriptor(mlopenFilterDescriptor_t filterDesc);

#endif // FilterDescriptor APIs

MLOPEN_EXPORT mlopenStatus_t mlopenCreateConvolutionDescriptor(
		mlopenConvolutionDescriptor_t *convDesc);

MLOPEN_EXPORT mlopenStatus_t mlopenInitConvolutionDescriptor(mlopenConvolutionDescriptor_t convDesc,
		mlopenConvolutionMode_t mode,
		int pad_h,
		int pad_w,
		int u,
		int v,
		int upscalex,
		int upscaley);

MLOPEN_EXPORT mlopenStatus_t mlopenGetConvolutionDescriptor(mlopenConvolutionDescriptor_t convDesc,
		mlopenConvolutionMode_t *mode,
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
MLOPEN_EXPORT mlopenStatus_t mlopenGetConvolutionForwardOutputDim(mlopenConvolutionDescriptor_t convDesc,
		const mlopenTensorDescriptor_t		inputTensorDesc,
		const mlopenTensorDescriptor_t		filterDesc,
		int									*n,
		int 								*c,
		int 								*h,
		int 								*w);
		

// TODO: Add APIs for N-dimensional filter descriptors. Tensorflow uses them.
//

MLOPEN_EXPORT mlopenStatus_t mlopenDestroyConvolutionDescriptor(mlopenConvolutionDescriptor_t convDesc);

// Same enum type for forward, backward filter and backward data
// algorthms
typedef enum {
	mlopenConvolutionNoWorkspace = 0,
	mlopenConvolutionFastest = 1,
	mlopenConvolutionWorkSpaceLimit = 2,
} mlopenConvPreference_t;

typedef enum {
	mlopenConvolutionFwdAlgoGEMM = 0,
	mlopenConvolutionFwdAlgoDirect = 1,
	mlopenConvolutionFwdAlgoFFT = 2,
	mlopenConvolutionFwdAlgoWinograd = 3,
} mlopenConvFwdAlgorithm_t;

typedef enum {
	mlopenConvolutionBwdFilterAlgo_0 = 0,
} mlopenConvBwdFilterAlgorithm_t;

typedef enum {
	mlopenConvolutionBwdDataAlgo_0 = 0,
} mlopenConvBwdDataAlgorithm_t;

// Same perf struct for forward, backward filter and backward
// data algorthms
typedef struct{
	union {
		mlopenConvFwdAlgorithm_t fwd_algo;
		mlopenConvBwdFilterAlgorithm_t bwd_filter_algo;
		mlopenConvBwdDataAlgorithm_t bwd_data_algo;
	};
	mlopenStatus_t status;
	float time;
	size_t memory;
} mlopenConvAlgoPerf_t;
/* This function attempts all MLOpen algorithms for mlopenConvolutionForward(),
 * and outputs performance metrics to a user- allocated array of
 * mlopenConvolutionFwdAlgoPerf_t. These metrics are written in sorted fashion
 * where the first element has the lowest compute time.
 *
 * [MD]: Ideally we want all the kernels to be
 * compiled, cached, etc. in this routine. Does this
 * mean that the user is required to call this
 * routine?
 * [MD]: Adding algo preference here itself such that this
 * routime works as both cuDNN's FindAlgorithm and GetAlgorithm
 * routines. I do not see any need of having two similar routines
 */
MLOPEN_EXPORT mlopenStatus_t mlopenFindConvolutionForwardAlgorithm(mlopenHandle_t handle,
		const mlopenTensorDescriptor_t		xDesc,
		const void							*x,
		const mlopenTensorDescriptor_t		wDesc,
		const void							*w,
		const mlopenConvolutionDescriptor_t	convDesc,
		const mlopenTensorDescriptor_t		yDesc,
		const void							*y,
		const int							requestAlgoCount,
		int									*returnedAlgoCount,
		mlopenConvAlgoPerf_t				*perfResults,
		mlopenConvPreference_t				preference,
		void								*workSpace,
		size_t								workSpaceSize,
		bool								exhaustiveSearch);

MLOPEN_EXPORT mlopenStatus_t mlopenConvolutionForward(mlopenHandle_t handle,
		const void							*alpha,
		const mlopenTensorDescriptor_t		xDesc,
		const void							*x,
		const mlopenTensorDescriptor_t		wDesc,
		const void							*w,
		const mlopenConvolutionDescriptor_t convDesc,
		mlopenConvFwdAlgorithm_t			algo,
		const void							*beta,
		const mlopenTensorDescriptor_t		yDesc,
		void								*y,
		void								*workSpace,
		size_t								workSpaceSize);

MLOPEN_EXPORT mlopenStatus_t mlopenFindConvolutionBackwardDataAlgorithm(mlopenHandle_t handle,
		const mlopenTensorDescriptor_t		dyDesc,
		const void							*dy,
		const mlopenTensorDescriptor_t		wDesc,
		const void							*w,
		const mlopenConvolutionDescriptor_t	convDesc,
		const mlopenTensorDescriptor_t		dxDesc,
		const void							*dx,
		const int							requestAlgoCount,
		int									*returnedAlgoCount,
		mlopenConvAlgoPerf_t				*perfResults,
		mlopenConvPreference_t				preference,
		void								*workSpace,
		size_t								workSpaceSize,
		bool								exhaustiveSearch);

MLOPEN_EXPORT mlopenStatus_t mlopenConvolutionBackwardData(mlopenHandle_t handle,
		const void							*alpha,
		const mlopenTensorDescriptor_t		dyDesc,
		const void							*dy,
		const mlopenTensorDescriptor_t		wDesc,
		const void							*w,
		const mlopenConvolutionDescriptor_t convDesc,
		mlopenConvBwdDataAlgorithm_t		algo,
		const void							*beta,
		const mlopenTensorDescriptor_t		dxDesc,
		void								*dx,
		void								*workSpace,
		size_t								workSpaceSize);

// Pooling APIs

MLOPEN_EXPORT mlopenStatus_t mlopenCreatePoolingDescriptor(mlopenPoolingDescriptor_t *poolDesc);

MLOPEN_EXPORT mlopenStatus_t mlopenSet2dPoolingDescriptor(
		mlopenPoolingDescriptor_t			poolDesc,
		mlopenPoolingMode_t					mode,
		int									windowHeight,
		int									windowWidth,
		int									pad_h,
		int									pad_w,
		int									u,
		int									v);
	
MLOPEN_EXPORT mlopenStatus_t mlopenGet2dPoolingDescriptor(
		const mlopenPoolingDescriptor_t		poolDesc,
		mlopenPoolingMode_t					*mode,
		int									*windowHeight,
		int									*windowWidth,
		int									*pad_h,
		int									*pad_w,
		int									*u,
		int									*v);

MLOPEN_EXPORT mlopenStatus_t mlopenSetNdPoolingDescriptor(
		mlopenPoolingDescriptor_t			poolDesc,
		mlopenPoolingMode_t					mode,
		int									nbDims,
		int									*windowDimA,
		int									*padA,
		int									*stridesA);

MLOPEN_EXPORT mlopenStatus_t mlopenGetNdPoolingDescriptor(
		const mlopenPoolingDescriptor_t		poolDesc,
		mlopenPoolingMode_t					*mode,
		int									*nbDims,
		int									*windowDimA,
		int									*padA,
		int									*stridesA);

MLOPEN_EXPORT mlopenStatus_t mlopenGetPoolingForwardOutputDim(
		const mlopenPoolingDescriptor_t		poolDesc,
		const mlopenTensorDescriptor_t		tensorDesc,
		int									*n,
		int									*c,
		int									*h,
		int									*w);

MLOPEN_EXPORT mlopenStatus_t mlopenPoolingForward(
		mlopenHandle_t						handle,
		const mlopenPoolingDescriptor_t		poolDesc,
		const void							*alpha,
		const mlopenTensorDescriptor_t		xDesc,
		const void							*x,
		const void							*beta,
		const mlopenTensorDescriptor_t		yDesc,
		void								*y,
		bool                                do_backward,
		void								*workSpace,
		size_t								workSpaceSize);


MLOPEN_EXPORT mlopenStatus_t mlopenPoolingBackward(
		mlopenHandle_t						handle,
		const mlopenPoolingDescriptor_t		poolDesc,
		const void							*alpha,
		const mlopenTensorDescriptor_t		yDesc,
		const void							*y,
		const mlopenTensorDescriptor_t		dyDesc,
		const void							*dy,
		const mlopenTensorDescriptor_t		xDesc,
		const void							*x,
		const void							*beta,
		const mlopenTensorDescriptor_t		dxDesc,
		void								*dx,
		const void							*workSpace);


MLOPEN_EXPORT mlopenStatus_t mlopenDestroyPoolingDescriptor(mlopenPoolingDescriptor_t poolDesc);

// LRN APIs

MLOPEN_EXPORT mlopenStatus_t mlopenCreateLRNDescriptor(mlopenLRNDescriptor_t *lrnDesc);

MLOPEN_EXPORT mlopenStatus_t mlopenSetLRNDescriptor(
	const mlopenLRNDescriptor_t			lrnDesc,
	mlopenLRNMode_t						mode,
	unsigned int						lrnN,
	double								lrnAlpha,
	double								lrnBeta,
	double								lrnK);


MLOPEN_EXPORT mlopenStatus_t mlopenGetLRNDescriptor(
		const mlopenLRNDescriptor_t			lrnDesc,
		mlopenLRNMode_t						*mode,
		unsigned int						*lrnN,
		double								*lrnAlpha,
		double								*lrnBeta,
		double								*lrnK);



MLOPEN_EXPORT mlopenStatus_t mlopenLRNForward(
		mlopenHandle_t						handle,
		const mlopenLRNDescriptor_t			lrnDesc,
		const void							*alpha,
		const mlopenTensorDescriptor_t		xDesc,
		const void							*x,
		const void							*beta,
		const mlopenTensorDescriptor_t		yDesc,
		void								*y,
		bool                                do_backward,
		void								*workSpace,
		size_t								*workSpaceSize);


MLOPEN_EXPORT mlopenStatus_t mlopenLRNBackward(
		mlopenHandle_t						handle,
		const mlopenLRNDescriptor_t			lrnDesc,
		const void							*alpha,
		const mlopenTensorDescriptor_t		yDesc,
		const void							*y,
		const mlopenTensorDescriptor_t		dyDesc,
		const void							*dy,
		const mlopenTensorDescriptor_t		xDesc,
		const void							*x,
		const void							*beta,
		const mlopenTensorDescriptor_t		dxDesc,
		void								*dx,
		const void							*workSpace);

MLOPEN_EXPORT mlopenStatus_t mlopenDestroyLRNDescriptor(mlopenLRNDescriptor_t lrnDesc);
#ifdef __cplusplus
}
#endif

#endif // _MLOPEN_H_

