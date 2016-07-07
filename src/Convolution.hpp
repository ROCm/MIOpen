#ifndef _CONVOLUTION_HPP_
#define _CONVOLUTION_HPP_

#include "MLOpen.h"
#include "Handle.hpp"
#include "Tensor.hpp"
#include "KernelCache.hpp"

struct mlopenConvolutionDescriptor {
	
	mlopenConvolutionDescriptor();
	~mlopenConvolutionDescriptor() {}

	mlopenStatus_t GetForwardOutputDim(const mlopenTensorDescriptor_t inputTensorDesc,
			const mlopenTensorDescriptor_t filterDesc,
			int *n,
			int *c,
			int *h, 
			int *w);

	template <typename Data_t>
	mlopenStatus_t FindConvFwdAlgorithm(mlopenHandle_t handle,
		const mlopenTensorDescriptor_t	xDesc,
		const Data_t					x,
		const mlopenTensorDescriptor_t	wDesc,
		const Data_t					w,
		const mlopenTensorDescriptor_t	yDesc,
		const Data_t					y,
		const int						requestAlgoCount,
		int								*returnedAlgoCount,
		mlopenConvAlgoPerf_t			*perfResults,
		mlopenConvPreference_t			preference,
		void							*workSpace,
		size_t							workSpaceSize);

	template <typename Data_t>
	mlopenStatus_t ConvolutionForward(mlopenHandle_t handle,
		const void							*alpha,
		const mlopenTensorDescriptor_t		xDesc,
		const Data_t						x,
		const mlopenTensorDescriptor_t		wDesc,
		const Data_t						w,
		mlopenConvFwdAlgorithm_t			algo,
		const void							*beta,
		const mlopenTensorDescriptor_t		yDesc,
		Data_t								y);

	mlopenConvolutionMode_t _mode;
	int _pad_h;
	int _pad_w;
	int _u;
	int _v;
	int _upscalex;
	int _upscaley;
	mlopenHandle_t _convHandle;
};

// Template Instantations
//
#if MLOpen_BACKEND_OPENCL
template<>
mlopenStatus_t mlopenConvolutionDescriptor::FindConvFwdAlgorithm<cl_mem>(mlopenHandle_t handle,
		const mlopenTensorDescriptor_t	xDesc,
		const cl_mem					x,
		const mlopenTensorDescriptor_t	wDesc,
		const cl_mem					w,
		const mlopenTensorDescriptor_t	yDesc,
		const cl_mem					y,
		const int						requestAlgoCount,
		int								*returnedAlgoCount,
		mlopenConvAlgoPerf_t			*perfResults,
		mlopenConvPreference_t			preference,
		void							*workSpace,
		size_t							workSpaceSize);

template<>
mlopenStatus_t mlopenConvolutionDescriptor::ConvolutionForward<cl_mem>(mlopenHandle_t handle,
		const void							*alpha,
		const mlopenTensorDescriptor_t		xDesc,
		const cl_mem						x,
		const mlopenTensorDescriptor_t		wDesc,
		const cl_mem						w,
		mlopenConvFwdAlgorithm_t			algo,
		const void							*beta,
		const mlopenTensorDescriptor_t		yDesc,
		cl_mem								y);

#elif MLOpen_BACKEND_HIP
template<>
mlopenStatus_t mlopenConvolutionDescriptor::FindConvFwdAlgorithm<void *>(mlopenHandle_t handle,
		const mlopenTensorDescriptor_t	xDesc,
		const void						*x,
		const mlopenTensorDescriptor_t	wDesc,
		const void						*w,
		const mlopenTensorDescriptor_t	yDesc,
		const void						*y,
		const int						requestAlgoCount,
		int								*returnedAlgoCount,
		mlopenConvAlgoPerf_t			*perfResults,
		mlopenConvPreference_t			preference,
		void							*workSpace,
		size_t							workSpaceSize);

template<>
mlopenStatus_t mlopenConvolutionDescriptor::ConvolutionForward<void *>(mlopenHandle_t handle,
		const void							*alpha,
		const mlopenTensorDescriptor_t		xDesc,
		const void							*x,
		const mlopenTensorDescriptor_t		wDesc,
		const void							*w,
		mlopenConvFwdAlgorithm_t			algo,
		const void							*beta,
		const mlopenTensorDescriptor_t		yDesc,
		void								*y);

#endif // HIP vs OpenCL

#endif // _CONVOLUTION_HPP_
