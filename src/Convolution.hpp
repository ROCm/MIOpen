#ifndef _MLOPEN_CONVOLUTION_HPP_
#define _MLOPEN_CONVOLUTION_HPP_

#include "MLOpen.h"
#include "Handle.hpp"
#include "Tensor.hpp"
#include "KernelCache.hpp"
#include "Common.hpp"

struct mlopenConvolutionDescriptor {
	
	mlopenConvolutionDescriptor();
	~mlopenConvolutionDescriptor() {}

	mlopenStatus_t GetForwardOutputDim(const mlopenTensorDescriptor_t inputTensorDesc,
			const mlopenTensorDescriptor_t filterDesc,
			int *n,
			int *c,
			int *h, 
			int *w);

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
		size_t							workSpaceSize,
		bool							exhaustiveSearch);

	mlopenStatus_t ConvolutionForward(mlopenHandle_t handle,
		const void							*alpha,
		const mlopenTensorDescriptor_t		xDesc,
		const Data_t						x,
		const mlopenTensorDescriptor_t		wDesc,
		const Data_t						w,
		mlopenConvFwdAlgorithm_t			algo,
		const void							*beta,
		const mlopenTensorDescriptor_t		yDesc,
		Data_t								y,
		void								*workSpace,
		size_t								workSpaceSize);

	mlopenStatus_t FindConvBwdDataAlgorithm(mlopenHandle_t handle,
		const mlopenTensorDescriptor_t	dyDesc,
		const Data_t					dy,
		const mlopenTensorDescriptor_t	wDesc,
		const Data_t					w,
		const mlopenTensorDescriptor_t	dxDesc,
		const Data_t					dx,
		const int						requestAlgoCount,
		int								*returnedAlgoCount,
		mlopenConvAlgoPerf_t			*perfResults,
		mlopenConvPreference_t			preference,
		void							*workSpace,
		size_t							workSpaceSize);

	mlopenStatus_t ConvolutionBackwardData(mlopenHandle_t handle,
		const void							*alpha,
		const mlopenTensorDescriptor_t		dyDesc,
		const Data_t						dy,
		const mlopenTensorDescriptor_t		wDesc,
		const Data_t						w,
		mlopenConvBwdDataAlgorithm_t		algo,
		const void							*beta,
		const mlopenTensorDescriptor_t		dxDesc,
		Data_t								dx,
		void								*workSpace,
		size_t								workSpaceSize);

	mlopenConvolutionMode_t _mode;
	int _pad_h;
	int _pad_w;
	int _u;
	int _v;
	int _upscalex;
	int _upscaley;
};

#endif // _MLOPEN_CONVOLUTION_HPP_
