#ifndef _MLOPEN_CONVOLUTION_HPP_
#define _MLOPEN_CONVOLUTION_HPP_

#include <mlopen.h>
#include <mlopen/context.hpp>
#include <mlopen/tensor.hpp>
#include <mlopen/kernel_cache.hpp>
#include <mlopen/common.hpp>

namespace mlopen {

struct ConvolutionDescriptor : mlopenConvolutionDescriptor {
	
	ConvolutionDescriptor(int p_pad_h = 0, int p_pad_w = 0, int p_u = 1, int p_v = 1, int p_upscalex = 1, int p_upscaley = 1);
	ConvolutionDescriptor(mlopenConvolutionMode_t p_mode, int p_pad_h = 0, int p_pad_w = 0, int p_u = 1, int p_v = 1, int p_upscalex = 1, int p_upscaley = 1);
	// ConvolutionDescriptor();
	// ~ConvolutionDescriptor() {}

	mlopenStatus_t GetForwardOutputDim(const mlopen::TensorDescriptor& inputTensorDesc,
			const mlopen::TensorDescriptor& filterDesc,
			int *n,
			int *c,
			int *h, 
			int *w);

	mlopenStatus_t FindConvFwdAlgorithm(mlopen::Context& handle,
		const mlopen::TensorDescriptor&	xDesc,
		const Data_t					x,
		const mlopen::TensorDescriptor&	wDesc,
		const Data_t					w,
		const mlopen::TensorDescriptor&	yDesc,
		const Data_t					y,
		const int						requestAlgoCount,
		int								*returnedAlgoCount,
		mlopenConvAlgoPerf_t			*perfResults,
		mlopenConvPreference_t			preference,
		void							*workSpace,
		size_t							workSpaceSize,
		bool							exhaustiveSearch);

	mlopenStatus_t ConvolutionForward(mlopen::Context& handle,
		const void							*alpha,
		const mlopen::TensorDescriptor&		xDesc,
		const Data_t						x,
		const mlopen::TensorDescriptor&		wDesc,
		const Data_t						w,
		mlopenConvFwdAlgorithm_t			algo,
		const void							*beta,
		const mlopen::TensorDescriptor&		yDesc,
		Data_t								y,
		void								*workSpace,
		size_t								workSpaceSize);

	mlopenStatus_t FindConvBwdDataAlgorithm(mlopen::Context& handle,
		const mlopen::TensorDescriptor&	dyDesc,
		const Data_t					dy,
		const mlopen::TensorDescriptor&	wDesc,
		const Data_t					w,
		const mlopen::TensorDescriptor&	dxDesc,
		const Data_t					dx,
		const int						requestAlgoCount,
		int								*returnedAlgoCount,
		mlopenConvAlgoPerf_t			*perfResults,
		mlopenConvPreference_t			preference,
		void							*workSpace,
		size_t							workSpaceSize);

	mlopenStatus_t ConvolutionBackwardData(mlopen::Context& handle,
		const void							*alpha,
		const mlopen::TensorDescriptor&		dyDesc,
		const Data_t						dy,
		const mlopen::TensorDescriptor&		wDesc,
		const Data_t						w,
		mlopenConvBwdDataAlgorithm_t		algo,
		const void							*beta,
		const mlopen::TensorDescriptor&		dxDesc,
		Data_t								dx,
		void								*workSpace,
		size_t								workSpaceSize);

	mlopenConvolutionMode_t mode;
	int pad_h;
	int pad_w;
	int u;
	int v;
	int upscalex;
	int upscaley;
};
}
MLOPEN_DEFINE_OBJECT(mlopenConvolutionDescriptor, mlopen::ConvolutionDescriptor);

#endif // _MLOPEN_CONVOLUTION_HPP_
