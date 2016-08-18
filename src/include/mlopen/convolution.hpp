#ifndef GUARD_MLOPEN_CONVOLUTION_HPP_
#define GUARD_MLOPEN_CONVOLUTION_HPP_

#include <mlopen.h>
#include <mlopen/handle.hpp>
#include <mlopen/tensor.hpp>
#include <mlopen/common.hpp>

namespace mlopen {

struct ConvolutionDescriptor : mlopenConvolutionDescriptor {
	
	ConvolutionDescriptor(int p_pad_h = 0, int p_pad_w = 0, int p_u = 1, int p_v = 1, int p_upscalex = 1, int p_upscaley = 1);
	ConvolutionDescriptor(mlopenConvolutionMode_t p_mode, int p_pad_h = 0, int p_pad_w = 0, int p_u = 1, int p_v = 1, int p_upscalex = 1, int p_upscaley = 1);

	std::tuple<int, int, int, int> GetForwardOutputDim(const TensorDescriptor& inputTensorDesc, const TensorDescriptor& filterDesc) const;
	TensorDescriptor GetForwardOutputTensor(const TensorDescriptor& inputTensorDesc, const TensorDescriptor& filterDesc) const;

	mlopenStatus_t FindConvFwdAlgorithm(mlopen::Handle& handle,
		const mlopen::TensorDescriptor&	xDesc,
		ConstData_t					x,
		const mlopen::TensorDescriptor&	wDesc,
		ConstData_t					w,
		const mlopen::TensorDescriptor&	yDesc,
		ConstData_t					y,
		const int						requestAlgoCount,
		int								*returnedAlgoCount,
		mlopenConvAlgoPerf_t			*perfResults,
		mlopenConvPreference_t			preference,
		void							*workSpace,
		size_t							workSpaceSize,
		bool							exhaustiveSearch) const;

	mlopenStatus_t ConvolutionForward(mlopen::Handle& handle,
		const void							*alpha,
		const mlopen::TensorDescriptor&		xDesc,
		ConstData_t						x,
		const mlopen::TensorDescriptor&		wDesc,
		ConstData_t						w,
		mlopenConvFwdAlgorithm_t			algo,
		const void							*beta,
		const mlopen::TensorDescriptor&		yDesc,
		Data_t								y,
		void								*workSpace,
		size_t								workSpaceSize) const;

	mlopenStatus_t FindConvBwdDataAlgorithm(mlopen::Handle& handle,
		const mlopen::TensorDescriptor&	dyDesc,
		ConstData_t					dy,
		const mlopen::TensorDescriptor&	wDesc,
		ConstData_t					w,
		const mlopen::TensorDescriptor&	dxDesc,
		ConstData_t					dx,
		const int						requestAlgoCount,
		int								*returnedAlgoCount,
		mlopenConvAlgoPerf_t			*perfResults,
		mlopenConvPreference_t			preference,
		void							*workSpace,
		size_t							workSpaceSize,
		bool							exhaustiveSearch) const;

	mlopenStatus_t ConvolutionBackwardData(mlopen::Handle& handle,
		const void							*alpha,
		const mlopen::TensorDescriptor&		dyDesc,
		ConstData_t						dy,
		const mlopen::TensorDescriptor&		wDesc,
		ConstData_t						w,
		mlopenConvBwdDataAlgorithm_t		algo,
		const void							*beta,
		const mlopen::TensorDescriptor&		dxDesc,
		Data_t								dx,
		void								*workSpace,
		size_t								workSpaceSize) const;

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

#endif // GUARD_MLOPEN_CONVOLUTION_HPP_
