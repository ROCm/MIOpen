#include "Convolution.hpp"
#include <errors.hpp>

extern "C"
mlopenStatus_t mlopenCreateConvolutionDescriptor(
		mlopenConvolutionDescriptor_t *convDesc) {

	return mlopen::try_([&] {
		mlopen::deref(convDesc) = new mlopenConvolutionDescriptor();
	});
}

extern "C"
mlopenStatus_t mlopenInitConvolutionDescriptor(mlopenConvolutionDescriptor_t convDesc,
		mlopenConvolutionMode_t	mode,
		int						pad_h,
		int						pad_w,
		int						u,
		int						v,
		int						upscalex,
		int						upscaley) {

	if(convDesc == nullptr) {
		return mlopenStatusBadParm;
	}
	if(pad_h < 0 || pad_w < 0) {
		return mlopenStatusBadParm;
	}
	if(u < 0 || v < 0) {
		return mlopenStatusBadParm;
	}

	return mlopen::try_([&] {
		convDesc->_mode		= mode;
		convDesc->_pad_h	= pad_h;
		convDesc->_pad_w	= pad_w;
		convDesc->_u		= u;
		convDesc->_v		= v;
		convDesc->_upscalex = upscalex;
		convDesc->_upscaley = upscaley;
	});
}

extern "C"
mlopenStatus_t mlopenGetConvolutionDescriptor(mlopenConvolutionDescriptor_t convDesc,
		mlopenConvolutionMode_t *mode,
		int						*pad_h,
		int						*pad_w,
		int						*u,
		int						*v,
		int						*upscalex,
		int						*upscaley) {

	return mlopen::try_([&] {
		mlopen::deref(mode)		= convDesc->_mode;
		mlopen::deref(pad_h)		= convDesc->_pad_h;
		mlopen::deref(pad_w)		= convDesc->_pad_w;
		mlopen::deref(u)			= convDesc->_u;
		mlopen::deref(v)			= convDesc->_v;
		mlopen::deref(upscalex)	= convDesc->_upscalex;
		mlopen::deref(upscaley)	= convDesc->_upscaley;
	});
}

extern "C"
mlopenStatus_t mlopenGetConvolutionForwardOutputDim(mlopenConvolutionDescriptor_t convDesc,
		const mlopenTensorDescriptor_t	inputTensorDesc,
		const mlopenTensorDescriptor_t	filterDesc,
		int								*n,
		int								*c,
		int								*h,
		int								*w) {

	return mlopen::try_([&] {
		convDesc->GetForwardOutputDim(inputTensorDesc,
				filterDesc,
				n, 
				c, 
				h,
				w);
	});

}

extern "C"
mlopenStatus_t mlopenDestroyConvolutionDescriptor(mlopenConvolutionDescriptor_t convDesc) {
	return mlopen::try_([&] {
		delete convDesc;
	});
}

extern "C"
mlopenStatus_t mlopenFindConvolutionForwardAlgorithm(mlopenHandle_t handle,
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
		bool								exhaustiveSearch) {

	return mlopen::try_([&] {
		convDesc->FindConvFwdAlgorithm(handle,
				xDesc,
				DataCast(x),
				wDesc,
				DataCast(w),
				yDesc,
				DataCast(y),
				requestAlgoCount,
				returnedAlgoCount,
				perfResults,
				preference,
				workSpace,
				workSpaceSize,
				exhaustiveSearch);
	});

}

extern "C"
mlopenStatus_t mlopenConvolutionForward(mlopenHandle_t handle,
		const void							*alpha,
		const mlopenTensorDescriptor_t		xDesc,
		const void							*x,
		const mlopenTensorDescriptor_t		wDesc,
		const void							*w,
		const mlopenConvolutionDescriptor_t convDesc,
		mlopenConvFwdAlgorithm_t			algo,
		const void							*beta,
		const mlopenTensorDescriptor_t		 yDesc,
		void								*y,
		void								*workSpace,
		size_t								workSpaceSize) {

	return mlopen::try_([&] {
		convDesc->ConvolutionForward(handle,
				alpha,
				xDesc,
				DataCast(x),
				wDesc,
				DataCast(w),
				algo,
				beta,
				yDesc,
				DataCast(y),
				workSpace,
				workSpaceSize);
	});


}

extern "C"
mlopenStatus_t mlopenFindConvolutionBackwardDataAlgorithm(mlopenHandle_t handle,
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
		size_t								workSpaceSize) {

	return mlopen::try_([&] {
		convDesc->FindConvBwdDataAlgorithm(handle,
				dyDesc,
				DataCast(dy),
				wDesc,
				DataCast(w),
				dxDesc,
				DataCast(dx),
				requestAlgoCount,
				returnedAlgoCount,
				perfResults,
				preference,
				workSpace,
				workSpaceSize);
	});

}

extern "C"
mlopenStatus_t mlopenConvolutionBackwardData(mlopenHandle_t handle,
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
		size_t								workSpaceSize) {

	return mlopen::try_([&] {
		convDesc->ConvolutionBackwardData(handle,
				alpha,
				dyDesc,
				DataCast(dy),
				wDesc,
				DataCast(w),
				algo,
				beta,
				dxDesc,
				DataCast(dx),
				workSpace,
				workSpaceSize);
	});

}

