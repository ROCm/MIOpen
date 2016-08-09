#include <mlopen/convolution.hpp>
#include <mlopen/errors.hpp>

extern "C"
mlopenStatus_t mlopenCreateConvolutionDescriptor(
		mlopenConvolutionDescriptor_t *convDesc) {

	return mlopen::try_([&] {
		mlopen::deref(convDesc) = new mlopen::ConvolutionDescriptor();
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
	
	return mlopen::try_([&] {
		mlopen::deref(convDesc) = mlopen::ConvolutionDescriptor(mode, pad_h, pad_w, u, v, upscalex, upscaley);
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
		mlopen::deref(mode)		= mlopen::deref(convDesc).mode;
		mlopen::deref(pad_h)		= mlopen::deref(convDesc).pad_h;
		mlopen::deref(pad_w)		= mlopen::deref(convDesc).pad_w;
		mlopen::deref(u)			= mlopen::deref(convDesc).u;
		mlopen::deref(v)			= mlopen::deref(convDesc).v;
		mlopen::deref(upscalex)	= mlopen::deref(convDesc).upscalex;
		mlopen::deref(upscaley)	= mlopen::deref(convDesc).upscaley;
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
		mlopen::tie_deref(n, c, h, w) = mlopen::deref(convDesc).GetForwardOutputDim(
			mlopen::deref(inputTensorDesc), 
			mlopen::deref(filterDesc));
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
		mlopen::deref(convDesc).FindConvFwdAlgorithm(mlopen::deref(handle),
				mlopen::deref(xDesc),
				DataCast(x),
				mlopen::deref(wDesc),
				DataCast(w),
				mlopen::deref(yDesc),
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
		mlopen::deref(convDesc).ConvolutionForward(mlopen::deref(handle),
				alpha,
				mlopen::deref(xDesc),
				DataCast(x),
				mlopen::deref(wDesc),
				DataCast(w),
				algo,
				beta,
				mlopen::deref(yDesc),
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
		mlopen::deref(convDesc).FindConvBwdDataAlgorithm(mlopen::deref(handle),
				mlopen::deref(dyDesc),
				DataCast(dy),
				mlopen::deref(wDesc),
				DataCast(w),
				mlopen::deref(dxDesc),
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
		mlopen::deref(convDesc).ConvolutionBackwardData(mlopen::deref(handle),
				alpha,
				mlopen::deref(dyDesc),
				DataCast(dy),
				mlopen::deref(wDesc),
				DataCast(w),
				algo,
				beta,
				mlopen::deref(dxDesc),
				DataCast(dx),
				workSpace,
				workSpaceSize);
	});

}

