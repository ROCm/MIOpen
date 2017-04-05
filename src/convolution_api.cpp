#include <miopen/convolution.hpp>
#include <miopen/errors.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

// TODO: Make miopenConvAlgoPerf_t loggable
inline std::ostream& operator<<(std::ostream& os, miopenConvAlgoPerf_t)
{
    return os;
}

extern "C"
miopenStatus_t miopenCreateConvolutionDescriptor(
		miopenConvolutionDescriptor_t *convDesc) {
	MIOPEN_LOG_FUNCTION(convDesc);
	return miopen::try_([&] {
		miopen::deref(convDesc) = new miopen::ConvolutionDescriptor();
	});
}

extern "C"
miopenStatus_t miopenInitConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc,
		miopenConvolutionMode_t	mode,
		int						pad_h,
		int						pad_w,
		int						u,
		int						v,
		int						upscalex,
		int						upscaley) {
	
	MIOPEN_LOG_FUNCTION(convDesc, mode, pad_h, pad_w, u, v, upscalex, upscaley);
	return miopen::try_([&] {
		miopen::deref(convDesc) = miopen::ConvolutionDescriptor(mode, pad_h, pad_w, u, v, upscalex, upscaley);
	});
}

extern "C"
miopenStatus_t miopenGetConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc,
		miopenConvolutionMode_t *mode,
		int						*pad_h,
		int						*pad_w,
		int						*u,
		int						*v,
		int						*upscalex,
		int						*upscaley) {

	MIOPEN_LOG_FUNCTION(convDesc, mode, pad_h, pad_w, u, v, upscalex, upscaley);
	return miopen::try_([&] {
		miopen::deref(mode)		= miopen::deref(convDesc).mode;
		miopen::deref(pad_h)		= miopen::deref(convDesc).pad_h;
		miopen::deref(pad_w)		= miopen::deref(convDesc).pad_w;
		miopen::deref(u)			= miopen::deref(convDesc).u;
		miopen::deref(v)			= miopen::deref(convDesc).v;
		miopen::deref(upscalex)	= miopen::deref(convDesc).upscalex;
		miopen::deref(upscaley)	= miopen::deref(convDesc).upscaley;
	});
}

extern "C"
miopenStatus_t miopenGetConvolutionForwardOutputDim(miopenConvolutionDescriptor_t convDesc,
		const miopenTensorDescriptor_t	inputTensorDesc,
		const miopenTensorDescriptor_t	filterDesc,
		int								*n,
		int								*c,
		int								*h,
		int								*w) {

	MIOPEN_LOG_FUNCTION(convDesc, inputTensorDesc, filterDesc, n, c, h, w);
	return miopen::try_([&] {
		miopen::tie_deref(n, c, h, w) = miopen::deref(convDesc).GetForwardOutputDim(
			miopen::deref(inputTensorDesc), 
			miopen::deref(filterDesc));
	});

}

extern "C"
miopenStatus_t miopenDestroyConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc) {
	MIOPEN_LOG_FUNCTION(convDesc);
	return miopen::try_([&] {
		delete convDesc;
	});
}

extern "C"
miopenStatus_t miopenConvolutionForwardGetWorkSpaceSize(
		const miopenTensorDescriptor_t		wDesc,
		const miopenTensorDescriptor_t		xDesc,
		const miopenTensorDescriptor_t		yDesc,
		const miopenConvolutionDescriptor_t convDesc,
		size_t								*workSpaceSize) {

	MIOPEN_LOG_FUNCTION(wDesc, yDesc, convDesc, workSpaceSize);
	miopen::try_([&] {
		miopen::deref(workSpaceSize) = miopen::deref(convDesc).ForwardGetWorkSpaceSize(
			miopen::deref(wDesc),
			miopen::deref(xDesc),
			miopen::deref(yDesc));
	});

	return(miopenStatusSuccess);
}

extern "C"
miopenStatus_t miopenFindConvolutionForwardAlgorithm(miopenHandle_t handle,
		const miopenTensorDescriptor_t		xDesc,
		const void							*x,
		const miopenTensorDescriptor_t		wDesc,
		const void							*w,
		const miopenConvolutionDescriptor_t	convDesc,
		const miopenTensorDescriptor_t		yDesc,
		void								*y,
		const int							requestAlgoCount,
		int									*returnedAlgoCount,
		miopenConvAlgoPerf_t				*perfResults,
		void								*workSpace,
		size_t								workSpaceSize,
		bool								exhaustiveSearch) {

	MIOPEN_LOG_FUNCTION(xDesc, x, wDesc, w, convDesc, yDesc, y, requestAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSize, exhaustiveSearch);
	return miopen::try_([&] {
		miopen::deref(convDesc).FindConvFwdAlgorithm(miopen::deref(handle),
				miopen::deref(xDesc),
				DataCast(x),
				miopen::deref(wDesc),
				DataCast(w),
				miopen::deref(yDesc),
				DataCast(y),
				requestAlgoCount,
				returnedAlgoCount,
				perfResults,
				DataCast(workSpace),
				workSpaceSize,
				exhaustiveSearch);
	});

}

extern "C"
miopenStatus_t miopenConvolutionForward(miopenHandle_t handle,
		const void							*alpha,
		const miopenTensorDescriptor_t		xDesc,
		const void							*x,
		const miopenTensorDescriptor_t		wDesc,
		const void							*w,
		const miopenConvolutionDescriptor_t convDesc,
		miopenConvFwdAlgorithm_t			algo,
		const void							*beta,
		const miopenTensorDescriptor_t		 yDesc,
		void								*y,
		void								*workSpace,
		size_t								workSpaceSize) {

	MIOPEN_LOG_FUNCTION(alpha, xDesc, x, wDesc, w, convDesc, algo, beta, yDesc, y, workSpace, workSpaceSize);
	return miopen::try_([&] {
		miopen::deref(convDesc).ConvolutionForward(miopen::deref(handle),
				alpha,
				miopen::deref(xDesc),
				DataCast(x),
				miopen::deref(wDesc),
				DataCast(w),
				algo,
				beta,
				miopen::deref(yDesc),
				DataCast(y),
				DataCast(workSpace),
				workSpaceSize);
	});


}

extern "C"
miopenStatus_t miopenConvolutionForwardBias(miopenHandle_t handle,
		const void						*alpha,
		const miopenTensorDescriptor_t	bDesc,
		const void						*b,
		const void						*beta,
		const miopenTensorDescriptor_t	yDesc,
		void							*y) {

	MIOPEN_LOG_FUNCTION(alpha, bDesc, b, beta, yDesc, y);
    return miopen::try_([&] {
		return AddTensor(miopen::deref(handle), 
				alpha,
				miopen::deref(bDesc),
				DataCast(b),
				beta,
				miopen::deref(yDesc),
				DataCast(y));
	});

}

extern "C"
miopenStatus_t miopenFindConvolutionBackwardDataAlgorithm(miopenHandle_t handle,
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
		bool								exhaustiveSearch) {

	MIOPEN_LOG_FUNCTION(dyDesc, dy, wDesc, w, convDesc, dxDesc, dx, requestAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSize, exhaustiveSearch);
	return miopen::try_([&] {
		miopen::deref(convDesc).FindConvBwdDataAlgorithm(miopen::deref(handle),
				miopen::deref(dyDesc),
				DataCast(dy),
				miopen::deref(wDesc),
				DataCast(w),
				miopen::deref(dxDesc),
				DataCast(dx),
				requestAlgoCount,
				returnedAlgoCount,
				perfResults,
				workSpace,
				workSpaceSize,
				exhaustiveSearch);
	});

}

extern "C"
miopenStatus_t miopenConvolutionBackwardData(miopenHandle_t handle,
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
		size_t								workSpaceSize) {

	MIOPEN_LOG_FUNCTION(alpha, dyDesc, dy, wDesc, w, convDesc, algo, beta, dxDesc, dx, workSpace, workSpaceSize);
	return miopen::try_([&] {
		miopen::deref(convDesc).ConvolutionBackwardData(miopen::deref(handle),
				alpha,
				miopen::deref(dyDesc),
				DataCast(dy),
				miopen::deref(wDesc),
				DataCast(w),
				algo,
				beta,
				miopen::deref(dxDesc),
				DataCast(dx),
				workSpace,
				workSpaceSize);
	});

}

extern "C"
miopenStatus_t miopenConvolutionBackwardWeightsGetWorkSpaceSize(
		const miopenTensorDescriptor_t		dyDesc,
		const miopenTensorDescriptor_t		xDesc,
		const miopenConvolutionDescriptor_t	convDesc,
		const miopenTensorDescriptor_t		dwDesc,
		size_t								*workSpaceSize) {

	MIOPEN_LOG_FUNCTION(dyDesc, xDesc, convDesc, dwDesc, workSpaceSize);
	return miopen::try_([&] {
		miopen::deref(workSpaceSize) = miopen::deref(convDesc).ConvolutionBackwardWeightsGetWorkSpaceSize(
			miopen::deref(dyDesc),
			miopen::deref(xDesc),
			miopen::deref(dwDesc));
	});
}

extern "C"
miopenStatus_t miopenFindConvolutionBackwardWeightsAlgorithm(miopenHandle_t handle,
		const miopenTensorDescriptor_t		dyDesc,
		const void							*dy,
		const miopenTensorDescriptor_t		xDesc,
		const void							*x,
		const miopenConvolutionDescriptor_t	convDesc,
		const miopenTensorDescriptor_t		dwDesc,
		void								*dw,
		const int							requestAlgoCount,
		int									*returnedAlgoCount,
		miopenConvAlgoPerf_t				*perfResults,
		void								*workSpace,
		size_t								workSpaceSize,
		bool								exhaustiveSearch) {

	MIOPEN_LOG_FUNCTION(dyDesc, dy, xDesc, x, convDesc, dwDesc, dw, requestAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSize, exhaustiveSearch);
	return miopen::try_([&] {
		miopen::deref(convDesc).FindConvBwdWeightsAlgorithm(miopen::deref(handle),
				miopen::deref(dyDesc),
				DataCast(dy),
				miopen::deref(xDesc),
				DataCast(x),
				miopen::deref(dwDesc),
				DataCast(dw),
				requestAlgoCount,
				returnedAlgoCount,
				perfResults,
				DataCast(workSpace),
				workSpaceSize,
				exhaustiveSearch);
	});

}

extern "C"
miopenStatus_t miopenConvolutionBackwardWeights(miopenHandle_t handle,
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
		size_t								workSpaceSize) {

	MIOPEN_LOG_FUNCTION(alpha, dyDesc, dy, xDesc, x, convDesc, algo, beta, dwDesc, dw, workSpace, workSpaceSize);
	return miopen::try_([&] {
        miopen::deref(convDesc).ConvolutionBackwardWeights(miopen::deref(handle),
				alpha,
				miopen::deref(dyDesc),
				DataCast(dy),
				miopen::deref(xDesc),
				DataCast(x),
				algo,
				beta,
				miopen::deref(dwDesc),
				DataCast(dw),
				DataCast(workSpace),
				workSpaceSize);
	});

}

extern "C"
miopenStatus_t miopenConvolutionBackwardBias(miopenHandle_t handle,
                                             const void						*alpha,
                                             const miopenTensorDescriptor_t	dyDesc,
                                             const void						*dy,
                                             const void						*beta,
                                             const miopenTensorDescriptor_t	dbDesc,
                                             void							*db) {
    return miopen::try_([&] {
        ConvolutionBackwardBias(miopen::deref(handle),
                                alpha,
                                miopen::deref(dyDesc),
                                DataCast(dy),
                                beta,
                                miopen::deref(dbDesc),
                                DataCast(db));
    });
}

