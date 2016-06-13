#include "Convolution.hpp"

mlopenConvolutionDescriptor::mlopenConvolutionDescriptor() : _pad_h(0), _pad_w(0), _u(1), _v(1), _upscalex(0), _upscaley(0) {
	_mode = mlopenConvolution;
}

mlopenStatus_t mlopenConvolutionDescriptor::FindConvFwdAlgorithm(mlopenHandle_t handle,
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
		size_t							workSpaceSize) {
	
	printf("To be implemented\n");

	if(handle == nullptr) {
		return mlopenStatusBadParm;
	}
	if(xDesc == nullptr || wDesc == nullptr || yDesc == nullptr) {
		return mlopenStatusBadParm;
	}
	if(x == nullptr || w == nullptr || y == nullptr) {
		return mlopenStatusBadParm;
	}
	if(returnedAlgoCount == nullptr || perfResults == nullptr) {
		return mlopenStatusBadParm;
	}
	if(requestAlgoCount < 1) {
		return mlopenStatusBadParm;
	}

	// Compile, cache kernels, etc.
	// Launch all kernels and store the perf, workspace limits, etc.
	
	std::string program_name; // CL kernel filename
	std::string kernel_name; // kernel name
	std::string parms; // kernel parameters

	mlopenStream_t queue;
	handle->GetStream(&queue);

#if MLOpen_BACKEND_OPENCL
	cl::Kernel kernel = KernelCache::get(reinterpret_cast<cl::CommandQueue&>(queue), program_name, kernel_name, parms); 
#endif

	return mlopenStatusSuccess;

}

mlopenStatus_t mlopenConvolutionDescriptor::ConvolutionForward(mlopenHandle_t handle,
		const void							*alpha,
		const mlopenTensorDescriptor_t		xDesc,
		const void							*x,
		const mlopenTensorDescriptor_t		wDesc,
		const void							*w,
		mlopenConvFwdAlgorithm_t			algo,
		const void							*beta,
		const mlopenTensorDescriptor_t		 yDesc,
		void								*y) {

	printf("To be implemented\n");

	if(handle == nullptr) {
		return mlopenStatusBadParm;
	}
	if(xDesc == nullptr || wDesc == nullptr || yDesc == nullptr) {
		return mlopenStatusBadParm;
	}
	if(x == nullptr || w == nullptr || y == nullptr) {
		return mlopenStatusBadParm;
	}
	if(xDesc->_CheckTensorDims(yDesc) != mlopenStatusSuccess || xDesc->_CheckTensorDims(wDesc) != mlopenStatusSuccess) {
		return mlopenStatusBadParm;
	}
	if(xDesc->_CheckTensorDataTypes(yDesc) != mlopenStatusSuccess || xDesc->_CheckTensorDataTypes(wDesc) != mlopenStatusSuccess) {
		return mlopenStatusBadParm;
	}
	if(xDesc->_dimA[0] != wDesc->_dimA[0]) {
		return mlopenStatusBadParm;
	}
	if(xDesc->_dims < 3) {
		return mlopenStatusBadParm;
	}

	std::string program_name; // CL kernel filename
	std::string kernel_name; // kernel name
	std::string parms; // kernel parameters

	mlopenStream_t queue;
	handle->GetStream(&queue);

#if MLOpen_BACKEND_OPENCL
	cl::Kernel kernel = KernelCache::get(reinterpret_cast<cl::CommandQueue&>(queue), program_name, kernel_name, parms); 
#endif

	return mlopenStatusSuccess;

}

