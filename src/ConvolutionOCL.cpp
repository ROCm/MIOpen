#include "Convolution.hpp"
#include "mlo_internal.hpp"

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
		size_t							workSpaceSize) {
	
	printf("To be implemented (FindConvFwdAlgo) \n");
#if 0
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
#endif 

	// Generate kernels if OpenCL
	// Compile, cache kernels, etc.
	// Launch all kernels and store the perf, workspace limits, etc.

	mlo_construct_direct2D construct_params(1); // forward
	{

		construct_params.setTimerIter(100);
// no bias
		construct_params.doBias(0);
// HOW TO DEFINE???
		construct_params.doSearch(true);
//
		construct_params.saveSearchRequest(true);


// TO DO WHERE IS THE PATH ?
		std::string kernel_path = "../src";

		construct_params.setKernelPath(kernel_path);

		std::string generic_comp_otions = std::string(" -I ") + kernel_path + " ";
//		if (debug)
		{
			generic_comp_otions += std::string(" -cl-std=CL2.0 ");

		}

		construct_params.setGeneralCompOptions(generic_comp_otions);

		mlopenStream_t queue;
		handle->GetStream(&queue);

		construct_params.setStream(queue);

		int nOut;
		int cOut;
		int hOut;
		int wOut;
		int nOutStride;
		int cOutStride;
		int hOutStride;
		int wOutStride;

		yDesc->Get4Dims(&nOut, &cOut, &hOut, &wOut);

		yDesc->Get4Strides(&nOutStride,
			&cOutStride,
			&hOutStride,
			&wOutStride);

// TO DO: Generalize data types
		size_t out_sz = nOutStride * cOutStride * hOutStride *wOutStride
			* sizeof(float);

		construct_params.setOutputDescr(hOut,
										wOut,
										cOut,
										hOutStride,
										cOutStride,
										nOutStride,
										out_sz,
										"NCHW",
										"FP32");

		int nIn;
		int cIn;
		int hIn;
		int wIn;
		int nInStride;
		int cInStride;
		int hInStride;
		int wInStride;

		xDesc->Get4Dims(&nIn, &cIn, &hIn, &wIn);

		yDesc->Get4Strides(&nInStride,
			&cInStride,
			&hInStride,
			&wInStride);

		// TO DO: Generalize data types
		size_t in_sz = nInStride * cInStride * hInStride *wInStride
			* sizeof(float);

		construct_params.setInputDescr(hIn,
			wIn,
			cIn,
			hInStride,
			cInStride,
			nInStride,
			in_sz,
			"NCHW",
			"FP32");

		construct_params.setBatchSize(nIn);

		int nWei;
		int cWei;
		int hWei;
		int wWei;
		int nWeiStride;
		int cWeiStride;
		int hWeiStride;
		int wWeiStride;

		wDesc->Get4Dims(&nWei, &cWei, &hWei, &wWei);

		wDesc->Get4Strides(&nWeiStride,
			&cWeiStride,
			&hWeiStride,
			&wWeiStride);

		// TO DO: Generalize data types
		size_t weights_sz = nWeiStride * cWeiStride * hWeiStride *wWeiStride
			* sizeof(float);

		construct_params.setKernelDescr(0, wWei, _pad_w, _u);
		construct_params.setKernelDescr(1, hWei, _pad_h, _v);

		construct_params.setWeightsSz(weights_sz);
		construct_params.setBiasSz(0);

//		construct_params.mloConstructDirect2D();

	}

	std::string program_name = "../src/Hello.cl"; // CL kernel filename
	std::string kernel_name = "hello_world_kernel"; // kernel name
	std::string parms; // kernel parameters

	// Get the queue associated with this handle
	mlopenStream_t queue;
	handle->GetStream(&queue);

	// Compile the kernel if not aleady compiled
	OCLKernel obj = KernelCache::get(queue, program_name, kernel_name);
	cl_int status;

	std::string kernName;
	obj.GetKernelName(kernName);

	printf("kname: %s\n", kernName.c_str());

	// Set kernel arguments
	// Use proper arguments

	int sz = 1024;
	obj.SetArgs(0, x, w, y, sz);

	size_t gd = 1024;
	size_t ld = 256;

	// Run the kernel
	obj.run(queue, 1, 0, gd, ld);

	clFinish(queue);

	return mlopenStatusSuccess;

}

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
		cl_mem								y) {

	printf("To be implemented (ConvolutionForward) \n");

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
//	OCLKernel kernel = KernelCache::get(queue, program_name, kernel_name);
#endif
	return mlopenStatusSuccess;

}

#endif
