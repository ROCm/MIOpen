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
	size_t input_sz = 0;
	size_t output_sz = 0;
	size_t weights_sz = 0;

	mlo_construct_direct2D construct_params(1); // forward
	{
		construct_params.setTimerIter(100);
// HOW TO DEFINE???
		construct_params.doSearch(true); // false);
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


		construct_params.setOutputDescr(
			"NCHW",
			"FP32",
			nOut,
			cOut,
			hOut,
			wOut,
			nOutStride,
			cOutStride,
			hOutStride,
			wOutStride);


		output_sz = nOut * cOut * hOut * wOut * sizeof(float);

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

		construct_params.setInputDescr(
			"NCHW",
			"FP32",
			nIn,
			cIn,
			hIn,
			wIn,
			nInStride,
			cInStride,
			hInStride,
			wInStride);
		input_sz = nIn * cIn * hIn * wIn * sizeof(float);

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


		construct_params.setWeightsDescr(
			"NCHW",
			"FP32",
			nWei,
			cWei,
			hWei,
			wWei,
			nWeiStride,
			cWeiStride,
			hWeiStride,
			wWeiStride
			);

		weights_sz = nWei * cWei * hWei * wWei * sizeof(float);

		construct_params.setConvDescr(_pad_h, _pad_w, _u, _v, _upscalex, _upscaley);

		construct_params.mloConstructDirect2D();
	}

	std::string program_name = std::string("../src/") +  construct_params.getKernelFile();  //"../src/Hello.cl"; // CL kernel filename
	std::string kernel_name = construct_params.getKernelName(); // "hello_world_kernel"; // kernel name
	std::string parms = construct_params.getCompilerOptions(); // kernel parameters

	// Get the queue associated with this handle
	mlopenStream_t queue;
	handle->GetStream(&queue);

	// Compile the kernel if not aleady compiled
	OCLKernel obj = KernelCache::get(queue, program_name, kernel_name, parms);
	cl_int status;

	std::string kernName;
	obj.GetKernelName(kernName);

	printf("kname: %s\n", kernName.c_str());

#if 1 // Test to see if we can launch the kernel and get the results back

	float * in_sys = new float[input_sz];
	float * wei_sys = new float[weights_sz];
	float * out_sys = new float[output_sz];

	for(int i = 0; i < input_sz; i++) {
		in_sys[i] = rand() * (1.0 / RAND_MAX);
	}
	for (int i = 0; i < weights_sz; i++) {
		wei_sys[i] = (double)(rand() * (1.0 / RAND_MAX) - 0.5) * 0.001;
	}

	cl_context ctx;
	clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, NULL);

	cl_mem in_dev = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input_sz, in_sys, &status);
	if(status != CL_SUCCESS) {
		printf("error %d\n", status);
	}
	cl_mem wei_dev = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_sz,wei_sys, NULL);
	cl_mem out_dev = clCreateBuffer(ctx, CL_MEM_READ_WRITE, output_sz,NULL, NULL);

//	status = clEnqueueWriteBuffer(queue, adev, CL_TRUE, 0, 4*sz, a, 0, NULL, NULL);
//	status |= clEnqueueWriteBuffer(queue, bdev, CL_TRUE, 0, 4*sz, b, 0, NULL, NULL);
//	status |= clEnqueueWriteBuffer(queue, cdev, CL_TRUE, 0, 4*sz, c, 0, NULL, NULL);


	// Set kernel arguments
	// Use proper arguments
	float padding_val = 0;
	obj.SetArgs(0, in_dev, wei_dev, out_dev, padding_val);

	const std::vector<size_t> & vld = construct_params.getLocalWkSize();

	const std::vector<size_t> & vgd = construct_params.getGlobalWkSize();

	int dim = (int)vld.size();
	size_t * gd = new size_t[dim];
	size_t * ld = new size_t[dim];

	for (int i = 0; i < dim; ++i)
	{
		gd[i] = vgd[i];
		ld[i] = vld[i];
	}
	// Run the kernel
	obj.run(queue, dim, 0, gd, ld);
	
	delete[] gd;
	delete[] ld;
	clFinish(queue);

	std::cout << "Conv's finished." << std::endl;
#endif // Test

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
