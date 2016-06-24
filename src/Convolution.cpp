#include "Convolution.hpp"

mlopenConvolutionDescriptor::mlopenConvolutionDescriptor() : _pad_h(0), _pad_w(0), _u(1), _v(1), _upscalex(0), _upscaley(0) {
	printf("In convolution Ctor\n");
	_mode = mlopenConvolution;
}

mlopenStatus_t mlopenConvolutionDescriptor::GetForwardOutputDim(const mlopenTensorDescriptor_t inputTensorDesc,
			const mlopenTensorDescriptor_t filterDesc,
			int *n,
			int *c,
			int *h, 
			int *w) {
	
	printf("To be implemented (GetForwardOutputDim)\n");

	return mlopenStatusSuccess;
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
	float *a = new float[1024];
	float *b = new float[1024];
	float *c = new float[1024];

	for(int i = 0; i < 1024; i++) {
		a[i] = 1.0;
		b[i] = 7.0;
		c[i] = 0.0;
	}
	int sz = 1024;

	cl_context ctx;
	clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, NULL);

	cl_mem adev = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 4*sz,NULL, &status);
	if(status != CL_SUCCESS) {
		printf("error %d\n", status);
	}
	cl_mem bdev = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 4*sz,NULL, NULL);
	cl_mem cdev = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4*sz,NULL, NULL);

	status = clEnqueueWriteBuffer(queue, adev, CL_TRUE, 0, 4*sz, a, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(queue, bdev, CL_TRUE, 0, 4*sz, b, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(queue, cdev, CL_TRUE, 0, 4*sz, c, 0, NULL, NULL);

	// Set kernel arguments
	obj.SetArgs(0, adev, bdev, cdev, sz);

	size_t gd = 1024;
	size_t ld = 256;

	// Run the kernel
	obj.run(queue, 1, 0, gd, ld);

	clFinish(queue);
	clEnqueueReadBuffer(queue, cdev, CL_TRUE, 0, 4*sz, c, 0, NULL, NULL);

	float sum = 0.0;
	for(int i = 0; i < 1024; i++) {
		b[i] = 6;
		sum += c[i];
	}

	printf("\nsum %f\n, ", sum);
	sum = 0.0;
	status |= clEnqueueWriteBuffer(queue, bdev, CL_TRUE, 0, 4*sz, b, 0, NULL, NULL);

	getchar();

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

