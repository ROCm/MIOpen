#include "Convolution.hpp"
#include "mlo_internal.hpp"

mlopenStatus_t mlopenConvolutionDescriptor::FindConvFwdAlgorithm(mlopenHandle_t handle,
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
	
	if(handle == nullptr) {
		return mlopenStatusBadParm;
	}
	if(xDesc == nullptr || wDesc == nullptr || yDesc == nullptr) {
		return mlopenStatusBadParm;
	}
	if(x == nullptr || w == nullptr || y == nullptr) {
		return mlopenStatusBadParm;
	}
#if 0
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
		std::string kernel_path = "../src/Kernels/";

		construct_params.setKernelPath(kernel_path);

		std::string generic_comp_otions = std::string(" -I ") + kernel_path + " ";
//		if (debug)
		{
			// generic_comp_otions += std::string(" -cl-std=CL2.0 ");

		}

		construct_params.setGeneralCompOptions(generic_comp_otions);

		mlopenStream_t queue;
		handle->GetStream(&queue);

		construct_params.setStream(queue);

		output_sz = construct_params.setOutputDescFromMLDesc(yDesc);
		input_sz = construct_params.setInputDescFromMLDesc(xDesc);
		weights_sz = construct_params.setWeightDescFromMLDesc(wDesc);

		construct_params.setConvDescr(_pad_h, _pad_w, _u, _v, _upscalex, _upscaley);

		construct_params.mloConstructDirect2D();
	}

	std::string program_name = std::string("../src/Kernels/") +  construct_params.getKernelFile();  //"../src/Hello.cl"; // CL kernel filename
	std::string kernel_name = construct_params.getKernelName(); // "hello_world_kernel"; // kernel name
	std::string parms = construct_params.getCompilerOptions(); // kernel parameters

	std::string network_config;
	construct_params.mloBuildConf_Key(network_config);
	// Get the queue associated with this handle
	mlopenStream_t queue;
	handle->GetStream(&queue);

	const std::vector<size_t> & vld = construct_params.getLocalWkSize();
	const std::vector<size_t> & vgd = construct_params.getGlobalWkSize();

	// Compile the kernel if not aleady compiled
	OCLKernel obj = KernelCache::get(queue, 
			"mlopenConvolutionFwdAlgoDirect",
			network_config,
			program_name, 
			kernel_name,
			vld,
			vgd,
			parms);

	cl_int status;

	std::string kernName;
	obj.GetKernelName(kernName);

	printf("kname: %s\n", kernName.c_str());

#if 0 // Test to see if we can launch the kernel and get the results back

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
#endif
	// Set kernel arguments
	// Use proper arguments
	float padding_val = 0;
	obj.SetArgs(0, x, w, y, padding_val);

	int dim = (int)vld.size();
	
	// Run the kernel
	obj.run(queue, dim, 0, vgd.data(), vld.data(), NULL);
	
	clFinish(queue);

	std::cout << "Conv's finished." << std::endl;

	return mlopenStatusSuccess;

}

mlopenStatus_t mlopenConvolutionDescriptor::ConvolutionForward(mlopenHandle_t handle,
		const void							*alpha,
		const mlopenTensorDescriptor_t		xDesc,
		const cl_mem						x,
		const mlopenTensorDescriptor_t		wDesc,
		const cl_mem						w,
		mlopenConvFwdAlgorithm_t			algo,
		const void							*beta,
		const mlopenTensorDescriptor_t		yDesc,
		cl_mem								y, 
		void								*workSpace,
		size_t								workSpaceSize) {

	if(handle == nullptr) {
		return mlopenStatusBadParm;
	}
	if(xDesc == nullptr || wDesc == nullptr || yDesc == nullptr) {
		return mlopenStatusBadParm;
	}
	if(x == nullptr || w == nullptr || y == nullptr) {
		return mlopenStatusBadParm;
	}
	if(xDesc->GetSize() != yDesc->GetSize() || xDesc->GetSize() != wDesc->GetSize()) {
		return mlopenStatusBadParm;
	}
	if(xDesc->GetType() != yDesc->GetType() || xDesc->GetType() != wDesc->GetType()) {
		return mlopenStatusBadParm;
	}
	if(xDesc->GetLengths()[1] != wDesc->GetLengths()[1]) {
		return mlopenStatusBadParm;
	}
	if(xDesc->GetSize() < 3) {
		return mlopenStatusBadParm;
	}
	
	// TODO: Replicating code for now.
	size_t input_sz = 0;
	size_t output_sz = 0;
	size_t weights_sz = 0;
	
	mlo_construct_direct2D construct_params(1); // forward
	{
		output_sz = construct_params.setOutputDescFromMLDesc(yDesc);
		input_sz = construct_params.setInputDescFromMLDesc(xDesc);
		weights_sz = construct_params.setWeightDescFromMLDesc(wDesc);
	}

	std::string network_config;
	construct_params.mloBuildConf_Key(network_config);
	// Get the queue associated with this handle
	mlopenStream_t queue;
	handle->GetStream(&queue);

	OCLKernel kernel;
	switch(algo) {
		case mlopenConvolutionFwdAlgoDirect:
			 // Compile the kernel if not aleady compiled
			 kernel = KernelCache::get( "mlopenConvolutionFwdAlgoDirect",
					 network_config);
		break;

	}
	cl_int status;

	std::string kernName;
	kernel.GetKernelName(kernName);

	printf("kname: %s\n", kernName.c_str());

	// Set kernel arguments
	// Use proper arguments
	float padding_val = 0;
	//kernel.SetArgs(0, in_dev, wei_dev, out_dev, padding_val);
	kernel.SetArgs(0, x, w, y, padding_val);

	const std::vector<size_t> & vld = kernel.GetLocalDims();
	const std::vector<size_t> & vgd = kernel.GetGlobalDims();

	int dim = (int)vld.size();
	// Run the kernel
	kernel.run(queue, dim, 0, vgd.data(), vld.data(), NULL);
	
	clFinish(queue);

	std::cout << "Conv's (forward) finished." << std::endl;

	return mlopenStatusSuccess;

}

// FindBackwardDataAlgorithm()
//
mlopenStatus_t mlopenConvolutionDescriptor::FindConvBwdDataAlgorithm(mlopenHandle_t handle,
		const mlopenTensorDescriptor_t	dyDesc,
		const cl_mem					dy,
		const mlopenTensorDescriptor_t	wDesc,
		const cl_mem					w,
		const mlopenTensorDescriptor_t	dxDesc,
		const cl_mem					dx,
		const int						requestAlgoCount,
		int								*returnedAlgoCount,
		mlopenConvAlgoPerf_t			*perfResults,
		mlopenConvPreference_t			preference,
		void							*workSpace,
		size_t							workSpaceSize) {
	
	if(handle == nullptr) {
		return mlopenStatusBadParm;
	}
	if(dxDesc == nullptr || wDesc == nullptr || dyDesc == nullptr) {
		return mlopenStatusBadParm;
	}
	if(dx == nullptr || w == nullptr || dy == nullptr) {
		return mlopenStatusBadParm;
	}
#if 0
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

	mlo_construct_direct2D construct_params(0); // backward
	{
		construct_params.setTimerIter(100);
// HOW TO DEFINE???
		construct_params.doSearch(true); // false);
//
		construct_params.saveSearchRequest(true);


// TO DO WHERE IS THE PATH ?
		std::string kernel_path = "../src/Kernels/";

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

		output_sz = construct_params.setOutputDescFromMLDesc(dxDesc);
		input_sz = construct_params.setInputDescFromMLDesc(dyDesc);
		weights_sz = construct_params.setWeightDescFromMLDesc(wDesc);

		construct_params.setConvDescr(_pad_h, _pad_w, _u, _v, _upscalex, _upscaley);

		construct_params.mloConstructDirect2D();
	}

	std::string program_name = std::string("../src/Kernels/") +  construct_params.getKernelFile();  
	std::string kernel_name = construct_params.getKernelName(); // kernel name
	std::string parms = construct_params.getCompilerOptions(); // kernel parameters

	std::string network_config;
	construct_params.mloBuildConf_Key(network_config);
	// Get the queue associated with this handle
	mlopenStream_t queue;
	handle->GetStream(&queue);

	const std::vector<size_t> & vld = construct_params.getLocalWkSize();
	const std::vector<size_t> & vgd = construct_params.getGlobalWkSize();

	// Compile the kernel if not aleady compiled
	OCLKernel obj = KernelCache::get(queue, 
			"mlopenConvolutionBwdDataAlgo_0",
			network_config,
			program_name, 
			kernel_name,
			vld,
			vgd,
			parms);

	cl_int status;

	std::string kernName;
	obj.GetKernelName(kernName);

	printf("kname: %s\n", kernName.c_str());

	// Set kernel arguments
	// Use proper arguments
	float padding_val = 0;
	obj.SetArgs(0, dy, w, dx, padding_val);

	int dim = (int)vld.size();
	
	// Run the kernel
	obj.run(queue, dim, 0, vgd.data(), vld.data(), NULL);
	
	clFinish(queue);

	std::cout << "Backward Conv's finished." << std::endl;

	return mlopenStatusSuccess;

}

// BackwardDataAlgorithm()
mlopenStatus_t mlopenConvolutionDescriptor::ConvolutionBackwardData(mlopenHandle_t handle,
		const void							*alpha,
		const mlopenTensorDescriptor_t		dyDesc,
		const cl_mem						dy,
		const mlopenTensorDescriptor_t		wDesc,
		const cl_mem						w,
		mlopenConvBwdDataAlgorithm_t		algo,
		const void							*beta,
		const mlopenTensorDescriptor_t		dxDesc,
		cl_mem								dx, 
		void								*workSpace,
		size_t								workSpaceSize) {

	if(handle == nullptr) {
		return mlopenStatusBadParm;
	}
	if(dxDesc == nullptr || wDesc == nullptr || dyDesc == nullptr) {
		return mlopenStatusBadParm;
	}
	if(dx == nullptr || w == nullptr || dy == nullptr) {
		return mlopenStatusBadParm;
	}
	if(dyDesc->GetSize() != dxDesc->GetSize() || dyDesc->GetSize() != wDesc->GetSize()) {
		return mlopenStatusBadParm;
	}
	if(dyDesc->GetType() != dxDesc->GetType() || dyDesc->GetType() != wDesc->GetType()) {
		return mlopenStatusBadParm;
	}
	if(dyDesc->GetLengths()[1] != wDesc->GetLengths()[1]) {
		return mlopenStatusBadParm;
	}
	if(dyDesc->GetSize() < 3) {
		return mlopenStatusBadParm;
	}

	// TODO: Replicating code for now.
	size_t input_sz = 0;
	size_t output_sz = 0;
	size_t weights_sz = 0;
	
	mlo_construct_direct2D construct_params(0); // backward
	{
		output_sz = construct_params.setOutputDescFromMLDesc(dxDesc);
		input_sz = construct_params.setInputDescFromMLDesc(dyDesc);
		weights_sz = construct_params.setWeightDescFromMLDesc(wDesc);
	}

	std::string network_config;
	construct_params.mloBuildConf_Key(network_config);
	// Get the queue associated with this handle
	mlopenStream_t queue;
	handle->GetStream(&queue);

	OCLKernel kernel;
	switch(algo) {
		case mlopenConvolutionBwdDataAlgo_0:
			 // Compile the kernel if not aleady compiled
			 kernel = KernelCache::get( "mlopenConvolutionBwdDataAlgo_0",
					 network_config);
//			 if(!kernel) {printf("kenrel not found in hash table\n");
		break;
		default:
			printf("Algorithm not found\n");
		break;

	}
	cl_int status;

	std::string kernName;
	kernel.GetKernelName(kernName);

	printf("kname: %s\n", kernName.c_str());

	// Set kernel arguments
	// Use proper arguments
	float padding_val = 0;
	kernel.SetArgs(0, dy, w, dx, padding_val);

	const std::vector<size_t> & vld = kernel.GetLocalDims();
	const std::vector<size_t> & vgd = kernel.GetGlobalDims();

	int dim = (int)vld.size();
	// Run the kernel
	kernel.run(queue, dim, 0, vgd.data(), vld.data(), NULL);
	
	clFinish(queue);

	std::cout << "Conv's (backward) finished." << std::endl;

	return mlopenStatusSuccess;

}
