#include <pooling.hpp>
#include "mlo_internal.hpp"

mlopenStatus_t mlopenPoolingDescriptor::Forward(
		mlopenHandle_t						handle,
		const void							*alpha,
		const mlopenTensorDescriptor_t		xDesc,
		const cl_mem						x,
		const void							*beta,
		const mlopenTensorDescriptor_t		yDesc,
		cl_mem								y) {
	mlopenStatus_t status = mlopenStatusSuccess;
	printf("in pooling forward\n");
	mlo_construct_pooling2D construct_params(1); // forward

	std::string kernel_path = "../src/Kernels/";

	construct_params.setKernelPath(kernel_path);

	mlopenAcceleratorQueue_t queue;
	handle->GetStream(&queue);

	construct_params.setStream(queue);

	construct_params.setOutputDescFromMLDesc(yDesc);
	construct_params.setInputDescFromMLDesc(xDesc);
	mlopenPoolingMode_t mode = GetMode();
	const std::vector<int> & lengths = GetLengths();
	const std::vector<int> & strides = GetStrides();
	const std::vector<int> & pads = GetPads();
	int pooling_method = (mode == mlopenPoolingMax) ? MLO_POOLING_OP_MAX : MLO_POOLING_OP_AVE;
	construct_params.setPoolingDescr(pooling_method, lengths[0], lengths[1], pads[0], pads[1], strides[0], strides[1]);

	status = (mlopenStatus_t)construct_params.mloConstruct();

	std::string program_name = kernel_path + construct_params.getKernelFile();  // CL kernel filename
	std::string kernel_name = construct_params.getKernelName(); // kernel name
	std::string parms = construct_params.getCompilerOptions(); // kernel parameters

	std::string network_config;
	construct_params.mloBuildConf_Key(network_config);

	const std::vector<size_t> & vld = construct_params.getLocalWkSize();
	const std::vector<size_t> & vgd = construct_params.getGlobalWkSize();

	// Compile the kernel if not aleady compiled
	OCLKernel obj = KernelCache::get(queue,
		"mlopenPooling2dForward",
		network_config,
		program_name,
		kernel_name,
		vld,
		vgd,
		parms);

	std::string kernName;
	obj.GetKernelName(kernName);

	printf("kname: %s\n", kernName.c_str());

	// Set kernel arguments
	// Use proper arguments
	float padding_val = 0;
	obj.SetArgs(0, x, y);

	int dim = (int)vld.size();

	// Run the kernel
	obj.run(queue, dim, 0, vgd.data(), vld.data(), NULL);

	clFinish(queue);

	std::cout << "Pooling Finished !!" << std::endl;


	return(status);
}

mlopenStatus_t mlopenPoolingDescriptor::Backward(
		mlopenHandle_t						handle,
		const void							*alpha,
		const mlopenTensorDescriptor_t		yDesc,
		const cl_mem						y,
		const mlopenTensorDescriptor_t		dyDesc,
		const cl_mem						dy,
		const mlopenTensorDescriptor_t		xDesc,
		const cl_mem						x,
		const void							*beta,
		const mlopenTensorDescriptor_t		dxDesc,
		cl_mem								dx) {

	mlopenStatus_t status = mlopenStatusSuccess;
	printf("in pooling backward\n");
	return(status);
}

