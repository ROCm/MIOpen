#include <lrn.hpp>
#include "mlo_internal.hpp"

mlopenStatus_t mlopenLRNDescriptor::Forward(
		mlopenHandle_t						handle,
		const void							*alpha,
		const mlopenTensorDescriptor_t		xDesc,
		const Data_t						x,
		const void							*beta,
		const mlopenTensorDescriptor_t		yDesc,
		Data_t								y,
		bool                                do_backward,
		Data_t								workSpace,
		size_t								*workSpaceSize) {
	
	mlopenStatus_t status = mlopenStatusSuccess;
	printf("in lrn forward\n");
	mlo_construct_norm construct_params(1); // forward

	std::string kernel_path = "../src/Kernels/";

	construct_params.setKernelPath(kernel_path);

	mlopenAcceleratorQueue_t queue;
	handle->GetStream(&queue);

	construct_params.setStream(queue);

	{
		int nOut;
		int cOut;
		int hOut;
		int wOut;
		int nOutStride;
		int cOutStride;
		int hOutStride;
		int wOutStride;

		std::tie(nOut, cOut, hOut, wOut) = tie4(yDesc->GetLengths());
		std::tie(nOutStride, cOutStride, hOutStride, wOutStride) = tie4(yDesc->GetStrides());


		construct_params.setTopDescr(
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
	}

	{
		int nIn;
		int cIn;
		int hIn;
		int wIn;
		int nInStride;
		int cInStride;
		int hInStride;
		int wInStride;

		std::tie(nIn, cIn, hIn, wIn) = tie4(xDesc->GetLengths());
		std::tie(nInStride, cInStride, hInStride, wInStride) = tie4(xDesc->GetStrides());

		construct_params.setBotDescr(
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
	}
	

	int norm_reg = GetMode();

	int local_area = GetN();

	double lrn_alpha = GetAlpha();
	double lrn_beta = GetBeta();
	double lrn_K = GetK();

	construct_params.doBackward(do_backward);
	construct_params.setNormDescr(norm_reg, local_area, lrn_alpha, lrn_beta, lrn_K);

	status = (mlopenStatus_t)construct_params.mloConstruct();
	if (x == 0 || y == 0)
	{
		*workSpaceSize = construct_params.getWorkSpaceSzBytes();
	}
	else
	{

		std::string program_name = kernel_path + construct_params.getKernelFile();  // CL kernel filename
		std::string kernel_name = construct_params.getKernelName(); // kernel name
		std::string parms = construct_params.getCompilerOptions(); // kernel parameters

		std::string network_config;
		construct_params.mloBuildConf_Key(network_config);

		const std::vector<size_t> & vld = construct_params.getLocalWkSize();
		const std::vector<size_t> & vgd = construct_params.getGlobalWkSize();

		// Compile the kernel if not aleady compiled
		OCLKernel obj = KernelCache::get(queue,
			"mlopenLRNForward",
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

		int norm_region;
		int local_ar;
		// whithin channel alphaoverarea is going to be culculate based on actual areal size (cut by borders).
		double norm_alpha;
		double norm_beta;
		double norm_K;
		double norm_alphaoverarea;

		construct_params.getNormDescr(norm_region, local_ar, norm_alpha, norm_beta, norm_K, norm_alphaoverarea);
		float f_norm_alpha = (float)norm_alpha;
		float f_norm_beta = (float)norm_beta;
		float f_norm_K = (float)norm_K;
		float f_norm_alphaoverarea = (float)norm_alphaoverarea;
		/*
		* Kernel argument list:
		cl_mem bot_mem,
		cl_mem top_mem,
		if do_scale
		cl_mem scale_mem,
		eif
		float f_norm_alphaoverarea,
		float f_norm_alpha,
		float f_norm_beta,
		float f_norm_K
		*/
		if (do_backward)
		{
			obj.SetArgs(0, x, y, workSpace, f_norm_alphaoverarea, f_norm_alpha, f_norm_beta, f_norm_K);
		}
		else
		{
			obj.SetArgs(0, x, y, f_norm_alphaoverarea, f_norm_alpha, f_norm_beta, f_norm_K);
		}

		int dim = (int)vld.size();

		// Run the kernel
		obj.run(queue, dim, 0, vgd.data(), vld.data(), NULL);

		clFinish(queue);

		std::cout << "LRN Forward Finished !!" << std::endl;

	}
	return(status);
}

mlopenStatus_t mlopenLRNDescriptor :: Backward(
		mlopenHandle_t						handle,
		const void							*alpha,
		const mlopenTensorDescriptor_t		yDesc,
		const Data_t						y,
		const mlopenTensorDescriptor_t		dyDesc,
		const Data_t						dy,
		const mlopenTensorDescriptor_t		xDesc,
		const Data_t						x,
		const void							*beta,
		const mlopenTensorDescriptor_t		dxDesc,
		Data_t								dx,
		const Data_t						workSpace) {

	mlopenStatus_t status = mlopenStatusSuccess;
	printf("in lrn backward\n");
	return(status);
}

