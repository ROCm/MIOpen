#include <mlopen/lrn.hpp>
#include <mlopen/mlo_internal.hpp>

namespace mlopen {

mlopenStatus_t LRNDescriptor::Forward(
		Handle						&handle,
		const void					* /*alpha*/,
		const TensorDescriptor		&xDesc,
		ConstData_t				x,
		const void					* /*beta*/,
		const TensorDescriptor		&yDesc,
		Data_t						y,
		bool                        do_backward,
		Data_t						workSpace) {

	mlopenStatus_t status = mlopenStatusSuccess;
	mlo_construct_norm construct_params(1); // forward

	construct_params.setStream(&handle);

	int nOut;
	int cOut;
	int hOut;
	int wOut;
	int nOutStride;
	int cOutStride;
	int hOutStride;
	int wOutStride;

	std::tie(nOut, cOut, hOut, wOut) = tie4(yDesc.GetLengths());
	std::tie(nOutStride, cOutStride, hOutStride, wOutStride) = tie4(yDesc.GetStrides());

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
	int nIn;
	int cIn;
	int hIn;
	int wIn;
	int nInStride;
	int cInStride;
	int hInStride;
	int wInStride;

	std::tie(nIn, cIn, hIn, wIn) = tie4(xDesc.GetLengths());
	std::tie(nInStride, cInStride, hInStride, wInStride) = tie4(xDesc.GetStrides());

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

	int norm_reg = GetMode();
	int local_area = GetN();
	double lrn_alpha = GetAlpha();
	double lrn_beta = GetBeta();
	double lrn_K = GetK();

	construct_params.doBackward(do_backward);
	construct_params.setNormDescr(norm_reg, local_area, lrn_alpha, lrn_beta, lrn_K);

	status = static_cast<mlopenStatus_t>(construct_params.mloConstruct());

	std::string program_name = construct_params.getKernelFile();  // CL kernel filename
	std::string kernel_name = construct_params.getKernelName(); // kernel name
	const std::string& compiler_parms = construct_params.getCompilerOptions(); // kernel parameters

	std::string network_config;
	construct_params.mloBuildConf_Key(network_config);

	const std::vector<size_t> & vld = construct_params.getLocalWkSize();
	const std::vector<size_t> & vgd = construct_params.getGlobalWkSize();

	int norm_region;
	int local_ar;
	// whithin channel alphaoverarea is going to be culculate based on actual areal size (cut by borders).
	double norm_alpha;
	double norm_beta;
	double norm_K;
	double norm_alphaoverarea;

	construct_params.getNormDescr(norm_region, local_ar, norm_alpha, norm_beta, norm_K, norm_alphaoverarea);
	float f_norm_alpha = static_cast<float>(norm_alpha);
	float f_norm_beta = static_cast<float>(norm_beta);
	float f_norm_K = static_cast<float>(norm_K);
	float f_norm_alphaoverarea = static_cast<float>(norm_alphaoverarea);

	KernelInvoke obj = 	handle.GetKernel("mlopenLRNForward",
				network_config,
				program_name,
				kernel_name,
				vld,
				vgd,
				compiler_parms);

	if (do_backward)
	{
		obj(x, y, workSpace, f_norm_alphaoverarea, f_norm_alpha, f_norm_beta, f_norm_K);
	}
	else
	{
		obj(x, y, f_norm_alphaoverarea, f_norm_alpha, f_norm_beta, f_norm_K);
	}

	return(status);
}

mlopenStatus_t LRNDescriptor :: Backward(
		Handle						&handle,
		const void					* /*alpha*/,
		const TensorDescriptor		&yDesc,
		ConstData_t		  		y,
		const TensorDescriptor		&dyDesc,
		ConstData_t		  		dy,
		const TensorDescriptor		&xDesc,
		ConstData_t		  		x,
		const void			  		* /*beta*/,
		const TensorDescriptor		&dxDesc,
		Data_t						dx,
		ConstData_t				workSpace) {

	mlopenStatus_t status = mlopenStatusSuccess;
	mlo_construct_norm construct_params(0); // backward

	construct_params.setStream(&handle);
	int ndOut;
	int cdOut;
	int hdOut;
	int wdOut;
	int ndOutStride;
	int cdOutStride;
	int hdOutStride;
	int wdOutStride;

	std::tie(ndOut, cdOut, hdOut, wdOut) = tie4(dyDesc.GetLengths());
	std::tie(ndOutStride, cdOutStride, hdOutStride, wdOutStride) = tie4(dyDesc.GetStrides());

	construct_params.setTopDfDescr(
			"NCHW",
			"FP32",
			ndOut,
			cdOut,
			hdOut,
			wdOut,
			ndOutStride,
			cdOutStride,
			hdOutStride,
			wdOutStride);

	int nOut;
	int cOut;
	int hOut;
	int wOut;
	int nOutStride;
	int cOutStride;
	int hOutStride;
	int wOutStride;

	std::tie(nOut, cOut, hOut, wOut) = tie4(yDesc.GetLengths());
	std::tie(nOutStride, cOutStride, hOutStride, wOutStride) = tie4(yDesc.GetStrides());

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

	int ndIn;
	int cdIn;
	int hdIn;
	int wdIn;
	int ndInStride;
	int cdInStride;
	int hdInStride;
	int wdInStride;

	std::tie(ndIn, cdIn, hdIn, wdIn) = tie4(dxDesc.GetLengths());
	std::tie(ndInStride, cdInStride, hdInStride, wdInStride) = tie4(dxDesc.GetStrides());

	construct_params.setBotDfDescr(
			"NCHW",
			"FP32",
			ndIn,
			cdIn,
			hdIn,
			wdIn,
			ndInStride,
			cdInStride,
			hdInStride,
			wdInStride);

	int nIn;
	int cIn;
	int hIn;
	int wIn;
	int nInStride;
	int cInStride;
	int hInStride;
	int wInStride;

	std::tie(nIn, cIn, hIn, wIn) = tie4(xDesc.GetLengths());
	std::tie(nInStride, cInStride, hInStride, wInStride) = tie4(xDesc.GetStrides());

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

	int norm_reg = GetMode();

	int local_area = GetN();

	double lrn_alpha = GetAlpha();
	double lrn_beta = GetBeta();
	double lrn_K = GetK();

	construct_params.setNormDescr(norm_reg, local_area, lrn_alpha, lrn_beta, lrn_K);

	status = static_cast<mlopenStatus_t>(construct_params.mloConstruct());

	std::string program_name = construct_params.getKernelFile();  // CL kernel filename
	std::string kernel_name = construct_params.getKernelName(); // kernel name
	std::string compiler_parms = construct_params.getCompilerOptions(); // kernel parameters

	std::string network_config;
	construct_params.mloBuildConf_Key(network_config);

	const std::vector<size_t> & vld = construct_params.getLocalWkSize();
	const std::vector<size_t> & vgd = construct_params.getGlobalWkSize();

	int norm_region;
	int local_ar;
	// whithin channel alphaoverarea is going to be culculate based on actual areal size (cut by borders).
	double norm_alpha;
	double norm_beta;
	double norm_K;
	double norm_alphaoverarea;

	construct_params.getNormDescr(norm_region, local_ar, norm_alpha, norm_beta, norm_K, norm_alphaoverarea);
	float f_norm_alpha = static_cast<float>(norm_alpha);
	float f_norm_beta = static_cast<float>(norm_beta);
	float f_norm_ratio = static_cast<float>(2. *norm_alpha * norm_beta / static_cast<double>(local_ar));

	handle.GetKernel("mlopenLRNBackward",
			network_config,
			program_name,
			kernel_name,
			vld,
			vgd,
			compiler_parms)(y, x, dy, workSpace, dx, f_norm_ratio, f_norm_alpha, f_norm_beta);


	return(status);
}
} // namespace mlopen
