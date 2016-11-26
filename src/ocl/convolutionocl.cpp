#include <mlopen/convolution.hpp>
#include <mlopen/util.hpp>
#include <mlopen/mlo_internal.hpp>

namespace mlopen {

void ConvolutionDescriptor::FindConvFwdAlgorithm(Handle& handle,
		const TensorDescriptor&		xDesc,
		const cl_mem				x,
		const TensorDescriptor&		wDesc,
		const cl_mem				w,
		const TensorDescriptor&		yDesc,
		const cl_mem				y,
		const int					 /*requestAlgoCount*/,
		int							* /*returnedAlgoCount*/,
		mlopenConvAlgoPerf_t		* /*perfResults*/,
		mlopenConvPreference_t		 /*preference*/,
		void						* /*workSpace*/,
		size_t						 /*workSpaceSize*/,
		bool						exhaustiveSearch) const {
	
	if(x == nullptr || w == nullptr || y == nullptr) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}
#if 0
	if(returnedAlgoCount == nullptr || perfResults == nullptr) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}
	if(requestAlgoCount < 1) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}
#endif 

	// Generate kernels if OpenCL
	// Compile, cache kernels, etc.
	// Launch all kernels and store the perf, workspace limits, etc.
	mlo_construct_direct2D construct_params(1); // forward
	{
		construct_params.setTimerIter(100);
		construct_params.doSearch(exhaustiveSearch);
		construct_params.saveSearchRequest(true);

		construct_params.setGeneralCompOptions("");

		construct_params.setStream(handle.GetStream());

		construct_params.setOutputDescFromMLDesc(yDesc);
		construct_params.setInputDescFromMLDesc(xDesc);
		construct_params.setWeightDescFromMLDesc(wDesc);

		construct_params.setConvDescr(pad_h, pad_w, u, v, upscalex, upscaley);

		construct_params.mloConstruct();
	}

	std::string program_name = construct_params.getKernelFile();  //"../src/Hello.cl"; // CL kernel filename
	std::string kernel_name = construct_params.getKernelName(); // "hello_world_kernel"; // kernel name
	std::string parms = construct_params.getCompilerOptions(); // kernel parameters

	std::string network_config;
	construct_params.mloBuildConf_Key(network_config);

	const std::vector<size_t> & vld = construct_params.getLocalWkSize();
	const std::vector<size_t> & vgd = construct_params.getGlobalWkSize();

	float padding_val = 0;
	handle.GetKernel("mlopenConvolutionFwdAlgoDirect",
			network_config,
			program_name, 
			kernel_name,
			vld,
			vgd,
			parms)(x, w, y, padding_val);
}

void ConvolutionDescriptor::ConvolutionForward(Handle& handle,
		const void					* /*alpha*/,
		const TensorDescriptor&		xDesc,
		const cl_mem				x,
		const TensorDescriptor&		wDesc,
		const cl_mem				w,
		mlopenConvFwdAlgorithm_t	algo,
		const void					* /*beta*/,
		const TensorDescriptor&		yDesc,
		cl_mem						y, 
		void						* /*workSpace*/,
		size_t						 /*workSpaceSize*/) const {

	if(x == nullptr || w == nullptr || y == nullptr) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}
	if(xDesc.GetSize() != yDesc.GetSize() || xDesc.GetSize() != wDesc.GetSize()) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}
	if(xDesc.GetType() != yDesc.GetType() || xDesc.GetType() != wDesc.GetType()) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}
	if(xDesc.GetLengths()[1] != wDesc.GetLengths()[1]) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}
	if(xDesc.GetSize() < 3) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}
	
	// TODO(paul): Replicating code for now.
	mlo_construct_direct2D construct_params(1); // forward
	{
		construct_params.setOutputDescFromMLDesc(yDesc);
		construct_params.setInputDescFromMLDesc(xDesc);
		construct_params.setWeightDescFromMLDesc(wDesc);
	}

	std::string network_config;
	construct_params.mloBuildConf_Key(network_config);

	std::string algorithm_name;
	switch(algo) {
		case mlopenConvolutionFwdAlgoDirect:
			algorithm_name = "mlopenConvolutionFwdAlgoDirect";
			break;
		case mlopenConvolutionFwdAlgoGEMM:
			algorithm_name = "mlopenConvolutionFwdAlgoGEMM";
			break;
		case mlopenConvolutionFwdAlgoFFT:
			algorithm_name = "mlopenConvolutionFwdAlgoFFT";
			break;
		case mlopenConvolutionFwdAlgoWinograd:
			algorithm_name = "mlopenConvolutionFwdAlgoWinograd";
			break;
	}

	float padding_val = 0;
	handle.GetKernel(algorithm_name, network_config)(x, w, y, padding_val);
}

// FindBackwardDataAlgorithm()
//
void ConvolutionDescriptor::FindConvBwdDataAlgorithm(Handle& handle,
		const TensorDescriptor&		dyDesc,
		const cl_mem				dy,
		const TensorDescriptor&		wDesc,
		const cl_mem				w,
		const TensorDescriptor&		dxDesc,
		const cl_mem				dx,
		const int					 /*requestAlgoCount*/,
		int							* /*returnedAlgoCount*/,
		mlopenConvAlgoPerf_t		* /*perfResults*/,
		mlopenConvPreference_t		 /*preference*/,
		void						* /*workSpace*/,
		size_t						 /*workSpaceSize*/,
		bool						exhaustiveSearch) const {
	
	if(dx == nullptr || w == nullptr || dy == nullptr) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}
#if 0
	if(returnedAlgoCount == nullptr || perfResults == nullptr) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}
	if(requestAlgoCount < 1) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}
#endif 

	if (u > 1 || v > 1)
	{
		printf("Algorithm has not been implemented\n");
	}
	else
	{
		// Generate kernels if OpenCL
		// Compile, cache kernels, etc.
		// Launch all kernels and store the perf, workspace limits, etc.
		mlo_construct_direct2D construct_params(0); // backward
		{
			construct_params.setTimerIter(100);
			construct_params.doSearch(exhaustiveSearch);
			construct_params.saveSearchRequest(true);

			construct_params.setGeneralCompOptions("");

			construct_params.setStream(handle.GetStream());

			construct_params.setOutputDescFromMLDesc(dyDesc);
			construct_params.setInputDescFromMLDesc(dxDesc);
			construct_params.setWeightDescFromMLDesc(wDesc);

			construct_params.setConvDescr(pad_h, pad_w, u, v, upscalex, upscaley);

			construct_params.mloConstruct();
		}

		std::string program_name = construct_params.getKernelFile();
		std::string kernel_name = construct_params.getKernelName(); // kernel name
		std::string parms = construct_params.getCompilerOptions(); // kernel parameters

		std::string network_config;
		construct_params.mloBuildConf_Key(network_config);

		const std::vector<size_t> & vld = construct_params.getLocalWkSize();
		const std::vector<size_t> & vgd = construct_params.getGlobalWkSize();

		float padding_val = 0;
		handle.GetKernel("mlopenConvolutionBwdDataAlgo_0",
			network_config,
			program_name,
			kernel_name,
			vld,
			vgd,
			parms)(dy, w, dx, padding_val);
	}
}

// BackwardDataAlgorithm()
void ConvolutionDescriptor::ConvolutionBackwardData(Handle& handle,
		const void						* /*alpha*/,
		const TensorDescriptor&			dyDesc,
		const cl_mem					dy,
		const TensorDescriptor&			wDesc,
		const cl_mem					w,
		mlopenConvBwdDataAlgorithm_t	/* algo */,
		const void						* /*beta*/,
		const TensorDescriptor&			dxDesc,
		cl_mem							dx, 
		void							* /*workSpace*/,
		size_t							 /*workSpaceSize*/) const {

	if(dx == nullptr || w == nullptr || dy == nullptr) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}
	if(dyDesc.GetSize() != dxDesc.GetSize() || dyDesc.GetSize() != wDesc.GetSize()) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}
	if(dyDesc.GetType() != dxDesc.GetType() || dyDesc.GetType() != wDesc.GetType()) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}
	if(dyDesc.GetLengths()[1] != wDesc.GetLengths()[0]) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}
	if(dyDesc.GetSize() < 3) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}

#if 0
	// TODO(paul): Replicating code for now.
	mlo_construct_direct2D construct_params(0); // backward
	{
		construct_params.setOutputDescFromMLDesc(dxDesc);
		construct_params.setInputDescFromMLDesc(dyDesc);
		construct_params.setWeightDescFromMLDesc(wDesc);
	}

	std::string network_config;
	construct_params.mloBuildConf_Key(network_config);

	std::string algorithm_name;
	switch(algo) {
		case mlopenConvolutionBwdDataAlgo_0:
			 algorithm_name = "mlopenConvolutionBwdDataAlgo_0";
		break;
	}
	float padding_val = 0;
	handle.GetKernel(algorithm_name, network_config)(dy, w, dx, padding_val);

#else

	{
		// Generate kernels if OpenCL
		// Compile, cache kernels, etc.
		// Launch all kernels and store the perf, workspace limits, etc.
		mlo_construct_direct2D construct_params(0); // backward
		{
			construct_params.doSearch(false);

			construct_params.setGeneralCompOptions("");

			construct_params.setStream(handle.GetStream());

			construct_params.setOutputDescFromMLDesc(dyDesc);
			construct_params.setInputDescFromMLDesc(dxDesc);
			construct_params.setWeightDescFromMLDesc(wDesc);

			construct_params.setConvDescr(pad_h, pad_w, u, v, upscalex, upscaley);

			construct_params.mloConstruct();
		}

		std::string program_name = construct_params.getKernelFile();
		std::string kernel_name = construct_params.getKernelName(); // kernel name
		std::string parms = construct_params.getCompilerOptions(); // kernel parameters

		std::string network_config;
		construct_params.mloBuildConf_Key(network_config);

		const std::vector<size_t> & vld = construct_params.getLocalWkSize();
		const std::vector<size_t> & vgd = construct_params.getGlobalWkSize();

		float padding_val = 0;
		handle.GetKernel("mlopenConvolutionBwdDataAlgo_0",
			network_config,
			program_name,
			kernel_name,
			vld,
			vgd,
			parms)(dy, w, dx, padding_val);
#endif
	}
}

// FindBackwardWeightsAlgorithm()
//
void ConvolutionDescriptor::FindConvBwdWeightsAlgorithm(Handle& handle,
		const TensorDescriptor&		dyDesc,
		const cl_mem				dy,
		const TensorDescriptor&		xDesc,
		const cl_mem				x,
		const TensorDescriptor&		dwDesc,
		const cl_mem				dw,
		const int					 /*requestAlgoCount*/,
		int							* /*returnedAlgoCount*/,
		mlopenConvAlgoPerf_t		* /*perfResults*/,
		mlopenConvPreference_t		 /*preference*/,
		cl_mem						workSpace,
		size_t						/*workSpaceSize*/,
		bool						/*exhaustiveSearch*/) const {
	
	if(x == nullptr || dw == nullptr || dy == nullptr) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}

	int in_n, in_c, in_h, in_w;
	std::tie(in_n, in_c, in_h, in_w) = tie4(xDesc.GetLengths());

	int wei_n, wei_h, wei_w;
	std::tie(wei_n, std::ignore, wei_h, wei_w) = tie4(dwDesc.GetLengths());

	int out_h, out_w;
	std::tie(std::ignore, std::ignore, out_h, out_w) = tie4(dyDesc.GetLengths());

//	int M = in_c * wei_h * wei_w;
//	int N = wei_n;
//	int K = out_h * out_w;
//	float alpha = 1.;
//	float beta = 1.;

	for(int i = 0; i < in_n; i++) {
		size_t in_offset = i * in_c * in_h * in_w;
		Im2ColGPU(handle, x, in_offset, in_c, in_h, in_w, wei_h, wei_w, out_h, out_w, pad_h, pad_w, v, u, workSpace);

	}
}

// BackwardWeightsAlgorithm()
void ConvolutionDescriptor::ConvolutionBackwardWeights(Handle& handle,
		const void						* /*alpha*/,
		const TensorDescriptor&			dyDesc,
		const cl_mem					dy,
		const TensorDescriptor&			xDesc,
		const cl_mem					x,
		mlopenConvBwdWeightsAlgorithm_t	/* algo */,
		const void						* /*beta*/,
		const TensorDescriptor&			dwDesc,
		cl_mem							dw, 
		cl_mem							workSpace,
		size_t							/*workSpaceSize*/) const {

	if(x == nullptr || dw == nullptr || dy == nullptr) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}
	if(dyDesc.GetSize() != dwDesc.GetSize() || dyDesc.GetSize() != xDesc.GetSize()) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}
	if(dyDesc.GetType() != dwDesc.GetType() || dyDesc.GetType() != xDesc.GetType()) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}
	if(dyDesc.GetLengths()[0] != xDesc.GetLengths()[0]) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}
	if(dyDesc.GetSize() < 3) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}

	int in_n, in_c, in_h, in_w;
	std::tie(in_n, in_c, in_h, in_w) = tie4(xDesc.GetLengths());

	int wei_n, wei_h, wei_w;
	std::tie(wei_n, std::ignore, wei_h, wei_w) = tie4(dwDesc.GetLengths());

	int out_h, out_w;
	std::tie(std::ignore, std::ignore, out_h, out_w) = tie4(dyDesc.GetLengths());

	for(int i = 0; i < in_n; i++) {
		size_t in_offset = i * in_c * in_h * in_w;
		Im2ColGPU(handle, x, in_offset, in_c, in_h, in_w, wei_h, wei_w, out_h, out_w, pad_h, pad_w, v, u, workSpace);

	}

}

}  // namespace mlopen
