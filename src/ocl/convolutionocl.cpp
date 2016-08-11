#include <mlopen/convolution.hpp>
#include <mlopen/mlo_internal.hpp>

namespace mlopen {

mlopenStatus_t ConvolutionDescriptor::FindConvFwdAlgorithm(mlopen::Context& handle,
		const mlopen::TensorDescriptor&	xDesc,
		const cl_mem					x,
		const mlopen::TensorDescriptor&	wDesc,
		const cl_mem					w,
		const mlopen::TensorDescriptor&	yDesc,
		const cl_mem					y,
		const int						requestAlgoCount,
		int								*returnedAlgoCount,
		mlopenConvAlgoPerf_t			*perfResults,
		mlopenConvPreference_t			preference,
		void							*workSpace,
		size_t							workSpaceSize,
		bool							exhaustiveSearch) const {
	
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
	mlo_construct_direct2D construct_params(1); // forward
	{
		construct_params.setTimerIter(100);
		construct_params.doSearch(exhaustiveSearch);
		construct_params.saveSearchRequest(true);


// TO DO WHERE IS THE PATH ?
		std::string kernel_path = "../src/Kernels/";

		construct_params.setKernelPath(kernel_path);

		construct_params.setGeneralCompOptions("");

		construct_params.setStream(handle.GetStream());

		construct_params.setOutputDescFromMLDesc(yDesc);
		construct_params.setInputDescFromMLDesc(xDesc);
		construct_params.setWeightDescFromMLDesc(wDesc);

		construct_params.setConvDescr(pad_h, pad_w, u, v, upscalex, upscaley);

		construct_params.mloConstructDirect2D();
	}

	std::string program_name = std::string("../src/Kernels/") +  construct_params.getKernelFile();  //"../src/Hello.cl"; // CL kernel filename
	std::string kernel_name = construct_params.getKernelName(); // "hello_world_kernel"; // kernel name
	std::string parms = construct_params.getCompilerOptions(); // kernel parameters

	std::string network_config;
	construct_params.mloBuildConf_Key(network_config);

	const std::vector<size_t> & vld = construct_params.getLocalWkSize();
	const std::vector<size_t> & vgd = construct_params.getGlobalWkSize();

	float padding_val = 0;
	handle.Run("mlopenConvolutionFwdAlgoDirect",
			network_config,
			program_name, 
			kernel_name,
			vld,
			vgd,
			parms)(x, w, y, padding_val);
	
	handle.Finish();

	return mlopenStatusSuccess;

}

mlopenStatus_t ConvolutionDescriptor::ConvolutionForward(mlopen::Context& handle,
		const void							*alpha,
		const mlopen::TensorDescriptor&		xDesc,
		const cl_mem						x,
		const mlopen::TensorDescriptor&		wDesc,
		const cl_mem						w,
		mlopenConvFwdAlgorithm_t			algo,
		const void							*beta,
		const mlopen::TensorDescriptor&		yDesc,
		cl_mem								y, 
		void								*workSpace,
		size_t								workSpaceSize) const {

	if(x == nullptr || w == nullptr || y == nullptr) {
		return mlopenStatusBadParm;
	}
	if(xDesc.GetSize() != yDesc.GetSize() || xDesc.GetSize() != wDesc.GetSize()) {
		return mlopenStatusBadParm;
	}
	if(xDesc.GetType() != yDesc.GetType() || xDesc.GetType() != wDesc.GetType()) {
		return mlopenStatusBadParm;
	}
	if(xDesc.GetLengths()[1] != wDesc.GetLengths()[1]) {
		return mlopenStatusBadParm;
	}
	if(xDesc.GetSize() < 3) {
		return mlopenStatusBadParm;
	}
	
	// TODO: Replicating code for now.
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
	handle.Run(algorithm_name, network_config)(x, w, y, padding_val);
	handle.Finish();

	return mlopenStatusSuccess;

}

// FindBackwardDataAlgorithm()
//
mlopenStatus_t ConvolutionDescriptor::FindConvBwdDataAlgorithm(mlopen::Context& handle,
		const mlopen::TensorDescriptor&	dyDesc,
		const cl_mem					dy,
		const mlopen::TensorDescriptor&	wDesc,
		const cl_mem					w,
		const mlopen::TensorDescriptor&	dxDesc,
		const cl_mem					dx,
		const int						requestAlgoCount,
		int								*returnedAlgoCount,
		mlopenConvAlgoPerf_t			*perfResults,
		mlopenConvPreference_t			preference,
		void							*workSpace,
		size_t							workSpaceSize,
		bool							exhaustiveSearch) const {
	
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
	mlo_construct_direct2D construct_params(0); // backward
	{
		construct_params.setTimerIter(100);
		construct_params.doSearch(exhaustiveSearch);
		construct_params.saveSearchRequest(true);


// TO DO WHERE IS THE PATH ?
		std::string kernel_path = "../src/Kernels/";

		construct_params.setKernelPath(kernel_path);

		construct_params.setGeneralCompOptions("");

		construct_params.setStream(handle.GetStream());

		construct_params.setOutputDescFromMLDesc(dxDesc);
		construct_params.setInputDescFromMLDesc(dyDesc);
		construct_params.setWeightDescFromMLDesc(wDesc);

		construct_params.setConvDescr(pad_h, pad_w, u, v, upscalex, upscaley);

		construct_params.mloConstructDirect2D();
	}

	std::string program_name = std::string("../src/Kernels/") +  construct_params.getKernelFile();  
	std::string kernel_name = construct_params.getKernelName(); // kernel name
	std::string parms = construct_params.getCompilerOptions(); // kernel parameters

	std::string network_config;
	construct_params.mloBuildConf_Key(network_config);

	const std::vector<size_t> & vld = construct_params.getLocalWkSize();
	const std::vector<size_t> & vgd = construct_params.getGlobalWkSize();

	float padding_val = 0;
	handle.Run("mlopenConvolutionBwdDataAlgo_0",
			network_config,
			program_name, 
			kernel_name,
			vld,
			vgd,
			parms)(dy, w, dx, padding_val);
	
	handle.Finish();

	return mlopenStatusSuccess;

}

// BackwardDataAlgorithm()
mlopenStatus_t ConvolutionDescriptor::ConvolutionBackwardData(mlopen::Context& handle,
		const void							*alpha,
		const mlopen::TensorDescriptor&		dyDesc,
		const cl_mem						dy,
		const mlopen::TensorDescriptor&		wDesc,
		const cl_mem						w,
		mlopenConvBwdDataAlgorithm_t		algo,
		const void							*beta,
		const mlopen::TensorDescriptor&		dxDesc,
		cl_mem								dx, 
		void								*workSpace,
		size_t								workSpaceSize) const {

	if(dx == nullptr || w == nullptr || dy == nullptr) {
		return mlopenStatusBadParm;
	}
	if(dyDesc.GetSize() != dxDesc.GetSize() || dyDesc.GetSize() != wDesc.GetSize()) {
		return mlopenStatusBadParm;
	}
	if(dyDesc.GetType() != dxDesc.GetType() || dyDesc.GetType() != wDesc.GetType()) {
		return mlopenStatusBadParm;
	}
	if(dyDesc.GetLengths()[1] != wDesc.GetLengths()[1]) {
		return mlopenStatusBadParm;
	}
	if(dyDesc.GetSize() < 3) {
		return mlopenStatusBadParm;
	}

	// TODO: Replicating code for now.
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
		default:
			printf("Algorithm not found\n");
		break;
	}
	float padding_val = 0;
	handle.Run(algorithm_name, network_config)(dy, w, dx, padding_val);
	handle.Finish();

	return mlopenStatusSuccess;

}
}
