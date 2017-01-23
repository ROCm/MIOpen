#include <mlopen/convolution.hpp>
#include <mlopen/util.hpp>
#include <mlopen/mlo_internal.hpp>

#include <tinygemm/tinygemm.hpp>
#include <mlopen/gemm.hpp>

namespace mlopen {

void ConvolutionDescriptor::FindConvFwdAlgorithm(Handle& handle,
		const TensorDescriptor&		xDesc,
		ConstData_t				x,
		const TensorDescriptor&		wDesc,
		ConstData_t				w,
		const TensorDescriptor&		yDesc,
		ConstData_t				y,
		const int					 /*requestAlgoCount*/,
		int							* /*returnedAlgoCount*/,
		mlopenConvAlgoPerf_t		*perfResults,
		mlopenConvPreference_t		 /*preference*/,
		Data_t						workSpace,
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

// GEMM based
	int in_n, in_c, in_h, in_w;
	std::tie(in_n, in_c, in_h, in_w) = tie4(xDesc.GetLengths());

	int wei_n, wei_h, wei_w;
	std::tie(wei_n, std::ignore, wei_h, wei_w) = tie4(wDesc.GetLengths());

	int out_h, out_w;
	std::tie(std::ignore, std::ignore, out_h, out_w) = tie4(yDesc.GetLengths());

	if(wei_h != 1 && wei_w != 1) {
		size_t in_offset = 0;
		Im2ColGPU(handle, x, in_offset, in_c, in_h, in_w, wei_h, wei_w, out_h, out_w, pad_h, pad_w, v, u, workSpace);
	}

	std::string network_config;
	GemmGeometry gg = CreateGemmGeometryConvFwd(xDesc, wDesc, yDesc, false, network_config);
	gg.FindSolution(.003, handle, workSpace, w, y, false);

	// Direct algo
	// Generate kernels if OpenCL
	// Compile, cache kernels, etc.
	// Launch all kernels and store the perf, workspace limits, etc.
	
	mlo_construct_direct2D construct_params(1); // forward
	construct_params.doSearch(exhaustiveSearch);
	construct_params.saveSearchRequest(true);

	construct_params.setGeneralCompOptions("");

	construct_params.setStream(&handle);

	construct_params.setOutputDescFromMLDesc(yDesc);
	construct_params.setInputDescFromMLDesc(xDesc);
	construct_params.setWeightDescFromMLDesc(wDesc);

	construct_params.setConvDescr(pad_h, pad_w, u, v, upscalex, upscaley);

	construct_params.mloConstruct();

	std::string program_name = construct_params.getKernelFile();  //"../src/Hello.cl"; // CL kernel filename
	std::string kernel_name = construct_params.getKernelName(); // "hello_world_kernel"; // kernel name
	std::string parms = construct_params.getCompilerOptions(); // kernel parameters

	construct_params.mloBuildConf_Key(network_config);

	const std::vector<size_t> & vld = construct_params.getLocalWkSize();
	const std::vector<size_t> & vgd = construct_params.getGlobalWkSize();

	// TODO: call the kernel here
	//float padding_val = 0;
	handle.GetKernel("mlopenConvolutionFwdAlgoDirect",
			network_config,
			program_name, 
			kernel_name,
			vld,
			vgd,
			parms);

	// FIXME: MD temporary hack for hipcaffe
	// should be ideally wrapped under mlopen::deref to check 
	// for the size of perfResults == requestedAlgoCount
	perfResults->fwd_algo = mlopenConvolutionFwdAlgoDirect;
}

void ConvolutionDescriptor::ConvolutionForward(Handle& handle,
		const void					* /*alpha*/,
		const TensorDescriptor&		xDesc,
		ConstData_t				x,
		const TensorDescriptor&		wDesc,
		ConstData_t				w,
		mlopenConvFwdAlgorithm_t	algo,
		const void					* /*beta*/,
		const TensorDescriptor&		yDesc,
		Data_t						y, 
		Data_t						workSpace,
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

	switch (algo)
	{
		case mlopenConvolutionFwdAlgoDirect:
		{
			// TODO(paul): Replicating code for now.
			mlo_construct_direct2D construct_params(1); // forward
			construct_params.setOutputDescFromMLDesc(yDesc);
			construct_params.setInputDescFromMLDesc(xDesc);
			construct_params.setWeightDescFromMLDesc(wDesc);

			std::string network_config;
			construct_params.mloBuildConf_Key(network_config);

			std::string algorithm_name = "mlopenConvolutionFwdAlgoDirect";
			float padding_val = 0;
			handle.GetKernel(algorithm_name, network_config)(x, w, y, padding_val);
		}
		break;

		case mlopenConvolutionFwdAlgoGEMM:
		{
			int in_n, in_c, in_h, in_w;
			std::tie(in_n, in_c, in_h, in_w) = tie4(xDesc.GetLengths());

			int wei_n, wei_h, wei_w;
			std::tie(wei_n, std::ignore, wei_h, wei_w) = tie4(wDesc.GetLengths());

			int out_h, out_w;
			std::tie(std::ignore, std::ignore, out_h, out_w) = tie4(yDesc.GetLengths());

			std::string network_config;
			CreateGemmGeometryConvFwd(xDesc, wDesc, yDesc, false, network_config);
			GemmGeometry gg = GetGemmGeometry("mlopenConvolutionFwdAlgoGEMM", network_config);
			
			float time_0 = 0;
			float t1 = 0;
			for(int i = 0; i < in_n; i++) {
				int out_offset = i * wei_n * out_h * out_w;
				if(wei_h != 1 && wei_w != 1) {
					size_t in_offset = i * in_c * in_h * in_w;
					Im2ColGPU(handle, x, in_offset, in_c, in_h, in_w, wei_h, wei_w, out_h, out_w, pad_h, pad_w, v, u, workSpace);
					if(handle.IsProfilingEnabled())
						t1 = handle.GetKernelTime();

					gg.RunGemm(handle, workSpace, w, y, 0, 0, out_offset);

					// Update times for both the kernels
					if(handle.IsProfilingEnabled()) {
						if(i == in_n - 1)
							handle.AccumKernelTime(t1+time_0);
						else
							handle.AccumKernelTime(t1);
						time_0 += handle.GetKernelTime();
					}
				}
				else if(wei_h == 1 && wei_w == 1) {
					int in_offset = i * in_c * in_h * in_w;
					gg.RunGemm(handle, x, w, y, in_offset, 0, out_offset);
					if(handle.IsProfilingEnabled()) {
						if(i == in_n - 1)
							handle.AccumKernelTime(time_0);
						time_0 += handle.GetKernelTime();
					}

				} 
			}
		}
		break;
		case mlopenConvolutionFwdAlgoFFT:
		break;
		case mlopenConvolutionFwdAlgoWinograd:
		break;

	}
}

// FindBackwardDataAlgorithm()
//
void ConvolutionDescriptor::FindConvBwdDataAlgorithm(Handle& handle,
		const TensorDescriptor&		dyDesc,
		ConstData_t				dy,
		const TensorDescriptor&		wDesc,
		ConstData_t				w,
		const TensorDescriptor&		dxDesc,
		ConstData_t				dx,
		const int					 /*requestAlgoCount*/,
		int							* /*returnedAlgoCount*/,
		mlopenConvAlgoPerf_t		*perfResults,
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

	// Generate kernels if OpenCL
	// Compile, cache kernels, etc.
	// Launch all kernels and store the perf, workspace limits, etc.
	mlo_construct_direct2D construct_params(0); // backward
	construct_params.doSearch(exhaustiveSearch);
	construct_params.saveSearchRequest(true);

	construct_params.setGeneralCompOptions("");

	construct_params.setStream(&handle);

	construct_params.setOutputDescFromMLDesc(dyDesc);
	construct_params.setInputDescFromMLDesc(dxDesc);
	construct_params.setWeightDescFromMLDesc(wDesc);

	construct_params.setConvDescr(pad_h, pad_w, u, v, upscalex, upscaley);

	construct_params.mloConstruct();

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

	// FIXME: MD temporary hack for hipcaffe
	// should be ideally wrapped under mlopen::deref to check 
	// for the size of perfResults == requestedAlgoCount
	perfResults->bwd_data_algo = mlopenConvolutionBwdDataAlgo_0;
	perfResults->time = handle.GetKernelTime();

}

// BackwardDataAlgorithm()
void ConvolutionDescriptor::ConvolutionBackwardData(Handle& handle,
		const void						* /*alpha*/,
		const TensorDescriptor&			dyDesc,
		ConstData_t					dy,
		const TensorDescriptor&			wDesc,
		ConstData_t					w,
		mlopenConvBwdDataAlgorithm_t	/* algo */,
		const void						* /*beta*/,
		const TensorDescriptor&			dxDesc,
		Data_t							dx, 
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

	// Launch all kernels and store the perf, workspace limits, etc.
	mlo_construct_direct2D construct_params(0); // backward
	construct_params.setOutputDescFromMLDesc(dyDesc);
	construct_params.setInputDescFromMLDesc(dxDesc);
	construct_params.setWeightDescFromMLDesc(wDesc);
	construct_params.setStream(&handle);

	std::string network_config;
	construct_params.mloBuildConf_Key(network_config);

	float padding_val = 0;
	handle.GetKernel("mlopenConvolutionBwdDataAlgo_0", network_config) (dy, w, dx, padding_val);
}

// ConvolutionBackwardWeightsGetWorkSpaceSize
void ConvolutionDescriptor::ConvolutionBackwardWeightsGetWorkSpaceSize(
		const TensorDescriptor&		 dyDesc,
		const TensorDescriptor&		 xDesc,
		const TensorDescriptor&		 dwDesc,
		size_t						*workSpaceSize)
{
	mlo_construct_BwdWrW2D construct_params(0); // backward with regards to weights
	construct_params.doSearch(false);
	construct_params.setOutputDescFromMLDesc(dyDesc);
	construct_params.setInputDescFromMLDesc(xDesc);
	construct_params.setWeightDescFromMLDesc(dwDesc);
	construct_params.setConvDescr(pad_h, pad_w, u, v, upscalex, upscaley);
	construct_params.mloConstruct();

	*workSpaceSize = construct_params.getWorkSpaceSzBytes();
}

// FindBackwardWeightsAlgorithm()
//
void ConvolutionDescriptor::FindConvBwdWeightsAlgorithm(Handle& handle,
		const TensorDescriptor&		dyDesc,
		ConstData_t				dy,
		const TensorDescriptor&		xDesc,
		ConstData_t				x,
		const TensorDescriptor&		dwDesc,
		ConstData_t				dw,
		const int					 /*requestAlgoCount*/,
		int							* /*returnedAlgoCount*/,
		mlopenConvAlgoPerf_t		*perfResults,
		mlopenConvPreference_t		 /*preference*/,
		Data_t						workSpace,
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

	if(wei_h != 1 && wei_w != 1) {
		size_t in_offset = 0;
		Im2ColGPU(handle, x, in_offset, in_c, in_h, in_w, wei_h, wei_w, out_h, out_w, pad_h, pad_w, v, u, workSpace);
	}

	std::string network_config;
	GemmGeometry gg = CreateGemmGeometryConvBwdWeights(dyDesc, xDesc, dwDesc, false, network_config);
	gg.FindSolution(.003, handle, workSpace, dy, dw, false);

// temprorary guard
	if((u == 1 && v == 1) || (wei_w >= 7 && u == 2 && v == 2))
	{
		mlo_construct_BwdWrW2D construct_params(0); // backward with regards to weights
		construct_params.doSearch(false);
		construct_params.setStream(&handle);
		construct_params.setOutputDescFromMLDesc(dyDesc);
		construct_params.setInputDescFromMLDesc(xDesc);
		construct_params.setWeightDescFromMLDesc(dwDesc);
		construct_params.setConvDescr(pad_h, pad_w, u, v, upscalex, upscaley);
		construct_params.mloConstruct();

		construct_params.mloBuildConf_Key(network_config);

		const std::vector<mlo_kernel_info> & bwd_wrw_info = construct_params.getKernelsInfo();
		/*
		 * get info for all kernels of the layer
		 * std::string _kernel_name;
		 * std::string _kernel_file;
		 * std::string _comp_options;
		 * std::vector<size_t> _g_wk;
		 * std::vector<size_t> _l_wk;
		 */

		//TODO: the kernels should be able to be called from Find()
		// Actually, that is requried to correctly populate the
		// PerfResults. May be we should clear the outputs in Find()
		// after the kernel finishes
		// main kernel
		if (bwd_wrw_info.size() == 1)
		{
			const mlo_kernel_info &bwd_wrw = bwd_wrw_info[0];
//			float padding_val = 0;

			handle.GetKernel("mlopenConvolutionBwdWeightsAlgoDirect_Main",
					network_config,
					std::get<1>(bwd_wrw),
					std::get<0>(bwd_wrw),
					std::get<4>(bwd_wrw),
					std::get<3>(bwd_wrw),
					std::get<2>(bwd_wrw));
			//				(dy, x, dw, padding_val);
		}
		else
		{
			auto bwd_wrw_main = bwd_wrw_info[0];
//			float padding_val = 0;

			handle.GetKernel("mlopenConvolutionBwdWeightsAlgoDirect_Main",
					network_config,
					std::get<1>(bwd_wrw_main),
					std::get<0>(bwd_wrw_main),
					std::get<4>(bwd_wrw_main),
					std::get<3>(bwd_wrw_main),
					std::get<2>(bwd_wrw_main));
			//					(dy, x, workSpace, padding_val);

			float time0 = handle.GetKernelTime();
			// second kernel hash
			network_config += "x1";
			// reduction  kernel
			auto bwd_wrw_red = bwd_wrw_info[1];

			handle.GetKernel("mlopenConvolutionBwdWeightsAlgoDirect_Red",
					network_config,
					std::get<1>(bwd_wrw_red),
					std::get<0>(bwd_wrw_red),
					std::get<4>(bwd_wrw_red),
					std::get<3>(bwd_wrw_red),
					std::get<2>(bwd_wrw_red));
			//					(workSpace, dw);

			handle.AccumKernelTime(time0);

		}

	// FIXME: MD temporary hack for hipcaffe
	// should be ideally wrapped under mlopen::deref to check 
	// for the size of perfResults == requestedAlgoCount
	perfResults->bwd_weights_algo = mlopenConvolutionBwdWeightsAlgoDirect;
	}
}

// BackwardWeightsAlgorithm()
void ConvolutionDescriptor::ConvolutionBackwardWeights(Handle& handle,
		const void						* /*alpha*/,
		const TensorDescriptor&			dyDesc,
		ConstData_t					dy,
		const TensorDescriptor&			xDesc,
		ConstData_t					x,
		mlopenConvBwdWeightsAlgorithm_t algo,
		const void						* /*beta*/,
		const TensorDescriptor&			dwDesc,
		Data_t							dw, 
		Data_t							workSpace,
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

	switch (algo)
	{
		case mlopenConvolutionBwdWeightsAlgoGEMM:
		{
			std::string network_config;
			CreateGemmGeometryConvBwdWeights(dyDesc, xDesc, dwDesc, false, network_config);
			GemmGeometry gg = GetGemmGeometry("mlopenConvolutionBwdWeightsAlgoGEMM", network_config);

			handle.ResetKernelTime();
			float time_0 = 0;
			float t1 = 0;
			for(int i = 0; i < in_n; i++) {
				int out_offset = i * wei_n * out_h * out_w;
				if(wei_h != 1 && wei_w != 1) {
					size_t in_offset = i * in_c * in_h * in_w;
					Im2ColGPU(handle, x, in_offset, in_c, in_h, in_w, wei_h, wei_w, out_h, out_w, pad_h, pad_w, v, u, workSpace);
					if(handle.IsProfilingEnabled())
						t1 = handle.GetKernelTime();

					gg.RunGemm(handle, workSpace, dy, dw, 0, out_offset, 0);

					// Update times for both the kernels
					if(handle.IsProfilingEnabled()) {
						if(i == in_n - 1)
							handle.AccumKernelTime(t1+time_0);
						else
							handle.AccumKernelTime(t1);
						time_0 += handle.GetKernelTime();
					}
				}
				else if(wei_h == 1 && wei_w == 1) {
					int in_offset = i * in_c * in_h * in_w;
					gg.RunGemm(handle, x, dy, dw, in_offset, out_offset, 0);

					if(handle.IsProfilingEnabled()) {
						if(i == in_n - 1)
							handle.AccumKernelTime(time_0);
						time_0 += handle.GetKernelTime();
					}
				}
			}

		}
		break;

		case mlopenConvolutionBwdWeightsAlgoDirect:
		{
			if ((u == 1 && v == 1) || (wei_w >= 7 && u == 2 && v == 2))
			{
				mlo_construct_BwdWrW2D construct_params(0); // backward with regards to weights
				construct_params.setOutputDescFromMLDesc(dyDesc);
				construct_params.setInputDescFromMLDesc(xDesc);
				construct_params.setWeightDescFromMLDesc(dwDesc);

				std::string network_config;
				construct_params.mloBuildConf_Key(network_config);
				const std::vector<mlo_kernel_info> & bwd_wrw_info = construct_params.getKernelsInfo();

				handle.ResetKernelTime();

				// main kernel
				if (bwd_wrw_info.size() == 1)
				{
					float padding_val = 0;
					handle.GetKernel("mlopenConvolutionBwdWeightsAlgoDirect_Main",
							network_config)	(dy, x, dw, padding_val);
				}
				else
				{
					float padding_val = 0;
					handle.GetKernel("mlopenConvolutionBwdWeightsAlgoDirect_Main",
							network_config)	(dy, x, workSpace, padding_val);

					float time0 = handle.GetKernelTime();
					// second kernel has
					network_config += "x1";
					// reduction  kernel
					handle.GetKernel("mlopenConvolutionBwdWeightsAlgoDirect_Red",
							network_config) (workSpace, dw);

					handle.AccumKernelTime(time0);
				}
			}
		}
		break;
	};
}

}  // namespace mlopen
