#include <mlopen/activ.hpp>
#include <mlopen/mlo_internal.hpp>
#include <mlopen/kernel_cache.hpp>

namespace mlopen {

	mlopenStatus_t ActivationDescriptor::Forward(
			Handle						&handle,
			const void					* /* alpha */,
			const TensorDescriptor		&xDesc,
			const Data_t				x,
			const void					* /* beta */,
			const TensorDescriptor		&yDesc,
			Data_t						y,
			bool                        do_backward,
			Data_t						/* workSpace */,
			size_t						*workSpaceSize) {

		mlopenStatus_t status = mlopenStatusSuccess;
		printf("in activation forward\n");


		mlo_construct_neuron construct_params(1); // forward

		std::string kernel_path = "../src/Kernels/";

		construct_params.setKernelPath(kernel_path);

		construct_params.setStream(handle.GetStream());

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

		double activ_alpha = GetAlpha();
		double activ_beta = GetBeta();
		double activ_power = GetPower();

		construct_params.doBackward(do_backward);
		construct_params.setNeuronDescr(static_cast<int>(mode), activ_power, activ_beta, activ_alpha);

// construct
		status = static_cast<mlopenStatus_t>(construct_params.mloConstruct());


		if (x == nullptr || y == nullptr)
		{
			*workSpaceSize = construct_params.getWorkSpaceSzBytes();
		}
		else
		{

			std::string program_name = kernel_path + construct_params.getKernelFile();  // CL kernel filename
			std::string kernel_name = construct_params.getKernelName(); // kernel name
			std::string compiler_options = construct_params.getCompilerOptions(); // kernel parameters

			std::string network_config;
			construct_params.mloBuildConf_Key(network_config);

			const std::vector<size_t> & vld = construct_params.getLocalWkSize();
			const std::vector<size_t> & vgd = construct_params.getGlobalWkSize();

			construct_params.getNeuronDescr(reinterpret_cast<int&>(mode), activ_power, activ_beta, activ_alpha);
			float f_activ_alpha = static_cast<float>(activ_alpha);
			float f_activ_beta = static_cast<float>(activ_beta);
			float f_activ_power = static_cast<float>(activ_power);

			handle.GetKernel("mlopenActivationForward",
						network_config,
						program_name,
						kernel_name,
						vld,
						vgd,
						compiler_options)(x, y, f_activ_power, f_activ_beta, f_activ_alpha);

			handle.Finish();

			std::cout << "Activation Forward Finished !!" << std::endl;

		}

		return(status);
	}

	mlopenStatus_t ActivationDescriptor :: Backward(
			Handle						&handle,
			const void					* /* alpha */,
			const TensorDescriptor		&yDesc,
			const Data_t		  		y,
			const TensorDescriptor		&dyDesc,
			const Data_t		  		dy,
			const TensorDescriptor		&xDesc,
			const Data_t		  		x,
			const void			  		* /* beta */,
			const TensorDescriptor		&dxDesc,
			Data_t						dx,
			const Data_t				/* workSpace */) {

		mlopenStatus_t status = mlopenStatusSuccess;
		printf("in activation backward\n");



		mlo_construct_neuron construct_params(0); // backward

		std::string kernel_path = "../src/Kernels/";

		construct_params.setKernelPath(kernel_path);

		construct_params.setStream(handle.GetStream());
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

		int activ_mode = GetMode();
		double activ_alpha = GetAlpha();
		double activ_beta = GetBeta();
		double activ_power = GetPower();

		construct_params.setNeuronDescr(activ_mode, activ_power, activ_beta, activ_alpha);

// construct
		status = static_cast<mlopenStatus_t>(construct_params.mloConstruct());

		std::string program_name = kernel_path + construct_params.getKernelFile();  // CL kernel filename
		std::string kernel_name = construct_params.getKernelName(); // kernel name
		std::string compiler_options = construct_params.getCompilerOptions(); // kernel parameters

		std::string network_config;
		construct_params.mloBuildConf_Key(network_config);

		const std::vector<size_t> & vld = construct_params.getLocalWkSize();
		const std::vector<size_t> & vgd = construct_params.getGlobalWkSize();


		float f_activ_alpha = static_cast<float>(GetAlpha());
		float f_activ_beta = static_cast<float>(GetBeta());
		float f_activ_power = static_cast<float>(GetPower());
		float f_diff_scale = f_activ_beta * f_activ_power;

		handle.GetKernel("mlopenActivationBackward",
			network_config,
			program_name,
			kernel_name,
			vld,
			vgd,
			compiler_options)(dx, dy, x, y, f_diff_scale, f_activ_power, f_activ_beta, f_activ_alpha);


		handle.Finish();

		std::cout << "Activation Backward Finished !!" << std::endl;

		return(status);
	}
}  // namespace mlopen
