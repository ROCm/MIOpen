#include <mlopen/gemm.hpp>

namespace mlopen {

GemmGeometry CreateGemmGeometryConvBwdWeights(
		const TensorDescriptor&		dyDesc,
		const TensorDescriptor&		xDesc,
		const TensorDescriptor&		dwDesc,
		bool						isColMajor,
		std::string					&network_config)
{
	int in_n, in_c, in_h, in_w;
	std::tie(in_n, in_c, in_h, in_w) = tie4(xDesc.GetLengths());

	int wei_n, wei_h, wei_w;
	std::tie(wei_n, std::ignore, wei_h, wei_w) = tie4(dwDesc.GetLengths());

	int out_h, out_w;
	std::tie(std::ignore, std::ignore, out_h, out_w) = tie4(dyDesc.GetLengths());

	// GEMM
	int N = in_c * wei_h * wei_w; 
	int M = wei_n; 
	int K = out_h * out_w;  
	bool tA = false;
	bool tB = true;
	bool tC = false;
	int lda = K;
	int ldb = K;
	int ldc = N;
	float alpha = 1.0;
	float beta = 1.0;

	// bool isColMajor, bool tA, bool tB, bool tC, lda, ldb, ldc, m, n, k, a_offset, b_offset, c_offset
	TinyGemmGeometry tgg;
	GemmGeometry gg;
	
	if (!isColMajor) {
		tgg = TinyGemmGeometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 0, 0);

		gg = GemmGeometry{std::array<int, 3>{{N, M, K}},
			std::array<int, 3>{{ldb, lda, ldc}},
			"mlopenConvolutionBwdWeightsAlgoGEMM",
			alpha, beta, tgg};
	}
	else {
		tgg = TinyGemmGeometry(true, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 0, 0);
		network_config = tgg.get_networkconfig_string();

		gg = GemmGeometry{std::array<int, 3>{{M, N, K}},
			std::array<int, 3>{{lda, ldb, ldc}},
			"mlopenConvolutionBwdWeightsAlgoGEMM", 
			alpha, beta, tgg};
	}
	network_config = tgg.get_networkconfig_string();
	return gg;
}

GemmGeometry CreateGemmGeometryConvFwd(
		const TensorDescriptor&		xDesc,
		const TensorDescriptor&		wDesc,
		const TensorDescriptor&		yDesc,
		bool						isColMajor,
		std::string					&network_config)
{
	int in_n, in_c, in_h, in_w;
	std::tie(in_n, in_c, in_h, in_w) = tie4(xDesc.GetLengths());

	int wei_n, wei_h, wei_w;
	std::tie(wei_n, std::ignore, wei_h, wei_w) = tie4(wDesc.GetLengths());

	int out_h, out_w;
	std::tie(std::ignore, std::ignore, out_h, out_w) = tie4(yDesc.GetLengths());

	// GEMM
	int K = in_c * wei_h * wei_w;
	int M = wei_n;
	int N = out_h * out_w;
	float alpha = 1.0;
	float beta = 0.0;
	bool tA = false;
	bool tB = false;
	bool tC = false;
	int lda = K;
	int ldb = N;
	int ldc = N;

	// bool isColMajor, bool tA, bool tB, bool tC, lda, ldb, ldc, m, n, k, a_offset, b_offset, c_offset
	TinyGemmGeometry tgg;
	GemmGeometry gg;
	
	if (!isColMajor) {
		tgg = TinyGemmGeometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 0, 0);

		gg = GemmGeometry{std::array<int, 3>{{N, M, K}}, 
			std::array<int, 3>{{ldb, lda, ldc}},
			"mlopenConvolutionFwdAlgoGEMM",
			alpha, beta, tgg};
	}
	else {
		tgg = TinyGemmGeometry(false, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 0, 0);

		gg = GemmGeometry{std::array<int, 3>{{M, N, K}},
			std::array<int, 3>{{lda, ldb, ldc}},
			"mlopenConvolutionFwdAlgoGEMM", 
			alpha, beta, tgg};
	}
	network_config = tgg.get_networkconfig_string();
	return gg;
}

GemmGeometry CreateMLOpenGemmGeometry( 
		int M, int N, int K,
		int lda, int ldb, int ldc,
		bool tA, bool tB,
		bool isDataColMajor,
		float alpha, float beta)
{
	TinyGemmGeometry tgg;
	
	// Assuming we are using tinygemm as only col major
	// Therefore, if the user provides data in col. major
	// then no transformations are requrired and vice versa
	if(isDataColMajor) {		
		tgg = TinyGemmGeometry(true, tA, tB, false, lda, ldb, ldc, M, N, K, 0, 0, 0);

		return GemmGeometry{std::array<int, 3>{{M, N, K}},
			std::array<int, 3>{{lda, ldb, ldc}},
			"mlopenGEMM",
			alpha, beta, tgg};
	}
	else {
		tgg = TinyGemmGeometry(true, tB, tA, false, ldb, lda, ldc, N, M, K, 0, 0, 0);

		return GemmGeometry{std::array<int, 3>{{N, M, K}},
			std::array<int, 3>{{ldb, lda, ldc}}, 
			"mlopenGEMM",
			alpha, beta, tgg};
	}
}

void GemmGeometry::EnableBetaKernel(bool enable,
		std::map<std::string, size_t> &beta_args)
{
	beta_kern_req = enable;
	beta_kern_args[0] = beta_args.at("dim_coal");
	beta_kern_args[1] = beta_args.at("dim_uncoal");
}

void GemmGeometry::FindSolution(float time,
		Handle			&handle,
		ConstData_t		a,
		ConstData_t		b,
		Data_t			c,
		bool			enforce_determinism)
{
	//tinygemm does not support m or n < 16
	
	//if(dims[0] < 16 || dims[1] < 16)
	//	return;

	// alloted_time, queue, a, b, c, enforce_determinism, float_type, geometry, alpha, beta, verbose 
	TinyGemmSolution soln = tinygemm::find(time, handle.GetStream(), a, b, c, enforce_determinism, 'f', tgg, alpha, beta, false);

	std::string program_name = soln.main_kernel;
	std::string kernel_name = soln.main_kernel_function_name;
	std::string network_config = tgg.get_networkconfig_string();

	auto main_kernel_worksize_params =  soln.get_main_kernel_worksize_params(dims[0], dims[1]);

	size_t local_work_size = main_kernel_worksize_params.at("local_work_size");
	size_t global_work_size = main_kernel_worksize_params.at("global_work_size");

	std::vector<size_t> vld (1, local_work_size);
	std::vector<size_t> vgd (1, global_work_size);

	handle.GetKernel(algorithm_name,
			network_config,
			program_name,
			kernel_name,
			vld,
			vgd,
			"");

	// beta kernel
	if(beta != 1.0 && !soln.betac_kernel.empty())
	{
		std::string beta_program_name = soln.betac_kernel;
		std::string beta_kernel_name = soln.betac_kernel_function_name;
		auto beta_kernel_worksize_params = soln.get_betac_kernel_worksize_params(dims[0], dims[1]);

		local_work_size = beta_kernel_worksize_params.at("local_work_size");
		global_work_size = beta_kernel_worksize_params.at("global_work_size");

		EnableBetaKernel(true, beta_kernel_worksize_params);

		vld[0] = local_work_size;
		vgd[1] = global_work_size;

		// TODO: remove placeholder
		handle.GetKernel(algorithm_name+"_beta",
				"placeholder",
				beta_program_name,
				beta_kernel_name,
				vld,
				vgd,
				"");
	}

	gemm_geo_map[std::make_pair(algorithm_name, network_config)] = *this;
}

void GemmGeometry::RunGemm(Handle &handle,
			ConstData_t		a,
			ConstData_t		b,
			Data_t			c,
			int				a_offset,
			int				b_offset,
			int				c_offset)
{

	std::string network_config = tgg.get_networkconfig_string();

	// beta kernel, if required
	if(beta_kern_req) {
		handle.GetKernel(algorithm_name+"_beta", "placeholder") (beta_kern_args[0], beta_kern_args[1],
				strides[2], c_offset, c, beta);
	}

	// main kernel
//  c, a, b, alpha, beta, lda, ldb, ldc, m, n, k, a_offset, b_offset, c_offset
	handle.GetKernel(algorithm_name, network_config)(c, a, b,
			alpha, beta,
			strides[0], strides[1], strides[2],
			dims[0], dims[1], dims[2],
			a_offset, b_offset, c_offset);
}


GemmGeometry GetGemmGeometry(std::string algorithm_name, std::string network_config)
{
	auto gemm_iterator = gemm_geo_map.find(std::make_pair(algorithm_name, network_config));
	if (gemm_iterator != gemm_geo_map.end())
	{
		return gemm_iterator->second;
	}
	else
	{
        MLOPEN_THROW("looking for gemm kernel (does not exist): " + algorithm_name + ", " + network_config);
	}
}

} // namespace mlopen
