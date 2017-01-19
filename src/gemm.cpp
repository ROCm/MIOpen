#include <mlopen/gemm.hpp>

namespace mlopen {

GemmGeometry CreateGemmGeometryConvBwdWeights(
		const TensorDescriptor&		dyDesc,
		const TensorDescriptor&		xDesc,
		const TensorDescriptor&		dwDesc,
		bool						isColMajor)
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
	
	if (isColMajor == true) {
		tgg = TinyGemmGeometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 0, 0);
		return GemmGeometry{std::array<int, 3>{N, M, K}, std::array<int, 3>{ldb, lda, ldc}, "mlopenConvolutionBwdWeightsAlgoGEMM", alpha, beta, tgg};
	}
	else {
		tgg = TinyGemmGeometry(false, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 0, 0);
		return GemmGeometry{std::array<int, 3>{M, N, K}, std::array<int, 3>{lda, ldb, ldc}, "mlopenConvolutionBwdWeightsAlgoGEMM", alpha, beta, tgg};
	}
}

GemmGeometry CreateGemmGeometryConvFwd(
		const TensorDescriptor&		xDesc,
		const TensorDescriptor&		wDesc,
		const TensorDescriptor&		yDesc,
		bool						isColMajor)
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
	
	if (isColMajor == true) {
		tgg = TinyGemmGeometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 0, 0);
		return GemmGeometry{std::array<int, 3>{N, M, K}, std::array<int, 3>{ldb, lda, ldc}, "mlopenConvolutionFwdAlgoGEMM", alpha, beta, tgg};
	}
	else {
		tgg = TinyGemmGeometry(false, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 0, 0);
		return GemmGeometry{std::array<int, 3>{M, N, K}, std::array<int, 3>{lda, ldb, ldc}, "mlopenConvolutionFwdAlgoGEMM", alpha, beta, tgg};
	}
}

void GemmGeometry::FindSolution(float time,
		Handle			&handle,
		cl_mem			a,
		cl_mem			b,
		cl_mem			c,
		bool			enforce_determinism)
{
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
}

void GemmGeometry::RunGemm(Handle &handle,
			cl_mem			a,
			cl_mem			b,
			cl_mem			c,
			int				a_offset,
			int				b_offset,
			int				c_offset)
{

	std::string network_config = tgg.get_networkconfig_string();

//  c, a, b, alpha, beta, lda, ldb, ldc, m, n, k, a_offset, b_offset, c_offset
	handle.GetKernel(algorithm_name, network_config)(c, a, b,
			alpha, beta,
			strides[0], strides[1], strides[2],
			dims[0], dims[1], dims[2],
			a_offset, b_offset, c_offset);
}

} // namespace mlopen
