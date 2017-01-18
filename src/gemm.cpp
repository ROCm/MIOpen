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
	float alpha = 1.0;
	float beta = 1.0;
	bool tA = false;
	bool tB = true;
	bool tC = false;
	unsigned int lda = K;
	unsigned int ldb = K;
	unsigned int ldc = N;

	// bool isColMajor, bool tA, bool tB, bool tC, lda, ldb, ldc, m, n, k, a_offset, b_offset, c_offset
	TinyGemmGeometry tgg;
	
	if (isColMajor == true) {
		tgg = TinyGemmGeometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 0, 0);
		return GemmGeometry{std::array<int, 3>{N, M, K}, std::array<int, 3>{ldb, lda, ldc}, tgg};
	}
	else {
		tgg = TinyGemmGeometry(false, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 0, 0);
		return GemmGeometry{std::array<int, 3>{M, N, K}, std::array<int, 3>{lda, ldb, ldc}, tgg};
	}
}

} // namespace mlopen
