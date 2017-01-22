#ifndef GUARD_MLOPEN_GEMM_HPP_
#define GUARD_MLOPEN_GEMM_HPP_

#include <tinygemm/tinygemm.hpp>
#include <mlopen/tensor.hpp>
#include <mlopen/handle.hpp>

namespace mlopen {

using tinygemm::TinyGemmGeometry;
using tinygemm::TinyGemmSolution;

struct GemmGeometry {
	std::array<int, 3> dims; // m, n, k
	std::array<int, 3> strides; // lda, ldb, ldc
	std::string algorithm_name;
	float alpha;
	float beta;
	TinyGemmGeometry tgg;
	bool beta_kern_req;
	std::array<int, 2> beta_kern_args;

	GemmGeometry(){}
	GemmGeometry(std::array<int, 3> pdims, std::array<int, 3>pstrides, std::string algo_name, float palpha, float pbeta, TinyGemmGeometry ptgg) : 
		dims(pdims), strides(pstrides), algorithm_name(algo_name), alpha(palpha), beta(pbeta), tgg(ptgg) 
	{
		beta_kern_req = false;
		beta_kern_args = {{0, 0}};
	}

	void EnableBetaKernel(bool enable, std::map<std::string, size_t> &beta_args);

	void FindSolution(float time,
			Handle			&handle,
			ConstData_t		a,
			ConstData_t		b,
			Data_t			c,
			bool			enforce_determinism);

	void RunGemm(Handle		&handle,
			ConstData_t		a,
			ConstData_t		b,
			Data_t			c,
			int				a_offset,
			int				b_offset,
			int				c_offset);
};

GemmGeometry CreateGemmGeometryConvBwdWeights(
		const TensorDescriptor&		dyDesc,
		const TensorDescriptor&		xDesc,
		const TensorDescriptor&		dwDesc,
		bool						isColMajor);

GemmGeometry CreateGemmGeometryConvFwd(
		const TensorDescriptor&		xDesc,
		const TensorDescriptor&		wDesc,
		const TensorDescriptor&		yDesc,
		bool						isColMajor);

GemmGeometry CreateMLOpenGemmGeometry( 
		int M, int N, int K,
		int lda, int ldb, int ldc,
		bool tA, bool tB,
		bool isDataColMajor,
		float alpha, float beta);

} // namespace mlopen

#endif // GUARD_MLOPEN_GEMM_HPP_

