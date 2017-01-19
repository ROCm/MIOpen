#ifndef GUARD_MLOPEN_GEMM_HPP_
#define GUARD_MLOPEN_GEMM_HPP_

#include <tinygemm/tinygemm.hpp>
#include <mlopen/tensor.hpp>
#include <mlopen/handle.hpp>

namespace mlopen {

using namespace tinygemm;

struct GemmGeometry {
	std::array<int, 3> dims; // m, n, k
	std::array<int, 3> strides; // lda, ldb, ldc
	std::string algorithm_name;
	float alpha;
	float beta;
	TinyGemmGeometry tgg;

	GemmGeometry(){}
	GemmGeometry(std::array<int, 3> pdims, std::array<int, 3>pstrides, std::string algo_name, float palpha, float pbeta, TinyGemmGeometry ptgg) : 
		dims(pdims), strides(pstrides), algorithm_name(algo_name), alpha(palpha), beta(pbeta), tgg(ptgg) 
	{}

	void FindSolution(float time,
			Handle			&handle,
			cl_mem			a,
			cl_mem			b,
			cl_mem			c,
			bool			enforce_determinism);

	void RunGemm(Handle		&handle,
			cl_mem			a,
			cl_mem			b,
			cl_mem			c,
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

} // mlopen namespace

#endif // GUARD_MLOPEN_GEMM_HPP_

