#ifndef GUARD_MLOPEN_GEMM_HPP_
#define GUARD_MLOPEN_GEMM_HPP_

#include <tinygemm/tinygemm.hpp>
#include <mlopen/tensor.hpp>

namespace mlopen {

using namespace tinygemm;

struct GemmGeometry {
	std::array<int, 3> dims; // m, n, k
	std::array<int, 3> strides; // lda, ldb, ldc
	TinyGemmGeometry tgg;
};

GemmGeometry CreateGemmGeometryConvBwdWeights(
		const TensorDescriptor&		dyDesc,
		const TensorDescriptor&		xDesc,
		const TensorDescriptor&		dwDesc,
		bool						isColMajor);

} // mlopen namespace

#endif // GUARD_MLOPEN_GEMM_HPP_

