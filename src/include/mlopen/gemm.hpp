#ifndef GUARD_MLOPEN_GEMM_HPP_
#define GUARD_MLOPEN_GEMM_HPP_

#include <tinygemm/tinygemm.hpp>
#include <mlopen/tensor.hpp>
#include <mlopen/gemm_geometry.hpp>

namespace mlopen {

GemmGeometry GetGemmGeometry(std::string algorithm_name, std::string network_config);

GemmGeometry CreateGemmGeometryConvBwdWeights(
        const TensorDescriptor&     dyDesc,
        const TensorDescriptor&     xDesc,
        const TensorDescriptor&     dwDesc,
        bool                        isDataColMajor,
        std::string                 &network_config);

GemmGeometry CreateGemmGeometryConvFwd(
        const TensorDescriptor&     xDesc,
        const TensorDescriptor&     wDesc,
        const TensorDescriptor&     yDesc,
        bool                        isDataColMajor,
        std::string                 &network_config);

GemmGeometry CreateMLOpenGemmGeometry( 
        int M, int N, int K,
        int lda, int ldb, int ldc,
        bool tA, bool tB,
        bool isDataColMajor,
        float alpha, float beta);

} // namespace mlopen

#endif // GUARD_MLOPEN_GEMM_HPP_

