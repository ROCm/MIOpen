#ifndef GUARD_MIOPEN_GEMM_HPP_
#define GUARD_MIOPEN_GEMM_HPP_

#include <miopen/tensor.hpp>
#include <miopen/gemm_geometry.hpp>

namespace miopen {

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

GemmGeometry CreateMIOpenGemmGeometry( 
        int M, int N, int K,
        int lda, int ldb, int ldc,
        bool tA, bool tB,
        bool isDataColMajor,
        float alpha, float beta);

} // namespace miopen

#endif // GUARD_MIOPEN_GEMM_HPP_

