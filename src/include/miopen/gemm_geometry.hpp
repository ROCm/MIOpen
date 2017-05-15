#ifndef GUARD_MIOPEN_GEMM_GEOMETRY_HPP_
#define GUARD_MIOPEN_GEMM_GEOMETRY_HPP_

#include <tinygemm/tinygemm.hpp>
#include <miopen/tensor.hpp>
#include <miopen/kernel_cache.hpp>

namespace miopen {

using tinygemm::TinyGemmGeometry;
using tinygemm::TinyGemmSolution;

struct GemmGeometry {
    std::array<int, 3> dims{}; // m, n, k
    std::array<int, 3> strides{}; // lda, ldb, ldc
    std::string algorithm_name;
    float alpha{};
    float beta{};
    TinyGemmGeometry tgg {};
    bool beta_kern_req{};
    std::array<int, 2> beta_kern_args{};

    GemmGeometry(){}
    GemmGeometry(std::array<int, 3> pdims, std::array<int, 3>pstrides, std::string algo_name, float palpha, float pbeta, TinyGemmGeometry ptgg) : 
        dims(pdims), strides(pstrides), algorithm_name(algo_name), alpha(palpha), beta(pbeta), tgg(ptgg) 
    {
        beta_kern_req = false;
        beta_kern_args = {{0, 0}};
    }

    void EnableBetaKernel(bool enable);

    void FindSolution(float time,
            Handle          &handle,
            ConstData_t     a,
            ConstData_t     b,
            Data_t          c,
            bool            enforce_determinism);

    void RunGemm(Handle     &handle,
            ConstData_t     a,
            ConstData_t     b,
            Data_t          c,
            int             a_offset,
            int             b_offset,
            int             c_offset);
};

using GemmKey = std::pair<std::string, std::string>;
std::unordered_map< GemmKey, GemmGeometry, SimpleHash>& gemm_geo_map();

} // namespace miopen

#endif // GUARD_MIOPEN_GEMM_GEOMETRY_HPP_
