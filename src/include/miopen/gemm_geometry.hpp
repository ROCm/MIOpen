#ifndef GUARD_MIOPEN_GEMM_GEOMETRY_HPP_
#define GUARD_MIOPEN_GEMM_GEOMETRY_HPP_


#include <miopengemm/miogemm.hpp>
#include <miopen/tensor.hpp>
#include <miopen/kernel_cache.hpp>

namespace miopen {

struct GemmGeometry {
    std::array<int, 3> dims{}; // m, n, k
    std::array<int, 3> strides{}; // lda, ldb, ldc
    std::string algorithm_name;
    float alpha{};
    float beta{};
    MIOpenGEMM::Geometry tgg {};
    bool beta_kern_req{};
    
    /* jn : if tinygemm returned a beta kernel. 
     * not the same as beta_kern_req(uired), as 
     * if beta == 1, beta kernel is returned but
     * not required.
     * we still need to know if it was returned,
     * as the function signature of the main kernel
     * is then different.
     * */
    bool beta_kern_returned{}; 
    std::array<int, 2> beta_kern_args{};

    GemmGeometry(){}
    GemmGeometry(std::array<int, 3> pdims, std::array<int, 3>pstrides, std::string algo_name, float palpha, float pbeta, MIOpenGEMM::Geometry ptgg) : 
        dims(pdims), strides(pstrides), algorithm_name(algo_name), alpha(palpha), beta(pbeta), tgg(ptgg) 
    {
        beta_kern_req = false;
        beta_kern_returned = false;
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
