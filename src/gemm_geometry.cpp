#include <miopen/gemm_geometry.hpp>

namespace miopen {

std::unordered_map< GemmKey, GemmGeometry, SimpleHash>& gemm_geo_map()
{
    static std::unordered_map< GemmKey, GemmGeometry, SimpleHash> data;
    return data;
}



void GemmGeometry::EnableBetaKernel(bool enable)
{
    beta_kern_req = enable;
}

void GemmGeometry::FindSolution(float time,
        Handle          &handle,
        ConstData_t     a,
        ConstData_t     b,
        Data_t          c,
        bool            enforce_determinism)
{

#if MIOPEN_BACKEND_OPENCL
    /* jn : using a simple version of find, until miopen supports workspace for gemm  */
    TinyGemmSolution soln = tinygemm::find(time, handle.GetStream(), a, b, c, enforce_determinism, tgg);
#else
    (void)time;
    (void)a;
    (void)b;
    (void)c;
    (void)tgg;
    (void)alpha;
    (void)beta;
    /* jn : this will not compile, not sure what to do for hip */
    TinyGemmSolution soln = tinygemm::get_default(enforce_determinism, 'f', tgg); 
#endif
    
    /* jn : the main kernel is at the back of the solution vector */
    std::string program_name = soln.v_tgks.back().kernstr;
    std::string kernel_name = soln.v_tgks.back().fname;
    std::string network_config = tgg.get_networkconfig_string();
    size_t local_work_size = soln.v_tgks.back().local_work_size;
    size_t global_work_size = soln.v_tgks.back().global_work_size;

    std::vector<size_t> vld {local_work_size, 1, 1};
    std::vector<size_t> vgd {global_work_size, 1, 1};

    handle.GetKernel(algorithm_name,
            network_config,
            program_name,
            kernel_name,
            vld,
            vgd,
            ""); /* jn : removed -w, tinygemm kernels should not generate warnings */

    
    /* the beta kernel is part of the solution */
    if(soln.v_tgks.size() == 2)
    {      
        std::string beta_program_name = soln.v_tgks[0].kernstr;
        std::string beta_kernel_name = soln.v_tgks[0].fname;
        local_work_size = soln.v_tgks[0].local_work_size;
        global_work_size = soln.v_tgks[0].local_work_size;

        EnableBetaKernel(true);

        vld[0] = local_work_size;
        vgd[0] = global_work_size;

        handle.GetKernel(algorithm_name+"_beta",
                network_config, /* jn : different network_configs require different beta kernels */
                beta_program_name,
                beta_kernel_name,
                vld,
                vgd,
                "");
   }

    gemm_geo_map()[std::make_pair(algorithm_name, network_config)] = *this;
}

void GemmGeometry::RunGemm(Handle &handle,
            ConstData_t     a,
            ConstData_t     b,
            Data_t          c,
            int             a_offset,
            int             b_offset,
            int             c_offset)
{
    std::string network_config = tgg.get_networkconfig_string();

    if(beta_kern_req) {
        handle.GetKernel(algorithm_name+"_beta", network_config) (c, c_offset, beta);
        handle.GetKernel(algorithm_name, network_config) (a, a_offset, b, b_offset, c, c_offset, alpha);
    }
    else {
      handle.GetKernel(algorithm_name, network_config) (a, a_offset, b, b_offset, c, c_offset, alpha, beta);
    }
}

} // namespace miopen
