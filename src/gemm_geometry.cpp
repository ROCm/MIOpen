#include <mlopen/gemm_geometry.hpp>

namespace mlopen {

std::unordered_map< GemmKey, GemmGeometry, SimpleHash> gemm_geo_map;

void GemmGeometry::EnableBetaKernel(bool enable,
        std::map<std::string, size_t> &beta_args)
{
    beta_kern_req = enable;
    beta_kern_args[0] = beta_args.at("dim_coal");
    beta_kern_args[1] = beta_args.at("dim_uncoal");
}

void GemmGeometry::FindSolution(float time,
        Handle          &handle,
        ConstData_t     a,
        ConstData_t     b,
        Data_t          c,
        bool            enforce_determinism)
{
    // alloted_time, queue, a, b, c, enforce_determinism, float_type, geometry, alpha, beta, verbose 
#if MLOPEN_BACKEND_OPENCL
    TinyGemmSolution soln = tinygemm::find(time, handle.GetStream(), a, b, c, enforce_determinism, 'f', tgg, alpha, beta);
#else
    (void)time;
    (void)a;
    (void)b;
    (void)c;
    (void)tgg;
    (void)alpha;
    (void)beta;
    TinyGemmSolution soln = tinygemm::get_default(enforce_determinism, 'f', tgg);
#endif

    std::string program_name = soln.main_kernel;
    std::string kernel_name = soln.main_kernel_function_name;
    std::string network_config = tgg.get_networkconfig_string();

    // std::cerr << "tinygemm get_networkconfig_string: " << network_config << std::endl;
    // std::cerr << "tinygemm get_hyper_param_string: " << soln.get_hyper_param_string() << std::endl;

    auto main_kernel_worksize_params =  soln.get_main_kernel_worksize_params(dims[0], dims[1]);

    size_t local_work_size = main_kernel_worksize_params.at("local_work_size");
    size_t global_work_size = main_kernel_worksize_params.at("global_work_size");

    std::vector<size_t> vld {local_work_size, 1, 1};
    std::vector<size_t> vgd {global_work_size, 1, 1};

    handle.GetKernel(algorithm_name,
            network_config,
            program_name,
            kernel_name,
            vld,
            vgd,
            "");

    // beta kernel
    if(beta != 1.0 && !soln.betac_kernel.empty())
    {
        std::string beta_program_name = soln.betac_kernel;
        std::string beta_kernel_name = soln.betac_kernel_function_name;
        auto beta_kernel_worksize_params = soln.get_betac_kernel_worksize_params(dims[0], dims[1]);

        local_work_size = beta_kernel_worksize_params.at("local_work_size");
        global_work_size = beta_kernel_worksize_params.at("global_work_size");

        EnableBetaKernel(true, beta_kernel_worksize_params);

        vld[0] = local_work_size;
        vgd[1] = global_work_size;

        handle.GetKernel(algorithm_name+"_beta",
                "placeholder", // TODO: hack for now because the kernel is not cached if config is ""
                beta_program_name,
                beta_kernel_name,
                vld,
                vgd,
                "");
    }

    gemm_geo_map[std::make_pair(algorithm_name, network_config)] = *this;
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

    // beta kernel, if required
    if(beta_kern_req) {
        handle.GetKernel(algorithm_name+"_beta", "placeholder") (beta_kern_args[0], beta_kern_args[1], strides[2], c_offset, c, beta);
    }

    // main kernel
//  c, a, b, alpha, beta, lda, ldb, ldc, m, n, k, a_offset, b_offset, c_offset
    handle.GetKernel(algorithm_name, network_config)(c, a, b,
            alpha, beta,
            strides[0], strides[1], strides[2],
            dims[0], dims[1], dims[2],
            a_offset, b_offset, c_offset);
}

} // namespace mlopen
