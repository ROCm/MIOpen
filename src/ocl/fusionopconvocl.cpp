#include <miopen/fusion.hpp>
#include <miopen/solver.hpp>
#include <miopen/gcn_asm_utils.hpp>

namespace miopen {

// Conv op in ocl
mlo_construct_direct2D_fusion ConvForwardOpDescriptor::ConstructParams(Handle& handle)
{
    TensorDescriptor o_desc;
    GetOutputDesc(o_desc);
    mlo_construct_direct2D_fusion construct_params(
        input_desc, filter_desc, o_desc, base_desc, 1); // forward
    construct_params.setStream(&handle);
    return construct_params;
}
miopenStatus_t ConvForwardOpDescriptor::GetNetworkConfig(std::string& network_config,
                                                         Handle& handle)
{
    mlo_construct_direct2D_fusion construct_params = ConstructParams(handle);

    std::string conv_config;
    construct_params.mloBuildConf_Key(conv_config);
    network_config += conv_config;
    return miopenStatusSuccess;
}

miopenStatus_t
ConvForwardOpDescriptor::GetCompileParms(std::string& compile_config,
                                         Handle& handle,
                                         FusionKernelSourceType source,
                                         const std::vector<solver::AnySolver>& solvers)
{
    mlo_construct_direct2D_fusion construct_params = ConstructParams(handle);
    const auto solution                            = FindFirstSolution(construct_params, solvers);
    if(!solution.Succeeded())
    {
        return solution.status;
    }
    kernel_info           = solution.construction_params[0];
    kernel_info_valid     = true;
    conv_compiler_options = solution.construction_params[0].comp_options;
    compile_config += conv_compiler_options;

    if(source == AsmText && !fusion::IsWinograd(solvers))
    {
        std::ostringstream options;
        GenerateClangDefsym(options, "fusion_mode", 1);
        compile_config += options.str();
    }
    return miopenStatusSuccess;
}
std::vector<size_t> ConvForwardOpDescriptor::GetLocalWGSz(Handle& /*handle*/,
                                                          std::string /*algorithm_name*/)
{
    if(!kernel_info_valid)
    {
        MIOPEN_THROW("GetCompileParms must be called before GetLocalWGSz");
    }
    return kernel_info.l_wk;
}

std::vector<size_t> ConvForwardOpDescriptor::GetGlobalWGSz(Handle& /*handle*/,
                                                           std::string /*algorithm_name*/)
{
    if(!kernel_info_valid)
    {
        MIOPEN_THROW("GetCompileParms must be called before GetGlobalWGSz");
    }
    return kernel_info.g_wk;
}
} // namespace miopen
