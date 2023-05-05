#include <miopen/fusion.hpp>

#include <miopen/gcn_asm_utils.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/solver.hpp>

namespace miopen {

// Conv op in ocl
mlo_construct_direct2D_fusion ConvForwardOpDescriptor::ConstructParams(Handle& handle)
{
    TensorDescriptor o_desc;
    GetOutputDesc(o_desc);
    mlo_construct_direct2D_fusion construct_params(
        input_desc, filter_desc, o_desc, base_desc, miopen::conv::Direction::Forward);
    construct_params.setStream(&handle);
    return construct_params;
}
miopenStatus_t ConvForwardOpDescriptor::GetNetworkConfig(std::stringstream& network_config,
                                                         Handle& handle)
{
    mlo_construct_direct2D_fusion construct_params = ConstructParams(handle);

    std::string conv_config;
    construct_params.mloBuildConf_Key(conv_config);
    network_config << conv_config;
    return miopenStatusSuccess;
}

} // namespace miopen
