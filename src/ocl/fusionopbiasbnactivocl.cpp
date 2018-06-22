#include <miopen/fusion.hpp>

namespace miopen {
miopenStatus_t FusionOpDescriptor::GetNetworkConfig(std::string& network_config, Handle& handle)
{
    (void)(network_config);
    (void)(handle);
    return miopenStatusSuccess;
}

miopenStatus_t
FusionOpDescriptor::GetCompileParms(std::string& compile_config, Handle& handle, bool is_asm)
{
    (void)(compile_config);
    (void)(handle);
    (void)(is_asm);
    return miopenStatusSuccess;
}

miopenStatus_t BiasFusionOpDescriptor::GetNetworkConfig(std::string& network_config, Handle& handle)
{
    (void)(handle);
    network_config += "biasOn"; // for bias
    return miopenStatusSuccess;
}

miopenStatus_t
BiasFusionOpDescriptor::GetCompileParms(std::string& compile_config, Handle& handle, bool is_asm)
{
    (void)(handle); // only convolution uses handle
    if(is_asm)
    {
        compile_config += " -Wa,-defsym,bias_mode=" + std::to_string(1);
    }
    else
    {
        compile_config += " -DMLO_CONV_BIAS=" + std::to_string(1);
    }
    return miopenStatusSuccess;
}

miopenStatus_t ActivFusionOpDescriptor::GetNetworkConfig(std::string& network_config,
                                                         Handle& handle)
{
    (void)(handle);
    network_config += "Activ" + std::to_string(activMode);
    return miopenStatusSuccess;
}

miopenStatus_t
ActivFusionOpDescriptor::GetCompileParms(std::string& compile_config, Handle& handle, bool is_asm)
{
    (void)(handle);
    if(is_asm)
    {
        compile_config += " -Wa,-defsym,activ_mode=" + std::to_string(activMode);
    }
    else
    {
        compile_config += " -DMIOPEN_YES_ACTIV=1 -DMIOPEN_NRN_OP_ID=" + std::to_string(activMode);
    }
    return miopenStatusSuccess;
}

miopenStatus_t BatchNormInferenceFusionOpDescriptor::GetNetworkConfig(std::string& network_config,
                                                                      Handle& handle)
{
    (void)(handle);
    network_config += std::to_string(mode);
    return miopenStatusSuccess;
}

miopenStatus_t BatchNormInferenceFusionOpDescriptor::GetCompileParms(std::string& compile_config,
                                                                     Handle& handle,
                                                                     bool is_asm)
{
    (void)(handle); // only convolution uses handle
    (void)(is_asm);
    compile_config += ""; // No opt parameters for forward inference.
    return miopenStatusSuccess;
}

} // namespace miopen
