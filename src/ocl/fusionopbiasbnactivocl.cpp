#include <miopen/fusion.hpp>

namespace miopen {
miopenStatus_t FusionOpDescriptor::GetNetworkConfig(std::string& network_config, Handle& handle)
{
    (void)(network_config);
    (void)(handle);
    return miopenStatusSuccess;
}

miopenStatus_t
FusionOpDescriptor::GetCompileParms(std::string& compile_config, Handle& handle, const FusionKernelSourceType source)
{
    (void)(compile_config);
    (void)(handle);
    (void)(source);
    MIOPEN_LOG_I2("");
    return miopenStatusSuccess;
}

std::vector<size_t> FusionOpDescriptor::GetLocalWGSz(Handle& handle, std::string algorithm_name)
{
    (void)handle;
    (void)(algorithm_name);
    MIOPEN_THROW("Op does not support local workgroup size");
}

std::vector<size_t> FusionOpDescriptor::GetGlobalWGSz(Handle& handle, std::string algorithm_name)
{
    (void)handle;
    (void)(algorithm_name);
    MIOPEN_THROW("Op does not support global workgroup size");
}

miopenStatus_t BiasFusionOpDescriptor::GetNetworkConfig(std::string& network_config, Handle& handle)
{
    (void)(handle);
    network_config += "biasOn"; // for bias
    return miopenStatusSuccess;
}

miopenStatus_t
BiasFusionOpDescriptor::GetCompileParms(std::string& compile_config, Handle& handle, const FusionKernelSourceType source)
{
    (void)(handle); // only convolution uses handle
    std::string add;
    switch (source)
    {
    case AsmText:
        add = " -Wa,-defsym,bias_mode=" + std::to_string(1);
        break;
    case OpenclText:
        add = " -DMLO_CONV_BIAS=" + std::to_string(1);
        break;
    case Binary: break;
    }
    MIOPEN_LOG_I2(add);
    compile_config += add;
    return miopenStatusSuccess;
}

std::vector<size_t> BiasFusionOpDescriptor::GetLocalWGSz(Handle& handle, std::string algorithm_name)
{
    (void)handle;
    (void)(algorithm_name);
    MIOPEN_THROW("Op does not support local workgroup size");
}

std::vector<size_t> BiasFusionOpDescriptor::GetGlobalWGSz(Handle& handle,
                                                          std::string algorithm_name)
{
    (void)handle;
    (void)(algorithm_name);
    MIOPEN_THROW("Op does not support global workgroup size");
}

miopenStatus_t ActivFusionOpDescriptor::GetNetworkConfig(std::string& network_config,
                                                         Handle& handle)
{
    (void)handle;
    network_config += "Activ" + std::to_string(activMode);
    return miopenStatusSuccess;
}

miopenStatus_t
ActivFusionOpDescriptor::GetCompileParms(std::string& compile_config, Handle& handle, const FusionKernelSourceType source)
{
    (void)handle;
    std::string add;
    switch (source)
    {
    case AsmText:
        add += " -Wa,-defsym,activ_mode=" + std::to_string(activMode);
        break;
    case OpenclText:
        add += " -DMIOPEN_YES_ACTIV=1 -DMIOPEN_NRN_OP_ID=" + std::to_string(activMode);
        break;
    case Binary: break;
    }
    compile_config += add;
    MIOPEN_LOG_I2(add);
    return miopenStatusSuccess;
}

std::vector<size_t> ActivFusionOpDescriptor::GetLocalWGSz(Handle& handle,
                                                          std::string algorithm_name)
{
    (void)handle;
    (void)(algorithm_name);
    MIOPEN_THROW("Op does not support local workgroup size");
}

std::vector<size_t> ActivFusionOpDescriptor::GetGlobalWGSz(Handle& handle,
                                                           std::string algorithm_name)
{
    (void)handle;
    (void)(algorithm_name);
    MIOPEN_THROW("Op does not support global workgroup size");
}

miopenStatus_t BatchNormInferenceFusionOpDescriptor::GetNetworkConfig(std::string& network_config,
                                                                      Handle& handle)
{
    (void)(handle);
    network_config += "bn" + std::to_string(mode);
    return miopenStatusSuccess;
}

miopenStatus_t BatchNormInferenceFusionOpDescriptor::GetCompileParms(std::string& compile_config,
                                                                     Handle& handle,
                                                                     const FusionKernelSourceType source)
{
    (void)(handle); // only convolution uses handle
    (void)source;
    assert(source == OpenclText);
    std::vector<size_t> vld{256, 1, 1};
    std::string add;
    if(mode == miopenBNSpatial)
        add += " -DSPATIAL_BN";
    else if(mode == miopenBNPerActivation)
        add += " -DPERACT_BN";

    if(input_desc.GetLengths().size() == 0)
        MIOPEN_THROW("The input descriptor is not set");

    int n, c, h, w;

    // The output_desc should be fully formed by this stage.
    std::tie(n, c, h, w) = tien<4>(input_desc.GetLengths());
    size_t read_unit = 0;
    size_t read_len  = (mode == miopenBNSpatial) ? h * w : c * h * w;

    if(mode /*ops_head->mode*/ == miopenBNSpatial)
    {
        read_unit = (read_len % 4 == 0) ? 4 : (read_len % 2 == 0) ? 2 : 1;
    }
    else
    {
        read_unit = 1;
    }
    add += " -DMIO_BN_CHW=" + std::to_string(c * h * w) + " -DMIO_BN_HW=" +
                      std::to_string(h * w) + " -DMIO_BN_N=" + std::to_string(n) +
                      " -DMIO_BN_GRP0=" + std::to_string(vld.at(0)) + " -DMIO_BN_GRP1=" +
                      std::to_string(1) + " -DMIO_BN_GRP2=" + std::to_string(1);

    std::string READ_TYPE = (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string(read_unit);
    add += " -DMIOPEN_READ_UNIT=" + std::to_string(read_unit);
    add += " -DMIOPEN_READ_TYPE=" + READ_TYPE;
    compile_config += add;
    MIOPEN_LOG_I2(add);
    return miopenStatusSuccess;
}

std::vector<size_t> BatchNormInferenceFusionOpDescriptor::GetLocalWGSz(Handle& handle,
                                                                       std::string algorithm_name)
{
    (void)handle;
    (void)algorithm_name;
    std::vector<size_t> vld{256, 1, 1};
    return vld;
}

std::vector<size_t> BatchNormInferenceFusionOpDescriptor::GetGlobalWGSz(Handle& handle,
                                                                        std::string algorithm_name)
{
    (void)handle;
    (void)algorithm_name;

    if(input_desc.GetLengths().size() == 0)
    {
        MIOPEN_THROW("Compile called for Fusion Plan without setting operator parameters");
    }

    int n, c, h, w;

    // The output_desc should be fully formed by this stage.
    std::tie(n, c, h, w) = tien<4>(input_desc.GetLengths());
    size_t read_unit = 0;
    size_t read_len  = (mode == miopenBNSpatial) ? h * w : c * h * w;

    if(mode /*ops_head->mode*/ == miopenBNSpatial)
    {
        read_unit = (read_len % 4 == 0) ? 4 : (read_len % 2 == 0) ? 2 : 1;
    }
    else
    {
        read_unit = 1;
    }

    size_t xgridsize = read_len / read_unit;
    size_t ygridsize = (mode == miopenBNSpatial) ? size_t(c) : 1;
    size_t zgridsize = 1;

    std::vector<size_t> vgd{};
    vgd.push_back(xgridsize);
    vgd.push_back(ygridsize);
    vgd.push_back(zgridsize);

    return vgd;
}

} // namespace miopen
