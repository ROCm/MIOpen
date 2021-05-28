#include <miopen/fusion.hpp>
#include <miopen/any_solver.hpp>

namespace miopen {

namespace fusion {

bool IsWinograd(const std::vector<solver::AnySolver>& ss)
{
    assert(ss.size() == 1);
    auto solverId = ss[0].GetSolverDbId();
    return (solverId == "ConvBinWinogradRxSFused" || solverId == "ConvBinWinogradRxSf2x3g1Fused");
}

} // namespace fusion

miopenStatus_t FusionOpDescriptor::GetNetworkConfig(std::string& /*network_config*/,
                                                    Handle& /*handle*/)
{
    return miopenStatusSuccess;
}

miopenStatus_t
FusionOpDescriptor::GetCompileParms(std::string& /*compile_config*/,
                                    Handle& /*handle*/,
                                    const FusionKernelSourceType /*source*/,
                                    const std::vector<solver::AnySolver>& /*solvers*/)
{
    MIOPEN_LOG_I2("");
    return miopenStatusSuccess;
}

std::vector<size_t> FusionOpDescriptor::GetLocalWGSz(Handle& /*handle*/,
                                                     std::string /*algorithm_name*/)
{
    MIOPEN_THROW("Op does not support local workgroup size");
}

std::vector<size_t> FusionOpDescriptor::GetGlobalWGSz(Handle& /*handle*/,
                                                      std::string /*algorithm_name*/)
{
    MIOPEN_THROW("Op does not support global workgroup size");
}

miopenStatus_t BiasFusionOpDescriptor::GetNetworkConfig(std::string& network_config,
                                                        Handle& /*handle*/)
{
    network_config += "biasOn"; // for bias
    return miopenStatusSuccess;
}

miopenStatus_t
BiasFusionOpDescriptor::GetCompileParms(std::string& compile_config,
                                        Handle& /*handle*/,
                                        FusionKernelSourceType source,
                                        const std::vector<solver::AnySolver>& solvers)
{
    std::string add;
    switch(source)
    {
    case AsmText:
        if(!fusion::IsWinograd(solvers))
            add = " -Wa,-defsym,bias_mode=" + std::to_string(1);
        break;
    case OpenclText: add = " -DMLO_CONV_BIAS=" + std::to_string(1); break;
    case Binary: break;
    }
    MIOPEN_LOG_I2(add);
    compile_config += add;
    return miopenStatusSuccess;
}

std::vector<size_t> BiasFusionOpDescriptor::GetLocalWGSz(Handle& /*handle*/,
                                                         std::string /*algorithm_name*/)
{
    MIOPEN_THROW("Op does not support local workgroup size");
}

std::vector<size_t> BiasFusionOpDescriptor::GetGlobalWGSz(Handle& /*handle*/,
                                                          std::string /*algorithm_name*/)
{
    MIOPEN_THROW("Op does not support global workgroup size");
}

miopenStatus_t ActivFwdFusionOpDescriptor::GetNetworkConfig(std::string& network_config,
                                                            Handle& /*handle*/)
{
    network_config += "ActivFwd" + std::to_string(activMode);
    return miopenStatusSuccess;
}

miopenStatus_t
ActivFwdFusionOpDescriptor::GetCompileParms(std::string& compile_config,
                                            Handle& /*handle*/,
                                            const FusionKernelSourceType source,
                                            const std::vector<solver::AnySolver>& solvers)
{
    std::string add;
    switch(source)
    {
    case AsmText:
        if(!fusion::IsWinograd(solvers))
            add = " -Wa,-defsym,enable_activ=1 -Wa,-defsym,activ_mode=" + std::to_string(activMode);
        break;
    case OpenclText:
        add = " -DMIOPEN_YES_ACTIV=1 -DMIOPEN_NRN_OP_ID=" + std::to_string(activMode);
        break;
    case Binary: break;
    }
    compile_config += add;
    MIOPEN_LOG_I2(add);
    return miopenStatusSuccess;
}

std::vector<size_t> ActivFwdFusionOpDescriptor::GetLocalWGSz(Handle& /*handle*/,
                                                             std::string /*algorithm_name*/)
{
    MIOPEN_THROW("Op does not support local workgroup size");
}

std::vector<size_t> ActivFwdFusionOpDescriptor::GetGlobalWGSz(Handle& /*handle*/,
                                                              std::string /*algorithm_name*/)
{
    MIOPEN_THROW("Op does not support global workgroup size");
}

// Activations backward prop ----------------------------
miopenStatus_t ActivBwdFusionOpDescriptor::GetNetworkConfig(std::string& network_config,
                                                            Handle& /*handle*/)
{
    network_config += "ActivBwd" + std::to_string(activMode);
    return miopenStatusSuccess;
}

miopenStatus_t
ActivBwdFusionOpDescriptor::GetCompileParms(std::string& compile_config,
                                            Handle& /*handle*/,
                                            const FusionKernelSourceType source,
                                            const std::vector<solver::AnySolver>& solvers)
{
    std::string add;
    switch(source)
    {
    case AsmText:
        if(!fusion::IsWinograd(solvers))
            add = " -Wa,-defsym,enable_activ=1 -Wa,-defsym,activ_mode=" + std::to_string(activMode);
        break;
    case OpenclText:
        add = " -DMIOPEN_YES_ACTIV=1 -DMIOPEN_NRN_OP_ID=" + std::to_string(activMode);
        break;
    case Binary: break;
    }
    compile_config += add;
    MIOPEN_LOG_I2(add);
    return miopenStatusSuccess;
}

std::vector<size_t> ActivBwdFusionOpDescriptor::GetLocalWGSz(Handle& /*handle*/,
                                                             std::string /*algorithm_name*/)
{
    MIOPEN_THROW("Op does not support local workgroup size");
}

std::vector<size_t> ActivBwdFusionOpDescriptor::GetGlobalWGSz(Handle& /*handle*/,
                                                              std::string /*algorithm_name*/)
{
    MIOPEN_THROW("Op does not support global workgroup size");
}

// END Activation bwd---------------------------------

/// BATCH NORMALIZATION inference start ================

miopenStatus_t BatchNormInferenceFusionOpDescriptor::GetNetworkConfig(std::string& network_config,
                                                                      Handle& /*handle*/)
{
    network_config += "bn" + std::to_string(mode);
    return miopenStatusSuccess;
}

miopenStatus_t BatchNormInferenceFusionOpDescriptor::GetCompileParms(
    std::string& compile_config,
    Handle& /*handle*/,
    FusionKernelSourceType source,
    const std::vector<solver::AnySolver>& /*solvers*/)
{
    if(source != OpenclText)
    {
        MIOPEN_THROW("Invalid source file type");
    }
    std::vector<size_t> vld{256, 1, 1};
    std::string add;
    if(mode == miopenBNSpatial)
        add += " -DSPATIAL_BN";
    else if(mode == miopenBNPerActivation)
        add += " -DPERACT_BN";

    if(input_desc.GetLengths().empty())
        MIOPEN_THROW("The input descriptor is not set");

    int n, c, h, w;

    // The output_desc should be fully formed by this stage.
    std::tie(n, c, h, w) = tien<4>(input_desc.GetLengths());
    size_t read_unit = 1;
    size_t read_len  = (mode == miopenBNSpatial) ? h * w : c * h * w;

    if(mode == miopenBNSpatial && input_desc.GetType() != miopenHalf)
    {
        read_unit = (read_len % 4 == 0) ? 4 : (read_len % 2 == 0) ? 2 : 1;
    }

    if(input_desc.GetType() == miopenHalf)
    {
        add += " -DMIOPEN_USE_FPMIX=1";
    }

    add += " -DMIO_BN_CHW=" + std::to_string(c * h * w) + " -DMIO_BN_HW=" + std::to_string(h * w) +
           " -DMIO_BN_N=" + std::to_string(n) + " -DMIO_BN_GRP0=" + std::to_string(vld.at(0)) +
           " -DMIO_BN_GRP1=" + std::to_string(1) + " -DMIO_BN_GRP2=" + std::to_string(1);

    std::string READ_TYPE = (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string(read_unit);
    add += " -DMIOPEN_READ_UNIT=" + std::to_string(read_unit);
    add += " -DMIOPEN_READ_TYPE=" + READ_TYPE;
    compile_config += add;
    MIOPEN_LOG_I2(add);
    return miopenStatusSuccess;
}

std::vector<size_t>
BatchNormInferenceFusionOpDescriptor::GetLocalWGSz(Handle& /*handle*/,
                                                   std::string /*algorithm_name*/)
{
    std::vector<size_t> vld{256, 1, 1};
    return vld;
}

std::vector<size_t>
BatchNormInferenceFusionOpDescriptor::GetGlobalWGSz(Handle& /*handle*/,
                                                    std::string /*algorithm_name*/)
{
    if(input_desc.GetLengths().empty())
    {
        MIOPEN_THROW("Compile called for Fusion Plan without setting operator parameters");
    }

    int n, c, h, w;

    // The output_desc should be fully formed by this stage.
    std::tie(n, c, h, w) = tien<4>(input_desc.GetLengths());
    size_t read_unit = 1;
    size_t read_len  = (mode == miopenBNSpatial) ? h * w : c * h * w;

    if(mode == miopenBNSpatial && input_desc.GetType() != miopenHalf)
    {
        read_unit = (read_len % 4 == 0) ? 4 : (read_len % 2 == 0) ? 2 : 1;
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
/// END BN inference ------------------------------------------

// BN Bwd Training start
void BatchNormBwdTrainFusionOpDescriptor::calcBNParams(Handle& handle,
                                                       std::vector<size_t> in_lens,
                                                       int& variant,
                                                       size_t& in_cstride,
                                                       size_t& in_nstride,
                                                       size_t& in_nchw,
                                                       unsigned int& ldsgcn,
                                                       unsigned int& ldsnogcn)
{
    size_t xlocalsize, ylocalsize, zlocalsize;
    std::tie(xlocalsize, ylocalsize, zlocalsize) = tien<3>(GetLocalWGSz(handle, ""));
    size_t zgridsize, ygridsize, xgridsize;
    std::tie(xgridsize, ygridsize, zgridsize) = tien<3>(GetGlobalWGSz(handle, ""));
    int n, c, h, w;
    variant = 0;
    std::tie(n, c, h, w) = tien<4>(in_lens);
    in_cstride = h * w;
    in_nstride = c * in_cstride;
    in_nchw    = n * in_nstride;

    variant = 0;

    if(mode == miopenBNSpatial)
    {
        ldsgcn   = xlocalsize / 64;
        ldsnogcn = xlocalsize;
        if(in_cstride > 1024)
        {
            variant = 1;
        }
        else if(in_cstride > 512)
        {
            variant = (n >= 32) ? 1 : 3;
        }
        else
        {
            variant = 0;
        }
    }
}
miopenStatus_t BatchNormBwdTrainFusionOpDescriptor::GetNetworkConfig(std::string& network_config,
                                                                     Handle& handle)
{
    int n, c, h, w;
    int variant = 0;
    std::tie(n, c, h, w) = tien<4>(input_desc.GetLengths());
    size_t in_cstride, in_nstride, in_nchw;
    size_t xlocalsize, ylocalsize, zlocalsize;
    std::tie(xlocalsize, ylocalsize, zlocalsize) = tien<3>(GetLocalWGSz(handle, ""));
    size_t zgridsize, ygridsize, xgridsize;
    std::tie(xgridsize, ygridsize, zgridsize) = tien<3>(GetGlobalWGSz(handle, ""));
    unsigned int ldsgcn   = 0;
    unsigned int ldsnogcn = 0;
    calcBNParams(handle,
                 input_desc.GetLengths(),
                 variant,
                 in_cstride,
                 in_nstride,
                 in_nchw,
                 ldsgcn,
                 ldsnogcn);

    if(input_desc.GetLengths().empty())
        MIOPEN_THROW("The input descriptor is not set");

    network_config += "variant" + std::to_string(variant) + "gx" + std::to_string(xgridsize) +
                      "gcn" + std::to_string(ldsgcn) + "gy" + std::to_string(ygridsize) + "lx" +
                      std::to_string(xlocalsize) + "ly" + std::to_string(ylocalsize) + "bn" +
                      std::to_string(mode) + "n" + std::to_string(n) + "cstride" +
                      std::to_string(in_cstride) + "nstride" + std::to_string(in_nstride) + "c" +
                      std::to_string(c) + "nchw" + std::to_string(in_nchw);

    return miopenStatusSuccess;
}

miopenStatus_t BatchNormBwdTrainFusionOpDescriptor::GetCompileParms(
    std::string& compile_config,
    Handle& handle,
    FusionKernelSourceType /*source*/,
    const std::vector<solver::AnySolver>& /*solvers*/)
{
    std::string add;
    int n, c, h, w;
    int variant = 0;
    std::tie(n, c, h, w) = tien<4>(input_desc.GetLengths());
    size_t in_cstride, in_nstride, in_nchw;
    size_t xlocalsize, ylocalsize, zlocalsize;
    std::tie(xlocalsize, ylocalsize, zlocalsize) = tien<3>(GetLocalWGSz(handle, ""));
    size_t zgridsize, ygridsize, xgridsize;
    std::tie(xgridsize, ygridsize, zgridsize) = tien<3>(GetGlobalWGSz(handle, ""));
    unsigned int ldsgcn   = 0;
    unsigned int ldsnogcn = 0;
    calcBNParams(handle,
                 input_desc.GetLengths(),
                 variant,
                 in_cstride,
                 in_nstride,
                 in_nchw,
                 ldsgcn,
                 ldsnogcn);

    if(input_desc.GetLengths().empty())
        MIOPEN_THROW("The input descriptor is not set");

    if(input_desc.GetType() == miopenHalf)
    {
        add += " -DMIOPEN_USE_FPMIX=1";
    }

    add += " -DMIO_BN_N=" + std::to_string(n) + " -DMIO_BN_C=" + std::to_string(c) +
           " -DMIO_BN_HW=" + std::to_string(in_cstride) + " -DMIO_BN_NHW=" +
           std::to_string(n * h * w) + " -DMIO_BN_CHW=" + std::to_string(in_nstride) +
           " -DMIO_BN_NCHW=" + std::to_string(in_nchw) + " -DMIO_BN_GRP0=" +
           std::to_string(xlocalsize) + " -DMIO_BN_GRP1=" + std::to_string(ylocalsize) +
           " -DMIO_BN_GRP2=" + std::to_string(zlocalsize) + " -DMIO_BN_LDS_SIZE=" +
           std::to_string(ldsnogcn) + " -DMIO_BN_LDSGCN_SIZE=" + std::to_string(ldsgcn) +
           " -DMIO_BN_USESAVED=" + std::to_string(static_cast<int>(true)) + " -DMIO_BN_VARIANT=" +
           std::to_string(variant) + " -DMIO_BN_CBA_WRITE_INTERMEDIATE=" + std::to_string(0) +
           " -DMIO_BN_GFX1030=" + ((handle.GetDeviceName() == "gfx1030") ? "1" : "0");

    compile_config += add;
    MIOPEN_LOG_I2(add);
    return miopenStatusSuccess;
}

std::vector<size_t>
BatchNormBwdTrainFusionOpDescriptor::GetLocalWGSz(Handle& /*handle*/,
                                                  std::string /*algorithm_name*/)
{
    size_t xlocalsize, ylocalsize, zlocalsize;
    int h, w;
    std::tie(std::ignore, std::ignore, h, w) = tien<4>(input_desc.GetLengths());
    size_t in_cstride = h * w;

    xlocalsize = 1;
    ylocalsize = 1;
    zlocalsize = 1;

    if(mode == miopenBNSpatial)
    {
        if(in_cstride <= 1024 && in_cstride > 512)
        {
            xlocalsize = std::min(64 * ((in_cstride + 63) / 64), static_cast<unsigned long>(1024));
        }
        else
        {
            xlocalsize = 1024;
        }
    }
    else
    {
        ylocalsize = (64 >= in_cstride) ? 64 : 256;
    }
    return {xlocalsize, ylocalsize, zlocalsize};
}

std::vector<size_t> BatchNormBwdTrainFusionOpDescriptor::GetGlobalWGSz(Handle& handle,
                                                                       std::string algorithm_name)
{
    int c, h, w;
    std::tie(std::ignore, c, h, w) = tien<4>(input_desc.GetLengths());
    size_t xlocalsize, ylocalsize;
    std::tie(xlocalsize, ylocalsize, std::ignore) = tien<3>(GetLocalWGSz(handle, algorithm_name));

    size_t xgridsize = 1;
    size_t zgridsize = 1;
    size_t ygridsize = 1;

    size_t in_cstride = h * w;

    if(mode == miopenBNSpatial)
    {
        if(in_cstride > 512)
        {
            xgridsize = c * xlocalsize;
        }
        else
        {
            xgridsize = 1024 * c;
        }
    }
    else
    {
        auto segment = int(std::ceil(double(in_cstride) / double(ylocalsize)));
        xgridsize    = c;
        ygridsize    = segment * ylocalsize;
    }
    std::vector<size_t> vgd{xgridsize, ygridsize, zgridsize};
    return vgd;
}

// BN Bwd Training end

/// BATCH NORMALIZATION training forward start ================

void BatchNormFwdTrainFusionOpDescriptor::calcBNParams(Handle& handle,
                                                       std::vector<size_t> in_lens,
                                                       int& variant,
                                                       size_t& in_cstride,
                                                       size_t& in_nstride,
                                                       size_t& in_nchw,
                                                       unsigned int& ldsgcn,
                                                       unsigned int& ldsnogcn)
{
    size_t xlocalsize, ylocalsize, zlocalsize;
    std::tie(xlocalsize, ylocalsize, zlocalsize) = tien<3>(GetLocalWGSz(handle, ""));
    size_t zgridsize, ygridsize, xgridsize;
    std::tie(xgridsize, ygridsize, zgridsize) = tien<3>(GetGlobalWGSz(handle, ""));
    int n, c, h, w;
    variant = 0;
    std::tie(n, c, h, w) = tien<4>(in_lens);
    in_cstride = h * w;
    in_nstride = c * in_cstride;
    in_nchw    = n * in_nstride;

    ldsgcn   = xlocalsize / 64;
    ldsnogcn = xlocalsize;

    variant = 0;

    if(mode == miopenBNSpatial)
    {
        if(in_cstride > 1024)
        {
            variant = 1;
        }
        else if(in_cstride > 512)
        {
            variant = 3;
        }
    }
}

miopenStatus_t BatchNormFwdTrainFusionOpDescriptor::GetNetworkConfig(std::string& network_config,
                                                                     Handle& handle)
{
    int n, c, h, w;
    int variant               = 0;
    const bool saveBatchStats = true;
    bool savePopStats         = runningMeanVar;
    std::tie(n, c, h, w) = tien<4>(input_desc.GetLengths());
    size_t in_cstride, in_nstride, in_nchw;
    size_t xlocalsize, ylocalsize, zlocalsize;
    std::tie(xlocalsize, ylocalsize, zlocalsize) = tien<3>(GetLocalWGSz(handle, ""));
    size_t zgridsize, ygridsize, xgridsize;
    std::tie(xgridsize, ygridsize, zgridsize) = tien<3>(GetGlobalWGSz(handle, ""));
    unsigned int ldsgcn, ldsnogcn;
    calcBNParams(handle,
                 input_desc.GetLengths(),
                 variant,
                 in_cstride,
                 in_nstride,
                 in_nchw,
                 ldsgcn,
                 ldsnogcn);

    if(input_desc.GetLengths().empty())
        MIOPEN_THROW("The input descriptor is not set");

    network_config += "variant" + std::to_string(variant) + "gx" + std::to_string(xgridsize) +
                      "gcn" + std::to_string(ldsgcn) + "gy" + std::to_string(ygridsize) + "lx" +
                      std::to_string(xlocalsize) + "ly" + std::to_string(ylocalsize) + "bn" +
                      std::to_string(mode) + "sbs" +
                      std::to_string(static_cast<int>(saveBatchStats)) + "sps" +
                      std::to_string(static_cast<int>(savePopStats)) + "n" + std::to_string(n) +
                      "hw" + std::to_string(in_cstride) + "chw" + std::to_string(in_nstride);

    return miopenStatusSuccess;
}

miopenStatus_t BatchNormFwdTrainFusionOpDescriptor::GetCompileParms(
    std::string& compile_config,
    Handle& handle,
    FusionKernelSourceType /*source*/,
    const std::vector<solver::AnySolver>& /*solvers*/)
{
    std::string add;
    int n, c, h, w;
    int variant               = 0;
    const bool saveBatchStats = true;
    bool savePopStats         = runningMeanVar;
    std::tie(n, c, h, w) = tien<4>(input_desc.GetLengths());
    size_t in_cstride, in_nstride, in_nchw;
    size_t xlocalsize, ylocalsize, zlocalsize;
    std::tie(xlocalsize, ylocalsize, zlocalsize) = tien<3>(GetLocalWGSz(handle, ""));
    size_t zgridsize, ygridsize, xgridsize;
    std::tie(xgridsize, ygridsize, zgridsize) = tien<3>(GetGlobalWGSz(handle, ""));
    unsigned int ldsgcn, ldsnogcn;
    calcBNParams(handle,
                 input_desc.GetLengths(),
                 variant,
                 in_cstride,
                 in_nstride,
                 in_nchw,
                 ldsgcn,
                 ldsnogcn);

    if(input_desc.GetLengths().empty())
        MIOPEN_THROW("The input descriptor is not set");

    size_t read_unit = 0;
    size_t read_len  = (mode == miopenBNSpatial) ? in_cstride : in_nstride;
    if(mode == miopenBNSpatial)
    {
        add += " -DSPATIAL_BN";
        read_unit = (read_len % 4 == 0) ? 4 : (read_len % 2 == 0) ? 2 : 1;
    }
    else
    {
        add += " -DPERACT_BN";
        read_unit = 1;
    }
    std::string READ_TYPE = (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string(read_unit);

    if(input_desc.GetType() == miopenHalf)
    {
        add += " -DMIOPEN_USE_FPMIX=1";
    }

    add += " -DMIO_BN_N=" + std::to_string(n) + " -DMIO_BN_C=" + std::to_string(c) +
           " -DMIO_BN_HW=" + std::to_string(in_cstride) + " -DMIO_BN_NHW=" +
           std::to_string(n * h * w) + " -DMIO_BN_CHW=" + std::to_string(in_nstride) +
           " -DMIO_BN_NCHW=" + std::to_string(in_nchw) + " -DMIO_BN_GRP0=" +
           std::to_string(xlocalsize) + " -DMIO_BN_GRP1=" + std::to_string(ylocalsize) +
           " -DMIO_BN_GRP2=" + std::to_string(zlocalsize) + " -DMIO_BN_LDS_SIZE=" +
           std::to_string(ldsnogcn) + " -DMIO_BN_LDSGCN_SIZE=" + std::to_string(ldsgcn) +
           " -DMIOPEN_READ_UNIT=" + std::to_string(read_unit) + " -DMIOPEN_READ_TYPE=" + READ_TYPE +
           " -DMIO_SAVE_MEAN_VARIANCE=" + (saveBatchStats ? "1" : "0") + " -DMIO_RUNNING_RESULT=" +
           ((savePopStats) ? "1" : "0") + " -DMIO_BN_VARIANT=" + std::to_string(variant) +
           " -DMIO_BN_GFX1030=" + ((handle.GetDeviceName() == "gfx1030") ? "1" : "0");

    compile_config += add;
    MIOPEN_LOG_I2(add);
    return miopenStatusSuccess;
}

std::vector<size_t>
BatchNormFwdTrainFusionOpDescriptor::GetLocalWGSz(Handle& /*handle*/,
                                                  std::string /*algorithm_name*/)
{
    size_t xlocalsize, ylocalsize, zlocalsize;
    int h, w;
    std::tie(std::ignore, std::ignore, h, w) = tien<4>(input_desc.GetLengths());
    size_t in_cstride = h * w;

    xlocalsize = 1024;
    ylocalsize = 1;
    zlocalsize = 1;

    if(mode == miopenBNSpatial)
    {
        if((in_cstride <= 1024) && (in_cstride > 512))
        {
            xlocalsize = 64 * ((in_cstride + 63) / 64);
        }
    }
    else
    {
        xlocalsize = 1;
        ylocalsize = 256;
    }
    std::vector<size_t> vgd{xlocalsize, ylocalsize, zlocalsize};
    return vgd;
}

std::vector<size_t> BatchNormFwdTrainFusionOpDescriptor::GetGlobalWGSz(Handle& handle,
                                                                       std::string algorithm_name)
{
    int c, h, w;
    std::tie(std::ignore, c, h, w) = tien<4>(input_desc.GetLengths());
    size_t xlocalsize, ylocalsize, zlocalsize;
    std::tie(xlocalsize, ylocalsize, zlocalsize) = tien<3>(GetLocalWGSz(handle, algorithm_name));

    size_t xgridsize = c * xlocalsize;
    size_t zgridsize = 1;
    size_t ygridsize = 1;

    size_t in_cstride = h * w;

    if(mode != miopenBNSpatial)
    {
        auto segment = int(std::ceil(double(in_cstride) / double(ylocalsize)));
        xgridsize    = c;
        ygridsize    = segment * ylocalsize;
    }
    std::vector<size_t> vgd{xgridsize, ygridsize, zgridsize};
    return vgd;
}

std::string BatchNormFwdTrainFusionOpDescriptor::GetArgKey(const std::string& k) const
{
    return k + std::to_string(GetIdx());
}
bool BatchNormFwdTrainFusionOpDescriptor::GetOpAttr(const std::string& sym, int& val) const
{
    if(sym == "bn_mode")
    {
        val = mode;
        return true;
    }
    else
    {
        return false;
    }
}
OpKernelArg BatchNormFwdTrainFusionOpDescriptor::GetOpAttr(const std::string& k) const
{
    int v;
    if(GetOpAttr(k, v))
    {
        return OpKernelArg(v);
    }
    else if(k == "diff_scale")
    {
        return OpKernelArg(static_cast<float>(0.0));
    }
    else if(k == "iNHW")
    {
        int n, h, w;
        std::tie(n, std::ignore, h, w) = tien<4>(input_desc.GetLengths());
        auto nhw = static_cast<float>(n * h * w);
        return OpKernelArg(static_cast<float>(1.0f / nhw));
    }
    else
        MIOPEN_THROW("BatchNormFwdTrainFusionOpDescriptor does not support attribute: " + k);
}
/// END BN traing forward

} // namespace miopen
