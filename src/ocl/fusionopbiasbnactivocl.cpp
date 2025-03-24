/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <miopen/fusion.hpp>
#include <miopen/any_solver.hpp>
#include <miopen/stringutils.hpp>

namespace miopen {

miopenStatus_t FusionOpDescriptor::GetNetworkConfig(std::ostringstream& /*network_config*/)
{
    return miopenStatusSuccess;
}

miopenStatus_t BiasFusionOpDescriptor::GetNetworkConfig(std::ostringstream& network_config)
{
    network_config << "biasOn"; // for bias
    return miopenStatusSuccess;
}

miopenStatus_t TensorScaleAddOpDescriptor::GetNetworkConfig(std::ostringstream& network_config)
{
    network_config << "tensorScaleAdd"; // for bias
    return miopenStatusSuccess;
}

miopenStatus_t ActivFwdFusionOpDescriptor::GetNetworkConfig(std::ostringstream& network_config)
{
    network_config << "ActivFwd" << std::to_string(activMode);
    return miopenStatusSuccess;
}

// Activations backward prop ----------------------------
miopenStatus_t ActivBwdFusionOpDescriptor::GetNetworkConfig(std::ostringstream& network_config)
{
    network_config << "ActivBwd" << std::to_string(activMode);
    return miopenStatusSuccess;
}

// END Activation bwd---------------------------------

/// BATCH NORMALIZATION inference start ================

miopenStatus_t
BatchNormInferenceFusionOpDescriptor::GetNetworkConfig(std::ostringstream& network_config)
{
    network_config << "bn" << std::to_string(mode);
    return miopenStatusSuccess;
}

std::vector<size_t>
BatchNormInferenceFusionOpDescriptor::GetLocalWGSz(const Handle& /*handle*/,
                                                   std::string /*algorithm_name*/)
{
    std::vector<size_t> vld{256, 1, 1};
    return vld;
}

std::vector<size_t>
BatchNormInferenceFusionOpDescriptor::GetGlobalWGSz(const Handle& /*handle*/,
                                                    std::string /*algorithm_name*/)
{
    if(input_desc.GetLengths().empty())
    {
        MIOPEN_THROW("Compile called for Fusion Plan without setting operator parameters");
    }

    int n, c, h, w;

    // The output_desc should be fully formed by this stage.
    std::tie(n, c, h, w) = tien<4>(input_desc.GetLengths());
    size_t read_unit     = 1;
    size_t read_len      = (mode == miopenBNSpatial) ? h * w : c * h * w;

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
void BatchNormBwdTrainFusionOpDescriptor::calcBNParams(std::vector<size_t> in_lens,
                                                       int& variant,
                                                       size_t& in_cstride,
                                                       size_t& in_nstride,
                                                       size_t& in_nchw,
                                                       unsigned int& ldsgcn,
                                                       unsigned int& ldsnogcn)
{
    const auto [xlocalsize, _0, _1] = tien<3>(GetLocalWGSz());
    int n, c, h, w;
    variant              = 0;
    std::tie(n, c, h, w) = tien<4>(in_lens);
    in_cstride           = static_cast<size_t>(h) * w;
    in_nstride           = c * in_cstride;
    in_nchw              = n * in_nstride;

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
miopenStatus_t
BatchNormBwdTrainFusionOpDescriptor::GetNetworkConfig(std::ostringstream& network_config)
{
    int n, c, h, w;
    int variant          = 0;
    std::tie(n, c, h, w) = tien<4>(input_desc.GetLengths());
    size_t in_cstride, in_nstride, in_nchw;
    size_t xlocalsize, ylocalsize, zlocalsize;
    std::tie(xlocalsize, ylocalsize, zlocalsize) = tien<3>(GetLocalWGSz());
    size_t zgridsize, ygridsize, xgridsize;
    std::tie(xgridsize, ygridsize, zgridsize) = tien<3>(GetGlobalWGSz());
    unsigned int ldsgcn                       = 0;
    unsigned int ldsnogcn                     = 0;
    calcBNParams(
        input_desc.GetLengths(), variant, in_cstride, in_nstride, in_nchw, ldsgcn, ldsnogcn);

    if(input_desc.GetLengths().empty())
        MIOPEN_THROW("The input descriptor is not set");

    network_config << "variant" << std::to_string(variant) << "gx" << std::to_string(xgridsize)
                   << "gcn" << std::to_string(ldsgcn) << "gy" << std::to_string(ygridsize) << "lx"
                   << std::to_string(xlocalsize) << "ly" << std::to_string(ylocalsize) << "bn"
                   << std::to_string(mode) << "n" << std::to_string(n) << "cstride"
                   << std::to_string(in_cstride) << "nstride" << std::to_string(in_nstride) << "c"
                   << std::to_string(c) << "nchw" << std::to_string(in_nchw);

    return miopenStatusSuccess;
}

std::vector<size_t> BatchNormBwdTrainFusionOpDescriptor::GetLocalWGSz()
{
    size_t xlocalsize, ylocalsize, zlocalsize;
    int h, w;
    std::tie(std::ignore, std::ignore, h, w) = tien<4>(input_desc.GetLengths());
    size_t in_cstride                        = static_cast<size_t>(h) * w;

    xlocalsize = 1;
    ylocalsize = 1;
    zlocalsize = 1;

    if(mode == miopenBNSpatial)
    {
        if(in_cstride <= 1024 && in_cstride > 512)
        {
            xlocalsize = std::min(64 * ((in_cstride + 63) / 64), static_cast<size_t>(1024));
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

std::vector<size_t> BatchNormBwdTrainFusionOpDescriptor::GetGlobalWGSz()
{
    int c, h, w;
    std::tie(std::ignore, c, h, w) = tien<4>(input_desc.GetLengths());
    size_t xlocalsize, ylocalsize;
    std::tie(xlocalsize, ylocalsize, std::ignore) = tien<3>(GetLocalWGSz());

    size_t xgridsize = 1;
    size_t zgridsize = 1;
    size_t ygridsize = 1;

    size_t in_cstride = static_cast<size_t>(h) * w;

    if(mode == miopenBNSpatial)
    {
        if(in_cstride > 512)
        {
            xgridsize = c * xlocalsize;
        }
        else
        {
            xgridsize = 1024 * static_cast<size_t>(c);
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

void BatchNormFwdTrainFusionOpDescriptor::calcBNParams(std::vector<size_t> in_lens,
                                                       int& variant,
                                                       size_t& in_cstride,
                                                       size_t& in_nstride,
                                                       size_t& in_nchw,
                                                       unsigned int& ldsgcn,
                                                       unsigned int& ldsnogcn)
{
    const auto [xlocalsize, _0, _1] = tien<3>(GetLocalWGSz());
    int n, c, h, w;
    variant              = 0;
    std::tie(n, c, h, w) = tien<4>(in_lens);
    in_cstride           = static_cast<size_t>(h) * w;
    in_nstride           = c * in_cstride;
    in_nchw              = n * in_nstride;

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

miopenStatus_t
BatchNormFwdTrainFusionOpDescriptor::GetNetworkConfig(std::ostringstream& network_config)
{
    int n, c, h, w;
    int variant               = 0;
    const bool saveBatchStats = true;
    bool savePopStats         = runningMeanVar;
    std::tie(n, c, h, w)      = tien<4>(input_desc.GetLengths());
    size_t in_cstride, in_nstride, in_nchw;
    size_t xlocalsize, ylocalsize, zlocalsize;
    std::tie(xlocalsize, ylocalsize, zlocalsize) = tien<3>(GetLocalWGSz());
    size_t zgridsize, ygridsize, xgridsize;
    std::tie(xgridsize, ygridsize, zgridsize) = tien<3>(GetGlobalWGSz());
    unsigned int ldsgcn, ldsnogcn;
    calcBNParams(
        input_desc.GetLengths(), variant, in_cstride, in_nstride, in_nchw, ldsgcn, ldsnogcn);

    if(input_desc.GetLengths().empty())
        MIOPEN_THROW("The input descriptor is not set");

    network_config << "variant" << std::to_string(variant) << "gx" << std::to_string(xgridsize)
                   << "gcn" << std::to_string(ldsgcn) << "gy" << std::to_string(ygridsize) << "lx"
                   << std::to_string(xlocalsize) << "ly" << std::to_string(ylocalsize) << "bn"
                   << std::to_string(mode) << "sbs"
                   << std::to_string(static_cast<int>(saveBatchStats)) << "sps"
                   << std::to_string(static_cast<int>(savePopStats)) << "n" << std::to_string(n)
                   << "hw" << std::to_string(in_cstride) << "chw" << std::to_string(in_nstride);

    return miopenStatusSuccess;
}

std::vector<size_t> BatchNormFwdTrainFusionOpDescriptor::GetLocalWGSz()
{
    size_t xlocalsize, ylocalsize, zlocalsize;
    int h, w;
    std::tie(std::ignore, std::ignore, h, w) = tien<4>(input_desc.GetLengths());
    size_t in_cstride                        = static_cast<size_t>(h) * w;

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

std::vector<size_t> BatchNormFwdTrainFusionOpDescriptor::GetGlobalWGSz()
{
    int c, h, w;
    std::tie(std::ignore, c, h, w) = tien<4>(input_desc.GetLengths());
    size_t xlocalsize, ylocalsize, zlocalsize;
    std::tie(xlocalsize, ylocalsize, zlocalsize) = tien<3>(GetLocalWGSz());

    size_t xgridsize = c * xlocalsize;
    size_t zgridsize = 1;
    size_t ygridsize = 1;

    size_t in_cstride = static_cast<size_t>(h) * w;

    if(mode != miopenBNSpatial)
    {
        auto segment = int(std::ceil(double(in_cstride) / double(ylocalsize)));
        xgridsize    = c;
        ygridsize    = segment * ylocalsize;
    }
    std::vector<size_t> vgd{xgridsize, ygridsize, zgridsize};
    return vgd;
}
/// END BN traing forward

} // namespace miopen
