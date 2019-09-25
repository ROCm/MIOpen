/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#include <miopen/config.h>
#include <miopen/convolution.hpp>
#include <miopen/env.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/miopen.h>
#include <miopen/mlo_internal.hpp>
#include <miopen/solver.hpp>
#include <miopen/tensor.hpp>
#include <miopen/algorithm.hpp>
#include <miopen/scgemm_utils.hpp>

#include <cassert>
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <ostream>

#include <boost/range/combine.hpp>
#include <boost/range/adaptors.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_SCGEMM)

// Workaround for issue 1430.
// Vega20 fails to access GPU memory larger than the return value of GetMaxMemoryAllocSize() of
// Vega10
#define MAX_MEM_ALLOC_SZ (std::min(handle.GetMaxMemoryAllocSize(), size_t(7287183769)))

namespace miopen {

ConvolutionDescriptor::ConvolutionDescriptor(std::size_t spatial_dim,
                                             miopenConvolutionMode_t c_mode,
                                             miopenPaddingMode_t p_mode,
                                             const std::vector<int>& p_pads,
                                             const std::vector<int>& p_strides,
                                             const std::vector<int>& p_dilations,
                                             const std::vector<int>& p_trans_output_pads,
                                             int p_group_count,
                                             float p_lowp_quant)
    : spatialDim(spatial_dim),
      mode(c_mode),
      paddingMode(p_mode),
      pads(p_pads),
      strides(p_strides),
      dilations(p_dilations),
      trans_output_pads(p_trans_output_pads),
      group_count(p_group_count),
      lowp_quant(p_lowp_quant)
{
    if(pads.size() != spatial_dim || strides.size() != spatial_dim ||
       dilations.size() != spatial_dim || trans_output_pads.size() != spatial_dim ||
       miopen::any_of(pads, [](auto v) { return v < 0; }) ||
       miopen::any_of(strides, [](auto v) { return v < 1; }) ||
       miopen::any_of(dilations, [](auto v) { return v < 1; }))
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "Invalid parameters, check usage. MIOPEN expects padding "
                     ">= 0, stride >= 1, dilation >= 1 and the same dilation "
                     "factor for horizontal and vertical direction");
    }
    if(!(mode == miopenConvolution || mode == miopenTranspose))
    {
        if(mode == miopenGroupConv || mode == miopenDepthwise)
        {
            mode = miopenConvolution;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm, "Convolution mode not supported");
        }
    }
    if(!(paddingMode == miopenPaddingSame || paddingMode == miopenPaddingValid ||
         paddingMode == miopenPaddingDefault))
    {
        MIOPEN_THROW(miopenStatusBadParm, "Padding mode not supported");
    }
}

ConvolutionDescriptor::ConvolutionDescriptor(const std::vector<int>& p_pads,
                                             const std::vector<int>& p_strides,
                                             const std::vector<int>& p_dilations,
                                             const std::vector<int>& p_trans_output_pads,
                                             int p_group_count,
                                             float p_lowp_quant)
    : ConvolutionDescriptor{p_pads.size(),
                            miopenConvolution,
                            miopenPaddingDefault,
                            p_pads,
                            p_strides,
                            p_dilations,
                            p_trans_output_pads,
                            p_group_count,
                            p_lowp_quant}
{
}

std::size_t ConvolutionDescriptor::GetSpatialDimension() const { return spatialDim; }

const std::vector<int>& ConvolutionDescriptor::GetConvPads() const { return pads; }

const std::vector<int>& ConvolutionDescriptor::GetConvStrides() const { return strides; }

const std::vector<int>& ConvolutionDescriptor::GetConvDilations() const { return dilations; }

const std::vector<int>& ConvolutionDescriptor::GetTransposeConvPads() const
{
    return trans_output_pads;
}

int ConvolutionDescriptor::GetGroupCount() const { return group_count; }

TensorDescriptor ConvolutionDescriptor::GetForwardOutputTensor(const TensorDescriptor& xDesc,
                                                               const TensorDescriptor& wDesc,
                                                               miopenDataType_t yType) const
{
    const std::size_t spatial_dim = GetSpatialDimension();

    assert(xDesc.GetLengths().size() == spatial_dim + 2);
    assert(wDesc.GetLengths().size() == spatial_dim + 2);

    if(xDesc.GetType() != wDesc.GetType())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Types do not match for the filter");
    }

    std::size_t in_n, in_c;
    std::tie(in_n, in_c) = miopen::tie_pick<0, 1>{}(xDesc.GetLengths());

    auto in_spatial = boost::adaptors::slice(xDesc.GetLengths(), 2, 2 + spatial_dim);

    std::size_t wei_k, wei_c;
    std::tie(wei_k, wei_c) = miopen::tie_pick<0, 1>{}(wDesc.GetLengths());

    auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);

    if(mode == miopenConvolution)
    {
        // for depthwise conv wei_c must be 1 while group_count must be wei_c
        if((group_count == 1 && in_c != wei_c) ||
           (group_count > 1 && (in_c % wei_c != 0 || wei_k % (in_c / wei_c) != 0)))
        {
            MIOPEN_THROW(miopenStatusBadParm, "Channels do not match for the filter");
        }
    }
    else if(mode == miopenTranspose)
    {
        if(in_c != wei_k || (group_count > 1 && (wei_k % group_count != 0)))
        {
            MIOPEN_THROW(miopenStatusBadParm, "Channels do not match for the filter");
        }

        if(miopen::any_of(boost::combine(GetTransposeConvPads(), GetConvStrides()), [](auto v) {
               auto trans_conv_pad = boost::get<0>(v);
               auto stride         = boost::get<1>(v);
               return trans_conv_pad >= stride;
           }))
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Output shape doesn't match due to invalid output padding");
        }
    }

    std::size_t out_c;
    std::vector<std::size_t> out_lens(spatial_dim + 2);

    auto out_spatial = boost::adaptors::slice(out_lens, 2, 2 + spatial_dim);

    if(paddingMode == miopenPaddingSame && mode == miopenConvolution &&
       miopen::all_of(GetConvDilations(), [](auto v) { return v == 1; }))
    {
        out_c = wei_k;

        for(int i = 0; i < spatial_dim; ++i)
        {
            out_spatial[i] = miopen::integer_division_ceil(in_spatial[i], GetConvStrides()[i]);
        }
    }
    else if(paddingMode == miopenPaddingValid && mode == miopenConvolution &&
            miopen::all_of(GetConvDilations(), [](auto v) { return v == 1; }))
    {
        out_c = wei_k;

        for(int i = 0; i < spatial_dim; ++i)
        {
            out_spatial[i] = miopen::integer_division_ceil(
                std::ptrdiff_t(in_spatial[i]) - wei_spatial[i] + 1, GetConvStrides()[i]);
        }
    }
    else if(paddingMode == miopenPaddingDefault || paddingMode == miopenPaddingSame ||
            paddingMode == miopenPaddingValid)
    {
        if(mode == miopenTranspose)
        {
            out_c = wei_c * group_count;

            for(int i = 0; i < spatial_dim; ++i)
            {
                out_spatial[i] = std::max<std::ptrdiff_t>(
                    1,
                    GetConvStrides()[i] * (std::ptrdiff_t(in_spatial[i]) - 1) + 1 +
                        GetConvDilations()[i] * (std::ptrdiff_t(wei_spatial[i]) - 1) -
                        2 * GetConvPads()[i] + GetTransposeConvPads()[i]);
            }
        }
        else
        {
            out_c = wei_k;

            for(int i = 0; i < spatial_dim; ++i)
            {
                out_spatial[i] = std::max<std::ptrdiff_t>(
                    1,
                    (ptrdiff_t(in_spatial[i]) -
                     (1 + GetConvDilations()[i] * (std::ptrdiff_t(wei_spatial[i]) - 1)) +
                     2 * GetConvPads()[i]) /
                            GetConvStrides()[i] +
                        1);
            }
        }
    }
    else
        MIOPEN_THROW(miopenStatusInvalidValue, "Invalid Padding Mode!");

    out_lens[0] = in_n;
    out_lens[1] = out_c;

    return TensorDescriptor((xDesc.GetType() == miopenInt8 || xDesc.GetType() == miopenInt8x4
                                 ? (yType == miopenInt32 ? yType : miopenFloat)
                                 : xDesc.GetType()),
                            out_lens);
}

std::size_t ConvolutionDescriptor::ForwardGetWorkSpaceSizeGEMM(const TensorDescriptor& wDesc,
                                                               const TensorDescriptor& yDesc) const
{
    const std::size_t spatial_dim = GetSpatialDimension();

    auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);
    auto out_spatial = boost::adaptors::slice(yDesc.GetLengths(), 2, 2 + spatial_dim);

    const std::size_t wei_c = wDesc.GetLengths()[1];

    const std::size_t workspace_size = wei_c * std::accumulate(wei_spatial.begin(),
                                                               wei_spatial.end(),
                                                               std::size_t(1),
                                                               std::multiplies<std::size_t>()) *
                                       std::accumulate(out_spatial.begin(),
                                                       out_spatial.end(),
                                                       std::size_t(1),
                                                       std::multiplies<std::size_t>()) *
                                       GetTypeSize(wDesc.GetType());

    // No workspace is needed for 1x1 convolutions
    if(miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
       miopen::all_of(GetConvPads(), [](auto v) { return v == 0; }) &&
       miopen::all_of(GetConvStrides(), [](auto v) { return v == 1; }))
    {
        if(wDesc.GetType() == miopenInt8)
            return workspace_size;
        else
            return 0;
    }

    return (wDesc.GetType() == miopenInt8 ? 2 * workspace_size : workspace_size);
}

std::size_t
ConvolutionDescriptor::ForwardGetWorkSpaceSizeGEMMTranspose(const TensorDescriptor& xDesc,
                                                            const TensorDescriptor& yDesc) const
{
    std::size_t in_n, in_c;
    std::tie(in_n, in_c) = miopen::tie_pick<0, 1>{}(xDesc.GetLengths());

    auto out_spatial = boost::adaptors::slice(yDesc.GetLengths(), 2, 2 + GetSpatialDimension());

    std::size_t x_t_size = in_n * in_c * std::accumulate(out_spatial.begin(),
                                                         out_spatial.end(),
                                                         std::size_t(1),
                                                         std::multiplies<std::size_t>()) *
                           GetTypeSize(xDesc.GetType());

    // Int8 also does "transpose_packed_MN2NM" which need additional workspace
    if(xDesc.GetType() == miopenInt8)
        x_t_size *= 2;

    const std::size_t y_t_size = yDesc.GetElementSize() * GetTypeSize(yDesc.GetType());

    return x_t_size + y_t_size;
}

/// There is assumption that if Winograd is applicable and granularity loss is low, then there is no
/// advantage in trying other algorithms as those either slower or use more workspace. This allows
/// for some related host-side optimizations.
///
/// These optimizations are kind of cutting corners, but advantages are quite high.
bool ConvolutionDescriptor::IsWinograd3x3SupportedAndFast(miopen::ConvolutionContext& ctx) const
{
    // Filter out configs where 3x3 Winograd does not have high WTI.
    if(!(ctx.n_outputs >= 16 && ctx.n_outputs % 2 == 0))
        return false;

    return solver::ConvBinWinograd3x3U{}.IsApplicable(ctx);
}

/// \todo Merge with ForwardGetWorkSpaceSizeGEMM
/// Use it instead of ForwardGetWorkSpaceSizeGEMM in ForwardGetWorkSpaceSize
std::size_t
ConvolutionDescriptor::ForwardGetValidWorkSpaceSizeGemm(Handle& handle,
                                                        const TensorDescriptor& wDesc,
                                                        const TensorDescriptor& xDesc,
                                                        const TensorDescriptor& yDesc) const
{

#if MIOPEN_USE_GEMM
    const std::size_t spatial_dim = GetSpatialDimension();
    auto wei_spatial              = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);
    auto in_spatial               = boost::adaptors::slice(xDesc.GetLengths(), 2, 2 + spatial_dim);

    // Use transpose path if input ht and width <= 14 for 1x1_stride=1 convolutions OR for
    // 1x1_stride=2
    if(GetSpatialDimension() == 2 &&
       (miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
        miopen::all_of(GetConvPads(), [](auto v) { return v == 0; })) &&
       ((miopen::all_of(in_spatial, [](auto v) { return v <= 14; }) &&
         miopen::all_of(GetConvStrides(), [](auto v) { return v == 1; })) ||
        miopen::all_of(GetConvStrides(), [](auto v) { return v == 2; })))
    {
        size_t gemm_trans = ForwardGetWorkSpaceSizeGEMMTranspose(xDesc, yDesc);
        /// \todo WORKAROUND for issue 1430
        if(gemm_trans > MAX_MEM_ALLOC_SZ /* handle.GetMaxMemoryAllocSize() */)
            gemm_trans = 0;
        return gemm_trans;
    }

    size_t workspace_size_gemm = ForwardGetWorkSpaceSizeGEMM(wDesc, yDesc) * group_count;
    /// \todo WORKAROUND for issue 1430
    if(workspace_size_gemm > MAX_MEM_ALLOC_SZ /* handle.GetMaxMemoryAllocSize() */)
        workspace_size_gemm = 0;

    return workspace_size_gemm;
#else
    (void)handle;
    (void)wDesc;
    (void)xDesc;
    (void)yDesc;
    return 0;
#endif
}

std::size_t
ConvolutionDescriptor::WrwGetValidWorkSpaceSizeGemm(const TensorDescriptor& dyDesc,
                                                    const TensorDescriptor& /*xDesc*/,
                                                    const TensorDescriptor& dwDesc) const
{
#if MIOPEN_USE_GEMM
    const std::size_t spatial_dim = GetSpatialDimension();
    const auto wei_spatial        = boost::adaptors::slice(dwDesc.GetLengths(), 2, 2 + spatial_dim);

    // if not 1x1
    if((miopen::any_of(wei_spatial, [](auto v) { return v != 1; }) ||
        miopen::any_of(GetConvPads(), [](auto v) { return v != 0; }) ||
        miopen::any_of(GetConvStrides(), [](auto v) { return v != 1; })))
        return BackwardWeightsGetWorkSpaceSizeGEMM(dyDesc, dwDesc) * group_count;

    if(miopen::any_of(wei_spatial, [](auto v) { return v == 1; }) &&
       miopen::any_of(GetConvPads(), [](auto v) { return v == 0; }) &&
       miopen::any_of(GetConvStrides(), [](auto v) { return v == 1; }))
        return 0;

    MIOPEN_THROW(miopenStatusNotImplemented);
#else
    std::ignore = dwDesc;
    std::ignore = dyDesc;
    return 0;
#endif
}

std::size_t ConvolutionDescriptor::ForwardGetWorkSpaceSize(Handle& handle,
                                                           const TensorDescriptor& wDesc,
                                                           const TensorDescriptor& xDesc,
                                                           const TensorDescriptor& yDesc) const
{
    MIOPEN_LOG_I("");

    auto ctx = ConvolutionContext{xDesc, wDesc, yDesc, *this, 1}; // Forward
    ctx.SetStream(&handle);
    ctx.DetectRocm();

    if(IsWinograd3x3SupportedAndFast(ctx))
        return 0;

    ctx.SetupFloats();
    ctx.do_search             = false;
    ctx.disable_perfdb_access = true;

    const size_t direct_workspace = ForwardBackwardDataGetWorkSpaceSizeDirect(ctx);

    const size_t implicit_gemm_workspace = ForwardGetWorkSpaceSizeImplicitGemm(ctx);

    const size_t workspace_size_scgemm = ForwardBackwardDataGetWorkSpaceSizeSCGemm(handle, ctx);

#if MIOPEN_USE_GEMM
    const std::size_t spatial_dim = GetSpatialDimension();
    const auto wei_spatial        = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);
    const auto in_spatial         = boost::adaptors::slice(xDesc.GetLengths(), 2, 2 + spatial_dim);

    size_t workspace_size_gemm = ForwardGetWorkSpaceSizeGEMM(wDesc, yDesc) * group_count;
    /// \todo WORKAROUND for issue 1430
    if(workspace_size_gemm > MAX_MEM_ALLOC_SZ /* handle.GetMaxMemoryAllocSize() */)
        workspace_size_gemm = 0;

    // Use transpose path if input ht and width <= 14 for 1x1_stride=1 convolutions OR for
    // 1x1_stride=2
    if(GetSpatialDimension() == 2 &&
       (miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
        miopen::all_of(GetConvPads(), [](auto v) { return v == 0; })) &&
       ((miopen::all_of(in_spatial, [](auto v) { return v <= 14; }) &&
         miopen::all_of(GetConvStrides(), [](auto v) { return v == 1; })) ||
        miopen::all_of(GetConvStrides(), [](auto v) { return v == 2; })))
    {
        size_t gemm_trans = ForwardGetWorkSpaceSizeGEMMTranspose(xDesc, yDesc);
        /// \todo WORKAROUND for issue 1430
        if(gemm_trans > MAX_MEM_ALLOC_SZ /* handle.GetMaxMemoryAllocSize() */)
            gemm_trans = 0;
        return std::max({gemm_trans, direct_workspace, workspace_size_scgemm});
    }

    if(miopen::any_of(GetConvDilations(), [](auto v) { return v > 1; }))
    {
        return std::max({workspace_size_gemm, direct_workspace, workspace_size_scgemm});
    }
#else
    size_t workspace_size_gemm = 0;
#endif

    const bool is_datatype_int8 =
        (wDesc.GetType() == miopenInt8 || wDesc.GetType() == miopenInt8x4);

    const size_t workspace_size_fft =
        (GetSpatialDimension() == 2 &&
         miopen::all_of(GetConvDilations(), [](auto v) { return v == 1; }) && !is_datatype_int8)
            ? ForwardGetWorkSpaceSizeFFT(wDesc, xDesc, yDesc)
            : 0;

    return std::max({workspace_size_fft,
                     workspace_size_gemm,
                     direct_workspace,
                     implicit_gemm_workspace,
                     workspace_size_scgemm});
}

std::size_t
ConvolutionDescriptor::BackwardDataGetWorkSpaceSize(Handle& handle,
                                                    const TensorDescriptor& wDesc,
                                                    const TensorDescriptor& dyDesc,
                                                    const TensorDescriptor& dxDesc) const
{
    MIOPEN_LOG_I("");

    auto ctx = ConvolutionContext{dxDesc, wDesc, dyDesc, *this, 0}; // Backward
    ctx.SetStream(&handle);
    ctx.DetectRocm();

    if(IsWinograd3x3SupportedAndFast(ctx))
        return 0;

    ctx.SetupFloats();
    ctx.do_search             = false;
    ctx.disable_perfdb_access = true;

    const size_t direct_workspace = ForwardBackwardDataGetWorkSpaceSizeDirect(ctx);

#if MIOPEN_USE_GEMM
    size_t workspace_size_gemm = BackwardDataGetWorkSpaceSizeGEMM(wDesc, dyDesc) * group_count;
    /// \todo WORKAROUND for issue 1430
    if(workspace_size_gemm > MAX_MEM_ALLOC_SZ /*  handle.GetMaxMemoryAllocSize() */)
        workspace_size_gemm = 0;

    const auto wei_spatial =
        boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + GetSpatialDimension());

    if(miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
       miopen::all_of(GetConvPads(), [](auto v) { return v == 0; }) &&
       miopen::all_of(GetConvStrides(), [](auto v) { return v == 2; }))
    {
        size_t gemm_trans = BackwardDataGetWorkSpaceSizeGEMMTranspose(dyDesc, dxDesc);
        /// \todo WORKAROUND for issue 1430
        if(gemm_trans > MAX_MEM_ALLOC_SZ /*  handle.GetMaxMemoryAllocSize() */)
            gemm_trans = 0;
        return std::max(gemm_trans, direct_workspace);
    }
    if(miopen::any_of(GetConvDilations(), [](auto v) { return v > 1; }))
        return std::max(workspace_size_gemm, direct_workspace);
#else
    size_t workspace_size_gemm = 0;
#endif

    const size_t workspace_size_fft =
        (GetSpatialDimension() == 2 &&
         miopen::all_of(GetConvDilations(), [](auto v) { return v == 1; }) &&
         wDesc.GetType() != miopenInt8)
            ? BackwardGetWorkSpaceSizeFFT(wDesc, dyDesc, dxDesc)
            : 0;

    return std::max({workspace_size_fft, workspace_size_gemm, direct_workspace});
}

std::size_t
ConvolutionDescriptor::BackwardDataGetWorkSpaceSizeGEMM(const TensorDescriptor& wDesc,
                                                        const TensorDescriptor& dyDesc) const
{
    const std::size_t spatial_dim = GetSpatialDimension();

    auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);
    auto out_spatial = boost::adaptors::slice(dyDesc.GetLengths(), 2, 2 + spatial_dim);

    const std::size_t wei_c = wDesc.GetLengths()[1];

    std::size_t gemm_size = wei_c * std::accumulate(wei_spatial.begin(),
                                                    wei_spatial.end(),
                                                    std::size_t(1),
                                                    std::multiplies<std::size_t>()) *
                            std::accumulate(out_spatial.begin(),
                                            out_spatial.end(),
                                            std::size_t(1),
                                            std::multiplies<std::size_t>()) *
                            GetTypeSize(dyDesc.GetType());

    // No workspace is needed for 1x1_stride=1 convolutions
    if(miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
       miopen::all_of(GetConvStrides(), [](auto v) { return v == 1; }) &&
       miopen::all_of(GetConvPads(), [](auto v) { return v == 0; }))
    {
        return 0;
    }

    return gemm_size;
}

std::size_t ConvolutionDescriptor::BackwardDataGetWorkSpaceSizeGEMMTranspose(
    const TensorDescriptor& dyDesc, const TensorDescriptor& dxDesc) const
{
    std::size_t in_n, in_c;
    std::tie(in_n, in_c) = miopen::tie_pick<0, 1>{}(dxDesc.GetLengths());

    auto out_spatial = boost::adaptors::slice(dyDesc.GetLengths(), 2, 2 + GetSpatialDimension());

    const std::size_t dx_t_size = in_n * in_c * std::accumulate(out_spatial.begin(),
                                                                out_spatial.end(),
                                                                std::size_t(1),
                                                                std::multiplies<std::size_t>()) *
                                  GetTypeSize(dxDesc.GetType());

    const std::size_t dy_t_size = dyDesc.GetElementSize() * GetTypeSize(dyDesc.GetType());

    return dx_t_size + dy_t_size;
}

std::size_t
ConvolutionDescriptor::BackwardWeightsGetWorkSpaceSizeGEMM(const TensorDescriptor& dyDesc,
                                                           const TensorDescriptor& dwDesc) const
{
    const std::size_t spatial_dim = GetSpatialDimension();

    auto out_spatial = boost::adaptors::slice(dyDesc.GetLengths(), 2, 2 + spatial_dim);
    auto wei_spatial = boost::adaptors::slice(dwDesc.GetLengths(), 2, 2 + spatial_dim);

    const std::size_t wei_c = dwDesc.GetLengths()[1];

    const std::size_t gemm_size =
        GetTypeSize(dyDesc.GetType()) * wei_c * std::accumulate(out_spatial.begin(),
                                                                out_spatial.end(),
                                                                std::size_t(1),
                                                                std::multiplies<std::size_t>()) *
        std::accumulate(
            wei_spatial.begin(), wei_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

    // No workspace is needed for 1x1_stride=1 convolutions
    if(miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
       miopen::all_of(GetConvStrides(), [](auto v) { return v == 1; }) &&
       miopen::all_of(GetConvPads(), [](auto v) { return v == 0; }))
    {
        return 0;
    }

    return gemm_size;
}

std::size_t ConvolutionDescriptor::ForwardGetWorkSpaceSizeImplicitGemm(
    const miopen::ConvolutionContext& ctx) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM{}))
    {
        return 0;
    }

    try
    {
        const auto ss  = FindAllImplicitGemmSolutions(ctx);
        std::size_t sz = 0;
        for(const auto& solution : ss)
        {
            if(sz < solution.workspce_sz)
            {
                MIOPEN_LOG_I2(sz << " < " << solution.workspce_sz);
                sz = solution.workspce_sz;
            }
        }
        return sz;
    }
    catch(const miopen::Exception&)
    {
        MIOPEN_LOG_E("failed in ForwardGetWorkSpaceSizeImplicitGemm");
        return 0;
    }
}

std::size_t ConvolutionDescriptor::ForwardBackwardDataGetWorkSpaceSizeDirect(
    const miopen::ConvolutionContext& ctx) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT{}))
    {
        return 0;
    }

    try
    {
        const auto sz_v = AllDirectForwardBackwardDataWorkspaceSize(ctx);
        std::size_t sz  = 0;
        for(const auto& pr : sz_v)
        {
            if(sz < pr.second)
            {
                MIOPEN_LOG_I2(sz << " < " << pr.second); // solution.workspce_sz);
                sz = pr.second;                          // solution.workspce_sz;
            }
        }
        return sz;
    }
    catch(const miopen::Exception&)
    {
        return 0;
    }
}

std::size_t ConvolutionDescriptor::ForwardBackwardDataGetWorkSpaceSizeSCGemm(
    Handle& handle, const miopen::ConvolutionContext& ctx) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_SCGEMM{}))
    {
        return 0;
    }

    std::size_t sz = 0;

#if MIOPEN_USE_SCGEMM
    sz = GetMaximumSCGemmConvFwdWorkSpaceSize(ctx);
    if(sz > MAX_MEM_ALLOC_SZ)
        sz = 0;
#else
    (void)handle;
    (void)ctx;
#endif

    return sz;
}

std::size_t
ConvolutionDescriptor::BackwardWeightsGetWorkSpaceSizeDirect(Handle& handle,
                                                             const TensorDescriptor& dyDesc,
                                                             const TensorDescriptor& xDesc,
                                                             const TensorDescriptor& dwDesc) const
{
    auto ctx = ConvolutionContext(xDesc, dwDesc, dyDesc, *this, 0);
    ctx.direction.SetBackwardWrW();
    ctx.do_search = false;
    ctx.SetStream(&handle);
    ctx.disable_perfdb_access = true;
    ctx.SetupFloats();
    ctx.DetectRocm();

    try
    {
        const auto sz_v = AllDirectBwdWrW2DWorkspaceSize(ctx);
        std::size_t sz  = 0;
        for(const auto& pr : sz_v)
        {
            if(sz < pr.second)
            {
                MIOPEN_LOG_I2(sz << " < " << pr.second);
                sz = pr.second;
            }
        }
        return sz;
    }
    catch(const miopen::Exception&)
    {
        return 0;
    }
}

std::size_t
ConvolutionDescriptor::BackwardWeightsGetWorkSpaceSizeWinograd(Handle& handle,
                                                               const TensorDescriptor& dyDesc,
                                                               const TensorDescriptor& xDesc,
                                                               const TensorDescriptor& dwDesc) const
{
    auto ctx = ConvolutionContext(xDesc, dwDesc, dyDesc, *this, 0);
    ctx.direction.SetBackwardWrW();
    ctx.do_search = false;
    ctx.SetStream(&handle);
    ctx.disable_perfdb_access = true;
    ctx.DetectRocm();

    try
    {
        const auto ss  = FindWinogradWrWAllSolutions(ctx);
        std::size_t sz = 0;
        for(const auto& solution : ss)
        {
            if(sz < solution.workspce_sz)
            {
                MIOPEN_LOG_I2(sz << " < " << solution.workspce_sz);
                sz = solution.workspce_sz;
            }
        }
        return sz;
    }
    catch(const miopen::Exception&)
    {
        return 0;
    }
}
std::size_t
ConvolutionDescriptor::BackwardWeightsGetWorkSpaceSize(Handle& handle,
                                                       const TensorDescriptor& dyDesc,
                                                       const TensorDescriptor& xDesc,
                                                       const TensorDescriptor& dwDesc) const
{
    MIOPEN_LOG_I("");

    std::size_t workspace_size = 0;
    {
        std::size_t workspace_size_gemm =
#if MIOPEN_USE_GEMM
            BackwardWeightsGetWorkSpaceSizeGEMM(dyDesc, dwDesc) * group_count;
        /// \todo WORKAROUND for issue 1430
        if(workspace_size_gemm > MAX_MEM_ALLOC_SZ /*  handle.GetMaxMemoryAllocSize() */)
            workspace_size_gemm =
#endif
                0;

        size_t direct_workspace =
            BackwardWeightsGetWorkSpaceSizeDirect(handle, dyDesc, xDesc, dwDesc);

        size_t winograd_workspace =
            BackwardWeightsGetWorkSpaceSizeWinograd(handle, dyDesc, xDesc, dwDesc);

        workspace_size =
            std::max(winograd_workspace, std::max(direct_workspace, workspace_size_gemm));
    }

    return workspace_size;
}

std::ostream& operator<<(std::ostream& stream, const ConvolutionDescriptor& c)
{
    stream << "conv" << c.spatialDim << "d, ";
    MIOPEN_LOG_ENUM(stream, c.mode, miopenConvolution, miopenTranspose) << ", ";
    MIOPEN_LOG_ENUM(
        stream, c.paddingMode, miopenPaddingDefault, miopenPaddingSame, miopenPaddingValid)
        << ", ";

    LogRange(stream << "{", c.GetConvPads(), ", ") << "}, ";
    LogRange(stream << "{", c.GetConvStrides(), ", ") << "}, ";
    LogRange(stream << "{", c.GetConvDilations(), ", ") << "}, ";

    if(c.group_count > 1)
    {
        stream << c.group_count << ", ";
    }

    if(c.mode == miopenTranspose)
    {
        LogRange(stream << "{", c.GetTransposeConvPads(), ", ") << "}, ";
    }

    return stream;
}
} // namespace miopen
