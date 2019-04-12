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

#include <cassert>
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <ostream>

#include <boost/range/combine.hpp>
#include <boost/range/adaptors.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT)

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
                                                               const TensorDescriptor& wDesc) const
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
                                 ? miopenFloat
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

// FIXME: This seems to duplicate
// mlo_construct_direct2D::mloIsCorrectBinaryWinograd3x3U()
// functionality thus violating the One Definition Rule.
bool ConvolutionDescriptor::IsWinograd3x3Supported(Handle& handle,
                                                   bool direction,
                                                   const TensorDescriptor& wDesc,
                                                   const TensorDescriptor& xDesc) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_ROCM_PRECOMPILED_BINARIES{}))
    {
        // Support for MIOPEN_DEBUG_AMD_ASM_KERNELS_PERF_FILTERING is not copypasted here.
        // Right now this does not matter as there is none perf filtering for Winograd
        return false;
    }

    if(GetSpatialDimension() != 2)
    {
        return false;
    }

    const auto device_name       = handle.GetDeviceName();
    const auto max_compute_units = handle.GetMaxComputeUnits();

    int _batch_sz, _n_inputs, _in_height, _in_width;
    int _n_outputs, _kernel_size0, _kernel_size1;
    int _n_outputs_w, _n_inputs_w;

    // Assumed rocm_meta_version::AMDHSA_1_0 or newer.
    if(!(device_name == "gfx803" || device_name == "gfx900" || device_name == "gfx906"))
    {
        return false;
    }
    const auto device_is_gfx8 = (device_name.find("gfx8") != std::string::npos);

    std::tie(_batch_sz, _n_inputs, _in_height, _in_width)             = tien<4>(xDesc.GetLengths());
    std::tie(_n_outputs_w, _n_inputs_w, _kernel_size0, _kernel_size1) = tien<4>(wDesc.GetLengths());

    _n_outputs = direction ? _n_outputs_w : _n_inputs_w;
    return GetConvPads()[0] == 1 && GetConvPads()[1] == 1 && _kernel_size0 == 3 &&
           _kernel_size1 == 3 && GetConvStrides()[0] == 1 && GetConvStrides()[1] == 1 &&
           _batch_sz < std::pow(2, 16) && _n_inputs < std::pow(2, 16) &&
           _n_outputs < std::pow(2, 16) && _in_height < std::pow(2, 16) &&
           _in_width < std::pow(2, 16) && max_compute_units < std::pow(2, 16) &&
           (_n_inputs * _in_height * _in_width) <= std::pow(2, 28) &&
           (_n_outputs * _in_height * _in_width) <= std::pow(2, 28) &&
           (_n_inputs * _kernel_size0 * _kernel_size1) <= std::pow(2, 28) &&
           (_n_outputs * _kernel_size0 * _kernel_size1) <= std::pow(2, 28) && _n_inputs % 2 == 0 &&
           _n_inputs >= (device_is_gfx8 ? 16 : 18) && (GetTypeSize(wDesc.GetType()) == 4) &&
           (GetTypeSize(xDesc.GetType()) == 4) && group_count == 1 && GetConvDilations()[0] == 1 &&
           GetConvDilations()[1] == 1;
}

std::size_t ConvolutionDescriptor::ForwardGetWorkSpaceSize(Handle& handle,
                                                           const TensorDescriptor& wDesc,
                                                           const TensorDescriptor& xDesc,
                                                           const TensorDescriptor& yDesc) const
{
    MIOPEN_LOG_I2("");
    {
        const std::size_t spatial_dim = GetSpatialDimension();

        auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);
        auto in_spatial  = boost::adaptors::slice(xDesc.GetLengths(), 2, 2 + spatial_dim);

        bool is_datatype_int8 = (wDesc.GetType() == miopenInt8 || wDesc.GetType() == miopenInt8x4);

        const size_t direct_workspace =
            (GetSpatialDimension() == 2 &&
             miopen::all_of(GetConvDilations(), [](auto v) { return v == 1; }) && !is_datatype_int8)
                ? ForwardBackwardDataGetWorkSpaceSizeDirect(handle, xDesc, yDesc, wDesc, 1)
                : 0;

        size_t workspace_size_gemm =
#if MIOPEN_USE_GEMM
            ForwardGetWorkSpaceSizeGEMM(wDesc, yDesc) * group_count;
        /// \todo WORKAROUND for issue 1430
        if(workspace_size_gemm > MAX_MEM_ALLOC_SZ /* handle.GetMaxMemoryAllocSize() */)
            workspace_size_gemm =
#endif
                0;

#if MIOPEN_USE_GEMM
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
            return std::max(gemm_trans, direct_workspace);
        }

        if(miopen::any_of(GetConvDilations(), [](auto v) { return v > 1; }))
        {
            return std::max(workspace_size_gemm, direct_workspace);
        }
#endif

        // Check if Winograd is available
        // If Winograd is present, there is no advantage in letting
        // the user run another algorithm as those both slower and
        // use more workspace.
        if(IsWinograd3x3Supported(handle, true, wDesc, xDesc) && !is_datatype_int8)
        {
            return 0;
        }
        else
        {
            size_t workspace_size_fft =
                (GetSpatialDimension() == 2 &&
                 miopen::all_of(GetConvDilations(), [](auto v) { return v == 1; }) &&
                 !is_datatype_int8)
                    ? ForwardGetWorkSpaceSizeFFT(wDesc, xDesc, yDesc)
                    : 0;

            return std::max(std::max(workspace_size_fft, workspace_size_gemm), direct_workspace);
        }
    }
}

std::size_t
ConvolutionDescriptor::BackwardDataGetWorkSpaceSize(Handle& handle,
                                                    const TensorDescriptor& wDesc,
                                                    const TensorDescriptor& dyDesc,
                                                    const TensorDescriptor& dxDesc) const
{
    MIOPEN_LOG_I2("");
    {
        auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + GetSpatialDimension());

        const size_t direct_workspace =
            (GetSpatialDimension() == 2 &&
             miopen::all_of(GetConvDilations(), [](auto v) { return v == 1; }) &&
             wDesc.GetType() != miopenInt8)
                ? ForwardBackwardDataGetWorkSpaceSizeDirect(handle, dxDesc, dyDesc, wDesc, 0)
                : 0;

        size_t workspace_size_gemm =
#if MIOPEN_USE_GEMM
            BackwardDataGetWorkSpaceSizeGEMM(wDesc, dyDesc) * group_count;
        /// \todo WORKAROUND for issue 1430
        if(workspace_size_gemm > MAX_MEM_ALLOC_SZ /*  handle.GetMaxMemoryAllocSize() */)
            workspace_size_gemm =
#endif
                0;

#if MIOPEN_USE_GEMM
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
#endif

        // Check if Winograd is available
        // If Winograd is present, there is no advantage in letting
        // the user run another algorithm as those both slower and
        // use more workspace.
        if(IsWinograd3x3Supported(handle, false, wDesc, dyDesc) && wDesc.GetType() != miopenInt8)
        {
            return 0;
        }
        else
        {
            const size_t workspace_size_fft =
                (GetSpatialDimension() == 2 &&
                 miopen::all_of(GetConvDilations(), [](auto v) { return v == 1; }) &&
                 wDesc.GetType() != miopenInt8)
                    ? BackwardGetWorkSpaceSizeFFT(wDesc, dyDesc, dxDesc)
                    : 0;

            return std::max(std::max(workspace_size_fft, workspace_size_gemm), direct_workspace);
        }
    }
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
    std::tie(in_n, in_c, std::ignore, std::ignore) = tien<4>(dxDesc.GetLengths());

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

std::size_t ConvolutionDescriptor::ForwardBackwardDataGetWorkSpaceSizeDirect(
    Handle& handle,
    const TensorDescriptor& xDesc,
    const TensorDescriptor& yDesc,
    const TensorDescriptor& wDesc,
    int direction) const // 1: Forward, 0: BackwardData
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT{}))
    {
        return 0;
    }

    /// \todo See issue #1587
    /// This must be handled in uniform way everywhere.
    /// Number of dimensions to be added to problem description.
    /// Non-2D problems to be filtered out in IsApplicable() methods.
    /// 3D problems to be properly serialized.
    if(GetSpatialDimension() != 2)
    {
        return 0;
    }

    mlo_construct_direct2D construct_params(xDesc, wDesc, yDesc, *this, direction);
    construct_params.setDoSearch(false);
    construct_params.setStream(&handle);
    construct_params.setWorkaroundDisableSearchEnforce(true);

    try
    {
        const auto ss  = FindAllSolutions(construct_params);
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
ConvolutionDescriptor::BackwardWeightsGetWorkSpaceSizeDirect(Handle& handle,
                                                             const TensorDescriptor& dyDesc,
                                                             const TensorDescriptor& xDesc,
                                                             const TensorDescriptor& dwDesc) const
{
    if(GetSpatialDimension() != 2)
    {
        return 0;
    }

    mlo_construct_BwdWrW2D construct_params(
        xDesc, dwDesc, dyDesc, *this, 0); // backward with regards to weights
    construct_params.setDoSearch(false);
    construct_params.setStream(&handle);
    construct_params.setWorkaroundDisableSearchEnforce(true);
    try
    {
        const auto ss  = FindAllSolutions(construct_params);
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

std::size_t ConvolutionDescriptor::ConvolutionBackwardWeightsGetWorkSpaceSize(
    Handle& handle,
    const TensorDescriptor& dyDesc,
    const TensorDescriptor& xDesc,
    const TensorDescriptor& dwDesc) const
{
    MIOPEN_LOG_I2("");

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
            (GetSpatialDimension() == 2 &&
             miopen::all_of(GetConvDilations(), [](auto v) { return v == 1; }) &&
             dwDesc.GetType() != miopenInt8)
                ? BackwardWeightsGetWorkSpaceSizeDirect(handle, dyDesc, xDesc, dwDesc)
                : 0;

        workspace_size = std::max(direct_workspace, workspace_size_gemm);
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
