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

#include <cassert>
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <ostream>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT)

namespace miopen {

// Workaround for issue 1430.
// Vega20 fails to access GPU memory larger than the return value of GetMaxMemoryAllocSize() of
// Vega10
#define MAX_MEM_ALLOC_SZ (std::min(handle.GetMaxMemoryAllocSize(), size_t(7287183769)))

ConvolutionDescriptor::ConvolutionDescriptor(miopenConvolutionMode_t c_mode,
                                             miopenPaddingMode_t p_mode,
                                             const std::vector<int>& p_pads,
                                             const std::vector<int>& p_strides,
                                             const std::vector<int>& p_dilations,
                                             const std::vector<int>& p_trans_output_pads,
                                             int p_group_count,
                                             float p_lowp_quant)
    : mode(c_mode),
      paddingMode(p_mode),
      pads(p_pads),
      strides(p_strides),
      dilations(p_dilations),
      trans_output_pads(p_trans_output_pads),
      group_count(p_group_count),
      lowp_quant(p_lowp_quant)
{
    if(pads.size() != strides.size() || pads.size() != dilations.size() ||
       pads.size() != GetTransposeConvPads().size() ||
       std::any_of(pads.begin(), pads.end(), [](int pad) { return pad < 0; }) ||
       std::any_of(strides.begin(), strides.end(), [](int stride) { return stride <= 0; }) ||
       std::any_of(dilations.begin(), dilations.end(), [](int dilation) { return dilation <= 0; }))
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
    : ConvolutionDescriptor{miopenConvolution,
                            miopenPaddingDefault,
                            p_pads,
                            p_strides,
                            p_dilations,
                            p_trans_output_pads,
                            p_group_count,
                            p_lowp_quant}
{
}

const std::vector<int>& ConvolutionDescriptor::GetConvPads() const { return pads; }

const std::vector<int>& ConvolutionDescriptor::GetConvStrides() const { return strides; }

const std::vector<int>& ConvolutionDescriptor::GetConvDilations() const { return dilations; }

const std::vector<int>& ConvolutionDescriptor::GetTransposeConvPads() const
{
    return trans_output_pads;
}

std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>
ConvolutionDescriptor::GetForwardOutputDim(const TensorDescriptor& inputTensorDesc,
                                           const TensorDescriptor& filterDesc) const
{
    assert(inputTensorDesc.GetLengths().size() == 4);
    assert(filterDesc.GetLengths().size() == 4);

    if(inputTensorDesc.GetType() != filterDesc.GetType())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Types do not match for the filter");
    }

    // We use signed integers here to avoid possible underflows
    std::ptrdiff_t input_n;
    std::ptrdiff_t input_c;
    std::ptrdiff_t input_h;
    std::ptrdiff_t input_w;

    std::tie(input_n, input_c, input_h, input_w) = miopen::tien<4>(inputTensorDesc.GetLengths());

    std::ptrdiff_t filter_k;
    std::ptrdiff_t filter_c;
    std::ptrdiff_t filter_h;
    std::ptrdiff_t filter_w;

    std::tie(filter_k, filter_c, filter_h, filter_w) = miopen::tien<4>(filterDesc.GetLengths());

    if(mode == miopenConvolution)
    {
        if((group_count == 1 && input_c != filter_c) ||
           (group_count >= 2 &&
            (input_c % filter_c != 0 ||
             filter_k % (input_c / filter_c) !=
                 0))) // for depthwise conv filter_c must be 1 while group_count must be filter_c
        {
            MIOPEN_THROW(miopenStatusBadParm, "Channels do not match for the filter");
        }
    }
    else if(mode == miopenTranspose)
    {
        if(input_c != filter_k || (group_count >= 2 && (filter_k % group_count != 0)))
        {
            MIOPEN_THROW(miopenStatusBadParm, "Channels do not match for the filter");
        }
        if(GetTransposeConvPads()[0] >= GetConvStrides()[0] ||
           GetTransposeConvPads()[1] >= GetConvStrides()[1])
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Output shape doesn't match due to invalid output padding");
        }
    }

    std::ptrdiff_t output_c;
    std::ptrdiff_t output_h;
    std::ptrdiff_t output_w;

    if(paddingMode == miopenPaddingSame && GetConvDilations()[0] == 1 &&
       GetConvDilations()[1] == 1 && mode == miopenConvolution)
    {
        output_c = filter_k;
        output_h = std::ceil(static_cast<double>(input_h) / GetConvStrides()[0]);
        output_w = std::ceil(static_cast<double>(input_w) / GetConvStrides()[1]);
    }
    else if(paddingMode == miopenPaddingValid && GetConvDilations()[0] == 1 &&
            GetConvDilations()[1] == 1 && mode == miopenConvolution)
    {
        output_c = filter_k;
        output_h = std::ceil(static_cast<double>(input_h - filter_h + 1) / GetConvStrides()[0]);
        output_w = std::ceil(static_cast<double>(input_w - filter_w + 1) / GetConvStrides()[1]);
    }
    else if(paddingMode == miopenPaddingDefault || paddingMode == miopenPaddingSame ||
            paddingMode == miopenPaddingValid)
    {
        if(mode == miopenTranspose)
        {
            output_c = filter_c * group_count;
            output_h = std::max<std::ptrdiff_t>(
                1,
                std::ptrdiff_t(GetConvStrides()[0] * (input_h - 1) + 1 +
                               GetConvDilations()[0] * (filter_h - 1.0) - 2 * GetConvPads()[0] +
                               GetTransposeConvPads()[0]));
            output_w = std::max<std::ptrdiff_t>(
                1,
                std::ptrdiff_t(GetConvStrides()[1] * (input_w - 1) + 1 +
                               GetConvDilations()[1] * (filter_w - 1.0) - 2 * GetConvPads()[1] +
                               GetTransposeConvPads()[1]));
        }
        else
        {
            output_c = filter_k;
            output_h = std::max<std::ptrdiff_t>(
                1,
                (input_h - (1 + GetConvDilations()[0] * (filter_h - 1)) + 2 * GetConvPads()[0]) /
                        GetConvStrides()[0] +
                    1);
            output_w = std::max<std::ptrdiff_t>(
                1,
                (input_w - (1 + GetConvDilations()[1] * (filter_w - 1)) + 2 * GetConvPads()[1]) /
                        GetConvStrides()[1] +
                    1);
        }
    }
    else
        MIOPEN_THROW(miopenStatusInvalidValue, "Invalid Padding Mode!");

    return std::make_tuple(input_n, output_c, output_h, output_w);
}

size_t ConvolutionDescriptor::ForwardGetWorkSpaceSizeGEMM(const TensorDescriptor& wDesc,
                                                          const TensorDescriptor& yDesc) const
{
    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = miopen::tien<4>(yDesc.GetLengths());

    int wei_c, wei_h, wei_w;
    std::tie(std::ignore, wei_c, wei_h, wei_w) = miopen::tien<4>(wDesc.GetLengths());

    size_t workspace_size = GetTypeSize(wDesc.GetType()) * wei_c * wei_h * wei_w * out_h * out_w;

    // No workspace is needed for 1x1_stride=1 convolutions
    if(wei_h == 1 && wei_w == 1 && GetConvStrides()[0] == 1 && GetConvStrides()[1] == 1 &&
       GetConvPads()[0] == 0 && GetConvPads()[1] == 0)
    {
        if(wDesc.GetType() == miopenInt8)
            return workspace_size;
        else
            return 0;
    }

    return (wDesc.GetType() == miopenInt8 ? 2 * workspace_size : workspace_size);
}

size_t
ConvolutionDescriptor::ForwardGetWorkSpaceSizeGEMMTranspose(const TensorDescriptor& xDesc,
                                                            const TensorDescriptor& yDesc) const
{

    int in_n, in_c;
    std::tie(in_n, in_c, std::ignore, std::ignore) = tien<4>(xDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(yDesc.GetLengths());

    size_t x_t_size = GetTypeSize(xDesc.GetType()) * in_n * in_c * out_h * out_w;
    if(xDesc.GetType() == miopenInt8)
        x_t_size *= 2;

    size_t y_t_size = yDesc.GetElementSize() * GetTypeSize(yDesc.GetType());

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

/// \todo GET RID OF THIS FUNCTION. --atamazov
/// At least, this function must be re-implemented by leveraging
/// IsApplicable() from respective Solvers.
bool ConvolutionDescriptor::IsDirectSupported(const TensorDescriptor& wDesc) const
{
    int k, c, _kernel_size0, _kernel_size1;
    std::tie(k, c, _kernel_size0, _kernel_size1) = tien<4>(wDesc.GetLengths());

    bool supported_filters =
        ((_kernel_size0 == 1 && _kernel_size1 == 1) || (_kernel_size0 == 3 && _kernel_size1 == 3) ||
         (_kernel_size0 == 5 && _kernel_size1 == 5) || (_kernel_size0 == 7 && _kernel_size1 == 7) ||
         (_kernel_size0 == 9 && _kernel_size1 == 9) ||
         (_kernel_size0 == 11 && _kernel_size1 == 11) ||
         (_kernel_size0 == 5 && _kernel_size1 == 10 && GetConvStrides()[0] == 2 &&
          GetConvStrides()[1] == 2 && GetConvPads()[0] == 0 && GetConvPads()[1] == 0) ||
         (_kernel_size0 == 5 && _kernel_size1 == 20 && GetConvStrides()[0] == 2 &&
          GetConvStrides()[1] == 2 && GetConvPads()[0] == 0 && GetConvPads()[1] == 0));

    bool workarounds = ((_kernel_size0 == 3 && _kernel_size1 == 3 &&
                         (GetConvStrides()[0] > 2 || GetConvStrides()[1] > 2)) ||
                        (_kernel_size0 == 1 && _kernel_size1 == 1 &&
                         (GetConvPads()[0] > 0 || GetConvPads()[1] > 0)) ||
                        (_kernel_size0 % 2 == 0 && _kernel_size1 % 2 == 0));

    return (supported_filters && !workarounds) || group_count >= 2;
}

size_t ConvolutionDescriptor::ForwardGetWorkSpaceSize(Handle& handle,
                                                      const TensorDescriptor& wDesc,
                                                      const TensorDescriptor& xDesc,
                                                      const TensorDescriptor& yDesc) const
{
    MIOPEN_LOG_I2("");
    {
        int wei_h, wei_w;
        std::tie(std::ignore, std::ignore, wei_h, wei_w) = tien<4>(wDesc.GetLengths());
        int in_c, in_h, in_w;
        std::tie(std::ignore, in_c, in_h, in_w) = tien<4>(xDesc.GetLengths());

        const size_t direct_workspace =
            (GetConvDilations()[0] == 1 && GetConvDilations()[1] == 1 &&
             wDesc.GetType() != miopenInt8)
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
        if((wei_h == 1 && wei_w == 1 && GetConvPads()[0] == 0 && GetConvPads()[1] == 0) &&
           ((in_h <= 14 && in_w <= 14 && GetConvStrides()[0] == 1 && GetConvStrides()[1] == 1) ||
            (GetConvStrides()[0] == 2 && GetConvStrides()[1] == 2)))
        {
            size_t gemm_trans = ForwardGetWorkSpaceSizeGEMMTranspose(xDesc, yDesc);
            /// \todo WORKAROUND for issue 1430
            if(gemm_trans > MAX_MEM_ALLOC_SZ /* handle.GetMaxMemoryAllocSize() */)
                gemm_trans = 0;
            return std::max(gemm_trans, direct_workspace);
        }
        if(GetConvDilations()[1] > 1 || GetConvDilations()[0] > 1)
            return std::max(workspace_size_gemm, direct_workspace);
#endif

        // Check if Winograd is available
        // If Winograd is present, there is no advantage in letting
        // the user run another algorithm as those both slower and
        // use more workspace.
        if(IsWinograd3x3Supported(handle, true, wDesc, xDesc) && wDesc.GetType() != miopenInt8)
        {
            return 0;
        }
        else
        {
            size_t workspace_size_fft = (GetConvDilations()[0] == 1 && GetConvDilations()[1] == 1 &&
                                         wDesc.GetType() != miopenInt8)
                                            ? ForwardGetWorkSpaceSizeFFT(wDesc, xDesc, yDesc)
                                            : 0;

            return std::max(std::max(workspace_size_fft, workspace_size_gemm), direct_workspace);
        }
    }
}

size_t ConvolutionDescriptor::BackwardDataGetWorkSpaceSize(Handle& handle,
                                                           const TensorDescriptor& wDesc,
                                                           const TensorDescriptor& dyDesc,
                                                           const TensorDescriptor& dxDesc) const
{
    MIOPEN_LOG_I2("");
    {
        int wei_h, wei_w;
        std::tie(std::ignore, std::ignore, wei_h, wei_w) = tien<4>(wDesc.GetLengths());

        const size_t direct_workspace =
            (GetConvDilations()[0] == 1 && GetConvDilations()[1] == 1 &&
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
        if(wei_h == 1 && wei_w == 1 && GetConvPads()[0] == 0 && GetConvPads()[1] == 0 &&
           (GetConvStrides()[0] == 2 && GetConvStrides()[1] == 2))
        {
            size_t gemm_trans = BackwardDataGetWorkSpaceSizeGEMMTranspose(dyDesc, dxDesc);
            /// \todo WORKAROUND for issue 1430
            if(gemm_trans > MAX_MEM_ALLOC_SZ /*  handle.GetMaxMemoryAllocSize() */)
                gemm_trans = 0;
            return std::max(gemm_trans, direct_workspace);
        }
        if(GetConvDilations()[1] > 1 || GetConvDilations()[0] > 1)
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
            size_t workspace_size_fft = (GetConvDilations()[0] == 1 && GetConvDilations()[1] == 1 &&
                                         wDesc.GetType() != miopenInt8)
                                            ? BackwardGetWorkSpaceSizeFFT(wDesc, dyDesc, dxDesc)
                                            : 0;

            return std::max(std::max(workspace_size_fft, workspace_size_gemm), direct_workspace);
        }
    }
}

// weights_n = output_c
// weights_c = input_c
// weights_h = 2*GetConvPads()[0] + input_h - GetConvStrides()[0]*(output_h - 1)
// weights_w = 2*GetConvPads()[1] + input_w - GetConvStrides()[1]*(output_w - 1)
std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>
ConvolutionDescriptor::GetBackwardsWeightsDim(const TensorDescriptor& inputTensorDesc,
                                              const TensorDescriptor& outputTensorDesc) const
{
    assert(inputTensorDesc.GetLengths().size() == 4);
    assert(outputTensorDesc.GetLengths().size() == 4);

    if(inputTensorDesc.GetType() != outputTensorDesc.GetType())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Types do not match for the filter");
    }

    std::size_t input_n;
    std::size_t input_c;
    std::size_t input_h;
    std::size_t input_w;

    std::tie(input_n, input_c, input_h, input_w) = miopen::tien<4>(inputTensorDesc.GetLengths());

    std::size_t output_n;
    std::size_t output_c;
    std::size_t output_h;
    std::size_t output_w;

    std::tie(output_n, output_c, output_h, output_w) =
        miopen::tien<4>(outputTensorDesc.GetLengths());

    // if(input_c != (filter_c * group_count)) {
    //  MIOPEN_THROW(miopenStatusBadParm, "Channels do not match for the filter");
    // }

    return std::make_tuple(output_c,
                           input_c / group_count,
                           2 * GetConvPads()[0] + input_h - GetConvStrides()[0] * (output_h - 1),
                           2 * GetConvPads()[1] + input_w - GetConvStrides()[1] * (output_w - 1));
}

std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>
ConvolutionDescriptor::GetBackwardOutputDim(const TensorDescriptor& outputTensorDesc,
                                            const TensorDescriptor& filterDesc) const
{
    assert(outputTensorDesc.GetLengths().size() == 4);
    assert(filterDesc.GetLengths().size() == 4);

    if(outputTensorDesc.GetType() != filterDesc.GetType())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Types do not match for the filter");
    }

    std::size_t output_n;
    std::size_t output_c;
    std::size_t output_h;
    std::size_t output_w;

    std::tie(output_n, output_c, output_h, output_w) =
        miopen::tien<4>(outputTensorDesc.GetLengths());

    std::size_t filter_k;
    std::size_t filter_c;
    std::size_t filter_h;
    std::size_t filter_w;

    std::tie(filter_k, filter_c, filter_h, filter_w) = miopen::tien<4>(filterDesc.GetLengths());

    if(output_c != filter_k)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Channels do not match for the filter");
    }

    return std::make_tuple(output_n,
                           filter_c * group_count,
                           GetConvStrides()[0] * (output_h - 1) - 2 * GetConvPads()[0] + filter_h,
                           GetConvStrides()[1] * (output_w - 1) - 2 * GetConvPads()[1] + filter_w);
}

TensorDescriptor
ConvolutionDescriptor::GetForwardOutputTensor(const TensorDescriptor& inputTensorDesc,
                                              const TensorDescriptor& filterDesc) const
{
    auto dims = this->GetForwardOutputDim(inputTensorDesc, filterDesc);
    return TensorDescriptor(
        (inputTensorDesc.GetType() == miopenInt8 ? miopenFloat : inputTensorDesc.GetType()),
        {std::get<0>(dims), std::get<1>(dims), std::get<2>(dims), std::get<3>(dims)});
}

TensorDescriptor
ConvolutionDescriptor::GetBackwardOutputTensor(const TensorDescriptor& outputTensorDesc,
                                               const TensorDescriptor& filterDesc) const
{
    auto dims = this->GetBackwardOutputDim(outputTensorDesc, filterDesc);
    return TensorDescriptor(
        outputTensorDesc.GetType(),
        {std::get<0>(dims), std::get<1>(dims), std::get<2>(dims), std::get<3>(dims)});
}

TensorDescriptor
ConvolutionDescriptor::GetBackwardWeightsTensor(const TensorDescriptor& inputTensorDesc,
                                                const TensorDescriptor& outputTensorDesc) const
{
    auto dims = this->GetBackwardsWeightsDim(inputTensorDesc, outputTensorDesc);
    return TensorDescriptor(
        outputTensorDesc.GetType(),
        {std::get<0>(dims), std::get<1>(dims), std::get<2>(dims), std::get<3>(dims)});
}

size_t ConvolutionDescriptor::BackwardDataGetWorkSpaceSizeGEMM(const TensorDescriptor& wDesc,
                                                               const TensorDescriptor& dyDesc) const
{
    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = miopen::tien<4>(dyDesc.GetLengths());
    int wei_c, wei_h, wei_w;
    std::tie(std::ignore, wei_c, wei_h, wei_w) = miopen::tien<4>(wDesc.GetLengths());
    size_t gemm_size = GetTypeSize(dyDesc.GetType()) * wei_c * wei_h * wei_w * out_h * out_w;

    // No workspace is needed for 1x1_stride=1 convolutions
    if(wei_h == 1 && wei_w == 1 && GetConvStrides()[0] == 1 && GetConvStrides()[1] == 1 &&
       GetConvPads()[0] == 0 && GetConvPads()[1] == 0)
    {
        return 0;
    }

    return gemm_size;
}

size_t ConvolutionDescriptor::BackwardDataGetWorkSpaceSizeGEMMTranspose(
    const TensorDescriptor& dyDesc, const TensorDescriptor& dxDesc) const
{
    int in_n, in_c;
    std::tie(in_n, in_c, std::ignore, std::ignore) = tien<4>(dxDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(dyDesc.GetLengths());

    size_t dx_t_size = GetTypeSize(dxDesc.GetType()) * in_n * in_c * out_h * out_w;

    size_t dy_t_size = dyDesc.GetElementSize() * GetTypeSize(dyDesc.GetType());

    return dx_t_size + dy_t_size;
}

size_t
ConvolutionDescriptor::BackwardWeightsGetWorkSpaceSizeGEMM(const TensorDescriptor& dyDesc,
                                                           const TensorDescriptor& dwDesc) const
{
    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = miopen::tien<4>(dyDesc.GetLengths());
    int wei_c, wei_h, wei_w;
    std::tie(std::ignore, wei_c, wei_h, wei_w) = miopen::tien<4>(dwDesc.GetLengths());
    size_t gemm_size = GetTypeSize(dyDesc.GetType()) * wei_c * wei_h * wei_w * out_h * out_w;

    // No workspace is needed for 1x1_stride=1 convolutions
    if(wei_h == 1 && wei_w == 1 && GetConvStrides()[0] == 1 && GetConvStrides()[1] == 1 &&
       GetConvPads()[0] == 0 && GetConvPads()[1] == 0)
    {
        return 0;
    }

    return gemm_size;
}

size_t ConvolutionDescriptor::ForwardBackwardDataGetWorkSpaceSizeDirect(
    Handle& handle,
    const TensorDescriptor& xDesc,
    const TensorDescriptor& yDesc,
    const TensorDescriptor& wDesc,
    int direction) const // 1: Forward, 0: BackwardData
{

    if(!IsDirectSupported(wDesc) || miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT{}))
        return 0;

    mlo_construct_direct2D construct_params(xDesc, wDesc, yDesc, *this, direction);
    construct_params.setDoSearch(false);
    construct_params.setStream(&handle);
    construct_params.setWorkaroundDisableSearchEnforce(true);

    try
    {
        const auto ss = FindAllSolutions(construct_params);
        size_t sz     = 0;
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

size_t
ConvolutionDescriptor::BackwardWeightsGetWorkSpaceSizeDirect(Handle& handle,
                                                             const TensorDescriptor& dyDesc,
                                                             const TensorDescriptor& xDesc,
                                                             const TensorDescriptor& dwDesc) const
{
    mlo_construct_BwdWrW2D construct_params(
        xDesc, dwDesc, dyDesc, *this, 0); // backward with regards to weights
    construct_params.setDoSearch(false);
    construct_params.setStream(&handle);
    construct_params.setWorkaroundDisableSearchEnforce(true);
    try
    {
        const auto ss = FindAllSolutions(construct_params);
        size_t sz     = 0;
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

size_t ConvolutionDescriptor::ConvolutionBackwardWeightsGetWorkSpaceSize(
    Handle& handle,
    const TensorDescriptor& dyDesc,
    const TensorDescriptor& xDesc,
    const TensorDescriptor& dwDesc) const
{
    MIOPEN_LOG_I2("");

    size_t workspace_size = 0;
    {
        size_t workspace_size_gemm =
#if MIOPEN_USE_GEMM
            BackwardWeightsGetWorkSpaceSizeGEMM(dyDesc, dwDesc) * group_count;
        /// \todo WORKAROUND for issue 1430
        if(workspace_size_gemm > MAX_MEM_ALLOC_SZ /*  handle.GetMaxMemoryAllocSize() */)
            workspace_size_gemm =
#endif
                0;

        size_t direct_workspace =
            (GetConvDilations()[0] == 1 && GetConvDilations()[1] == 1 &&
             dwDesc.GetType() != miopenInt8)
                ? BackwardWeightsGetWorkSpaceSizeDirect(handle, dyDesc, xDesc, dwDesc)
                : 0;

        workspace_size = std::max(direct_workspace, workspace_size_gemm);
    }

    return workspace_size;
}

std::ostream& operator<<(std::ostream& stream, const ConvolutionDescriptor& c)
{
    MIOPEN_LOG_ENUM(stream, c.mode, miopenConvolution, miopenTranspose) << ", ";
    stream << c.GetConvPads()[0] << ", ";
    stream << c.GetConvPads()[1] << ", ";
    stream << c.GetConvStrides()[0] << ", ";
    stream << c.GetConvStrides()[1] << ", ";
    stream << c.GetConvDilations()[0] << ", ";
    stream << c.GetConvDilations()[1] << ", ";
    if(c.group_count > 1)
        stream << c.group_count << ", ";
    if(c.mode == miopenTranspose)
    {
        stream << c.GetTransposeConvPads()[0] << ", ";
        stream << c.GetTransposeConvPads()[1] << ", ";
    }
    return stream;
}

} // namespace miopen
