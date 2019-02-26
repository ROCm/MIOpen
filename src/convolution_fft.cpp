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
#include <miopen/convolution.hpp>
#include <miopen/convolution_fft.hpp>
#include <miopen/env.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

static size_t GetWorkSpaceSizeFFT(const TensorDescriptor& wDesc,
                                  const TensorDescriptor& xDesc,
                                  const TensorDescriptor& yDesc,
                                  const std::tuple<int, int, int, int> cparam,
                                  bool fwd)
{
    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = miopen::tien<4>(xDesc.GetLengths());

    int out_n, out_c, out_h, out_w;
    std::tie(out_n, out_c, out_h, out_w) = miopen::tien<4>(yDesc.GetLengths());

    int wei_k, wei_c, wei_h, wei_w;
    std::tie(wei_k, wei_c, wei_h, wei_w) = miopen::tien<4>(wDesc.GetLengths());

    bool supported = true;

    // FFT convolutions only works for specific config(s)
    // coverage to expand gradually

    supported = ((in_n < 1) || (in_n > 512)) ? false : supported;
    supported = ((wei_k < 1) || (wei_k > 512)) ? false : supported;
    supported = ((in_c * in_n) % 16 != 0) ? false : supported;
    supported = ((wei_c * wei_k) % 16 != 0) ? false : supported;
    supported = ((out_c * out_n) % 16 != 0) ? false : supported;

    supported = ((std::tie(in_h, in_w) != std::make_tuple(28, 28)) &&
                 (std::tie(in_h, in_w) != std::make_tuple(27, 27)) &&
                 (std::tie(in_h, in_w) != std::make_tuple(14, 14)) &&
                 (std::tie(in_h, in_w) != std::make_tuple(7, 7)))
                    ? false
                    : supported;

    supported = (std::tie(wei_h, wei_w) != std::make_tuple(5, 5)) ? false : supported;
    supported = (cparam != std::make_tuple(2, 2, 1, 1)) ? false : supported;
    supported = (yDesc.GetType() != miopenFloat) ? false : supported;

    const int N       = FFTConvParams::TileSize(in_h, in_w);
    const int Padding = FFTConvParams::TransposePadding;

    if(supported)
    {
        int temp_size = 0;

        if(fwd)
        {
            int temp_size1 = (in_c * in_n + Padding) + (wei_k * wei_c + Padding);
            int temp_size2 = (out_n * out_c + Padding);
            temp_size      = temp_size1 > temp_size2 ? temp_size1 : temp_size2;
        }
        else
        {
            int temp_size1 = (out_n * out_c + Padding) + (wei_k * wei_c + Padding);
            int temp_size2 = (in_c * in_n + Padding);
            temp_size      = temp_size1 > temp_size2 ? temp_size1 : temp_size2;
        }

        return 2 * 2 * N * temp_size * sizeof(float);
    }
    else
        return 0;
}

size_t ConvolutionDescriptor::ForwardGetWorkSpaceSizeFFT(const TensorDescriptor& wDesc,
                                                         const TensorDescriptor& xDesc,
                                                         const TensorDescriptor& yDesc) const
{
    return GetWorkSpaceSizeFFT(
        wDesc,
        xDesc,
        yDesc,
        std::make_tuple(
            GetConvPads()[0], GetConvPads()[1], GetConvStrides()[0], GetConvStrides()[1]),
        true);
}

size_t ConvolutionDescriptor::BackwardGetWorkSpaceSizeFFT(const TensorDescriptor& wDesc,
                                                          const TensorDescriptor& dyDesc,
                                                          const TensorDescriptor& dxDesc) const
{
    return GetWorkSpaceSizeFFT(
        wDesc,
        dxDesc,
        dyDesc,
        std::make_tuple(
            GetConvPads()[0], GetConvPads()[1], GetConvStrides()[0], GetConvStrides()[1]),
        false);
}

} // namespace miopen
