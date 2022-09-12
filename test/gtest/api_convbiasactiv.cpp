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
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/solver_id.hpp>
#include <serialize.hpp>

#include "tensor_util.hpp"
#include "get_handle.hpp"

// Demonstrate some basic assertions.
TEST(ConvBiasActivFwd, DriveAPI)
{
    const double double_zero = 0.0f;
    tensor<float> input{16, 128, 16, 16};
    tensor<float> weights{1, 128, 3, 3};
    tensor<float> z{};
    miopenConvolutionDescriptor_t conv_desc;
    miopenActivationDescriptor_t activ_desc;
    input.generate(tensor_elem_gen_integer{17});
    weights.generate(tensor_elem_gen_integer{17});
    miopenCreateConvolutionDescriptor(&conv_desc);
    miopenCreateActivationDescriptor(&activ_desc);
    miopenSetActivationDescriptor(
        activ_desc, miopenActivationRELU, double_zero, double_zero, double_zero);
    miopenInitConvolutionDescriptor(conv_desc, miopenConvolution, 0, 0, 1, 1, 1, 1);
    int n, c, h, w;
    miopenGetConvolutionForwardOutputDim(conv_desc, &input.desc, &weights.desc, &n, &c, &h, &w);
    tensor<float> output{static_cast<size_t>(n),
                         static_cast<size_t>(c),
                         static_cast<size_t>(h),
                         static_cast<size_t>(w)};
    tensor<float> bias{1, static_cast<size_t>(c), 1, 1};
    const float alpha                   = 1.0f;
    auto&& handle                       = get_handle();
    auto in_dev                         = handle.Write(input.data);
    auto wei_dev                        = handle.Write(weights.data);
    auto out_dev                        = handle.Write(output.data);
    auto bias_dev                       = handle.Write(bias.data);
    const miopenConvFwdAlgorithm_t algo = miopenConvolutionFwdAlgoDirect;
    const auto status                   = miopenConvolutionBiasActivationForward(&get_handle(),
                                                               &alpha,
                                                               &input.desc,
                                                               in_dev.get(),
                                                               &weights.desc,
                                                               wei_dev.get(),
                                                               conv_desc,
                                                               algo,
                                                               nullptr,
                                                               0,
                                                               &alpha,
                                                               &z.desc,
                                                               nullptr,
                                                               &bias.desc,
                                                               bias_dev.get(),
                                                               activ_desc,
                                                               &output.desc,
                                                               out_dev.get());
    EXPECT_EQ(status, miopenStatusSuccess);
}
