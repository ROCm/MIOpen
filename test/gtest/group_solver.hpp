/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#pragma once

#include <random>

#include "get_handle.hpp"
#include <miopen/conv/data_invoke_params.hpp>

#include "../driver/tensor_driver.hpp"
#include "conv_common.hpp"

template <typename T>
miopenDataType_t GetDataType();

template <>
miopenDataType_t GetDataType<float>()
{
    return miopenFloat;
}

template <>
miopenDataType_t GetDataType<half_float::half>()
{
    return miopenHalf;
}

template <>
miopenDataType_t GetDataType<int8_t>()
{
    return miopenInt8;
}

struct ConvTestCase
{
    size_t G;
    size_t N;
    size_t C;
    size_t H;
    size_t W;
    size_t k;
    size_t y;
    size_t x;
    size_t pad_x;
    size_t pad_y;
    size_t stride_x;
    size_t stride_y;
    size_t dilation_x;
    size_t dilation_y;
    miopenConvolutionMode_t conv_mode;
    friend std::ostream& operator<<(std::ostream& os, const ConvTestCase& tc)
    {
        return os << " G:" << tc.G << " N:" << tc.N << " C:" << tc.C << " H:" << tc.H
                  << " W:" << tc.W << " k:" << tc.k << " y:" << tc.y << " x:" << tc.x
                  << " pad_y:" << tc.pad_y << " pad_x:" << tc.pad_x << " stride_y:" << tc.stride_y
                  << " dilation_y:" << tc.dilation_y << " conv_mode:" << tc.conv_mode;
    }

    std::vector<size_t> GetInput() { return {N, C, H, W}; }
    std::vector<size_t> GetWeights()
    {
        EXPECT_EQUAL(C % G, 0);
        return {k, C / G, y, x};
    }

    miopen::ConvolutionDescriptor GetConv()
    {
        return miopen::ConvolutionDescriptor{
            2,
            miopenConvolution,
            miopenPaddingDefault,
            {static_cast<int>(pad_y), static_cast<int>(pad_x)},
            {static_cast<int>(stride_y), static_cast<int>(stride_x)},
            {static_cast<int>(dilation_y), static_cast<int>(dilation_x)},
            {0, 0},
            static_cast<int>(G),
            1.0};
    }
};

std::vector<ConvTestCase> ConvTestConfigs()
{ // g  n  c   h   w   k   y  x pad_x pad_y stri_x stri_y dia_x dia_y
    return {{1, 256, 192, 28, 28, 192, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {1, 256, 12, 28, 28, 12, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {4, 256, 192, 28, 28, 192, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {8, 256, 192, 28, 28, 192, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {8, 256, 384, 28, 28, 384, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {32, 256, 1024, 28, 28, 2048, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution}};
}

inline int SetTensorLayout(miopen::TensorDescriptor& desc)
{
    // get layout string names
    std::string layout_str = desc.GetLayout_str();

    std::vector<std::size_t> lens = desc.GetLengths();
    std::vector<int> int_lens(lens.begin(), lens.end());

    // set the strides for the tensor
    return SetTensorNd(&desc, int_lens, layout_str, desc.GetType());
}

template <typename T = float>
struct ConvFwdSolverTest
    : public ::testing::TestWithParam<
          std::tuple<miopenConvFwdAlgorithm_t, ConvTestCase, miopenTensorLayout_t>>
{
protected:
    void SetUp() override
    {
        test_skipped                               = false;
        std::tie(algo, conv_config, tensor_layout) = GetParam();

        input   = tensor<T>{tensor_layout, conv_config.GetInput()};
        weights = tensor<T>{tensor_layout, conv_config.GetWeights()};
        SetTensorLayout(input.desc);
        SetTensorLayout(weights.desc);
        auto gen_value = [](auto...) {
            return prng::gen_A_to_B(static_cast<T>(-3.0), static_cast<T>(3.0));
        };
        input.generate(gen_value);
        weights.generate(gen_value);

        conv_desc = conv_config.GetConv();

        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, GetDataType<T>());
        output = tensor<T>{tensor_layout, output_desc.GetLengths()};
        SetTensorLayout(output.desc);
        std::fill(output.begin(), output.end(), std::numeric_limits<double>::quiet_NaN());

        auto&& handle = get_handle();
        in_dev        = handle.Write(input.data);
        wei_dev       = handle.Write(weights.data);
        out_dev       = handle.Write(output.data);
    }
    void TearDown() override
    {
        if(test_skipped)
            return;

        auto&& handle = get_handle();

        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, GetDataType<T>());
        ref_out     = tensor<T>{tensor_layout, output_desc.GetLengths()};
        ref_out     = ref_conv_fwd(input, weights, output, conv_desc);
        output.data = handle.Read<T>(out_dev, output.data.size());
        EXPECT_FALSE(miopen::range_zero(ref_out)) << "Cpu data is all zeros";
        EXPECT_FALSE(miopen::range_zero(output)) << "Gpu data is all zeros";
        EXPECT_TRUE(miopen::range_distance(ref_out) == miopen::range_distance(output));

        const double tolerance = 80;
        double threshold       = std::numeric_limits<T>::epsilon() * tolerance;
        auto error             = miopen::rms_range(ref_out, output);

        EXPECT_FALSE(miopen::find_idx(ref_out, miopen::not_finite) >= 0)
            << "Non finite number found in the CPU data";

        EXPECT_TRUE(error < threshold)
            << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;
    }
    ConvTestCase conv_config;
    miopen::ConvolutionDescriptor conv_desc;
    tensor<T> input;
    tensor<T> weights;
    tensor<T> output;
    tensor<T> ref_out;
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr wei_dev;
    miopen::Allocator::ManageDataPtr out_dev;
    miopenConvFwdAlgorithm_t algo = miopenConvolutionFwdAlgoImplicitGEMM;
    bool test_skipped             = false;
    miopenTensorLayout_t tensor_layout;
};
