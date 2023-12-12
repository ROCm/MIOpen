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

#include "conv3d_test_case.hpp"

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
struct ConvWrwSolverTest
    : public ::testing::TestWithParam<
          std::tuple<miopenConvBwdWeightsAlgorithm_t, Conv3DTestCase, miopenTensorLayout_t>>
{
protected:
    void SetUp() override
    {
        test_skipped = false;

        std::tie(algo, conv_config, tensor_layout) = GetParam();
        input   = tensor<T>{tensor_layout, conv_config.GetInput()};
        weights = tensor<T>{tensor_layout, conv_config.GetWeights()};

        auto gen_value = [](auto...) {
            return prng::gen_A_to_B(static_cast<T>(-3.0), static_cast<T>(3.0));
        };
        input.generate(gen_value);

        std::fill(weights.begin(), weights.end(), 0);
        conv_desc = conv_config.GetConv();

        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, GetDataType<T>());
        output = tensor<T>{tensor_layout, output_desc.GetLengths()};
        output.generate(gen_value);
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

        ref_wei      = tensor<T>{tensor_layout, weights.desc.GetLengths()};
        ref_wei      = ref_conv_wrw(input, weights, output, conv_desc);
        weights.data = handle.Read<T>(wei_dev, weights.data.size());
        EXPECT_FALSE(miopen::range_zero(ref_wei)) << "Cpu data is all zeros";
        EXPECT_FALSE(miopen::range_zero(weights)) << "Gpu data is all zeros";
        EXPECT_FALSE(miopen::find_idx(ref_wei, miopen::not_finite) >= 0)
            << "Non finite number found in the CPU data";
        EXPECT_FALSE(miopen::find_idx(weights, miopen::not_finite) >= 0)
            << "Non finite number found in the CK GPU data";
        EXPECT_TRUE(miopen::range_distance(ref_wei) == miopen::range_distance(weights));

        const double tolerance = 80;
        double threshold       = 1e-5 * tolerance;
        auto error             = miopen::rms_range(ref_wei, weights);

        EXPECT_FALSE(miopen::find_idx(ref_wei, miopen::not_finite) >= 0)
            << "Non finite number found in the CPU data";

        EXPECT_TRUE(error < threshold)
            << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;
    }
    Conv3DTestCase conv_config;
    miopen::ConvolutionDescriptor conv_desc;
    tensor<T> input;
    tensor<T> weights;
    tensor<T> output;
    tensor<T> ref_wei;
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr wei_dev;
    miopen::Allocator::ManageDataPtr out_dev;
    miopenConvBwdWeightsAlgorithm_t algo = miopenConvolutionBwdWeightsAlgoImplicitGEMM;
    bool test_skipped                    = false;
    miopenTensorLayout_t tensor_layout;
};
