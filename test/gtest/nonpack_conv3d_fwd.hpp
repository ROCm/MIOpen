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

struct NonPackTestCase : Conv3DTestCase
{
    size_t i0;
    size_t i1;
    size_t i2;
    size_t i3;
    size_t i4;
    size_t w0;
    size_t w1;
    size_t w2;
    size_t w3;
    size_t w4;
    size_t o0;
    size_t o1;
    size_t o2;
    size_t o3;
    size_t o4;
    std::vector<size_t> GetInputStrides() { return {i0, i1, i2, i3, i4}; }
    std::vector<size_t> GetWeightStrides() { return {w0, w1, w2, w3, w4}; }
    std::vector<size_t> GetOutputStrides() { return {o0, o1, o2, o3, o4}; }
};

template <>
std::vector<NonPackTestCase> ConvTestConfigs()
{ // g   n   c   k   image   filter   pad   stride   dilation
    // clang-format off
    return {{
                {1, 4, 16, 16, {4, 9, 16}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenConvolution},
                10240,
                1,
                2560,
                160,
                16,
                432,
                1,
                144,
                48,
                16,
                9216,
                1,
                2304,
                256,
                16,
            },
            {
                {1, 1, 64, 128, {3, 16, 16}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenConvolution},
                65536,
                1,
                24000,
                2048,
                64,
                1728,
                1,
                576,
                192,
                64,
                98304,
                1,
                32768,
                2048,
                128,
            }};
    // clang-format on
}

template <typename T = float>
struct ConvNonpackFwdSolverTest3D
    : public ::testing::TestWithParam<std::tuple<miopenConvFwdAlgorithm_t,
                                                 NonPackTestCase,
                                                 double,
                                                 double,
                                                 miopenTensorLayout_t>>
{
protected:
    void SetUp() override
    {
        test_skipped = false;

        std::tie(algo, conv_config, alpha_val, beta_val, tensor_layout) = GetParam();
        input   = tensor<T>{tensor_layout, conv_config.GetInput(), conv_config.GetInputStrides()};
        weights = tensor<T>{tensor_layout, conv_config.GetWeights()};
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<> d{-3, 3};
        auto gen_value = [&](auto...) { return d(gen); };
        input.generate(gen_value);
        weights.generate(gen_value);

        conv_desc = conv_config.GetConv();

        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<T>{});
        output = tensor<T>{tensor_layout, output_desc.GetLengths()};
        // since now we do alpha*value + output*beta
        // we set output with values other then nan.
        std::fill(output.begin(), output.end(), 0.0);
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
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<T>{});
        ref_out = tensor<T>{tensor_layout, output_desc.GetLengths()};
        ref_out = ref_conv_fwd(
            input, weights, output, conv_desc, miopen::Scalar(alpha_val), miopen::Scalar(beta_val));

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
    NonPackTestCase conv_config;
    miopen::ConvolutionDescriptor conv_desc;
    tensor<T> input;
    tensor<T> weights;
    tensor<T> output;
    tensor<T> ref_out;

    float alpha_val = 1.0f;
    float beta_val  = 0.0f;

    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr wei_dev;
    miopen::Allocator::ManageDataPtr out_dev;
    miopenConvFwdAlgorithm_t algo = miopenConvolutionFwdAlgoImplicitGEMM;
    bool test_skipped             = false;
    miopenTensorLayout_t tensor_layout;
};
