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
#pragma once

#include <random>

#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/solver_id.hpp>
#include <serialize.hpp>
#include <fusionHost.hpp>

#include "tensor_util.hpp"
#include "get_handle.hpp"
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

struct ConvTestCase
{
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
    size_t dialtion_x;
    size_t dilation_y;
    friend std::ostream& operator<<(std::ostream& os, const ConvTestCase& tc)
    {
        return os << "(N: " << tc.N << " C:" << tc.C << " H:" << tc.H << " W:" << tc.W
                  << " k: " << tc.k << " y:" << tc.y << " x:" << tc.x << " pad_y:" << tc.pad_y
                  << " pad_x:" << tc.pad_x << " stride_y:" << tc.stride_y
                  << " dilation_y:" << tc.dilation_y << " )";
    }
    std::vector<size_t> GetInput() { return {N, C, H, W}; }
    std::vector<size_t> GetWeights() { return {k, C, y, x}; }
    miopen::ConvolutionDescriptor GetConv()
    {
        return miopen::ConvolutionDescriptor{
            {static_cast<int>(pad_y), static_cast<int>(pad_x)},
            {static_cast<int>(stride_y), static_cast<int>(stride_x)},
            {static_cast<int>(dilation_y), static_cast<int>(dilation_y)}};
    }
};

std::vector<ConvTestCase> GetNetwork1()
{
    // pyt_mlperf_resnet50v1.5
    return {{64, 1024, 14, 14, 2048, 1, 1, 0, 0, 2, 2, 1, 1},
            {64, 1024, 14, 14, 256, 1, 1, 0, 0, 1, 1, 1, 1},
            {64, 1024, 14, 14, 512, 1, 1, 0, 0, 1, 1, 1, 1},
            {64, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1},
            {64, 128, 28, 28, 512, 1, 1, 0, 0, 1, 1, 1, 1},
            {64, 128, 56, 56, 128, 3, 3, 1, 1, 2, 2, 1, 1},
            {64, 2048, 7, 7, 512, 1, 1, 0, 0, 1, 1, 1, 1},
            {64, 256, 14, 14, 1024, 1, 1, 0, 0, 1, 1, 1, 1},
            {64, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1},
            {64, 256, 28, 28, 256, 3, 3, 1, 1, 2, 2, 1, 1},
            {64, 256, 56, 56, 128, 1, 1, 0, 0, 1, 1, 1, 1},
            {64, 256, 56, 56, 512, 1, 1, 0, 0, 2, 2, 1, 1},
            {64, 256, 56, 56, 64, 1, 1, 0, 0, 1, 1, 1, 1},
            {64, 3, 224, 224, 64, 7, 7, 3, 3, 2, 2, 1, 1},
            {64, 512, 14, 14, 512, 3, 3, 1, 1, 2, 2, 1, 1},
            {64, 512, 28, 28, 1024, 1, 1, 0, 0, 2, 2, 1, 1},
            {64, 512, 28, 28, 128, 1, 1, 0, 0, 1, 1, 1, 1},
            {64, 512, 28, 28, 256, 1, 1, 0, 0, 1, 1, 1, 1},
            {64, 512, 7, 7, 2048, 1, 1, 0, 0, 1, 1, 1, 1},
            {64, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1, 1, 1},
            {64, 64, 56, 56, 256, 1, 1, 0, 0, 1, 1, 1, 1},
            {64, 64, 56, 56, 64, 1, 1, 0, 0, 1, 1, 1, 1},
            {64, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1}};
}

template <typename T = float>
struct ConvBiasActivInferTest
    : public ::testing::TestWithParam<std::tuple<miopenActivationMode_t, ConvTestCase>>
{
protected:
    void SetUp() override
    {
        test_skipped                      = false;
        std::tie(activ_mode, conv_config) = GetParam();
        input                             = tensor<T>{conv_config.GetInput()};
        weights                           = tensor<T>{conv_config.GetWeights()};
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<> d{-3, 3};
        auto gen_value = [&](auto...) { return d(gen); };
        input.generate(gen_value);
        weights.generate(gen_value);
        activ_desc = {activ_mode, activ_alpha, activ_beta, activ_gamma};
        conv_desc  = conv_config.GetConv();
        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, GetDataType<T>());
        output = tensor<T>{output_desc.GetLengths()};
        bias   = tensor<T>{1, static_cast<size_t>(conv_config.k), 1, 1};
        bias.generate(gen_value);
        auto&& handle = get_handle();
        std::fill(output.begin(), output.end(), std::numeric_limits<double>::quiet_NaN());
        in_dev   = handle.Write(input.data);
        wei_dev  = handle.Write(weights.data);
        out_dev  = handle.Write(output.data);
        bias_dev = handle.Write(bias.data);

        // Setup the Fusionplan
        fusePlanDesc = miopen::FusionPlanDescriptor(miopenVerticalFusion, input.desc);
        auto convOp  = std::make_shared<miopen::ConvForwardOpDescriptor>(conv_desc, weights.desc);
        auto biasOp  = std::make_shared<miopen::BiasFusionOpDescriptor>(bias.desc);
        auto activOp = std::make_shared<miopen::ActivFwdFusionOpDescriptor>(activ_desc.GetMode());
        EXPECT_EQ(fusePlanDesc.AddOp(convOp), miopenStatusSuccess);
        // miopen::OperatorArgs params;
        convOp->SetArgs(params, &alpha, &beta, wei_dev.get());
        EXPECT_EQ(fusePlanDesc.AddOp(biasOp), miopenStatusSuccess);
        biasOp->SetArgs(params, &alpha, &beta, bias_dev.get());
        EXPECT_EQ(fusePlanDesc.AddOp(activOp), miopenStatusSuccess);
        activOp->SetArgs(params, &alpha, &beta, activ_alpha, activ_beta, activ_gamma);
    }
    void TearDown() override
    {
        if(test_skipped)
            return;
        conv_stats stats;
        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopenFloat);
        ref_out = tensor<T>{output_desc.GetLengths()};
        ref_out = ref_conv_fwd(input, weights, output, conv_desc);
        cpu_bias_forward(ref_out, bias);
        activationHostInfer(
            activ_mode, activ_gamma, activ_beta, activ_alpha, ref_out.data, ref_out.data);
        auto&& handle = get_handle();
        output.data   = handle.Read<T>(out_dev, output.data.size());
        EXPECT_FALSE(miopen::range_zero(ref_out)) << "CPU data is all zeros";
        EXPECT_FALSE(miopen::range_zero(output)) << "GPU data is all zeros";
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
    miopen::ActivationDescriptor activ_desc;
    tensor<T> input;
    tensor<T> weights;
    tensor<T> output;
    tensor<T> bias;
    tensor<T> ref_out;
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr wei_dev;
    miopen::Allocator::ManageDataPtr out_dev;
    miopen::Allocator::ManageDataPtr bias_dev;
    bool test_skipped = false;
    miopenActivationMode_t activ_mode;
    miopen::FusionPlanDescriptor fusePlanDesc;
    // std::unique_ptr<miopen::fusion::FusionInvokeParams> plan_params;
    miopen::OperatorArgs params;
    const float alpha       = static_cast<float>(1.0f);
    const float beta        = static_cast<float>(0);
    const float activ_alpha = static_cast<double>(0.5f);
    const float activ_beta  = static_cast<double>(0.5f);
    const float activ_gamma = static_cast<double>(0.5f);
};
