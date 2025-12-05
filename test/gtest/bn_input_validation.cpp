/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#include <miopen/errors.hpp>

#include "bn_test_data.hpp"
#include "get_handle.hpp"

// Test that batch norm training APIs reject invalid inputs matching PyTorch behavior
// PyTorch errors:
// - Spatial: "Expected more than 1 value per channel when training"
// - PerActivation: "Expected more than 1 value per channel when training" (N > 1)

struct GPU_BNInputValidation2D_FP32
    : public testing::TestWithParam<
          std::tuple<BN2DTestCase, miopenTensorLayout_t, miopenBatchNormMode_t>>
{
};

struct GPU_BNInputValidation3D_FP32
    : public testing::TestWithParam<
          std::tuple<BN3DTestCase, miopenTensorLayout_t, miopenBatchNormMode_t>>
{
};

TEST_P(GPU_BNInputValidation2D_FP32, RejectsInvalidInput)
{
    auto [config, layout, bn_mode] = GetParam();
    auto&& handle                  = get_handle();
    float alpha = 1.0f, beta = 0.0f;

    if(config.Direction == miopen::batchnorm::Direction::ForwardTraining)
    {
        BNFwdTrainTestData<float, float, float, float, float, double, BN2DTestCase> data;
        data.SetUpImpl(config, bn_mode, layout);

        auto status = miopenBatchNormalizationForwardTraining(&handle,
                                                              bn_mode,
                                                              &alpha,
                                                              &beta,
                                                              &data.input.desc,
                                                              data.in_dev.get(),
                                                              &data.output.desc,
                                                              data.out_dev.get(),
                                                              &data.scale.desc,
                                                              data.scale_dev.get(),
                                                              data.shift_dev.get(),
                                                              data.averageFactor,
                                                              data.runMean_dev.get(),
                                                              data.runVariance_dev.get(),
                                                              data.epsilon,
                                                              data.saveMean_dev.get(),
                                                              data.saveVariance_dev.get());

        const char* mode_str   = (bn_mode == miopenBNSpatial) ? "Spatial" : "PerActivation";
        const char* layout_str = (layout == miopenTensorNCHW) ? "NCHW" : "NHWC";

        EXPECT_EQ(status, miopenStatusBadParm)
            << "ForwardTraining should reject " << mode_str << " mode with " << layout_str
            << " layout: N=" << config.N << " H=" << config.H << " W=" << config.W
            << " (N*H*W=" << (config.N * config.H * config.W) << ")";
    }
    else if(config.Direction == miopen::batchnorm::Direction::Backward)
    {
        BNBwdTestData<float, float, float, float, float, float, double, BN2DTestCase> data;
        data.SetUpImpl(config, bn_mode, layout);

        auto status = miopenBatchNormalizationBackward(&handle,
                                                       bn_mode,
                                                       &alpha,
                                                       &beta,
                                                       &alpha,
                                                       &beta,
                                                       &data.input.desc,
                                                       data.in_dev.get(),
                                                       &data.dy.desc,
                                                       data.dy_dev.get(),
                                                       &data.output.desc,
                                                       data.out_dev.get(),
                                                       &data.bnScale.desc,
                                                       data.bnScale_dev.get(),
                                                       data.dScale_dev.get(),
                                                       data.dBias_dev.get(),
                                                       data.epsilon,
                                                       data.savedMean_dev.get(),
                                                       data.savedInvVar_dev.get());

        const char* mode_str   = (bn_mode == miopenBNSpatial) ? "Spatial" : "PerActivation";
        const char* layout_str = (layout == miopenTensorNCHW) ? "NCHW" : "NHWC";

        EXPECT_EQ(status, miopenStatusBadParm)
            << "Backward should reject " << mode_str << " mode with " << layout_str
            << " layout: N=" << config.N << " H=" << config.H << " W=" << config.W
            << " (N*H*W=" << (config.N * config.H * config.W) << ")";
    }
}

TEST_P(GPU_BNInputValidation3D_FP32, RejectsInvalidInput)
{
    auto [config, layout, bn_mode] = GetParam();
    auto&& handle                  = get_handle();
    float alpha = 1.0f, beta = 0.0f;

    if(config.Direction == miopen::batchnorm::Direction::ForwardTraining)
    {
        BNFwdTrainTestData<float, float, float, float, float, double, BN3DTestCase> data;
        data.SetUpImpl(config, bn_mode, layout);

        auto status = miopenBatchNormalizationForwardTraining(&handle,
                                                              bn_mode,
                                                              &alpha,
                                                              &beta,
                                                              &data.input.desc,
                                                              data.in_dev.get(),
                                                              &data.output.desc,
                                                              data.out_dev.get(),
                                                              &data.scale.desc,
                                                              data.scale_dev.get(),
                                                              data.shift_dev.get(),
                                                              data.averageFactor,
                                                              data.runMean_dev.get(),
                                                              data.runVariance_dev.get(),
                                                              data.epsilon,
                                                              data.saveMean_dev.get(),
                                                              data.saveVariance_dev.get());

        const char* mode_str   = (bn_mode == miopenBNSpatial) ? "Spatial" : "PerActivation";
        const char* layout_str = (layout == miopenTensorNCDHW) ? "NCDHW" : "NDHWC";

        EXPECT_EQ(status, miopenStatusBadParm)
            << "ForwardTraining 3D should reject " << mode_str << " mode with " << layout_str
            << " layout: N=" << config.N << " D=" << config.D << " H=" << config.H
            << " W=" << config.W << " (N*D*H*W=" << (config.N * config.D * config.H * config.W)
            << ")";
    }
    else if(config.Direction == miopen::batchnorm::Direction::Backward)
    {
        BNBwdTestData<float, float, float, float, float, float, double, BN3DTestCase> data;
        data.SetUpImpl(config, bn_mode, layout);

        auto status = miopenBatchNormalizationBackward(&handle,
                                                       bn_mode,
                                                       &alpha,
                                                       &beta,
                                                       &alpha,
                                                       &beta,
                                                       &data.input.desc,
                                                       data.in_dev.get(),
                                                       &data.dy.desc,
                                                       data.dy_dev.get(),
                                                       &data.output.desc,
                                                       data.out_dev.get(),
                                                       &data.bnScale.desc,
                                                       data.bnScale_dev.get(),
                                                       data.dScale_dev.get(),
                                                       data.dBias_dev.get(),
                                                       data.epsilon,
                                                       data.savedMean_dev.get(),
                                                       data.savedInvVar_dev.get());

        const char* mode_str   = (bn_mode == miopenBNSpatial) ? "Spatial" : "PerActivation";
        const char* layout_str = (layout == miopenTensorNCDHW) ? "NCDHW" : "NDHWC";

        EXPECT_EQ(status, miopenStatusBadParm)
            << "Backward 3D should reject " << mode_str << " mode with " << layout_str
            << " layout: N=" << config.N << " D=" << config.D << " H=" << config.H
            << " W=" << config.W << " (N*D*H*W=" << (config.N * config.D * config.H * config.W)
            << ")";
    }
}

// Instantiate tests for 2D with all combinations of layouts and modes
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_BNInputValidation2D_FP32,
    testing::Combine(testing::ValuesIn(Network2DInvalidTraining<BN2DTestCase>()),
                     testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                     testing::ValuesIn({miopenBNSpatial, miopenBNPerActivation})));

// Instantiate tests for 3D with all combinations of layouts and modes
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_BNInputValidation3D_FP32,
    testing::Combine(testing::ValuesIn(Network3DInvalidTraining<BN3DTestCase>()),
                     testing::ValuesIn({miopenTensorNCDHW, miopenTensorNDHWC}),
                     testing::ValuesIn({miopenBNSpatial, miopenBNPerActivation})));
