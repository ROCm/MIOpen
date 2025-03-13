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

#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/solver_id.hpp>
#include <serialize.hpp>
#include <fusionHost.hpp>

#include "tensor_util.hpp"
#include "get_handle.hpp"
#include "random.hpp"
struct BNTestCase
{
    size_t N;
    size_t C;
    size_t H;
    size_t W;
    miopenBatchNormMode_t mode;
    miopen::batchnorm::Direction Direction;
    bool save;
    bool keepRunning;

    friend std::ostream& operator<<(std::ostream& ss, const BNTestCase& tc)
    {
        return ss << "(N: " << tc.N << " C:" << tc.C << " H:" << tc.H << " W:" << tc.W
                  << " mode: " << tc.mode << " Direction: " << static_cast<int>(tc.Direction)
                  << " save: " << tc.save << " keepRunning: " << tc.keepRunning;
    }
    std::vector<size_t> GetInput() { return {N, C, H, W}; }
};

inline std::vector<BNTestCase> Networkna1()
{
    // pyt_mlperf_resnet50v1.5
    return {
        {64, 128, 56, 56, miopenBNSpatial, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 2048, 7, 7, miopenBNSpatial, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 2048, 7, 7, miopenBNSpatial, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 2048, 7, 7, miopenBNSpatial, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 256, 14, 14, miopenBNSpatial, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 256, 14, 14, miopenBNSpatial, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 256, 14, 14, miopenBNSpatial, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 256, 28, 28, miopenBNSpatial, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 256, 28, 28, miopenBNSpatial, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 256, 28, 28, miopenBNSpatial, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 256, 56, 56, miopenBNSpatial, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 256, 56, 56, miopenBNSpatial, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 256, 56, 56, miopenBNSpatial, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 512, 14, 14, miopenBNSpatial, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 512, 14, 14, miopenBNSpatial, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 512, 14, 14, miopenBNSpatial, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 512, 28, 28, miopenBNSpatial, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 512, 28, 28, miopenBNSpatial, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 512, 28, 28, miopenBNSpatial, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 512, 7, 7, miopenBNSpatial, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 512, 7, 7, miopenBNSpatial, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 512, 7, 7, miopenBNSpatial, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 64, 112, 112, miopenBNSpatial, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 64, 112, 112, miopenBNSpatial, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 64, 112, 112, miopenBNSpatial, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 64, 56, 56, miopenBNSpatial, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 64, 56, 56, miopenBNSpatial, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 64, 56, 56, miopenBNSpatial, miopen::batchnorm::Direction::ForwardInference, 1, 0}};
}

template <typename T>
struct BNActivInferTest
    : public ::testing::TestWithParam<std::tuple<miopenActivationMode_t, BNTestCase>>
{
protected:
    void SetUp() override
    {
        test_skipped                    = false;
        std::tie(activ_mode, bn_config) = GetParam();
        bn_mode                         = bn_config.mode;
        input                           = tensor<T>{bn_config.GetInput()};
        auto derivedBnDesc              = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, bn_mode);
        scale       = tensor<T>{derivedBnDesc.GetLengths()};
        shift       = tensor<T>{derivedBnDesc.GetLengths()};
        estMean     = tensor<T>{derivedBnDesc.GetLengths()};
        estVariance = tensor<T>{derivedBnDesc.GetLengths()};

        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        input.generate(gen_value);
        scale.generate(gen_value);
        shift.generate(gen_value);
        estMean.generate(gen_value);
        auto gen_var = [](auto...) { return static_cast<T>(1e-2 * (prng::gen_0_to_B(100) + 1)); };
        estVariance.generate(gen_var);
        activ_desc    = {activ_mode, activ_alpha, activ_beta, activ_gamma};
        output        = tensor<T>{bn_config.GetInput()};
        auto&& handle = get_handle();
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());
        in_dev          = handle.Write(input.data);
        scale_dev       = handle.Write(scale.data);
        shift_dev       = handle.Write(shift.data);
        estMean_dev     = handle.Write(estMean.data);
        estVariance_dev = handle.Write(estVariance.data);
        out_dev         = handle.Write(output.data);

        // Setup the Fusionplan
        fusePlanDesc = miopen::FusionPlanDescriptor(miopenVerticalFusion, input.desc);
        auto bnOp =
            std::make_shared<miopen::BatchNormInferenceFusionOpDescriptor>(bn_mode, bn_desc);
        EXPECT_EQ(fusePlanDesc.AddOp(bnOp), miopenStatusSuccess);
        bnOp->SetArgs(params,
                      &alpha,
                      &beta,
                      scale_dev.get(),
                      shift_dev.get(),
                      estMean_dev.get(),
                      estVariance_dev.get(),
                      epsilon);
        auto activOp = std::make_shared<miopen::ActivFwdFusionOpDescriptor>(activ_desc.GetMode());
        EXPECT_EQ(fusePlanDesc.AddOp(activOp), miopenStatusSuccess);
        activOp->SetArgs(params, &alpha, &beta, activ_alpha, activ_beta, activ_gamma);
    }

    void TearDown() override
    {
        if(test_skipped)
            return;
        ref_out = tensor<T>{bn_config.GetInput()};
        if(bn_mode == miopenBNPerActivation)
        {
            batchNormPerActivHostInference(
                input, ref_out, scale, shift, epsilon, estMean, estVariance);
        }
        else
        {
            batchNormSpatialHostInference(
                input, ref_out, scale, shift, epsilon, estMean, estVariance);
        }
        activationHostInfer(
            activ_mode, activ_gamma, activ_beta, activ_alpha, ref_out.data, ref_out.data);
        auto&& handle = get_handle();
        output.data   = handle.Read<T>(out_dev, output.data.size());
        EXPECT_FALSE(miopen::range_zero(ref_out)) << "CPU data is all zeros";
        EXPECT_FALSE(miopen::range_zero(output)) << "GPU data is all zeros";
        EXPECT_FALSE(miopen::find_idx(output, miopen::not_finite) >= 0)
            << "Non finite number found in the GPU data";
        EXPECT_TRUE(miopen::range_distance(ref_out) == miopen::range_distance(output));
        const double tolerance = 80;
        double threshold       = std::numeric_limits<T>::epsilon() * tolerance;
        auto error             = miopen::rms_range(ref_out, output);
        EXPECT_FALSE(miopen::find_idx(ref_out, miopen::not_finite) >= 0)
            << "Non finite number found in the CPU data";
        EXPECT_TRUE(error < threshold)
            << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;
    }
    BNTestCase bn_config;
    miopen::TensorDescriptor bn_desc;
    miopen::ActivationDescriptor activ_desc;
    miopenBatchNormMode_t bn_mode;
    tensor<T> input;
    tensor<T> output;
    tensor<T> ref_out;
    tensor<T> scale;
    tensor<T> shift;
    tensor<T> estMean;
    tensor<T> estVariance;
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr out_dev;
    miopen::Allocator::ManageDataPtr scale_dev;
    miopen::Allocator::ManageDataPtr shift_dev;
    miopen::Allocator::ManageDataPtr estMean_dev;
    miopen::Allocator::ManageDataPtr estVariance_dev;
    bool test_skipped = false;
    miopenActivationMode_t activ_mode;
    miopen::FusionPlanDescriptor fusePlanDesc;
    miopen::OperatorArgs params;
    const float alpha       = static_cast<float>(1.0f);
    const float beta        = static_cast<float>(0);
    const float activ_alpha = static_cast<double>(0.5f);
    const float activ_beta  = static_cast<double>(0.5f);
    const float activ_gamma = static_cast<double>(0.5f);
    double epsilon          = 1.0e-5;
};
