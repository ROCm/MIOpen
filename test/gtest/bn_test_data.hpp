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

#include <miopen/miopen.h>
#include <miopen/solver_id.hpp>
#include <miopen/batchnorm/problem_description.hpp>
#include <serialize.hpp>
#include <fusionHost.hpp>

#include "tensor_util.hpp"
#include "get_handle.hpp"

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
    std::vector<size_t> GetInput() const { return {N, C, H, W}; }
};

std::vector<BNTestCase> Network1()
{
    // pyt_mlperf_resnet50v1.5
    return {
        {16, 8, 128, 256, miopenBNSpatial, miopen::batchnorm::Direction::ForwardInference, 1, 0},
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

template <typename T, typename TConfig>
struct BNTestData
{
    void SetUpImpl(const TConfig& config, miopenTensorLayout_t t_layout)
    {
        bn_config     = config;
        tensor_layout = t_layout;
        CreateTensors();
        InitTensorsWithRandValue();
        SetDirection();
        SetBNMode();
        WriteToGPU();
    }
    const miopen::TensorDescriptor& GetInputDesc() const { return input.desc; }

    tensor<T> input;
    tensor<T> output;
    tensor<T> ref_out;
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr out_dev;

    miopen::FusionPlanDescriptor fusePlanDesc;
    miopen::OperatorArgs params;
    miopen::TensorDescriptor bn_desc;
    miopenBatchNormMode_t bn_mode;
    miopen::batchnorm::Direction direction;
    miopenTensorLayout_t tensor_layout;
    TConfig bn_config;

private:
    void CreateTensors()
    {
        input   = tensor<T>{miopen_type<T>{}, tensor_layout, bn_config.GetInput()};
        output  = tensor<T>{miopen_type<T>{}, tensor_layout, bn_config.GetInput()};
        ref_out = output;
    }

    void InitTensorsWithRandValue()
    {
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::uniform_int_distribution<> d{0, 100};
        auto gen_value = [&](auto...) {
            return 1e-2 * static_cast<T>(d(gen)) * ((d(gen) % 2 == 1) ? -1 : 1);
        };
        input.generate(gen_value);
    }

    void SetDirection() { direction = bn_config.Direction; }
    void SetBNMode() { bn_mode = bn_config.mode; }
    void WriteToGPU()
    {
        auto&& handle = get_handle();
        in_dev        = handle.Write(input.data);
        out_dev       = handle.Write(output.data);
    }
};

template <typename T, typename TConfig>
struct BNInferTestData : public BNTestData<T, TConfig>
{
    void SetUpImpl(const TConfig& config, miopenTensorLayout_t t_layout)
    {
        BNTestData<T, TConfig>::SetUpImpl(config, t_layout);
        CreateTensors();
        InitTensorsWithRandValue();
        WriteToGPU();
    }

    tensor<T> scale;
    tensor<T> shift;
    tensor<T> estMean;
    tensor<T> estVariance;
    miopen::Allocator::ManageDataPtr scale_dev;
    miopen::Allocator::ManageDataPtr shift_dev; // bias
    miopen::Allocator::ManageDataPtr estMean_dev;
    miopen::Allocator::ManageDataPtr estVariance_dev;
    double epsilon          = 1.0e-5;
    float alpha       = static_cast<float>(1.0f);
    float beta        = static_cast<float>(0);
    const float activ_alpha = static_cast<double>(0.5f);
    const float activ_beta  = static_cast<double>(0.5f);
    const float activ_gamma = static_cast<double>(0.5f);

private:
    void CreateTensors()
    {
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc,
                                         BNTestData<T, TConfig>::input.desc,
                                         BNTestData<T, TConfig>::bn_mode);
        scale       = tensor<T>{miopen_type<T>{},
                          BNTestData<T, TConfig>::tensor_layout,
                          derivedBnDesc.GetLengths()};
        shift       = tensor<T>{miopen_type<T>{},
                          BNTestData<T, TConfig>::tensor_layout,
                          derivedBnDesc.GetLengths()};
        estMean     = tensor<T>{miopen_type<T>{},
                            BNTestData<T, TConfig>::tensor_layout,
                            derivedBnDesc.GetLengths()};
        estVariance = tensor<T>{miopen_type<T>{},
                                BNTestData<T, TConfig>::tensor_layout,
                                derivedBnDesc.GetLengths()};
    }

    void InitTensorsWithRandValue()
    {
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::uniform_int_distribution<> d{0, 100};
        auto gen_value = [&](auto...) {
            return 1e-2 * static_cast<T>(d(gen)) * ((d(gen) % 2 == 1) ? -1 : 1);
        };
        scale.generate(gen_value);
        shift.generate(gen_value);
        estMean.generate(gen_value);
        auto gen_var = [&](auto...) { return 1e-2 * (static_cast<T>(d(gen)) + 1); };
        estVariance.generate(gen_var);
    }
    void WriteToGPU()
    {
        auto&& handle   = get_handle();
        scale_dev       = handle.Write(scale.data);
        shift_dev       = handle.Write(shift.data);
        estMean_dev     = handle.Write(estMean.data);
        estVariance_dev = handle.Write(estVariance.data);
    }
};
