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
#include "random.hpp"

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

template <typename T>
std::vector<T> NetworkSmall();

template <typename T>
std::vector<T> NetworkLarge();

template <>
inline std::vector<BNTestCase> NetworkLarge()
{
    // pyt_mlperf_resnet50v1.5
    return {
        {192, 1, 8, 8, miopenBNSpatial, miopen::batchnorm::Direction::Backward, 1, 0},
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

template <>
inline std::vector<BNTestCase> NetworkSmall()
{
    // pyt_mlperf_resnet50v1.5
    return {
        {192, 2, 8, 8, miopenBNSpatial, miopen::batchnorm::Direction::Backward, 1, 0},
        {16, 8, 132, 28, miopenBNSpatial, miopen::batchnorm::Direction::Backward, 1, 0},
        {16, 8, 128, 256, miopenBNSpatial, miopen::batchnorm::Direction::ForwardTraining, 1, 0},
        {64, 2048, 17, 17, miopenBNSpatial, miopen::batchnorm::Direction::Backward, 0, 1},

    };
}

template <typename XDataType, typename YDataType, typename TConfig>
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

    tensor<XDataType> input;
    tensor<YDataType> output;
    tensor<YDataType> ref_out;
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
        input   = tensor<XDataType>{tensor_layout, bn_config.GetInput()};
        output  = tensor<YDataType>{tensor_layout, bn_config.GetInput()};
        ref_out = tensor<YDataType>{tensor_layout, bn_config.GetInput()};
    }

    void InitTensorsWithRandValue()
    {
        input.generate(
            [](auto...) { return prng::gen_descreet_uniform_sign<XDataType>(1e-2, 100); });
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

template <typename XDataType,
          typename YDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType,
          typename TConfig>
struct BNInferTestData : public BNTestData<XDataType, YDataType, TConfig>
{
    void SetUpImpl(const TConfig& config, miopenTensorLayout_t t_layout)
    {
        BNTestData<XDataType, YDataType, TConfig>::SetUpImpl(config, t_layout);
        CreateTensors();
        InitTensorsWithRandValue();
        WriteToGPU();
    }

    tensor<ScaleDataType> scale;
    tensor<BiasDataType> shift;
    tensor<MeanVarDataType> estMean;
    tensor<MeanVarDataType> estVariance;
    miopen::Allocator::ManageDataPtr scale_dev;
    miopen::Allocator::ManageDataPtr shift_dev;
    miopen::Allocator::ManageDataPtr estMean_dev;
    miopen::Allocator::ManageDataPtr estVariance_dev;
    double epsilon          = 1.0e-5;
    float alpha             = static_cast<float>(1.0f);
    float beta              = static_cast<float>(0);
    const float activ_alpha = static_cast<double>(0.5f);
    const float activ_beta  = static_cast<double>(0.5f);
    const float activ_gamma = static_cast<double>(0.5f);

private:
    void CreateTensors()
    {
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc,
                                         BNTestData<XDataType, YDataType, TConfig>::input.desc,
                                         BNTestData<XDataType, YDataType, TConfig>::bn_mode);
        scale   = tensor<ScaleDataType>{BNTestData<XDataType, YDataType, TConfig>::tensor_layout,
                                      derivedBnDesc.GetLengths()};
        shift   = tensor<BiasDataType>{BNTestData<XDataType, YDataType, TConfig>::tensor_layout,
                                     derivedBnDesc.GetLengths()};
        estMean = tensor<MeanVarDataType>{BNTestData<XDataType, YDataType, TConfig>::tensor_layout,
                                          derivedBnDesc.GetLengths()};
        estVariance = tensor<MeanVarDataType>{
            BNTestData<XDataType, YDataType, TConfig>::tensor_layout, derivedBnDesc.GetLengths()};
    }

    void InitTensorsWithRandValue()
    {
        auto gen_value = [](auto...) {
            return prng::gen_descreet_uniform_sign<ScaleDataType>(1e-2, 100);
        };
        scale.generate(gen_value);
        shift.generate(gen_value);
        estMean.generate(gen_value);

        auto gen_var = [](auto...) {
            return static_cast<MeanVarDataType>(1e-2 * (prng::gen_0_to_B(100) + 1));
        };
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

template <typename XDataType,
          typename DxDataType,
          typename DyDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename DscaleDbiasDataType,
          typename MeanVarDataType,
          typename TConfig>
struct BNBwdTestData : public BNTestData<XDataType, DyDataType, TConfig>
{
    void SetUpImpl(const TConfig& config, miopenTensorLayout_t t_layout)
    {
        BNTestData<XDataType, DxDataType, TConfig>::SetUpImpl(config, t_layout);
        CreateTensors();
        InitTensorsWithRandValue();
        WriteToGPU();
    }

    tensor<ScaleDataType> bnScale;

    tensor<MeanVarDataType> savedMean;
    tensor<MeanVarDataType> savedInvVar;

    tensor<DyDataType> dy;
    tensor<DscaleDbiasDataType> dScale;
    tensor<DscaleDbiasDataType> dBias;
    tensor<DscaleDbiasDataType> dScale_ref;
    tensor<DscaleDbiasDataType> dBias_ref;

    miopen::Allocator::ManageDataPtr bnScale_dev;
    miopen::Allocator::ManageDataPtr savedMean_dev;
    miopen::Allocator::ManageDataPtr savedInvVar_dev;

    miopen::Allocator::ManageDataPtr dy_dev;
    miopen::Allocator::ManageDataPtr dScale_dev;
    miopen::Allocator::ManageDataPtr dBias_dev;
    miopen::Allocator::ManageDataPtr dScale_ref_dev;
    miopen::Allocator::ManageDataPtr dBias_ref_dev;
    double epsilon = std::numeric_limits<float>::epsilon();

    float alphaDataDiff = static_cast<float>(1), betaDataDiff = static_cast<float>(0);
    float alphaParamDiff = static_cast<float>(1), betaParamDiff = static_cast<float>(0);

private:
    void CreateTensors()
    {
        dy = tensor<DyDataType>{BNTestData<XDataType, DyDataType, TConfig>::tensor_layout,
                                BNTestData<XDataType, DyDataType, TConfig>::bn_config.GetInput()};

        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc,
                                         BNTestData<XDataType, DyDataType, TConfig>::input.desc,
                                         BNTestData<XDataType, DyDataType, TConfig>::bn_mode);
        bnScale   = tensor<ScaleDataType>{BNTestData<XDataType, DyDataType, TConfig>::tensor_layout,
                                        derivedBnDesc.GetLengths()};
        savedMean = tensor<MeanVarDataType>{
            BNTestData<XDataType, DyDataType, TConfig>::tensor_layout, derivedBnDesc.GetLengths()};
        savedInvVar = tensor<MeanVarDataType>{
            BNTestData<XDataType, DyDataType, TConfig>::tensor_layout, derivedBnDesc.GetLengths()};
        dScale = tensor<DscaleDbiasDataType>{
            BNTestData<XDataType, DyDataType, TConfig>::tensor_layout, derivedBnDesc.GetLengths()};
        dBias = tensor<DscaleDbiasDataType>{
            BNTestData<XDataType, DyDataType, TConfig>::tensor_layout, derivedBnDesc.GetLengths()};
        dScale_ref = dScale;
        dBias_ref  = dBias;
    }

    void InitTensorsWithRandValue()
    {
        auto gen_value = [](auto...) {
            return prng::gen_descreet_uniform_sign<ScaleDataType>(1e-2, 100);
        };
        dy.generate(gen_value);
        bnScale.generate(gen_value);
        savedMean.generate(gen_value);

        auto gen_var = [](auto...) {
            return static_cast<MeanVarDataType>(1e-2 * (prng::gen_0_to_B(100) + 1));
        };
        savedInvVar.generate(gen_var);

        std::fill(dScale.begin(), dScale.end(), 0.);
        std::fill(dBias.begin(), dBias.end(), 0.);

        std::fill(dScale_ref.begin(), dScale_ref.end(), 0.);
        std::fill(dBias_ref.begin(), dBias_ref.end(), 0.);
    }
    void WriteToGPU()
    {
        auto&& handle = get_handle();

        bnScale_dev     = handle.Write(bnScale.data);
        savedMean_dev   = handle.Write(savedMean.data);
        savedInvVar_dev = handle.Write(savedInvVar.data);
        dy_dev          = handle.Write(dy.data);

        dScale_dev = handle.Write(dScale.data);
        dBias_dev  = handle.Write(dBias.data);
    }
};

template <typename XDataType,
          typename YDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename AccDataType,
          typename TConfig>
struct BNFwdTrainTestData : public BNTestData<XDataType, YDataType, TConfig>
{
    void SetUpImpl(const TConfig& config, miopenTensorLayout_t t_layout)
    {
        BNTestData<XDataType, YDataType, TConfig>::SetUpImpl(config, t_layout);
        CreateTensors();
        InitTensorsWithRandValue();
        WriteToGPU();
    }

    tensor<ScaleDataType> scale;
    tensor<BiasDataType> shift;
    tensor<AccDataType> saveMean;
    tensor<AccDataType> saveVariance;
    tensor<AccDataType> runMean;
    tensor<AccDataType> runVariance;

    tensor<AccDataType> saveMean_ref;
    tensor<AccDataType> saveVariance_ref;
    tensor<AccDataType> runMean_ref;
    tensor<AccDataType> runVariance_ref;

    miopen::Allocator::ManageDataPtr scale_dev;
    miopen::Allocator::ManageDataPtr shift_dev; // bias
    miopen::Allocator::ManageDataPtr saveMean_dev;
    miopen::Allocator::ManageDataPtr saveVariance_dev;
    miopen::Allocator::ManageDataPtr runMean_dev;
    miopen::Allocator::ManageDataPtr runVariance_dev;
    double epsilon          = 1.0e-5;
    double averageFactor    = 0.1;
    float alpha             = static_cast<float>(1.0f);
    float beta              = static_cast<float>(0);
    const float activ_alpha = static_cast<double>(0.5f);
    const float activ_beta  = static_cast<double>(0.5f);
    const float activ_gamma = static_cast<double>(0.5f);

private:
    void CreateTensors()
    {
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc,
                                         BNTestData<XDataType, YDataType, TConfig>::input.desc,
                                         BNTestData<XDataType, YDataType, TConfig>::bn_mode);
        scale    = tensor<ScaleDataType>{BNTestData<XDataType, YDataType, TConfig>::tensor_layout,
                                      derivedBnDesc.GetLengths()};
        shift    = tensor<BiasDataType>{BNTestData<XDataType, YDataType, TConfig>::tensor_layout,
                                     derivedBnDesc.GetLengths()};
        saveMean = tensor<AccDataType>{BNTestData<XDataType, YDataType, TConfig>::tensor_layout,
                                       derivedBnDesc.GetLengths()};
        saveVariance = tensor<AccDataType>{BNTestData<XDataType, YDataType, TConfig>::tensor_layout,
                                           derivedBnDesc.GetLengths()};
        runMean      = tensor<AccDataType>{BNTestData<XDataType, YDataType, TConfig>::tensor_layout,
                                      derivedBnDesc.GetLengths()};
        runVariance  = tensor<AccDataType>{BNTestData<XDataType, YDataType, TConfig>::tensor_layout,
                                          derivedBnDesc.GetLengths()};
    }

    void InitTensorsWithRandValue()
    {
        auto gen_value = [](auto...) {
            return prng::gen_descreet_uniform_sign<ScaleDataType>(1e-2, 100);
        };
        scale.generate(gen_value);
        shift.generate(gen_value);

        auto gen_var = [](auto...) {
            return static_cast<AccDataType>(1e-2 * (prng::gen_0_to_B(100) + 1));
        };
        runMean.generate(gen_var);
        runVariance.generate(gen_var);

        saveMean_ref     = saveMean;
        saveVariance_ref = saveVariance;
        runMean_ref      = runMean;
        runVariance_ref  = runVariance;
    }
    void WriteToGPU()
    {
        auto&& handle    = get_handle();
        scale_dev        = handle.Write(scale.data);
        shift_dev        = handle.Write(shift.data);
        saveMean_dev     = handle.Write(saveMean.data);
        saveVariance_dev = handle.Write(saveVariance.data);
        runMean_dev      = handle.Write(runMean.data);
        runVariance_dev  = handle.Write(runVariance.data);
    }
};
