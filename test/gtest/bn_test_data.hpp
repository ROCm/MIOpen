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

struct BN2DTestCase
{
    size_t N;
    size_t C;
    size_t H;
    size_t W;
    miopen::batchnorm::Direction Direction;
    bool save;
    bool keepRunning;

    friend std::ostream& operator<<(std::ostream& ss, const BN2DTestCase& tc)
    {
        return ss << "(N: " << tc.N << " C:" << tc.C << " H:" << tc.H << " W:" << tc.W
                  << " Direction: " << static_cast<int>(tc.Direction) << " save: " << tc.save
                  << " keepRunning: " << tc.keepRunning;
    }
    std::vector<size_t> GetInput() const { return {N, C, H, W}; }
};

struct BN3DTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    miopen::batchnorm::Direction Direction;
    bool save;
    bool keepRunning;

    friend std::ostream& operator<<(std::ostream& ss, const BN3DTestCase& tc)
    {
        return ss << "(N: " << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << " Direction: " << static_cast<int>(tc.Direction)
                  << " save: " << tc.save << " keepRunning: " << tc.keepRunning;
    }
    std::vector<size_t> GetInput() const { return {N, C, D, H, W}; }
};

template <typename T>
std::vector<T> Network2DSmall();

template <typename T>
std::vector<T> Network2DLarge();

template <typename T>
std::vector<T> Network3DBN();

template <typename T>
std::vector<T> Network3DSerialCase();

template <>
inline std::vector<BN2DTestCase> Network2DLarge()
{
    // pyt_mlperf_resnet50v1.5
    // clang-format off
    return {
        {64, 1, 1024, 1024, miopen::batchnorm::Direction::Backward, 1, 0},
        {192, 1, 8, 8, miopen::batchnorm::Direction::Backward, 1, 0},
        {12, 40, 122, 122, miopen::batchnorm::Direction::Backward, 1, 0},
        {64, 256, 14, 14, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 256, 14, 14, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 256, 14, 14, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 256, 28, 28, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 256, 28, 28, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 256, 28, 28, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 256, 56, 56, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 256, 56, 56, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 256, 56, 56, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 512, 14, 14, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 512, 14, 14, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 512, 14, 14, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 512, 28, 28, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 512, 28, 28, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 512, 28, 28, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 512, 7, 7, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 512, 7, 7, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 512, 7, 7, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 64, 112, 112, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 64, 112, 112, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 64, 112, 112, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 64, 56, 56, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 64, 56, 56, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 64, 56, 56, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 2048, 7, 7, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 2048, 17, 17, miopen::batchnorm::Direction::Backward, 0, 1},
        {128, 256, 14, 14, miopen::batchnorm::Direction::Backward, 0, 1},
        {128, 256, 16, 16, miopen::batchnorm::Direction::Backward, 0, 1},
        {670, 1, 224, 224, miopen::batchnorm::Direction::Backward, 0, 1},
        {768, 1, 14, 14, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {768, 1, 23, 23, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {832, 1, 14, 14, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {832, 1, 28, 28, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        // {1, 512, 7, 7, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        // {1, 512, 7, 7, miopen::batchnorm::Direction::Backward, 1, 1},
        // edge cases
        {69328, 1, 22, 22, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {69328, 1, 13, 79, miopen::batchnorm::Direction::ForwardTraining, 1, 1}
        };
    // clang-format on
}

// These are very large tensors which caused memory insufficient error
// when ran parallely by ctest. Hence, these are run serially.
// Shape: (2, 2048, 16, 128, 128) --> Size: 1.07e+09
// For now any test case with tensor size greater then 1e09 need to be run serially.
template <>
inline std::vector<BN3DTestCase> Network3DSerialCase()
{
    return {{2, 2048, 16, 128, 128, miopen::batchnorm::Direction::Backward, 0, 1}};
}

template <>
inline std::vector<BN2DTestCase> Network2DSmall()
{
    // pyt_mlperf_resnet50v1.5
    // clang-format off
    return {
        {12, 40, 122, 122, miopen::batchnorm::Direction::Backward, 1, 0},
        {16, 8, 132, 28, miopen::batchnorm::Direction::Backward, 1, 0},
        {192, 2, 8, 8, miopen::batchnorm::Direction::Backward, 1, 0},
        {16, 8, 56, 56, miopen::batchnorm::Direction::Backward, 1, 0},
        {16, 8, 128, 256, miopen::batchnorm::Direction::ForwardTraining, 1, 0},
    };
    // clang-format on
}

template <>
inline std::vector<BN3DTestCase> Network3DBN()
{
    // clang-format off
    return {
        {2, 2, 3, 224, 224, miopen::batchnorm::Direction::Backward, 1, 0},
        {16, 8, 132, 28, 28, miopen::batchnorm::Direction::Backward, 1, 0},
        {16, 8, 16, 128, 128, miopen::batchnorm::Direction::ForwardTraining, 1, 0}
    };
    // clang-format on
}

template <typename XDataType, typename YDataType, typename AccDataType, typename TConfig>
struct BNTestData
{
    void
    SetUpImpl(const TConfig& config, miopenBatchNormMode_t t_bnmode, miopenTensorLayout_t t_layout)
    {
        bn_config     = config;
        tensor_layout = t_layout;
        bn_mode       = t_bnmode;
        CreateTensors();
        InitTensorsWithRandValue();
        SetDirection();
        WriteToGPU();
    }
    const miopen::TensorDescriptor& GetInputDesc() const { return input.desc; }

    tensor<XDataType> input;
    tensor<YDataType> output;
    tensor<AccDataType> out_ref;
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
        out_ref = tensor<AccDataType>{tensor_layout, bn_config.GetInput()};
    }

    void InitTensorsWithRandValue()
    {
        // -2.0 to 2.0
        input.generate(uniform_signed_initializer<XDataType>(2e-3 /*scale*/, 1000 /*range*/));
    }

    void SetDirection() { direction = bn_config.Direction; }
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
          typename AccDataType,
          typename TConfig>
struct BNInferTestData : public BNTestData<XDataType, YDataType, AccDataType, TConfig>
{
    void
    SetUpImpl(const TConfig& config, miopenBatchNormMode_t t_bnmode, miopenTensorLayout_t t_layout)
    {
        BNTestData<XDataType, YDataType, AccDataType, TConfig>::SetUpImpl(
            config, t_bnmode, t_layout);
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
    double epsilon = 1.0e-5;
    float alpha    = static_cast<float>(1.0f);
    float beta     = static_cast<float>(0);
    double activ_alpha;
    double activ_beta;
    miopenActivationMode_t activ_mode;

private:
    void CreateTensors()
    {
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(
            derivedBnDesc,
            BNTestData<XDataType, YDataType, AccDataType, TConfig>::input.desc,
            BNTestData<XDataType, YDataType, AccDataType, TConfig>::bn_mode);
        scale = tensor<ScaleDataType>{
            BNTestData<XDataType, YDataType, AccDataType, TConfig>::tensor_layout,
            derivedBnDesc.GetLengths()};
        shift = tensor<BiasDataType>{
            BNTestData<XDataType, YDataType, AccDataType, TConfig>::tensor_layout,
            derivedBnDesc.GetLengths()};
        estMean = tensor<MeanVarDataType>{
            BNTestData<XDataType, YDataType, AccDataType, TConfig>::tensor_layout,
            derivedBnDesc.GetLengths()};
        estVariance = tensor<MeanVarDataType>{
            BNTestData<XDataType, YDataType, AccDataType, TConfig>::tensor_layout,
            derivedBnDesc.GetLengths()};
    }

    void InitTensorsWithRandValue()
    {
        // -2.0 to 2.0
        scale.generate(uniform_signed_initializer<ScaleDataType>(2e-3 /*scale*/, 1000 /*range*/));
        shift.generate(uniform_signed_initializer<BiasDataType>(2e-3 /*scale*/, 1000 /*range*/));
        estMean.generate(
            uniform_signed_initializer<MeanVarDataType>(2e-3 /*scale*/, 1000 /*range*/));
        // estVaraince has to be +ve number otherwise 1/sqrt(-ve) would
        // give img number
        estVariance.generate(
            uniform_unsigned_initializer<MeanVarDataType>(2e-3 /*scale*/, 1000 /*range*/));
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
          typename ScaleDataType,
          typename DscaleDbiasDataType,
          typename MeanVarDataType,
          typename AccDataType,
          typename TConfig>
struct BNBwdTestData : public BNTestData<XDataType, DyDataType, AccDataType, TConfig>
{
    void
    SetUpImpl(const TConfig& config, miopenBatchNormMode_t t_bnmode, miopenTensorLayout_t t_layout)
    {
        BNTestData<XDataType, DxDataType, AccDataType, TConfig>::SetUpImpl(
            config, t_bnmode, t_layout);
        CreateTensors();
        InitTensorsWithRandValue();
        WriteToGPU();
    }

    tensor<ScaleDataType> bnScale;
    tensor<ScaleDataType> bnBias;

    tensor<MeanVarDataType> savedMean;
    tensor<MeanVarDataType> savedInvVar;

    tensor<DyDataType> dy;
    tensor<DscaleDbiasDataType> dScale;
    tensor<DscaleDbiasDataType> dBias;
    tensor<AccDataType> dScale_ref;
    tensor<AccDataType> dBias_ref;

    miopen::Allocator::ManageDataPtr bnScale_dev;
    miopen::Allocator::ManageDataPtr bnBias_dev;
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

    double activ_alpha;
    double activ_beta;
    miopenActivationMode_t activ_mode;

private:
    void CreateTensors()
    {
        dy = tensor<DyDataType>{
            BNTestData<XDataType, DyDataType, AccDataType, TConfig>::tensor_layout,
            BNTestData<XDataType, DyDataType, AccDataType, TConfig>::bn_config.GetInput()};

        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(
            derivedBnDesc,
            BNTestData<XDataType, DyDataType, AccDataType, TConfig>::input.desc,
            BNTestData<XDataType, DyDataType, AccDataType, TConfig>::bn_mode);
        bnScale = tensor<ScaleDataType>{
            BNTestData<XDataType, DyDataType, AccDataType, TConfig>::tensor_layout,
            derivedBnDesc.GetLengths()};
        bnBias = tensor<ScaleDataType>{
            BNTestData<XDataType, DyDataType, AccDataType, TConfig>::tensor_layout,
            derivedBnDesc.GetLengths()};
        savedMean = tensor<MeanVarDataType>{
            BNTestData<XDataType, DyDataType, AccDataType, TConfig>::tensor_layout,
            derivedBnDesc.GetLengths()};
        savedInvVar = tensor<MeanVarDataType>{
            BNTestData<XDataType, DyDataType, AccDataType, TConfig>::tensor_layout,
            derivedBnDesc.GetLengths()};
        dScale = tensor<DscaleDbiasDataType>{
            BNTestData<XDataType, DyDataType, AccDataType, TConfig>::tensor_layout,
            derivedBnDesc.GetLengths()};
        dBias = tensor<DscaleDbiasDataType>{
            BNTestData<XDataType, DyDataType, AccDataType, TConfig>::tensor_layout,
            derivedBnDesc.GetLengths()};
        dScale_ref = tensor<AccDataType>{
            BNTestData<XDataType, DyDataType, AccDataType, TConfig>::tensor_layout,
            derivedBnDesc.GetLengths()};
        dBias_ref = tensor<AccDataType>{
            BNTestData<XDataType, DyDataType, AccDataType, TConfig>::tensor_layout,
            derivedBnDesc.GetLengths()};
    }

    void InitTensorsWithRandValue()
    {
        dy.generate(uniform_signed_initializer<DyDataType>(2e-3 /*scale*/, 1000 /*range*/));
        bnScale.generate(uniform_signed_initializer<ScaleDataType>(2e-3 /*scale*/, 1000 /*range*/));
        bnBias.generate(uniform_signed_initializer<ScaleDataType>(2e-3 /*scale*/, 1000 /*range*/));
        savedMean.generate(
            uniform_signed_initializer<MeanVarDataType>(2e-3 /*scale*/, 1000 /*range*/));
        savedInvVar.generate(
            uniform_signed_initializer<MeanVarDataType>(2e-3 /*scale*/, 1000 /*range*/));

        std::fill(dScale.begin(), dScale.end(), 0.);
        std::fill(dBias.begin(), dBias.end(), 0.);

        std::fill(dScale_ref.begin(), dScale_ref.end(), 0.);
        std::fill(dBias_ref.begin(), dBias_ref.end(), 0.);
    }
    void WriteToGPU()
    {
        auto&& handle = get_handle();

        bnScale_dev     = handle.Write(bnScale.data);
        bnBias_dev      = handle.Write(bnBias.data);
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
          typename RunSaveDataType,
          typename AccDataType,
          typename TConfig>
struct BNFwdTrainTestData : public BNTestData<XDataType, YDataType, AccDataType, TConfig>
{
    void
    SetUpImpl(const TConfig& config, miopenBatchNormMode_t t_bnmode, miopenTensorLayout_t t_layout)
    {
        BNTestData<XDataType, YDataType, AccDataType, TConfig>::SetUpImpl(
            config, t_bnmode, t_layout);
        CreateTensors();
        InitTensorsWithRandValue();
        WriteToGPU();
    }

    tensor<ScaleDataType> scale;
    tensor<BiasDataType> shift;
    tensor<RunSaveDataType> saveMean;
    tensor<RunSaveDataType> saveVariance;
    tensor<RunSaveDataType> runMean;
    tensor<RunSaveDataType> runVariance;

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
    double epsilon       = 1.0e-5;
    double averageFactor = 0.1;
    float alpha          = static_cast<float>(1.0f);
    float beta           = static_cast<float>(0);
    double activ_alpha;
    double activ_beta;
    miopenActivationMode_t activ_mode;

private:
    void CreateTensors()
    {
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(
            derivedBnDesc,
            BNTestData<XDataType, YDataType, AccDataType, TConfig>::input.desc,
            BNTestData<XDataType, YDataType, AccDataType, TConfig>::bn_mode);
        scale = tensor<ScaleDataType>{
            BNTestData<XDataType, YDataType, AccDataType, TConfig>::tensor_layout,
            derivedBnDesc.GetLengths()};
        shift = tensor<BiasDataType>{
            BNTestData<XDataType, YDataType, AccDataType, TConfig>::tensor_layout,
            derivedBnDesc.GetLengths()};
        saveMean = tensor<RunSaveDataType>{
            BNTestData<XDataType, YDataType, AccDataType, TConfig>::tensor_layout,
            derivedBnDesc.GetLengths()};
        saveVariance = tensor<RunSaveDataType>{
            BNTestData<XDataType, YDataType, AccDataType, TConfig>::tensor_layout,
            derivedBnDesc.GetLengths()};
        runMean = tensor<RunSaveDataType>{
            BNTestData<XDataType, YDataType, AccDataType, TConfig>::tensor_layout,
            derivedBnDesc.GetLengths()};
        runVariance = tensor<RunSaveDataType>{
            BNTestData<XDataType, YDataType, AccDataType, TConfig>::tensor_layout,
            derivedBnDesc.GetLengths()};
        // ref
        saveMean_ref = tensor<AccDataType>{
            BNTestData<XDataType, YDataType, AccDataType, TConfig>::tensor_layout,
            derivedBnDesc.GetLengths()};
        saveVariance_ref = tensor<AccDataType>{
            BNTestData<XDataType, YDataType, AccDataType, TConfig>::tensor_layout,
            derivedBnDesc.GetLengths()};
        runMean_ref = tensor<AccDataType>{
            BNTestData<XDataType, YDataType, AccDataType, TConfig>::tensor_layout,
            derivedBnDesc.GetLengths()};
        runVariance_ref = tensor<AccDataType>{
            BNTestData<XDataType, YDataType, AccDataType, TConfig>::tensor_layout,
            derivedBnDesc.GetLengths()};
    }

    void InitTensorsWithRandValue()
    {
        // -2.0 to 2.0
        scale.generate(uniform_signed_initializer<ScaleDataType>(2e-3 /*scale*/, 1000 /*range*/));
        shift.generate(uniform_signed_initializer<BiasDataType>(2e-3 /*scale*/, 1000 /*range*/));
        runMean.generate(
            uniform_signed_initializer<RunSaveDataType>(2e-3 /*scale*/, 1000 /*range*/));
        runVariance.generate(
            uniform_signed_initializer<RunSaveDataType>(2e-3 /*scale*/, 1000 /*range*/));

        std::transform(saveMean.data.begin(),
                       saveMean.data.end(),
                       saveMean_ref.data.begin(),
                       [](float val) { return static_cast<AccDataType>(val); });
        std::transform(saveVariance.data.begin(),
                       saveVariance.data.end(),
                       saveVariance_ref.data.begin(),
                       [](float val) { return static_cast<AccDataType>(val); });
        std::transform(runMean.data.begin(),
                       runMean.data.end(),
                       runMean_ref.data.begin(),
                       [](float val) { return static_cast<AccDataType>(val); });
        std::transform(runVariance.data.begin(),
                       runVariance.data.end(),
                       runVariance_ref.data.begin(),
                       [](float val) { return static_cast<AccDataType>(val); });
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
