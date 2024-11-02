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

#include <miopen/miopen.h>
#include <gtest/gtest.h>
#include <miopen/solver/ck_utility_common.hpp>

#include "bn_test_data.hpp"
#include "test_operations.hpp"

// Define an enum to identify which version of BN api to call
enum BNApiType
{
    testBNAPIV1,
    testBNAPIV2,
};

// Assuming miopenTensorLayout_t and testAPI_t are the types of your enums
static std::string LayoutToString(int tensor_format)
{
    switch(tensor_format)
    {
    case miopenTensorNCHW: return "NCHW";
    case miopenTensorNHWC: return "NHWC";
    default: return "UnknownTensorFormat";
    }
}

static std::string ApiVerisonToString(int api_version)
{
    switch(api_version)
    {
    case testBNAPIV1: return "testBNAPIV1";
    case testBNAPIV2: return "testBNAPIV2";
    default: return "UnknownAPIVersion";
    }
}

// Custom test name generator to handle enums
struct TestNameGenerator
{
    std::string operator()(
        const testing::TestParamInfo<std::tuple<BNTestCase, miopenTensorLayout_t, BNApiType>>& info)
        const
    {
        const auto& layout_type = std::get<1>(info.param);
        const auto& api_type    = std::get<2>(info.param);

        std::string tensor_name = LayoutToString(layout_type);
        std::string api_name    = ApiVerisonToString(api_type);

        return tensor_name + "_" + api_name + "_" + std::to_string(info.index);
    }
};

template <typename XDataType,
          typename YDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType>
struct BNInferTest
    : public ::testing::TestWithParam<std::tuple<BNTestCase, miopenTensorLayout_t, BNApiType>>
{
protected:
    void SetUp() override
    {
        std::tie(bn_config, tensor_layout, api_type) = GetParam();
        bn_infer_test_data.SetUpImpl(bn_config, tensor_layout);

        auto&& handle = get_handle();
        if(!miopen::solver::ck_utility::is_ck_whitelist(handle.GetStream()))
        {
            test_skipped = true;
            GTEST_SKIP() << "Not Applicable on " << handle.GetDeviceName() << " Architecture";
        }
        miopenStatus_t res = miopenStatusUnknownError;
        if(api_type == BNApiType::testBNAPIV1)
        {
            res = miopenBatchNormalizationForwardInference(&handle,
                                                           bn_config.mode,
                                                           &bn_infer_test_data.alpha,
                                                           &bn_infer_test_data.beta,
                                                           &bn_infer_test_data.input.desc,
                                                           bn_infer_test_data.in_dev.get(),
                                                           &bn_infer_test_data.output.desc,
                                                           bn_infer_test_data.out_dev.get(),
                                                           &bn_infer_test_data.scale.desc,
                                                           bn_infer_test_data.scale_dev.get(),
                                                           bn_infer_test_data.shift_dev.get(),
                                                           bn_infer_test_data.estMean_dev.get(),
                                                           bn_infer_test_data.estVariance_dev.get(),
                                                           bn_infer_test_data.epsilon);
        }
        else if(api_type == BNApiType::testBNAPIV2)
        {
            res = miopenBatchNormalizationForwardInference_V2(
                &handle,
                bn_config.mode,
                &bn_infer_test_data.alpha,
                &bn_infer_test_data.beta,
                &bn_infer_test_data.input.desc,
                bn_infer_test_data.in_dev.get(),
                &bn_infer_test_data.output.desc,
                bn_infer_test_data.out_dev.get(),
                &bn_infer_test_data.scale.desc,
                &bn_infer_test_data.shift.desc,
                &bn_infer_test_data.estMean.desc,
                &bn_infer_test_data.estVariance.desc,
                bn_infer_test_data.scale_dev.get(),
                bn_infer_test_data.shift_dev.get(),
                bn_infer_test_data.estMean_dev.get(),
                bn_infer_test_data.estVariance_dev.get(),
                bn_infer_test_data.epsilon);
        }
        else
            GTEST_FAIL() << "ERROR: unknown bn api type!!";
        if(res != miopenStatusSuccess)
        {
            GTEST_FAIL() << "miopenBatchNormalizationForwardInference failed";
        }

        std::fill(bn_infer_test_data.output.begin(),
                  bn_infer_test_data.output.end(),
                  std::numeric_limits<YDataType>::quiet_NaN());
    }

    void TearDown() override
    {
        if(test_skipped || Test::HasFailure())
        {
            return;
        }

        auto&& handle                  = get_handle();
        bn_infer_test_data.output.data = handle.Read<YDataType>(
            bn_infer_test_data.out_dev, bn_infer_test_data.output.data.size());
        test::ComputeCPUBNInference(bn_infer_test_data);
        // 4e-3 is tolerance used by CK kernel.
        test::CompareTensor<YDataType>(bn_infer_test_data.output, bn_infer_test_data.ref_out, 4e-3);
    }

    BNTestCase bn_config;
    bool test_skipped = false;
    BNInferTestData<XDataType, YDataType, ScaleDataType, BiasDataType, MeanVarDataType, BNTestCase>
        bn_infer_test_data;
    miopenTensorLayout_t tensor_layout;
    BNApiType api_type;
};

template <typename XDataType,
          typename DxDataType,
          typename DyDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename DscaleDbiasDataType,
          typename MeanVarDataType>
struct BNBwdTest
    : public ::testing::TestWithParam<std::tuple<BNTestCase, miopenTensorLayout_t, BNApiType>>
{
protected:
    void SetUp() override
    {
        std::tie(bn_config, tensor_layout, api_type) = GetParam();
        bn_bwd_test_data.SetUpImpl(bn_config, tensor_layout);

        auto&& handle = get_handle();
        if(!miopen::solver::ck_utility::is_ck_whitelist(handle.GetStream()))
        {
            test_skipped = true;
            GTEST_SKIP() << "Not Applicable on " << handle.GetDeviceName() << " Architecture";
        }
        miopenStatus_t res = miopenStatusUnknownError;
        if(api_type == BNApiType::testBNAPIV1)
        {
            res = miopenBatchNormalizationBackward(&handle,
                                                   bn_config.mode,
                                                   &bn_bwd_test_data.alphaDataDiff,
                                                   &bn_bwd_test_data.betaDataDiff,
                                                   &bn_bwd_test_data.alphaParamDiff,
                                                   &bn_bwd_test_data.betaParamDiff,
                                                   &bn_bwd_test_data.input.desc,
                                                   bn_bwd_test_data.in_dev.get(),
                                                   &bn_bwd_test_data.dy.desc,
                                                   bn_bwd_test_data.dy_dev.get(),
                                                   &bn_bwd_test_data.output.desc,
                                                   bn_bwd_test_data.out_dev.get(),
                                                   &bn_bwd_test_data.bnScale.desc,
                                                   bn_bwd_test_data.bnScale_dev.get(),
                                                   bn_bwd_test_data.dScale_dev.get(),
                                                   bn_bwd_test_data.dBias_dev.get(),
                                                   bn_bwd_test_data.epsilon,
                                                   bn_bwd_test_data.savedMean_dev.get(),
                                                   bn_bwd_test_data.savedInvVar_dev.get());
        }
        else if(api_type == BNApiType::testBNAPIV2)
        {
            res = miopenBatchNormalizationBackward_V2(&handle,
                                                      bn_config.mode,
                                                      &bn_bwd_test_data.alphaDataDiff,
                                                      &bn_bwd_test_data.betaDataDiff,
                                                      &bn_bwd_test_data.alphaParamDiff,
                                                      &bn_bwd_test_data.betaParamDiff,
                                                      &bn_bwd_test_data.input.desc,
                                                      bn_bwd_test_data.in_dev.get(),
                                                      &bn_bwd_test_data.dy.desc,
                                                      bn_bwd_test_data.dy_dev.get(),
                                                      &bn_bwd_test_data.output.desc,
                                                      bn_bwd_test_data.out_dev.get(),
                                                      &bn_bwd_test_data.bnScale.desc,
                                                      &bn_bwd_test_data.dBias.desc,
                                                      &bn_bwd_test_data.savedMean.desc,
                                                      &bn_bwd_test_data.savedInvVar.desc,
                                                      bn_bwd_test_data.bnScale_dev.get(),
                                                      bn_bwd_test_data.dScale_dev.get(),
                                                      bn_bwd_test_data.dBias_dev.get(),
                                                      bn_bwd_test_data.epsilon,
                                                      bn_bwd_test_data.savedMean_dev.get(),
                                                      bn_bwd_test_data.savedInvVar_dev.get());
        }
        else
            GTEST_FAIL() << "ERROR: unknown bn api type!!";
        if(res != miopenStatusSuccess)
        {
            GTEST_FAIL() << "miopenBatchNormalizationBackward failed";
        }

        std::fill(bn_bwd_test_data.output.begin(),
                  bn_bwd_test_data.output.end(),
                  std::numeric_limits<DxDataType>::quiet_NaN());
    }

    void TearDown() override
    {
        if(test_skipped || Test::HasFailure())
        {
            return;
        }

        auto&& handle = get_handle();
        bn_bwd_test_data.output.data =
            handle.Read<DyDataType>(bn_bwd_test_data.out_dev, bn_bwd_test_data.output.data.size());
        bn_bwd_test_data.dScale.data = handle.Read<DscaleDbiasDataType>(
            bn_bwd_test_data.dScale_dev, bn_bwd_test_data.dScale.data.size());
        bn_bwd_test_data.dBias.data = handle.Read<DscaleDbiasDataType>(
            bn_bwd_test_data.dBias_dev, bn_bwd_test_data.dBias.data.size());

        test::ComputeCPUBNBwd(bn_bwd_test_data);

        test::CompareTensor<DxDataType>(bn_bwd_test_data.output, bn_bwd_test_data.ref_out, bwd_tol);
        test::CompareTensor<DscaleDbiasDataType>(
            bn_bwd_test_data.dScale, bn_bwd_test_data.dScale_ref, bwd_tol);
        test::CompareTensor<DscaleDbiasDataType>(
            bn_bwd_test_data.dBias, bn_bwd_test_data.dBias_ref, bwd_tol);
    }

    BNTestCase bn_config;
    bool test_skipped = false;
    BNBwdTestData<XDataType,
                  DxDataType,
                  DyDataType,
                  AccDataType,
                  ScaleDataType,
                  DscaleDbiasDataType,
                  MeanVarDataType,
                  BNTestCase>
        bn_bwd_test_data;
    miopenTensorLayout_t tensor_layout;
    BNApiType api_type;
    double bwd_tol = 4e-3;
};

template <typename XDataType,
          typename YDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename AccDataType>
struct BNFwdTrainTest
    : public ::testing::TestWithParam<std::tuple<BNTestCase, miopenTensorLayout_t, BNApiType>>
{
protected:
    void SetUp() override
    {
        std::tie(bn_config, tensor_layout, api_type) = GetParam();
        bn_fwd_train_test_data.SetUpImpl(bn_config, tensor_layout);

        auto&& handle = get_handle();
        if(!miopen::solver::ck_utility::is_ck_whitelist(handle.GetStream()))
        {
            test_skipped = true;
            GTEST_SKIP() << "Not Applicable on " << handle.GetDeviceName() << " Architecture";
        }
        miopenStatus_t res = miopenStatusUnknownError;
        if(api_type == BNApiType::testBNAPIV1)
        {
            res = miopenBatchNormalizationForwardTraining(
                &handle,
                bn_config.mode,
                &bn_fwd_train_test_data.alpha,
                &bn_fwd_train_test_data.beta,
                &bn_fwd_train_test_data.input.desc,
                bn_fwd_train_test_data.in_dev.get(),
                &bn_fwd_train_test_data.output.desc,
                bn_fwd_train_test_data.out_dev.get(),
                &bn_fwd_train_test_data.scale.desc,
                bn_fwd_train_test_data.scale_dev.get(),
                bn_fwd_train_test_data.shift_dev.get(),
                bn_fwd_train_test_data.averageFactor,
                bn_fwd_train_test_data.runMean_dev.get(),
                bn_fwd_train_test_data.runVariance_dev.get(),
                bn_fwd_train_test_data.epsilon,
                bn_fwd_train_test_data.saveMean_dev.get(),
                bn_fwd_train_test_data.saveVariance_dev.get());
        }
        else if(api_type == BNApiType::testBNAPIV2)
        {
            res = miopenBatchNormalizationForwardTraining_V2(
                &handle,
                bn_config.mode,
                &bn_fwd_train_test_data.alpha,
                &bn_fwd_train_test_data.beta,
                &bn_fwd_train_test_data.input.desc,
                bn_fwd_train_test_data.in_dev.get(),
                &bn_fwd_train_test_data.output.desc,
                bn_fwd_train_test_data.out_dev.get(),
                &bn_fwd_train_test_data.scale.desc,
                &bn_fwd_train_test_data.shift.desc,
                &bn_fwd_train_test_data.saveMean.desc,
                &bn_fwd_train_test_data.saveVariance.desc,
                bn_fwd_train_test_data.scale_dev.get(),
                bn_fwd_train_test_data.shift_dev.get(),
                bn_fwd_train_test_data.averageFactor,
                bn_fwd_train_test_data.runMean_dev.get(),
                bn_fwd_train_test_data.runVariance_dev.get(),
                bn_fwd_train_test_data.epsilon,
                bn_fwd_train_test_data.saveMean_dev.get(),
                bn_fwd_train_test_data.saveVariance_dev.get());
        }
        else
            GTEST_FAIL() << "ERROR: unknown bn api type!!";
        if(res != miopenStatusSuccess)
        {
            GTEST_FAIL() << "miopenBatchNormalizationForwardTraining failed";
        }

        std::fill(bn_fwd_train_test_data.output.begin(),
                  bn_fwd_train_test_data.output.end(),
                  std::numeric_limits<YDataType>::quiet_NaN());
        std::fill(bn_fwd_train_test_data.saveMean_ref.begin(),
                  bn_fwd_train_test_data.saveMean_ref.end(),
                  std::numeric_limits<YDataType>::quiet_NaN());
        std::fill(bn_fwd_train_test_data.saveVariance_ref.begin(),
                  bn_fwd_train_test_data.saveVariance_ref.end(),
                  std::numeric_limits<YDataType>::quiet_NaN());
    }

    void TearDown() override
    {
        if(test_skipped || Test::HasFailure())
        {
            return;
        }

        auto&& handle                      = get_handle();
        bn_fwd_train_test_data.output.data = handle.Read<YDataType>(
            bn_fwd_train_test_data.out_dev, bn_fwd_train_test_data.output.data.size());

        bn_fwd_train_test_data.saveMean.data = handle.Read<AccDataType>(
            bn_fwd_train_test_data.saveMean_dev, bn_fwd_train_test_data.saveMean.data.size());
        bn_fwd_train_test_data.saveVariance.data =
            handle.Read<AccDataType>(bn_fwd_train_test_data.saveVariance_dev,
                                     bn_fwd_train_test_data.saveVariance_ref.data.size());
        bn_fwd_train_test_data.runMean.data = handle.Read<AccDataType>(
            bn_fwd_train_test_data.runMean_dev, bn_fwd_train_test_data.runMean_ref.data.size());
        bn_fwd_train_test_data.runVariance.data =
            handle.Read<AccDataType>(bn_fwd_train_test_data.runVariance_dev,
                                     bn_fwd_train_test_data.runVariance_ref.data.size());
        test::ComputeCPUBNFwdTrain(bn_fwd_train_test_data);

        // 4e-3 is tolerance used by CK kernel.
        test::CompareTensor<YDataType>(
            bn_fwd_train_test_data.output, bn_fwd_train_test_data.ref_out, 4e-3);
        test::CompareTensor<AccDataType>(
            bn_fwd_train_test_data.saveMean, bn_fwd_train_test_data.saveMean_ref, 4e-3);
        test::CompareTensor<AccDataType>(
            bn_fwd_train_test_data.saveVariance, bn_fwd_train_test_data.saveVariance_ref, 4e-3);
        test::CompareTensor<AccDataType>(
            bn_fwd_train_test_data.runMean, bn_fwd_train_test_data.runMean_ref, 4e-3);
        test::CompareTensor<AccDataType>(
            bn_fwd_train_test_data.runVariance, bn_fwd_train_test_data.runVariance_ref, 4e-3);
    }

    BNTestCase bn_config;
    bool test_skipped = false;
    BNFwdTrainTestData<XDataType, YDataType, ScaleDataType, BiasDataType, AccDataType, BNTestCase>
        bn_fwd_train_test_data;
    miopenTensorLayout_t tensor_layout;
    BNApiType api_type;
};
