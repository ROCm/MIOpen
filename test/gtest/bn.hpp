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

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_FIND_ENFORCE)
#define WORKAROUND_SWDEV_547301 1
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
    case miopenTensorNCDHW: return "NCDHW";
    case miopenTensorNHWC: return "NHWC";
    case miopenTensorNDHWC: return "NDHWC";
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

static std::string BNModeToString(int bn_mode)
{
    switch(bn_mode)
    {
    case miopenBNPerActivation: return "BNPerActivation";
    case miopenBNSpatial: return "BNSpatial";
    default: return "UnknownBNMode";
    }
}

inline miopenTuningPolicy_t GetTuningPolicy(void)
{
    auto str = env::value(MIOPEN_FIND_ENFORCE);
    if(str.empty())
        return miopenTuningPolicy_t::miopenTuningPolicyNone;
    for(auto& c : str)
        c = toupper(static_cast<unsigned char>(c));
    if(str == "NONE")
    {
        return miopenTuningPolicy_t::miopenTuningPolicyNone;
    }
    else if(str == "DB_UPDATE")
    {
        return miopenTuningPolicy_t::miopenTuningPolicyDbUpdate;
    }
    else if(str == "SEARCH")
    {
        return miopenTuningPolicy_t::miopenTuningPolicySearch;
    }
    else if(str == "SEARCH_DB_UPDATE")
    {
        return miopenTuningPolicy_t::miopenTuningPolicySearchDbUpdate;
    }
    else if(str == "DB_CLEAN")
    {
        return miopenTuningPolicy_t::miopenTuningPolicyDbClean;
    }
    else
    {
        MIOPEN_LOG_NQE("Wrong MIOPEN_FIND_ENFORCE value, using None.");
        return miopenTuningPolicy_t::miopenTuningPolicyNone;
    }
}

// Custom test name generator to handle enums
template <typename TestCase>
struct TestNameGenerator
{
    std::string
    operator()(const testing::TestParamInfo<std::tuple<TestCase,
                                                       miopenTensorLayout_t,
                                                       miopenBatchNormMode_t,
                                                       BNApiType,
                                                       miopenActivationMode_t>>& info) const
    {
        constexpr int dimension = std::is_same<TestCase, BN2DTestCase>::value   ? 2
                                  : std::is_same<TestCase, BN3DTestCase>::value ? 3
                                                                                : -1;
        static_assert(dimension > 0);

        const auto& layout_type    = std::get<1>(info.param);
        const auto& batchnorm_mode = std::get<2>(info.param);
        const auto& api_type       = std::get<3>(info.param);

        std::string tensor_name  = LayoutToString(layout_type);
        std::string bn_mode_name = BNModeToString(batchnorm_mode);
        std::string api_name     = ApiVerisonToString(api_type);

        std::ostringstream oss;
        oss << tensor_name + "_" + bn_mode_name + "_" + api_name + "_Dim_" +
                   std::to_string(dimension) + "_test_id_" + std::to_string(info.index);
        return oss.str();
    }
};

template <typename XDataType,
          typename YDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType,
          typename AccDataType,
          typename TestCase>
struct BNInferTest : public ::testing::TestWithParam<std::tuple<TestCase,
                                                                miopenTensorLayout_t,
                                                                miopenBatchNormMode_t,
                                                                BNApiType,
                                                                miopenActivationMode_t>>
{
protected:
    void SetUp() override
    {
        std::tie(bn_config, tensor_layout, bn_mode, api_type, bn_infer_test_data.activ_mode) =
            this->GetParam();
        bn_infer_test_data.SetUpImpl(bn_config, bn_mode, tensor_layout);

        bn_infer_test_data.activ_alpha = static_cast<double>(0.1f);
        bn_infer_test_data.activ_beta  = static_cast<double>(0.3f);

        auto&& handle                      = get_handle();
        miopenStatus_t res                 = miopenStatusUnknownError;
        miopenTuningPolicy_t tuning_policy = GetTuningPolicy();
        if(bn_infer_test_data.activ_mode > 0)
        {
            miopenCreateActivationDescriptor(&activ_desc);
            miopenSetActivationDescriptor(activ_desc,
                                          bn_infer_test_data.activ_mode,
                                          bn_infer_test_data.activ_alpha,
                                          bn_infer_test_data.activ_beta,
                                          static_cast<double>(0.0));
            if(tuning_policy == miopenTuningPolicy_t::miopenTuningPolicySearch)
            {
                miopenSetTuningPolicy(&handle, tuning_policy); // set tuning
            }
            res =
                miopenBatchNormForwardInferenceActivation(&handle,
                                                          bn_mode,
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
                                                          bn_infer_test_data.epsilon,
                                                          activ_desc);
            if(tuning_policy == miopenTuningPolicy_t::miopenTuningPolicySearch)
            {
                miopenSetTuningPolicy(&handle,
                                      miopenTuningPolicy_t::miopenTuningPolicyNone); // unset tuning
            }
            miopenDestroyActivationDescriptor(activ_desc);
        }
        else
        {
            if(api_type == BNApiType::testBNAPIV1)
            {
                if(tuning_policy == miopenTuningPolicy_t::miopenTuningPolicySearch)
                {
                    miopenSetTuningPolicy(&handle, tuning_policy); // set tuning
                }
                res = miopenBatchNormalizationForwardInference(
                    &handle,
                    bn_mode,
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
                if(tuning_policy == miopenTuningPolicy_t::miopenTuningPolicySearch)
                {
                    miopenSetTuningPolicy(
                        &handle, miopenTuningPolicy_t::miopenTuningPolicyNone); // unset tuning
                }
            }
            else if(api_type == BNApiType::testBNAPIV2)
            {
                if(tuning_policy == miopenTuningPolicy_t::miopenTuningPolicySearch)
                {
                    miopenSetTuningPolicy(&handle, tuning_policy); // set tuning
                }
                res = miopenBatchNormalizationForwardInference_V2(
                    &handle,
                    bn_mode,
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
                if(tuning_policy == miopenTuningPolicy_t::miopenTuningPolicySearch)
                {
                    miopenSetTuningPolicy(
                        &handle, miopenTuningPolicy_t::miopenTuningPolicyNone); // unset tuning
                }
            }
            else
                GTEST_FAIL() << "ERROR: unknown bn api type!!";
        }
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
        if(test_skipped || ::testing::Test::HasFailure())
        {
            return;
        }

        auto&& handle                  = get_handle();
        bn_infer_test_data.output.data = handle.Read<YDataType>(
            bn_infer_test_data.out_dev, bn_infer_test_data.output.data.size());
        test::ComputeCPUBNInference(bn_infer_test_data);
        activationHostInfer(bn_infer_test_data.activ_mode,
                            static_cast<double>(0.0),
                            bn_infer_test_data.activ_beta,
                            bn_infer_test_data.activ_alpha,
                            bn_infer_test_data.out_ref.data,
                            bn_infer_test_data.out_ref.data);
        auto tolerance = 4e-3;
#if WORKAROUND_SWDEV_547301
        // Workaround to let BN Infer tests pass on Navi4x,SWDEV-547301
        tolerance = miopen::StartsWith(handle.GetDeviceName(), "gfx120") ? 8e-3 : 4e-3;
#endif
        test::CompareTensor<YDataType>(
            bn_infer_test_data.output, bn_infer_test_data.out_ref, tolerance);
    }

    TestCase bn_config;
    bool test_skipped = false;
    BNInferTestData<XDataType,
                    YDataType,
                    ScaleDataType,
                    BiasDataType,
                    MeanVarDataType,
                    AccDataType,
                    TestCase>
        bn_infer_test_data;
    miopenTensorLayout_t tensor_layout;
    miopenBatchNormMode_t bn_mode;
    BNApiType api_type;
    miopenActivationDescriptor_t activ_desc;
};

template <typename XDataType,
          typename DxDataType,
          typename DyDataType,
          typename ScaleDataType,
          typename DscaleDbiasDataType,
          typename MeanVarDataType,
          typename AccDataType,
          typename TestCase>
struct BNBwdTest : public ::testing::TestWithParam<std::tuple<TestCase,
                                                              miopenTensorLayout_t,
                                                              miopenBatchNormMode_t,
                                                              BNApiType,
                                                              miopenActivationMode_t>>
{
protected:
    void SetUp() override
    {
        std::tie(bn_config, tensor_layout, bn_mode, api_type, bn_bwd_test_data.activ_mode) =
            this->GetParam();
        bn_bwd_test_data.SetUpImpl(bn_config, bn_mode, tensor_layout);

        bn_bwd_test_data.activ_alpha = bn_bwd_test_data.activ_mode == miopenActivationCLAMP
                                           ? static_cast<double>(0.1f)
                                           : static_cast<double>(0.5f);
        bn_bwd_test_data.activ_beta  = static_cast<double>(0.3f);

        auto&& handle      = get_handle();
        miopenStatus_t res = miopenStatusUnknownError;
        if(bn_bwd_test_data.activ_mode > 0)
        {
            miopenCreateActivationDescriptor(&activ_desc);
            miopenSetActivationDescriptor(activ_desc,
                                          bn_bwd_test_data.activ_mode,
                                          bn_bwd_test_data.activ_alpha,
                                          bn_bwd_test_data.activ_beta,
                                          static_cast<double>(0.0));
            res = miopenBatchNormBackwardActivation(&handle,
                                                    bn_mode,
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
                                                    bn_bwd_test_data.bnBias_dev.get(),
                                                    bn_bwd_test_data.dScale_dev.get(),
                                                    bn_bwd_test_data.dBias_dev.get(),
                                                    bn_bwd_test_data.epsilon,
                                                    bn_bwd_test_data.savedMean_dev.get(),
                                                    bn_bwd_test_data.savedInvVar_dev.get(),
                                                    activ_desc);
            miopenDestroyActivationDescriptor(activ_desc);
        }
        else
        {
            if(api_type == BNApiType::testBNAPIV1)
            {
                res = miopenBatchNormalizationBackward(&handle,
                                                       bn_mode,
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
                                                          bn_mode,
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
        }
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
        if(test_skipped || ::testing::Test::HasFailure())
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

        test::CompareTensor<DxDataType, AccDataType>(
            bn_bwd_test_data.output, bn_bwd_test_data.out_ref, bwd_tol);
        test::CompareTensor<DscaleDbiasDataType, AccDataType>(
            bn_bwd_test_data.dScale, bn_bwd_test_data.dScale_ref, bwd_tol);
        test::CompareTensor<DscaleDbiasDataType, AccDataType>(
            bn_bwd_test_data.dBias, bn_bwd_test_data.dBias_ref, bwd_tol);
    }

    TestCase bn_config;
    bool test_skipped = false;
    BNBwdTestData<XDataType,
                  DxDataType,
                  DyDataType,
                  ScaleDataType,
                  DscaleDbiasDataType,
                  MeanVarDataType,
                  AccDataType,
                  TestCase>
        bn_bwd_test_data;
    miopenTensorLayout_t tensor_layout;
    miopenBatchNormMode_t bn_mode;
    BNApiType api_type;
    miopenActivationDescriptor_t activ_desc;
    double bwd_tol = 4e-3;
};

template <typename XDataType,
          typename YDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename RunSaveDataType,
          typename AccDataType,
          typename TestCase>
struct BNFwdTrainTest : public ::testing::TestWithParam<std::tuple<TestCase,
                                                                   miopenTensorLayout_t,
                                                                   miopenBatchNormMode_t,
                                                                   BNApiType,
                                                                   miopenActivationMode_t>>
{
protected:
    void SetUp() override
    {
        std::tie(bn_config, tensor_layout, bn_mode, api_type, bn_fwd_train_test_data.activ_mode) =
            this->GetParam();
        bn_fwd_train_test_data.SetUpImpl(bn_config, bn_mode, tensor_layout);

        bn_fwd_train_test_data.activ_alpha = static_cast<double>(0.1f);
        bn_fwd_train_test_data.activ_beta  = static_cast<double>(0.3f);

        auto&& handle      = get_handle();
        miopenStatus_t res = miopenStatusUnknownError;
        if(bn_fwd_train_test_data.activ_mode > 0)
        {
            miopenCreateActivationDescriptor(&activ_desc);
            miopenSetActivationDescriptor(activ_desc,
                                          bn_fwd_train_test_data.activ_mode,
                                          bn_fwd_train_test_data.activ_alpha,
                                          bn_fwd_train_test_data.activ_beta,
                                          static_cast<double>(0.0));
            res = miopenBatchNormForwardTrainingActivation(
                &handle,
                bn_mode,
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
                bn_fwd_train_test_data.saveVariance_dev.get(),
                activ_desc);
            miopenDestroyActivationDescriptor(activ_desc);
        }
        else
        {
            if(api_type == BNApiType::testBNAPIV1)
            {
                res = miopenBatchNormalizationForwardTraining(
                    &handle,
                    bn_mode,
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
                    bn_mode,
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
        }
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
        if(test_skipped || ::testing::Test::HasFailure())
        {
            return;
        }

        auto&& handle                      = get_handle();
        bn_fwd_train_test_data.output.data = handle.Read<YDataType>(
            bn_fwd_train_test_data.out_dev, bn_fwd_train_test_data.output.data.size());

        bn_fwd_train_test_data.saveMean.data = handle.Read<RunSaveDataType>(
            bn_fwd_train_test_data.saveMean_dev, bn_fwd_train_test_data.saveMean.data.size());
        bn_fwd_train_test_data.saveVariance.data =
            handle.Read<RunSaveDataType>(bn_fwd_train_test_data.saveVariance_dev,
                                         bn_fwd_train_test_data.saveVariance_ref.data.size());
        bn_fwd_train_test_data.runMean.data = handle.Read<RunSaveDataType>(
            bn_fwd_train_test_data.runMean_dev, bn_fwd_train_test_data.runMean_ref.data.size());
        bn_fwd_train_test_data.runVariance.data =
            handle.Read<RunSaveDataType>(bn_fwd_train_test_data.runVariance_dev,
                                         bn_fwd_train_test_data.runVariance_ref.data.size());
        test::ComputeCPUBNFwdTrain(bn_fwd_train_test_data);
        activationHostInfer(bn_fwd_train_test_data.activ_mode,
                            static_cast<double>(0.0),
                            bn_fwd_train_test_data.activ_beta,
                            bn_fwd_train_test_data.activ_alpha,
                            bn_fwd_train_test_data.out_ref.data,
                            bn_fwd_train_test_data.out_ref.data);

        // 4e-3 is tolerance used by CK kernel.
        test::CompareTensor<YDataType>(
            bn_fwd_train_test_data.output, bn_fwd_train_test_data.out_ref, 4e-3);
        test::CompareTensor<RunSaveDataType>(
            bn_fwd_train_test_data.saveMean, bn_fwd_train_test_data.saveMean_ref, 4e-3);
        test::CompareTensor<RunSaveDataType>(
            bn_fwd_train_test_data.saveVariance, bn_fwd_train_test_data.saveVariance_ref, 4e-3);
        test::CompareTensor<RunSaveDataType>(
            bn_fwd_train_test_data.runMean, bn_fwd_train_test_data.runMean_ref, 4e-3);
        test::CompareTensor<RunSaveDataType>(
            bn_fwd_train_test_data.runVariance, bn_fwd_train_test_data.runVariance_ref, 4e-3);
    }

    TestCase bn_config;
    bool test_skipped = false;
    BNFwdTrainTestData<XDataType,
                       YDataType,
                       ScaleDataType,
                       BiasDataType,
                       RunSaveDataType,
                       AccDataType,
                       TestCase>
        bn_fwd_train_test_data;
    miopenTensorLayout_t tensor_layout;
    miopenBatchNormMode_t bn_mode;
    BNApiType api_type;
    miopenActivationDescriptor_t activ_desc;
};
