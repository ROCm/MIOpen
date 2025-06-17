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
#include "conv_common.hpp"

#include "conv_test_base.hpp"
#include "conv_tensor_gen.hpp"

template <typename T = float, typename TestCaseType = ConvTestCaseBase>
struct ConvActivInferTest : public ::testing::TestWithParam<std::tuple<miopenActivationMode_t,
                                                                       TestCaseType,
                                                                       miopenTensorLayout_t,
                                                                       float,
                                                                       float,
                                                                       float>>,
                            ConvFwdSolverTestBase<T, T, TestCaseType>
{
protected:
    void SetUp() override
    {
        test_skipped = false;
        std::tie(activ_mode, conv_config, tensor_layout, activ_alpha, activ_beta, activ_gamma) =
            this->GetParam();

        cfsb::SetUpImpl(conv_config, tensor_layout);
        activ_desc = {activ_mode, activ_alpha, activ_beta, activ_gamma};
        std::fill(
            cfsb::output.begin(), cfsb::output.end(), std::numeric_limits<double>::quiet_NaN());

        // Setup the Fusionplan
        fusePlanDesc = miopen::FusionPlanDescriptor(miopenVerticalFusion, cfsb::input.desc);
        auto convOp =
            std::make_shared<miopen::ConvForwardOpDescriptor>(cfsb::conv_desc, cfsb::weights.desc);
        auto activOp = std::make_shared<miopen::ActivFwdFusionOpDescriptor>(activ_desc.GetMode());
        EXPECT_EQ(fusePlanDesc.AddOp(convOp), miopenStatusSuccess);
        convOp->SetArgs(params, &alpha, &beta, cfsb::wei_dev.get());
        EXPECT_EQ(fusePlanDesc.AddOp(activOp), miopenStatusSuccess);
        activOp->SetArgs(params, &alpha, &beta, activ_alpha, activ_beta, activ_gamma);
    }
    void TearDown() override
    {
        if(test_skipped)
            return;
        conv_stats stats;
        cfsb::TearDownConv();

        activationHostInfer(activ_mode,
                            activ_gamma,
                            activ_beta,
                            activ_alpha,
                            cfsb::ref_out.data,
                            cfsb::ref_out.data);
        cfsb::ThresholdChecks();
    }
    TestCaseType conv_config;
    miopen::ActivationDescriptor activ_desc;
    bool test_skipped = false;
    miopenActivationMode_t activ_mode;
    miopen::FusionPlanDescriptor fusePlanDesc;
    miopen::OperatorArgs params;
    const float alpha = static_cast<float>(1.0f);
    const float beta  = static_cast<float>(0);
    float activ_alpha = static_cast<double>(0.25f);
    float activ_beta  = static_cast<double>(0.75f);
    float activ_gamma = static_cast<double>(0.5f);
    miopenTensorLayout_t tensor_layout;
    using cfsb = ConvFwdSolverTestBase<T, T, TestCaseType>;
    Workspace wspace{};
};
