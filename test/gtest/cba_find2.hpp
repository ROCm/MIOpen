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
#include <miopen/problem.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/fusion/fusion_invoke_params.hpp>
#include <serialize.hpp>
#include <fusionHost.hpp>

#include "tensor_util.hpp"
#include "get_handle.hpp"
#include "conv_common.hpp"

#include "conv_test_base.hpp"
#include "conv_tensor_gen.hpp"

template <typename T = float>
struct ConvBiasActivInferFind2Test
    : public ::testing::TestWithParam<
          std::tuple<miopenActivationMode_t, ConvTestCaseBase, miopenTensorLayout_t>>,
      ConvFwdSolverTestBase<T, T>
{
protected:
    ConvTestCaseBase conv_config;
    miopen::ActivationDescriptor activ_desc;
    tensor<T> bias;
    miopen::Allocator::ManageDataPtr bias_dev;
    bool test_skipped = false;
    miopenActivationMode_t activ_mode;
    /*
    miopen::FusionPlanDescriptor fusePlanDesc;
    */
    miopen::OperatorArgs params;
    const float alpha       = static_cast<float>(1.0f);
    const float beta        = static_cast<float>(0);
    const float activ_alpha = static_cast<double>(0.5f);
    const float activ_beta  = static_cast<double>(0.5f);
    const float activ_gamma = static_cast<double>(0.5f);
    miopenTensorLayout_t tensor_layout;

    miopen::FusedProblem fused_problem;
    miopen::AnyInvokeParams invoke_params;

    using cfsb = ConvFwdSolverTestBase<T, T>;

    void SetUp() override
    {
        test_skipped                                     = false;
        std::tie(activ_mode, conv_config, tensor_layout) = GetParam();

        cfsb::SetUpImpl(conv_config, tensor_layout);
        activ_desc = {activ_mode, activ_alpha, activ_beta, activ_gamma};
        bias       = tensor<T>{1, static_cast<size_t>(conv_config.k), 1, 1};
        bias.generate(tensor_elem_gen_integer{3});
        std::fill(
            cfsb::output.begin(), cfsb::output.end(), std::numeric_limits<double>::quiet_NaN());
        bias_dev = get_handle().Write(bias.data);

        // Setup the fusion problem
        fused_problem = miopen::FusedProblem{{
            MakeConvProblem(),
            MakeBiasProblem(),
            MakeActivationProblem(),
        }};

        fused_problem.PropagateDescriptors();
        ASSERT_NO_THROW(invoke_params = MakeInvokeParams());
    }

    void TearDown() override
    {
        if(test_skipped)
            return;
        ValidateResult();
    }

    void ValidateResult()
    {
        CalculateCPUValuesIfNeeded();
        cfsb::ThresholdChecks();
    }

private:
    bool cpu_values_calculated = false;

    [[nodiscard]] miopen::Problem MakeConvProblem() const
    {
        auto problem = miopen::Problem{};
        problem.SetOperatorDescriptor(cfsb::conv_desc);
        problem.RegisterTensorDescriptor(miopenTensorConvolutionX, cfsb::input.desc);
        problem.RegisterTensorDescriptor(miopenTensorConvolutionW, cfsb::weights.desc);
        problem.RegisterTensorDescriptor(miopenTensorConvolutionY, cfsb::output.desc);
        return problem;
    }

    [[nodiscard]] miopen::Problem MakeBiasProblem() const
    {
        auto problem = miopen::Problem{};
        problem.SetOperatorDescriptor(miopen::BiasDescriptor{});
        problem.RegisterTensorDescriptor(miopenTensorBias, bias.desc);
        return problem;
    }

    [[nodiscard]] miopen::Problem MakeActivationProblem() const
    {
        auto problem = miopen::Problem{};
        problem.SetOperatorDescriptor(activ_desc);
        return problem;
    }

    [[nodiscard]] miopen::AnyInvokeParams MakeInvokeParams()
    {
        return fused_problem.MakeInvokeParams(
            [&](auto id, const auto& desc) {
                if(id == miopenTensorConvolutionX)
                {
                    EXPECT_EQ(desc, cfsb::input.desc);
                    return cfsb::in_dev.get();
                }
                if(id == miopenTensorConvolutionW)
                {
                    EXPECT_EQ(desc, cfsb::weights.desc);
                    return cfsb::wei_dev.get();
                }
                if(id == miopenTensorActivationY)
                {
                    EXPECT_EQ(desc, cfsb::output.desc);
                    return cfsb::out_dev.get();
                }
                if(id == miopenTensorBias)
                {
                    EXPECT_EQ(desc, bias.desc);
                    return bias_dev.get();
                }
                MIOPEN_THROW(miopenStatusInternalError);
            },
            params);
    }

    void CalculateCPUValuesIfNeeded()
    {
        if(cpu_values_calculated)
            return;

        cpu_values_calculated = true;
        cfsb::TearDownConv();
        cpu_bias_forward(cfsb::ref_out, bias);

        activationHostInfer(activ_mode,
                            activ_gamma,
                            activ_beta,
                            activ_alpha,
                            cfsb::ref_out.data,
                            cfsb::ref_out.data);
    }
};
