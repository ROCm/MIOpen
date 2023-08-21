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

#include <gtest/gtest.h>

#include "bn_test_base.hpp"
#include "test_fusion_plan_base.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batchnorm_backward.hpp"
#include <ck/library/tensor_operation_instance/gpu/batchnorm_backward.hpp>
template <typename T>
struct BNInferTest : public ::testing::TestWithParam<std::tuple<BNTestCase, miopenTensorLayout_t>>
{
protected:
    void SetUp() override
    {
        test_skipped                       = false;
        std::tie(bn_config, tensor_layout) = GetParam();
        bn_infer_data.SetUpImpl(bn_config, tensor_layout);

        test::FusionPlan::InitFusionPlan(fusePlanDesc, bn_infer_data);
        test::FusionPlan::AddBnInfer(fusePlanDesc, params, bn_infer_data);
    }

    void TearDown() override
    {
        if(test_skipped)
            return;
        auto&& handle = get_handle();
        bn_infer_data.output.data =
            handle.Read<T>(bn_infer_data.out_dev, bn_infer_data.output.data.size());
        test::FusionPlan::ComputeRefBNInfer(bn_infer_data);
        test::FusionPlan::BnCmpare(bn_infer_data.output, bn_infer_data.ref_out);
    }
    BNTestCase bn_config;

    bool test_skipped = false;
    miopen::FusionPlanDescriptor fusePlanDesc;
    miopen::OperatorArgs params;

    BNInferSolverTest<T, BNTestCase> bn_infer_data;

    miopenTensorLayout_t tensor_layout;
};


template <typename T>
struct BNBwdTest : public ::testing::TestWithParam<std::tuple<BNTestCase, miopenTensorLayout_t>>
{
protected:
    void SetUp() override
    {
        test_skipped                       = false;
        std::tie(bn_config, tensor_layout) = GetParam();
        bn_bwd_data.SetUpImpl(bn_config, tensor_layout);

        test::FusionPlan::InitFusionPlan(fusePlanDesc, bn_bwd_data);
        test::FusionPlan::AddBwdTrain(fusePlanDesc, params, bn_bwd_data);
    }

    void TearDown() override
    {
        if(test_skipped)
            return;
        auto&& handle = get_handle();
        std::cout << "\n\n before start output\n\n";
        for(const auto&it : bn_bwd_data.output.data)
        {
            std::cout << it  << " , ";
        }
        std::cout << "\n\n before end output\n\n";
        bn_bwd_data.output.data =
            handle.Read<T>(bn_bwd_data.out_dev, bn_bwd_data.output.data.size());
        std::cout << "bn_bwd_data.out_devptr " << bn_bwd_data.out_dev.get() << std::endl;
        std::cout << "\n\n after start output\n\n";
        for(const auto&it : bn_bwd_data.output.data)
        {
            std::cout << it  << " , ";
        }
        std::cout << "\n\n after end output\n\n";
        const auto fusion_problem = miopen::FusionDescription{&fusePlanDesc};
        const auto& bn_problem =
            fusion_problem.GetBnProblem(0, miopen::batchnorm::Direction::Backward);
        test::FusionPlan::ComputeRefBNBwdTrain(bn_bwd_data, bn_problem);

        std::cout << "\n\n start ref output\n\n";
        for(const auto&it : bn_bwd_data.ref_out.data)
        {
            std::cout << it  << " , ";
        }
        std::cout << "\n\n end ref output\n\n";
        test::FusionPlan::BnCmpare(bn_bwd_data.output, bn_bwd_data.ref_out);
    }
    BNTestCase bn_config;

    bool test_skipped = false;
    miopen::FusionPlanDescriptor fusePlanDesc;
    miopen::OperatorArgs params;

    BNBwdSolverTest<T, BNTestCase> bn_bwd_data;

    miopenTensorLayout_t tensor_layout;
};
