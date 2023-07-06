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

#include "bn_test_base.hpp"
#include "test_fusion_plan_base.hpp"

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
        test::FusionPlan::ComputeRefBN(bn_infer_data);
        auto&& handle = get_handle();
        bn_infer_data.output.data =
            handle.Read<T>(bn_infer_data.out_dev, bn_infer_data.output.data.size());
        test::FusionPlan::ComputeRefBN(bn_infer_data);
        test::FusionPlan::BnCmpare(bn_infer_data.output, bn_infer_data.ref_out);
    }
    BNTestCase bn_config;

    bool test_skipped = false;
    miopen::FusionPlanDescriptor fusePlanDesc;
    miopen::OperatorArgs params;

    BNInferSolverTest<T, BNTestCase> bn_infer_data;

    miopenTensorLayout_t tensor_layout;
};
