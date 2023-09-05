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

#include "bn_test_data.hpp"
#include "test_operations.hpp"

template <typename XDataType, typename YDataType, typename ScaleDataType, 
          typename BiasDataType, typename MeanVarDataType>
struct BNInferTest : public ::testing::TestWithParam<std::tuple<BNTestCase, miopenTensorLayout_t>>
{
protected:
    void SetUp() override
    {
        test_skipped                       = false;
        std::tie(bn_config, tensor_layout) = GetParam();
        bn_infer_test_data.SetUpImpl(bn_config, tensor_layout);

        auto&& handle = get_handle();
        miopenBatchNormalizationForwardInference(
        &handle,
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

        std::fill(bn_infer_test_data.output.begin(), bn_infer_test_data.output.end(), std::numeric_limits<YDataType>::quiet_NaN());
    }

    void TearDown() override
    {
        if(test_skipped)
            return;
        auto&& handle = get_handle();
        bn_infer_test_data.output.data =
            handle.Read<YDataType>(bn_infer_test_data.out_dev, bn_infer_test_data.output.data.size());
        test::ComputeCPUBNInference(bn_infer_test_data);
        
        if constexpr(std::is_same_v<YDataType, double>)
        {
            // tolerance for CK solver tolerance for
            test::CompareTensor<YDataType>(bn_infer_test_data.output, bn_infer_test_data.ref_out, 1e-8);
        }
        else{
            test::CompareTensor<YDataType>(bn_infer_test_data.output, bn_infer_test_data.ref_out);
        }

    }

    BNTestCase bn_config;
    bool test_skipped = false;
    BNInferTestData<XDataType, YDataType, ScaleDataType, 
          BiasDataType, MeanVarDataType, BNTestCase> bn_infer_test_data;
    miopenTensorLayout_t tensor_layout;
};
