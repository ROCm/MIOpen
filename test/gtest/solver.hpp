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
#include "cpu_conv.hpp"
#include "get_handle.hpp"
#include "tensor_util.hpp"
#include <fusionHost.hpp>
#include <miopen/conv/data_invoke_params.hpp>

#include "conv_test_base.hpp"

template <typename T = float>
struct ConvFwdSolverTest
    : public ::testing::TestWithParam<
          std::tuple<miopenConvFwdAlgorithm_t, ConvTestCase, miopenTensorLayout_t>>,
      ConvFwdSolverTestBase<T>
{
public:
    void SetUp() override
    {
        test_skipped                               = false;
        std::tie(algo, conv_config, tensor_layout) = GetParam();
        ConvFwdSolverTestBase<T>::SetUpImpl(conv_config, tensor_layout);
    }
    void TearDown() override
    {
        if(test_skipped)
            return;
        ConvFwdSolverTestBase<T>::TearDownConv();
        ConvFwdSolverTestBase<T>::ThresholdChecks();
    }
    ConvTestCase conv_config;
    miopenConvFwdAlgorithm_t algo = miopenConvolutionFwdAlgoDirect;
    bool test_skipped             = false;
    miopenTensorLayout_t tensor_layout;
};
