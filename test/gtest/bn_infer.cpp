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

#include "bn.hpp"

struct BNInferTestHalf
    : BNInferTest<half_float::half, half_float::half, half_float::half, half_float::half, float>
{
};

struct BNInferTestFloat : BNInferTest<float, float, float, float, float>
{
};

struct BNInferTestDouble : BNInferTest<double, double, double, double, double>
{
};

struct BNInferTestBFloat16 : BNInferTest<bfloat16, bfloat16, bfloat16, bfloat16, double>
{
};

TEST_P(BNInferTestHalf, BnInferCKHalf) {}

TEST_P(BNInferTestFloat, BnInferCKFloat) {}

// Currently disabled since miopen::batchnorm::MakeForwardTrainingNetworkConfig
// only supports half and float
TEST_P(BNInferTestDouble, DISABLED_BnInferCKDouble) {}
TEST_P(BNInferTestBFloat16, DISABLED_BnInferCKBFloat16) {}

INSTANTIATE_TEST_SUITE_P(BNInferTestHalfNHWCSuite,
                         BNInferTestHalf,
                         testing::Combine(testing::ValuesIn(Network1()),
                                          testing::Values(miopenTensorNHWC)));

INSTANTIATE_TEST_SUITE_P(BNInferTestFloatNHWCSuite,
                         BNInferTestFloat,
                         testing::Combine(testing::ValuesIn(Network1()),
                                          testing::Values(miopenTensorNHWC)));

INSTANTIATE_TEST_SUITE_P(BNInferTestFloatNHWCSuite,
                         BNInferTestDouble,
                         testing::Combine(testing::ValuesIn(Network1()),
                                          testing::Values(miopenTensorNHWC)));

INSTANTIATE_TEST_SUITE_P(BNInferTestFloatNHWCSuite,
                         BNInferTestBFloat16,
                         testing::Combine(testing::ValuesIn(Network1()),
                                          testing::Values(miopenTensorNHWC)));
