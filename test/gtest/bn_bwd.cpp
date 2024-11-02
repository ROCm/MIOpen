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
/* typename XDataType,
   typename DxDataType,
   typename DyDataType,
   typename AccDataType,
   typename ScaleDataType,
   typename DscaleDbiasDataType,
   typename MeanVarDataType> */

struct GPU_BN_CK_BWD_Large_FP16
    : BNBwdTest<half_float::half, float, float, float, half_float::half, float, float>
{
};

struct GPU_BN_OCL_BWD_Large_FP16
    : BNBwdTest<half_float::half, half_float::half, half_float::half, float, float, float, float>
{
};

struct GPU_BN_CK_BWD_Large_BFP16 : BNBwdTest<bfloat16, float, float, float, bfloat16, float, float>
{
};

struct GPU_BN_OCL_BWD_Large_BFP16
    : BNBwdTest<bfloat16, bfloat16, bfloat16, float, float, float, float>
{
};

struct GPU_BN_BWD_Small_FP32 : BNBwdTest<float, float, float, float, float, float, float>
{
};

struct GPU_BN_BWD_Large_FP32 : BNBwdTest<float, float, float, float, float, float, float>
{
};

struct GPU_BN_BWD_Small_FP64 : BNBwdTest<double, double, double, double, double, double, double>
{
};

struct GPU_BN_BWD_Large_FP64 : BNBwdTest<double, double, double, double, double, double, double>
{
};

// fp16
TEST_P(GPU_BN_CK_BWD_Large_FP16, DISABLED_BnV2LargeBWDCKfp16) {}
TEST_P(GPU_BN_OCL_BWD_Large_FP16, BnV2LargeBWDOCLfp16) {}

// bfp16
TEST_P(GPU_BN_CK_BWD_Large_BFP16, DISABLED_BnV2LargeBWDCKbfp16) {}
TEST_P(GPU_BN_OCL_BWD_Large_BFP16, BnV2LargeBWDOCLbfp16) {}

// fp32 (float)
TEST_P(GPU_BN_BWD_Small_FP32, BnV1SmallBWDCKfp32) {}
TEST_P(GPU_BN_BWD_Large_FP32, BnV2LargeBWDCKfp32) {}

// fp64
TEST_P(GPU_BN_BWD_Small_FP64, DISABLED_BnV1SmallBWDCKfp64) {}
TEST_P(GPU_BN_BWD_Large_FP64, DISABLED_BnV2LargeBWDCKfp64) {}

// fp16
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_CK_BWD_Large_FP16,
                         testing::Combine(testing::ValuesIn(NetworkSmall<BNTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({testBNAPIV2})),
                         TestNameGenerator());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_OCL_BWD_Large_FP16,
                         testing::Combine(testing::ValuesIn(NetworkLarge<BNTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW}),
                                          testing::ValuesIn({testBNAPIV2})),
                         TestNameGenerator());

// bfp16
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_CK_BWD_Large_BFP16,
                         testing::Combine(testing::ValuesIn(NetworkLarge<BNTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({testBNAPIV2})),
                         TestNameGenerator());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_OCL_BWD_Large_BFP16,
                         testing::Combine(testing::ValuesIn(NetworkLarge<BNTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW}),
                                          testing::ValuesIn({testBNAPIV2})),
                         TestNameGenerator());

// fp32
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_BWD_Small_FP32,
                         testing::Combine(testing::ValuesIn(NetworkSmall<BNTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW}),
                                          testing::ValuesIn({testBNAPIV1})),
                         TestNameGenerator());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_BWD_Large_FP32,
                         testing::Combine(testing::ValuesIn(NetworkLarge<BNTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW}),
                                          testing::ValuesIn({testBNAPIV2})),
                         TestNameGenerator());
// // fp64
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_BWD_Small_FP64,
                         testing::Combine(testing::ValuesIn(NetworkSmall<BNTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({testBNAPIV1})),
                         TestNameGenerator());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_BWD_Large_FP64,
                         testing::Combine(testing::ValuesIn(NetworkLarge<BNTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({testBNAPIV2})),
                         TestNameGenerator());
