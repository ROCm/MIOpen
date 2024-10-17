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

// https://github.com/ROCm/MIOpen/issues/1549
// NCHW solver accepts
// XDataType       : half_float::half
// YDataYype       : half_float::half
// ScaleDataType   : half_float::half
// BiasDataType    : half_float::half
// MeanVarDataType : half_float::half
// struct GPU_BN_V1_BwdNCHW_FP16 : BNBwdTest<half_float::half, half_float::half, half_float::half,
// half_float::half, half_float::half, half_float::half, half_float::half>
// {
// };

// NHWC solver accepts
// XDataType       : half_float::half
// YDataYype       : half_float::half
// ScaleDataType   : half_float::half
// BiasDataType    : half_float::half
// MeanVarDataType : float
struct GPU_BN_V2_BwdNHWC_FP16
    : BNBwdTest<half_float::half, float, float, float, half_float::half, float, float>
{
};

// bf16 NHWC solver accepts is only on CK solver
// XDataType       : bfloat16
// YDataYype       : bfloat16
// ScaleDataType   : bfloat16
// BiasDataType    : bfloat16
// MeanVarDataType : float
struct GPU_BN_V1_BwdNHWC_BFP16 : BNBwdTest<bfloat16, float, float, float, bfloat16, float, float>
{
};

struct GPU_BN_V2_BwdNHWC_BFP16 : BNBwdTest<bfloat16, float, float, float, bfloat16, float, float>
{
};

struct GPU_BN_V1_Bwd_FP32 : BNBwdTest<float, float, float, float, float, float, float>
{
};

struct GPU_BN_V2_Bwd_FP32 : BNBwdTest<float, float, float, float, float, float, float>
{
};

struct GPU_BN_V1_BwdNHWC_FP64 : BNBwdTest<double, double, double, double, double, double, double>
{
};

struct GPU_BN_V2_BwdNHWC_FP64 : BNBwdTest<double, double, double, double, double, double, double>
{
};

// fp16
// TEST_P(GPU_BN_V1_BwdNCHW_FP16, BnV1BwdHalf) {}
TEST_P(GPU_BN_V2_BwdNHWC_FP16, BnV2BwdCKHalf) {}

// float
TEST_P(GPU_BN_V1_Bwd_FP32, BnV1BwdFloat) {}
TEST_P(GPU_BN_V2_Bwd_FP32, BnV2BwdFloat) {}

// bfp16 is only on CK solver
TEST_P(GPU_BN_V1_BwdNHWC_BFP16, BnV1BwdCKBfloat) {}
TEST_P(GPU_BN_V2_BwdNHWC_BFP16, BnV2BwdCKBfloat) {}

// double is only on CK solver
TEST_P(GPU_BN_V1_BwdNHWC_FP64, BnV1BwdCKDouble) {}
TEST_P(GPU_BN_V2_BwdNHWC_FP64, BnV2BwdCKDouble) {}

// // fp16
// INSTANTIATE_TEST_SUITE_P(Smoke,
//                          GPU_BN_V1_BwdNCHW_FP16,
//                          testing::Combine(testing::ValuesIn(NetworkSmall<BNTestCase>()),
//                                           testing::Values(miopenTensorNCHW),
//                                           testing::ValuesIn({testBNAPIV1})));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_V2_BwdNHWC_FP16,
                         testing::Combine(testing::ValuesIn(NetworkLarge<BNTestCase>()),
                                          testing::Values(miopenTensorNHWC),
                                          testing::ValuesIn({testBNAPIV2})));

// fp32
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_V1_Bwd_FP32,
                         testing::Combine(testing::ValuesIn(NetworkSmall<BNTestCase>()),
                                          testing::Values(miopenTensorNCHW),
                                          testing::ValuesIn({testBNAPIV1})));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_V2_Bwd_FP32,
                         testing::Combine(testing::ValuesIn(NetworkLarge<BNTestCase>()),
                                          testing::Values(miopenTensorNHWC),
                                          testing::ValuesIn({testBNAPIV2})));

// bfp16 is only on CK solver
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_V1_BwdNHWC_BFP16,
                         testing::Combine(testing::ValuesIn(NetworkSmall<BNTestCase>()),
                                          testing::Values(miopenTensorNHWC),
                                          testing::ValuesIn({testBNAPIV1})));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_V2_BwdNHWC_BFP16,
                         testing::Combine(testing::ValuesIn(NetworkLarge<BNTestCase>()),
                                          testing::Values(miopenTensorNHWC),
                                          testing::ValuesIn({testBNAPIV2})));

// fp64 is only on CK solver
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_V1_BwdNHWC_FP64,
                         testing::Combine(testing::ValuesIn(NetworkSmall<BNTestCase>()),
                                          testing::Values(miopenTensorNHWC),
                                          testing::ValuesIn({testBNAPIV1})));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_V2_BwdNHWC_FP64,
                         testing::Combine(testing::ValuesIn(NetworkLarge<BNTestCase>()),
                                          testing::Values(miopenTensorNHWC),
                                          testing::ValuesIn({testBNAPIV2})));
