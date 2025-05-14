/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

// XDataType       : half_float::half
// YDataYype       : half_float::half
// ScaleDataType   : half_float::half
// BiasDataType    : half_float::half
// MeanVarDataType : float
// AccDataType     : double
struct GPU_BNOCLInferSerialRun3D_FP16
    : BNInferTest<half_float::half, half_float::half, float, float, float, double, BN3DTestCase>
{
};

// XDataType       : bfloat16
// YDataYype       : bfloat16
// ScaleDataType   : float
// BiasDataType    : float
// MeanVarDataType : float
struct GPU_BNOCLInferSerialRun3D_BFP16
    : BNInferTest<bfloat16, bfloat16, float, float, float, double, BN3DTestCase>
{
};

struct GPU_BNInferSerialRun3D_FP32
    : BNInferTest<float, float, float, float, float, double, BN3DTestCase>
{
};

struct GPU_BNInferSerialRun3D_FP64
    : BNInferTest<double, double, double, double, double, double, BN3DTestCase>
{
};

// fp16
TEST_P(GPU_BNOCLInferSerialRun3D_FP16, BnV2SerialRunInferOCLfp16_3D) {}

// bfp16
TEST_P(GPU_BNOCLInferSerialRun3D_BFP16, BnV2SerialRunInferOCLbfp16_3D) {}

// fp32 (float)
TEST_P(GPU_BNInferSerialRun3D_FP32, DISABLED_BnV2SerialRunInferfp32_3D) {}

// fp16
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_BNOCLInferSerialRun3D_FP16,
                         testing::Combine(testing::ValuesIn(Network3DSerialCase<BN3DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCDHW, miopenTensorNDHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1, testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN3DTestCase>());
// bfp16
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_BNOCLInferSerialRun3D_BFP16,
                         testing::Combine(testing::ValuesIn(Network3DSerialCase<BN3DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCDHW, miopenTensorNDHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1, testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN3DTestCase>());

// fp32
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_BNInferSerialRun3D_FP32,
                         testing::Combine(testing::ValuesIn(Network3DSerialCase<BN3DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCDHW, miopenTensorNDHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN3DTestCase>());
