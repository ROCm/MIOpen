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

// XDataType       : half_float::half
// YDataYype       : half_float::half
// ScaleDataType   : half_float::half
// BiasDataType    : half_float::half
// MeanVarDataType : float
// AccDataType     : double
struct GPU_BNCKInferLarge2D_FP16 : BNInferTest<half_float::half,
                                               half_float::half,
                                               half_float::half,
                                               half_float::half,
                                               float,
                                               double,
                                               BN2DTestCase>
{
};

struct GPU_BNOCLInferLarge2D_FP16
    : BNInferTest<half_float::half, half_float::half, float, float, float, double, BN2DTestCase>
{
};

struct GPU_BNOCLInferLarge3D_FP16
    : BNInferTest<half_float::half, half_float::half, float, float, float, double, BN3DTestCase>
{
};

// XDataType       : bfloat16
// YDataYype       : bfloat16
// ScaleDataType   : bfloat16
// BiasDataType    : bfloat16
// MeanVarDataType : float
struct GPU_BNCKInferLarge2D_BFP16
    : BNInferTest<bfloat16, bfloat16, bfloat16, bfloat16, float, double, BN2DTestCase>
{
};

// XDataType       : bfloat16
// YDataYype       : bfloat16
// ScaleDataType   : float
// BiasDataType    : float
// MeanVarDataType : float
struct GPU_BNOCLInferLarge2D_BFP16
    : BNInferTest<bfloat16, bfloat16, float, float, float, double, BN2DTestCase>
{
};

struct GPU_BNOCLInferLarge3D_BFP16
    : BNInferTest<bfloat16, bfloat16, float, float, float, double, BN3DTestCase>
{
};

struct GPU_BNInferSmall2D_FP32
    : BNInferTest<float, float, float, float, float, double, BN2DTestCase>
{
};
struct GPU_BNInferSmall3D_FP32
    : BNInferTest<float, float, float, float, float, double, BN3DTestCase>
{
};
struct GPU_BNInferLarge2D_FP32
    : BNInferTest<float, float, float, float, float, double, BN2DTestCase>
{
};

struct GPU_BNInferSmall2D_FP64
    : BNInferTest<double, double, double, double, double, double, BN2DTestCase>
{
};

struct GPU_BNInferLarge2D_FP64
    : BNInferTest<double, double, double, double, double, double, BN2DTestCase>
{
};

// fp16
TEST_P(GPU_BNCKInferLarge2D_FP16, DISABLED_BnV2LargeInferCKfp16_2D) {}
TEST_P(GPU_BNOCLInferLarge2D_FP16, BnV2LargeInferOCLfp16_2D) {}
TEST_P(GPU_BNOCLInferLarge3D_FP16, BnV2LargeInferOCLfp16_3D) {}

// bfp16
TEST_P(GPU_BNCKInferLarge2D_BFP16, DISABLED_BnV2LargeInferCKbfp16_2D) {}
TEST_P(GPU_BNOCLInferLarge2D_BFP16, BnV2LargeInferOCLbfp16_2D) {}
TEST_P(GPU_BNOCLInferLarge3D_BFP16, BnV2LargeInferOCLbfp16_3D) {}

// fp32 (float)
TEST_P(GPU_BNInferSmall2D_FP32, BnV1SmallInferfp32_2D) {}
TEST_P(GPU_BNInferLarge2D_FP32, BnV2LargeInferfp32_2D) {}
TEST_P(GPU_BNInferSmall3D_FP32, BnV1SmallInferfp32_3D) {}

// fp64
TEST_P(GPU_BNInferSmall2D_FP64, DISABLED_BnV1SmallInferfp64_2D) {}
TEST_P(GPU_BNInferLarge2D_FP64, DISABLED_BnV2LargeInferfp64_2D) {}

// fp16
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNCKInferLarge2D_FP16,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNOCLInferLarge2D_FP16,
                         testing::Combine(testing::ValuesIn(Network2DLarge<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1, testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNOCLInferLarge3D_FP16,
                         testing::Combine(testing::ValuesIn(Network3DBN<BN3DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCDHW, miopenTensorNDHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1, testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN3DTestCase>());
// bfp16
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNCKInferLarge2D_BFP16,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNOCLInferLarge2D_BFP16,
                         testing::Combine(testing::ValuesIn(Network2DLarge<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1, testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNOCLInferLarge3D_BFP16,
                         testing::Combine(testing::ValuesIn(Network3DBN<BN3DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCDHW, miopenTensorNDHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1, testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN3DTestCase>());

// fp32
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNInferSmall2D_FP32,
                         testing::Combine(testing::ValuesIn(Network2DLarge<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNInferLarge2D_FP32,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNInferSmall3D_FP32,
                         testing::Combine(testing::ValuesIn(Network3DBN<BN3DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCDHW, miopenTensorNDHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN3DTestCase>());
// fp64
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNInferSmall2D_FP64,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNInferLarge2D_FP64,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());
