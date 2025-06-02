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
   typename ScaleDataType,
   typename DscaleDbiasDataType,
   typename MeanVarDataType,
   typename AccDataType> */

struct GPU_BNBWDSmall_FP32 : BNBwdTest<half_float::half,
                                       float,
                                       float,
                                       float,
                                       half_float::half,
                                       float,
                                       double,
                                       BN2DTestCase>
{
};

struct GPU_BNOCLBWDLarge2D_FP16 : BNBwdTest<half_float::half,
                                            half_float::half,
                                            half_float::half,
                                            float,
                                            float,
                                            float,
                                            double,
                                            BN2DTestCase>
{
};

struct GPU_BNOCLBWDLarge3D_FP16 : BNBwdTest<half_float::half,
                                            half_float::half,
                                            half_float::half,
                                            float,
                                            float,
                                            float,
                                            double,
                                            BN3DTestCase>
{
};

struct GPU_BNCKBWDLarge2D_BFP16
    : BNBwdTest<bfloat16, float, float, float, bfloat16, float, double, BN2DTestCase>
{
};

struct GPU_BNOCLBWDLarge2D_BFP16
    : BNBwdTest<bfloat16, bfloat16, bfloat16, float, float, float, double, BN2DTestCase>
{
};

struct GPU_BNOCLBWDLarge3D_BFP16
    : BNBwdTest<bfloat16, bfloat16, bfloat16, float, float, float, double, BN3DTestCase>
{
};

struct GPU_BNBWDSmall2D_FP32
    : BNBwdTest<float, float, float, float, float, float, double, BN2DTestCase>
{
};

struct GPU_BNBWDLarge2D_FP32
    : BNBwdTest<float, float, float, float, float, float, double, BN2DTestCase>
{
};

struct GPU_BNBWDLarge3D_FP32
    : BNBwdTest<float, float, float, float, float, float, double, BN3DTestCase>
{
};

struct GPU_BNBWDSmall2D_FP64
    : BNBwdTest<double, double, double, double, double, double, double, BN2DTestCase>
{
};

struct GPU_BNBWDLarge2D_FP64
    : BNBwdTest<double, double, double, double, double, double, double, BN2DTestCase>
{
};

// fp16
TEST_P(GPU_BNBWDSmall_FP32, DISABLED_BnV2LargeBWDCK2D_fp16) {}
TEST_P(GPU_BNOCLBWDLarge2D_FP16, BnV2LargeBWDOCL2D_fp16) {}
TEST_P(GPU_BNOCLBWDLarge3D_FP16, BnV2LargeBWDOCL3D_fp16) {}

// bfp16
TEST_P(GPU_BNCKBWDLarge2D_BFP16, DISABLED_BnV2LargeBWDCKbfp16_2D) {}
TEST_P(GPU_BNOCLBWDLarge2D_BFP16, BnV2LargeBWDOCLbfp16_2D) {}
TEST_P(GPU_BNOCLBWDLarge3D_BFP16, BnV2LargeBWDOCLbfp16_3D) {}

// fp32 (float)
TEST_P(GPU_BNBWDSmall2D_FP32, BnV1SmallBWDCKfp32_2D) {}
TEST_P(GPU_BNBWDLarge2D_FP32, BnV2LargeBWDCKfp32_2D) {}
TEST_P(GPU_BNBWDLarge3D_FP32, BnV2LargeBWDCKfp32_3D) {}

// fp64
TEST_P(GPU_BNBWDSmall2D_FP64, DISABLED_BnV1SmallBWDCKfp64_2D) {}
TEST_P(GPU_BNBWDLarge2D_FP64, DISABLED_BnV2LargeBWDCKfp64_2D) {}

// fp16
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNBWDSmall_FP32,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNOCLBWDLarge2D_FP16,
                         testing::Combine(testing::ValuesIn(Network2DLarge<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNOCLBWDLarge3D_FP16,
                         testing::Combine(testing::ValuesIn(Network3DBN<BN3DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCDHW, miopenTensorNDHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN3DTestCase>());

// bfp16
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNCKBWDLarge2D_BFP16,
                         testing::Combine(testing::ValuesIn(Network2DLarge<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNOCLBWDLarge2D_BFP16,
                         testing::Combine(testing::ValuesIn(Network2DLarge<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNOCLBWDLarge3D_BFP16,
                         testing::Combine(testing::ValuesIn(Network3DBN<BN3DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCDHW, miopenTensorNDHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN3DTestCase>());

// fp32
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNBWDSmall2D_FP32,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNBWDLarge2D_FP32,
                         testing::Combine(testing::ValuesIn(Network2DLarge<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNBWDLarge3D_FP32,
                         testing::Combine(testing::ValuesIn(Network3DBN<BN3DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCDHW, miopenTensorNDHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN3DTestCase>());
// fp64
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNBWDSmall2D_FP64,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNBWDLarge2D_FP64,
                         testing::Combine(testing::ValuesIn(Network2DLarge<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());
