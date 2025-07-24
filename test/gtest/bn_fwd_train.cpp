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

// XDataType
// YDataYype
// ScaleDataType
// BiasDataType
// RunSaveDataType
// AccDataType

struct GPU_BNCKFWDTrainLarge2D_FP16 : BNFwdTrainTest<half_float::half,
                                                     half_float::half,
                                                     half_float::half,
                                                     half_float::half,
                                                     float,
                                                     double,
                                                     BN2DTestCase>
{
};

struct GPU_BNOCLFWDTrainLarge2D_FP16
    : BNFwdTrainTest<half_float::half, half_float::half, float, float, float, double, BN2DTestCase>
{
};

struct GPU_BNOCLFWDTrainLarge3D_FP16
    : BNFwdTrainTest<half_float::half, half_float::half, float, float, float, double, BN3DTestCase>
{
};

struct GPU_BNCKFWDTrainLarge2D_BFP16
    : BNFwdTrainTest<bfloat16, bfloat16, bfloat16, bfloat16, float, double, BN2DTestCase>
{
};

struct GPU_BNOCLFWDTrainLarge2D_BFP16
    : BNFwdTrainTest<bfloat16, bfloat16, float, float, float, double, BN2DTestCase>
{
};

struct GPU_BNOCLFWDTrainLarge3D_BFP16
    : BNFwdTrainTest<bfloat16, bfloat16, float, float, float, double, BN3DTestCase>
{
};

struct GPU_BNFWDTrainSmall2D_FP32
    : BNFwdTrainTest<float, float, float, float, float, double, BN2DTestCase>
{
};

struct GPU_BNFWDTrainSmall3D_FP32
    : BNFwdTrainTest<float, float, float, float, float, double, BN3DTestCase>
{
};

struct GPU_BNFWDTrainLarge2D_FP32
    : BNFwdTrainTest<float, float, float, float, float, double, BN2DTestCase>
{
};

struct GPU_BNFWDTrainSmall2D_FP64
    : BNFwdTrainTest<double, double, double, double, double, double, BN2DTestCase>
{
};

struct GPU_BNFWDTrainLarge2D_FP64
    : BNFwdTrainTest<double, double, double, double, double, double, BN2DTestCase>
{
};

struct GPU_BNFWDTrainSmall3D_FP64
    : BNFwdTrainTest<double, double, double, double, double, double, BN3DTestCase>
{
};

// fp16
TEST_P(GPU_BNCKFWDTrainLarge2D_FP16, DISABLED_BnV2LargeFWD_TrainCKfp16) {}
TEST_P(GPU_BNOCLFWDTrainLarge2D_FP16, BnV2LargeFWD_TrainOCLfp16) {}
TEST_P(GPU_BNOCLFWDTrainLarge3D_FP16, BnV2LargeFWD_TrainOCL_3D_fp16) {}

// bfp16
TEST_P(GPU_BNCKFWDTrainLarge2D_BFP16, DISABLED_BnV2LargeFWD_TrainCKbfp16) {}
TEST_P(GPU_BNOCLFWDTrainLarge2D_BFP16, BnV2LargeFWD_TrainOCLbfp16) {}
TEST_P(GPU_BNOCLFWDTrainLarge3D_BFP16, BnV2LargeFWD_TrainOCL_3Dbfp16) {}

// fp32 (float)
TEST_P(GPU_BNFWDTrainSmall2D_FP32, BnV1SmallFWD_TrainCKfp32) {}
TEST_P(GPU_BNFWDTrainLarge2D_FP32, BnV2LargeFWD_TrainCKfp32) {}
TEST_P(GPU_BNFWDTrainSmall3D_FP32, BnV1SmallFWD_TrainC_3DKfp32) {}

// fp64
TEST_P(GPU_BNFWDTrainSmall2D_FP64, DISABLED_BnV1SmallFWD_TrainCKfp64) {}
TEST_P(GPU_BNFWDTrainLarge2D_FP64, DISABLED_BnV2Large2DFWD_TrainCKfp64) {}
TEST_P(GPU_BNFWDTrainSmall3D_FP64, DISABLED_BnV2Large3DFWD_TrainCKfp64) {}

// fp16

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNCKFWDTrainLarge2D_FP16,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNOCLFWDTrainLarge2D_FP16,
                         testing::Combine(testing::ValuesIn(Network2DLarge<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1, testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNOCLFWDTrainLarge3D_FP16,
                         testing::Combine(testing::ValuesIn(Network3DBN<BN3DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCDHW, miopenTensorNDHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1, testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN3DTestCase>());

// bfp16
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNCKFWDTrainLarge2D_BFP16,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNOCLFWDTrainLarge2D_BFP16,
                         testing::Combine(testing::ValuesIn(Network2DLarge<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1, testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNOCLFWDTrainLarge3D_BFP16,
                         testing::Combine(testing::ValuesIn(Network3DBN<BN3DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCDHW, miopenTensorNDHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1, testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN3DTestCase>());
// fp32
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNFWDTrainSmall2D_FP32,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNFWDTrainLarge2D_FP32,
                         testing::Combine(testing::ValuesIn(Network2DLarge<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNFWDTrainSmall3D_FP32,
                         testing::Combine(testing::ValuesIn(Network3DBN<BN3DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCDHW, miopenTensorNDHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1, testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN3DTestCase>());

// fp64
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNFWDTrainSmall2D_FP64,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNFWDTrainLarge2D_FP64,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNFWDTrainSmall3D_FP64,
                         testing::Combine(testing::ValuesIn(Network3DBN<BN3DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCDHW, miopenTensorNDHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1, testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN3DTestCase>());
