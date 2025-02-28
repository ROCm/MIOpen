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

// XDataType
// YDataYype
// ScaleDataType
// BiasDataType
// RunSaveDataType
// AccDataType

struct GPU_BNCKFWDTrainSerialRun2D_FP16 : BNFwdTrainTest<half_float::half,
                                                         half_float::half,
                                                         half_float::half,
                                                         half_float::half,
                                                         float,
                                                         double,
                                                         BN2DTestCase>
{
};

struct GPU_BNOCLFWDTrainSerialRun2D_FP16
    : BNFwdTrainTest<half_float::half, half_float::half, float, float, float, double, BN2DTestCase>
{
};

struct GPU_BNOCLFWDTrainSerialRun3D_FP16
    : BNFwdTrainTest<half_float::half, half_float::half, float, float, float, double, BN3DTestCase>
{
};

struct GPU_BNCKFWDTrainSerialRun2D_BFP16
    : BNFwdTrainTest<bfloat16, bfloat16, bfloat16, bfloat16, float, double, BN2DTestCase>
{
};

struct GPU_BNOCLFWDTrainSerialRun2D_BFP16
    : BNFwdTrainTest<bfloat16, bfloat16, float, float, float, double, BN2DTestCase>
{
};

struct GPU_BNOCLFWDTrainSerialRun3D_BFP16
    : BNFwdTrainTest<bfloat16, bfloat16, float, float, float, double, BN3DTestCase>
{
};

struct GPU_BNFWDTrainSerialRun2D_FP32
    : BNFwdTrainTest<float, float, float, float, float, double, BN2DTestCase>
{
};

struct GPU_BNFWDTrainSerialRun3D_FP32
    : BNFwdTrainTest<float, float, float, float, float, double, BN3DTestCase>
{
};

struct GPU_BNFWDTrainSerialRun2D_FP64
    : BNFwdTrainTest<double, double, double, double, double, double, BN2DTestCase>
{
};

struct GPU_BNFWDTrainSerialRun3D_FP64
    : BNFwdTrainTest<double, double, double, double, double, double, BN3DTestCase>
{
};

// fp16
TEST_P(GPU_BNCKFWDTrainSerialRun2D_FP16, DISABLED_BnV2SerialRunFWD_TrainCKfp16) {}
TEST_P(GPU_BNOCLFWDTrainSerialRun2D_FP16, BnV2SerialRunFWD_TrainOCLfp16) {}
TEST_P(GPU_BNOCLFWDTrainSerialRun3D_FP16, BnV2SerialRunFWD_TrainOCL_3D_fp16) {}

// bfp16
TEST_P(GPU_BNCKFWDTrainSerialRun2D_BFP16, DISABLED_BnV2SerialRunFWD_TrainCKbfp16) {}
TEST_P(GPU_BNOCLFWDTrainSerialRun2D_BFP16, BnV2SerialRunFWD_TrainOCLbfp16) {}
TEST_P(GPU_BNOCLFWDTrainSerialRun3D_BFP16, BnV2SerialRunFWD_TrainOCL_3Dbfp16) {}

// fp32 (float)
TEST_P(GPU_BNFWDTrainSerialRun2D_FP32, BnV1SerialRunFWD_Train2Dfp32) {}
TEST_P(GPU_BNFWDTrainSerialRun3D_FP32, BnV1SerialRunFWD_Train2Dfp32) {}

// fp64
TEST_P(GPU_BNFWDTrainSerialRun2D_FP64, DISABLED_BnV1SerialRunFWD_Train3Dfp64) {}
TEST_P(GPU_BNFWDTrainSerialRun3D_FP64, DISABLED_BnV2SerialRunFWD_Train3Dfp64) {}

// fp16

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNCKFWDTrainSerialRun2D_FP16,
                         testing::Combine(testing::ValuesIn(Network2DSerialCase<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNOCLFWDTrainSerialRun2D_FP16,
                         testing::Combine(testing::ValuesIn(Network2DSerialCase<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1, testBNAPIV2})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNOCLFWDTrainSerialRun3D_FP16,
                         testing::Combine(testing::ValuesIn(Network3DSerialCase<BN3DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCDHW, miopenTensorNDHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1, testBNAPIV2})),
                         TestNameGenerator<BN3DTestCase>());

// bfp16
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNCKFWDTrainSerialRun2D_BFP16,
                         testing::Combine(testing::ValuesIn(Network2DSerialCase<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNOCLFWDTrainSerialRun2D_BFP16,
                         testing::Combine(testing::ValuesIn(Network2DSerialCase<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1, testBNAPIV2})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNOCLFWDTrainSerialRun3D_BFP16,
                         testing::Combine(testing::ValuesIn(Network3DSerialCase<BN3DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCDHW, miopenTensorNDHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1, testBNAPIV2})),
                         TestNameGenerator<BN3DTestCase>());
// fp32
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNFWDTrainSerialRun2D_FP32,
                         testing::Combine(testing::ValuesIn(Network2DSerialCase<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNFWDTrainSerialRun3D_FP32,
                         testing::Combine(testing::ValuesIn(Network3DSerialCase<BN3DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCDHW, miopenTensorNDHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1, testBNAPIV2})),
                         TestNameGenerator<BN3DTestCase>());

// fp64
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNFWDTrainSerialRun2D_FP64,
                         testing::Combine(testing::ValuesIn(Network2DSerialCase<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV2})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNFWDTrainSerialRun3D_FP64,
                         testing::Combine(testing::ValuesIn(Network3DSerialCase<BN3DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCDHW, miopenTensorNDHWC}),
                                          testing::ValuesIn({miopenBNSpatial,
                                                             miopenBNPerActivation}),
                                          testing::ValuesIn({testBNAPIV1, testBNAPIV2})),
                         TestNameGenerator<BN3DTestCase>());
