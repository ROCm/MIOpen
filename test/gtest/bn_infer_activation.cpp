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

struct GPU_BNOCLInferLargeFusedActivation2D_FP16
    : BNInferTest<half_float::half, half_float::half, float, float, float, double, BN2DTestCase>
{
};

struct GPU_BNOCLInferLargeFusedActivation2D_BFP16
    : BNInferTest<bfloat16, bfloat16, float, float, float, double, BN2DTestCase>
{
};

// fp16
TEST_P(GPU_BNOCLInferLargeFusedActivation2D_FP16, BnV2LargeInferOCLfp16_2D) {}

// bfp16
TEST_P(GPU_BNOCLInferLargeFusedActivation2D_BFP16, BnV2LargeInferOCLbfp16_2D) {}

// fp16
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNOCLInferLargeFusedActivation2D_FP16,
                         testing::Combine(testing::ValuesIn(Network2DLarge<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial}),
                                          testing::ValuesIn({testBNAPIV1}),
                                          testing::ValuesIn({miopenActivationCLAMP})),
                         TestNameGenerator<BN2DTestCase>());

// bfp16
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNOCLInferLargeFusedActivation2D_BFP16,
                         testing::Combine(testing::ValuesIn(Network2DLarge<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial}),
                                          testing::ValuesIn({testBNAPIV1}),
                                          testing::ValuesIn({miopenActivationCLAMP})),
                         TestNameGenerator<BN2DTestCase>());
