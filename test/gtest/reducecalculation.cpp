/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include "reducecalculation.hpp"

namespace reducecalculation {

struct GPU_ReduceCalculationTest_FP32 : ReduceCalculationTest<float>
{
};

struct GPU_ReduceCalculationTest_FP16 : ReduceCalculationTest<half_float::half>
{
};

struct GPU_ReduceCalculationTest_BFP16 : ReduceCalculationTest<bfloat16>
{
};

} // namespace reducecalculation
using namespace reducecalculation;

TEST_P(GPU_ReduceCalculationTest_FP32, ReduceCalculationTestFw)
{
    RunTest();
    Verify();
};

TEST_P(GPU_ReduceCalculationTest_FP16, ReduceCalculationTestFw)
{
    RunTest();
    Verify();
};

TEST_P(GPU_ReduceCalculationTest_BFP16, ReduceCalculationTestFw)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(
    FullSUM,
    GPU_ReduceCalculationTest_FP32,
    testing::ValuesIn(ReduceCalculationTestConfigs(MIOPEN_REDUCE_CALCULATION_SUM)));
INSTANTIATE_TEST_SUITE_P(
    FullPROD,
    GPU_ReduceCalculationTest_FP32,
    testing::ValuesIn(ReduceCalculationTestConfigs(MIOPEN_REDUCE_CALCULATION_PROD)));
INSTANTIATE_TEST_SUITE_P(
    FullSUM,
    GPU_ReduceCalculationTest_FP16,
    testing::ValuesIn(ReduceCalculationTestConfigs(MIOPEN_REDUCE_CALCULATION_SUM)));
INSTANTIATE_TEST_SUITE_P(
    FullPROD,
    GPU_ReduceCalculationTest_FP16,
    testing::ValuesIn(ReduceCalculationTestConfigs(MIOPEN_REDUCE_CALCULATION_PROD)));
INSTANTIATE_TEST_SUITE_P(
    FullSUM,
    GPU_ReduceCalculationTest_BFP16,
    testing::ValuesIn(ReduceCalculationTestConfigs(MIOPEN_REDUCE_CALCULATION_SUM)));
INSTANTIATE_TEST_SUITE_P(
    FullPROD,
    GPU_ReduceCalculationTest_BFP16,
    testing::ValuesIn(ReduceCalculationTestConfigs(MIOPEN_REDUCE_CALCULATION_PROD)));
