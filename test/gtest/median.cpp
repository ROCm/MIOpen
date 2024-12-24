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
#include "median.hpp"

using float16 = half_float::half;

// FORWARD TEST
using GPU_Median_fwd_FP32  = MedianTestFwd<float>;
using GPU_Median_fwd_FP16  = MedianTestFwd<float16>;
using GPU_Median_fwd_BFP16 = MedianTestFwd<bfloat16>;

TEST_P(GPU_Median_fwd_FP32, MedianTestFwd)
{
    RunTest();
    Verify();
};

TEST_P(GPU_Median_fwd_FP16, MedianTestFwd)
{
    RunTest();
    Verify();
};

TEST_P(GPU_Median_fwd_BFP16, MedianTestFwd)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Median_fwd_FP32, testing::ValuesIn(MedianTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Median_fwd_FP16, testing::ValuesIn(MedianTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Median_fwd_BFP16, testing::ValuesIn(MedianTestConfigs()));

INSTANTIATE_TEST_SUITE_P(Full, GPU_Median_fwd_FP32, testing::ValuesIn(MedianTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Full, GPU_Median_fwd_FP16, testing::ValuesIn(MedianTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Full, GPU_Median_fwd_BFP16, testing::ValuesIn(MedianTestConfigs()));

// BACKWARD TEST
using GPU_Median_bwd_FP32  = MedianTestBwd<float>;
using GPU_Median_bwd_FP16  = MedianTestBwd<float16>;
using GPU_Median_bwd_BFP16 = MedianTestBwd<bfloat16>;

TEST_P(GPU_Median_bwd_FP32, MedianTestBwd)
{
    RunTest();
    Verify();
};

TEST_P(GPU_Median_bwd_FP16, MedianTestBwd)
{
    RunTest();
    Verify();
};

TEST_P(GPU_Median_bwd_BFP16, MedianTestBwd)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Median_bwd_FP32, testing::ValuesIn(MedianTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Median_bwd_FP16, testing::ValuesIn(MedianTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Median_bwd_BFP16, testing::ValuesIn(MedianTestConfigs()));

INSTANTIATE_TEST_SUITE_P(Full, GPU_Median_bwd_FP32, testing::ValuesIn(MedianTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Full, GPU_Median_bwd_FP16, testing::ValuesIn(MedianTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Full, GPU_Median_bwd_BFP16, testing::ValuesIn(MedianTestConfigs()));
