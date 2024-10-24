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
#include "adaptiveavgpool.hpp"
#include "gtest/gtest.h"
using float16 = half_float::half;

// FORWARD TEST
using GPU_AdaptiveAvgPool_fwd_FP32  = AdaptiveAvgPoolTestFwd<float>;
using GPU_AdaptiveAvgPool_fwd_FP16  = AdaptiveAvgPoolTestFwd<float16>;
using GPU_AdaptiveAvgPool_fwd_BFP16 = AdaptiveAvgPoolTestFwd<bfloat16>;

TEST_P(GPU_AdaptiveAvgPool_fwd_FP32, AdaptiveAvgPoolTestFwd)
{
    RunTest();
    Verify();
};

TEST_P(GPU_AdaptiveAvgPool_fwd_FP16, AdaptiveAvgPoolTestFwd)
{
    RunTest();
    Verify();
};

TEST_P(GPU_AdaptiveAvgPool_fwd_BFP16, AdaptiveAvgPoolTestFwd)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_AdaptiveAvgPool_fwd_FP32,
                         testing::ValuesIn(AdaptiveAvgPoolTestConfigsFwdFp32()));
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_AdaptiveAvgPool_fwd_FP16,
                         testing::ValuesIn(AdaptiveAvgPoolTestConfigsFwdFp16()));
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_AdaptiveAvgPool_fwd_BFP16,
                         testing::ValuesIn(AdaptiveAvgPoolTestConfigsFwdBfp16()));

// BACKWARD TEST
using GPU_AdaptiveAvgPool_bwd_FP32  = AdaptiveAvgPoolTestBwd<float>;
using GPU_AdaptiveAvgPool_bwd_FP16  = AdaptiveAvgPoolTestBwd<float16>;
using GPU_AdaptiveAvgPool_bwd_BFP16 = AdaptiveAvgPoolTestBwd<bfloat16>;

TEST_P(GPU_AdaptiveAvgPool_bwd_FP32, AdaptiveAvgPoolTestBwd)
{
    RunTest();
    Verify();
};

TEST_P(GPU_AdaptiveAvgPool_bwd_FP16, AdaptiveAvgPoolTestBwd)
{
    RunTest();
    Verify();
};

TEST_P(GPU_AdaptiveAvgPool_bwd_BFP16, AdaptiveAvgPoolTestBwd)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_AdaptiveAvgPool_bwd_FP32,
                         testing::ValuesIn(AdaptiveAvgPoolTestConfigsBwdFp32()));
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_AdaptiveAvgPool_bwd_FP16,
                         testing::ValuesIn(AdaptiveAvgPoolTestConfigsBwdFp16()));
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_AdaptiveAvgPool_bwd_BFP16,
                         testing::ValuesIn(AdaptiveAvgPoolTestConfigsBwdBfp16()));
