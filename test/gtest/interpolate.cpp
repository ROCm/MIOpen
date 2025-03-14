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
#include "interpolate.hpp"
using float16 = half_float::half;

// FORWARD TEST
using GPU_Interpolate_fwd_FP32  = InterpolateTestFwd<float>;
using GPU_Interpolate_fwd_FP16  = InterpolateTestFwd<float16>;
using GPU_Interpolate_fwd_BFP16 = InterpolateTestFwd<bfloat16>;

TEST_P(GPU_Interpolate_fwd_FP32, InterpolateTest)
{
    RunTest();
    Verify();
};

TEST_P(GPU_Interpolate_fwd_FP16, InterpolateTest)
{
    RunTest();
    Verify();
};

TEST_P(GPU_Interpolate_fwd_BFP16, InterpolateTest)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Interpolate_fwd_FP32,
                         testing::ValuesIn(InterpolateTestFwdConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Interpolate_fwd_FP16,
                         testing::ValuesIn(InterpolateTestFwdConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Interpolate_fwd_BFP16,
                         testing::ValuesIn(InterpolateTestFwdConfigs()));

// BACKWARD TEST
using GPU_Interpolate_bwd_FP32  = InterpolateTestBwd<float>;
using GPU_Interpolate_bwd_FP16  = InterpolateTestBwd<float16>;
using GPU_Interpolate_bwd_BFP16 = InterpolateTestBwd<bfloat16>;

TEST_P(GPU_Interpolate_bwd_FP32, InterpolateTestBwd)
{
    RunTest();
    Verify();
};

TEST_P(GPU_Interpolate_bwd_FP16, InterpolateTestBwd)
{
    RunTest();
    Verify();
};

TEST_P(GPU_Interpolate_bwd_BFP16, InterpolateTestBwd)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Interpolate_bwd_FP32,
                         testing::ValuesIn(InterpolateTestBwdConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Interpolate_bwd_FP16,
                         testing::ValuesIn(InterpolateTestBwdConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Interpolate_bwd_BFP16,
                         testing::ValuesIn(InterpolateTestBwdConfigs()));
