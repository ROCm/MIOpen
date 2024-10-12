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

#include "rope.hpp"

namespace rope {

struct GPU_RoPEFwdTest_FP32 : RoPEFwdTest<float>
{
};

struct GPU_RoPEFwdTest_FP16 : RoPEFwdTest<half_float::half>
{
};

struct GPU_RoPEFwdTest_BFP16 : RoPEFwdTest<bfloat16>
{
};

struct GPU_RoPEBwdTest_FP32 : RoPEBwdTest<float>
{
};

struct GPU_RoPEBwdTest_FP16 : RoPEBwdTest<half_float::half>
{
};

struct GPU_RoPEBwdTest_BFP16 : RoPEBwdTest<bfloat16>
{
};

} // namespace rope
using namespace rope;

TEST_P(GPU_RoPEFwdTest_FP32, RoPEFwdTest)
{
    RunTest();
    Verify();
};

TEST_P(GPU_RoPEFwdTest_FP16, RoPEFwdTest)
{
    RunTest();
    Verify();
};

TEST_P(GPU_RoPEFwdTest_BFP16, RoPEFwdTest)
{
    RunTest();
    Verify();
};

TEST_P(GPU_RoPEBwdTest_FP32, RoPEBwdTest)
{
    RunTest();
    Verify();
};

TEST_P(GPU_RoPEBwdTest_FP16, RoPEBwdTest)
{
    RunTest();
    Verify();
};

TEST_P(GPU_RoPEBwdTest_BFP16, RoPEBwdTest)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Full, GPU_RoPEFwdTest_FP32, testing::ValuesIn(RoPETestConfigs()));
INSTANTIATE_TEST_SUITE_P(Full, GPU_RoPEFwdTest_FP16, testing::ValuesIn(RoPETestConfigs()));
INSTANTIATE_TEST_SUITE_P(Full, GPU_RoPEFwdTest_BFP16, testing::ValuesIn(RoPETestConfigs()));
INSTANTIATE_TEST_SUITE_P(Full, GPU_RoPEBwdTest_FP32, testing::ValuesIn(RoPETestConfigs()));
INSTANTIATE_TEST_SUITE_P(Full, GPU_RoPEBwdTest_FP16, testing::ValuesIn(RoPETestConfigs()));
INSTANTIATE_TEST_SUITE_P(Full, GPU_RoPEBwdTest_BFP16, testing::ValuesIn(RoPETestConfigs()));
