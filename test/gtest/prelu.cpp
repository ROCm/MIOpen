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

#include "prelu.hpp"

namespace prelu {

struct GPU_PReLU_bwd_FP32 : PReLUTest<float>
{
};

struct GPU_PReLU_bwd_FP16 : PReLUTest<half>
{
};

struct GPU_PReLU_bwd_BFP16 : PReLUTest<bfloat16>
{
};

} // namespace prelu
using namespace prelu;

TEST_P(GPU_PReLU_bwd_FP32, Test)
{
    RunTest();
    Verify();
};

TEST_P(GPU_PReLU_bwd_FP16, Test)
{
    RunTest();
    Verify();
};

TEST_P(GPU_PReLU_bwd_BFP16, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_PReLU_bwd_FP32, testing::ValuesIn(PReLUSmokeTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_PReLU_bwd_FP16, testing::ValuesIn(PReLUSmokeTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_PReLU_bwd_BFP16, testing::ValuesIn(PReLUSmokeTestConfigs()));

INSTANTIATE_TEST_SUITE_P(Full, GPU_PReLU_bwd_FP32, testing::ValuesIn(PReLUFullTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Full, GPU_PReLU_bwd_FP16, testing::ValuesIn(PReLUFullTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Full, GPU_PReLU_bwd_BFP16, testing::ValuesIn(PReLUFullTestConfigs()));

INSTANTIATE_TEST_SUITE_P(Perf, GPU_PReLU_bwd_FP32, testing::ValuesIn(PReLUPerfTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Perf, GPU_PReLU_bwd_FP16, testing::ValuesIn(PReLUPerfTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Perf, GPU_PReLU_bwd_BFP16, testing::ValuesIn(PReLUPerfTestConfigs()));
