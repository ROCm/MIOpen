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

#include "logsumexp.hpp"
#include <miopen/bfloat16.hpp>

namespace logsumexp {

using GPU_LogSumExp_fwd_FP32  = LogSumExpForwardTest<float>;
using GPU_LogSumExp_fwd_FP16  = LogSumExpForwardTest<half_float::half>;
using GPU_LogSumExp_fwd_BFP16 = LogSumExpForwardTest<bfloat16>;
using GPU_LogSumExp_bwd_FP32  = LogSumExpBackwardTest<float>;
using GPU_LogSumExp_bwd_FP16  = LogSumExpBackwardTest<half_float::half>;
using GPU_LogSumExp_bwd_BFP16 = LogSumExpBackwardTest<bfloat16>;

} // namespace logsumexp
using namespace logsumexp;

TEST_P(GPU_LogSumExp_fwd_FP32, Test)
{
    RunTest();
    Verify();
}

TEST_P(GPU_LogSumExp_fwd_FP16, Test)
{
    RunTest();
    Verify();
}

TEST_P(GPU_LogSumExp_fwd_BFP16, Test)
{
    RunTest();
    Verify();
}

TEST_P(GPU_LogSumExp_bwd_FP32, Test)
{
    RunTest();
    Verify();
}

TEST_P(GPU_LogSumExp_bwd_FP16, Test)
{
    RunTest();
    Verify();
}

TEST_P(GPU_LogSumExp_bwd_BFP16, Test)
{
    RunTest();
    Verify();
}

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_LogSumExp_fwd_FP32, testing::ValuesIn(LogSumExpTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_LogSumExp_fwd_FP16, testing::ValuesIn(LogSumExpTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_LogSumExp_fwd_BFP16, testing::ValuesIn(LogSumExpTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_LogSumExp_bwd_FP32, testing::ValuesIn(LogSumExpTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_LogSumExp_bwd_FP16, testing::ValuesIn(LogSumExpTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_LogSumExp_bwd_BFP16, testing::ValuesIn(LogSumExpTestConfigs()));
