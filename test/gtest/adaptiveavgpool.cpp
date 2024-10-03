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
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace adaptiveavgpool {

std::string GetFloatArg()
{
    const auto& tmp = env::value(MIOPEN_TEST_FLOAT_ARG);
    if(tmp.empty())
    {
        return "";
    }
    return tmp;
}

struct GPU_AdaptiveAvgpool_fwd_FP32 : AdaptiveAvgPoolTestFwd<float>
{
};

struct GPU_AdaptiveAvgpool_fwd_FP16 : AdaptiveAvgPoolTestFwd<half>
{
};

struct GPU_AdaptiveAvgpool_fwd_BFP16 : AdaptiveAvgPoolTestFwd<bfloat16>
{
};

struct GPU_AdaptiveAvgpool_bwd_FP32 : AdaptiveAvgPoolTestBwd<float>
{
};

struct GPU_AdaptiveAvgpool_bwd_FP16 : AdaptiveAvgPoolTestBwd<half>
{
};

struct GPU_AdaptiveAvgpool_bwd_BFP16 : AdaptiveAvgPoolTestBwd<bfloat16>
{
};

} // namespace adaptiveavgpool
using namespace adaptiveavgpool;

// FORWARD TEST
TEST_P(GPU_AdaptiveAvgpool_fwd_FP32, AdaptiveAvgPoolTestFwd)
{
    if(!MIOPEN_TEST_ALL ||
       (env::enabled(MIOPEN_TEST_ALL) && env::value(MIOPEN_TEST_FLOAT_ARG) == "--float"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(GPU_AdaptiveAvgpool_fwd_FP16, AdaptiveAvgPoolTestFwd)
{
    if(!MIOPEN_TEST_ALL ||
       (env::enabled(MIOPEN_TEST_ALL) && env::value(MIOPEN_TEST_FLOAT_ARG) == "--half"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(GPU_AdaptiveAvgpool_fwd_BFP16, AdaptiveAvgPoolTestFwd)
{
    if(!MIOPEN_TEST_ALL ||
       (env::enabled(MIOPEN_TEST_ALL) && env::value(MIOPEN_TEST_FLOAT_ARG) == "--bfloat16"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_AdaptiveAvgpool_fwd_FP32,
                         testing::ValuesIn(AdaptiveAvgPoolTestConfigsFwdFp32()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_AdaptiveAvgpool_fwd_FP16,
                         testing::ValuesIn(AdaptiveAvgPoolTestConfigsFwdFp16()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_AdaptiveAvgpool_fwd_BFP16,
                         testing::ValuesIn(AdaptiveAvgPoolTestConfigsFwdBfp16()));

// BACKWARD TEST
TEST_P(GPU_AdaptiveAvgpool_bwd_FP32, AdaptiveAvgPoolTestBwd)
{
    if(!MIOPEN_TEST_ALL ||
       (env::enabled(MIOPEN_TEST_ALL) && env::value(MIOPEN_TEST_FLOAT_ARG) == "--float"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(GPU_AdaptiveAvgpool_bwd_FP16, AdaptiveAvgPoolTestBwd)
{
    if(!MIOPEN_TEST_ALL ||
       (env::enabled(MIOPEN_TEST_ALL) && env::value(MIOPEN_TEST_FLOAT_ARG) == "--half"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(GPU_AdaptiveAvgpool_bwd_BFP16, AdaptiveAvgPoolTestBwd)
{
    if(!MIOPEN_TEST_ALL ||
       (env::enabled(MIOPEN_TEST_ALL) && env::value(MIOPEN_TEST_FLOAT_ARG) == "--bfloat16"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_AdaptiveAvgpool_bwd_FP32,
                         testing::ValuesIn(AdaptiveAvgPoolTestConfigsBwdFp32()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_AdaptiveAvgpool_bwd_FP16,
                         testing::ValuesIn(AdaptiveAvgPoolTestConfigsBwdFp16()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_AdaptiveAvgpool_bwd_BFP16,
                         testing::ValuesIn(AdaptiveAvgPoolTestConfigsBwdBfp16()));
