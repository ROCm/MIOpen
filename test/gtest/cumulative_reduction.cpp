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

#include "cumulative_reduction.hpp"
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace cumulative_reduction {

bool CheckFloatArg(std::string arg)
{
    if(!MIOPEN_TEST_ALL ||
       (env::enabled(MIOPEN_TEST_ALL) && (env::value(MIOPEN_TEST_FLOAT_ARG) == arg)))
    {
        return true;
    }
    return false;
}

struct CumulativeReductionTestFloat : CumulativeReductionTest<float>
{
};

struct CumulativeReductionTestHalf : CumulativeReductionTest<half>
{
};

struct CumulativeReductionTestBfloat16 : CumulativeReductionTest<bfloat16>
{
};

} // namespace cumulative_reduction
using namespace cumulative_reduction;

TEST_P(CumulativeReductionTestFloat, CumulativeReductionTest)
{
    if(CheckFloatArg("--float"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(CumulativeReductionTestHalf, CumulativeReductionTest)
{
    if(CheckFloatArg("--half"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(CumulativeReductionTestBfloat16, CumulativeReductionTest)
{
    if(CheckFloatArg("--bfloat16"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(CumulativeReductionTestSet,
                         CumulativeReductionTestFloat,
                         testing::ValuesIn(CumulativeReductionTestConfigs()));
INSTANTIATE_TEST_SUITE_P(CumulativeReductionTestSet,
                         CumulativeReductionTestHalf,
                         testing::ValuesIn(CumulativeReductionTestConfigs()));
INSTANTIATE_TEST_SUITE_P(CumulativeReductionTestSet,
                         CumulativeReductionTestBfloat16,
                         testing::ValuesIn(CumulativeReductionTestConfigs()));
