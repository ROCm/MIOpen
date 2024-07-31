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

#include "reduceextreme.hpp"
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace reduceextreme {

std::string GetFloatArg()
{
    const auto& tmp = env::value(MIOPEN_TEST_FLOAT_ARG);
    if(tmp.empty())
    {
        return "";
    }
    return tmp;
}

struct ReduceExtremeTestFloat : ReduceExtremeTest<float>
{
};

struct ReduceExtremeTestHalf : ReduceExtremeTest<half_float::half>
{
};

struct ReduceExtremeTestBFloat16 : ReduceExtremeTest<bfloat16>
{
};

} // namespace reduceextreme
using namespace reduceextreme;

TEST_P(ReduceExtremeTestFloat, ReduceExtremeTestFw)
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

TEST_P(ReduceExtremeTestHalf, ReduceExtremeTestFw)
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

TEST_P(ReduceExtremeTestBFloat16, ReduceExtremeTestFw)
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

INSTANTIATE_TEST_SUITE_P(ReduceExtremeTestSetMIN,
                         ReduceExtremeTestFloat,
                         testing::ValuesIn(ReduceExtremeTestConfigs(MIOPEN_REDUCE_EXTREME_MIN)));
INSTANTIATE_TEST_SUITE_P(ReduceExtremeTestSetMAX,
                         ReduceExtremeTestFloat,
                         testing::ValuesIn(ReduceExtremeTestConfigs(MIOPEN_REDUCE_EXTREME_MAX)));
INSTANTIATE_TEST_SUITE_P(ReduceExtremeTestSetARGMIN,
                         ReduceExtremeTestFloat,
                         testing::ValuesIn(ReduceExtremeTestConfigs(MIOPEN_REDUCE_EXTREME_ARGMIN)));
INSTANTIATE_TEST_SUITE_P(ReduceExtremeTestSetARGMAX,
                         ReduceExtremeTestFloat,
                         testing::ValuesIn(ReduceExtremeTestConfigs(MIOPEN_REDUCE_EXTREME_ARGMAX)));
INSTANTIATE_TEST_SUITE_P(ReduceExtremeTestSetMIN,
                         ReduceExtremeTestHalf,
                         testing::ValuesIn(ReduceExtremeTestConfigs(MIOPEN_REDUCE_EXTREME_MIN)));
INSTANTIATE_TEST_SUITE_P(ReduceExtremeTestSetMAX,
                         ReduceExtremeTestHalf,
                         testing::ValuesIn(ReduceExtremeTestConfigs(MIOPEN_REDUCE_EXTREME_MAX)));
INSTANTIATE_TEST_SUITE_P(ReduceExtremeTestSetARGMIN,
                         ReduceExtremeTestHalf,
                         testing::ValuesIn(ReduceExtremeTestConfigs(MIOPEN_REDUCE_EXTREME_ARGMIN)));
INSTANTIATE_TEST_SUITE_P(ReduceExtremeTestSetARGMAX,
                         ReduceExtremeTestHalf,
                         testing::ValuesIn(ReduceExtremeTestConfigs(MIOPEN_REDUCE_EXTREME_ARGMAX)));
INSTANTIATE_TEST_SUITE_P(ReduceExtremeTestSetMIN,
                         ReduceExtremeTestBFloat16,
                         testing::ValuesIn(ReduceExtremeTestConfigs(MIOPEN_REDUCE_EXTREME_MIN)));
INSTANTIATE_TEST_SUITE_P(ReduceExtremeTestSetMAX,
                         ReduceExtremeTestBFloat16,
                         testing::ValuesIn(ReduceExtremeTestConfigs(MIOPEN_REDUCE_EXTREME_MAX)));
INSTANTIATE_TEST_SUITE_P(ReduceExtremeTestSetARGMIN,
                         ReduceExtremeTestBFloat16,
                         testing::ValuesIn(ReduceExtremeTestConfigs(MIOPEN_REDUCE_EXTREME_ARGMIN)));
INSTANTIATE_TEST_SUITE_P(ReduceExtremeTestSetARGMAX,
                         ReduceExtremeTestBFloat16,
                         testing::ValuesIn(ReduceExtremeTestConfigs(MIOPEN_REDUCE_EXTREME_ARGMAX)));
