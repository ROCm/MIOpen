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
#include "miopen/bfloat16.hpp"
#include <miopen/env.hpp>
#include "interpolate.hpp"

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace interpolate {

std::string GetFloatArg()
{
    const auto& tmp = miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG));
    if(tmp.empty())
    {
        return "";
    }
    return tmp;
}

struct InterpolateTestFloat : InterpolateTest<float>
{
};

struct InterpolateTestHalf : InterpolateTest<half>
{
};

struct InterpolateTestBFloat16 : InterpolateTest<bfloat16>
{
};

struct InterpolateTestFloatBwd : InterpolateTestBwd<float>
{
};

struct InterpolateTestHalfBwd : InterpolateTestBwd<half>
{
};

struct InterpolateTestBFloat16Bwd : InterpolateTestBwd<bfloat16>
{
};

} // namespace interpolate
using namespace interpolate;

// FORWARD TEST
TEST_P(InterpolateTestFloat, InterpolateTest)
{
    if((miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && GetFloatArg() == "--float") ||
       miopen::IsUnset(ENV(MIOPEN_TEST_ALL)))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(InterpolateTestHalf, InterpolateTest)
{
    if((miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && GetFloatArg() == "--half") ||
       miopen::IsUnset(ENV(MIOPEN_TEST_ALL)))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(InterpolateTestBFloat16, InterpolateTest)
{
    if((miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && GetFloatArg() == "--bfloat16") ||
       miopen::IsUnset(ENV(MIOPEN_TEST_ALL)))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(InterpolateTestSet,
                         InterpolateTestFloat,
                         testing::ValuesIn(InterpolateTestConfigs()));
INSTANTIATE_TEST_SUITE_P(InterpolateTestSet,
                         InterpolateTestHalf,
                         testing::ValuesIn(InterpolateTestConfigs()));
INSTANTIATE_TEST_SUITE_P(InterpolateTestSet,
                         InterpolateTestBFloat16,
                         testing::ValuesIn(InterpolateTestConfigs()));

// BACKWARD TEST
TEST_P(InterpolateTestFloatBwd, InterpolateTestBwd)
{
    if((miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && GetFloatArg() == "--float") ||
       miopen::IsUnset(ENV(MIOPEN_TEST_ALL)))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(InterpolateTestHalfBwd, InterpolateTestBwd)
{
    if((miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && GetFloatArg() == "--half") ||
       miopen::IsUnset(ENV(MIOPEN_TEST_ALL)))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(InterpolateTestBFloat16Bwd, InterpolateTestBwd)
{
    if((miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && GetFloatArg() == "--bfloat16") ||
       miopen::IsUnset(ENV(MIOPEN_TEST_ALL)))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(InterpolateTestSet,
                         InterpolateTestFloatBwd,
                         testing::ValuesIn(InterpolateTestConfigs()));
INSTANTIATE_TEST_SUITE_P(InterpolateTestSet,
                         InterpolateTestHalfBwd,
                         testing::ValuesIn(InterpolateTestConfigs()));
INSTANTIATE_TEST_SUITE_P(InterpolateTestSet,
                         InterpolateTestBFloat16Bwd,
                         testing::ValuesIn(InterpolateTestConfigs()));
