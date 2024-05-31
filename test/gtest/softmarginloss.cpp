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

#include "softmarginloss.hpp"
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace softmarginloss {

std::string GetFloatArg()
{
    const auto& tmp = miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG));
    if(tmp.empty())
    {
        return "";
    }
    return tmp;
}

struct SoftMarginLossUnreducedForwardTestFloat : SoftMarginLossUnreducedForwardTest<float>
{
};

struct SoftMarginLossUnreducedForwardTestHalf : SoftMarginLossUnreducedForwardTest<half_float::half>
{
};

struct SoftMarginLossUnreducedForwardTestBFloat16 : SoftMarginLossUnreducedForwardTest<bfloat16>
{
};

struct SoftMarginLossUnreducedBackwardTestFloat : SoftMarginLossUnreducedBackwardTest<float>
{
};

struct SoftMarginLossUnreducedBackwardTestHalf
    : SoftMarginLossUnreducedBackwardTest<half_float::half>
{
};

struct SoftMarginLossUnreducedBackwardTestBFloat16 : SoftMarginLossUnreducedBackwardTest<bfloat16>
{
};

struct SoftMarginLossReducedForwardTestFloat : SoftMarginLossReducedForwardTest<float>
{
};

struct SoftMarginLossReducedForwardTestHalf : SoftMarginLossReducedForwardTest<half_float::half>
{
};

struct SoftMarginLossReducedForwardTestBFloat16 : SoftMarginLossReducedForwardTest<bfloat16>
{
};

struct SoftMarginLossReducedBackwardTestFloat : SoftMarginLossReducedBackwardTest<float>
{
};

struct SoftMarginLossReducedBackwardTestHalf : SoftMarginLossReducedBackwardTest<half_float::half>
{
};

struct SoftMarginLossReducedBackwardTestBFloat16 : SoftMarginLossReducedBackwardTest<bfloat16>
{
};

} // namespace softmarginloss

using namespace softmarginloss;

TEST_P(SoftMarginLossUnreducedForwardTestFloat, )
{
    if(miopen::IsUnset(ENV(MIOPEN_TEST_ALL)) ||
       (miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--float")))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(SoftMarginLossUnreducedForwardTestHalf, )
{
    if(miopen::IsUnset(ENV(MIOPEN_TEST_ALL)) ||
       (miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--half")))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(SoftMarginLossUnreducedForwardTestBFloat16, )
{
    if(miopen::IsUnset(ENV(MIOPEN_TEST_ALL)) ||
       (miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--bfloat16")))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};
INSTANTIATE_TEST_SUITE_P(SoftMarginLossTestSet,
                         SoftMarginLossUnreducedForwardTestFloat,
                         testing::ValuesIn(SoftMarginLossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(SoftMarginLossTestSet,
                         SoftMarginLossUnreducedForwardTestHalf,
                         testing::ValuesIn(SoftMarginLossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(SoftMarginLossTestSet,
                         SoftMarginLossUnreducedForwardTestBFloat16,
                         testing::ValuesIn(SoftMarginLossTestConfigs()));

TEST_P(SoftMarginLossUnreducedBackwardTestFloat, )
{
    if(miopen::IsUnset(ENV(MIOPEN_TEST_ALL)) ||
       (miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--float")))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(SoftMarginLossUnreducedBackwardTestHalf, )
{
    if(miopen::IsUnset(ENV(MIOPEN_TEST_ALL)) ||
       (miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--half")))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(SoftMarginLossUnreducedBackwardTestBFloat16, )
{
    if(miopen::IsUnset(ENV(MIOPEN_TEST_ALL)) ||
       (miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--bfloat16")))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};
INSTANTIATE_TEST_SUITE_P(SoftMarginLossTestSet,
                         SoftMarginLossUnreducedBackwardTestFloat,
                         testing::ValuesIn(SoftMarginLossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(SoftMarginLossTestSet,
                         SoftMarginLossUnreducedBackwardTestHalf,
                         testing::ValuesIn(SoftMarginLossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(SoftMarginLossTestSet,
                         SoftMarginLossUnreducedBackwardTestBFloat16,
                         testing::ValuesIn(SoftMarginLossTestConfigs()));

TEST_P(SoftMarginLossReducedForwardTestFloat, )
{
    if(miopen::IsUnset(ENV(MIOPEN_TEST_ALL)) ||
       (miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--float")))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(SoftMarginLossReducedForwardTestHalf, )
{
    if(miopen::IsUnset(ENV(MIOPEN_TEST_ALL)) ||
       (miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--half")))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(SoftMarginLossReducedForwardTestBFloat16, )
{
    if(miopen::IsUnset(ENV(MIOPEN_TEST_ALL)) ||
       (miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--bfloat16")))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};
INSTANTIATE_TEST_SUITE_P(SoftMarginLossTestSet,
                         SoftMarginLossReducedForwardTestFloat,
                         testing::ValuesIn(SoftMarginLossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(SoftMarginLossTestSet,
                         SoftMarginLossReducedForwardTestHalf,
                         testing::ValuesIn(SoftMarginLossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(SoftMarginLossTestSet,
                         SoftMarginLossReducedForwardTestBFloat16,
                         testing::ValuesIn(SoftMarginLossTestConfigs()));

TEST_P(SoftMarginLossReducedBackwardTestFloat, )
{
    if(miopen::IsUnset(ENV(MIOPEN_TEST_ALL)) ||
       (miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--float")))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(SoftMarginLossReducedBackwardTestHalf, )
{
    if(miopen::IsUnset(ENV(MIOPEN_TEST_ALL)) ||
       (miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--half")))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(SoftMarginLossReducedBackwardTestBFloat16, )
{
    if(miopen::IsUnset(ENV(MIOPEN_TEST_ALL)) ||
       (miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--bfloat16")))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};
INSTANTIATE_TEST_SUITE_P(SoftMarginLossTestSet,
                         SoftMarginLossReducedBackwardTestFloat,
                         testing::ValuesIn(SoftMarginLossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(SoftMarginLossTestSet,
                         SoftMarginLossReducedBackwardTestHalf,
                         testing::ValuesIn(SoftMarginLossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(SoftMarginLossTestSet,
                         SoftMarginLossReducedBackwardTestBFloat16,
                         testing::ValuesIn(SoftMarginLossTestConfigs()));
