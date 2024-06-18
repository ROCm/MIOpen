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

#include "multilabelsoftmarginloss.hpp"
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace multilabelsoftmarginloss {

std::string GetFloatArg()
{
    const auto& tmp = miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG));
    if(tmp.empty())
    {
        return "";
    }
    return tmp;
}

struct MultilabelSoftMarginLossUnreducedForwardTestFloat
    : MultilabelSoftMarginLossUnreducedForwardTest<float>
{
};

struct MultilabelSoftMarginLossUnreducedForwardTestHalf
    : MultilabelSoftMarginLossUnreducedForwardTest<half_float::half>
{
};

struct MultilabelSoftMarginLossUnreducedForwardTestBFloat16
    : MultilabelSoftMarginLossUnreducedForwardTest<bfloat16>
{
};

struct MultilabelSoftMarginLossReducedForwardTestFloat
    : MultilabelSoftMarginLossReducedForwardTest<float>
{
};

struct MultilabelSoftMarginLossReducedForwardTestHalf
    : MultilabelSoftMarginLossReducedForwardTest<half_float::half>
{
};

struct MultilabelSoftMarginLossReducedForwardTestBFloat16
    : MultilabelSoftMarginLossReducedForwardTest<bfloat16>
{
};

} // namespace multilabelsoftmarginloss

using namespace multilabelsoftmarginloss;

TEST_P(MultilabelSoftMarginLossUnreducedForwardTestFloat, )
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

TEST_P(MultilabelSoftMarginLossUnreducedForwardTestHalf, )
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

TEST_P(MultilabelSoftMarginLossUnreducedForwardTestBFloat16, )
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
INSTANTIATE_TEST_SUITE_P(MultilabelSoftMarginLossTestSet,
                         MultilabelSoftMarginLossUnreducedForwardTestFloat,
                         testing::ValuesIn(MultilabelSoftMarginLossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(MultilabelSoftMarginLossTestSet,
                         MultilabelSoftMarginLossUnreducedForwardTestHalf,
                         testing::ValuesIn(MultilabelSoftMarginLossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(MultilabelSoftMarginLossTestSet,
                         MultilabelSoftMarginLossUnreducedForwardTestBFloat16,
                         testing::ValuesIn(MultilabelSoftMarginLossTestConfigs()));

TEST_P(MultilabelSoftMarginLossReducedForwardTestFloat, )
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

TEST_P(MultilabelSoftMarginLossReducedForwardTestHalf, )
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

TEST_P(MultilabelSoftMarginLossReducedForwardTestBFloat16, )
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
INSTANTIATE_TEST_SUITE_P(MultilabelSoftMarginLossTestSet,
                         MultilabelSoftMarginLossReducedForwardTestFloat,
                         testing::ValuesIn(MultilabelSoftMarginLossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(MultilabelSoftMarginLossTestSet,
                         MultilabelSoftMarginLossReducedForwardTestHalf,
                         testing::ValuesIn(MultilabelSoftMarginLossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(MultilabelSoftMarginLossTestSet,
                         MultilabelSoftMarginLossReducedForwardTestBFloat16,
                         testing::ValuesIn(MultilabelSoftMarginLossTestConfigs()));
