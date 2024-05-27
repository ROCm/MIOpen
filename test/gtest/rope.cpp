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
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace rope {

std::string GetFloatArg()
{
    const auto& tmp = miopen::GetStringEnv(MIOPEN_ENV(MIOPEN_TEST_FLOAT_ARG));
    if(tmp.empty())
    {
        return "";
    }
    return tmp;
}

struct RoPEFwdTestFloat : RoPEFwdTest<float>
{
};

struct RoPEFwdTestHalf : RoPEFwdTest<half_float::half>
{
};

struct RoPEFwdTestBFloat16 : RoPEFwdTest<bfloat16>
{
};

struct RoPEBwdTestFloat : RoPEBwdTest<float>
{
};

struct RoPEBwdTestHalf : RoPEBwdTest<half_float::half>
{
};

struct RoPEBwdTestBFloat16 : RoPEBwdTest<bfloat16>
{
};

} // namespace rope
using namespace rope;

TEST_P(RoPEFwdTestFloat, RoPEFwdTest)
{
    auto TypeArg = miopen::GetStringEnv(MIOPEN_ENV(MIOPEN_TEST_FLOAT_ARG));
    if((miopen::IsUnset(MIOPEN_ENV(MIOPEN_TEST_ALL)) ||
        (miopen::IsEnabled(MIOPEN_ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--float"))))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(RoPEFwdTestHalf, RoPEFwdTest)
{
    auto TypeArg = miopen::GetStringEnv(MIOPEN_ENV(MIOPEN_TEST_FLOAT_ARG));
    if((miopen::IsUnset(MIOPEN_ENV(MIOPEN_TEST_ALL)) ||
        (miopen::IsEnabled(MIOPEN_ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--half"))))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(RoPEFwdTestBFloat16, RoPEFwdTest)
{
    auto TypeArg = miopen::GetStringEnv(MIOPEN_ENV(MIOPEN_TEST_FLOAT_ARG));
    if((miopen::IsUnset(MIOPEN_ENV(MIOPEN_TEST_ALL)) ||
        (miopen::IsEnabled(MIOPEN_ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--bfloat16"))))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(RoPEBwdTestFloat, RoPEBwdTest)
{
    auto TypeArg = miopen::GetStringEnv(MIOPEN_ENV(MIOPEN_TEST_FLOAT_ARG));
    if((miopen::IsUnset(MIOPEN_ENV(MIOPEN_TEST_ALL)) ||
        (miopen::IsEnabled(MIOPEN_ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--float"))))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(RoPEBwdTestHalf, RoPEBwdTest)
{
    auto TypeArg = miopen::GetStringEnv(MIOPEN_ENV(MIOPEN_TEST_FLOAT_ARG));
    if((miopen::IsUnset(MIOPEN_ENV(MIOPEN_TEST_ALL)) ||
        (miopen::IsEnabled(MIOPEN_ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--half"))))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(RoPEBwdTestBFloat16, RoPEBwdTest)
{
    auto TypeArg = miopen::GetStringEnv(MIOPEN_ENV(MIOPEN_TEST_FLOAT_ARG));
    if((miopen::IsUnset(MIOPEN_ENV(MIOPEN_TEST_ALL)) ||
        (miopen::IsEnabled(MIOPEN_ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--bfloat16"))))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(RoPETestSet, RoPEFwdTestFloat, testing::ValuesIn(RoPETestConfigs()));
INSTANTIATE_TEST_SUITE_P(RoPETestSet, RoPEFwdTestHalf, testing::ValuesIn(RoPETestConfigs()));
INSTANTIATE_TEST_SUITE_P(RoPETestSet, RoPEFwdTestBFloat16, testing::ValuesIn(RoPETestConfigs()));
INSTANTIATE_TEST_SUITE_P(RoPETestSet, RoPEBwdTestFloat, testing::ValuesIn(RoPETestConfigs()));
INSTANTIATE_TEST_SUITE_P(RoPETestSet, RoPEBwdTestHalf, testing::ValuesIn(RoPETestConfigs()));
INSTANTIATE_TEST_SUITE_P(RoPETestSet, RoPEBwdTestBFloat16, testing::ValuesIn(RoPETestConfigs()));
