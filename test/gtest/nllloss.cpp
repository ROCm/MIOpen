/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <cstdio>
#include <miopen/env.hpp>
#include "nllloss.hpp"

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace nllloss {

std::string GetFloatArg()
{
    const auto& tmp = miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG));
    if(tmp.empty())
    {
        return "";
    }
    return tmp;
}

struct NLLLossTestFloat : NLLLossTest<float>
{
};

struct NLLLossTestHalf : NLLLossTest<half_float::half>
{
};

struct NLLLossTestBFloat16 : NLLLossTest<bfloat16>
{
};

} // namespace nllloss
using namespace nllloss;

TEST_P(NLLLossTestFloat, NLLLossTestFw)
{
    if(miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) || (GetFloatArg() == "--float"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(NLLLossTestHalf, NLLLossTestFw)
{
    if(miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) || (GetFloatArg() == "--half"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(NLLLossTestBFloat16, NLLLossTestFw)
{
    if(miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) || (GetFloatArg() == "--bfloat16"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(NLLLossTestSet, NLLLossTestFloat, testing::ValuesIn(NLLLossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(NLLLossTestSet, NLLLossTestHalf, testing::ValuesIn(NLLLossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(NLLLossTestSet,
                         NLLLossTestBFloat16,
                         testing::ValuesIn(NLLLossTestConfigs()));
