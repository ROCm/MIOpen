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
#include <miopen/env.hpp>
#include "transformers_adam_w.hpp"

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace transformers_adam_w {

std::string GetFloatArg()
{
    const auto& tmp = miopen::GetStringEnv(MIOPEN_ENV(MIOPEN_TEST_FLOAT_ARG));
    if(tmp.empty())
    {
        return "";
    }
    return tmp;
}

struct TransformersAdamWTestFloat : TransformersAdamWTest<float, float>
{
};

struct TransformersAdamWTestFloat16 : TransformersAdamWTest<half_float::half, half_float::half>
{
};

struct TransformersAmpAdamWTestFloat : TransformersAdamWTest<float, half_float::half, true>
{
};

} // namespace transformers_adam_w
using namespace transformers_adam_w;

TEST_P(TransformersAdamWTestFloat, TransformersAdamWFloatTest)
{
    if(miopen::IsEnabled(MIOPEN_ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--float"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(TransformersAdamWTestFloat16, TransformersAdamWFloat16Test)
{
    if(miopen::IsEnabled(MIOPEN_ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--half"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(TransformersAmpAdamWTestFloat, TransformersAmpAdamWTest)
{
    if(miopen::IsEnabled(MIOPEN_ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--float"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(TransformersAdamWTestSet,
                         TransformersAdamWTestFloat,
                         testing::ValuesIn(TransformersAdamWTestConfigs()));
INSTANTIATE_TEST_SUITE_P(TransformersAdamWTestSet,
                         TransformersAdamWTestFloat16,
                         testing::ValuesIn(TransformersAdamWTestConfigs()));
INSTANTIATE_TEST_SUITE_P(TransformersAdamWTestSet,
                         TransformersAmpAdamWTestFloat,
                         testing::ValuesIn(TransformersAdamWTestConfigs()));