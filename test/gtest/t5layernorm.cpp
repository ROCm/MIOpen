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

#include "t5layernorm.hpp"
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace t5layernorm {

std::string GetFloatArg()
{
    const auto& tmp = env::value(MIOPEN_TEST_FLOAT_ARG);
    if(tmp.empty())
    {
        return "";
    }
    return tmp;
}

struct T5LayerNormTestFloat : T5LayerNormTest<float>
{
};

struct T5LayerNormTestHalf : T5LayerNormTest<half_float::half>
{
};

struct T5LayerNormTestBFloat16 : T5LayerNormTest<bfloat16>
{
};

struct T5LayerNormBwdTestFloat : T5LayerNormBwdTest<float>
{
};

struct T5LayerNormBwdTestHalf : T5LayerNormBwdTest<half_float::half>
{
};

struct T5LayerNormBwdTestBFloat16 : T5LayerNormBwdTest<bfloat16>
{
};

} // namespace t5layernorm
using namespace t5layernorm;

TEST_P(T5LayerNormTestFloat, T5LayerNormTestFw)
{
    auto TypeArg = env::value(MIOPEN_TEST_FLOAT_ARG);
    if(env::enabled(MIOPEN_TEST_ALL) && GetFloatArg() == "--float")
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(T5LayerNormTestHalf, T5LayerNormTestFw)
{
    auto TypeArg = env::value(MIOPEN_TEST_FLOAT_ARG);
    if(env::enabled(MIOPEN_TEST_ALL) && GetFloatArg() == "--half")
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(T5LayerNormTestBFloat16, T5LayerNormTestFw)
{
    auto TypeArg = env::value(MIOPEN_TEST_FLOAT_ARG);
    if(env::enabled(MIOPEN_TEST_ALL) && GetFloatArg() == "--bfloat16")
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(T5LayerNormBwdTestFloat, T5LayerNormBwdTestFw)
{
    auto TypeArg = env::value(MIOPEN_TEST_FLOAT_ARG);
    if(env::enabled(MIOPEN_TEST_ALL) && GetFloatArg() == "--float")
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(T5LayerNormBwdTestHalf, T5LayerNormBwdTestFw)
{
    auto TypeArg = env::value(MIOPEN_TEST_FLOAT_ARG);
    if(env::enabled(MIOPEN_TEST_ALL) && GetFloatArg() == "--half")
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(T5LayerNormBwdTestBFloat16, T5LayerNormBwdTestFw)
{
    auto TypeArg = env::value(MIOPEN_TEST_FLOAT_ARG);
    if(env::enabled(MIOPEN_TEST_ALL) && GetFloatArg() == "--bfloat16")
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(T5LayerNormTestSet,
                         T5LayerNormTestFloat,
                         testing::ValuesIn(T5LayerNormTestConfigs()));
INSTANTIATE_TEST_SUITE_P(T5LayerNormTestSet,
                         T5LayerNormTestHalf,
                         testing::ValuesIn(T5LayerNormTestConfigs()));
INSTANTIATE_TEST_SUITE_P(T5LayerNormTestSet,
                         T5LayerNormTestBFloat16,
                         testing::ValuesIn(T5LayerNormTestConfigs()));
INSTANTIATE_TEST_SUITE_P(T5LayerNormTestSet,
                         T5LayerNormBwdTestFloat,
                         testing::ValuesIn(T5LayerNormTestConfigs()));
INSTANTIATE_TEST_SUITE_P(T5LayerNormTestSet,
                         T5LayerNormBwdTestHalf,
                         testing::ValuesIn(T5LayerNormTestConfigs()));
INSTANTIATE_TEST_SUITE_P(T5LayerNormTestSet,
                         T5LayerNormBwdTestBFloat16,
                         testing::ValuesIn(T5LayerNormTestConfigs()));
