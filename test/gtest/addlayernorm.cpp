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

#include "addlayernorm.hpp"
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace layernorm {

std::string GetFloatArg()
{
    const auto& tmp = miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG));
    if(tmp.empty())
    {
        return "";
    }
    return tmp;
}

struct AddLayerNormTestFloat : AddLayerNormTest<float>
{
};

struct AddLayerNormTestHalf : AddLayerNormTest<half_float::half>
{
};

struct AddLayerNormTestBFloat16 : AddLayerNormTest<bfloat16>
{
};

} // namespace layernorm
using namespace layernorm;

TEST_P(AddLayerNormTestFloat, AddLayerNormTestFw)
{
    auto TypeArg = miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG));
    if(miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--float"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(AddLayerNormTestHalf, AddLayerNormTestFw)
{
    auto TypeArg = miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG));
    if(miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--half"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(AddLayerNormTestBFloat16, AddLayerNormTestFw)
{
    auto TypeArg = miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG));
    if(miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--bfloat16"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(AddLayerNormTestSet,
                         AddLayerNormTestFloat,
                         testing::ValuesIn(AddLayerNormTestConfigs()));
INSTANTIATE_TEST_SUITE_P(AddLayerNormTestSet,
                         AddLayerNormTestHalf,
                         testing::ValuesIn(AddLayerNormTestConfigs()));
INSTANTIATE_TEST_SUITE_P(AddLayerNormTestSet,
                         AddLayerNormTestBFloat16,
                         testing::ValuesIn(AddLayerNormTestConfigs()));
