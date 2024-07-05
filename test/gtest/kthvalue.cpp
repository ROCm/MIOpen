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

#include "kthvalue.hpp"
#include "miopen/bfloat16.hpp"
#include "tensor_holder.hpp"
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace kthvalue {

std::string GetFloatArg()
{
    const auto& tmp = miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG));
    if(tmp.empty())
    {
        return "";
    }
    return tmp;
}

struct KthvalueForwardTestFloat32 : KthvalueFwdTest<float>
{
};

struct KthvalueForwardTestFloat16 : KthvalueFwdTest<half>
{
};

// struct KthvalueForwardTestBFloat16 : KthvalueFwdTest<bfloat16>
// {
// };

using namespace kthvalue;

TEST_P(KthvalueForwardTestFloat32, KthvalueForwardTest)
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

TEST_P(KthvalueForwardTestFloat16, KthvalueForwardTest)
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

// TEST_P(KthvalueForwardTestBFloat16, KthvalueForwardTest)
// {
//     if(miopen::IsUnset(ENV(MIOPEN_TEST_ALL)) ||
//        (miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--bfloat16")))
//     {
//         RunTest();
//         Verify();
//     }
//     else
//     {
//         GTEST_SKIP();
//     }
// };

INSTANTIATE_TEST_SUITE_P(KthvalueForwardTestSet,
                         KthvalueForwardTestFloat32,
                         testing::ValuesIn(KthvalueTestConfigs()));

INSTANTIATE_TEST_SUITE_P(KthvalueForwardTestSet,
                         KthvalueForwardTestFloat16,
                         testing::ValuesIn(KthvalueTestConfigs()));

// INSTANTIATE_TEST_SUITE_P(KthvalueForwardTestSet,
//                          KthvalueForwardTestBFloat16,
//                          testing::ValuesIn(KthvalueTestConfigs()));
} // namespace kthvalue
