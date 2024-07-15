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

#include "multimarginloss.hpp"
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace multimarginloss {

std::string GetFloatArg()
{
    const auto& tmp = env::value(MIOPEN_TEST_FLOAT_ARG);
    if(tmp.empty())
    {
        return "";
    }
    return tmp;
}

struct MultiMarginLossForwardTestFloat : MultiMarginLossForwardTest<float>
{
};

struct MultiMarginLossForwardTestHalf : MultiMarginLossForwardTest<half_float::half>
{
};

struct MultiMarginLossForwardTestBFloat16 : MultiMarginLossForwardTest<bfloat16>
{
};

} // namespace multimarginloss

using multimarginloss::MultiMarginLossForwardTestBFloat16;
using multimarginloss::MultiMarginLossForwardTestFloat;
using multimarginloss::MultiMarginLossForwardTestHalf;

TEST_P(MultiMarginLossForwardTestFloat, MMLFwdTest)
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

TEST_P(MultiMarginLossForwardTestHalf, MMLFwdTest)
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

TEST_P(MultiMarginLossForwardTestBFloat16, MMLFwdTest)
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

std::vector<MultiMarginLossTestCase> MultiMarginLossTestConfigs()
{
    // clang-format off
    return {
    {{22, 12}, true, MIOPEN_LOSS_REDUCTION_MEAN, 1}, 
    {{22, 12}, false, MIOPEN_LOSS_REDUCTION_SUM, 1}, 
    {{22, 12}, true, MIOPEN_LOSS_REDUCTION_NONE, 1}, 
    {{9456, 13}, false, MIOPEN_LOSS_REDUCTION_MEAN, 2 }, 
    {{9456, 13}, true, MIOPEN_LOSS_REDUCTION_SUM, 2 }, 
    {{9456, 13}, false, MIOPEN_LOSS_REDUCTION_NONE, 2 }, 
    {{543210, 7}, true, MIOPEN_LOSS_REDUCTION_MEAN, 2 }, 
    {{543210, 7}, false, MIOPEN_LOSS_REDUCTION_SUM, 2 }, 
    {{543210, 7}, true, MIOPEN_LOSS_REDUCTION_NONE, 2 }, 
    {{3995776, 6}, true, MIOPEN_LOSS_REDUCTION_MEAN, 1 }, 
    {{3995776, 6}, true, MIOPEN_LOSS_REDUCTION_SUM, 1 }, 
    {{3995776, 6}, true, MIOPEN_LOSS_REDUCTION_NONE, 1 }, 
    };
    // clang-format on
}

INSTANTIATE_TEST_SUITE_P(MultiMarginLossTestSet,
                         MultiMarginLossForwardTestFloat,
                         testing::ValuesIn(MultiMarginLossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(MultiMarginLossTestSet,
                         MultiMarginLossForwardTestHalf,
                         testing::ValuesIn(MultiMarginLossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(MultiMarginLossTestSet,
                         MultiMarginLossForwardTestBFloat16,
                         testing::ValuesIn(MultiMarginLossTestConfigs()));
