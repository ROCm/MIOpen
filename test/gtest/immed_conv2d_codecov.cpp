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
#include "conv_common.hpp"
#include "get_handle.hpp"
#include <miopen/env.hpp>
#include <gtest/gtest.h>
#include "test_env.hpp"

#include "immed_conv2d.hpp"

MIOPEN_DECLARE_ENV_VAR_BOOL(CODECOV_TEST)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLAGS_ARGS)

namespace immed_conv2d_codecov {

static bool SkipTest(void) { return !env::enabled(CODECOV_TEST); }

void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

class Conv2dFloat_immed_conv2d_codecov : public testing::TestWithParam<std::vector<std::string>>
{
};

class Conv2dHalf_immed_conv2d_codecov : public testing::TestWithParam<std::vector<std::string>>
{
};

class Conv2dBFloat16_immed_conv2d_codecov : public testing::TestWithParam<std::vector<std::string>>
{
};

class Conv2dInt8_immed_conv2d_codecov : public testing::TestWithParam<std::vector<std::string>>
{
};

void Run2dDriver(miopenDataType_t prec)
{

    std::vector<std::string> params;
    switch(prec)
    {
    case miopenHalf: params = Conv2dHalf_immed_conv2d_codecov::GetParam(); break;
    case miopenBFloat16: params = Conv2dBFloat16_immed_conv2d_codecov::GetParam(); break;
    case miopenFloat: params = Conv2dFloat_immed_conv2d_codecov::GetParam(); break;
    case miopenInt8: params = Conv2dInt8_immed_conv2d_codecov::GetParam(); break;
    case miopenFloat8:
    case miopenBFloat8:
    case miopenInt32:
    case miopenInt64:
    case miopenDouble:
        FAIL() << "miopenInt32, miopenDouble, miopenFloat8, miopenBFloat8 "
                  "data type not supported by "
                  "immed_conv2d_codecov test";

    default: params = Conv2dFloat_immed_conv2d_codecov::GetParam();
    }

    for(const auto& test_value : params)
    {
        std::vector<std::string> tokens;
        GetArgs(test_value, tokens);
        std::vector<const char*> ptrs;

        std::transform(tokens.begin(), tokens.end(), std::back_inserter(ptrs), [](const auto& str) {
            return str.data();
        });

        testing::internal::CaptureStderr();
        test_drive<conv2d_driver>(ptrs.size(), ptrs.data());
        auto capture = testing::internal::GetCapturedStderr();
        std::cout << capture;
    }
};

bool IsTestSupportedForDevice(const miopen::Handle& handle) { return true; }

std::vector<std::string> GetTestCases(const std::string& precision)
{
    const auto& flag_arg = env::value(MIOPEN_TEST_FLAGS_ARGS);

    const std::vector<std::string> test_cases = {
        // clang-format off
    {"test_immed_conv2d " + precision + " --input  2 2 14 14 --weights 8 2 3 3 --pads_strides_dilations 0 0 1 1 1 1 " + flag_arg}
        // clang-format on
    };

    return test_cases;
}

} // namespace immed_conv2d_codecov
using namespace immed_conv2d_codecov;

TEST_P(Conv2dFloat_immed_conv2d_codecov, FloatTest)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && !SkipTest() && IsTestRunWith("--float"))
    {
        Run2dDriver(miopenFloat);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(Conv2dHalf_immed_conv2d_codecov, HalfTest)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && !SkipTest() && IsTestRunWith("--half"))
    {
        Run2dDriver(miopenHalf);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(Conv2dBFloat16_immed_conv2d_codecov, BFloat16Test)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && !SkipTest() && IsTestRunWith("--bfloat16"))
    {
        Run2dDriver(miopenBFloat16);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(Conv2dInt8_immed_conv2d_codecov, Int8Test)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && !SkipTest() && IsTestRunWith("--int8"))
    {
        Run2dDriver(miopenInt8);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(ImmedConv2D,
                         Conv2dFloat_immed_conv2d_codecov,
                         testing::Values(GetTestCases("--float")));

INSTANTIATE_TEST_SUITE_P(ImmedConv2D,
                         Conv2dHalf_immed_conv2d_codecov,
                         testing::Values(GetTestCases("--half")));

INSTANTIATE_TEST_SUITE_P(ImmedConv2D,
                         Conv2dBFloat16_immed_conv2d_codecov,
                         testing::Values(GetTestCases("--bfloat16")));

INSTANTIATE_TEST_SUITE_P(ImmedConv2D,
                         Conv2dInt8_immed_conv2d_codecov,
                         testing::Values(GetTestCases("--int8")));
