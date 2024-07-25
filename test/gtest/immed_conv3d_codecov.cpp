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
#include <gtest/gtest.h>
#include <miopen/env.hpp>
#include "get_handle.hpp"
#include "test_env.hpp"

#include "immed_conv3d.hpp"

MIOPEN_DECLARE_ENV_VAR_BOOL(CODECOV_TEST)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLAGS_ARGS)

namespace immed_conv3d_codecov {

class ImmedConv3dFloat : public testing::TestWithParam<std::vector<std::string>>
{
};

class ImmedConv3dHalf : public testing::TestWithParam<std::vector<std::string>>
{
};

class ImmedConv3dBFloat16 : public testing::TestWithParam<std::vector<std::string>>
{
};

class ImmedConv3dInt8 : public testing::TestWithParam<std::vector<std::string>>
{
};

static bool SkipTest(void) { return !env::enabled(CODECOV_TEST); }

void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

void Run3dDriver(miopenDataType_t prec)
{

    std::vector<std::string> params;
    switch(prec)
    {
    case miopenHalf: params = ImmedConv3dHalf::GetParam(); break;
    case miopenBFloat16: params = ImmedConv3dBFloat16::GetParam(); break;
    case miopenFloat: params = ImmedConv3dFloat::GetParam(); break;
    case miopenInt8: params = ImmedConv3dInt8::GetParam(); break;
    case miopenFloat8:
    case miopenBFloat8:
    case miopenInt32:
    case miopenInt64:
    case miopenDouble:
        FAIL() << "miopenInt32, miopenDouble, miopenFloat8, miopenBFloat8 "
                  "data type not supported by "
                  "immed_conv3d_codecov test";

    default: params = ImmedConv3dFloat::GetParam();
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
        test_drive<conv3d_driver>(ptrs.size(), ptrs.data());
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
    {"test_immed_conv3d " + precision + " --input 1 4 4 4 4 --weights 2 4 3 3 3 --pads_strides_dilations 0 0 0 1 1 1 1 1 1 " + flag_arg}
        // clang-format on
    };

    return test_cases;
}

} // namespace immed_conv3d_codecov
using namespace immed_conv3d_codecov;

TEST_P(ImmedConv3dFloat, FloatTest_immed_conv3d_codecov)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && !SkipTest() && IsTestRunWith("--float"))
    {
        Run3dDriver(miopenFloat);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(ImmedConv3dHalf, HalfTest_immed_conv3d_codecov)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && !SkipTest() && IsTestRunWith("--half"))
    {
        Run3dDriver(miopenHalf);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(ImmedConv3dBFloat16, BFloat16Test_immed_conv3d_codecov)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && !SkipTest() && IsTestRunWith("--bfloat16"))
    {
        Run3dDriver(miopenBFloat16);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(ImmedConv3dInt8, Int8Test_immed_conv3d_codecov)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && !SkipTest() && IsTestRunWith("--int8"))
    {
        Run3dDriver(miopenInt8);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(ImmedConv3D, ImmedConv3dFloat, testing::Values(GetTestCases("--float")));

INSTANTIATE_TEST_SUITE_P(ImmedConv3D, ImmedConv3dHalf, testing::Values(GetTestCases("--half")));

INSTANTIATE_TEST_SUITE_P(ImmedConv3D,
                         ImmedConv3dBFloat16,
                         testing::Values(GetTestCases("--bfloat16")));

INSTANTIATE_TEST_SUITE_P(ImmedConv3D, ImmedConv3dInt8, testing::Values(GetTestCases("--int8")));
