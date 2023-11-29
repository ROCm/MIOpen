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
#include <tuple>
#include <miopen/miopen.h>
#include <gtest/gtest.h>
#include "../conv3d.cpp"
#include "get_handle.hpp"

using TestCase = std::tuple<std::vector<std::string>, std::string>;

MIOPEN_DECLARE_ENV_VAR(MIOPEN_TEST_ALL)

MIOPEN_DECLARE_ENV_VAR(MIOPEN_TEST_GPU_XNACK_ENABLED)

static bool SkipTest(void)
{
    return miopen::IsEnabled(MIOPEN_TEST_GPU_XNACK_ENABLED{}) ||
           miopen::IsDisabled(MIOPEN_TEST_ALL{});
}

void GetArgs(const TestCase& param, std::vector<std::string>& tokens)
{
    auto env_vars = std::get<0>(param);
    for(auto& elem : env_vars)
    {
        putenv(elem.data());
    }

    auto cmd = std::get<1>(param);

    std::stringstream ss(cmd);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

class Conv3dFloat : public testing::TestWithParam<std::vector<TestCase>>
{
};

void Run3dDriver(miopenDataType_t prec)
{
    std::vector<TestCase> params;
    switch(prec)
    {
    case miopenFloat: params = Conv3dFloat::GetParam(); break;
    case miopenBFloat16:
    case miopenHalf:
    case miopenInt8:
    case miopenInt32:
    case miopenDouble:
    case miopenFloat8:
    case miopenBFloat8:
        FAIL() << "miopenHalf, miopenInt8, miopenBFloat16, miopenInt32, "
                  "miopenDouble, miopenFloat8, miopenBFloat8 "
                  "data type not supported by conv_igemm_dynamic test";

    default: params = Conv3dFloat::GetParam(); break;
    }

    for(const auto& test_value : params)
    {
        std::vector<std::string> tokens;
        GetArgs(test_value, tokens);
        std::vector<const char*> ptrs;

        std::transform(tokens.begin(),
                       tokens.end(),
                       std::back_inserter(ptrs),
                       [](const std::string& str) { return str.data(); });

        testing::internal::CaptureStderr();
        test_drive<conv3d_driver>(ptrs.size(), ptrs.data());
        auto capture = testing::internal::GetCapturedStderr();
        std::cout << capture;
    }
}

bool IsTestSupportedForDevice(const miopen::Handle& handle)
{
    std::string devName = handle.GetDeviceName();
    if(devName == "gfx94" || devName == "gfx103" || miopen::StartsWith(devName, "gfx110"))
        return true;
    else
        return false;
}

TEST_P(Conv3dFloat, FloatTest)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && !SkipTest())
    {
        Run3dDriver(miopenFloat);
    }
    else
    {
        GTEST_SKIP();
    }
};

std::vector<TestCase> GetTestCases(const std::string& precision)
{

    const std::vector<TestCase> test_cases = {
        // clang-format off
    // test_conv3d_extra
    {{}, precision + "--input 2 16 50 50 50 --weights 32 16 5 5 5 --pads_strides_dilations 0 0 0 1 1 1 1 1 1" },
    {{}, precision + "--input 2 16 50 50 50 --weights 32 16 5 5 5 --pads_strides_dilations 0 0 0 2 2 2 1 1 1" },
    {{}, precision + "--input 2 16 50 50 50 --weights 32 16 5 5 5 --pads_strides_dilations 2 2 2 1 1 1 1 1 1" },
    {{}, precision + "--input 2 16 50 50 50 --weights 32 16 5 5 5 --pads_strides_dilations 0 0 0 1 1 1 2 2 2" },
    //test_conv3d_extra reduced set
    {{}, precision + "--input 2 16 50 50 50 --weights 32 16 5 5 5 --pads_strides_dilations 0 0 0 1 1 1 1 1 1" },
    {{}, precision + "--input 2 16 50 50 50 --weights 32 16 5 5 5 --pads_strides_dilations 0 0 0 2 2 2 1 1 1" },
    {{}, precision + "--input 2 16 50 50 50 --weights 32 16 5 5 5 --pads_strides_dilations 2 2 2 1 1 1 1 1 1" },
    {{}, precision + "--input 2 16 50 50 50 --weights 32 16 5 5 5 --pads_strides_dilations 0 0 0 1 1 1 2 2 2" },
    {{}, precision + "--input 1 16 4 161 700 --weights 16 16 3 11 11 --pads_strides_dilations 1 1 1 1 1 1 1 1 1" },
    {{}, precision + "--input 1 16 4 161 700 --weights 16 16 3 11 11 --pads_strides_dilations 0 0 0 1 1 1 1 1 1" },
    {{}, precision + "--input 1 16 4 140 602 --weights 16 16 3 11 11 --pads_strides_dilations 1 1 1 1 1 1 1 1 1" },
    {{}, precision + "--input 1 16 4 140 602 --weights 16 16 3 11 11 --pads_strides_dilations 0 0 0 1 1 1 1 1 1" }

    };
    return test_cases;
    };

INSTANTIATE_TEST_SUITE_P(Conv3dFloatTest,
                         Conv3dFloat,
                         testing::Values(GetTestCases("--float")));
