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
#include <miopen/env.hpp>
#include "../conv2d.hpp"
#include "get_handle.hpp"

using TestCase = std::tuple<std::vector<std::string>, std::string>;

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)
MIOPEN_DECLARE_ENV_VAR_BOOL(IMPLICITGEMM_TESTING_ENV)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)

namespace env = miopen::env;

namespace test_conv_for_implicit_gemm {

static bool SkipTest()
{
    if(!MIOPEN_TEST_ALL)
        return false;
    if(env::enabled(IMPLICITGEMM_TESTING_ENV))
        return false;
    return true;
}

static bool IsTestRunWith(const char* float_arg)
{
    assert(float_arg != nullptr);
    if(!MIOPEN_TEST_ALL)
        return true; // standalone run
    return env::enabled(MIOPEN_TEST_ALL) && env::value(MIOPEN_TEST_FLOAT_ARG) == float_arg;
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

class ConfigWithHalf : public testing::TestWithParam<std::vector<TestCase>>
{
};

class ConfigWithBF16 : public testing::TestWithParam<std::vector<TestCase>>
{
};

void Run2dDriver(miopenDataType_t prec)
{
    std::vector<TestCase> params;

    switch(prec)
    {
    case miopenHalf: params = ConfigWithHalf::GetParam(); break;
    case miopenBFloat16: params = ConfigWithBF16::GetParam(); break;
    case miopenFloat:
    case miopenInt8:
    case miopenInt32:
    case miopenInt64:
    case miopenDouble:
    case miopenFloat8:
    case miopenBFloat8:
    default:
        FAIL() << "miopenFloat, miopenInt8, miopenInt32, miopenDouble, miopenFloat8, miopenBFloat8 "
                  "data type not supported by test_conv_for_implicit_gemm test";
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
        test_drive<conv2d_driver>(ptrs.size(), ptrs.data(), "test_conv2d");
        auto capture = testing::internal::GetCapturedStderr();
        EXPECT_FALSE(capture.find("No suitable algorithm was found") != std::string::npos);
        std::cout << capture;
    }
}

bool IsTestSupportedForDevice(const miopen::Handle& handle)
{
    std::string devName = handle.GetDeviceName();
    if(devName == "gfx900" || devName == "gfx906" || devName == "gfx908" || devName == "gfx90a" ||
       miopen::StartsWith(devName, "gfx94") || miopen::StartsWith(devName, "gfx103") ||
       miopen::StartsWith(devName, "gfx110"))
    {
        return true;
    }
    return false;
}

std::vector<TestCase> GetTestCases(const std::string& precision)
{

    std::vector<std::string> env = {
        "MIOPEN_FIND_MODE=normal",
        "MIOPEN_DEBUG_CONV_WINOGRAD=0",
        "MIOPEN_DEBUG_CONV_GEMM=0",
        "MIOPEN_DEBUG_CONV_DIRECT=0",
        "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1",
        "MIOPEN_DEBUG_CONV_FFT=0",
    };

    std::string flags = " --verbose ";

    std::string psd0 = " --pads_strides_dilations 0 0 2 2 1 1";
    std::string psd1 = " --pads_strides_dilations 0 0 1 1 1 1";
    std::string psd2 = " --pads_strides_dilations 1 1 2 2 1 1";
    std::string psd3 = " --pads_strides_dilations 0 0 1 1 2 2";
    std::string psd4 = " --pads_strides_dilations 1 1 1 1 1 1";
    std::string psd5 = " --pads_strides_dilations 2 2 2 2 1 1";

    const std::vector<TestCase> test_cases = {

        // clang-format off
        TestCase{env, precision + flags + "--input 64 16 28 28 --weights 192 16 3 3 " + psd0},
        TestCase{env, precision + flags + "--input 64 16 14 14 --weights 160 16 3 3 " + psd0},
        TestCase{env, precision + flags +   "--input 64 16 7 7 --weights 128 16 3 3 " + psd0},
        TestCase{env, precision + flags +   "--input 64 16 55 55 --weights 96 16 1 7 " + psd0},
        TestCase{env, precision + flags +   "--input 64 16 28 28 --weights 64 16 1 7 " + psd0},
        TestCase{env, precision + flags +   "--input 64 16 14 14 --weights 32 16 1 7 " + psd0},
        TestCase{env, precision + flags +   "--input 64 32 28 28 --weights 192 32 3 3 " + psd0},
        TestCase{env, precision + flags +   "--input 64 32 14 14 --weights 160 32 3 3 " + psd0},
        TestCase{env, precision + flags +   "--input 64 32 7 7 --weights 128 32 3 3 " + psd0},
        TestCase{env, precision + flags +   "--input 64 32 55 55 --weights 96 32 1 7 " + psd0},
        TestCase{env, precision + flags +   "--input 64 32 28 28 --weights 64 32 1 7 " + psd0},
        TestCase{env, precision + flags +   "--input 64 32 14 14 --weights 32 32 1 7 " + psd0},

        TestCase{env, precision + flags +   "--input 64 64 56 56 --weights 256 64 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 64 56 56 --weights 64 64 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 64 73 73 --weights 80 64 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 64 56 56 --weights 64 64 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 128 55 55 --weights 16 128 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 128 28 28 --weights 16 128 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 128 14 14 --weights 16 128 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 128 7 7 --weights 16 128 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 16 64 56 56 --weights 256 64 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 16 64 56 56 --weights 64 64 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 16 64 73 73 --weights 80 64 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 16 64 56 56 --weights 64 64 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 16 128 55 55 --weights 16 128 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 16 128 28 28 --weights 16 128 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 16 128 7 7 --weights 16 128 1 1 " + psd1},

        TestCase{env, precision + flags +   "--input 64 64 55 55 --weights 16 128 1 1 " + psd0},
        TestCase{env, precision + flags +   "--input 64 128 28 28 --weights 16 128 1 1 " + psd0},
        TestCase{env, precision + flags +   "--input 64 128 14 14 --weights 16 128 1 1 " + psd0},
        TestCase{env, precision + flags +   "--input 64 128 7 7 --weights 16 128 1 1 " + psd0},

        TestCase{env, precision + flags +   "--input 64 128 28 28 --weights 512 128 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 160 73 73 --weights 64 160 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 192 35 35 --weights 32 192 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 192 35 35 --weights 48 192 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 192 35 35 --weights 64 192 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 192 28 28 --weights 16 192 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 192 28 28 --weights 32 192 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 192 28 28 --weights 64 192 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 192 28 28 --weights 96 192 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 256 35 35 --weights 48 256 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 256 35 35 --weights 64 256 1 1 " + psd1},

        TestCase{env, precision + flags +   "--input 64 256 56 56 --weights 128 256 1 1 " + psd0},
        TestCase{env, precision + flags +   "--input 64 256 56 56 --weights 512 256 1 1 " + psd0},


        TestCase{env, precision + flags +   "--input 64 256 56 56 --weights 64 256 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 256 28 28 --weights 128 256 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 256 28 28 --weights 32 256 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 256 28 28 --weights 64 256 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 288 35 35 --weights 48 288 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 288 35 35 --weights 64 288 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 384 35 35 --weights 192 384 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 384 35 35 --weights 64 384 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 384 35 35 --weights 96 384 1 1 " + psd1},

        TestCase{env, precision + flags +   "--input 64 480 14 14 --weights 16 480 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 480 14 14 --weights 192 480 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 480 14 14 --weights 64 480 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 480 14 14 --weights 96 480 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 512 28 28 --weights 128 512 1 1 " + psd1},

        TestCase{env, precision + flags +   "--input 64 512 28 28 --weights 256 512 1 1 " + psd0},

        TestCase{env, precision + flags +   "--input 64 512 14 14 --weights 112 512 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 512 14 14 --weights 128 512 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 512 14 14 --weights 144 512 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 512 14 14 --weights 160 512 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 512 14 14 --weights 24 512 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 512 14 14 --weights 32 512 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 64 512 14 14 --weights 64 512 1 1 " + psd1},

        TestCase{env, precision + flags +   "--input 128 832 7 7 --weights 32 832 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 128 832 7 7 --weights 192 832 1 1 " + psd1},
        TestCase{env, precision + flags +   "--input 128 832 7 7 --weights 128 832 1 1 " + psd1},

        TestCase{env, precision + flags +   "--input 128 832 7 7 --weights 32 832 1 1 " + psd3},
        TestCase{env, precision + flags +   "--input 128 832 7 7 --weights 192 832 1 1 " + psd3},
        TestCase{env, precision + flags +   "--input 128 832 7 7 --weights 128 832 1 1 " + psd3},
        TestCase{env, precision + flags +   "--input 16 2048 7 7 --weights 192 2048 1 1 " + psd3},

        TestCase{env, precision + flags +   "--input 64 32 28 28 --weights 192 32 3 3 " + psd2},
        TestCase{env, precision + flags +   "--input 8 16 14 14 --weights 32 16 1 1 " + psd4},
        TestCase{env, precision + flags +   "--input 64 32 14 14 --weights 192 32 3 3 " + psd2},
        TestCase{env, precision + flags +   "--input 64 32 7 7 --weights 192 32 3 3 " + psd2},
        TestCase{env, precision + flags +   "--input 64 32 28 28 --weights 192 32 3 3 " + psd5},
        TestCase{env, precision + flags +   "--input 64 32 14 14 --weights 192 32 3 3 " + psd5},
        TestCase{env, precision + flags +   "--input 64 32 7 7 --weights 192 32 3 3 " + psd5}
    };

    return test_cases ;
}

} //namespace test_conv_for_implicit_gemm

using namespace test_conv_for_implicit_gemm;

TEST_P(ConfigWithBF16, Test_conv_for_implicit_gemm_bf16)
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

TEST_P(ConfigWithHalf, Test_conv_for_implicit_gemm_half)
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

INSTANTIATE_TEST_SUITE_P(ConvIgemm,
                             ConfigWithBF16,
                             testing::Values(GetTestCases("--bfloat16")));

INSTANTIATE_TEST_SUITE_P(ConvIgemm,
                             ConfigWithHalf,
                             testing::Values(GetTestCases("--half")));
