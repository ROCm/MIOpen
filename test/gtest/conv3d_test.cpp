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
#include <miopen/env.hpp>
#include <gtest/gtest.h>
#include "../conv3d.cpp"
#include "get_handle.hpp"

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_COMPOSABLEKERNEL)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

using TestCase = std::tuple<std::vector<std::string>, std::string>;

static bool IsTestRunWith(const char* float_arg)
{
    assert(float_arg != nullptr);
    const char* const p_envVar = miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG));
    return (p_envVar != nullptr && std::strcmp(p_envVar, float_arg) == 0);
}

void GetArgs(const std::string& cmd, std::vector<std::string>& tokens)
{
    std::stringstream ss(cmd);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

class Conv3dFloat : public testing::TestWithParam<std::string>
{
};

void Run3dDriver(miopenDataType_t prec)
{
    std::string test_value;
    switch(prec)
    {
    case miopenHalf: params = ConfigWithHalf::GetParam(); break;
    case miopenInt8: params = ConfigWithInt8::GetParam(); break;
    case miopenFloat:
    case miopenBFloat16:
    case miopenInt32:
    case miopenDouble:
    case miopenFloat8:
    case miopenBFloat8:
        MIOPEN_THROW(miopenStatusBadParm,
                     "miopenBFloat16, miopenFloat, miopenInt32, miopenDouble, "
                     "miopenFloat8, miopenBFloat8 data types not supported by "
                     "conv_igemm_mlir_xdlops test");
        break;
    default: FAIL() << "Unknown data type for conv3d test";
    }
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

bool IsTestSupportedForDevice(const miopen::Handle& handle)
{
    std::string devName = handle.GetDeviceName();
    if(devName == "gfx94" || devName == "gfx103" || devName == "gfx110")
        return true;
    else
        return false;
}

TEST_P(Conv3dFloat, FloatTest)
{
#if MIOPEN_BACKEND_OPENCL
    GTEST_SKIP() << "MIOPEN_BACKEND_HIP needed for this test";
#else
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) &&
       miopen::IsEnvvarValueEnabled("MIOPEN_TEST_COMPOSABLEKERNEL") &&
       miopen::IsEnvvarValueEnabled("MIOPEN_TEST_ALL") && IsTestRunWith("--float"))
    {
        Run3dDriver(miopenFloat);
    }
    else
    {
        GTEST_SKIP();
    }
#endif
}

std::vector<std::string> GetTestCases(const std::string& precision)
{
    std::string psd0 = " --pads_strides_dilations 0 0 0 1 1 1 1 1 1";
    std::string psd1 = " --pads_strides_dilations 0 0 0 2 2 2 1 1 1";
    std::string psd2 = " --pads_strides_dilations 2 2 2 1 1 1 1 1 1";
    std::string psd3 = " --pads_strides_dilations 0 0 0 1 1 1 2 2 2";
    std::string psd4 = " --pads_strides_dilations 1 1 1 1 1 1 1 1 1";
    std::string psd5 = " --pads_strides_dilations 0 0 0 1 1 1 1 1 1";

    std::vector<std::string> test_cases = {
        // clang-format off
    // test_conv3d_extra
    {precision + "--input 2 16 50 50 50 --weights 32 16 5 5 5" + psd0},
    {precision + "--input 2 16 50 50 50 --weights 32 16 5 5 5" + psd1},
    {precision + "--input 2 16 50 50 50 --weights 32 16 5 5 5" + psd2},
    {precision + "--input 2 16 50 50 50 --weights 32 16 5 5 5" + psd3},
    //test_conv3d_extra reduced set
    {precision + "--input 2 16 50 50 50 --weights 32 16 5 5 5" + psd0 },
    {precision + "--input 2 16 50 50 50 --weights 32 16 5 5 5" + psd1 },
    {precision + "--input 2 16 50 50 50 --weights 32 16 5 5 5" + psd2 },
    {precision + "--input 2 16 50 50 50 --weights 32 16 5 5 5" + psd3 },
    {precision + "--input 1 16 4 161 700 --weights 16 16 3 11 11" + psd4 },
    {precision + "--input 1 16 4 161 700 --weights 16 16 3 11 11" + psd5 },
    {precision + "--input 1 16 4 140 602 --weights 16 16 3 11 11" + psd4 },
    {precision + "--input 1 16 4 140 602 --weights 16 16 3 11 11" + psd5 }

    };
    return test_cases;
}

INSTANTIATE_TEST_SUITE_P(Conv3dFloatTest,
                         Conv3dFloat,
                         testing::ValuesIn(GetTestCases("--float")));
