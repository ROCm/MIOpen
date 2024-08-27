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
#include "../conv3d.hpp"
#include "get_handle.hpp"

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace conv3d_test {

static bool IsTestRunWith(const char* float_arg)
{
    assert(float_arg != nullptr);
    if(!MIOPEN_TEST_ALL)
        return true; // standalone run
    return env::enabled(MIOPEN_TEST_ALL) && env::value(MIOPEN_TEST_FLOAT_ARG) == float_arg;
}

void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

class GPU_Conv3d_Test_FP32 : public testing::TestWithParam<std::vector<std::string>>
{
};

void Run3dDriver(miopenDataType_t prec)
{
    std::vector<std::string> params;
    switch(prec)
    {
    case miopenFloat: params = GPU_Conv3d_Test_FP32::GetParam(); break;
    case miopenInt8:
    case miopenHalf:
    case miopenBFloat16:
    case miopenInt32:
    case miopenInt64:
    case miopenDouble:
        FAIL() << "miopenHalf, miopenBFloat16, miopenInt8, miopenInt32, "
                  "miopenDouble data "
                  "type not supported by "
                  "test_conv3d_extra test";

    default: params = GPU_Conv3d_Test_FP32::GetParam();
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
        test_drive<conv3d_driver>(ptrs.size(), ptrs.data(), "test_conv3d");
        auto capture = testing::internal::GetCapturedStderr();
        std::cout << capture;
    }
};

std::vector<std::string> GetTestCases(const std::string& precision)
{
    const std::string psd0 = " --pads_strides_dilations 0 0 0 1 1 1 1 1 1";
    const std::string psd1 = " --pads_strides_dilations 0 0 0 2 2 2 1 1 1";
    const std::string psd2 = " --pads_strides_dilations 2 2 2 1 1 1 1 1 1";
    const std::string psd3 = " --pads_strides_dilations 0 0 0 1 1 1 2 2 2";
    const std::string psd4 = " --pads_strides_dilations 1 1 1 1 1 1 1 1 1";
    const std::string psd5 = " --pads_strides_dilations 0 0 0 1 1 1 1 1 1";

    const std::vector<std::string> test_cases = {
        // clang-format off
    // test_conv3d_extra
    {precision + " --input 2 16 50 50 50 --weights 32 16 5 5 5" + psd0},
    {precision + " --input 2 16 50 50 50 --weights 32 16 5 5 5" + psd1},
    {precision + " --input 2 16 50 50 50 --weights 32 16 5 5 5" + psd2},
    {precision + " --input 2 16 50 50 50 --weights 32 16 5 5 5" + psd3},
    //test_conv3d_extra reduced set
    {precision + " --input 2 16 50 50 50 --weights 32 16 5 5 5" + psd0 },
    {precision + " --input 2 16 50 50 50 --weights 32 16 5 5 5" + psd1 },
    {precision + " --input 2 16 50 50 50 --weights 32 16 5 5 5" + psd2 },
    {precision + " --input 2 16 50 50 50 --weights 32 16 5 5 5" + psd3 },
    {precision + " --input 1 16 4 161 700 --weights 16 16 3 11 11" + psd4 },
    {precision + " --input 1 16 4 161 700 --weights 16 16 3 11 11" + psd5 },
    {precision + " --input 1 16 4 140 602 --weights 16 16 3 11 11" + psd4 },
    {precision + " --input 1 16 4 140 602 --weights 16 16 3 11 11" + psd5 }
        // clang-format on
    };
    return test_cases;
}

} // namespace conv3d_test

using namespace conv3d_test;

TEST_P(GPU_Conv3d_Test_FP32, FloatTest_conv3d_test)
{
    if(IsTestRunWith("--float"))
    {
        Run3dDriver(miopenFloat);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(Full, GPU_Conv3d_Test_FP32, testing::Values(GetTestCases("--float")));
