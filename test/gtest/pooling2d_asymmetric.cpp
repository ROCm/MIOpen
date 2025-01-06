/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#include "pooling2d.hpp"

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLAGS_ARGS)

namespace env = miopen::env;

namespace pooling2d_asymmetric {

class GPU_AsymPooling2d_FP32 : public testing::TestWithParam<std::vector<std::string>>
{
};

class GPU_AsymPooling2d_FP16 : public testing::TestWithParam<std::vector<std::string>>
{
};

void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

void Run2dDriver(miopenDataType_t prec)
{

    std::vector<std::string> params;
    switch(prec)
    {
    case miopenFloat: params = GPU_AsymPooling2d_FP32::GetParam(); break;
    case miopenHalf: params = GPU_AsymPooling2d_FP16::GetParam(); break;
    case miopenBFloat16:
    case miopenInt8:
    case miopenFloat8:
    case miopenBFloat8:
    case miopenInt32:
    case miopenInt64:
    case miopenDouble:
        FAIL()
            << "miopenBFloat16, miopenInt8, miopenInt32, miopenDouble, miopenFloat8, miopenBFloat8 "
               "data type not supported by "
               "pooling2d_asymmetric test";

    default: params = GPU_AsymPooling2d_FP32::GetParam();
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
        test_drive<pooling2d_driver>(ptrs.size(), ptrs.data());
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
    {"test_pooling2d " + precision + " --all --dataset 1 --limit 0 " + flag_arg}
        // clang-format on
    };

    return test_cases;
}

} // namespace pooling2d_asymmetric
using namespace pooling2d_asymmetric;

/*
TEST_P(GPU_AsymPooling2d_FP32, FloatTest_pooling2d_asymmetric)
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
*/

TEST_P(GPU_AsymPooling2d_FP16, HalfTest_pooling2d_asymmetric)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle))
    {
        Run2dDriver(miopenHalf);
    }
    else
    {
        GTEST_SKIP();
    }
};

// INSTANTIATE_TEST_SUITE_P(Full, GPU_AsymPooling2d_FP32, testing::Values(GetTestCases("--float")));

INSTANTIATE_TEST_SUITE_P(Full, GPU_AsymPooling2d_FP16, testing::Values(GetTestCases("--half")));
