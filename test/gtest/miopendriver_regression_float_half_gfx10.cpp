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
#include <miopen/miopen.h>
#include <gtest/gtest_common.hpp>
#include <gtest/gtest.h>
#include <miopen/env.hpp>
#include "get_handle.hpp"
#include "../conv2d.hpp"

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_WITH_MIOPENDRIVER)

namespace miopendriver_regression_float_half_gfx10 {
void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

auto GetTestCases(const std::string& precision)
{
    std::string cmd = "MIOpenDriver ";

    if(precision == "--float")
        cmd.append("bnorm");
    else if(precision == "--half")
        cmd.append("bnormfp16");

    // clang-format off
    return std::vector<std::string>{
        {cmd + " -n 256 -c 512 -H 18 -W 18 -m 1 --forw 0 -b 1 -r 1"},
        {cmd + " -n 256 -c 512 -H 28 -W 28 -m 1 --forw 0 -b 1 -r 1"}
    };
    // clang-format on
}

using TestCase = decltype(GetTestCases())::value_type;

class ConfigWithFloat_miopendriver_regression_float_half_gfx10
    : public testing::TestWithParam<std::vector<TestCase>>
{
};

class ConfigWithHalf_miopendriver_regression_float_half_gfx10
    : public testing::TestWithParam<std::vector<TestCase>>
{
};

bool IsTestSupportedForDevice()
{
    using namespace miopen::debug;
    using e_mask = enabled<Gpu::gfx103X>;
    using d_mask = disabled<Gpu::gfx900, Gpu::gfx906, Gpu::gfx908, Gpu::gfx90A>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

static bool SkipTest(const std::string& float_arg)
{
    if(IsTestSupportedForDevice() && miopen::IsEnabled(ENV(MIOPEN_TEST_WITH_MIOPENDRIVER)) &&
       (miopen::IsUnset(ENV(MIOPEN_TEST_ALL))       // standalone run
        || (miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) // or full tests enabled
            && miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG)) == float_arg)))
        return false;
    return true;
}

void Run2dDriver(miopenDataType_t prec)
{
    std::vector<std::string> params;
    switch(prec)
    {
    case miopenFloat:
        params = ConfigWithFloat_miopendriver_regression_float_half_gfx10::GetParam();
        break;
    case miopenHalf:
        params = ConfigWithHalf_miopendriver_regression_float_half_gfx10::GetParam();
        break;
    case miopenInt8:
    case miopenBFloat16:
    case miopenInt32:
    case miopenFloat8:
    case miopenBFloat8:
    case miopenDouble:
    default:
        FAIL() << "miopenInt8, miopenBFloat16, miopenInt32, miopenFloat8, miopenBFloat8, "
                  "miopenDouble data type "
                  "not supported by miopendriver_regression_float_half_gfx10 test";
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

} // namespace miopendriver_regression_float_half_gfx10
using namespace miopendriver_regression_float_half_gfx10;

TEST_P(ConfigWithFloat_miopendriver_regression_float_half_gfx10,
       FloatTest_miopendriver_regression_float_half_gfx10)
{
    if(SkipTest("--float"))
    {
        GTEST_SKIP();
    }
    else
    {
        Run2dDriver(miopenFloat);
    }
};

TEST_P(ConfigWithHalf_miopendriver_regression_float_half_gfx10,
       HalfTest_miopendriver_regression_float_half_gfx10)
{
    if(SkipTest("--half"))
    {
        GTEST_SKIP();
    }
    else
    {
        Run2dDriver(miopenHalf);
    }
};

INSTANTIATE_TEST_SUITE_P(MiopendriverRegressionFloatHalfGfx10,
                         ConfigWithFloat_miopendriver_regression_float_half_gfx10,
                         testing::Values(GetTestCases("--float")));
INSTANTIATE_TEST_SUITE_P(MiopendriverRegressionFloatHalfGfx10,
                         ConfigWithHalf_miopendriver_regression_float_half_gfx10,
                         testing::Values(GetTestCases("--half")));
