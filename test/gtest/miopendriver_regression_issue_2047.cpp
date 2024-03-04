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

namespace miopendriver_regression_issue_2047 {
void SetupEnvVar(void)
{
    miopen::UpdateEnvVar(ENV(MIOPEN_FIND_MODE), std::string_view("normal"));
    miopen::UpdateEnvVar(ENV(MIOPEN_DEBUG_FIND_ONLY_SOLVER), std::string_view("GemmFwdRest"));
}

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
        cmd.append("conv");
    else if(precision == "--half")
        cmd.append("convfp16");
    else if(precision == "--bfloat16")
        cmd.append("convbfp16");
    else if(precision == "--int8")
        cmd.append("convint8");

    // clang-format off
    return std::vector<std::string>{
        {cmd + " -n 1 -c 1 --in_d 2 -H 1 -W 2 -k 2 --fil_d 2 -y 1 -x 2 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -i 1 -t 1 -w 1"}
    };
    // clang-format on
}

using TestCase = decltype(GetTestCases())::value_type;

class ConfigWithFloat_miopendriver_regression_issue_2047
    : public testing::TestWithParam<std::vector<TestCase>>
{
};

class ConfigWithHalf_miopendriver_regression_issue_2047
    : public testing::TestWithParam<std::vector<TestCase>>
{
};

class ConfigWithBf16_miopendriver_regression_issue_2047
    : public testing::TestWithParam<std::vector<TestCase>>
{
};

class ConfigWithInt8_miopendriver_regression_issue_2047
    : public testing::TestWithParam<std::vector<TestCase>>
{
};

bool IsTestSupportedForDevice()
{
    using namespace miopen::debug;
    using e_mask = enabled<Gpu::gfx103X, Gpu::gfx110X>;
    using d_mask = disabled<Gpu::gfx900>;
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
        params = ConfigWithFloat_miopendriver_regression_issue_2047::GetParam();
        break;
    case miopenHalf: params = ConfigWithHalf_miopendriver_regression_issue_2047::GetParam(); break;
    case miopenBFloat16:
        params = ConfigWithBf16_miopendriver_regression_issue_2047::GetParam();
        break;
    case miopenInt8: params = ConfigWithInt8_miopendriver_regression_issue_2047::GetParam(); break;
    case miopenInt32:
    case miopenFloat8:
    case miopenBFloat8:
    case miopenDouble:
    default:
        FAIL() << "miopenInt32, miopenFloat8, miopenBFloat8, miopenDouble data type "
                  "not supported by miopendriver_regression_issue_2047 test";
    }

    SetupEnvVar();

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

} // namespace miopendriver_regression_issue_2047
using namespace miopendriver_regression_issue_2047;

TEST_P(ConfigWithFloat_miopendriver_regression_issue_2047,
       FloatTest_miopendriver_regression_issue_2047)
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

TEST_P(ConfigWithHalf_miopendriver_regression_issue_2047,
       HalfTest_miopendriver_regression_issue_2047)
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

TEST_P(ConfigWithBf16_miopendriver_regression_issue_2047,
       Bf16Test_miopendriver_regression_issue_2047)
{
    if(SkipTest("--bfloat16"))
    {
        GTEST_SKIP();
    }
    else
    {
        Run2dDriver(miopenBFloat16);
    }
};

TEST_P(ConfigWithInt8_miopendriver_regression_issue_2047,
       Int8Test_miopendriver_regression_issue_2047)
{
    if(SkipTest("--int8"))
    {
        GTEST_SKIP();
    }
    else
    {
        Run2dDriver(miopenInt8);
    }
};

INSTANTIATE_TEST_SUITE_P(MiopendriverConv2dTran,
                         ConfigWithFloat_miopendriver_regression_issue_2047,
                         testing::Values(GetTestCases("--float")));
INSTANTIATE_TEST_SUITE_P(MiopendriverConv2dTran,
                         ConfigWithHalf_miopendriver_regression_issue_2047,
                         testing::Values(GetTestCases("--half")));
INSTANTIATE_TEST_SUITE_P(MiopendriverConv2dTran,
                         ConfigWithBf16_miopendriver_regression_issue_2047,
                         testing::Values(GetTestCases("--bfloat16")));
INSTANTIATE_TEST_SUITE_P(MiopendriverConv2dTran,
                         ConfigWithInt8_miopendriver_regression_issue_2047,
                         testing::Values(GetTestCases("--int8")));
