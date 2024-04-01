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
#include "../conv2d.hpp"
#include <miopen/miopen.h>
#include <gtest/gtest_common.hpp>
#include <gtest/gtest.h>
#include <miopen/env.hpp>
#include "get_handle.hpp"

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)

namespace regression_issue_2012 {
void SetupEnvVar(void) { miopen::UpdateEnvVar(ENV(MIOPEN_FIND_MODE), std::string("normal")); }

std::vector<std::string> GetArgs(const std::string& param)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    return {begin, end};
}

std::vector<std::string> GetTestCases(void)
{
    const std::string& cmd       = "test_conv2d ";
    const std::string& float_arg = miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG));
    const std::string& args =
        " --verbose --disable-forward --disable-backward-data --disable-validation";

    // clang-format off
    return std::vector<std::string>{
        {cmd + float_arg + " --cmode conv --pmode default --group-count 1 --input 128, 832, 7,  7  --weights 32,  832, 1, 1 --pads_strides_dilations 0 0 1 1 1 1" + args},
        {cmd + float_arg + " --cmode conv --pmode default --group-count 1 --input 64,  192, 28, 28 --weights 64,  192, 1, 1 --pads_strides_dilations 0 0 1 1 1 1" + args},
        {cmd + float_arg + " --cmode conv --pmode default --group-count 1 --input 64,  256, 28, 28 --weights 128, 256, 1, 1 --pads_strides_dilations 0 0 1 1 1 1" + args},
        {cmd + float_arg + " --cmode conv --pmode default --group-count 1 --input 64,  480, 14, 14 --weights 64,  480, 1, 1 --pads_strides_dilations 0 0 1 1 1 1" + args},
        {cmd + float_arg + " --cmode conv --pmode default --group-count 1 --input 64,  512, 14, 14 --weights 128, 512, 1, 1 --pads_strides_dilations 0 0 1 1 1 1" + args},
        {cmd + float_arg + " --cmode conv --pmode default --group-count 1 --input 64,  512, 28, 28 --weights 128, 512, 1, 1 --pads_strides_dilations 0 0 1 1 1 1" + args},
        {cmd + float_arg + " --cmode conv --pmode default --group-count 1 --input 64,  64,  56, 56 --weights 256, 64,  1, 1 --pads_strides_dilations 0 0 1 1 1 1" + args}
    };
    // clang-format on
}

using TestCase = decltype(GetTestCases())::value_type;

class ConfigWithFloat_regression_issue_2012 : public testing::TestWithParam<std::vector<TestCase>>
{
};

bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::Default>;
    using d_mask = disabled<Gpu::gfx900, Gpu::gfx906, Gpu::gfx90A>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

void Run2dDriver(void)
{
    if(!(IsTestSupportedForDevice() &&
         miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG)) == "--float"))
    {
        GTEST_SKIP();
    }
    SetupEnvVar();
    std::vector<std::string> params = ConfigWithFloat_regression_issue_2012::GetParam();

    for(const auto& test_value : params)
    {
        std::vector<std::string> tokens = GetArgs(test_value);
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

} // namespace regression_issue_2012
using namespace regression_issue_2012;

TEST_P(ConfigWithFloat_regression_issue_2012, FloatTest_regression_issue_2012) { Run2dDriver(); };

INSTANTIATE_TEST_SUITE_P(RegressionIssue2012,
                         ConfigWithFloat_regression_issue_2012,
                         testing::Values(GetTestCases()));
