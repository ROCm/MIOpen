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
#include "../conv3d.hpp"
#include <miopen/miopen.h>
#include <gtest/gtest_common.hpp>
#include <gtest/gtest.h>
#include <miopen/env.hpp>
#include "get_handle.hpp"

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)

namespace regression_half_vega {
void SetupEnvVar(void)
{
    miopen::UpdateEnvVar(ENV(MIOPEN_FIND_MODE), std::string("normal"));
    miopen::UpdateEnvVar(ENV(MIOPEN_DEBUG_FIND_ONLY_SOLVER), std::string("GemmBwdRest"));
}

void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

std::vector<std::string> GetTestCases(void)
{
    const std::string& cmd       = "test_conv3d ";
    const std::string& float_arg = miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG));
    const std::string& conv_verbose_b =
        float_arg + " --verbose --disable-forward --disable-backward-weights";

    // clang-format off
    return std::vector<std::string>{
        {cmd + conv_verbose_b + " --input 64 256 28 28 --weights 64  64  1 1 --pads_strides_dilations 0 0 1 1 1 1 --group-count 4"}
    };
    // clang-format on
}

using TestCase = decltype(GetTestCases())::value_type;

class ConfigWithHalf_regression_half_vega : public testing::TestWithParam<std::vector<TestCase>>
{
};

bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::Default>;
    using d_mask = disabled<Gpu::gfx908, Gpu::gfx90A>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

void Run2dDriver(void)
{
    if(!(IsTestSupportedForDevice()                      //
         && (miopen::IsUnset(ENV(MIOPEN_TEST_ALL))       // standalone run
             || (miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) // or --float full tests enabled
                 && miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG)) == "--half"))))
    {
        GTEST_SKIP();
    }
    std::vector<std::string> params = ConfigWithHalf_regression_half_vega::GetParam();

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

} // namespace regression_half_vega
using namespace regression_half_vega;

TEST_P(ConfigWithHalf_regression_half_vega, FloatTest_regression_half_vega) { Run2dDriver(); };

INSTANTIATE_TEST_SUITE_P(RegressionHalfVega,
                         ConfigWithHalf_regression_half_vega,
                         testing::Values(GetTestCases()));
