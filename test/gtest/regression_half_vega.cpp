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

namespace env = miopen::env;

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)

namespace regression_half_vega {
void SetupEnvVar()
{
    env::update(MIOPEN_FIND_MODE, "normal");
    env::update(MIOPEN_DEBUG_FIND_ONLY_SOLVER, "GemmBwdRest");
}

std::vector<std::string> GetArgs(const std::string& param)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    return {begin, end};
}

std::vector<std::string> GetTestCases()
{
    const std::string& cmd = "test_conv3d ";
    std::string float_arg  = env::value(MIOPEN_TEST_FLOAT_ARG);
    if(float_arg.empty())
        float_arg = "--half";
    const std::string& conv_verbose_b =
        float_arg + " --verbose --disable-forward --disable-backward-weights";

    // clang-format off
    return std::vector<std::string>{
        {cmd + conv_verbose_b + " --cmode conv --pmode default --group-count 1 --batch_size 2 --input_channels 64 --output_channels 32 --spatial_dim_elements 128 128 128 --filter_dims 3 3 3 --pads_strides_dilations 1 1 1 1 1 1 1 1 1 --trans_output_pads 0 0 0 --in_layout NCDHW --fil_layout NCDHW --out_layout NCDHW"}
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

void Run2dDriver()
{
    if(!(IsTestSupportedForDevice()            //
         && (!MIOPEN_TEST_ALL                  // standalone run
             || (env::enabled(MIOPEN_TEST_ALL) // or --float full tests enabled
                 && env::value(MIOPEN_TEST_FLOAT_ARG) == "--half"))))
    {
        GTEST_SKIP();
    }
    SetupEnvVar();
    std::vector<std::string> params = ConfigWithHalf_regression_half_vega::GetParam();

    for(const auto& test_value : params)
    {
        std::vector<std::string> tokens = GetArgs(test_value);
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
