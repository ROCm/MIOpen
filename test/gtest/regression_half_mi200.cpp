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
#include "get_handle.hpp"

namespace regression_half_mi200 {
void SetupEnvVar(void)
{
    env::update(MIOPEN_FIND_MODE, "normal");
    env::update(MIOPEN_DEBUG_FIND_ONLY_SOLVER, "ConvHipImplicitGemmForwardV4R4Xdlops");
}

std::vector<std::string> GetArgs(const std::string& param)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    return {begin, end};
}

std::vector<std::string> GetTestCases(const std::string& float_arg)
{
    const std::string& cmd = "test_conv2d ";
    const std::string& args =
        " --verbose --disable-forward --disable-backward-data --disable-validation";

    // clang-format off
    return std::vector<std::string>{
        {cmd + float_arg + " --cmode conv --pmode default --input 120 64 75 75 --weights 128 64 1 1 --pads_strides_dilations 0 0 2 2 1 1" + args}
    };
    // clang-format on
}

using TestCase = decltype(GetTestCases(std::string{}))::value_type;

class GPU_regression_mi200_FP16 : public testing::TestWithParam<std::vector<TestCase>>
{
};

bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::Default>;
    using d_mask = disabled<Gpu::gfx900, Gpu::gfx906, Gpu::gfx908>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

void Run2dDriver(void)
{
    if(!IsTestSupportedForDevice())
    {
        GTEST_SKIP();
    }

    SetupEnvVar();
    std::vector<std::string> params = GPU_regression_mi200_FP16::GetParam();

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

} // namespace regression_half_mi200
using namespace regression_half_mi200;

TEST_P(GPU_regression_mi200_FP16, FloatTest_regression_half_mi200) { Run2dDriver(); };

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_regression_mi200_FP16, testing::Values(GetTestCases("--half")));
