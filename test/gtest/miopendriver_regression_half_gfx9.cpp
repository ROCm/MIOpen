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
#include "miopendriver_common.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <gtest/gtest_common.hpp>

#include <miopen/env.hpp>
#include <miopen/miopen.h>
#include <miopen/process.hpp>

using ::testing::HasSubstr;
using ::testing::Not;

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_WITH_MIOPENDRIVER)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPENDRIVER_MODE_CONV)

namespace miopendriver_regression_half_gfx9 {

std::vector<std::string> GetTestCases()
{
    const std::string& cmd       = MIOpenDriverExePath().string();
    const std::string& modeConvolutionArg = miopen::GetStringEnv(ENV(MIOPENDRIVER_MODE_CONV));

    // clang-format off
    return std::vector<std::string>{
        // Regression test for:
        //   [SWDEV-375617] Fix 3d convolution Host API bug
        //   https://github.com/ROCm/MIOpen/pull/1935
        {cmd + " " + modeConvolutionArg + " -n 2 -c 64 --in_d 128 -H 128 -W 128 -k 32 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1"}
    };
    // clang-format on
}

using TestCase = decltype(GetTestCases())::value_type;

class MIOpenDriverRegressionHalfGfx9Test : public testing::TestWithParam<std::vector<TestCase>>
{
};

bool IsTestSupportedForDevice()
{
    using namespace miopen::debug;
    using e_mask = enabled<Gpu::gfx94X>;
    using d_mask = disabled<Gpu::gfx900, Gpu::gfx906, Gpu::gfx908, Gpu::gfx103X>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

void RunMIOpenDriver()
{
    bool runTestSuite = miopen::IsEnabled(ENV(MIOPEN_TEST_WITH_MIOPENDRIVER))
                            && IsTestSupportedForDevice()
                                && miopen::IsEnabled(ENV(MIOPEN_TEST_ALL))
                                    && miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG)) == "--half";

    if(!runTestSuite)
    {
        GTEST_SKIP();
    }

    std::vector<std::string> params = MIOpenDriverRegressionHalfGfx9Test::GetParam();
    for(const auto& testCommand : params)
    {
        int commandResult = 0;
        miopen::Process p{testCommand};

        // TODO bharriso - get decision for capturing output, and either remove this if we can ignore, 
        //                 or add capturing output + check here for regrex not matching FAILED. 
        EXPECT_NO_THROW(commandResult = p());
        EXPECT_EQ(commandResult, 0) << "MIOpenDriver exited with non-zero value when running command: " << testCommand;
    }
};

} // namespace miopendriver_regression_half_gfx9
using namespace miopendriver_regression_half_gfx9;

TEST_P(MIOpenDriverRegressionHalfGfx9Test, MIOpenDriverRegressionHalfGfx9) { RunMIOpenDriver(); };

INSTANTIATE_TEST_SUITE_P(MIOpenDriverRegressionHalfGfx9TestSet,
                         MIOpenDriverRegressionHalfGfx9Test,
                         testing::Values(GetTestCases()));
