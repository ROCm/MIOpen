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

namespace miopendriver_conv2d_trans {

std::vector<std::string> GetTestCases()
{
    const std::string& cmd = MIOpenDriverExePath().string();
    const std::string& modeConvolutionArg =
        miopen::GetStringEnv(MIOPEN_ENV(MIOPENDRIVER_MODE_CONV));

    // clang-format off
    return std::vector<std::string>{
        // Why we have to use the driver:
        //   The transposed convolutions are paritally implemented in the convolution_api layer,
        //   but test apps (including test_conv*) were designed as unit tests and, therefore, do not use the public API.
        // Also serves as a regression test for https://github.com/ROCm/MIOpen/issues/2459.
        {cmd + " " + modeConvolutionArg + " -m trans -x 1 -y 1 -W 112 -H 112 -c 64 -n 8 -k 32 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F 0 -V 1"},
        {cmd + " " + modeConvolutionArg + " -m trans -x 1 -y 7 -W 17 -H 17 -c 32 -n 128 -k 16 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -g 2 -F 0 -V 1"},
        {cmd + " " + modeConvolutionArg + " -m trans -x 10 -y 5 -W 341 -H 79 -c 32 -n 4 -k 8 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 4 -F 0 -V 1"},
        {cmd + " " + modeConvolutionArg + " -m trans -x 20 -y 5 -W 700 -H 161 -c 1 -n 4 -k 32 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F 0 -V 1"},
        {cmd + " " + modeConvolutionArg + " -m trans -x 3 -y 3 -W 108 -H 108 -c 3 -n 8 -k 64 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -F 0 -V 1"},
        {cmd + " " + modeConvolutionArg + " -m trans -x 5 -y 5 -W 175 -H 40 -c 128 -n 16 -k 256 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -F 0 -V 1"},
        {cmd + " " + modeConvolutionArg + " -m trans -x 5 -y 5 -W 700 -H 161 -c 1 -n 16 -k 64 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -F 0 -V 1"},
        {cmd + " " + modeConvolutionArg + " -m trans -x 7 -y 7 -W 224 -H 224 -c 3 -n 16 -k 64 -p 3 -q 3 -u 2 -v 2 -l 1 -j 1 -g 1 -F 0 -V 1"}
    };
    // clang-format on
}

using TestCase = decltype(GetTestCases())::value_type;

class MIOpenDriverConv2dTransTest : public testing::TestWithParam<std::vector<TestCase>>
{
};

bool IsTestSupportedForDevice()
{
    using namespace miopen::debug;
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx103X, Gpu::gfx110X>;
    using d_mask = disabled<Gpu::gfx900>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

void RunMIOpenDriver()
{
    bool runTestSuite = miopen::IsEnabled(MIOPEN_ENV(MIOPEN_TEST_WITH_MIOPENDRIVER)) &&
                        IsTestSupportedForDevice() &&
                        miopen::IsEnabled(MIOPEN_ENV(MIOPEN_TEST_ALL)) &&
                        (miopen::GetStringEnv(MIOPEN_ENV(MIOPEN_TEST_FLOAT_ARG)) == "--float" ||
                         miopen::GetStringEnv(MIOPEN_ENV(MIOPEN_TEST_FLOAT_ARG)) == "--half" ||
                         miopen::GetStringEnv(MIOPEN_ENV(MIOPEN_TEST_FLOAT_ARG)) == "--bf16");

    if(!runTestSuite)
    {
        GTEST_SKIP();
    }

    std::vector<std::string> params = MIOpenDriverConv2dTransTest::GetParam();
    for(const auto& testCommand : params)
    {
        int commandResult = 0;
        miopen::Process p{testCommand};

        // TODO bharriso - get decision for capturing output, and either remove this if we can
        // ignore,
        //                 or add capturing output + check here for regrex not matching FAILED.
        EXPECT_NO_THROW(commandResult = p());
        EXPECT_EQ(commandResult, 0)
            << "MIOpenDriver exited with non-zero value when running command: " << testCommand;
    }
};

} // namespace miopendriver_conv2d_trans
using namespace miopendriver_conv2d_trans;

TEST_P(MIOpenDriverConv2dTransTest, MIOpenDriverConv2dTrans) { RunMIOpenDriver(); };

INSTANTIATE_TEST_SUITE_P(MIOpenDriverConv2dTransTestSet,
                         MIOpenDriverConv2dTransTest,
                         testing::Values(GetTestCases()));
