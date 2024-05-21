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

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_WITH_MIOPENDRIVER)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPENDRIVER_MODE_GEMM)

namespace miopendriver_gemm {

std::vector<std::string> GetTestCases()
{
    const std::string& cmd          = MIOpenDriverExePath().string();
    const std::string& modeGemmnArg = miopen::GetStringEnv(MIOPEN_ENV(MIOPENDRIVER_MODE_GEMM));

    // clang-format off
    return std::vector<std::string>{
        {cmd + " " + modeGemmnArg + " -m 256 -n 512 -k 1024 -i 1 -V 1"}
    };
    // clang-format on
}

using TestCase = decltype(GetTestCases())::value_type;

class MIOpenDriverGemmTest : public testing::TestWithParam<std::vector<TestCase>>
{
};

bool IsTestSupportedForDevice()
{
    using namespace miopen::debug;
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx103X, Gpu::gfx110X>;
    using d_mask = disabled<Gpu::Default>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

void RunMIOpenDriver()
{
    bool runTestSuite = miopen::IsEnabled(MIOPEN_ENV(MIOPEN_TEST_WITH_MIOPENDRIVER)) &&
                        IsTestSupportedForDevice() &&
                        (miopen::GetStringEnv(MIOPEN_ENV(MIOPEN_TEST_FLOAT_ARG)) == "--float" ||
                         miopen::GetStringEnv(MIOPEN_ENV(MIOPEN_TEST_FLOAT_ARG)) == "--half");

    if(!runTestSuite)
    {
        GTEST_SKIP();
    }

    std::vector<std::string> commands = MIOpenDriverGemmTest::GetParam();
    for(const auto& testCommand : commands)
    {
        int commandResult = 0;
        miopen::Process p{testCommand};

        // TODO bharriso - get decision for capturing output, and either remove this if we can
        // ignore,
        //                 or add capturing output + check here.
        EXPECT_NO_THROW(commandResult = p());
        EXPECT_EQ(commandResult, 0)
            << "MIOpenDriver exited with non-zero value when running command: " << testCommand;
    }
};

} // namespace miopendriver_gemm
using namespace miopendriver_gemm;

TEST_P(MIOpenDriverGemmTest, MIOpenDriverGemm) { RunMIOpenDriver(); };

INSTANTIATE_TEST_SUITE_P(MIOpenDriverGemmTestSet,
                         MIOpenDriverGemmTest,
                         testing::Values(GetTestCases()));
