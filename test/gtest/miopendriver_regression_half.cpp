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

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_WITH_MIOPENDRIVER)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPENDRIVER_MODE_POOL)

namespace miopendriver_regression_half {

std::vector<std::string> GetTestCases()
{
    const std::string& modePoolingArg = miopen::GetStringEnv(ENV(MIOPENDRIVER_MODE_POOL));

    // clang-format off
    return std::vector<std::string>{
        // WORKAROUND_ISSUE_2110_2: tests for 2110 and 2160 shall be added to "test_pooling3d --all" but this is
        // impossible until backward pooling limitation (issue #2110 (2)) is fully fixed.
        // Partial (3D only) regression test for https://github.com/ROCm/MIOpen/issues/2160.
        {modePoolingArg + " -M 0 --input 1x64x41x40x70 -y 41 -x 40 -Z 70 -m avg -F 1 -t 1 -i 1"},
        // Partial (3D only) regression test for https://github.com/ROCm/MIOpen/issues/2110 (1).
        {modePoolingArg + " -M 0 --input 1x64x41x40x100 -y 4 -x 4 -Z 100 -m max -F 1 -t 1 -i 1"}
    };
    // clang-format on
}

using TestCase = decltype(GetTestCases())::value_type;

class MIOpenDriverRegressionHalfTest : public testing::TestWithParam<std::vector<TestCase>>
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
    bool runTestSuite = miopen::IsEnabled(ENV(MIOPEN_TEST_WITH_MIOPENDRIVER)) &&
                        IsTestSupportedForDevice() && miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) &&
                        miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG)) == "--half";

    if(!runTestSuite)
    {
        GTEST_SKIP();
    }

    RunMIOpenDriverTestCommand(MIOpenDriverRegressionHalfTest::GetParam());
};

} // namespace miopendriver_regression_half
using namespace miopendriver_regression_half;

TEST_P(MIOpenDriverRegressionHalfTest, MIOpenDriverRegressionHalf) { RunMIOpenDriver(); };

INSTANTIATE_TEST_SUITE_P(MIOpenDriverRegressionHalfTestSet,
                         MIOpenDriverRegressionHalfTest,
                         testing::Values(GetTestCases()));
