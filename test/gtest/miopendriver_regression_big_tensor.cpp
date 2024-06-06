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
MIOPEN_DECLARE_ENV_VAR_STR(MIOPENDRIVER_MODE_CONV)

namespace miopendriver_regression_big_tensor {

std::vector<std::string> GetTestCases()
{
    const std::string& modeConvolutionArg = miopen::GetStringEnv(ENV(MIOPENDRIVER_MODE_CONV));

    // clang-format off
    return std::vector<std::string>{
        // Regression test for https://github.com/ROCm/MIOpen/issues/1661
        // Issue #1697: this is large test which has to run in serial and not enabled on gfx900/gfx906
        {modeConvolutionArg + " -W 5078 -H 4903 -c 24 -n 5 -k 1 --fil_w 3 --fil_h 3 --pad_w 6 --pad_h 4 -F 1"}
    };
    // clang-format on
}

using TestCase = decltype(GetTestCases())::value_type;

class MIOpenDriverRegressionBigTensorTest : public testing::TestWithParam<std::vector<TestCase>>
{
};

bool IsTestSupportedForDevice()
{
    using namespace miopen::debug;
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx103X>;
    using d_mask = disabled<Gpu::gfx900, Gpu::gfx906, Gpu::gfx908>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

void RunMIOpenDriver()
{
    bool runTestSuite = miopen::IsEnabled(ENV(MIOPEN_TEST_WITH_MIOPENDRIVER)) &&
                        IsTestSupportedForDevice() && miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) &&
                        miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG)) == "--float";

    if(!runTestSuite)
    {
        GTEST_SKIP();
    }

    RunMIOpenDriverTestCommand(MIOpenDriverRegressionBigTensorTest::GetParam());
};

} // namespace miopendriver_regression_big_tensor
using namespace miopendriver_regression_big_tensor;

TEST_P(MIOpenDriverRegressionBigTensorTest, MIOpenDriverRegressionBigTensor) { RunMIOpenDriver(); };

INSTANTIATE_TEST_SUITE_P(MIOpenDriverRegressionBigTensorTestSet,
                         MIOpenDriverRegressionBigTensorTest,
                         testing::Values(GetTestCases()));
