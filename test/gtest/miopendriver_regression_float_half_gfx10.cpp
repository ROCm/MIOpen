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

#include <miopen/env.hpp>
#include <miopen/miopen.h>
#include <miopen/process.hpp>

namespace miopendriver_regression_float_half_gfx10 {

std::vector<std::string> GetTestCases(const std::string& modeBatchNormArg)
{
    // clang-format off
    return std::vector<std::string>{
        // Regression test for:
        // [Navi21] Fixing Batchnorm backward precision issues by adjusting workgroup size (SWDEV-292187, SWDEV-319919)
        // https://github.com/ROCm/MIOpen/pull/1386
        {modeBatchNormArg + " -n 256 -c 512 -H 18 -W 18 -m 1 --forw 0 -b 1 -r 1"},
        {modeBatchNormArg + " -n 256 -c 512 -H 28 -W 28 -m 1 --forw 0 -b 1 -r 1"}
    };
    // clang-format on
}

using TestCase = decltype(GetTestCases(""))::value_type;

class GPU_MIOpenDriverRegressionGfx10Test_FP32
    : public testing::TestWithParam<std::vector<TestCase>>
{
};

class GPU_MIOpenDriverRegressionGfx10Test_FP16
    : public testing::TestWithParam<std::vector<TestCase>>
{
};

void RunMIOpenDriver(const std::vector<TestCase>& testCases)
{
    using e_mask = enabled<Gpu::gfx103X>;
    using d_mask = disabled<Gpu::gfx900, Gpu::gfx906, Gpu::gfx908, Gpu::gfx90A>;
    if(!ShouldRunMIOpenDriverTest<d_mask, e_mask>())
    {
        GTEST_SKIP();
    }

    RunMIOpenDriverTestCommand(testCases);
};

} // namespace miopendriver_regression_float_half_gfx10
using namespace miopendriver_regression_float_half_gfx10;

TEST_P(GPU_MIOpenDriverRegressionGfx10Test_FP32, MIOpenDriverRegressionFloatHalfGfx10)
{
    RunMIOpenDriver(GetParam());
};

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MIOpenDriverRegressionGfx10Test_FP32,
                         testing::Values(GetTestCases(miopendriver::basearg::bn::Float)));

TEST_P(GPU_MIOpenDriverRegressionGfx10Test_FP16, MIOpenDriverRegressionFloatHalfGfx10)
{
    RunMIOpenDriver(GetParam());
};

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MIOpenDriverRegressionGfx10Test_FP16,
                         testing::Values(GetTestCases(miopendriver::basearg::bn::Half)));
