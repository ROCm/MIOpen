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

#include <miopen/miopen.h>
#include <miopen/process.hpp>

namespace miopendriver_regression_half_gfx9 {

std::vector<std::string> GetTestCases()
{
    const std::string& modeConvolutionArg = miopendriver::basearg::conv::Half;

    // clang-format off
    return std::vector<std::string>{
        // Regression test for:
        //   [SWDEV-375617] Fix 3d convolution Host API bug
        //   https://github.com/ROCm/MIOpen/pull/1935
        {modeConvolutionArg + " -n 2 -c 64 --in_d 128 -H 128 -W 128 -k 32 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1"}
    };
    // clang-format on
}

using TestCase = decltype(GetTestCases())::value_type;

class GPU_MIOpenDriverRegressionGfx9Test_FP16 : public testing::TestWithParam<std::vector<TestCase>>
{
};

void RunMIOpenDriver()
{
    using e_mask = enabled<Gpu::gfx94X>;
    using d_mask = disabled<Gpu::gfx900, Gpu::gfx906, Gpu::gfx908, Gpu::gfx103X>;
    if(!ShouldRunMIOpenDriverTest<d_mask, e_mask>())
    {
        GTEST_SKIP();
    }

    RunMIOpenDriverTestCommand(GPU_MIOpenDriverRegressionGfx9Test_FP16::GetParam());
};

} // namespace miopendriver_regression_half_gfx9
using namespace miopendriver_regression_half_gfx9;

TEST_P(GPU_MIOpenDriverRegressionGfx9Test_FP16, MIOpenDriverRegressionHalfGfx9)
{
    RunMIOpenDriver();
};

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MIOpenDriverRegressionGfx9Test_FP16,
                         testing::Values(GetTestCases()));
