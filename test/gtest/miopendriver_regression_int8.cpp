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

namespace miopendriver_regression_int8 {

std::vector<std::string> GetTestCases()
{
    const std::string& modeConvolutionArg = miopendriver::basearg::conv::Int8;

    // clang-format off
    return std::vector<std::string>{
        {modeConvolutionArg + " --forw 1 --in_layout NCHW --out_layout NCHW --fil_layout NCHW -n 256 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1"}
    };
    // clang-format on
}

using TestCase = decltype(GetTestCases())::value_type;

class GPU_MIOpenDriverRegressionTest_I8 : public testing::TestWithParam<std::vector<TestCase>>
{
};

void RunMIOpenDriver()
{
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx103X, Gpu::gfx110X>;
    using d_mask = disabled<Gpu::Default>;
    if(!ShouldRunMIOpenDriverTest<d_mask, e_mask>("--int8", true))
    {
        GTEST_SKIP();
    }

    miopen::ProcessEnvironmentMap environmentVariables = {
        {"MIOPEN_FIND_MODE", "1"}, {"MIOPEN_DEBUG_FIND_ONLY_SOLVER", "ConvDirectNaiveConvFwd"}};

    RunMIOpenDriverTestCommand(GPU_MIOpenDriverRegressionTest_I8::GetParam(), environmentVariables);
};

} // namespace miopendriver_regression_int8
using namespace miopendriver_regression_int8;

TEST_P(GPU_MIOpenDriverRegressionTest_I8, MIOpenDriverRegressionInt8) { RunMIOpenDriver(); };

INSTANTIATE_TEST_SUITE_P(Full, GPU_MIOpenDriverRegressionTest_I8, testing::Values(GetTestCases()));
