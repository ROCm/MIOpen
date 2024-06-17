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

namespace miopendriver_regression_issue_2047 {

std::vector<std::string> GetTestCases(const std::string& modeConvolutionArg)
{
    // clang-format off
    return std::vector<std::string>{
        // Regression test for: MIOpenIm3d2Col stuck with ROCm update, https://github.com/ROCm/MIOpen/issues/2047
        {modeConvolutionArg + " -n 1 -c 1 --in_d 2 -H 1 -W 2 -k 2 --fil_d 2 -y 1 -x 2 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -i 1 -t 1 -w 1"}
    };
    // clang-format on
}

using TestCase = decltype(GetTestCases(""))::value_type;

class MIOpenDriverRegressionIssue2047TestFloat
    : public testing::TestWithParam<std::vector<TestCase>>
{
};

class MIOpenDriverRegressionIssue2047TestHalf : public testing::TestWithParam<std::vector<TestCase>>
{
};

class MIOpenDriverRegressionIssue2047TestBFloat16
    : public testing::TestWithParam<std::vector<TestCase>>
{
};

class MIOpenDriverRegressionIssue2047TestInt8 : public testing::TestWithParam<std::vector<TestCase>>
{
};

void RunMIOpenDriver(const std::string& floatArg, const std::vector<TestCase>& testCases)
{
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx103X, Gpu::gfx110X>;
    using d_mask = disabled<Gpu::Default>;
    if(!ShouldRunMIOpenDriverTest<d_mask, e_mask>(floatArg, false))
    {
        GTEST_SKIP();
    }

    RunMIOpenDriverTestCommand(
        testCases,
        {{"MIOPEN_FIND_MODE", "normal"}, {"MIOPEN_DEBUG_FIND_ONLY_SOLVER", "GemmFwdRest"}});
};

} // namespace miopendriver_regression_issue_2047
using namespace miopendriver_regression_issue_2047;

TEST_P(MIOpenDriverRegressionIssue2047TestFloat, MIOpenDriverRegressionIssue2047)
{
    RunMIOpenDriver("--float", GetParam());
};

INSTANTIATE_TEST_SUITE_P(MIOpenDriverRegressionIssue2047TestSet,
                         MIOpenDriverRegressionIssue2047TestFloat,
                         testing::Values(GetTestCases(miopendriver::basearg::conv::Float)));

TEST_P(MIOpenDriverRegressionIssue2047TestHalf, MIOpenDriverRegressionIssue2047)
{
    RunMIOpenDriver("--half", GetParam());
};

INSTANTIATE_TEST_SUITE_P(MIOpenDriverRegressionIssue2047TestSet,
                         MIOpenDriverRegressionIssue2047TestHalf,
                         testing::Values(GetTestCases(miopendriver::basearg::conv::Half)));

TEST_P(MIOpenDriverRegressionIssue2047TestBFloat16, MIOpenDriverRegressionIssue2047)
{
    RunMIOpenDriver("--bfloat16", GetParam());
};

INSTANTIATE_TEST_SUITE_P(MIOpenDriverRegressionIssue2047TestSet,
                         MIOpenDriverRegressionIssue2047TestBFloat16,
                         testing::Values(GetTestCases(miopendriver::basearg::conv::BFloat16)));

TEST_P(MIOpenDriverRegressionIssue2047TestInt8, MIOpenDriverRegressionIssue2047)
{
    RunMIOpenDriver("--int8", GetParam());
};

INSTANTIATE_TEST_SUITE_P(MIOpenDriverRegressionIssue2047TestSet,
                         MIOpenDriverRegressionIssue2047TestInt8,
                         testing::Values(GetTestCases(miopendriver::basearg::conv::Int8)));
