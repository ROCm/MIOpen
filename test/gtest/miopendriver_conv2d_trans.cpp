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

namespace miopendriver_conv2d_trans {

std::vector<std::string> GetTestCases(const std::string& modeConvolutionArg)
{
    // clang-format off
    return std::vector<std::string>{
        // Why we have to use the driver:
        //   The transposed convolutions are paritally implemented in the convolution_api layer,
        //   but test apps (including test_conv*) were designed as unit tests and, therefore, do not use the public API.
        // Also serves as a regression test for https://github.com/ROCm/MIOpen/issues/2459.
        {modeConvolutionArg + " -m trans -x 1 -y 1 -W 112 -H 112 -c 64 -n 8 -k 32 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F 0 -V 1"},
        {modeConvolutionArg + " -m trans -x 1 -y 7 -W 17 -H 17 -c 32 -n 128 -k 16 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -g 2 -F 0 -V 1"},
        {modeConvolutionArg + " -m trans -x 10 -y 5 -W 341 -H 79 -c 32 -n 4 -k 8 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 4 -F 0 -V 1"},
        {modeConvolutionArg + " -m trans -x 20 -y 5 -W 700 -H 161 -c 1 -n 4 -k 32 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F 0 -V 1"},
        {modeConvolutionArg + " -m trans -x 3 -y 3 -W 108 -H 108 -c 3 -n 8 -k 64 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -F 0 -V 1"},
        {modeConvolutionArg + " -m trans -x 5 -y 5 -W 175 -H 40 -c 128 -n 16 -k 256 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -F 0 -V 1"},
        {modeConvolutionArg + " -m trans -x 5 -y 5 -W 700 -H 161 -c 1 -n 16 -k 64 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -F 0 -V 1"},
        {modeConvolutionArg + " -m trans -x 7 -y 7 -W 224 -H 224 -c 3 -n 16 -k 64 -p 3 -q 3 -u 2 -v 2 -l 1 -j 1 -g 1 -F 0 -V 1"}
    };
    // clang-format on
}

using TestCase = decltype(GetTestCases(""))::value_type;

class GPU_MIOpenDriverConv2dTransTest_FP32 : public testing::TestWithParam<std::vector<TestCase>>
{
};

class GPU_MIOpenDriverConv2dTransTest_FP16 : public testing::TestWithParam<std::vector<TestCase>>
{
};

class GPU_MIOpenDriverConv2dTransTest_BFP16 : public testing::TestWithParam<std::vector<TestCase>>
{
};

void RunMIOpenDriver(const std::vector<TestCase>& testCases)
{
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx103X, Gpu::gfx110X>;
    using d_mask = disabled<Gpu::gfx900>;
    if(!ShouldRunMIOpenDriverTest<d_mask, e_mask>())
    {
        GTEST_SKIP();
    }

    RunMIOpenDriverTestCommand(testCases);
};

} // namespace miopendriver_conv2d_trans
using namespace miopendriver_conv2d_trans;

TEST_P(GPU_MIOpenDriverConv2dTransTest_FP32, MIOpenDriverConv2dTrans)
{
    RunMIOpenDriver(GetParam());
};

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MIOpenDriverConv2dTransTest_FP32,
                         testing::Values(GetTestCases(miopendriver::basearg::conv::Float)));

TEST_P(GPU_MIOpenDriverConv2dTransTest_FP16, MIOpenDriverConv2dTrans)
{
    RunMIOpenDriver(GetParam());
};

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MIOpenDriverConv2dTransTest_FP16,
                         testing::Values(GetTestCases(miopendriver::basearg::conv::Half)));

TEST_P(GPU_MIOpenDriverConv2dTransTest_BFP16, MIOpenDriverConv2dTrans)
{
    RunMIOpenDriver(GetParam());
};

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MIOpenDriverConv2dTransTest_BFP16,
                         testing::Values(GetTestCases(miopendriver::basearg::conv::BFloat16)));
