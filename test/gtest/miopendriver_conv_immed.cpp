/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

namespace miopen_conv_immed {

std::vector<std::string> GetTestCases(const std::string& modeConvolutionArg)
{
    return std::vector<std::string>{
        {modeConvolutionArg +
         " -n 560 -c 128 -H 15 -W 20 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 "
         "-m conv -g 1 -F 1 -t 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC --iter 1"}};
}

using TestCase = decltype(GetTestCases(""))::value_type;

class GPU_MIOpenDriverConvImmedTest_FP32 : public testing::TestWithParam<std::vector<TestCase>>
{
};

class GPU_MIOpenDriverConvImmedTest_FP16 : public testing::TestWithParam<std::vector<TestCase>>
{
};

class GPU_MIOpenDriverConvImmedTest_BFP16 : public testing::TestWithParam<std::vector<TestCase>>
{
};

void RunMIOpenDriver(const std::vector<TestCase>& testCases)
{
    using e_mask = enabled<Gpu::gfx94X>;
    using d_mask = disabled<Gpu::gfx900, Gpu::gfx906, Gpu::gfx908, Gpu::gfx103X>;
    if(!ShouldRunMIOpenDriverTest<d_mask, e_mask>())
    {
        GTEST_SKIP();
    }

    auto test_dir_path = "/tmp/miopen_immed_ws_warning_test";
    miopen::fs::remove_all(test_dir_path);
    miopen::ProcessEnvironmentMap envs = {};
    envs["MIOPEN_FIND_MODE"]           = "2";
    envs["MIOPEN_LOG_LEVEL"]           = "5";
    envs["MIOPEN_USER_DB_PATH"]        = test_dir_path;

    // AI workspace warning
    testing::internal::CaptureStderr();

    RunMIOpenDriverTestCommand(testCases, envs);

    auto output = testing::internal::GetCapturedStderr();

    EXPECT_THAT(output, Not(testing::HasSubstr("Warning [IsEnoughWorkspace]")));

    miopen::fs::remove_all(test_dir_path);

    // WTI workspace warning
    envs["MIOPEN_SYSTEM_DB_PATH"] = test_dir_path;

    testing::internal::CaptureStderr();

    RunMIOpenDriverTestCommand(testCases, envs);

    output = testing::internal::GetCapturedStderr();

    EXPECT_THAT(output, Not(testing::HasSubstr("Warning [IsEnoughWorkspace]")));

    miopen::fs::remove_all(test_dir_path);
};

} // namespace miopen_conv_immed
using namespace miopen_conv_immed;

TEST_P(GPU_MIOpenDriverConvImmedTest_FP32, MIOpenDriverConvImmed) { RunMIOpenDriver(GetParam()); };

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MIOpenDriverConvImmedTest_FP32,
                         testing::Values(GetTestCases(miopendriver::basearg::conv::Float)));

TEST_P(GPU_MIOpenDriverConvImmedTest_FP16, MIOpenDriverConvImmed) { RunMIOpenDriver(GetParam()); };

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MIOpenDriverConvImmedTest_FP16,
                         testing::Values(GetTestCases(miopendriver::basearg::conv::Half)));

TEST_P(GPU_MIOpenDriverConvImmedTest_BFP16, MIOpenDriverConvImmed) { RunMIOpenDriver(GetParam()); };

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MIOpenDriverConvImmedTest_BFP16,
                         testing::Values(GetTestCases(miopendriver::basearg::conv::BFloat16)));
