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
#include "../reduce_test.hpp"
#include <miopen/miopen.h>
#include <gtest/gtest_common.hpp>
#include <gtest/gtest.h>
#include "get_handle.hpp"

namespace reduce_custom_fp32 {
std::vector<std::string> GetArgs(const std::string& param)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    return {begin, end};
}

std::vector<std::string> GetTestCases(const std::string& float_arg)
{
    const std::string& cmd = "test_reduce_test ";

    // clang-format off
    return std::vector<std::string>{
        {cmd + float_arg + " --scales 1 0 --CompType 1 --D 1024 30528 1 --I 0 --N 1 --ReduceOp 0 --R 0 1 2"}
    };
    // clang-format on
}

using TestCase = decltype(GetTestCases(""))::value_type;

class GPU_reduce_custom_fp32_FP32 : public testing::TestWithParam<std::vector<TestCase>>
{
};

class GPU_reduce_custom_fp32_FP16 : public testing::TestWithParam<std::vector<TestCase>>
{
};

class GPU_reduce_custom_fp32_BFP16 : public testing::TestWithParam<std::vector<TestCase>>
{
};

class GPU_reduce_custom_fp32_I8 : public testing::TestWithParam<std::vector<TestCase>>
{
};

bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx103X, Gpu::gfx110X>;
    using d_mask = disabled<Gpu::Default>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

void Run2dDriver(void)
{
    if(!IsTestSupportedForDevice())
    {
        GTEST_SKIP();
    }
    std::vector<std::string> params = GPU_reduce_custom_fp32_FP32::GetParam();

    for(const auto& test_value : params)
    {
        std::vector<std::string> tokens = GetArgs(test_value);
        std::vector<const char*> ptrs;

        std::transform(tokens.begin(), tokens.end(), std::back_inserter(ptrs), [](const auto& str) {
            return str.data();
        });
        testing::internal::CaptureStderr();
        test_drive<reduce_driver>(ptrs.size(), ptrs.data());
        auto capture = testing::internal::GetCapturedStderr();
        std::cout << capture;
    }
};

} // namespace reduce_custom_fp32
using namespace reduce_custom_fp32;

TEST_P(GPU_reduce_custom_fp32_FP32, FloatTest_reduce_custom_fp32) { Run2dDriver(); };
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_reduce_custom_fp32_FP32,
                         testing::Values(GetTestCases("--float")));

TEST_P(GPU_reduce_custom_fp32_FP16, HalfTest_reduce_custom_fp16) { Run2dDriver(); };
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_reduce_custom_fp32_FP16,
                         testing::Values(GetTestCases("--half")));

TEST_P(GPU_reduce_custom_fp32_BFP16, BHalfTest_reduce_custom_bfp16) { Run2dDriver(); };
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_reduce_custom_fp32_BFP16,
                         testing::Values(GetTestCases("--bfloat16")));

TEST_P(GPU_reduce_custom_fp32_I8, IntTest_reduce_custom_i8) { Run2dDriver(); };
INSTANTIATE_TEST_SUITE_P(Full, GPU_reduce_custom_fp32_I8, testing::Values(GetTestCases("--int8")));
