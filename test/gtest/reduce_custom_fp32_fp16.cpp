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

namespace reduce_custom_fp32_fp16 {
std::vector<std::string> GetArgs(const std::string& param)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    return {begin, end};
}

std::vector<std::string> GetTestCases(const std::string& precision)
{
    const std::string& cmd = "test_reduce_test " + precision;

    // clang-format off
    return std::vector<std::string>{
        {cmd + " --scales 1 0 --CompType 1 --D 8 2 1 --I 0 --N 1 --ReduceOp 0 --R 0 1 2"},
        {cmd + " --scales 1 0 --CompType 1 --D 160 10 1 --I 0 --N 1 --ReduceOp 0 --R 0 1 2"},
        {cmd + " --scales 1 0 --CompType 1 --D 7 1024 1 --I 0 --N 1 --ReduceOp 0 --R 0 1 2"},
        {cmd + " --scales 1 0 --CompType 1 --D 3 1 1 --I 0 --N 1 --ReduceOp 0 --R 0 1 2"},
        {cmd + " --scales 1 0 --CompType 1 --D 3 1 1 --I 0 --N 1 --ReduceOp 1 --R 0 1 2"},
        {cmd + " --scales 1 0 --CompType 1 --D 3 1 1 --I 1 --N 1 --ReduceOp 3 --R 0 1 2"},
        {cmd + " --scales 1 0 --CompType 1 --D 3 2 1 --I 1 --N 1 --ReduceOp 3 --R 1 2"},
        {cmd + " --scales 1 0 --CompType 1 --D 6 2 1 --I 0 --N 1 --ReduceOp 3 --R 1 2"},
        {cmd + " --scales 1 0 --CompType 1 --D 6 2 1 --I 0 --N 1 --ReduceOp 2 --R 1 2"},
        {cmd + " --scales 1 0 --CompType 1 --D 2 2 1 --I 0 --N 1 --ReduceOp 0 --R 1 2"},
        {cmd + " --scales 1 0 --CompType 1 --D 4 3 1 --I 0 --N 1 --ReduceOp 3 --R 1 2"},
        {cmd + " --scales 1 0 --CompType 1 --D 3 4 1 --I 0 --N 1 --ReduceOp 3 --R 1 2"},
        {cmd + " --scales 1 0 --CompType 1 --D 3 4 1 --I 0 --N 1 --ReduceOp 3 --R 0 2"},
        {cmd + " --scales 1 0 --CompType 1 --D 2048 32 1 --I 0 --N 1 --ReduceOp 3 --R 0 2"},
        {cmd + " --scales 1 0 --CompType 1 --D 4 3 1 --I 0 --N 1 --ReduceOp 2 --R 1 2"},
        {cmd + " --scales 1 0 --CompType 1 --D 3 4 1 --I 0 --N 1 --ReduceOp 2 --R 0 2"},
        {cmd + " --scales 1 0 --CompType 1 --D 2048 32 1 --I 0 --N 1 --ReduceOp 2 --R 0 2"},
        {cmd + " --scales 1 0 --CompType 1 --D 3 4 1 --I 0 --N 1 --ReduceOp 2 --R 0 2"},
        {cmd + " --scales 1 0 --CompType 1 --D 12 11 1 --I 0 --N 1 --ReduceOp 0 --R 0 1 2"},
        {cmd + " --scales 1 0 --CompType 1 --D 13 4 7 7 --I 0 --N 1 --ReduceOp 0 --R 0 1 2 3"},
        {cmd + " --scales 1 0 --CompType 1 --D 64 3 280 81 --I 0 --N 0 --ReduceOp 0 --R 0"}
    };
    // clang-format on
}

class GPU_reduce_custom_fp32_fp16_FP32 : public testing::TestWithParam<std::vector<std::string>>
{
};

class GPU_reduce_custom_fp32_fp16_FP16 : public testing::TestWithParam<std::vector<std::string>>
{
};

bool IsTestSupportedForDevice(void)
{
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx103X, Gpu::gfx110X>;
    using d_mask = disabled<Gpu::Default>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

void Run2dDriver(miopenDataType_t prec)
{
    std::vector<std::string> params;
    switch(prec)
    {
    case miopenFloat: params = GPU_reduce_custom_fp32_fp16_FP32::GetParam(); break;
    case miopenHalf: params = GPU_reduce_custom_fp32_fp16_FP16::GetParam(); break;
    case miopenInt8:
    case miopenBFloat16:
    case miopenInt32:
    case miopenInt64:
    case miopenDouble:
    case miopenFloat8:
    case miopenBFloat8:
    default:
        FAIL() << "miopenInt8, miopenBFloat16, miopenInt32, "
                  "miopenDouble, miopenFloat8, miopenBFloat8 "
                  "data type not supported by reduce_custom_fp32_fp16 test";
    }

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

} // namespace reduce_custom_fp32_fp16
using namespace reduce_custom_fp32_fp16;

TEST_P(GPU_reduce_custom_fp32_fp16_FP32, FloatTest_reduce_custom_fp32_fp16)
{
    if(!IsTestSupportedForDevice())
    {
        GTEST_SKIP();
    }

    Run2dDriver(miopenFloat);
};

TEST_P(GPU_reduce_custom_fp32_fp16_FP16, HalfTest_reduce_custom_fp32_fp16)
{
    if(!IsTestSupportedForDevice())
    {
        GTEST_SKIP();
    }

    Run2dDriver(miopenHalf);
};

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_reduce_custom_fp32_fp16_FP32,
                         testing::Values(GetTestCases("--float")));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_reduce_custom_fp32_fp16_FP16,
                         testing::Values(GetTestCases("--half")));
