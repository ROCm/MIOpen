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
#include <miopen/env.hpp>
#include "get_handle.hpp"

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)

namespace reduce_custom_fp32 {
std::vector<std::string> GetArgs(const std::string& param)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    return {begin, end};
}

std::vector<std::string> GetTestCases(void)
{
    const std::string& cmd = "test_reduce_test ";
    std::string float_arg  = env::value(MIOPEN_TEST_FLOAT_ARG);
    if(float_arg.empty())
        float_arg = "--float";

    // clang-format off
    return std::vector<std::string>{
        {cmd + float_arg + " --scales 1 0 --CompType 1 --D 1024 30528 1 --I 0 --N 1 --ReduceOp 0 --R 0 1 2"}
    };
    // clang-format on
}

using TestCase = decltype(GetTestCases())::value_type;

class GPU_reduce_custom_fp32_FP32 : public testing::TestWithParam<std::vector<TestCase>>
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
    if(!(IsTestSupportedForDevice() &&
         (!MIOPEN_TEST_ALL                  // standalone run
          || (env::enabled(MIOPEN_TEST_ALL) // or --float full tests enabled
              && env::value(MIOPEN_TEST_FLOAT_ARG) == "--float"))))
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

INSTANTIATE_TEST_SUITE_P(Full, GPU_reduce_custom_fp32_FP32, testing::Values(GetTestCases()));
