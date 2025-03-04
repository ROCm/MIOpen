/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <miopen/miopen.h>
#include <gtest/gtest.h>
#include "../gru.hpp"
#include "get_handle.hpp"

namespace env = miopen::env;

namespace deepbench_gru {

void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

class GPU_DeepBenchGRU_FP32 : public testing::TestWithParam<std::vector<std::string>>
{
};

void Run2dDriverFloat(void)
{
    std::vector<std::string> params = GPU_DeepBenchGRU_FP32::GetParam();

    for(const auto& test_value : params)
    {
        std::vector<std::string> tokens;
        GetArgs(test_value, tokens);
        std::vector<const char*> ptrs;

        std::transform(tokens.begin(), tokens.end(), std::back_inserter(ptrs), [](const auto& str) {
            return str.data();
        });

        testing::internal::CaptureStderr();
        test_drive<gru_driver>(ptrs.size(), ptrs.data(), "deepbench_gru");
        auto capture = testing::internal::GetCapturedStderr();
        std::cout << capture;
    }
};

std::vector<std::string> GetTestCases(const std::string& precision)
{
    std::string flags = " --verbose " + precision;
    std::string commonFlags =
        " --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill";

    const std::vector<std::string> test_cases = {
        // clang-format off
    {flags + " --batch-size 32 --seq-len 1500 --vector-len 2816 --hidden-size 2816" + commonFlags},
    {flags + " --batch-size 32 --seq-len 750 --vector-len 2816 --hidden-size 2816" + commonFlags},
    {flags + " --batch-size 32 --seq-len 375 --vector-len 2816 --hidden-size 2816" + commonFlags},
    {flags + " --batch-size 32 --seq-len 187 --vector-len 2816 --hidden-size 2816" + commonFlags},
    {flags + " --batch-size 32 --seq-len 1500 --vector-len 2048 --hidden-size 2048" + commonFlags},
    {flags + " --batch-size 32 --seq-len 750 --vector-len 2048 --hidden-size 2048" + commonFlags},
    {flags + " --batch-size 32 --seq-len 375 --vector-len 2048 --hidden-size 2048" + commonFlags},
    {flags + " --batch-size 32 --seq-len 187 --vector-len 2048 --hidden-size 2048" + commonFlags},
    {flags + " --batch-size 32 --seq-len 1500 --vector-len 1536 --hidden-size 1536" + commonFlags},
    {flags + " --batch-size 32 --seq-len 750 --vector-len 1536 --hidden-size 1536" + commonFlags},
    {flags + " --batch-size 32 --seq-len 375 --vector-len 1536 --hidden-size 1536" + commonFlags},
    {flags + " --batch-size 32 --seq-len 187 --vector-len 1536 --hidden-size 1536" + commonFlags},
    {flags + " --batch-size 32 --seq-len 1500 --vector-len 2560 --hidden-size 2560" + commonFlags},
    {flags + " --batch-size 32 --seq-len 750 --vector-len 2560 --hidden-size 2560" + commonFlags},
    {flags + " --batch-size 32 --seq-len 375 --vector-len 2560 --hidden-size 2560" + commonFlags},
    {flags + " --batch-size 32 --seq-len 187 --vector-len 2560 --hidden-size 2560" + commonFlags},
    {flags + " --batch-size 32 --seq-len 1 --vector-len 512 --hidden-size 512" + commonFlags},
    {flags + " --batch-size 32 --seq-len 1500 --vector-len 1024 --hidden-size 1024" + commonFlags},
    {flags + " --batch-size 64 --seq-len 1500 --vector-len 1024 --hidden-size 1024" + commonFlags}
        // clang-format on
    };

    return test_cases;
}

} // namespace deepbench_gru

using namespace deepbench_gru;

TEST_P(GPU_DeepBenchGRU_FP32, FloatTest_deepbench_gru) { Run2dDriverFloat(); };

INSTANTIATE_TEST_SUITE_P(Full, GPU_DeepBenchGRU_FP32, testing::Values(GetTestCases("--float")));
