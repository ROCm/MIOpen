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
#include <miopen/env.hpp>
#include "../rnn_vanilla.hpp"
#include "get_handle.hpp"

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_DEEPBENCH)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace env = miopen::env;

namespace deepbench_rnn {
static bool SkipTest(void) { return !env::enabled(MIOPEN_TEST_DEEPBENCH); }

void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

class DeepBenchRNNConfigWithFloat : public testing::TestWithParam<std::vector<std::string>>
{
};

void Run2dDriverFloat(void)
{

    std::vector<std::string> params = DeepBenchRNNConfigWithFloat::GetParam();

    for(const auto& test_value : params)
    {
        std::vector<std::string> tokens;
        GetArgs(test_value, tokens);
        std::vector<const char*> ptrs;

        std::transform(tokens.begin(), tokens.end(), std::back_inserter(ptrs), [](const auto& str) {
            return str.data();
        });

        testing::internal::CaptureStderr();
        test_drive<rnn_vanilla_driver>(ptrs.size(), ptrs.data(), "deepbench_rnn");
        auto capture = testing::internal::GetCapturedStderr();
        std::cout << capture;
    }
};

std::vector<std::string> GetTestCases(const std::string& precision)
{
    std::string flags = "--verbose " + precision;

    std::string postFlags =
        "--num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill";

    const std::vector<std::string> test_cases = {
        // clang-format off
    {flags + " --batch-size 16 --seq-len 50 --vector-len 1760 --hidden-size 1760 " + postFlags},
    {flags + " --batch-size 32 --seq-len 50 --vector-len 1760 --hidden-size 1760 " + postFlags},
    {flags + " --batch-size 64 --seq-len 50 --vector-len 1760 --hidden-size 1760 " + postFlags},
    {flags + " --batch-size 128 --seq-len 50 --vector-len 1760 --hidden-size 1760 " + postFlags},
    {flags + " --batch-size 16 --seq-len 50 --vector-len 2048 --hidden-size 2048 " + postFlags},
    {flags + " --batch-size 32 --seq-len 50 --vector-len 2048 --hidden-size 2048 " + postFlags},
    {flags + " --batch-size 64 --seq-len 50 --vector-len 2048 --hidden-size 2048 " + postFlags},
    {flags + " --batch-size 128 --seq-len 50 --vector-len 2048 --hidden-size 2048 " + postFlags},
    {flags + " --batch-size 16 --seq-len 50 --vector-len 2560 --hidden-size 2560 " + postFlags},
    {flags + " --batch-size 32 --seq-len 50 --vector-len 2560 --hidden-size 2560 " + postFlags},
    {flags + " --batch-size 64 --seq-len 50 --vector-len 2560 --hidden-size 2560 " + postFlags},
    {flags + " --batch-size 128 --seq-len 50 --vector-len 2560 --hidden-size 2560 " + postFlags}
        // clang-format on
    };

    return test_cases;
}

} // namespace deepbench_rnn

using namespace deepbench_rnn;

TEST_P(DeepBenchRNNConfigWithFloat, FloatTest_deepbench_rnn)
{
    if(SkipTest())
    {
        GTEST_SKIP();
    }
    else
    {
        Run2dDriverFloat();
    }
};

INSTANTIATE_TEST_SUITE_P(ConvTrans,
                         DeepBenchRNNConfigWithFloat,
                         testing::Values(GetTestCases("--float")));
