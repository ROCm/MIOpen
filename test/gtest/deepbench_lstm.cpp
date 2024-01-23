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
#include <utility>

#include "lstm.hpp"
#include "get_handle.hpp"
#include <miopen/env.hpp>
#include <gtest/gtest_common.hpp>
#include <gtest/gtest.h>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_DEEPBENCH)

namespace deepbench_lstm {

static bool Skip(miopenDataType_t prec)
{
    switch(prec)
    {
    case miopenFloat: return !miopen::IsEnabled(ENV(MIOPEN_TEST_DEEPBENCH));
    case miopenHalf:
    case miopenFloat8:
    case miopenBFloat8:
    case miopenInt8:
    case miopenBFloat16:
    case miopenInt32:
    case miopenDouble:
    default: MIOPEN_THROW("Unsupported datatype");
    }
    return true;
}

void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

auto GetTestCases(std::string precision)
{
    std::string flags = "test_lstm --verbose " + precision;
    std::string commonFlags =
        " --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill";

    // clang-format off
    return std::vector<std::string>{
        {flags + " --batch-size 16 --seq-len 25 --vector-len 512 --hidden-size 512" + commonFlags},
        {flags + " --batch-size 32 --seq-len 25 --vector-len 512 --hidden-size 512" + commonFlags},
        {flags + " --batch-size 64 --seq-len 25 --vector-len 512 --hidden-size 512" + commonFlags},
        {flags + " --batch-size 128 --seq-len 25 --vector-len 512 --hidden-size 512" + commonFlags},
        {flags + " --batch-size 16 --seq-len 25 --vector-len 1024 --hidden-size 1024" + commonFlags},
        {flags + " --batch-size 32 --seq-len 25 --vector-len 1024 --hidden-size 1024" + commonFlags},
        {flags + " --batch-size 64 --seq-len 25 --vector-len 1024 --hidden-size 1024" + commonFlags},
        {flags + " --batch-size 128 --seq-len 25 --vector-len 1024 --hidden-size 1024" + commonFlags},
        {flags + " --batch-size 16 --seq-len 25 --vector-len 2048 --hidden-size 2048" + commonFlags},
        {flags + " --batch-size 32 --seq-len 25 --vector-len 2048 --hidden-size 2048" + commonFlags},
        {flags + " --batch-size 64 --seq-len 25 --vector-len 2048 --hidden-size 2048" + commonFlags},
        {flags + " --batch-size 128 --seq-len 25 --vector-len 2048 --hidden-size 2048" + commonFlags},
        {flags + " --batch-size 16 --seq-len 25 --vector-len 4096 --hidden-size 4096" + commonFlags},
        {flags + " --batch-size 32 --seq-len 25 --vector-len 4096 --hidden-size 4096" + commonFlags},
        {flags + " --batch-size 64 --seq-len 25 --vector-len 4096 --hidden-size 4096" + commonFlags},
        {flags + " --batch-size 128 --seq-len 25 --vector-len 4096 --hidden-size 4096" + commonFlags},
        {flags + " --batch-size 8 --seq-len 50 --vector-len 1536 --hidden-size 1536" + commonFlags},
        {flags + " --batch-size 16 --seq-len 50 --vector-len 1536 --hidden-size 1536" + commonFlags},
        {flags + " --batch-size 32 --seq-len 50 --vector-len 1536 --hidden-size 1536" + commonFlags},
        {flags + " --batch-size 16 --seq-len 150 --vector-len 256 --hidden-size 256" + commonFlags},
        {flags + " --batch-size 32 --seq-len 150 --vector-len 256 --hidden-size 256" + commonFlags},
        {flags + " --batch-size 64 --seq-len 150 --vector-len 256 --hidden-size 256" + commonFlags}
    };
    // clang-format on
}

using TestCase = decltype(GetTestCases({}))::value_type;

class ConfigWithFloat : public testing::TestWithParam<std::vector<TestCase>>
{
};

class ConfigWithHalf : public testing::TestWithParam<std::vector<TestCase>>
{
};

static bool IsTestSupportedForDevice()
{
    using namespace miopen::debug;
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx103X, Gpu::gfx110X, Gpu::gfx908>;
    using d_mask = disabled<Gpu::Default>;
    return miopen::debug::IsTestSupportedForDevice<d_mask, e_mask>();
}

void Run2dDriver(miopenDataType_t prec)
{
    if(!IsTestSupportedForDevice())
    {
        GTEST_SKIP();
    }
    if(Skip(prec))
    {
        GTEST_SKIP();
    }
    std::vector<std::string> params;
    switch(prec)
    {
    case miopenFloat: params = ConfigWithFloat::GetParam(); break;
    case miopenHalf: params = ConfigWithHalf::GetParam(); break;
    case miopenFloat8:
    case miopenBFloat8:
    case miopenInt8:
    case miopenBFloat16:
    case miopenInt32:
    case miopenDouble:
        FAIL() << "miopenInt8, miopenBFloat16, miopenInt32, "
                  "miopenDouble, miopenFloat8, miopenBFloat8 "
                  "data types not supported by "
                  "deepbench_lstm test";

    default: params = ConfigWithFloat::GetParam();
    }

    for(const auto& test_value : params)
    {
        std::vector<std::string> tokens;
        GetArgs(test_value, tokens);
        std::vector<const char*> ptrs;

        std::transform(tokens.begin(), tokens.end(), std::back_inserter(ptrs), [](const auto& str) {
            return str.data();
        });
        testing::internal::CaptureStderr();
        test_drive<lstm_driver>(ptrs.size(), ptrs.data());
        auto capture = testing::internal::GetCapturedStderr();
        std::cout << capture;
    }
};
} // namespace deepbench_lstm

using namespace deepbench_lstm;

TEST_P(ConfigWithFloat, FloatTest_deepbench_lstm) { Run2dDriver(miopenFloat); };

INSTANTIATE_TEST_SUITE_P(DeepbenchLstm, ConfigWithFloat, testing::Values(GetTestCases("--float")));
