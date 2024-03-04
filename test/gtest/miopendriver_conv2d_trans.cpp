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
#include <miopen/miopen.h>
#include <gtest/gtest_common.hpp>
#include <gtest/gtest.h>
#include <miopen/env.hpp>
#include "get_handle.hpp"
#include "../conv2d.hpp"

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_WITH_MIOPENDRIVER)

namespace miopendriver_conv2d_trans {
void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

auto GetTestCases(const std::string& precision)
{
    std::string cmd = "MIOpenDriver ";

    if(precision == "--float")
        cmd.append("conv");
    else if(precision == "--half")
        cmd.append("convfp16");
    else if(precision == "--bfloat16")
        cmd.append("convbfp16");

    // clang-format off
    return std::vector<std::string>{
        {cmd + " -m trans -x 1 -y 1 -W 112 -H 112 -c 64 -n 8 -k 32 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F 0 -V 1"},
        {cmd + " -m trans -x 1 -y 7 -W 17 -H 17 -c 32 -n 128 -k 16 -p 3 -q 0 -u 1 -v 1 -l 1 -j 1 -g 2 -F 0 -V 1"},
        {cmd + " -m trans -x 10 -y 5 -W 341 -H 79 -c 32 -n 4 -k 8 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 4 -F 0 -V 1"},
        {cmd + " -m trans -x 20 -y 5 -W 700 -H 161 -c 1 -n 4 -k 32 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -F 0 -V 1"},
        {cmd + " -m trans -x 3 -y 3 -W 108 -H 108 -c 3 -n 8 -k 64 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -F 0 -V 1"},
        {cmd + " -m trans -x 5 -y 5 -W 175 -H 40 -c 128 -n 16 -k 256 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -F 0 -V 1"},
        {cmd + " -m trans -x 5 -y 5 -W 700 -H 161 -c 1 -n 16 -k 64 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -F 0 -V 1"},
        {cmd + " -m trans -x 7 -y 7 -W 224 -H 224 -c 3 -n 16 -k 64 -p 3 -q 3 -u 2 -v 2 -l 1 -j 1 -g 1 -F 0 -V 1"}
    };
    // clang-format on
}

using TestCase = decltype(GetTestCases())::value_type;

class ConfigWithFloat_miopendriver_conv2d_trans
    : public testing::TestWithParam<std::vector<TestCase>>
{
};

class ConfigWithHalf_miopendriver_conv2d_trans
    : public testing::TestWithParam<std::vector<TestCase>>
{
};

class ConfigWithBf16_miopendriver_conv2d_trans
    : public testing::TestWithParam<std::vector<TestCase>>
{
};

bool IsTestSupportedForDevice()
{
    using namespace miopen::debug;
    using e_mask = enabled<Gpu::gfx103X, Gpu::gfx110X>;
    using d_mask = disabled<Gpu::gfx900>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

static bool SkipTest(const std::string& float_arg)
{
    if(IsTestSupportedForDevice() && miopen::IsEnabled(ENV(MIOPEN_TEST_WITH_MIOPENDRIVER)) &&
       (miopen::IsUnset(ENV(MIOPEN_TEST_ALL))       // standalone run
        || (miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) // or full tests enabled
            && miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG)) == float_arg)))
        return false;
    return true;
}

void Run2dDriver(miopenDataType_t prec)
{
    std::vector<std::string> params;
    switch(prec)
    {
    case miopenFloat: params = ConfigWithFloat_miopendriver_conv2d_trans::GetParam(); break;
    case miopenHalf: params = ConfigWithHalf_miopendriver_conv2d_trans::GetParam(); break;
    case miopenBFloat16: params = ConfigWithBf16_miopendriver_conv2d_trans::GetParam(); break;
    case miopenInt8:
    case miopenInt32:
    case miopenFloat8:
    case miopenBFloat8:
    case miopenDouble:
    default:
        FAIL() << "miopenInt8, miopenInt32, miopenFloat8, miopenBFloat8, miopenDouble data type "
                  "not supported by miopendriver_conv2d_trans test";
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
        test_drive<conv2d_driver>(ptrs.size(), ptrs.data());
        auto capture = testing::internal::GetCapturedStderr();
        std::cout << capture;
    }
};

} // namespace miopendriver_conv2d_trans
using namespace miopendriver_conv2d_trans;

TEST_P(ConfigWithFloat_miopendriver_conv2d_trans, FloatTest_miopendriver_conv2d_trans)
{
    if(SkipTest("--float"))
    {
        GTEST_SKIP();
    }
    else
    {
        Run2dDriver(miopenFloat);
    }
};

TEST_P(ConfigWithHalf_miopendriver_conv2d_trans, HalfTest_miopendriver_conv2d_trans)
{
    if(SkipTest("--half"))
    {
        GTEST_SKIP();
    }
    else
    {
        Run2dDriver(miopenHalf);
    }
};

TEST_P(ConfigWithBf16_miopendriver_conv2d_trans, Bf16Test_miopendriver_conv2d_trans)
{
    if(SkipTest("--bfloat16"))
    {
        GTEST_SKIP();
    }
    else
    {
        Run2dDriver(miopenBFloat16);
    }
};

INSTANTIATE_TEST_SUITE_P(MiopendriverConv2dTran,
                         ConfigWithFloat_miopendriver_conv2d_trans,
                         testing::Values(GetTestCases("--float")));
INSTANTIATE_TEST_SUITE_P(MiopendriverConv2dTran,
                         ConfigWithHalf_miopendriver_conv2d_trans,
                         testing::Values(GetTestCases("--half")));
INSTANTIATE_TEST_SUITE_P(MiopendriverConv2dTran,
                         ConfigWithBf16_miopendriver_conv2d_trans,
                         testing::Values(GetTestCases("--bfloat16")));
