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

#include "lstm.hpp"
#include "get_handle.hpp"
#include <miopen/env.hpp>
#include <gtest/gtest.h>
#include <boost/algorithm/string.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_DEEPBENCH)

static bool SkipTest(void) { return miopen::IsDisabled(ENV(MIOPEN_TEST_DEEPBENCH)); }

void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

class ConfigWithFloat : public testing::TestWithParam<std::vector<std::string>>
{
};

void Run2dDriver(miopenDataType_t prec)
{

    std::vector<std::string> params;
    switch(prec)
    {
    case miopenFloat: params = ConfigWithFloat::GetParam(); break;
    case miopenHalf:
    case miopenFloat8:
    case miopenBFloat8:
    case miopenInt8:
    case miopenBFloat16:
    case miopenInt32:
    case miopenDouble:
        FAIL() << "miopenHalf, miopenInt8, miopenBFloat16, miopenInt32, miopenDouble "
                  "data type not supported by "
                  "lstm_extra test";

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

bool IsTestSupportedForDevice(const miopen::Handle& handle)
{
    std::string devName = handle.GetDeviceName();
    if(devName == "gfx900" || devName == "gfx906" || devName == "gfx908" || devName == "gfx90a" ||
       miopen::StartsWith(devName, "gfx94") || miopen::StartsWith(devName, "gfx103") ||
       miopen::StartsWith(devName, "gfx110"))
        return true;
    else
        return false;
}

TEST_P(ConfigWithFloat, FloatTest)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && !SkipTest())
    {
        Run2dDriver(miopenFloat);
    }
    else
    {
        GTEST_SKIP();
    }
};

std::vector<std::string> GetTestCases(void)
{
    std::string flags = "test_lstm --verbose ";
    std::string commonFlags =
        "--batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 --hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0";

    const std::vector<std::string> test_cases = {
    // clang-format off
    {flags + commonFlags + " -dir-mode 0 --no-hx"},
    {flags + commonFlags + " -dir-mode 0 --no-dhy"},
    {flags + commonFlags + " -dir-mode 0 --no-hx --no-dhy"},
    {flags + commonFlags + " -dir-mode 0 --no-cx"},
    {flags + commonFlags + " -dir-mode 0 --no-hx --no-cx"},
    {flags + commonFlags + " -dir-mode 0 --no-dcy"},
    {flags + commonFlags + " -dir-mode 0 --no-cx --no-dcy"},
    {flags + commonFlags + " -dir-mode 1 --no-hx"},
    {flags + commonFlags + " -dir-mode 1 --no-dhy"},
    {flags + commonFlags + " -dir-mode 1 --no-hx --no-dhy"},
    {flags + commonFlags + " -dir-mode 1 --no-cx"},
    {flags + commonFlags + " -dir-mode 1 --no-hx --no-cx"},
    {flags + commonFlags + " -dir-mode 1 --no-dcy"},
    {flags + commonFlags + " -dir-mode 1 --no-cx --no-dcy"},
    {flags + commonFlags + " -dir-mode 0 --no-hy"},
    {flags + commonFlags + " -dir-mode 0 --no-dhx"},
    {flags + commonFlags + " -dir-mode 0 --no-hy --no-dhx"},
    {flags + commonFlags + " -dir-mode 0 --no-cy"},
    {flags + commonFlags + " -dir-mode 0 --no-hy --no-cy"},
    {flags + commonFlags + " -dir-mode 0 --no-dcx"},
    {flags + commonFlags + " -dir-mode 0 --no-cy --no-dcx"},
    {flags + commonFlags + " -dir-mode 1 --no-hy"},
    {flags + commonFlags + " -dir-mode 1 --no-dhx"},
    {flags + commonFlags + " -dir-mode 1 --no-hy --no-dhx"},
    {flags + commonFlags + " -dir-mode 1 --no-cy"},
    {flags + commonFlags + " -dir-mode 1 --no-hy --no-cy"},
    {flags + commonFlags + " -dir-mode 1 --no-dcx"},
    {flags + commonFlags + " -dir-mode 1 --no-cy --no-dcx"},
    {flags + commonFlags + " -dir-mode 0 --no-hx --no-dhy --no-cx --no-dcy --no-hy --no-dhx --no-cy --no-dcx"},
    {flags + commonFlags + " -dir-mode 1 --no-hx --no-dhy --no-cx --no-dcy --no-hy --no-dhx --no-cy --no-dcx"}
    // clang-format on
    };

    return test_cases;
}

INSTANTIATE_TEST_SUITE_P(ConvTrans, ConfigWithFloat, testing::Values(GetTestCases()));


