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

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_FLOAT)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_HALF)

static bool SkipTest(void) { return !miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)); }

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

class ConfigWithHalf : public testing::TestWithParam<std::vector<std::string>>
{
};

void Run2dDriver(miopenDataType_t prec)
{

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

bool IsTestSupportedForDevice(const miopen::Handle& handle) { return true; }

TEST_P(ConfigWithFloat, FloatTest)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && !SkipTest() && miopen::IsEnabled(ENV(MIOPEN_TEST_FLOAT)))
    {
        Run2dDriver(miopenFloat);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(ConfigWithHalf, HalfTest)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && !SkipTest() && miopen::IsEnabled(ENV(MIOPEN_TEST_HALF)))
    {
        Run2dDriver(miopenHalf);
    }
    else
    {
        GTEST_SKIP();
    }
};

std::vector<std::string> GetTestCases(std::string precision)
{
    std::string flags       = "test_lstm --verbose " + precision;
    std::string commonFlags = " --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 "
                              "--hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0";

    const std::vector<std::string> test_cases = {
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
        {flags + commonFlags +
         " -dir-mode 0 --no-hx --no-dhy --no-cx --no-dcy --no-hy --no-dhx --no-cy --no-dcx"},
        {flags + commonFlags +
         " -dir-mode 1 --no-hx --no-dhy --no-cx --no-dcy --no-hy --no-dhx --no-cy --no-dcx"}};

    return test_cases;
}

INSTANTIATE_TEST_SUITE_P(LstmExtra, ConfigWithFloat, testing::Values(GetTestCases("--float")));

INSTANTIATE_TEST_SUITE_P(LstmExtra, ConfigWithHalf, testing::Values(GetTestCases("--half")));
