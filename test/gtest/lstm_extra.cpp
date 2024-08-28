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
#include <gtest/gtest_common.hpp>
#include <gtest/gtest.h>
#include <boost/algorithm/string.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)

namespace env = miopen::env;

namespace lstm_extra {
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
    std::string flags       = "test_lstm --verbose " + precision;
    std::string commonFlags = " --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 "
                              "--hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0";

    // clang-format off
    return std::vector<std::string>{
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
    };
    // clang-format on
}

using TestCase = decltype(GetTestCases({}))::value_type;

class GPU_lstm_extra_FP32 : public testing::TestWithParam<std::vector<TestCase>>
{
};

bool IsTestSupportedForDevice()
{
    using namespace miopen::debug;
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx103X, Gpu::gfx110X>;
    using d_mask = disabled<Gpu::Default>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

void Run2dDriver(miopenDataType_t prec)
{
    if(!(IsTestSupportedForDevice()            //
         && (!MIOPEN_TEST_ALL                  // standalone run
             || (env::enabled(MIOPEN_TEST_ALL) // or --float full tests enabled
                 && env::value(MIOPEN_TEST_FLOAT_ARG) == "--float"))))
    {
        GTEST_SKIP();
    }
    std::vector<std::string> params = GPU_lstm_extra_FP32::GetParam();

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

} // namespace lstm_extra
using namespace lstm_extra;

TEST_P(GPU_lstm_extra_FP32, FloatTest_lstm_extra) { Run2dDriver(miopenFloat); };

INSTANTIATE_TEST_SUITE_P(Full, GPU_lstm_extra_FP32, testing::Values(GetTestCases("--float")));
