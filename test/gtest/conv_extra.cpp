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
#include "../conv2d.hpp"
#include <miopen/miopen.h>
#include <gtest/gtest_common.hpp>
#include <gtest/gtest.h>
#include <miopen/env.hpp>
#include "get_handle.hpp"

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLAGS_ARGS)

namespace env = miopen::env;

namespace conv_extra {
void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

std::vector<std::string> GetTestCases(void)
{
    const std::string cmd               = "test_conv2d";
    const std::string cmd_v             = cmd + " --verbose";
    const std::string default_float_arg = " --float";
    const std::string cmd_v_float       = cmd_v + default_float_arg;

    std::string float_arg = env::value(MIOPEN_TEST_FLOAT_ARG);
    if(float_arg.empty())
        float_arg = default_float_arg;
    else
        float_arg = " " + float_arg;

    std::string flag_arg = env::value(MIOPEN_TEST_FLAGS_ARGS);

    // clang-format off
    return std::vector<std::string>{
        // {cmd_v_float + " --input	1	1	1	1	--weights	1	1	2	2	--pads_strides_dilations	0	0	3	3	1	1"},
        {cmd_v_float + " --input	4	1	161	700	--weights	4	1	5	20	--pads_strides_dilations	0	0	2	2	1	1"},
        {cmd_v_float + " --input	4	1	161	700	--weights	4	1	5	20	--pads_strides_dilations	0	0	2	2	1	1"},
        {cmd_v_float + " --input	4	32	79	341	--weights	4	32	5	10	--pads_strides_dilations	0	0	2	2	1	1"},
        {cmd_v_float + " --input	4	32	79	341	--weights	4	32	5	10	--pads_strides_dilations	0	0	2	2	1	1"},
        {cmd_v_float + " --input	4	3	227	227	--weights	4	3	11	11	--pads_strides_dilations	0	0	4	4	1	1"},
        {cmd_v_float + " --input	4	3	224	224	--weights	4	3	11	11	--pads_strides_dilations	2	2	4	4	1	1"},
        {cmd_v_float + " --input	16	1	48	480	--weights	16	1	3	3	--pads_strides_dilations	1	1	1	1	1	1"},
        // Forward disabled since FFT fails verification for the forward direction
        {cmd_v_float + " --input	32	64	27	27	--weights	192	64	5	5	--pads_strides_dilations	2	2	1	1	1	1 --disable-forward"},
        // {cmd_v_float + " --input	4	64	14	14	--weights	24	64	5	5	--pads_strides_dilations	2	2	1	1	1	1"},
        {cmd_v_float + " --input	4	96	14	14	--weights	32	96	5	5	--pads_strides_dilations	2	2	1	1	1	1"},
        {cmd_v_float + " --input	4	16	14	14	--weights	4	16	5	5	--pads_strides_dilations	2	2	1	1	1	1"},
        {cmd_v_float + " --input	4	32	14	14	--weights	4	32	5	5	--pads_strides_dilations	2	2	1	1	1	1"},

        {cmd + float_arg + " --input 16 3 64 128 --weights 96 3 11 11 --pads_strides_dilations 0 0 1 1 1 1 " + flag_arg},
        {cmd + float_arg + " --input 16 3 32 32 --weights 96 3 11 11 --pads_strides_dilations 0 0 2 2 1 1  " + flag_arg},
        {cmd + float_arg + " --input 16 3 64 128 --weights 96 3 11 11 --pads_strides_dilations 5 5 2 2 1 1 " + flag_arg},
        {cmd + float_arg + " --input 16 3 32 32 --weights 96 3 11 11 --pads_strides_dilations 5 5 2 2 1 1  " + flag_arg},

        {cmd + float_arg + " --input 2 16 1024 2048 --weights 32 16 3 3 --pads_strides_dilations 0 0 1 1 1 1 " + flag_arg},
        {cmd + float_arg + " --input 2 16 1024 2048 --weights 32 16 3 3 --pads_strides_dilations 1 1 1 1 1 1 " + flag_arg},
        {cmd + float_arg + " --input 2 16 3072 3072 --weights 32 16 3 3 --pads_strides_dilations 0 0 2 2 1 1 " + flag_arg},
        {cmd + float_arg + " --input 2 16 3072 3072 --weights 32 16 3 3 --pads_strides_dilations 2 2 2 2 1 1 " + flag_arg},

        {cmd + float_arg + " --input 128 320 1 7 --weights 256 320 1 1 --pads_strides_dilations 0 0 1 1 1 1 " + flag_arg},
        {cmd + float_arg + " --input 128 1024 1 7 --weights 2048 1024 1 1 --pads_strides_dilations 1 1 1 1 1 1 " + flag_arg},
        {cmd + float_arg + " --input 352 192 7 1 --weights 320 192 1 1 --pads_strides_dilations 0 0 1 1 1 1 " + flag_arg},
        {cmd + float_arg + " --input 352 16 7 1 --weights 32 16 1 1 --pads_strides_dilations 2 2 1 1 1 1 " + flag_arg}
    };
    // clang-format on
}

using TestCase = decltype(GetTestCases())::value_type;

class ConfigWithFloat_conv_extra : public testing::TestWithParam<std::vector<TestCase>>
{
};

bool IsTestSupportedForDevice()
{
    using namespace miopen::debug;
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx103X, Gpu::gfx110X>;
    using d_mask = disabled<Gpu::Default>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

void Run2dDriver(void)
{
    if(!IsTestSupportedForDevice() && !env::enabled(MIOPEN_TEST_ALL))
    {
        GTEST_SKIP();
    }
    std::vector<std::string> params = ConfigWithFloat_conv_extra::GetParam();

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

} // namespace conv_extra
using namespace conv_extra;

TEST_P(ConfigWithFloat_conv_extra, FloatTest_conv_extra) { Run2dDriver(); };

INSTANTIATE_TEST_SUITE_P(ConvExtra, ConfigWithFloat_conv_extra, testing::Values(GetTestCases()));
