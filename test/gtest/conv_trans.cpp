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
#include <miopen/miopen.h>
#include <miopen/env.hpp>
#include "../conv2d.hpp"
#include "get_handle.hpp"

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace env = miopen::env;

namespace conv_trans {

void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

class ConfigWithFloat_conv_trans : public testing::TestWithParam<std::vector<std::string>>
{
};

void Run2dDriver(miopenDataType_t prec)
{

    std::vector<std::string> params;
    switch(prec)
    {
    case miopenFloat: params = ConfigWithFloat_conv_trans::GetParam(); break;
    case miopenHalf:
    case miopenFloat8:
    case miopenBFloat8:
    case miopenInt8:
    case miopenBFloat16:
    case miopenInt32:
    case miopenInt64:
    case miopenDouble:
        FAIL() << "miopenHalf, miopenInt8, miopenBFloat16, miopenInt32, miopenDouble "
                  "data type not supported by "
                  "conv_trans test";

    default: params = ConfigWithFloat_conv_trans::GetParam();
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
        test_drive<conv2d_driver>(ptrs.size(), ptrs.data(), "test_conv2d");
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

std::vector<std::string> GetTestCases(const std::string& precision)
{
    std::string flags = " --verbose " + precision + " ";

    std::string psd0 = " --pads_strides_dilations 0 0 1 1 1 1";
    std::string psd1 = " --pads_strides_dilations 0 0 2 2 1 1";
    std::string psd2 = " --pads_strides_dilations 2 2 1 1 1 1";

    std::string cmode_t = " --cmode trans";
    std::string pmode_d = " --pmode	default";
    std::string pmode_s = " --pmode	same";
    std::string pmode_v = " --pmode	valid";

    std::string gc_2  = " --group-count 2";
    std::string gc_3  = " --group-count 3";
    std::string gc_4  = " --group-count 4";
    std::string gc_8  = " --group-count 8";
    std::string gc_32 = " --group-count 32";

    const std::vector<std::string> test_cases = {
        // clang-format off
    {flags + "--input	8	128	28	28	--weights	128	128	1	1" + psd0 + cmode_t + pmode_d},
    {flags + "--input	8	256	28	28	--weights	256	256	1	1" + psd0 + cmode_t + pmode_s},
    {flags + "--input	8	32	28	28	--weights	32	32	5	5" + psd1 + cmode_t + pmode_d},
    {flags + "--input	8	512	14	14	--weights	512	512	1	1" + psd1 + cmode_t + pmode_s},
    {flags + "--input	8	512	4	4	--weights	512	512	1	1" + psd0 + cmode_t + pmode_v},
    {flags + "--input	8	64	56	56	--weights	64	64	1	1" + psd1 + cmode_t + pmode_v},
    {flags + "--input	100	3	64	64	--weights	3	3	1	1" + psd2 + cmode_t + pmode_d},
    {flags + "--input	100	6	4	4	--weights	6	4	1	1" + psd2 + cmode_t + pmode_d},
    {flags + "--input	8	128	28	28	--weights	128	16	1	1" + psd0 + cmode_t + pmode_d + gc_8},
    {flags + "--input	8	256	28	28	--weights	256	64	1	1" + psd0 + cmode_t + pmode_s + gc_4},
    {flags + "--input	8	32	28	28	--weights	32	1	5	5" + psd1 + cmode_t + pmode_d + gc_32},
    {flags + "--input	8	512	14	14	--weights	512	16	1	1" + psd1 + cmode_t + pmode_s + gc_32},
    {flags + "--input	8	512	4	4	--weights	512	16	1	1" + psd0 + cmode_t + pmode_v + gc_32},
    {flags + "--input	8	64	56	56	--weights	64	2	1	1" + psd1 + cmode_t + pmode_v + gc_32},
    {flags + "--input	100	3	64	64	--weights	3	3	1	1" + psd2 + cmode_t + pmode_d + gc_3},
    {flags + "--input	100	6	4	4	--weights	6	4	1	1" + psd2 + cmode_t + pmode_d + gc_2}
        // clang-format on
    };

    return test_cases;
}

} // namespace conv_trans
using namespace conv_trans;

TEST_P(ConfigWithFloat_conv_trans, FloatTest_conv_trans)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && env::enabled(MIOPEN_TEST_ALL))
    {
        Run2dDriver(miopenFloat);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(ConvTrans,
                         ConfigWithFloat_conv_trans,
                         testing::Values(GetTestCases("--float")));
