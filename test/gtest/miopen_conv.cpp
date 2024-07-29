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
#include "../conv2d.hpp"
#include "get_handle.hpp"

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_CONV)

namespace miopen_conv {

bool SkipTest() { return env::disabled(MIOPEN_TEST_CONV); }

void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

class Conv2dFloat_miopen_conv : public testing::TestWithParam<std::vector<std::string>>
{
};

void Run2dDriver(miopenDataType_t prec)
{
    std::vector<std::string> params;
    switch(prec)
    {
    case miopenFloat: params = Conv2dFloat_miopen_conv::GetParam(); break;
    case miopenInt8:
    case miopenBFloat8:
    case miopenFloat8:
    case miopenHalf:
    case miopenBFloat16:
    case miopenInt32:
    case miopenInt64:
    case miopenDouble:
        FAIL() << "miopenInt8, miopenBFloat8, miopenFloat8, miopenHalf, miopenBFloat16, "
                  "miopenInt32, "
                  "miopenDouble data "
                  "type not supported by "
                  "miopen_conv test";

    default: params = Conv2dFloat_miopen_conv::GetParam();
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
        test_drive<conv2d_driver>(ptrs.size(), ptrs.data(), "miopen_conv");
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
    std::string v = " --verbose " + precision;

    std::vector<std::string> test_cases = {
        // clang-format off
    {v + "	--input	1	3	32	32	--weights	1	3	7	7	--pads_strides_dilations	1	1	1	1	1	1"},
    {v + "	--input	1	3	227	227	--weights	1	3	7	7	--pads_strides_dilations	1	1	1	1	1	1"},
    {v + "	--input	1	64	56	56	--weights	1	64	1	1	--pads_strides_dilations	0	0	2	2	1	1"},
    {v + "	--input	1	3	32	32	--weights	1	3	3	3	--pads_strides_dilations	2	2	1	1	1	1"},
    {v + "	--input	1	3	224	224	--weights	1	3	3	3	--pads_strides_dilations	2	2	1	1	1	1"},
    {v + "	--input	1	3	227	227	--weights	1	3	3	3	--pads_strides_dilations	2	2	1	1	1	1"},
    {v + "	--input	1	3	231	231	--weights	1	3	3	3	--pads_strides_dilations	2	2	1	1	1	1"},
    {v + "	--input	1	3	224	224	--weights	1	3	5	5	--pads_strides_dilations	2	2	1	1	1	1"},
    {v + "	--input	1	3	227	227	--weights	1	3	5	5	--pads_strides_dilations	2	2	1	1	1	1"},
    {v + "	--input	1	3	231	231	--weights	1	3	5	5	--pads_strides_dilations	2	2	1	1	1	1"},
    {v + "	--input	1	3	32	32	--weights	1	3	7	7	--pads_strides_dilations	2	2	1	1	1	1"},
    {v + "	--input	1	3	224	224	--weights	1	3	7	7	--pads_strides_dilations	2	2	1	1	1	1"},
    {v + "	--input	1	3	227	227	--weights	1	3	7	7	--pads_strides_dilations	2	2	1	1	1	1"},
    {v + "	--input	1	3	231	231	--weights	1	3	7	7	--pads_strides_dilations	2	2	1	1	1	1"},
    {v + "	--input	1	64	56	56	--weights	1	64	3	3	--pads_strides_dilations	2	2	1	1	1	1"},
    {v + "	--input	1	64	112	112	--weights	1	64	3	3	--pads_strides_dilations	2	2	1	1	1	1"},
    {v + "	--input	1	64	512	1024	--weights	1	64	3	3	--pads_strides_dilations	2	2	1	1	1	1"},
    {v + "	--input	1	96	27	27	--weights	1	96	3	3	--pads_strides_dilations	2	2	1	1	1	1"},
    {v + "	--input	1	96	28	28	--weights	1	96	3	3	--pads_strides_dilations	2	2	1	1	1	1"},
    {v + "	--input	1	3	32	32	--weights	1	3	3	3	--pads_strides_dilations	0	0	4	4	1	1"},
    {v + "	--input	1	3	224	224	--weights	1	3	3	3	--pads_strides_dilations	0	0	4	4	1	1"},
    {v + "	--input	1	3	227	227	--weights	1	3	3	3	--pads_strides_dilations	0	0	4	4	1	1"},
    {v + "	--input	1	3	231	231	--weights	1	3	3	3	--pads_strides_dilations	0	0	4	4	1	1"},
    {v + "	--input	1	3	32	32	--weights	1	3	5	5	--pads_strides_dilations	0	0	4	4	1	1"},
    {v + "	--input	1	3	224	224	--weights	1	3	5	5	--pads_strides_dilations	0	0	4	4	1	1"},
    {v + "	--input	1	3	227	227	--weights	1	3	5	5	--pads_strides_dilations	0	0	4	4	1	1"},
    {v + "	--input	1	3	231	231	--weights	1	3	5	5	--pads_strides_dilations	0	0	4	4	1	1"},
    {v + "	--input	1	3	32	32	--weights	1	3	7	7	--pads_strides_dilations	0	0	4	4	1	1"},
    {v + "	--input	1	3	224	224	--weights	1	3	7	7	--pads_strides_dilations	0	0	4	4	1	1"},
    {v + "	--input	1	3	227	227	--weights	1	3	7	7	--pads_strides_dilations	0	0	4	4	1	1"},
    {v + "	--input	1	3	231	231	--weights	1	3	7	7	--pads_strides_dilations	0	0	4	4	1	1"},
    {v + "	--input	1	16	14	14	--weights	1	16	5	5	--pads_strides_dilations	0	0	4	4	1	1"},
    {v + "	--input	1	16	28	28	--weights	1	16	5	5	--pads_strides_dilations	0	0	4	4	1	1"},
    {v + "	--input	1	24	14	14	--weights	1	24	5	5	--pads_strides_dilations	0	0	4	4	1	1"},
    {v + "	--input	1	32	7	7	--weights	1	32	5	5	--pads_strides_dilations	0	0	4	4	1	1"},
    {v + "	--input	1	32	8	8	--weights	1	32	5	5	--pads_strides_dilations	0	0	4	4	1	1"},
    {v + "	--input	1	32	14	14	--weights	1	32	5	5	--pads_strides_dilations	0	0	4	4	1	1"},
    {v + "	--input	1	32	16	16	--weights	1	32	5	5	--pads_strides_dilations	0	0	4	4	1	1"},
    {v + "	--input	1	32	28	28	--weights	1	32	5	5	--pads_strides_dilations	0	0	4	4	1	1"},
    {v + "	--input	1	48	7	7	--weights	1	48	5	5	--pads_strides_dilations	0	0	4	4	1	1"}
        // clang-format on
    };

    return test_cases;
}

} // namespace miopen_conv
using namespace miopen_conv;

TEST_P(Conv2dFloat_miopen_conv, FloatTest)
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

INSTANTIATE_TEST_SUITE_P(MiopenConv,
                         Conv2dFloat_miopen_conv,
                         testing::Values(GetTestCases("--float")));
