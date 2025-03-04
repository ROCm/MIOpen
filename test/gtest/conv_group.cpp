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
#include "get_handle.hpp"

namespace conv_group {
void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

std::vector<std::string> GetTestCases(const std::string& precision)
{
    std::string cmd_v = "test_conv2d --verbose " + precision;

    // clang-format off
    return std::vector<std::string>{
        {cmd_v + " --input	16	128	56	56	--weights	256	4	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32"},
        {cmd_v + " --input	16	256	56	56	--weights	512	8	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	32"},
        {cmd_v + " --input	16	256	28	28	--weights	512	8	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32"},
        {cmd_v + " --input	16	512	28	28	--weights	1024	16	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	32"},
        {cmd_v + " --input	16	512	14	14	--weights	1024	16	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32"},
        {cmd_v + " --input	16	1024	14	14	--weights	2048	32	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	32"},
        {cmd_v + " --input	16	1024	7	7	--weights	2048	32	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32"},
        {cmd_v + " --input	32	128	56	56	--weights	256	4	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32"},
        {cmd_v + " --input	32	256	56	56	--weights	512	8	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	32"},
        //
        // Workaround for "Memory access fault by GPU node" during "HIP Release All" - WrW disabled.
        {cmd_v + " --input	32	256	28	28	--weights	512	8	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32 --disable-backward-weights"},
        {cmd_v + " --input	32	512	28	28	--weights	1024	16	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	32"},
        {cmd_v + " --input	32	512	14	14	--weights	1024	16	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32"},
        {cmd_v + " --input	32	1024	14	14	--weights	2048	32	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	32"},
        {cmd_v + " --input	32	1024	7	7	--weights	2048	32	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	32"},
        {cmd_v + " --input	4	4	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1	--group-count	4"},
        {cmd_v + " --input	8	2	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1	--group-count	2"},
        {cmd_v + " --input	16	4	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1	--group-count	4"},
        {cmd_v + " --input	32	2	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1	--group-count	2"},
        {cmd_v + " --input	4	32	79	341	--weights	32	16	5	10	--pads_strides_dilations	0	0	2	2	1	1	--group-count	2"},
        {cmd_v + " --input	8	32	79	341	--weights	32	16	5	10	--pads_strides_dilations	0	0	2	2	1	1	--group-count	2"},
        {cmd_v + " --input	16	32	79	341	--weights	32	16	5	10	--pads_strides_dilations	0	0	2	2	1	1	--group-count	2"},
        {cmd_v + " --input	32	32	79	341	--weights	32	16	5	10	--pads_strides_dilations	0	0	2	2	1	1	--group-count	2"},
        {cmd_v + " --input	16	4	48	480	--weights	16	1	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	4"},
        {cmd_v + " --input	16	16	24	240	--weights	32	1	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	16"},
        {cmd_v + " --input	16	32	12	120	--weights	64	8	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	4"},
        {cmd_v + " --input	16	64	6	60	--weights	128	16	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	4"},
        {cmd_v + " --input	8	3	108	108	--weights	63	1	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	3"},
        {cmd_v + " --input	8	64	54	54	--weights	64	8	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	8"},
        {cmd_v + " --input	8	128	27	27	--weights	128	16	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	8"},
        {cmd_v + " --input	8	3	224	224	--weights	63	1	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	3"},
        {cmd_v + " --input	8	64	112	112	--weights	128	32	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	2"},
        {cmd_v + " --input	16	9	224	224	--weights	63	3	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	3"},
        //
        // Workaround for "Memory access fault by GPU node" during "FP32 gfx908 Hip Release All subset" - WrW disabled.
        {cmd_v + " --input	16	64	112	112	--weights	128	16	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	4 --disable-backward-weights"},
        {cmd_v + " --input	16	3	224	224	--weights	63	1	7	7	--pads_strides_dilations	3	3	2	2	1	1	--group-count	3"},
        {cmd_v + " --input	16	192	28	28	--weights	32	12	5	5	--pads_strides_dilations	2	2	1	1	1	1	--group-count	16"},
        {cmd_v + " --input	16	832	7	7	--weights	128	52	5	5	--pads_strides_dilations	2	2	1	1	1	1	--group-count	16"},
        {cmd_v + " --input	16	192	28	28	--weights	32	24	1	1	--pads_strides_dilations	0	0	1	1	1	1	--group-count	8"},
        {cmd_v + " --input	16	832	7	7	--weights	128	104	1	1	--pads_strides_dilations	0	0	1	1	1	1	--group-count	8"},
        {cmd_v + " --input	11	23	161	700	--weights	46	1	7	7	--pads_strides_dilations	1	1	2	2	1	1	--group-count	23"},
        {cmd_v + " --input	8	7	224	224	--weights	63	1	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	7"},
        {cmd_v + " --input	8	7	224	224	--weights	63	1	3	3	--pads_strides_dilations	0	0	1	1	1	1	--group-count	7"},
        {cmd_v + " --input	8	7	224	224	--weights	63	1	3	3	--pads_strides_dilations	0	0	2	2	1	1	--group-count	7"},
        {cmd_v + " --input	8	7	224	224	--weights	63	1	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	7"},
        {cmd_v + " --input	8	7	224	224	--weights	63	1	3	3	--pads_strides_dilations	2	2	2	2	1	1	--group-count	7"},
        {cmd_v + " --input	8	3	108	108	--weights	63	1	3	3	--pads_strides_dilations	1	1	1	1	1	1	--group-count	3"},
        {cmd_v + " --input	8	3	108	108	--weights	63	1	3	3	--pads_strides_dilations	0	0	1	1	1	1	--group-count	3"},
        {cmd_v + " --input	8	3	108	108	--weights	63	1	3	3	--pads_strides_dilations	0	0	2	2	1	1	--group-count	3"},
        {cmd_v + " --input	8	3	108	108	--weights	63	1	3	3	--pads_strides_dilations	1	1	2	2	1	1	--group-count	3"},
        {cmd_v + " --input	8	3	108	108	--weights	63	1	3	3	--pads_strides_dilations	2	2	2	2	1	1	--group-count	3"}
    };
    // clang-format on
}

using TestCase = decltype(GetTestCases(std::string{}))::value_type;

class GPU_ConvGroup_FP32 : public testing::TestWithParam<std::vector<TestCase>>
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
    if(!IsTestSupportedForDevice())
    {
        GTEST_SKIP();
    }
    std::vector<std::string> params = GPU_ConvGroup_FP32::GetParam();

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

} // namespace conv_group
using namespace conv_group;

TEST_P(GPU_ConvGroup_FP32, FloatTest_conv_group) { Run2dDriver(); };

INSTANTIATE_TEST_SUITE_P(Full, GPU_ConvGroup_FP32, testing::Values(GetTestCases("--float")));
