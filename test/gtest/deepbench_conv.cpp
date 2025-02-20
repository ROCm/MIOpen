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
#include <tuple>
#include <string_view>

#include "gtest_common.hpp"

#include "../conv2d.hpp"

namespace deepbench_conv {

auto GetTestCases()
{
    const std::string v = " --verbose";

    return std::vector{
        // clang-format off
    std::pair{std::tuple<>{}, v + "	--input	4	1	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	8	1	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	16	1	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	32	1	161	700	--weights	32	1	5	20	--pads_strides_dilations	0	0	2	2	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	4	32	79	341	--weights	32	32	5	10	--pads_strides_dilations	0	0	2	2	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	8	32	79	341	--weights	32	32	5	10	--pads_strides_dilations	0	0	2	2	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	16	32	79	341	--weights	32	32	5	10	--pads_strides_dilations	0	0	2	2	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	32	32	79	341	--weights	32	32	5	10	--pads_strides_dilations	0	0	2	2	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	16	1	48	480	--weights	16	1	3	3	--pads_strides_dilations	1	1	1	1	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	16	16	24	240	--weights	32	16	3	3	--pads_strides_dilations	1	1	1	1	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	16	32	12	120	--weights	64	32	3	3	--pads_strides_dilations	1	1	1	1	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	16	64	6	60	--weights	128	64	3	3	--pads_strides_dilations	1	1	1	1	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	8	3	108	108	--weights	64	3	3	3	--pads_strides_dilations	1	1	2	2	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	8	64	54	54	--weights	64	64	3	3	--pads_strides_dilations	1	1	1	1	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	8	128	27	27	--weights	128	128	3	3	--pads_strides_dilations	1	1	1	1	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	8	128	14	14	--weights	256	128	3	3	--pads_strides_dilations	1	1	1	1	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	8	256	7	7	--weights	512	256	3	3	--pads_strides_dilations	1	1	1	1	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	8	3	224	224	--weights	64	3	3	3	--pads_strides_dilations	1	1	1	1	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	8	64	112	112	--weights	128	64	3	3	--pads_strides_dilations	1	1	1	1	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	8	128	56	56	--weights	256	128	3	3	--pads_strides_dilations	1	1	1	1	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	8	256	28	28	--weights	512	256	3	3	--pads_strides_dilations	1	1	1	1	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	8	512	14	14	--weights	512	512	3	3	--pads_strides_dilations	1	1	1	1	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	8	512	7	7	--weights	512	512	3	3	--pads_strides_dilations	1	1	1	1	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	16	3	224	224	--weights	64	3	3	3	--pads_strides_dilations	1	1	1	1	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	16	64	112	112	--weights	128	64	3	3	--pads_strides_dilations	1	1	1	1	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	16	128	56	56	--weights	256	128	3	3	--pads_strides_dilations	1	1	1	1	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	16	256	28	28	--weights	512	256	3	3	--pads_strides_dilations	1	1	1	1	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	16	512	14	14	--weights	512	512	3	3	--pads_strides_dilations	1	1	1	1	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	16	512	7	7	--weights	512	512	3	3	--pads_strides_dilations	1	1	1	1	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	16	3	224	224	--weights	64	3	7	7	--pads_strides_dilations	3	3	2	2	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	16	192	28	28	--weights	32	192	5	5	--pads_strides_dilations	2	2	1	1	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	16	512	14	14	--weights	48	512	5	5	--pads_strides_dilations	2	2	1	1	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	16	832	7	7	--weights	128	832	5	5	--pads_strides_dilations	2	2	1	1	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	16	192	28	28	--weights	32	192	1	1	--pads_strides_dilations	0	0	1	1	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	16	512	14	14	--weights	48	512	1	1	--pads_strides_dilations	0	0	1	1	1	1"},
    std::pair{std::tuple<>{}, v + "	--input	16	832	7	7	--weights	128	832	1	1	--pads_strides_dilations	0	0	1	1	1	1"}
        // clang-format on
    };
}

using TestCase = decltype(GetTestCases())::value_type;

class GPU_Conv2d_DeepBench_FP32 : public FloatTestCase<std::vector<TestCase>>
{
};

bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx103X, Gpu::gfx110X>;
    using d_mask = disabled<Gpu::Default>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}
} // namespace deepbench_conv
using namespace deepbench_conv;

TEST_P(GPU_Conv2d_DeepBench_FP32, FloatTest_deepbench_conv)
{
    if(IsTestSupportedForDevice())
    {
        invoke_with_params<conv2d_driver, GPU_Conv2d_DeepBench_FP32>(default_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(Full, GPU_Conv2d_DeepBench_FP32, testing::Values(GetTestCases()));
