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
#include "../conv3d.hpp"
#include <miopen/miopen.h>
#include <gtest/gtest_common.hpp>
#include <gtest/gtest.h>
#include "get_handle.hpp"

namespace conv_3d {
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
    std::string cmd_v = "test_conv3d --verbose " + precision + " ";

    // clang-format off
    return std::vector<std::string>{
        {cmd_v + " --conv_dim_type conv3d --input 16    32   4    9     9  --weights    64    32   3  3    3  --pads_strides_dilations  0  0  0    2  2   2    1   1   1  --group-count   1   --cmode conv   --pmode   default"},
        {cmd_v + " --conv_dim_type conv3d --input  4     3   4  227   227  --weights     4     3   3 11   11  --pads_strides_dilations  0  0  0    1  1   1    1   1   1  --group-count   1   --cmode conv   --pmode   default"},
        {cmd_v + " --conv_dim_type conv3d --input 16   128   4   56    56  --weights   256     4   3  3    3  --pads_strides_dilations  1  1  1    1  1   1    1   1   1  --group-count   32  --cmode conv   --pmode   default"},
        {cmd_v + " --conv_dim_type conv3d --input 16   128  56   56    56  --weights   256     4   3  3    3  --pads_strides_dilations  1  2  3    1  1   1    1   2   3  --group-count   32  --cmode conv   --pmode   default"},
        {cmd_v + " --conv_dim_type conv3d --input  4     4   4  161   700  --weights    32     1   3  5   20  --pads_strides_dilations  1  1  1    2  2   2    1   1   1  --group-count   4   --cmode conv   --pmode   default"},
        {cmd_v + " --conv_dim_type conv3d --input  8   512   4   28    28  --weights   512   128   1  1    1  --pads_strides_dilations  0  0  0    1  1   1    1   1   1  --group-count   4   --cmode conv   --pmode   same"},
        {cmd_v + " --conv_dim_type conv3d --input  8   512   4   56    56  --weights   512   128   1  1    1  --pads_strides_dilations  0  0  0    2  2   2    1   1   1  --group-count   4   --cmode conv   --pmode   same"},
        {cmd_v + " --conv_dim_type conv3d --input  8   512   3   14    14  --weights   512   128   1  1    1  --pads_strides_dilations  0  0  0    2  2   2    1   1   1  --trans_output_pads 0 0 0 --group-count   1   --cmode trans  --pmode   same"},
        {cmd_v + " --conv_dim_type conv3d --input 16    64   3    4     4  --weights    64    32   1  3    3  --pads_strides_dilations  0  0  0    2  2   2    1   1   1  --trans_output_pads 0 0 0 --group-count   4   --cmode trans  --pmode   default"},
        {cmd_v + " --conv_dim_type conv3d --input 16    32   4    9     9  --weights    64    32   3  3    3  --pads_strides_dilations  0  0  0    1  2   3    1   2   3  --group-count   1   --cmode conv   --pmode   default"},
        {cmd_v + " --conv_dim_type conv3d --input  4     3   4  227   227  --weights     4     3   3 11   11  --pads_strides_dilations  0  0  0    1  1   1    1   2   3  --group-count   1   --cmode conv   --pmode   default"},
        {cmd_v + " --conv_dim_type conv3d --input 16   128   4   56    56  --weights   256     4   3  3    3  --pads_strides_dilations  1  2  3    1  1   1    1   2   3  --group-count   32  --cmode conv   --pmode   default"},
        {cmd_v + " --conv_dim_type conv3d --input  4     4   4  161   700  --weights    32     1   3  5   20  --pads_strides_dilations  1  2  3    1  2   3    1   2   3  --group-count   4   --cmode conv   --pmode   default"},
        {cmd_v + " --conv_dim_type conv3d --input  8   512   4   28    28  --weights   512   128   1  1    1  --pads_strides_dilations  0  0  0    1  1   1    1   2   3  --group-count   4   --cmode conv   --pmode   same"},
        {cmd_v + " --conv_dim_type conv3d --input  8   512   4   56    56  --weights   512   128   1  1    1  --pads_strides_dilations  0  0  0    1  2   3    1   2   3  --group-count   4   --cmode conv   --pmode   same"},
        {cmd_v + " --conv_dim_type conv3d --input  8   512   3   14    14  --weights   512   128   1  1    1  --pads_strides_dilations  0  0  0    1  2   3    1   2   3  --trans_output_pads 0 0 0 --group-count   1   --cmode trans  --pmode   same"},
        {cmd_v + " --conv_dim_type conv3d --input 16    64   3    4     4  --weights    64    32   1  3    3  --pads_strides_dilations  0  0  0    1  2   3    1   2   3  --trans_output_pads 0 0 0 --group-count   4   --cmode trans  --pmode   default"}
    };
    // clang-format on
}

using TestCase = decltype(GetTestCases(std::string{}))::value_type;

class GPU_conv3d_FP32 : public testing::TestWithParam<std::vector<TestCase>>
{
};

bool IsTestSupportedForDevice()
{
    using namespace miopen::debug;
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx103X, Gpu::gfx110X>;
    using d_mask = disabled<Gpu::Default>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

void Run2dDriver()
{
    if(!IsTestSupportedForDevice())
    {
        GTEST_SKIP();
    }

    std::vector<std::string> params = GPU_conv3d_FP32::GetParam();

    for(const auto& test_value : params)
    {
        std::vector<std::string> tokens;
        GetArgs(test_value, tokens);
        std::vector<const char*> ptrs;

        std::transform(tokens.begin(), tokens.end(), std::back_inserter(ptrs), [](const auto& str) {
            return str.data();
        });
        testing::internal::CaptureStderr();
        test_drive<conv3d_driver>(ptrs.size(), ptrs.data());
        auto capture = testing::internal::GetCapturedStderr();
        std::cout << capture;
    }
};

} // namespace conv_3d
using namespace conv_3d;

TEST_P(GPU_conv3d_FP32, FloatTest_conv_3d) { Run2dDriver(); };

INSTANTIATE_TEST_SUITE_P(Full, GPU_conv3d_FP32, testing::Values(GetTestCases("--float")));
