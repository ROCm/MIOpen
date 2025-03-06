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

namespace conv_ck_igemm_fwd_v6r1_dlops_nchw {

auto GetTestCases()
{
    const auto env =
        std::tuple{std::pair{MIOPEN_FIND_MODE, "normal"},
                   std::pair{MIOPEN_DEBUG_FIND_ONLY_SOLVER, "ConvCkIgemmFwdV6r1DlopsNchw"},
                   std::pair{MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW, true}};

    const std::string v           = " --verbose";
    const std::string dis_bk_data = " --disable-backward-data";
    const std::string dis_bk_wei  = " --disable-backward-weights";

    return std::vector{
        // clang-format off
    std::pair{env, v + " --input 128 1024 14 14  --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env, v + " --input 128  256 14 14  --weights  256 1024 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env, v + " --input 128 1024 14 14  --weights  512 1024 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env, v + " --input 128  128 28 28  --weights  128 1024 3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env, v + " --input 128  128 28 28  --weights  512  128 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env, v + " --input 128  128 58 58  --weights  128  128 3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env, v + " --input 128 2048  7  7  --weights  512 2048 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env, v + " --input 128  256 14 14  --weights 1024  256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env, v + " --input 128  256 14 14  --weights  256  256 3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env, v + " --input 128  256 30 30  --weights  256  256 3 3 --pads_strides_dilations 0 0 2 2 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env, v + " --input 128  256 56 56  --weights  128  256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env, v + " --input 128  256 56 56  --weights  512  256 1 1 --pads_strides_dilations 0 0 2 2 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env, v + " --input 128  256 56 56  --weights   64  256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env, v + " --input 128  512 16 16  --weights  512  512 3 3 --pads_strides_dilations 0 0 2 2 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env, v + " --input 128  512 28 28  --weights 1024  512 1 1 --pads_strides_dilations 0 0 2 2 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env, v + " --input 128  512 28 28  --weights  128  512 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env, v + " --input 128  512 28 28  --weights  256  512 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env, v + " --input 128  512  7  7  --weights 2048  512 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env, v + " --input 128  512  7  7  --weights  512  512 3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env, v + " --input 128   64 56 56  --weights  256   64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env, v + " --input 128   64 56 56  --weights   64   64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env, v + " --input 128   64 56 56  --weights   64   64 3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_bk_data + dis_bk_wei}
        // clang-format on
    };
}

using TestCase = decltype(GetTestCases())::value_type;

class GPU_Conv2d_conv_ck_igemm_fwd_v6r1_dlops_nchw_FP32
    : public FloatTestCase<std::vector<TestCase>>
{
};

class GPU_Conv2d_conv_ck_igemm_fwd_v6r1_dlops_nchw_FP16 : public HalfTestCase<std::vector<TestCase>>
{
};

bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::Default>;
    using d_mask = disabled<Gpu::gfx908>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

} // namespace conv_ck_igemm_fwd_v6r1_dlops_nchw
using namespace conv_ck_igemm_fwd_v6r1_dlops_nchw;

TEST_P(GPU_Conv2d_conv_ck_igemm_fwd_v6r1_dlops_nchw_FP32, FloatTest)
{
    if(IsTestSupportedForDevice())
    {
        invoke_with_params<conv2d_driver, GPU_Conv2d_conv_ck_igemm_fwd_v6r1_dlops_nchw_FP32>(
            default_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(GPU_Conv2d_conv_ck_igemm_fwd_v6r1_dlops_nchw_FP16, HalfTest)
{
    if(IsTestSupportedForDevice())
    {
        invoke_with_params<conv2d_driver, GPU_Conv2d_conv_ck_igemm_fwd_v6r1_dlops_nchw_FP16>(
            default_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Conv2d_conv_ck_igemm_fwd_v6r1_dlops_nchw_FP32,
                         testing::Values(GetTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Conv2d_conv_ck_igemm_fwd_v6r1_dlops_nchw_FP16,
                         testing::Values(GetTestCases()));
