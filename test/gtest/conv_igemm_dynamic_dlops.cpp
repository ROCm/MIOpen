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

namespace {

auto GetTestCases()
{
    const auto env_fwd = std::tuple{
        std::pair{MIOPEN_FIND_MODE, "normal"},
        std::pair{MIOPEN_DEBUG_FIND_ONLY_SOLVER, "ConvAsmImplicitGemmGTCDynamicFwdDlopsNCHWC"}};

    const std::string v           = " --verbose";
    const std::string dis_bk_data = " --disable-backward-data";
    const std::string dis_bk_wei  = " --disable-backward-weights";
    const std::string in_nchw     = " --in_layout NCHW";
    const std::string fil_nchw    = " --fil_layout NCHW";
    const std::string fil_chwn    = " --fil_layout CHWN";
    const std::string out_nchw    = " --out_layout NCHW";
    const std::string tensor      = " --tensor_vect 1";
    const std::string vlen4       = " --vector_length 4";
    const std::string vlen8       = " --vector_length 8";

    const std::string common_base = " --cmode convfp16" + dis_bk_data + dis_bk_wei + in_nchw;

    const std::string nchwc_nchwc_base       = common_base + fil_nchw + out_nchw + tensor;
    const std::string nchwc_nchwc_fwd_fp16x4 = nchwc_nchwc_base + vlen4;
    const std::string nchwc_nchwc_fwd_fp16x8 = nchwc_nchwc_base + vlen8;

    const std::string nchwc_chwnc_base       = common_base + fil_chwn + out_nchw + tensor;
    const std::string nchwc_chwnc_fwd_fp16x4 = nchwc_chwnc_base + vlen4;
    const std::string nchwc_chwnc_fwd_fp16x8 = nchwc_chwnc_base + vlen8;

    return std::vector{
        // clang-format off
    //nchwc_nchwc_fwd_fp16x4
    std::pair{env_fwd, v + " --input  1 8  10  10  --weights 8 8 3 3     --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  32 160 73 73 --weights  64 160 1 1 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  16  64 56 56 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input   2 256 40 52 --weights 256 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input   2  64 32 28 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  32 128 14 14 --weights  64 128 1 1 --pads_strides_dilations 0 0 2 2 1 1" + nchwc_nchwc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  64  64 17 17 --weights 192  64 1 7 --pads_strides_dilations 0 3 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  64  64 17 17 --weights 192  64 7 1 --pads_strides_dilations 3 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input   4 128 28 28 --weights 128 128 2 2 --pads_strides_dilations 0 0 2 2 1 1" + nchwc_nchwc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  32 128  8  8 --weights 192 128 3 1 --pads_strides_dilations 1 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  64 192 17 17 --weights 160 192 3 3 --pads_strides_dilations 0 0 2 2 1 1" + nchwc_nchwc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  64  32 73 73 --weights  64  32 3 3 --pads_strides_dilations 1 1 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  16  64 56 56 --weights  64  64 3 3 --pads_strides_dilations 1 1 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  16  16 25 25 --weights  64  16 3 3 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  4  32 79 141 --weights 64  32 5 10 --pads_strides_dilations 0 0 2 2 1 1" + nchwc_nchwc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  400  256 7 7 --weights 1024 256 7 7 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  400  256 1 1 --weights 1024 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},

    //nchwc_chwnc_fwd_fp16x4
    std::pair{env_fwd, v + " --input  64 256  7  7 --weights 256 3 3  128  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  32 160 73 73 --weights 160 1 1   64  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  16  64 56 56 --weights  64 1 1   64  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input   2 256 40 52 --weights 256 1 1  256  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input   2  64 32 28 --weights  64 1 1   64  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  32 128 14 14 --weights 128 1 1   64  --pads_strides_dilations 0 0 2 2 1 1" + nchwc_chwnc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  64  64 17 17 --weights  64 3 7  192  --pads_strides_dilations 0 3 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  64  64 17 17 --weights  64 7 1  192  --pads_strides_dilations 3 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input   4 128 28 28 --weights 128 2 2  128  --pads_strides_dilations 0 0 2 2 1 1" + nchwc_chwnc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  32 128  8  8 --weights 128 3 1  192  --pads_strides_dilations 1 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  64 192 17 17 --weights 192 3 3  160  --pads_strides_dilations 0 0 2 2 1 1" + nchwc_chwnc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  64  32 73 73 --weights  32 3 3   64  --pads_strides_dilations 1 1 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  16  64 56 56 --weights  64 3 3   64  --pads_strides_dilations 1 1 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  16  16 25 25 --weights  16 3 3   64  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  4  32 79 141 --weights  32 5 10  64  --pads_strides_dilations 0 0 2 2 1 1" + nchwc_chwnc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  400  256 7 7 --weights 256 7 7 1024  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},
    std::pair{env_fwd, v + " --input  400  256 1 1 --weights 256 1 1 1024  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},

    //nchwc_nchwc_fwd_fp16x8
    std::pair{env_fwd, v + " --input  1 8  10  10  --weights 8 8 3 3     --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  32 160 73 73 --weights  64 160 1 1 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  16  64 56 56 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input   2 256 40 52 --weights 256 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input   2  64 32 28 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  32 128 14 14 --weights  64 128 1 1 --pads_strides_dilations 0 0 2 2 1 1" + nchwc_nchwc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  64  64 17 17 --weights 192  64 1 7 --pads_strides_dilations 0 3 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  64  64 17 17 --weights 192  64 7 1 --pads_strides_dilations 3 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input   4 128 28 28 --weights 128 128 2 2 --pads_strides_dilations 0 0 2 2 1 1" + nchwc_nchwc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  32 128  8  8 --weights 192 128 3 1 --pads_strides_dilations 1 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  64 192 17 17 --weights 160 192 3 3 --pads_strides_dilations 0 0 2 2 1 1" + nchwc_nchwc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  64  32 73 73 --weights  64  32 3 3 --pads_strides_dilations 1 1 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  16  64 56 56 --weights  64  64 3 3 --pads_strides_dilations 1 1 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  16  16 25 25 --weights  64  16 3 3 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  4  32 79 141 --weights 64  32 5 10 --pads_strides_dilations 0 0 2 2 1 1" + nchwc_nchwc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  400  256 7 7 --weights 1024 256 7 7 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  400  256 1 1 --weights 1024 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},

    //nchwc_chwnc_fwd_fp16x8
    std::pair{env_fwd, v + " --input  64 256  7  7 --weights 256 1 1  128  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  32 160 73 73 --weights 160 1 1   64  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  16  64 56 56 --weights  64 1 1   64  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input   2 256 40 52 --weights 256 1 1  256  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input   2  64 32 28 --weights  64 1 1   64  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  32 128 14 14 --weights 128 1 1   64  --pads_strides_dilations 0 0 2 2 1 1" + nchwc_chwnc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  64  64 17 17 --weights  64 1 7  192  --pads_strides_dilations 0 3 1 1 1 1" + nchwc_chwnc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  64  64 17 17 --weights  64 7 1  192  --pads_strides_dilations 3 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input   4 128 28 28 --weights 128 2 2  128  --pads_strides_dilations 0 0 2 2 1 1" + nchwc_chwnc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  32 128  8  8 --weights 128 3 1  192  --pads_strides_dilations 1 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  64 192 17 17 --weights 192 3 3  160  --pads_strides_dilations 0 0 2 2 1 1" + nchwc_chwnc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  64  32 73 73 --weights  32 3 3   64  --pads_strides_dilations 1 1 1 1 1 1" + nchwc_chwnc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  16  64 56 56 --weights  64 3 3   64  --pads_strides_dilations 1 1 1 1 1 1" + nchwc_chwnc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  16  16 25 25 --weights  16 3 3   64  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  4  32 79 141 --weights 32 5 10  64   --pads_strides_dilations 0 0 2 2 1 1" + nchwc_chwnc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  400  256 7 7 --weights  256 7 7 1024 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x8},
    std::pair{env_fwd, v + " --input  400  256 1 1 --weights  256 1 1 1024 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x8}
        // clang-format on
    };
}

using TestCase = decltype(GetTestCases())::value_type;

bool SkipTest() { return get_handle_xnack(); }

bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::gfx103X>;
    using d_mask = disabled<Gpu::gfx900, Gpu::gfx906, Gpu::gfx908, Gpu::gfx90A>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

} // namespace

class GPU_Conv2dDefault_FP16 : public HalfTestCase<std::vector<TestCase>>
{
};

TEST_P(GPU_Conv2dDefault_FP16, HalfTest_conv_igemm_dynamic_dlops)
{
    if(IsTestSupportedForDevice() && !SkipTest())
    {
        invoke_with_params<conv2d_driver, GPU_Conv2dDefault_FP16>(default_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(Full, GPU_Conv2dDefault_FP16, testing::Values(GetTestCases()));
