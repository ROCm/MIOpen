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

namespace conv_igemm_dynamic {

auto GetTestCases()
{
    const auto env =
        std::tuple{std::pair{MIOPEN_FIND_MODE, "normal"},
                   std::pair{MIOPEN_DEBUG_FIND_ONLY_SOLVER, "ConvAsmImplicitGemmV4R1DynamicFwd"}};
    const auto env_1x1 = std::tuple{
        std::pair{MIOPEN_FIND_MODE, "normal"},
        std::pair{MIOPEN_DEBUG_FIND_ONLY_SOLVER, "ConvAsmImplicitGemmV4R1DynamicFwd_1x1"}};
    const auto env_wrw =
        std::tuple{std::pair{MIOPEN_FIND_MODE, "normal"},
                   std::pair{MIOPEN_DEBUG_FIND_ONLY_SOLVER, "ConvAsmImplicitGemmV4R1DynamicWrw"}};
    const auto env_bwd =
        std::tuple{std::pair{MIOPEN_FIND_MODE, "normal"},
                   std::pair{MIOPEN_DEBUG_FIND_ONLY_SOLVER, "ConvAsmImplicitGemmV4R1DynamicBwd"}};

    const std::string v           = " --verbose";
    const std::string dis_bk_data = " --disable-backward-data";
    const std::string dis_bk_wei  = " --disable-backward-weights";
    const std::string dis_fwd     = " --disable-forward";
    const std::string dis_vali    = " --disable-validation";

    auto basic_tests = std::vector{
        // clang-format off
    std::pair{env    , v + " --input  16  16 56 56 --weights 64  16 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env    , v + " --input  16  64 34 34 --weights 64  64 3 3 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env    , v + " --input  32  32 17 17 --weights 32  32 1 7 --pads_strides_dilations 0 3 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env_1x1, v + " --input  16 384  8  8 --weights 64 384 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env_wrw, v + " --input  64  64 28 28 --weights 32  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    std::pair{env_wrw, v + " --input  16  128 36 36 --weights 32  128 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    std::pair{env_bwd, v + " --input  64  64 28 28 --weights 16  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei},
    std::pair{env_bwd, v + " --input  16  128 36 36 --weights 32  128 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei}
        // clang-format on
    };

    basic_tests.insert(basic_tests.end(),
                       {
                           // clang-format off
    std::pair{env    , v + " --input  64   64 56 56 --weights 256  64  1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env    , v + " --input  64  256 34 34 --weights 256  256 3 3 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env    , v + " --input 128  128 35 35 --weights 128  128 3 3 --pads_strides_dilations 0 0 2 2 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env    , v + " --input  64 1536  8  8 --weights 256 1536 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env    , v + " --input 128   48  7  7 --weights 128   48 5 5 --pads_strides_dilations 2 2 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env    , v + " --input 128  128 17 17 --weights 128  128 1 7 --pads_strides_dilations 0 3 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env_1x1, v + " --input 128  256 28 28 --weights 128  256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env_1x1, v + " --input  64 1536  8  8 --weights 256 1536 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env_1x1, v + " --input 128  768 17 17 --weights 128  768 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env_wrw, v + " --input  64   64 56 56 --weights 256  64  1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    std::pair{env_wrw, v + " --input  32  128 34 34 --weights 64  128  3 3 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    std::pair{env_wrw, v + " --input 128  128 35 35 --weights 128  128 3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_fwd + dis_bk_data},
    std::pair{env_wrw, v + " --input 128  256 56 56 --weights 64  256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    std::pair{env_wrw, v + " --input  64  512 28 28 --weights 256 512 1 1 --pads_strides_dilations 0 0 2 2 1 1" + dis_fwd + dis_bk_data},
    std::pair{env_wrw, v + " --input  64  512 14 14 --weights 256 512 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    std::pair{env_bwd, v + " --input  64   64 56 56 --weights 256  64  1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei},
    std::pair{env_bwd, v + " --input  32  128 34 34 --weights 64  128  3 3 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei},
    std::pair{env_bwd, v + " --input 128  128 35 35 --weights 128  128 3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_fwd + dis_bk_wei},
    std::pair{env_bwd, v + " --input 128  256 56 56 --weights 64  256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei}
                           // clang-format on
                       });

    return basic_tests;
}

using TestCase = decltype(GetTestCases())::value_type;

bool SkipTest() { return get_handle_xnack(); }

class GPU_Conv2dDynamic_FP32 : public FloatTestCase<std::vector<TestCase>>
{
};

bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::Default>;
    using d_mask = disabled<Gpu::gfx908, Gpu::gfx90A>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

} // namespace conv_igemm_dynamic
using namespace conv_igemm_dynamic;

TEST_P(GPU_Conv2dDynamic_FP32, FloatTest_conv_igemm_dynamic)
{
    if(IsTestSupportedForDevice() && !SkipTest())
    {
        invoke_with_params<conv2d_driver, GPU_Conv2dDynamic_FP32>(default_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(Full, GPU_Conv2dDynamic_FP32, testing::Values(GetTestCases()));
