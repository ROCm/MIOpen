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

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)

namespace conv_igemm_dynamic_xdlops {

auto GetTestCases()
{
    const auto env_xdlops =
        std::tuple{std::pair{ENV(MIOPEN_FIND_MODE), "normal"},
                   std::pair{ENV(MIOPEN_DEBUG_FIND_ONLY_SOLVER),
                             "ConvAsmImplicitGemmGTCDynamicBwdXdlops;"
                             "ConvAsmImplicitGemmGTCDynamicFwdXdlops;"
                             "ConvAsmImplicitGemmGTCDynamicWrwXdlops"}};

    const std::string cmd_v       = " test_conv2d --verbose";
    const std::string dis_bk_data = " --disable-backward-data";
    const std::string dis_bk_wei  = " --disable-backward-weights";
    const std::string dis_fwd     = " --disable-forward";
    const std::string dis_vali    = " --disable-validation";

    return std::vector{
        // clang-format off
    //bwd
    std::pair{env_xdlops, cmd_v + " --input  64  64 28 28 --weights 16  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei},
    std::pair{env_xdlops, cmd_v + " --input  16  128 36 36 --weights 32  128 1 1  --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei},
    std::pair{env_xdlops, cmd_v + " --input  64   64 56 56 --weights 256  64  1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei},
    std::pair{env_xdlops, cmd_v + " --input  64  224 17 17 --weights 224  224  1 7 --pads_strides_dilations 0 3 1 1 1 1" + dis_fwd + dis_bk_wei},
    std::pair{env_xdlops, cmd_v + " --input  128  128 35 35 --weights 256  128  3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_fwd + dis_bk_wei},
    std::pair{env_xdlops, cmd_v + " --input  128  128 64 64 --weights 256  128  3 3 --pads_strides_dilations 1 1 2 2 1 1" + dis_fwd + dis_bk_wei},
    std::pair{env_xdlops, cmd_v + " --input  128  768 17 17 --weights 256  768  3 3 --pads_strides_dilations 1 1 1 1 2 2" + dis_fwd + dis_bk_wei},
    std::pair{env_xdlops, cmd_v + " --input  3  256 28 28 --weights 80  256  1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei},
    std::pair{env_xdlops, cmd_v + " --input  2  256 12 18 --weights 256  256  3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_fwd + dis_bk_wei},
    std::pair{env_xdlops, cmd_v + " --input  400  256 7 7 --weights 1024  256  7 7 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei},
    std::pair{env_xdlops, cmd_v + " --input  400  256 1 1 --weights 1024  256  1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei},
    std::pair{env_xdlops, cmd_v + " --input  8  16 5 5 --weights 8  16  2 2 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei},
    std::pair{env_xdlops, cmd_v + " --input  256 2048 2 2 --weights 1024  2048  1 1 --pads_strides_dilations 0 0 2 2 1 1" + dis_fwd + dis_bk_wei},
    //fwd
    //Be careful to add testings for (x=1, y=1, c % 8 != 0) due to WORKAROUND_SWDEV_306318
    std::pair{env_xdlops, cmd_v + "  --input 64 1024 14 14 --weights 1024 1024 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env_xdlops, cmd_v + "  --input 64 256 56 56 --weights 512 256 1 1 --pads_strides_dilations 0 0 2 2 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env_xdlops, cmd_v + "  --input 64 2048 7 7 --weights 2048 2048 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env_xdlops, cmd_v + "  --input 128 128 17 17 --weights 128 128 7 1 --pads_strides_dilations 3 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env_xdlops, cmd_v + "  --input 128 128 17 17 --weights 128 128 1 7 --pads_strides_dilations 0 3 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env_xdlops, cmd_v + "  --input 128 192 17 17 --weights 320 192 3 3 --pads_strides_dilations 0 0 2 2 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env_xdlops, cmd_v + "  --input 128 256 35 35 --weights 64 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env_xdlops, cmd_v + "  --input 128 48 35 35 --weights 64 48 5 5 --pads_strides_dilations 2 2 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env_xdlops, cmd_v + "  --input 64 512 7 7 --weights 512 512 3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env_xdlops, cmd_v + "  --input 32 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env_xdlops, cmd_v + "  --input 2 256 100 104 --weights 12 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    std::pair{env_xdlops, cmd_v + "  --input 1 256 28 28 --weights 80 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    //ho=wo=1 stride=2
    std::pair{env_xdlops, cmd_v + "  --input  256 2048 2 2 --weights 1024  2048  1 1 --pads_strides_dilations 0 0 2 2 1 1 " + dis_bk_data + dis_bk_wei},
    //wrw
    std::pair{env_xdlops, cmd_v + "  --input  64  64 28 28 --weights 32  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    std::pair{env_xdlops, cmd_v + "  --input  16  128 36 36 --weights 32  128 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    std::pair{env_xdlops, cmd_v + "  --input  64   64 56 56 --weights 256  64  1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    std::pair{env_xdlops, cmd_v + "  --input  64  224 17 17 --weights 224  224  1 7 --pads_strides_dilations 0 3 1 1 1 1" + dis_fwd + dis_bk_data},
    std::pair{env_xdlops, cmd_v + "  --input  128  128 35 35 --weights 256  128  3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_fwd + dis_bk_data},
    std::pair{env_xdlops, cmd_v + "  --input  128  128 64 64 --weights 256  128  3 3 --pads_strides_dilations 1 1 2 2 1 1" + dis_fwd + dis_bk_data},
    std::pair{env_xdlops, cmd_v + "  --input  128  768 17 17 --weights 256  768  3 3 --pads_strides_dilations 1 1 1 1 2 2" + dis_fwd + dis_bk_data},
    std::pair{env_xdlops, cmd_v + "  --input  3  256 28 28 --weights 80  256  1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    std::pair{env_xdlops, cmd_v + "  --input  2  256 12 18 --weights 256  256  3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_fwd + dis_bk_data},
    std::pair{env_xdlops, cmd_v + "  --input  4  512 128 128 --weights 12  512  1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    //regression test for issue 540
    std::pair{env_xdlops, cmd_v + "  --input  4 32 79 141 --weights 64 32 5 10 --pads_strides_dilations 0 0 2 2 1 1" + dis_fwd + dis_bk_data},
    std::pair{env_xdlops, cmd_v + "  --input  400  256 7 7 --weights 1024  256  7 7 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    std::pair{env_xdlops, cmd_v + "  --input  400  256 1 1 --weights 1024  256  1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    //Regression test for SWDEV-295434 (FP16 only).
    std::pair{env_xdlops, cmd_v + "  --input  120  256 3 3 --weights 340  256  3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_fwd + dis_bk_data},
    //ho=wo=1 stride=2
    std::pair{env_xdlops, cmd_v + "  --input  256 2048 2 2 --weights 1024  2048  1 1 --pads_strides_dilations 0 0 2 2 1 1 " + dis_fwd + dis_bk_data}
        // clang-format on
    };
}

using TestCase = decltype(GetTestCases())::value_type;

static bool SkipTest(const std::string& float_arg)
{
    if(!miopen::IsSet(ENV(MIOPEN_TEST_ALL)))
        return false;
    if(miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)))
        if(miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG)) == float_arg)
            return false;
    return true;
}

bool IsTestSupportedForDevice(const miopen::Handle& handle)
{
    const auto target = handle.GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false;
    using e_mask = enabled<Gpu::Default>;
    using d_mask = disabled<Gpu::gfx900, Gpu::gfx906, Gpu::gfx90A>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

} // namespace conv_igemm_dynamic_xdlops
using namespace conv_igemm_dynamic_xdlops;

class Conv2dDefaultFloat : public FloatTestCase<std::vector<TestCase>>
{
};

class Conv2dDefaultHalf : public HalfTestCase<std::vector<TestCase>>
{
};

TEST_P(Conv2dDefaultFloat, FloatTest_conv_igemm_dynamic_xdlops)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && !SkipTest("--float"))
    {
        invoke_with_params<conv2d_driver, Conv2dDefaultFloat>(default_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(Conv2dDefaultHalf, HalfTest_conv_igemm_dynamic_xdlops)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && !SkipTest("--half"))
    {
        invoke_with_params<conv2d_driver, Conv2dDefaultHalf>(default_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(ConvIgemmDynamic, Conv2dDefaultFloat, testing::Values(GetTestCases()));
INSTANTIATE_TEST_SUITE_P(ConvIgemmDynamic, Conv2dDefaultHalf, testing::Values(GetTestCases()));
