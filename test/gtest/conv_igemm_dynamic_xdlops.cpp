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
#include <miopen/miopen.h>
#include <gtest/gtest.h>
#include "../conv2d.hpp"
#include "get_handle.hpp"

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_GPU_XNACK_ENABLED)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_FIND_MODE)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEBUG_FIND_ONLY_SOLVER)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)

namespace conv_igemm_dynamic_xdlops {

static bool SkipTest(void)
{
    return !(miopen::IsUnset(ENV(MIOPEN_TEST_ALL)) &&
             miopen::IsUnset(ENV(MIOPEN_TEST_FLOAT_ARG))) ||
           miopen::IsEnabled(ENV(MIOPEN_TEST_GPU_XNACK_ENABLED)) ||
           miopen::IsDisabled(ENV(MIOPEN_TEST_ALL));
}

void SetupEnvVar(void)
{
    miopen::UpdateEnvVar(ENV(MIOPEN_FIND_MODE), std::string("normal"));
    miopen::UpdateEnvVar(
        ENV(MIOPEN_DEBUG_FIND_ONLY_SOLVER),
        std::string("ConvAsmImplicitGemmGTCDynamicBwdXdlops;ConvAsmImplicitGemmGTCDynamicFwdXdlops;"
                    "ConvAsmImplicitGemmGTCDynamicWrwXdlops"));
}

void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

class Conv2dFloat : public testing::TestWithParam<std::vector<std::string>>
{
};

class Conv2dHalf : public testing::TestWithParam<std::vector<std::string>>
{
};

void Run2dDriver(miopenDataType_t prec)
{

    std::vector<std::string> params;
    switch(prec)
    {
    case miopenFloat: params = Conv2dFloat::GetParam(); break;
    case miopenHalf: params = Conv2dHalf::GetParam(); break;
    case miopenInt8:
    case miopenBFloat16:
    case miopenInt32:
    case miopenDouble:
    case miopenFloat8:
    case miopenBFloat8:
        FAIL() << "miopenInt8, miopenBFloat16, miopenInt32, "
                  "miopenDouble, miopenFloat8, miopenBFloat8 "
                  "data type not supported by conv_igemm_dynamic_xdlops test";

    default: params = Conv2dFloat::GetParam();
    }

    SetupEnvVar();

    for(const auto& test_value : params)
    {
        std::vector<std::string> tokens;
        GetArgs(test_value, tokens);
        std::vector<const char*> ptrs;

        std::transform(tokens.begin(),
                       tokens.end(),
                       std::back_inserter(ptrs),
                       [](const std::string& str) { return str.data(); });

        testing::internal::CaptureStderr();
        test_drive<conv2d_driver>(ptrs.size(), ptrs.data());
        auto capture = testing::internal::GetCapturedStderr();
        std::cout << capture;
    }
};

bool IsTestSupportedForDevice(const miopen::Handle& handle)
{
    const auto target   = handle.GetTargetProperties();
    std::string devName = handle.GetDeviceName();
    if(target.Xnack() && *target.Xnack())
        return false;

    if(devName == "gfx908")
        return true;
    else
        return false;
}

std::vector<std::string> GetTestCases(const std::string& precision)
{
    const std::string flags       = "test_conv2d " + precision + " --verbose ";
    const std::string dis_bk_data = " --disable-backward-data";
    const std::string dis_bk_wei  = " --disable-backward-weights";
    const std::string dis_fwd     = " --disable-forward";
    const std::string dis_vali    = " --disable-validation";

    const std::vector<std::string> test_cases = {
        // clang-format off
    //bwd
    {flags + " --input  64  64 28 28 --weights 16  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei},
    {flags + " --input  16  128 36 36 --weights 32  128 1 1  --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei},
    {flags + " --input  64   64 56 56 --weights 256  64  1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei},
    {flags + " --input  64  224 17 17 --weights 224  224  1 7 --pads_strides_dilations 0 3 1 1 1 1" + dis_fwd + dis_bk_wei},
    {flags + " --input  128  128 35 35 --weights 256  128  3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_fwd + dis_bk_wei},
    {flags + " --input  128  128 64 64 --weights 256  128  3 3 --pads_strides_dilations 1 1 2 2 1 1" + dis_fwd + dis_bk_wei},
    {flags + " --input  128  768 17 17 --weights 256  768  3 3 --pads_strides_dilations 1 1 1 1 2 2" + dis_fwd + dis_bk_wei},
    {flags + " --input  3  256 28 28 --weights 80  256  1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei},
    {flags + " --input  2  256 12 18 --weights 256  256  3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_fwd + dis_bk_wei},
    {flags + " --input  400  256 7 7 --weights 1024  256  7 7 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei},
    {flags + " --input  400  256 1 1 --weights 1024  256  1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei},
    {flags + " --input  8  16 5 5 --weights 8  16  2 2 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei},
    //ho=wo=1 stride=2
    {flags + " --input  256 2048 2 2 --weights 1024  2048  1 1 --pads_strides_dilations 0 0 2 2 1 1" + dis_fwd + dis_bk_wei},
    //fwd
    //Be careful to add testings for (x=1, y=1, c % 8 != 0) due to WORKAROUND_SWDEV_306318
    {flags + "  --input 64 1024 14 14 --weights 1024 1024 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    {flags + "  --input 64 256 56 56 --weights 512 256 1 1 --pads_strides_dilations 0 0 2 2 1 1" + dis_bk_data + dis_bk_wei},
    {flags + "  --input 64 2048 7 7 --weights 2048 2048 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    {flags + "  --input 128 128 17 17 --weights 128 128 7 1 --pads_strides_dilations 3 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    {flags + "  --input 128 128 17 17 --weights 128 128 1 7 --pads_strides_dilations 0 3 1 1 1 1" + dis_bk_data + dis_bk_wei},
    {flags + "  --input 128 192 17 17 --weights 320 192 3 3 --pads_strides_dilations 0 0 2 2 1 1" + dis_bk_data + dis_bk_wei},
    {flags + "  --input 128 256 35 35 --weights 64 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    {flags + "  --input 128 48 35 35 --weights 64 48 5 5 --pads_strides_dilations 2 2 1 1 1 1" + dis_bk_data + dis_bk_wei},
    {flags + "  --input 64 512 7 7 --weights 512 512 3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_bk_data + dis_bk_wei},
    {flags + "  --input 32 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1" + dis_bk_data + dis_bk_wei},
    {flags + "  --input 2 256 100 104 --weights 12 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    {flags + "  --input 1 256 28 28 --weights 80 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    //ho=wo=1 stride=2
    {flags + "  --input  256 2048 2 2 --weights 1024  2048  1 1 --pads_strides_dilations 0 0 2 2 1 1 " + dis_bk_data + dis_bk_wei},
    //wrw
    {flags + "  --input  64  64 28 28 --weights 32  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    {flags + "  --input  16  128 36 36 --weights 32  128 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    {flags + "  --input  64   64 56 56 --weights 256  64  1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    {flags + "  --input  64  224 17 17 --weights 224  224  1 7 --pads_strides_dilations 0 3 1 1 1 1" + dis_fwd + dis_bk_data},
    {flags + "  --input  128  128 35 35 --weights 256  128  3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_fwd + dis_bk_data},
    {flags + "  --input  128  128 64 64 --weights 256  128  3 3 --pads_strides_dilations 1 1 2 2 1 1" + dis_fwd + dis_bk_data},
    {flags + "  --input  128  768 17 17 --weights 256  768  3 3 --pads_strides_dilations 1 1 1 1 2 2" + dis_fwd + dis_bk_data},
    {flags + "  --input  3  256 28 28 --weights 80  256  1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    {flags + "  --input  2  256 12 18 --weights 256  256  3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_fwd + dis_bk_data},
    {flags + "  --input  4  512 128 128 --weights 12  512  1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    //regression test for issue 540
    {flags + "  --input  4 32 79 141 --weights 64 32 5 10 --pads_strides_dilations 0 0 2 2 1 1" + dis_fwd + dis_bk_data},
    {flags + "  --input  400  256 7 7 --weights 1024  256  7 7 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    {flags + "  --input  400  256 1 1 --weights 1024  256  1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    //Regression test for SWDEV-295434 (FP16 only).
    {flags + "  --input  120  256 3 3 --weights 340  256  3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_fwd + dis_bk_data},
    //ho=wo=1 stride=2
    {flags + "  --input  256 2048 2 2 --weights 1024  2048  1 1 --pads_strides_dilations 0 0 2 2 1 1 " + dis_fwd + dis_bk_data}
        // clang-format on
    };
    return test_cases;
}

} // namespace conv_igemm_dynamic_xdlops
using namespace conv_igemm_dynamic_xdlops;

TEST_P(Conv2dFloat, FloatTest_Conv_Igemm_Dynamic_dlops)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && !SkipTest() &&
       miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG)) == "--float")
    {
        Run2dDriver(miopenFloat);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(Conv2dHalf, HalfTest_conv_igemm_dynamic_xdlops)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && !SkipTest() &&
       miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG)) == "--half")
    {
        Run2dDriver(miopenHalf);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(ConvIgemmDynamic, Conv2dFloat, testing::Values(GetTestCases("--float")));

INSTANTIATE_TEST_SUITE_P(ConvIgemmDynamic, Conv2dHalf, testing::Values(GetTestCases("--half")));
