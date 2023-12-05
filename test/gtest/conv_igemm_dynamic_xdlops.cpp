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

using TestCase = std::tuple<std::vector<std::string>, std::string>;

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_GPU_XNACK_ENABLED)

static bool SkipTest(void)
{
    return miopen::IsEnabled(ENV(MIOPEN_TEST_GPU_XNACK_ENABLED)) ||
           miopen::IsDisabled(ENV(MIOPEN_TEST_ALL));
}

void GetArgs(const TestCase& param, std::vector<std::string>& tokens)
{
    auto env_vars = std::get<0>(param);
    for(auto& elem : env_vars)
    {
        putenv(elem.data());
    }

    auto cmd = std::get<1>(param);

    std::stringstream ss(cmd);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

class Conv2dFloat : public testing::TestWithParam<std::vector<TestCase>>
{
};

class Conv2dHalf : public testing::TestWithParam<std::vector<TestCase>>
{
};

void Run2dDriver(miopenDataType_t prec)
{

    std::vector<TestCase> params;
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
    std::string devName = handle.GetDeviceName();
    if(devName == "gfx908")
        return true;
    else
        return false;
}

TEST_P(Conv2dFloat, FloatTest)
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

TEST_P(Conv2dHalf, HalfTest)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && !SkipTest())
    {
        Run2dDriver(miopenHalf);
    }
    else
    {
        GTEST_SKIP();
    }
};

std::vector<TestCase> GetTestCases(const std::string& precision)
{

    std::vector<std::string> env_xdlops = {
        "MIOPEN_FIND_MODE=normal",
        "MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvAsmImplicitGemmGTCDynamicBwdXdlops;"
        "ConvAsmImplicitGemmGTCDynamicFwdXdlops;ConvAsmImplicitGemmGTCDynamicWrwXdlops"};

    const std::string v           = " --verbose";
    const std::string dis_bk_data = " --disable-backward-data";
    const std::string dis_bk_wei  = " --disable-backward-weights";
    const std::string dis_fwd     = " --disable-forward";
    const std::string dis_vali    = " --disable-validation";

    const std::vector<TestCase> test_cases = {
        // clang-format off
    //bwd
    TestCase{env_xdlops, precision + v + " --input  64  64 28 28 --weights 16  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei},
    TestCase{env_xdlops, precision + v + " --input  16  128 36 36 --weights 32  128 1 1  --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei},
    TestCase{env_xdlops, precision + v + " --input  64   64 56 56 --weights 256  64  1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei},
    TestCase{env_xdlops, precision + v + " --input  64  224 17 17 --weights 224  224  1 7 --pads_strides_dilations 0 3 1 1 1 1" + dis_fwd + dis_bk_wei},
    TestCase{env_xdlops, precision + v + " --input  128  128 35 35 --weights 256  128  3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_fwd + dis_bk_wei},
    TestCase{env_xdlops, precision + v + " --input  128  128 64 64 --weights 256  128  3 3 --pads_strides_dilations 1 1 2 2 1 1" + dis_fwd + dis_bk_wei},
    TestCase{env_xdlops, precision + v + " --input  128  768 17 17 --weights 256  768  3 3 --pads_strides_dilations 1 1 1 1 2 2" + dis_fwd + dis_bk_wei},
    TestCase{env_xdlops, precision + v + " --input  3  256 28 28 --weights 80  256  1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei},
    TestCase{env_xdlops, precision + v + " --input  2  256 12 18 --weights 256  256  3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_fwd + dis_bk_wei},
    TestCase{env_xdlops, precision + v + " --input  400  256 7 7 --weights 1024  256  7 7 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei},
    TestCase{env_xdlops, precision + v + " --input  400  256 1 1 --weights 1024  256  1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei},
    TestCase{env_xdlops, precision + v + " --input  8  16 5 5 --weights 8  16  2 2 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei},
    //ho=wo=1 stride=2
    TestCase{env_xdlops, precision + v + " --input  256 2048 2 2 --weights 1024  2048  1 1 --pads_strides_dilations 0 0 2 2 1 1" + dis_fwd + dis_bk_wei},
    //fwd
    //Be careful to add testings for (x=1, y=1, c % 8 != 0) due to WORKAROUND_SWDEV_306318
    TestCase{env_xdlops, precision + v + "  --input 64 1024 14 14 --weights 1024 1024 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_xdlops, precision + v + "  --input 64 256 56 56 --weights 512 256 1 1 --pads_strides_dilations 0 0 2 2 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_xdlops, precision + v + "  --input 64 2048 7 7 --weights 2048 2048 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_xdlops, precision + v + "  --input 128 128 17 17 --weights 128 128 7 1 --pads_strides_dilations 3 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_xdlops, precision + v + "  --input 128 128 17 17 --weights 128 128 1 7 --pads_strides_dilations 0 3 1 1 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_xdlops, precision + v + "  --input 128 192 17 17 --weights 320 192 3 3 --pads_strides_dilations 0 0 2 2 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_xdlops, precision + v + "  --input 128 256 35 35 --weights 64 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_xdlops, precision + v + "  --input 128 48 35 35 --weights 64 48 5 5 --pads_strides_dilations 2 2 1 1 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_xdlops, precision + v + "  --input 64 512 7 7 --weights 512 512 3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_xdlops, precision + v + "  --input 32 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_xdlops, precision + v + "  --input 2 256 100 104 --weights 12 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_xdlops, precision + v + "  --input 1 256 28 28 --weights 80 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    //ho=wo=1 stride=2
    TestCase{env_xdlops, precision + v + "  --input  256 2048 2 2 --weights 1024  2048  1 1 --pads_strides_dilations 0 0 2 2 1 1 " + dis_bk_data + dis_bk_wei},
    //wrw
    TestCase{env_xdlops, precision + v + "  --input  64  64 28 28 --weights 32  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_xdlops, precision + v + "  --input  16  128 36 36 --weights 32  128 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_xdlops, precision + v + "  --input  64   64 56 56 --weights 256  64  1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_xdlops, precision + v + "  --input  64  224 17 17 --weights 224  224  1 7 --pads_strides_dilations 0 3 1 1 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_xdlops, precision + v + "  --input  128  128 35 35 --weights 256  128  3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_xdlops, precision + v + "  --input  128  128 64 64 --weights 256  128  3 3 --pads_strides_dilations 1 1 2 2 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_xdlops, precision + v + "  --input  128  768 17 17 --weights 256  768  3 3 --pads_strides_dilations 1 1 1 1 2 2" + dis_fwd + dis_bk_data},
    TestCase{env_xdlops, precision + v + "  --input  3  256 28 28 --weights 80  256  1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_xdlops, precision + v + "  --input  2  256 12 18 --weights 256  256  3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_xdlops, precision + v + "  --input  4  512 128 128 --weights 12  512  1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    //regression test for issue 540
    TestCase{env_xdlops, precision + v + "  --input  4 32 79 141 --weights 64 32 5 10 --pads_strides_dilations 0 0 2 2 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_xdlops, precision + v + "  --input  400  256 7 7 --weights 1024  256  7 7 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_xdlops, precision + v + "  --input  400  256 1 1 --weights 1024  256  1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    //Regression test for SWDEV-295434 (FP16 only).
    TestCase{env_xdlops, precision + v + "  --input  120  256 3 3 --weights 340  256  3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_fwd + dis_bk_data},
    //ho=wo=1 stride=2
    TestCase{env_xdlops, precision + v + "  --input  256 2048 2 2 --weights 1024  2048  1 1 --pads_strides_dilations 0 0 2 2 1 1 " + dis_fwd + dis_bk_data}
        // clang-format on
    };
    return test_cases;
}

INSTANTIATE_TEST_SUITE_P(ConvIgemmDynamic, Conv2dFloat, testing::Values(GetTestCases("--float")));

INSTANTIATE_TEST_SUITE_P(ConvIgemmDynamic, Conv2dHalf, testing::Values(GetTestCases("--half")));
