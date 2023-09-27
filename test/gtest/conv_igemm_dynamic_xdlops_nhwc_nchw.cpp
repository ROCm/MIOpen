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
#include <miopen/miopen.h>
#include "../conv2d.hpp"
#include "get_handle.hpp"

using TestCase = std::tuple<std::vector<std::string>, std::string>;

MIOPEN_DECLARE_ENV_VAR(MIOPEN_TEST_ALL)

MIOPEN_DECLARE_ENV_VAR(MIOPEN_TEST_GPU_XNACK_ENABLED)

static bool SkipTest(void)
{
    return miopen::IsEnabled(MIOPEN_TEST_GPU_XNACK_ENABLED{}) ||
           miopen::IsDisabled(MIOPEN_TEST_ALL{});
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
    case miopenInt8x4:
    case miopenInt32:
    case miopenDouble:
    case miopenFloat8:
    case miopenBFloat8:
        FAIL() << "miopenHalf, miopenInt8, miopenBFloat16, miopenInt8x4, miopenInt32, "
                  "miopenDouble, miopenFloat8, miopenBFloat8 "
                  "data type not supported by conv_igemm_dynamic test";

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
    if(devName == "gfx908" || devName == "gfx90a" || miopen::StartsWith(devName, "gfx94"))
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

    std::vector<std::string> env_nhwc_fwd = {
        "MIOPEN_FIND_MODE=normal",
        "MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC"};

    std::vector<std::string> env_nhwc_bwd = {
        "MIOPEN_FIND_MODE=normal",
        "MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC"};

    std::vector<std::string> env_nhwc_wrw = {
        "MIOPEN_FIND_MODE=normal",
        "MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvAsmImplicitGemmGTCDynamicWrwXdlops"};

    std::string v             = " --verbose";
    std::string dis_bk_data   = " --disable-backward-data";
    std::string dis_bk_wei    = " --disable-backward-weights";
    std::string dis_fwd       = " --disable-forward";
    std::string dis_vali      = " --disable-validation";
    std::string in_nhwc       = " --in_layout NHWC";
    std::string fil_nhwc      = " --fil_layout NHWC";
    std::string out_nhwc      = " --out_layout NHWC";
    std::string args_nhwc_wrw = dis_fwd + dis_bk_data + in_nhwc + fil_nhwc + out_nhwc;

    const std::vector<TestCase> test_cases = {
        // clang-format off
    //nhwc_fwd
    TestCase{env_nhwc_fwd, precision + v + "  --input  64 256  7  7 --weights 128 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_fwd, precision + v + "  --input  32 160 73 73 --weights  64 160 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_fwd, precision + v + "  --input  16  64 56 56 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_fwd, precision + v + "  --input   2 256 40 52 --weights 256 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_fwd, precision + v + "  --input   2  64 59 57 --weights  12  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_fwd, precision + v + "  --input  32 128 14 14 --weights  64 128 1 1 --pads_strides_dilations 0 0 2 2 1 1" + dis_bk_data + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_fwd, precision + v + "  --input  64  64 17 17 --weights 192  64 1 7 --pads_strides_dilations 0 3 1 1 1 1" + dis_bk_data + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_fwd, precision + v + "  --input  64  64 17 17 --weights 192  64 7 1 --pads_strides_dilations 3 0 1 1 1 1" + dis_bk_data + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_fwd, precision + v + "  --input   4 128 28 28 --weights 128 128 2 2 --pads_strides_dilations 0 0 2 2 1 1" + dis_bk_data + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_fwd, precision + v + "  --input  32 128  8  8 --weights 192 128 3 1 --pads_strides_dilations 1 0 1 1 1 1" + dis_bk_data + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_fwd, precision + v + "  --input  64 192 17 17 --weights 160 192 3 3 --pads_strides_dilations 0 0 2 2 1 1" + dis_bk_data + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_fwd, precision + v + "  --input  64  32 73 73 --weights  64  32 3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_bk_data + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_fwd, precision + v + "  --input  16  64 56 56 --weights  64  64 3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_bk_data + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_fwd, precision + v + "  --input  64   3 78 78 --weights  64   3 7 7 --pads_strides_dilations 0 0 2 2 1 1" + dis_bk_data + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_fwd, precision + v + "  --input  16 192 17 17 --weights 224 192 1 7 --pads_strides_dilations 0 3 1 1 1 1" + dis_bk_data + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_fwd, precision + v + "  --input  16   3 17 17 --weights  64   3 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_fwd, precision + v + "  --input   2  64 19 19 --weights 510  64 3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_bk_data + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    // tensor larger than 4GB
    TestCase{env_nhwc_fwd, precision + v + "  --input 2048  1 512 1024 --weights 1  1 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    // ho=wo=1 stride=2
    TestCase{env_nhwc_fwd, precision + v + "  --input  256 2048 2 2 --weights 1024  2048  1 1 --pads_strides_dilations 0 0 2 2 1 1" + dis_bk_data + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    
    //nhwc_fwd_nchw
    TestCase{env_nhwc_fwd, precision + v + "  --input  64 256   7   7 --weights 128 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_nhwc_fwd, precision + v + "  --input  32 160  73  73 --weights  64 160 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_nhwc_fwd, precision + v + "  --input  16  64  56  56 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_nhwc_fwd, precision + v + "  --input   2 256  40  52 --weights 256 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_nhwc_fwd, precision + v + "  --input   2  64  59  57 --weights  12  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_nhwc_fwd, precision + v + "  --input  32 128  14  14 --weights  64 128 1 1 --pads_strides_dilations 0 0 2 2 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_nhwc_fwd, precision + v + "  --input  64  64  17  17 --weights 192  64 1 7 --pads_strides_dilations 0 3 1 1 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_nhwc_fwd, precision + v + "  --input  64  64  17  17 --weights 192  64 7 1 --pads_strides_dilations 3 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_nhwc_fwd, precision + v + "  --input   4 128  28  28 --weights 128 128 2 2 --pads_strides_dilations 0 0 2 2 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_nhwc_fwd, precision + v + "  --input  32 128   8   8 --weights 192 128 3 1 --pads_strides_dilations 1 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_nhwc_fwd, precision + v + "  --input  64 192  17  17 --weights 160 192 3 3 --pads_strides_dilations 0 0 2 2 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_nhwc_fwd, precision + v + "  --input  64  32  73  73 --weights  64  32 3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_nhwc_fwd, precision + v + "  --input  16  64  56  56 --weights  64  64 3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_nhwc_fwd, precision + v + "  --input  64   3  78  78 --weights  64   3 7 7 --pads_strides_dilations 0 0 2 2 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_nhwc_fwd, precision + v + "  --input  16 192  17  17 --weights 224 192 1 7 --pads_strides_dilations 0 3 1 1 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_nhwc_fwd, precision + v + "  --input  16   3  17  17 --weights  64   3 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_bk_data + dis_bk_wei},
    TestCase{env_nhwc_fwd, precision + v + "  --input   2  64  19  19 --weights 510  64 3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_bk_data + dis_bk_wei},
    // TODO: disabled for WORKAROUND_ISSUE_1979
    //TestCase{env_nhwc_fwd, precision + v + "  --input  16   3 224 224 --weights  63   1 3 3 --pads_strides_dilations 1 1 1 1 1 1 --group-count 3" + dis_bk_data + dis_bk_wei},

    //nhwc_bwd
    TestCase{env_nhwc_bwd, precision + v + "  --input  64 256  7  7 --weights 128 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_bwd, precision + v + "  --input  32 160 73 73 --weights  64 160 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_bwd, precision + v + "  --input  16  64 56 56 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_bwd, precision + v + "  --input   2 256 40 52 --weights 256 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_bwd, precision + v + "  --input   2  64 32 28 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_bwd, precision + v + "  --input  32 128 14 14 --weights  64 128 1 1 --pads_strides_dilations 0 0 2 2 1 1" + dis_fwd + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_bwd, precision + v + "  --input  64  64 17 17 --weights 192  64 1 7 --pads_strides_dilations 0 3 1 1 1 1" + dis_fwd + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_bwd, precision + v + "  --input  64  64 17 17 --weights 192  64 7 1 --pads_strides_dilations 3 0 1 1 1 1" + dis_fwd + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_bwd, precision + v + "  --input   4 128 28 28 --weights 128 128 2 2 --pads_strides_dilations 0 0 2 2 1 1" + dis_fwd + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_bwd, precision + v + "  --input  32 128  8  8 --weights 192 128 3 1 --pads_strides_dilations 1 0 1 1 1 1" + dis_fwd + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_bwd, precision + v + "  --input  64 192 17 17 --weights 160 192 3 3 --pads_strides_dilations 0 0 2 2 1 1" + dis_fwd + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_bwd, precision + v + "  --input  64  32 73 73 --weights  64  32 3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_fwd + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_bwd, precision + v + "  --input  16  64 56 56 --weights  64  64 3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_fwd + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_bwd, precision + v + "  --input  16  16 25 25 --weights  64  16 3 3 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_bwd, precision + v + "  --input  15 256 1  1  --weights 340 256 3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_fwd + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_bwd, precision + v + "  --input  15 128 10 10 --weights 340 128 3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_fwd + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    TestCase{env_nhwc_bwd, precision + v + "  --input   2  64 19 19 --weights 510  64 3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_fwd + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    // tensor larger than 4GB
    TestCase{env_nhwc_bwd, precision + v + "  --input 2048  1 512 1024 --weights 1  1 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},
    // ho=wo=1 stride=2
    TestCase{env_nhwc_bwd, precision + v + "  --input  256 2048 2 2 --weights 1024  2048  1 1 --pads_strides_dilations 0 0 2 2 1 1" + dis_fwd + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc},

    //nhwc_bwd_nchw
    TestCase{env_nhwc_bwd, precision + v + "  --input  64 256  7  7 --weights 128 256 1 1 --pads_strides_dilations 0 0 1 1 1 1 " + dis_fwd + dis_bk_wei},
    TestCase{env_nhwc_bwd, precision + v + "  --input  32 160 73 73 --weights  64 160 1 1 --pads_strides_dilations 0 0 1 1 1 1 " + dis_fwd + dis_bk_wei},
    TestCase{env_nhwc_bwd, precision + v + "  --input  16  64 56 56 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1 " + dis_fwd + dis_bk_wei},
    TestCase{env_nhwc_bwd, precision + v + "  --input   2 256 40 52 --weights 256 256 1 1 --pads_strides_dilations 0 0 1 1 1 1 " + dis_fwd + dis_bk_wei},
    TestCase{env_nhwc_bwd, precision + v + "  --input   2  64 32 28 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1 " + dis_fwd + dis_bk_wei},
    TestCase{env_nhwc_bwd, precision + v + "  --input  32 128 14 14 --weights  64 128 1 1 --pads_strides_dilations 0 0 2 2 1 1 " + dis_fwd + dis_bk_wei},
    TestCase{env_nhwc_bwd, precision + v + "  --input  64  64 17 17 --weights 192  64 1 7 --pads_strides_dilations 0 3 1 1 1 1 " + dis_fwd + dis_bk_wei},
    TestCase{env_nhwc_bwd, precision + v + "  --input  64  64 17 17 --weights 192  64 7 1 --pads_strides_dilations 3 0 1 1 1 1 " + dis_fwd + dis_bk_wei},
    TestCase{env_nhwc_bwd, precision + v + "  --input   4 128 28 28 --weights 128 128 2 2 --pads_strides_dilations 0 0 2 2 1 1 " + dis_fwd + dis_bk_wei},
    TestCase{env_nhwc_bwd, precision + v + "  --input  32 128  8  8 --weights 192 128 3 1 --pads_strides_dilations 1 0 1 1 1 1 " + dis_fwd + dis_bk_wei},
    TestCase{env_nhwc_bwd, precision + v + "  --input  64 192 17 17 --weights 160 192 3 3 --pads_strides_dilations 0 0 2 2 1 1 " + dis_fwd + dis_bk_wei},
    TestCase{env_nhwc_bwd, precision + v + "  --input  64  32 73 73 --weights  64  32 3 3 --pads_strides_dilations 1 1 1 1 1 1 " + dis_fwd + dis_bk_wei},
    TestCase{env_nhwc_bwd, precision + v + "  --input  16  64 56 56 --weights  64  64 3 3 --pads_strides_dilations 1 1 1 1 1 1 " + dis_fwd + dis_bk_wei},
    TestCase{env_nhwc_bwd, precision + v + "  --input  16  16 25 25 --weights  64  16 3 3 --pads_strides_dilations 0 0 1 1 1 1 " + dis_fwd + dis_bk_wei},
    TestCase{env_nhwc_bwd, precision + v + "  --input  15 256 1  1  --weights 340 256 3 3 --pads_strides_dilations 1 1 1 1 1 1 " + dis_fwd + dis_bk_wei},
    TestCase{env_nhwc_bwd, precision + v + "  --input  15 128 10 10 --weights 340 128 3 3 --pads_strides_dilations 1 1 1 1 1 1 " + dis_fwd + dis_bk_wei},
    TestCase{env_nhwc_bwd, precision + v + "  --input   2  64 19 19 --weights 510  64 3 3 --pads_strides_dilations 1 1 1 1 1 1 " + dis_fwd + dis_bk_wei},

    //nhwc_wrw
    TestCase{env_nhwc_wrw, precision + v + "  --input  64 256  7  7 --weights 128 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + args_nhwc_wrw},
    TestCase{env_nhwc_wrw, precision + v + "  --input  32 160 73 73 --weights  64 160 1 1 --pads_strides_dilations 0 0 1 1 1 1" + args_nhwc_wrw},
    TestCase{env_nhwc_wrw, precision + v + "  --input  16  64 56 56 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + args_nhwc_wrw},
    TestCase{env_nhwc_wrw, precision + v + "  --input   2 256 40 52 --weights 256 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + args_nhwc_wrw},
    TestCase{env_nhwc_wrw, precision + v + "  --input   2  64 32 28 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + args_nhwc_wrw},
    TestCase{env_nhwc_wrw, precision + v + "  --input  32 128 14 14 --weights  64 128 1 1 --pads_strides_dilations 0 0 2 2 1 1" + args_nhwc_wrw},
    TestCase{env_nhwc_wrw, precision + v + "  --input  64  64 17 17 --weights 192  64 1 7 --pads_strides_dilations 0 3 1 1 1 1" + args_nhwc_wrw},
    TestCase{env_nhwc_wrw, precision + v + "  --input  64  64 17 17 --weights 192  64 7 1 --pads_strides_dilations 3 0 1 1 1 1" + args_nhwc_wrw},
    TestCase{env_nhwc_wrw, precision + v + "  --input   4 128 28 28 --weights 128 128 2 2 --pads_strides_dilations 0 0 2 2 1 1" + args_nhwc_wrw},
    TestCase{env_nhwc_wrw, precision + v + "  --input  32 128  8  8 --weights 192 128 3 1 --pads_strides_dilations 1 0 1 1 1 1" + args_nhwc_wrw},
    TestCase{env_nhwc_wrw, precision + v + "  --input  64 192 17 17 --weights 160 192 3 3 --pads_strides_dilations 0 0 2 2 1 1" + args_nhwc_wrw},
    TestCase{env_nhwc_wrw, precision + v + "  --input  64  32 73 73 --weights  64  32 3 3 --pads_strides_dilations 1 1 1 1 1 1" + args_nhwc_wrw},
    TestCase{env_nhwc_wrw, precision + v + "  --input  16  64 56 56 --weights  64  64 3 3 --pads_strides_dilations 1 1 1 1 1 1" + args_nhwc_wrw},
    TestCase{env_nhwc_wrw, precision + v + "  --input  16  16 25 25 --weights  64  16 3 3 --pads_strides_dilations 0 0 1 1 1 1" + args_nhwc_wrw},
    TestCase{env_nhwc_wrw, precision + v + "  --input  4 32 79 141 --weights 64 32 5 10 --pads_strides_dilations 0 0 2 2 1 1" + args_nhwc_wrw},
    TestCase{env_nhwc_wrw, precision + v + "  --input  400  256 7 7 --weights 1024  256  7 7 --pads_strides_dilations 0 0 1 1 1 1" + args_nhwc_wrw},
    TestCase{env_nhwc_wrw, precision + v + "  --input  400  256 1 1 --weights 1024  256  1 1 --pads_strides_dilations 0 0 1 1 1 1" + args_nhwc_wrw},
    TestCase{env_nhwc_wrw, precision + v + "  --input  1 3 32 32 --weights 1 3 11 11 --pads_strides_dilations 1 1 2 2 2 1" + args_nhwc_wrw},
    TestCase{env_nhwc_wrw, precision + v + "  --input  1 3 224 224 --weights 1 3 3 3 --pads_strides_dilations 0 0 1 1 2 2" + args_nhwc_wrw},
    TestCase{env_nhwc_wrw, precision + v + "  --input  1 1 8 8 --weights 1 1 2 2 --pads_strides_dilations 0 0 1 1 2 2" + args_nhwc_wrw},
    TestCase{env_nhwc_wrw, precision + v + "  --input  1 128 56 56 --weights 1 128 5 5 --pads_strides_dilations 0 0 2 2 1 1" + args_nhwc_wrw},
    TestCase{env_nhwc_wrw, precision + v + "  --input  2 64 19 19 --weights 510 64 3 3 --pads_strides_dilations 1 1 1 1 1 1" + args_nhwc_wrw},
    // ho=wo=1 stride=2
    TestCase{env_nhwc_wrw, precision + v + "  --input  256 2048 2 2 --weights 1024  2048  1 1 --pads_strides_dilations 0 0 2 2 1 1" + args_nhwc_wrw},

    //nhwc_wrw_nchw
    TestCase{env_nhwc_wrw, precision + v + "  --input  64 256  7  7 --weights 128 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_nhwc_wrw, precision + v + "  --input  32 160 73 73 --weights  64 160 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_nhwc_wrw, precision + v + "  --input  16  64 56 56 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_nhwc_wrw, precision + v + "  --input   2 256 40 52 --weights 256 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_nhwc_wrw, precision + v + "  --input   2  64 32 28 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_nhwc_wrw, precision + v + "  --input  32 128 14 14 --weights  64 128 1 1 --pads_strides_dilations 0 0 2 2 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_nhwc_wrw, precision + v + "  --input  64  64 17 17 --weights 192  64 1 7 --pads_strides_dilations 0 3 1 1 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_nhwc_wrw, precision + v + "  --input  64  64 17 17 --weights 192  64 7 1 --pads_strides_dilations 3 0 1 1 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_nhwc_wrw, precision + v + "  --input   4 128 28 28 --weights 128 128 2 2 --pads_strides_dilations 0 0 2 2 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_nhwc_wrw, precision + v + "  --input  32 128  8  8 --weights 192 128 3 1 --pads_strides_dilations 1 0 1 1 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_nhwc_wrw, precision + v + "  --input  64 192 17 17 --weights 160 192 3 3 --pads_strides_dilations 0 0 2 2 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_nhwc_wrw, precision + v + "  --input  64  32 73 73 --weights  64  32 3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_nhwc_wrw, precision + v + "  --input  16  64 56 56 --weights  64  64 3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_nhwc_wrw, precision + v + "  --input  16  16 25 25 --weights  64  16 3 3 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_nhwc_wrw, precision + v + "  --input  4 32 79 141 --weights 64 32 5 10 --pads_strides_dilations 0 0 2 2 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_nhwc_wrw, precision + v + "  --input  400  256 7 7 --weights 1024  256  7 7 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_nhwc_wrw, precision + v + "  --input  400  256 1 1 --weights 1024  256  1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_nhwc_wrw, precision + v + "  --input  1 3 32 32 --weights 1 3 11 11 --pads_strides_dilations 1 1 2 2 2 1" + dis_fwd + dis_bk_data},
    TestCase{env_nhwc_wrw, precision + v + "  --input  1 3 224 224 --weights 1 3 3 3 --pads_strides_dilations 0 0 1 1 2 2" + dis_fwd + dis_bk_data},
    TestCase{env_nhwc_wrw, precision + v + "  --input  1 1 8 8 --weights 1 1 2 2 --pads_strides_dilations 0 0 1 1 2 2" + dis_fwd + dis_bk_data},
    TestCase{env_nhwc_wrw, precision + v + "  --input  1 128 56 56 --weights 1 128 5 5 --pads_strides_dilations 0 0 2 2 1 1" + dis_fwd + dis_bk_data},
    TestCase{env_nhwc_wrw, precision + v + "  --input  2 64 19 19 --weights 510 64 3 3 --pads_strides_dilations 1 1 1 1 1 1" + dis_fwd + dis_bk_data}
        // clang-format on
    };
    return test_cases;
}

INSTANTIATE_TEST_SUITE_P(ConvIgemmDynamic, Conv2dFloat, testing::Values(GetTestCases("--float")));

INSTANTIATE_TEST_SUITE_P(ConvIgemmDynamic, Conv2dHalf, testing::Values(GetTestCases("--half")));
