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

class Conv2dHalf : public testing::TestWithParam<std::vector<TestCase>>
{
};

void Run2dDriver(miopenDataType_t prec)
{

    std::vector<TestCase> params;
    switch(prec)
    {
    case miopenHalf: params = Conv2dHalf::GetParam(); break;
    case miopenFloat:
    case miopenInt8:
    case miopenBFloat16:
    case miopenInt32:
    case miopenDouble:
    case miopenFloat8:
    case miopenBFloat8:
        FAIL() << "miopenFloat, miopenInt8, miopenBFloat16, miopenInt32, "
                  "miopenDouble, miopenFloat8, miopenBFloat8 "
                  "data type not supported by conv_igemm_dynamic_dlops test";

    default: params = Conv2dHalf::GetParam();
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
    if(miopen::StartsWith(devName, "GFX103"))
        return true;
    else
        return false;
}

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

    std::vector<std::string> env_fwd = {
        "MIOPEN_FIND_MODE=normal",
        "MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvAsmImplicitGemmGTCDynamicFwdDlopsNCHWC"};

    std::string v           = " --verbose";
    std::string dis_bk_data = " --disable-backward-data";
    std::string dis_bk_wei  = " --disable-backward-weights";
    std::string in_nchw     = " --in_layout NCHW";
    std::string fil_nchw    = " --fil_layout NCHW";
    std::string fil_chwn    = " --fil_layout CHWN";
    std::string out_nchw    = " --out_layout NCHW";
    std::string tensor      = " --tensor_vect 1";
    std::string vlen4       = " --vector_length 4";
    std::string vlen8       = " --vector_length 8";

    std::string common_base = " --cmode convfp16" + dis_bk_data + dis_bk_wei + in_nchw;

    std::string nchwc_nchwc_base       = common_base + fil_nchw + out_nchw + tensor;
    std::string nchwc_nchwc_fwd_fp16x4 = nchwc_nchwc_base + vlen4;
    std::string nchwc_nchwc_fwd_fp16x8 = nchwc_nchwc_base + vlen8;

    std::string nchwc_chwnc_base       = common_base + fil_chwn + out_nchw + tensor;
    std::string nchwc_chwnc_fwd_fp16x4 = nchwc_chwnc_base + vlen4;
    std::string nchwc_chwnc_fwd_fp16x8 = nchwc_chwnc_base + vlen8;

    const std::vector<TestCase> test_cases = {
        // clang-format off
    //nchwc_nchwc_fwd_fp16x4
    TestCase{env_fwd, precision + v + " --input  1 8  10  10  --weights 8 8 3 3     --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  32 160 73 73 --weights  64 160 1 1 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  16  64 56 56 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input   2 256 40 52 --weights 256 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input   2  64 32 28 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  32 128 14 14 --weights  64 128 1 1 --pads_strides_dilations 0 0 2 2 1 1" + nchwc_nchwc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  64  64 17 17 --weights 192  64 1 7 --pads_strides_dilations 0 3 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  64  64 17 17 --weights 192  64 7 1 --pads_strides_dilations 3 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input   4 128 28 28 --weights 128 128 2 2 --pads_strides_dilations 0 0 2 2 1 1" + nchwc_nchwc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  32 128  8  8 --weights 192 128 3 1 --pads_strides_dilations 1 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  64 192 17 17 --weights 160 192 3 3 --pads_strides_dilations 0 0 2 2 1 1" + nchwc_nchwc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  64  32 73 73 --weights  64  32 3 3 --pads_strides_dilations 1 1 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  16  64 56 56 --weights  64  64 3 3 --pads_strides_dilations 1 1 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  16  16 25 25 --weights  64  16 3 3 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  4  32 79 141 --weights 64  32 5 10 --pads_strides_dilations 0 0 2 2 1 1" + nchwc_nchwc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  400  256 7 7 --weights 1024 256 7 7 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  400  256 1 1 --weights 1024 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x4},

    //nchwc_chwnc_fwd_fp16x4
    TestCase{env_fwd, precision + v + " --input  64 256  7  7 --weights 256 3 3  128  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  32 160 73 73 --weights 160 1 1   64  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  16  64 56 56 --weights  64 1 1   64  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input   2 256 40 52 --weights 256 1 1  256  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input   2  64 32 28 --weights  64 1 1   64  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  32 128 14 14 --weights 128 1 1   64  --pads_strides_dilations 0 0 2 2 1 1" + nchwc_chwnc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  64  64 17 17 --weights  64 3 7  192  --pads_strides_dilations 0 3 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  64  64 17 17 --weights  64 7 1  192  --pads_strides_dilations 3 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input   4 128 28 28 --weights 128 2 2  128  --pads_strides_dilations 0 0 2 2 1 1" + nchwc_chwnc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  32 128  8  8 --weights 128 3 1  192  --pads_strides_dilations 1 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  64 192 17 17 --weights 192 3 3  160  --pads_strides_dilations 0 0 2 2 1 1" + nchwc_chwnc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  64  32 73 73 --weights  32 3 3   64  --pads_strides_dilations 1 1 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  16  64 56 56 --weights  64 3 3   64  --pads_strides_dilations 1 1 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  16  16 25 25 --weights  16 3 3   64  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  4  32 79 141 --weights  32 5 10  64  --pads_strides_dilations 0 0 2 2 1 1" + nchwc_chwnc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  400  256 7 7 --weights 256 7 7 1024  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},
    TestCase{env_fwd, precision + v + " --input  400  256 1 1 --weights 256 1 1 1024  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x4},

    //nchwc_nchwc_fwd_fp16x8
    TestCase{env_fwd, precision + v + " --input  1 8  10  10  --weights 8 8 3 3     --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  32 160 73 73 --weights  64 160 1 1 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  16  64 56 56 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input   2 256 40 52 --weights 256 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input   2  64 32 28 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  32 128 14 14 --weights  64 128 1 1 --pads_strides_dilations 0 0 2 2 1 1" + nchwc_nchwc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  64  64 17 17 --weights 192  64 1 7 --pads_strides_dilations 0 3 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  64  64 17 17 --weights 192  64 7 1 --pads_strides_dilations 3 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input   4 128 28 28 --weights 128 128 2 2 --pads_strides_dilations 0 0 2 2 1 1" + nchwc_nchwc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  32 128  8  8 --weights 192 128 3 1 --pads_strides_dilations 1 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  64 192 17 17 --weights 160 192 3 3 --pads_strides_dilations 0 0 2 2 1 1" + nchwc_nchwc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  64  32 73 73 --weights  64  32 3 3 --pads_strides_dilations 1 1 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  16  64 56 56 --weights  64  64 3 3 --pads_strides_dilations 1 1 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  16  16 25 25 --weights  64  16 3 3 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  4  32 79 141 --weights 64  32 5 10 --pads_strides_dilations 0 0 2 2 1 1" + nchwc_nchwc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  400  256 7 7 --weights 1024 256 7 7 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  400  256 1 1 --weights 1024 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_nchwc_fwd_fp16x8},

    //nchwc_chwnc_fwd_fp16x8
    TestCase{env_fwd, precision + v + " --input  64 256  7  7 --weights 256 1 1  128  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  32 160 73 73 --weights 160 1 1   64  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  16  64 56 56 --weights  64 1 1   64  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input   2 256 40 52 --weights 256 1 1  256  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input   2  64 32 28 --weights  64 1 1   64  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  32 128 14 14 --weights 128 1 1   64  --pads_strides_dilations 0 0 2 2 1 1" + nchwc_chwnc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  64  64 17 17 --weights  64 1 7  192  --pads_strides_dilations 0 3 1 1 1 1" + nchwc_chwnc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  64  64 17 17 --weights  64 7 1  192  --pads_strides_dilations 3 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input   4 128 28 28 --weights 128 2 2  128  --pads_strides_dilations 0 0 2 2 1 1" + nchwc_chwnc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  32 128  8  8 --weights 128 3 1  192  --pads_strides_dilations 1 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  64 192 17 17 --weights 192 3 3  160  --pads_strides_dilations 0 0 2 2 1 1" + nchwc_chwnc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  64  32 73 73 --weights  32 3 3   64  --pads_strides_dilations 1 1 1 1 1 1" + nchwc_chwnc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  16  64 56 56 --weights  64 3 3   64  --pads_strides_dilations 1 1 1 1 1 1" + nchwc_chwnc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  16  16 25 25 --weights  16 3 3   64  --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  4  32 79 141 --weights 32 5 10  64   --pads_strides_dilations 0 0 2 2 1 1" + nchwc_chwnc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  400  256 7 7 --weights  256 7 7 1024 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x8},
    TestCase{env_fwd, precision + v + " --input  400  256 1 1 --weights  256 1 1 1024 --pads_strides_dilations 0 0 1 1 1 1" + nchwc_chwnc_fwd_fp16x8}
        // clang-format on
    };
    return test_cases;
}

INSTANTIATE_TEST_SUITE_P(ConvIgemmDynamicDlopsFwd,
                         Conv2dHalf,
                         testing::Values(GetTestCases("--half")));
