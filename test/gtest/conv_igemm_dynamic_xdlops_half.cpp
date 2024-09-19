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
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_FIND_MODE)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEBUG_FIND_ONLY_SOLVER)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)

namespace conv_igemm_dynamic_xdlops_half {

static bool SkipTest(const std::string& float_arg)
{
    if(!MIOPEN_TEST_ALL)
        return false;
    if(env::enabled(MIOPEN_TEST_ALL))
        if(env::value(MIOPEN_TEST_FLOAT_ARG) == float_arg)
            return false;
    return true;
}

void SetupEnvVar()
{
    env::update(MIOPEN_FIND_MODE, "normal");
    env::update(MIOPEN_DEBUG_FIND_ONLY_SOLVER,
                "ConvAsmImplicitGemmGTCDynamicFwdXdlops;ConvAsmImplicitGemmGTCDynamicWrwXdlops");
}

void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

class GPU_Conv2dHalf_conv_igemm_dynamic_xdlops_FP16
    : public testing::TestWithParam<std::vector<std::string>>
{
};

void Run2dDriver(miopenDataType_t prec)
{

    std::vector<std::string> params;
    switch(prec)
    {
    case miopenHalf: params = GPU_Conv2dHalf_conv_igemm_dynamic_xdlops_FP16::GetParam(); break;
    case miopenFloat:
    case miopenInt8:
    case miopenBFloat16:
    case miopenInt32:
    case miopenInt64:
    case miopenDouble:
    case miopenFloat8:
    case miopenBFloat8:
        FAIL() << "miopenFloat, miopenInt8, miopenBFloat16, miopenInt32, "
                  "miopenDouble, miopenFloat8, miopenBFloat8 "
                  "data type not supported by conv_igemm_dynamic_xdlops_half test";

    default: params = GPU_Conv2dHalf_conv_igemm_dynamic_xdlops_FP16::GetParam();
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

    const std::vector<std::string> test_cases = {
        // clang-format off
    //fwd
    {flags + " --input 64 3 224 224 --weights 64 3 7 7 --pads_strides_dilations 3 3 2 2 1 1" + dis_bk_data + dis_bk_wei},
    {flags + " --input 64 3 230 230 --weights 64 3 7 7 --pads_strides_dilations 0 0 2 2 1 1" + dis_bk_data + dis_bk_wei},

    //wrw
    {flags + " --input  1 3 32 32 --weights 1 3 11 11 --pads_strides_dilations 1 1 2 2 2 1" + dis_fwd + dis_bk_data},
    {flags + " --input  1 3 224 224 --weights 1 3 3 3 --pads_strides_dilations 0 0 1 1 2 2" + dis_fwd + dis_bk_data},
    {flags + " --input  1 1 8 8 --weights 1 1 2 2 --pads_strides_dilations 0 0 1 1 2 2" + dis_fwd + dis_bk_data},
    {flags + " --input  1 128 56 56 --weights 1 128 5 5 --pads_strides_dilations 0 0 2 2 1 1" + dis_fwd + dis_bk_data}
        // clang-format on
    };
    return test_cases;
}

} // namespace conv_igemm_dynamic_xdlops_half
using namespace conv_igemm_dynamic_xdlops_half;

TEST_P(GPU_Conv2dHalf_conv_igemm_dynamic_xdlops_FP16, HalfTest_conv_igemm_dynamic_xdlops_half)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && !SkipTest("--half"))
    {
        Run2dDriver(miopenHalf);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Conv2dHalf_conv_igemm_dynamic_xdlops_FP16,
                         testing::Values(GetTestCases("--half")));
