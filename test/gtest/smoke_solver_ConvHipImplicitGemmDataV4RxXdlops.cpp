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

class Conv2dBFloat16 : public testing::TestWithParam<std::vector<TestCase>>
{
};

void Run2dDriver(miopenDataType_t prec)
{

    std::vector<TestCase> params;
    switch(prec)
    {
    case miopenHalf: params = Conv2dHalf::GetParam(); break;
    case miopenBFloat16: params = Conv2dBFloat16::GetParam(); break;
    case miopenFloat:
    case miopenInt8:
    case miopenInt32:
    case miopenDouble:
    case miopenFloat8:
    case miopenBFloat8:
        FAIL() << "miopenFloat, miopenInt8, miopenInt32, "
                  "miopenDouble, miopenFloat8, miopenBFloat8 "
                  "data type not supported by "
                  "smoke_solver_ConvHipImplicitGemmDataV4RxXdlops test";

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
        // TEST_TUNING - the test should fail if output contains "Error" or "failed".
        EXPECT_FALSE(capture.find("Error") != std::string::npos ||
                     capture.find("failed") != std::string::npos);
        std::cout << capture;
    }
};

bool IsTestSupportedForDevice(const miopen::Handle& handle)
{
    std::string devName = handle.GetDeviceName();
    if(devName == "gfx908" || devName == "gfx90a")
        return true;
    else
        return false;
}

TEST_P(Conv2dHalf, HalfTest)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle))
    {
        Run2dDriver(miopenHalf);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(Conv2dBFloat16, BFloat16Test)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle))
    {
        Run2dDriver(miopenBFloat16);
    }
    else
    {
        GTEST_SKIP();
    }
};

std::vector<TestCase> GetTestCases(void)
{
    std::vector<std::string> env_fwd = {
        "MIOPEN_FIND_ENFORCE=SEARCH_DB_UPDATE",
        "MIOPEN_DEBUG_TUNING_ITERATIONS_MAX=5",
        "MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL=0",
        "MIOPEN_FIND_MODE=normal",
        "MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvHipImplicitGemmForwardV4R4Xdlops"};

    std::vector<std::string> env_fwd_padded = {
        "MIOPEN_FIND_ENFORCE=SEARCH_DB_UPDATE",
        "MIOPEN_DEBUG_TUNING_ITERATIONS_MAX=5",
        "MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL=0",
        "MIOPEN_FIND_MODE=normal",
        "MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvHipImplicitGemmForwardV4R4Xdlops_Padded_Gemm"};

    std::vector<std::string> env_fwd_v4r5 = {
        "MIOPEN_FIND_ENFORCE=SEARCH_DB_UPDATE",
        "MIOPEN_DEBUG_TUNING_ITERATIONS_MAX=5",
        "MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL=0",
        "MIOPEN_FIND_MODE=normal",
        "MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvHipImplicitGemmForwardV4R5Xdlops"};

    std::vector<std::string> env_bwd = {
        "MIOPEN_FIND_ENFORCE=SEARCH_DB_UPDATE",
        "MIOPEN_DEBUG_TUNING_ITERATIONS_MAX=5",
        "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1_XDLOPS=1",
        "MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL=0",
        "MIOPEN_FIND_MODE=normal",
        "MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvHipImplicitGemmBwdDataV4R1Xdlops"};

    std::vector<std::string> env_wrw = {
        "MIOPEN_FIND_ENFORCE=SEARCH_DB_UPDATE",
        "MIOPEN_DEBUG_TUNING_ITERATIONS_MAX=5",
        "MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL=0",
        "MIOPEN_FIND_MODE=normal",
        "MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvHipImplicitGemmWrwV4R4Xdlops"};

    std::vector<std::string> env_wrw_padded = {
        "MIOPEN_FIND_ENFORCE=SEARCH_DB_UPDATE",
        "MIOPEN_DEBUG_TUNING_ITERATIONS_MAX=5",
        "MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL=0",
        "MIOPEN_FIND_MODE=normal",
        "MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvHipImplicitGemmWrwV4R4Xdlops_Padded_Gemm"};

    std::string vf = " --verbose --disable-backward-data --disable-backward-weights";
    std::string vb = " --verbose --disable-forward --disable-backward-weights";
    std::string vw = " --verbose --disable-forward --disable-backward-data";

    const std::vector<TestCase> test_cases = {
        // clang-format off
    TestCase{env_fwd, vf + " --input 128 48 13 13 --weights 192 48 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    TestCase{env_bwd, vb + " --input 64 64 55 55 --weights 64 64 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    TestCase{env_wrw, vw + " --input 1 192 28 28 --weights 16 192 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    TestCase{env_fwd, vf + " --input 16 1 7 7 --weights 1 1 3 3 --pads_strides_dilations 0 0 1 1 1 1"},
    TestCase{env_wrw_padded, vw + " --input 256 2 5 5 --weights 1 2 3 3 --pads_strides_dilations 1 1 2 2 1 1"},
    TestCase{env_fwd_v4r5, vf + " --input 128 16 54 54 --weights 64 16 3 3 --pads_strides_dilations 1 1 1 1 1 1"}
        // clang-format on
    };
    return test_cases;
}

INSTANTIATE_TEST_SUITE_P(SmokeSolverConvHipImplicitGemmDataV4RxXdlops,
                         Conv2dHalf,
                         testing::Values(GetTestCases()));

INSTANTIATE_TEST_SUITE_P(SmokeSolverConvHipImplicitGemmDataV4RxXdlops,
                         Conv2dBFloat16,
                         testing::Values(GetTestCases()));
