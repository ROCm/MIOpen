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

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_GPU_XNACK_ENABLED)
namespace smoke_solver_convasmbwdwrw3x3_fp16 {

using TestCase = std::tuple<std::vector<std::string>, std::string>;

static bool SkipTest(void) { return miopen::IsEnabled(ENV(MIOPEN_TEST_GPU_XNACK_ENABLED)); }

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

class SmokeSolverASMConv2dBwdHalf : public testing::TestWithParam<std::vector<TestCase>>
{
};

void Run2dDriver(miopenDataType_t prec)
{

    std::vector<TestCase> params;
    switch(prec)
    {
    case miopenHalf: params = SmokeSolverASMConv2dBwdHalf::GetParam(); break;
    case miopenFloat:
    case miopenBFloat16:
    case miopenInt8:
    case miopenInt32:
    case miopenDouble:
    case miopenFloat8:
    case miopenBFloat8:
        FAIL() << "miopenFloat, miopenBFloat16, miopenInt8, miopenInt32, "
                  "miopenDouble, miopenFloat8, miopenBFloat8 "
                  "data type not supported by smoke_solver_convasmbwdwrw3x3_fp16 test";

    default: params = SmokeSolverASMConv2dBwdHalf::GetParam();
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
    if(devName == "gfx900" || devName == "gfx906" || devName == "gfx908" || devName == "gfx90a")
        return true;
    else
        return false;
}

std::vector<TestCase> GetTestCases(void)
{
    std::vector<std::string> env_wrw = {"MIOPEN_FIND_ENFORCE=SEARCH_DB_UPDATE",
                                        "MIOPEN_DEBUG_TUNING_ITERATIONS_MAX=5",
                                        "MIOPEN_FIND_MODE=normal",
                                        "MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvAsmBwdWrW3x3"};

    std::string vw = " --verbose --disable-forward --disable-backward-data";

    const std::vector<TestCase> test_cases = {
        // clang-format off
    //smoke_solver_ConvAsmBwdWrW3x3_fp32   
    TestCase{env_wrw, vw + " --input 2 4 3 3 --weights 4 4 3 3 --pads_strides_dilations 1 1 1 1 1 1"}
        // clang-format on
    };
    return test_cases;
}

} // namespace smoke_solver_convasmbwdwrw3x3_fp16

using namespace smoke_solver_convasmbwdwrw3x3_fp16;

TEST_P(SmokeSolverASMConv2dBwdHalf, HalfTest_smoke_solver_convasmbwdwrw3x3_fp16)
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

INSTANTIATE_TEST_SUITE_P(SmokeSolverConvAsmBwdWrw3x3Fp16,
                         SmokeSolverASMConv2dBwdHalf,
                         testing::Values(GetTestCases()));
