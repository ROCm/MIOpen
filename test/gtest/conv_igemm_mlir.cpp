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
#include <miopen/env.hpp>
#include "../conv2d.hpp"
#include "get_handle.hpp"

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_MLIR)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace conv_igemm_mlir {

using TestCase = std::tuple<std::vector<std::string>, std::string>;

std::string GetFloatArg()
{
    const auto& tmp = miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG));
    if(tmp.empty())
    {
        return "";
    }
    return tmp;
};

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

class ConvIgemmMlirConfigFloat : public testing::TestWithParam<std::vector<TestCase>>
{
};
class ConvIgemmMlirConfigHalf : public testing::TestWithParam<std::vector<TestCase>>
{
};
class ConvIgemmMlirConfigInt8 : public testing::TestWithParam<std::vector<TestCase>>
{
};

void Run2dDriver(miopenDataType_t prec)
{

    std::vector<TestCase> params;
    switch(prec)
    {
    case miopenHalf: params = ConvIgemmMlirConfigHalf::GetParam(); break;
    case miopenInt8: params = ConvIgemmMlirConfigInt8::GetParam(); break;
    case miopenFloat: params = ConvIgemmMlirConfigFloat::GetParam(); break;
    case miopenBFloat16:
    case miopenInt32:
    case miopenFloat8:
    case miopenBFloat8:
    case miopenDouble:
        MIOPEN_THROW(miopenStatusBadParm,
                     "miopenBFloat16, miopenInt32, miopenFloat8, miopenBFloat8, "
                     "miopenDouble data type not supported by conv_igemm_mlir test");

    default: params = ConvIgemmMlirConfigFloat::GetParam();
    }

    for(const auto& test_value : params)
    {
        std::vector<std::string> tokens;
        GetArgs(test_value, tokens);
        std::vector<const char*> ptrs;

        std::transform(tokens.begin(), tokens.end(), std::back_inserter(ptrs), [](const auto& str) {
            return str.data();
        });

        testing::internal::CaptureStderr();
        test_drive<conv2d_driver>(ptrs.size(), ptrs.data());
        auto capture = testing::internal::GetCapturedStderr();
        EXPECT_FALSE(capture.find("Perf Db: record not found") != std::string::npos);
    }
};

TEST_P(ConvIgemmMlirConfigFloat, FloatTest)
{
#if MIOPEN_USE_MLIR

    const auto& handle = get_handle();
    if((miopen::StartsWith(handle.GetDeviceName(), "gfx103") ||
        miopen::StartsWith(handle.GetDeviceName(), "gfx906")) &&
       miopen::IsEnabled(ENV(MIOPEN_TEST_MLIR)) && miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) &&
       GetFloatArg() == "--float")
    {
        Run2dDriver(miopenFloat);
    }
    else
    {
        GTEST_SKIP();
    }

#else
    GTEST_SKIP();
#endif
};

TEST_P(ConvIgemmMlirConfigHalf, HalfTest)
{
#if MIOPEN_USE_MLIR

    const auto& handle = get_handle();
    if((miopen::StartsWith(handle.GetDeviceName(), "gfx103") ||
        miopen::StartsWith(handle.GetDeviceName(), "gfx906")) &&
       miopen::IsEnabled(ENV(MIOPEN_TEST_MLIR)) && miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) &&
       GetFloatArg() == "--half")
    {
        Run2dDriver(miopenHalf);
    }
    else
    {
        GTEST_SKIP();
    }

#else
    GTEST_SKIP();
#endif
};

TEST_P(ConvIgemmMlirConfigInt8, Int8Test)
{
#if MIOPEN_USE_MLIR

    const auto& handle = get_handle();
    if((miopen::StartsWith(handle.GetDeviceName(), "gfx103") ||
        miopen::StartsWith(handle.GetDeviceName(), "gfx906")) &&
       miopen::IsEnabled(ENV(MIOPEN_TEST_MLIR)) && miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) &&
       GetFloatArg() == "--int8")
    {
        Run2dDriver(miopenInt8);
    }
    else
    {
        GTEST_SKIP();
    }

#else
    GTEST_SKIP();
#endif
};

std::vector<TestCase> GetTestCases(const std::string& precision)
{
    std::vector<std::string> igemm_fwd = {"MIOPEN_FIND_MODE=normal",
                                          "MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmFwd"};
    std::string flags_fwd    = " --verbose --disable-backward-data --disable-backward-weights";
    std::string layout       = " --in_layout NHWC --fil_layout NHWC --out_layout NHWC";
    std::string groupCount_4 = " --group-count 4";

    // FWD test cases for precision == "--int8"
    std::vector<TestCase> test_cases = {
        // clang-format off
    TestCase{igemm_fwd, precision + flags_fwd + " --input 256 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1"},
    TestCase{igemm_fwd, precision + flags_fwd + " --input 256 128  28 28 --weights 128  128  3 3 --pads_strides_dilations 1 1 1 1 1 1"},
    TestCase{igemm_fwd, precision + flags_fwd + " --input 256 128  28 28 --weights 128  128  3 3 --pads_strides_dilations 1 1 1 1 1 1" + layout},
    TestCase{igemm_fwd, precision + flags_fwd + " --input 128 512  7  7  --weights 512  512  3 3 --pads_strides_dilations 1 1 1 1 1 1"},
    TestCase{igemm_fwd, precision + flags_fwd + " --input 128 512  7  7  --weights 512  512  3 3 --pads_strides_dilations 1 1 1 1 1 1" + layout},
    TestCase{igemm_fwd, precision + flags_fwd + " --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    TestCase{igemm_fwd, precision + flags_fwd + " --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1" + layout},
    TestCase{igemm_fwd, precision + flags_fwd + " --input 256 256  56 56 --weights 256  64   1 1 --pads_strides_dilations 0 0 1 1 1 1" + groupCount_4}
        // clang-format on
    };

    std::vector<std::string> igemm_bwd = {"MIOPEN_FIND_MODE=normal",
                                          "MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmBwd"};
    std::vector<std::string> igemm_wrw = {"MIOPEN_FIND_MODE=normal",
                                          "MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmWrW"};

    std::string flags_bwd     = " --verbose --disable-forward --disable-backward-weights";
    std::string flags_wrw     = " --verbose --disable-forward --disable-backward-data";
    std::string groupCount_32 = " --group-count 32";

    // BWD WRW test cases
    const std::vector<TestCase> test_cases_bwd_wrw = {
        // clang-format off
    TestCase{igemm_bwd, precision + flags_bwd + " --input 256 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1"},
    TestCase{igemm_bwd, precision + flags_bwd + " --input 256 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1" + layout},
    TestCase{igemm_bwd, precision + flags_bwd + " --input 256 128  28 28 --weights 128  128  3 3 --pads_strides_dilations 1 1 1 1 1 1"},
    TestCase{igemm_bwd, precision + flags_bwd + " --input 256 128  28 28 --weights 128  128  3 3 --pads_strides_dilations 1 1 1 1 1 1" + layout},
    TestCase{igemm_bwd, precision + flags_bwd + " --input 128 512  7  7  --weights 512  512  3 3 --pads_strides_dilations 1 1 1 1 1 1"},
    TestCase{igemm_bwd, precision + flags_bwd + " --input 128 512  7  7  --weights 512  512  3 3 --pads_strides_dilations 1 1 1 1 1 1" + layout},
    TestCase{igemm_bwd, precision + flags_bwd + " --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    TestCase{igemm_bwd, precision + flags_bwd + " --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1" + layout},

    TestCase{igemm_wrw, precision + flags_wrw + " --input 64  1024 14 14 --weights 256  1024 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    TestCase{igemm_wrw, precision + flags_wrw + " --input 64  1024 14 14 --weights 256  1024 1 1 --pads_strides_dilations 0 0 1 1 1 1" + layout},
    TestCase{igemm_wrw, precision + flags_wrw + " --input 256 256  14 14 --weights 256  256  3 3 --pads_strides_dilations 0 0 2 2 1 1"},
    TestCase{igemm_wrw, precision + flags_wrw + " --input 256 256  14 14 --weights 256  256  3 3 --pads_strides_dilations 0 0 2 2 1 1" + layout},
    TestCase{igemm_wrw, precision + flags_wrw + " --input 128 2048 7  7  --weights 512  2048 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    TestCase{igemm_wrw, precision + flags_wrw + " --input 128 2048 7  7  --weights 512  2048 1 1 --pads_strides_dilations 0 0 1 1 1 1" + layout},
    TestCase{igemm_wrw, precision + flags_wrw + " --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1" + layout},
    TestCase{igemm_wrw, precision + flags_wrw + " --input 256 1024 14 14 --weights 1024 32   1 1 --pads_strides_dilations 0 0 1 1 1 1" + groupCount_32}
        // clang-format on
    };

    // FWD BWD WRW cases in test_cases
    if(precision == "--float" || precision == "--half")
    {
        test_cases.reserve(test_cases_bwd_wrw.size());
        test_cases.insert(test_cases.end(), test_cases_bwd_wrw.begin(), test_cases_bwd_wrw.end());
    }

    return test_cases;
}
// Float for FWD, BWD, WRW
INSTANTIATE_TEST_SUITE_P(ConvIgemmMlir, ConvIgemmMlirConfigFloat, testing::Values(GetTestCases("--float")));
// Half for FWD, BWD, WRW
INSTANTIATE_TEST_SUITE_P(ConvIgemmMlir, ConvIgemmMlirConfigHalf, testing::Values(GetTestCases("--half")));
// Int8 for FWD
INSTANTIATE_TEST_SUITE_P(ConvIgemmMlir, ConvIgemmMlirConfigInt8, testing::Values(GetTestCases("--int8")));

} //namespace conv_igemm_mlir 
