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
#include "get_handle.hpp"
#include "test_env.hpp"

#include "../conv2d.hpp"

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_COMPOSABLEKERNEL)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace conv_hip_igemm_xdlops {

void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

class ConvHipIgemmXdlopsConfigInt8 : public testing::TestWithParam<std::vector<std::string>>
{
};

void Run2dDriver(miopenDataType_t prec)
{
    std::vector<std::string> params;
    switch(prec)
    {
    case miopenInt8: params = ConvHipIgemmXdlopsConfigInt8::GetParam(); break;
    case miopenFloat8:
    case miopenBFloat8:
    case miopenHalf:
    case miopenBFloat16:
    case miopenFloat:
    case miopenInt32:
    case miopenInt64:
    case miopenDouble:
        FAIL() << "miopenHalf, miopenBFloat16, miopenFloat, miopenInt32, "
                  "miopenDouble data "
                  "type not supported by "
                  "test_conv_hip_igemm_xdlops test";

    default: params = ConvHipIgemmXdlopsConfigInt8::GetParam();
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
        test_drive<conv2d_driver>(ptrs.size(), ptrs.data(), "test_conv2d");
        auto capture = testing::internal::GetCapturedStderr();
        std::cout << capture;
    }
};

bool IsTestSupportedForDevice(const miopen::Handle& handle)
{
    std::string devName = handle.GetDeviceName();
    if(devName == "gfx908" || devName == "gfx90a" || devName == "gfx94")
        return true;
    else
        return false;
}

std::vector<std::string> GetTestCases(const std::string& precision)
{
    std::string fwd         = " --disable-backward-data --disable-backward-weights --verbose";
    std::string bwd         = " --disable-forward --disable-backward-weights --verbose";
    std::string layout      = " --in_layout NHWC --fil_layout NHWC --out_layout NHWC";
    std::string output_int8 = " --output_type int8";
    std::string output_fp32 = " --output_type fp32";
    std::string output_fp16 = " --output_type fp16";
    std::string psd0        = " --pads_strides_dilations 0 0 1 1 1 1";
    std::string psd1        = " --pads_strides_dilations 1 1 1 1 1 1";

    std::vector<std::string> test_cases = {
        // clang-format off
    {precision + fwd + " --input 256 128  28 28 --weights 128  128  3 3" + output_int8 + layout + psd1},
    {precision + fwd + " --input 128 512  7  7  --weights 512  512  3 3" + output_int8 + layout + psd1},
    {precision + fwd + " --input 128 64   56 56 --weights 64   64   1 1" + output_int8 + layout + psd0},
    {precision + fwd + " --input 256 256  56 56 --weights 256  64   1 1" + output_int8 + layout + psd0},

    {precision + fwd + " --input 256 128  28 28 --weights 128  128  3 3" + output_fp32 + layout + psd1},
    {precision + fwd + " --input 128 512  7  7  --weights 512  512  3 3" + output_fp32 + layout + psd1},
    {precision + fwd + " --input 128 64   56 56 --weights 64   64   1 1" + output_fp32 + layout + psd0},
    {precision + fwd + " --input 256 256  56 56 --weights 256  64   1 1" + output_fp32 + layout + psd0},
    {precision + fwd + " --input 256 128  28 28 --weights 128  128  3 3" + output_fp16 + layout + psd1},
    {precision + fwd + " --input 128 512  7  7  --weights 512  512  3 3" + output_fp16 + layout + psd1},
    {precision + fwd + " --input 128 64   56 56 --weights 64   64   1 1" + output_fp16 + layout + psd0},
    {precision + fwd + " --input 256 256  56 56 --weights 256  64   1 1" + output_fp16 + layout + psd0},

    {precision + bwd + " --input 256 128  28 28 --weights 128  128  3 3" + output_fp32 + layout + psd1},
    {precision + bwd + " --input 128 512  7  7  --weights 512  512  3 3" + output_fp32 + layout + psd1},
    {precision + bwd + " --input 128 64   56 56 --weights 64   64   1 1" + output_fp32 + layout + psd0},
    {precision + bwd + " --input 256 256  56 56 --weights 256  64   1 1" + output_fp32 + layout + psd0},
    {precision + bwd + " --input 256 128  28 28 --weights 128  128  3 3" + output_fp16 + layout + psd1},
    {precision + bwd + " --input 128 512  7  7  --weights 512  512  3 3" + output_fp16 + layout + psd1},
    {precision + bwd + " --input 128 64   56 56 --weights 64   64   1 1" + output_fp16 + layout + psd0},
    {precision + bwd + " --input 256 256  56 56 --weights 256  64   1 1" + output_fp16 + layout + psd0}
        // clang-format on
    };

    return test_cases;
}

} // namespace conv_hip_igemm_xdlops
using namespace conv_hip_igemm_xdlops;

TEST_P(ConvHipIgemmXdlopsConfigInt8, Int8Test)
{
#if MIOPEN_BACKEND_OPENCL

    GTEST_SKIP() << "MIOPEN_BACKEND_HIP needed for this test";

#else // MIOPEN_BACKEND_HIP, OCL_DISABLED
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && env::enabled(MIOPEN_TEST_COMPOSABLEKERNEL) &&
       env::enabled(MIOPEN_TEST_ALL) && IsTestRunWith("--int8"))
    {
        Run2dDriver(miopenInt8);
    }
    else
    {
        GTEST_SKIP();
    }
#endif
};

INSTANTIATE_TEST_SUITE_P(ConvHipIgemmXdlops,
                         ConvHipIgemmXdlopsConfigInt8,
                         testing::Values(GetTestCases("--int8")));
