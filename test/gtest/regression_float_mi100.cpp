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
#include "get_handle.hpp"

#include "../conv2d.hpp"

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace regression_float_mi100 {

auto GetTestCases()
{
    // Regression test for SWDEV-305815 (issue 1206)
    const auto env = std::tuple{std::pair{MIOPEN_DEBUG_CONV_WINOGRAD, false},
                                std::pair{MIOPEN_DEBUG_CONV_FFT, false},
                                std::pair{MIOPEN_DEBUG_CONV_DIRECT, false},
                                std::pair{MIOPEN_DEBUG_CONV_GEMM, false},
                                std::pair{MIOPEN_DEBUG_CONV_IMPLICIT_GEMM, false},
                                std::pair{MIOPEN_LOG_LEVEL, 1}};

    const std::string v          = " --verbose";
    const std::string dis_fwd    = " --disable-forward";
    const std::string dis_bk_wei = " --disable-backward-weights";

    return std::vector{
        // clang-format off
    std::pair{env, v + " --input 32 256 38 38 --weights 256 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_fwd + dis_bk_wei}
        // clang-format on
    };
}

using TestCase = decltype(GetTestCases())::value_type;

bool SkipTest() { return env::disabled(MIOPEN_TEST_ALL); }

class GPU_Conv2d_regression_mi100_FP32 : public FloatTestCase<std::vector<TestCase>>
{
};

bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::Default>;
    using d_mask = disabled<Gpu::gfx900, Gpu::gfx906, Gpu::gfx90A>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

} // namespace regression_float_mi100
using namespace regression_float_mi100;

TEST_P(GPU_Conv2d_regression_mi100_FP32, FloatTest)
{
    if(IsTestSupportedForDevice() && !SkipTest())
    {
        invoke_with_params<conv2d_driver, GPU_Conv2d_regression_mi100_FP32>(default_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Conv2d_regression_mi100_FP32, testing::Values(GetTestCases()));
