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

#include "../conv2d.hpp"

namespace {

auto GetTestCases()
{
    const auto env5x10f = std::tuple{std::pair{MIOPEN_FIND_MODE, "normal"},
                                     std::pair{MIOPEN_DEBUG_FIND_ONLY_SOLVER, "ConvAsm5x10u2v2f1"}};
    const auto env5x10b = std::tuple{std::pair{MIOPEN_FIND_MODE, "normal"},
                                     std::pair{MIOPEN_DEBUG_FIND_ONLY_SOLVER, "ConvAsm5x10u2v2b1"}};
    const auto env7x7 =
        std::tuple{std::pair{MIOPEN_FIND_MODE, "normal"},
                   std::pair{MIOPEN_DEBUG_FIND_ONLY_SOLVER, "ConvAsm7x7c3h224w224k64u2v2p3q3f1"}};

    const std::string vf = " --verbose --disable-backward-data --disable-backward-weights";
    const std::string vb = " --verbose --disable-forward --disable-backward-weights";

    return std::vector{
        // clang-format off
    std::pair{env5x10f, vf + " --input 1 1 5 10 --weights 16 1 5 10 --pads_strides_dilations 0 0 2 2 1 1"},
    std::pair{env5x10b, vb + " --input 1 1 16 160 --weights 16 16 5 10 --pads_strides_dilations 0 0 2 2 1 1"},
    std::pair{env7x7,   vf + " --input 1 3 224 224 --weights 64 3 7  7 --pads_strides_dilations 3 3 2 2 1 1"}
        // clang-format on
    };
}

using TestCase = decltype(GetTestCases())::value_type;

bool SkipTest() { return get_handle_xnack(); }

bool IsTestSupportedForDevice()
{
    // GFX90A_DISABLED is because of WORKAROUND_ISSUE_1146
    using e_mask = enabled<Gpu::Default>;
    using d_mask = disabled<Gpu::gfx90A>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

} // namespace

class GPU_Conv2dDefault_FP32 : public FloatTestCase<std::vector<TestCase>>
{
};

TEST_P(GPU_Conv2dDefault_FP32, FloatTest_smoke_solver_ConvAsm_5x10_7x7)
{
    if(IsTestSupportedForDevice() && !SkipTest())
    {
        invoke_with_params<conv2d_driver, GPU_Conv2dDefault_FP32>(default_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Conv2dDefault_FP32, testing::Values(GetTestCases()));
