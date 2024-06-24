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
    const auto env_bwd = std::tuple{
        std::pair{MIOPEN_FIND_MODE, "normal"},
        std::pair{MIOPEN_DEBUG_FIND_ONLY_SOLVER, "ConvDirectNaiveConvBwd"},
        std::pair{MIOPEN_DRIVER_USE_GPU_REFERENCE, false},
    };

    const auto env_wrw = std::tuple{
        std::pair{MIOPEN_FIND_MODE, "normal"},
        std::pair{MIOPEN_DEBUG_FIND_ONLY_SOLVER, "ConvDirectNaiveConvWrw"},
        std::pair{MIOPEN_DRIVER_USE_GPU_REFERENCE, false},
    };

    const std::string vb = " --verbose --disable-forward --disable-backward-weights";
    const std::string vw = " --verbose --disable-forward --disable-backward-data";

    return std::vector{
        // clang-format off
    std::pair{env_bwd, vb + " --input 1 16 14 14 --weights 48 16 5 5 --pads_strides_dilations 2 2 1 1 1 1"},
    std::pair{env_wrw, vw + " --input 1 16 14 14 --weights 48 16 5 5 --pads_strides_dilations 2 2 1 1 1 1"}
        // clang-format on
    };
}

using TestCase = decltype(GetTestCases())::value_type;

bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx103X, Gpu::gfx110X>;
    using d_mask = disabled<Gpu::Default>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

} // namespace

class Conv2dNoGpuRefFloat : public FloatTestCase<std::vector<TestCase>>
{
};

class Conv2dNoGpuRefHalf : public HalfTestCase<std::vector<TestCase>>
{
};

class Conv2dNoGpuRefBf16 : public Bf16TestCase<std::vector<TestCase>>
{
};

TEST_P(Conv2dNoGpuRefFloat, FloatTest_smoke_solver_ConvDirectNaiveConv_Bw)
{
    if(IsTestSupportedForDevice())
    {
        invoke_with_params<conv2d_driver, Conv2dNoGpuRefFloat>(default_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(Conv2dNoGpuRefHalf, HalfTest_smoke_solver_ConvDirectNaiveConv_Bw)
{
    if(IsTestSupportedForDevice())
    {
        invoke_with_params<conv2d_driver, Conv2dNoGpuRefHalf>(default_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(Conv2dNoGpuRefBf16, Bf16Test_smoke_solver_ConvDirectNaiveConv_Bw)
{
    if(IsTestSupportedForDevice())
    {
        invoke_with_params<conv2d_driver, Conv2dNoGpuRefBf16>(default_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(SmokeSolverConvDirectNaiveConvBw,
                         Conv2dNoGpuRefFloat,
                         testing::Values(GetTestCases()));
INSTANTIATE_TEST_SUITE_P(SmokeSolverConvDirectNaiveConvBw,
                         Conv2dNoGpuRefHalf,
                         testing::Values(GetTestCases()));
INSTANTIATE_TEST_SUITE_P(SmokeSolverConvDirectNaiveConvBw,
                         Conv2dNoGpuRefBf16,
                         testing::Values(GetTestCases()));
