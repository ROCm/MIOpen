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
    const auto env = std::tuple{std::pair{MIOPEN_FIND_MODE, "normal"},
                                std::pair{MIOPEN_DEBUG_FIND_ONLY_SOLVER, "ConvWinoFuryRxS<2-3>"}};

    const std::string vf = " --verbose --disable-backward-data --disable-backward-weights";
    const std::string vb = " --verbose --disable-forward --disable-backward-weights";

    return std::vector{
        // clang-format off
    std::pair{env, vf + " --input 1 16 16 16 --weights 16 16 3 3 --pads_strides_dilations 1 1 1 1 1 1"},
    std::pair{env, vb + " --input 1 16 16 16 --weights 16 16 3 3 --pads_strides_dilations 1 1 1 1 1 1"}
        // clang-format on
    };
}

using TestCase = decltype(GetTestCases())::value_type;

bool SkipTest() { return false; }

bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::gfx110X>;
    using d_mask = disabled<Gpu::gfx900, Gpu::gfx906, Gpu::gfx908, Gpu::gfx90A>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

} // namespace

class Conv2dDefaultHalf : public HalfTestCase<std::vector<TestCase>>
{
};

TEST_P(Conv2dDefaultHalf, HalfTest_smoke_solver_ConvWinoFuryRxSf2x3)
{
    if(IsTestSupportedForDevice() && !SkipTest())
    {
        invoke_with_params<conv2d_driver, Conv2dDefaultHalf>(default_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(SmokeSolverConvWinoFuryRxSf2x3,
                         Conv2dDefaultHalf,
                         testing::Values(GetTestCases()));
