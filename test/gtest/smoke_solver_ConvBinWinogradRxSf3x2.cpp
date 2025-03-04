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
    const auto env3x2 =
        std::tuple{std::pair{MIOPEN_FIND_ENFORCE, "SEARCH_DB_UPDATE"},
                   std::pair{MIOPEN_DEBUG_TUNING_ITERATIONS_MAX, 5},
                   std::pair{MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL, 0},
                   std::pair{MIOPEN_FIND_MODE, "normal"},
                   std::pair{MIOPEN_DEBUG_FIND_ONLY_SOLVER, "ConvBinWinogradRxSf3x2"}};

    return std::vector{
        // clang-format off
    std::pair{env3x2,std::string(" --input 1 40 20 20 --weights 20 40 3 3 --pads_strides_dilations 1 1 1 1 1 1")}
        // clang-format on
    };
}

using TestCase = decltype(GetTestCases())::value_type;

bool SkipTest() { return get_handle_xnack(); }

bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx103X>;
    using d_mask = disabled<Gpu::gfx900>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

} // namespace

class GPU_Conv2dTuningAltBinWinogradRxSf3x2_FP32 : public FloatTestCase<std::vector<TestCase>>
{
};

class GPU_Conv2dTuningAltBinWinogradRxSf3x2_FP16 : public HalfTestCase<std::vector<TestCase>>
{
};

TEST_P(GPU_Conv2dTuningAltBinWinogradRxSf3x2_FP32, FloatTest_smoke_solver_ConvBinWinogradRxSf3x2)
{
    if(IsTestSupportedForDevice() && !SkipTest())
    {
        invoke_with_params<conv2d_driver, GPU_Conv2dTuningAltBinWinogradRxSf3x2_FP32>(
            default_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(GPU_Conv2dTuningAltBinWinogradRxSf3x2_FP16, HalfTest_smoke_solver_ConvBinWinogradRxSf3x2)
{
    if(IsTestSupportedForDevice() && !SkipTest())
    {
        invoke_with_params<conv2d_driver, GPU_Conv2dTuningAltBinWinogradRxSf3x2_FP16>(
            default_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Conv2dTuningAltBinWinogradRxSf3x2_FP32,
                         testing::Values(GetTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Conv2dTuningAltBinWinogradRxSf3x2_FP16,
                         testing::Values(GetTestCases()));
