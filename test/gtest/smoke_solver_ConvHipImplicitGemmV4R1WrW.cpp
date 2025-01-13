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
    const auto env_wrw =
        std::tuple{std::pair{MIOPEN_FIND_ENFORCE, "SEARCH_DB_UPDATE"},
                   std::pair{MIOPEN_DEBUG_TUNING_ITERATIONS_MAX, 5},
                   std::pair{MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL, 0},
                   std::pair{MIOPEN_FIND_MODE, "normal"},
                   std::pair{MIOPEN_DEBUG_FIND_ONLY_SOLVER, "ConvHipImplicitGemmV4R1WrW"}};

    const std::string vw = " --verbose --disable-forward --disable-backward-data";

    return std::vector{
        // clang-format off
    std::pair{env_wrw, vw + " --input 64 64 55 55 --weights 64 64 1 1 --pads_strides_dilations 0 0 1 1 1 1"}
        // clang-format on
    };
}

using TestCase = decltype(GetTestCases())::value_type;

bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::gfx103X>;
    using d_mask = disabled<Gpu::Default>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

} // namespace

class GPU_Conv2dTuningAltGemmV4R1WrW_FP32 : public FloatTestCase<std::vector<TestCase>>
{
};

class GPU_Conv2dTuningAltGemmV4R1WrW_FP16 : public HalfTestCase<std::vector<TestCase>>
{
};

class GPU_Conv2dTuningAltGemmV4R1WrW_BFP16 : public Bf16TestCase<std::vector<TestCase>>
{
};

TEST_P(GPU_Conv2dTuningAltGemmV4R1WrW_FP32, FloatTest_smoke_solver_ConvHipImplicitGemmV4R1WrW)
{
    if(IsTestSupportedForDevice())
    {
        invoke_with_params<conv2d_driver, GPU_Conv2dTuningAltGemmV4R1WrW_FP32>(tuning_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(GPU_Conv2dTuningAltGemmV4R1WrW_FP16, HalfTest_smoke_solver_ConvHipImplicitGemmV4R1WrW)
{
    if(IsTestSupportedForDevice())
    {
        invoke_with_params<conv2d_driver, GPU_Conv2dTuningAltGemmV4R1WrW_FP16>(tuning_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(GPU_Conv2dTuningAltGemmV4R1WrW_BFP16, Bf16Test_smoke_solver_ConvHipImplicitGemmV4R1WrW)
{
    if(IsTestSupportedForDevice())
    {
        invoke_with_params<conv2d_driver, GPU_Conv2dTuningAltGemmV4R1WrW_BFP16>(tuning_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Conv2dTuningAltGemmV4R1WrW_FP32,
                         testing::Values(GetTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Conv2dTuningAltGemmV4R1WrW_FP16,
                         testing::Values(GetTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Conv2dTuningAltGemmV4R1WrW_BFP16,
                         testing::Values(GetTestCases()));
