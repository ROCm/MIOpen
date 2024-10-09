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
    const auto env_fwd =
        std::tuple{std::pair{MIOPEN_FIND_MODE, "normal"},
                   std::pair{MIOPEN_DEBUG_FIND_ONLY_SOLVER, "ConvOclDirectFwd11x11"}};

    const auto env_fwd_gen =
        std::tuple{std::pair{MIOPEN_FIND_MODE, "normal"},
                   std::pair{MIOPEN_DEBUG_FIND_ONLY_SOLVER, "ConvOclDirectFwdGen"}};

    const auto env_wrw = std::tuple{std::pair{MIOPEN_FIND_MODE, "normal"},
                                    std::pair{MIOPEN_DEBUG_FIND_ONLY_SOLVER, "ConvOclBwdWrW53"}};

    const std::string vf = " --verbose --disable-backward-data --disable-backward-weights";
    const std::string vw = " --verbose --disable-forward --disable-backward-data";

    return std::vector{
        // clang-format off
    std::pair{env_fwd,     vf + " --input 1 1 44 44 --weights 1 1 11 11 --pads_strides_dilations 0 0 4 4 1 1"},
    std::pair{env_fwd_gen, vf + " --input 1 1 6 6 --weights 1 1 3 3 --pads_strides_dilations 0 0 2 2 1 1"},
    std::pair{env_wrw,     vw + " --input 16 1 7 7 --weights 1 1 3 3 --pads_strides_dilations 0 0 1 1 1 1"}
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

class GPU_Conv2dDefault_FP32 : public FloatTestCase<std::vector<TestCase>>
{
};

class GPU_Conv2dDefault_FP16 : public HalfTestCase<std::vector<TestCase>>
{
};

class GPU_Conv2dDefault_BFP16 : public Bf16TestCase<std::vector<TestCase>>
{
};

TEST_P(GPU_Conv2dDefault_FP32, FloatTest_smoke_solver_ConvOcl_Fwd11x11_FwdGen_WrW53)
{
    if(IsTestSupportedForDevice())
    {
        invoke_with_params<conv2d_driver, GPU_Conv2dDefault_FP32>(default_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(GPU_Conv2dDefault_FP16, HalftTest_smoke_solver_ConvOcl_Fwd11x11_FwdGen_WrW53)
{
    if(IsTestSupportedForDevice())
    {
        invoke_with_params<conv2d_driver, GPU_Conv2dDefault_FP16>(default_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(GPU_Conv2dDefault_BFP16, Bf16Test_smoke_solver_ConvOcl_Fwd11x11_FwdGen_WrW53)
{
    if(IsTestSupportedForDevice())
    {
        invoke_with_params<conv2d_driver, GPU_Conv2dDefault_BFP16>(default_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Conv2dDefault_FP32, testing::Values(GetTestCases()));
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Conv2dDefault_FP16, testing::Values(GetTestCases()));
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Conv2dDefault_BFP16, testing::Values(GetTestCases()));
