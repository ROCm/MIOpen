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

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_GPU_XNACK_ENABLED)

namespace {

auto GetTestCases()
{
    const auto env_w1 = std::tuple{
        std::pair{ENV(MIOPEN_FIND_ENFORCE), "SEARCH_DB_UPDATE"},
        std::pair{ENV(MIOPEN_DEBUG_TUNING_ITERATIONS_MAX), 5},
        std::pair{ENV(MIOPEN_FIND_MODE), "normal"},
        std::pair{ENV(MIOPEN_DEBUG_FIND_ONLY_SOLVER), "ConvAsmBwdWrW1x1"}};

    const std::string vw = " --verbose --disable-forward --disable-backward-data";

    return std::vector{
        // clang-format off
    std::pair{env_w1, vw + " --input 1 4 5 5 --weights 4 4 1 1 --pads_strides_dilations 0 0 2 2 1 1"}
        // clang-format on
    };
}

using TestCase = decltype(GetTestCases())::value_type;

bool SkipTest() { return miopen::IsEnabled(ENV(MIOPEN_TEST_GPU_XNACK_ENABLED)); }

bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::Default>;
    using d_mask = disabled<Gpu::Default>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

} // namespace

class Conv2dTuningFloat : public FloatTestCase<std::vector<TestCase>>
{
};

class Conv2dTuningHalf : public HalfTestCase<std::vector<TestCase>>
{
};

class Conv2dTuningBf16 : public Bf16TestCase<std::vector<TestCase>>
{
};

TEST_P(Conv2dTuningFloat, FloatTest_smoke_solver_convasmbwdwrw)
{
    if(IsTestSupportedForDevice() && !SkipTest())
    {
        invoke_with_params<conv2d_driver, Conv2dTuningFloat>(tuning_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(Conv2dTuningHalf, HalfTest_smoke_solver_convasmbwdwrw)
{
    if(IsTestSupportedForDevice() && !SkipTest())
    {
        invoke_with_params<conv2d_driver, Conv2dTuningHalf>(tuning_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(Conv2dTuningBf16, Bf16Test_smoke_solver_convasmbwdwrw)
{
    if(IsTestSupportedForDevice() && !SkipTest())
    {
        invoke_with_params<conv2d_driver, Conv2dTuningBf16>(tuning_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(SmokeSolverConvAsmBwd, Conv2dTuningFloat, testing::Values(GetTestCases()));
INSTANTIATE_TEST_SUITE_P(SmokeSolverConvAsmBwd, Conv2dTuningHalf, testing::Values(GetTestCases()));
INSTANTIATE_TEST_SUITE_P(SmokeSolverConvAsmBwd, Conv2dTuningBf16, testing::Values(GetTestCases()));
