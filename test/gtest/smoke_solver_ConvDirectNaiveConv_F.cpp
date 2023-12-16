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

auto GetTestCases()
{
    const auto env = std::tuple{
        std::pair{ENV(MIOPEN_FIND_MODE), std::string_view("normal")},
        std::pair{ENV(MIOPEN_DEBUG_FIND_ONLY_SOLVER), std::string_view("ConvDirectNaiveConvFwd")},
        std::pair{ENV(MIOPEN_DRIVER_USE_GPU_REFERENCE), std::string_view("0")},
    };

    const std::string vf = " --verbose --disable-backward-data --disable-backward-weights";

    return std::vector{
        // clang-format off
    std::pair{env, vf + " --input 1 16 14 14 --weights 48 16 5 5 --pads_strides_dilations 2 2 1 1 1 1"}
        // clang-format on
    };
}

using TestCase = decltype(GetTestCases())::value_type;

class Conv2dFloat : public FloatTestCase<std::vector<TestCase>>
{
};

class Conv2dHalf : public HalfTestCase<std::vector<TestCase>>
{
};

class Conv2dBf16 : public Bf16TestCase<std::vector<TestCase>>
{
};

class Conv2dInt8 : public Int8TestCase<std::vector<TestCase>>
{
};

bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx103X, Gpu::gfx110X>;
    using d_mask = disabled<Gpu::Default>;
    return IsTestSupportedForDevice<d_mask, e_mask>();
}

TEST_P(Conv2dFloat, FloatTest)
{
    if(IsTestSupportedForDevice())
    {
        invoke_with_params<conv2d_driver, Conv2dFloat>(default_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(Conv2dHalf, HalftTest)
{
    if(IsTestSupportedForDevice())
    {
        invoke_with_params<conv2d_driver, Conv2dHalf>(default_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(Conv2dBf16, Bf16Test)
{
    if(IsTestSupportedForDevice())
    {
        invoke_with_params<conv2d_driver, Conv2dBf16>(default_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(Conv2dInt8, Int8Test)
{
    if(IsTestSupportedForDevice())
    {
        invoke_with_params<conv2d_driver, Conv2dInt8>(default_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(SmokeSolverConvDirectNaiveConvF,
                         Conv2dFloat,
                         testing::Values(GetTestCases()));
INSTANTIATE_TEST_SUITE_P(SmokeSolverConvDirectNaiveConvF,
                         Conv2dHalf,
                         testing::Values(GetTestCases()));
INSTANTIATE_TEST_SUITE_P(SmokeSolverConvDirectNaiveConvF,
                         Conv2dBf16,
                         testing::Values(GetTestCases()));
INSTANTIATE_TEST_SUITE_P(SmokeSolverConvDirectNaiveConvF,
                         Conv2dInt8,
                         testing::Values(GetTestCases()));
