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

namespace {

auto GetTestCases()
{
    const auto env_1uv2 = std::tuple{
        std::pair{MIOPEN_ENV(MIOPEN_FIND_ENFORCE), std::string_view("SEARCH_DB_UPDATE")},
        std::pair{MIOPEN_ENV(MIOPEN_DEBUG_TUNING_ITERATIONS_MAX), std::string_view("5")},
        std::pair{MIOPEN_ENV(MIOPEN_FIND_MODE), std::string_view("normal")},
        std::pair{MIOPEN_ENV(MIOPEN_DEBUG_FIND_ONLY_SOLVER), std::string_view("ConvAsm1x1UV2")}};

    const auto env_3u = std::tuple{
        std::pair{MIOPEN_ENV(MIOPEN_FIND_ENFORCE), std::string_view("SEARCH_DB_UPDATE")},
        std::pair{MIOPEN_ENV(MIOPEN_DEBUG_TUNING_ITERATIONS_MAX), std::string_view("5")},
        std::pair{MIOPEN_ENV(MIOPEN_FIND_MODE), std::string_view("normal")},
        std::pair{MIOPEN_ENV(MIOPEN_DEBUG_FIND_ONLY_SOLVER), std::string_view("ConvAsm3x3U")}};

    const std::string vf = " --verbose --disable-backward-data --disable-backward-weights";
    const std::string vb = " --verbose --disable-forward --disable-backward-weights";

    return std::vector{
        // clang-format off
    std::pair{env_1uv2, vf + " --input 1 4 2 2 --weights 4 4 1 1 --pads_strides_dilations 0 0 2 2 1 1"},
    std::pair{env_1uv2, vb + " --input 1 4 2 2 --weights 4 4 1 1 --pads_strides_dilations 0 0 2 2 1 1"},
    std::pair{env_3u, vf + " --input 1 4 10 10 --weights 4 4 3 3 --pads_strides_dilations 1 1 1 1 1 1"},
    std::pair{env_3u, vb + " --input 1 4 10 10 --weights 4 4 3 3 --pads_strides_dilations 1 1 1 1 1 1"},
        // clang-format on
    };
}

using TestCase = decltype(GetTestCases())::value_type;

bool SkipTest() { return get_handle_xnack(); }

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

TEST_P(Conv2dTuningFloat, FloatTest_smoke_solver_convasm)
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

INSTANTIATE_TEST_SUITE_P(SmokeSolverConvAsm, Conv2dTuningFloat, testing::Values(GetTestCases()));
