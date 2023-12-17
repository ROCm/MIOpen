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

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_MLIR)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace conv_igemm_mlir_xdlops_fwd {

auto GetTestCases()
{
    const auto fwd = std::tuple{
        std::pair{ENV(MIOPEN_FIND_MODE), std::string_view("normal")},
        std::pair{ENV(MIOPEN_DEBUG_FIND_ONLY_SOLVER), std::string_view("ConvMlirIgemmFwdXdlops")}};

    const std::string flags_fwd = " --verbose --disable-backward-data --disable-backward-weights";
    const std::string layout    = " --in_layout NHWC --fil_layout NHWC --out_layout NHWC";
    const std::string groupCount_4 = " --group-count 4";

    // FWD test cases for precision == "--int8"
    return std::vector{
        // clang-format off
    std::pair{fwd, flags_fwd + " --input 256 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1"},
    std::pair{fwd, flags_fwd + " --input 256 128  28 28 --weights 128  128  3 3 --pads_strides_dilations 1 1 1 1 1 1"},
    std::pair{fwd, flags_fwd + " --input 256 128  28 28 --weights 128  128  3 3 --pads_strides_dilations 1 1 1 1 1 1" + layout},
    std::pair{fwd, flags_fwd + " --input 128 512  7  7  --weights 512  512  3 3 --pads_strides_dilations 1 1 1 1 1 1"},
    std::pair{fwd, flags_fwd + " --input 128 512  7  7  --weights 512  512  3 3 --pads_strides_dilations 1 1 1 1 1 1" + layout},
    std::pair{fwd, flags_fwd + " --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    std::pair{fwd, flags_fwd + " --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1" + layout},
    std::pair{fwd, flags_fwd + " --input 256 256  56 56 --weights 256  64   1 1 --pads_strides_dilations 0 0 1 1 1 1" + groupCount_4}
        // clang-format on
    };
}

using TestCase = decltype(GetTestCases())::value_type;

bool SkipTest()
{
    return !(miopen::IsEnabled(ENV(MIOPEN_TEST_MLIR))) || miopen::IsDisabled(ENV(MIOPEN_TEST_ALL));
}

class Conv2dHalf : public FloatTestCase<std::vector<TestCase>>
{
};

class Conv2dInt8 : public Int8TestCase<std::vector<TestCase>>
{
};

bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::Default>;
    using d_mask = disabled<Gpu::gfx900, Gpu::gfx906>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

} // namespace conv_igemm_mlir_xdlops_fwd
using namespace conv_igemm_mlir_xdlops_fwd;

TEST_P(Conv2dHalf, HalfTest_conv_igemm_mlir_xdlops_fwd)
{
    if(IsTestSupportedForDevice() && !SkipTest())
    {
        invoke_with_params<conv2d_driver, Conv2dHalf>(db_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(Conv2dInt8, Int8Test_conv_igemm_mlir_xdlops_fwd)
{
    if(IsTestSupportedForDevice() && !SkipTest())
    {
        invoke_with_params<conv2d_driver, Conv2dInt8>(db_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

// Half for FWD, BWD, WRW
INSTANTIATE_TEST_SUITE_P(ConvIgemmMlirXdlops, Conv2dHalf, testing::Values(GetTestCases()));
// Int8 for FWD
INSTANTIATE_TEST_SUITE_P(ConvIgemmMlirXdlops, Conv2dInt8, testing::Values(GetTestCases()));
