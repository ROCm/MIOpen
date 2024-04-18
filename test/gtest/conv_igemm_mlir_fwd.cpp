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

namespace {

auto GetTestCases()
{
    const auto igemm_fwd = std::tuple{std::pair{MIOPEN_FIND_MODE, "normal"},
                                      std::pair{MIOPEN_DEBUG_FIND_ONLY_SOLVER, "ConvMlirIgemmFwd"}};

    const std::string vf     = " --verbose --disable-backward-data --disable-backward-weights";
    const std::string layout = " --in_layout NHWC --fil_layout NHWC --out_layout NHWC";
    const std::string groupCount_4 = " --group-count 4";

    // FWD test cases for precision == "--int8"
    return std::vector{
        // clang-format off
    std::pair{igemm_fwd, vf + " --input 256 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1"},
    std::pair{igemm_fwd, vf + " --input 256 128  28 28 --weights 128  128  3 3 --pads_strides_dilations 1 1 1 1 1 1"},
    std::pair{igemm_fwd, vf + " --input 256 128  28 28 --weights 128  128  3 3 --pads_strides_dilations 1 1 1 1 1 1" + layout},
    std::pair{igemm_fwd, vf + " --input 128 512  7  7  --weights 512  512  3 3 --pads_strides_dilations 1 1 1 1 1 1"},
    std::pair{igemm_fwd, vf + " --input 128 512  7  7  --weights 512  512  3 3 --pads_strides_dilations 1 1 1 1 1 1" + layout},
    std::pair{igemm_fwd, vf + " --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    std::pair{igemm_fwd, vf + " --input 128 64   56 56 --weights 64   64   1 1 --pads_strides_dilations 0 0 1 1 1 1" + layout},
    std::pair{igemm_fwd, vf + " --input 256 256  56 56 --weights 256  64   1 1 --pads_strides_dilations 0 0 1 1 1 1" + groupCount_4}
        // clang-format on
    };
}

using TestCase = decltype(GetTestCases())::value_type;

bool SkipTest() { return !(env::enabled(MIOPEN_TEST_MLIR)) || env::disabled(MIOPEN_TEST_ALL); }

bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::gfx103X>;
    using d_mask = disabled<Gpu::gfx900, Gpu::gfx908, Gpu::gfx90A>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

} // namespace

class Conv2dDefaultFloat : public FloatTestCase<std::vector<TestCase>>
{
};
class Conv2dDefaultHalf : public HalfTestCase<std::vector<TestCase>>
{
};
class Conv2dDefaultInt8 : public Int8TestCase<std::vector<TestCase>>
{
};

TEST_P(Conv2dDefaultFloat, FloatTest_conv_igemm_mlir_fwd)
{
    if(IsTestSupportedForDevice() && !SkipTest())
    {
        invoke_with_params<conv2d_driver, Conv2dDefaultFloat>(db_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(Conv2dDefaultHalf, HalfTest_conv_igemm_mlir_fwd)
{
    if(IsTestSupportedForDevice() && !SkipTest())
    {
        invoke_with_params<conv2d_driver, Conv2dDefaultHalf>(db_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(Conv2dDefaultInt8, Int8Test_conv_igemm_mlir_fwd)
{
    if(IsTestSupportedForDevice() && !SkipTest())
    {
        invoke_with_params<conv2d_driver, Conv2dDefaultInt8>(db_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

// Float for FWD, BWD, WRW
INSTANTIATE_TEST_SUITE_P(ConvIgemmMlir, Conv2dDefaultFloat, testing::Values(GetTestCases()));
// Half for FWD, BWD, WRW
INSTANTIATE_TEST_SUITE_P(ConvIgemmMlir, Conv2dDefaultHalf, testing::Values(GetTestCases()));
// Int8 for FWD
INSTANTIATE_TEST_SUITE_P(ConvIgemmMlir, Conv2dDefaultInt8, testing::Values(GetTestCases()));
