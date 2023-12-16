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

auto GetTestCases()
{
    // MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=1 is necessary due to WORKAROUND_iGemm_936 in
    // Jenkinsfile, which disables ConvHipImplicitGemmV4R1Fwd, but we still want to check that the
    // solver is not broken.
    const auto env_fwd = std::tuple{
        std::pair{ENV(MIOPEN_FIND_ENFORCE), std::string_view("SEARCH_DB_UPDATE")},
        std::pair{ENV(MIOPEN_DEBUG_TUNING_ITERATIONS_MAX), std::string_view("5")},
        std::pair{ENV(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1), std::string_view("1")},
        std::pair{ENV(MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL), std::string_view("0")},
        std::pair{ENV(MIOPEN_FIND_MODE), std::string_view("normal")},
        std::pair{ENV(MIOPEN_DEBUG_FIND_ONLY_SOLVER),
                  std::string_view("ConvHipImplicitGemmV4R1Fwd")}};

    const std::string vf = " --verbose --disable-backward-data --disable-backward-weights";

    return std::vector{
        // clang-format off
    std::pair{env_fwd, vf + " --input 256 32 27 27 --weights 128 32 1 1 --pads_strides_dilations 0 0 1 1 1 1"}
        // clang-format on
    };
}

using TestCase = decltype(GetTestCases())::value_type;

class Conv2dFloat : public FloatTestCase<std::vector<TestCase>>
{
};

bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::gfx103X>;
    using d_mask = disabled<Gpu::Default>;
    return IsTestSupportedForDevice<d_mask, e_mask>();
}

TEST_P(Conv2dFloat, FloatTest)
{
    if(IsTestSupportedForDevice())
    {
        invoke_with_params<conv2d_driver, Conv2dFloat>(tuning_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(SmokeSolverConvHipImplicitGemmV4R1FwdFp32,
                         Conv2dFloat,
                         testing::Values(GetTestCases()));
