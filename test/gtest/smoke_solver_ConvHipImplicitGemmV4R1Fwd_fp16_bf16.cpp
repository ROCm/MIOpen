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
    // MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=1 is necessary due to WORKAROUND_iGemm_936 in
    // Jenkinsfile, which disables ConvHipImplicitGemmV4R1Fwd, but we still want to check that the
    // solver is not broken. smoke_solver_ConvHipImplicitGemmV4R1Fwd is split to BF16+FP16 and
    // FP32 tests because of WORKAROUND_ISSUE_2038, which disables validation of FP16 and BF16
    // datatypes in this test, see
    // https://github.com/ROCm/MIOpen/pull/2043#issuecomment-1482657160
    const auto env_fwd =
        std::tuple{std::pair{ENV(MIOPEN_FIND_ENFORCE), "SEARCH_DB_UPDATE"},
                   std::pair{ENV(MIOPEN_DEBUG_TUNING_ITERATIONS_MAX), 5},
                   std::pair{ENV(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1), true},
                   std::pair{ENV(MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL), 0},
                   std::pair{ENV(MIOPEN_FIND_MODE), "normal"},
                   std::pair{ENV(MIOPEN_DEBUG_FIND_ONLY_SOLVER), "ConvHipImplicitGemmV4R1Fwd"}};

    const std::string vf = " --verbose --disable-backward-data --disable-backward-weights";

    const auto dis_vali = " --disable-validation";

    return std::vector{
        // clang-format off
    std::pair{env_fwd, vf + " --input 256 32 27 27 --weights 128 32 1 1 --pads_strides_dilations 0 0 1 1 1 1" + dis_vali}
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

class Conv2dTuningV4R1Half : public HalfTestCase<std::vector<TestCase>>
{
};

class Conv2dTuningV4R1Bf16 : public Bf16TestCase<std::vector<TestCase>>
{
};

TEST_P(Conv2dTuningV4R1Half, HalfTest_smoke_solver_ConvHipImplicitGemmV4R1Fwd_fp16_bf16)
{
    if(IsTestSupportedForDevice())
    {
        invoke_with_params<conv2d_driver, Conv2dTuningV4R1Half>(tuning_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(Conv2dTuningV4R1Bf16, Bf16Test_smoke_solver_ConvHipImplicitGemmV4R1Fwd_fp16_bf16)
{
    if(IsTestSupportedForDevice())
    {
        invoke_with_params<conv2d_driver, Conv2dTuningV4R1Bf16>(tuning_check);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(SmokeSolverConvHipImplicitGemmV4R1FwdFp16Bf16,
                         Conv2dTuningV4R1Half,
                         testing::Values(GetTestCases()));

INSTANTIATE_TEST_SUITE_P(SmokeSolverConvHipImplicitGemmV4R1FwdFp16Bf16,
                         Conv2dTuningV4R1Bf16,
                         testing::Values(GetTestCases()));
