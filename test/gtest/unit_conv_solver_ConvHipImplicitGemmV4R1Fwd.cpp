/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include "unit_conv_solver.hpp"

#ifndef HIP_PACKAGE_VERSION_FLAT
#error "HIP_PACKAGE_VERSION_FLAT undefined"
#endif

/// [iGemmfwd][test_conv2d][gfx906][half] Verification failed
/// https://github.com/ROCm/MIOpen/issues/936
/// WORKAROUND_ISSUE_936

/// Mismatch in ConvHipImplicitGemmV4R1Fwd
/// https://github.com/ROCm/MIOpen/issues/2038
/// WORKAROUND_ISSUE_2038

#define SOLVER_NAME ConvHipImplicitGemmV4R1Fwd

namespace {

auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{256, 32, 27, 27}, {128, 32, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        // clang-format on
    };
}

template <miopenDataType_t datatype>
const auto& GetTestParams()
{
    static const auto params = [] {
        Gpu supported_gpus = Gpu::gfx900 | Gpu::gfx906 | Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx103X;
        if constexpr(datatype != miopenFloat)
        {
            supported_gpus = supported_gpus | Gpu::gfx94X | Gpu::gfx950;
        }
        auto p = miopen::unit_tests::UnitTestConvSolverParams(supported_gpus);
        p.EnableDeprecatedSolvers();
        p.Tunable(5);
        p.SetConvAttrFp16Alt(0);
        /// \todo 250.0f is too much. The solver needs to be checked.
        p.SetTolerance(Gpu::gfx908, miopenHalf, 250.0f);
        p.SetTolerance(Gpu::gfx90A, miopenHalf, 250.0f);
        p.SetTolerance(Gpu::gfx94X, miopenHalf, 250.0f);
        p.SetTolerance(Gpu::gfx950, miopenHalf, 250.0f);
        p.SetTolerance(Gpu::gfx908, miopenBFloat16, 30.0f);
        p.SetTolerance(Gpu::gfx90A, miopenBFloat16, 30.0f);
        p.SetTolerance(Gpu::gfx950, miopenBFloat16, 30.0f);
        return p;
    }();
    return params;
}

const auto& GetTestParamsFP16() { return GetTestParams<miopenHalf>(); }
const auto& GetTestParamsBFP16() { return GetTestParams<miopenBFloat16>(); }
const auto& GetTestParamsFP32() { return GetTestParams<miopenFloat>(); }

} // namespace

using GPU_UnitTestConvSolverHipImplicitGemmV4R1Fwd_FP16  = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverHipImplicitGemmV4R1Fwd_BFP16 = GPU_UnitTestConvSolverFwd_BFP16;
using GPU_UnitTestConvSolverHipImplicitGemmV4R1Fwd_FP32  = GPU_UnitTestConvSolverFwd_FP32;
using CPU_UnitTestConvSolverHipImplicitGemmV4R1FwdDevApplicabilityFwd_FP16 =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;
using CPU_UnitTestConvSolverHipImplicitGemmV4R1FwdDevApplicabilityFwd_FP32 =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverHipImplicitGemmV4R1Fwd_FP16, SOLVER_NAME)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmV4R1Fwd{});
};

TEST_P(GPU_UnitTestConvSolverHipImplicitGemmV4R1Fwd_BFP16, SOLVER_NAME)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmV4R1Fwd{});
};

TEST_P(GPU_UnitTestConvSolverHipImplicitGemmV4R1Fwd_FP32, SOLVER_NAME)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmV4R1Fwd{});
};

TEST_P(CPU_UnitTestConvSolverHipImplicitGemmV4R1FwdDevApplicabilityFwd_FP16, SOLVER_NAME)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmV4R1Fwd{});
};

TEST_P(CPU_UnitTestConvSolverHipImplicitGemmV4R1FwdDevApplicabilityFwd_FP32, SOLVER_NAME)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmV4R1Fwd{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipImplicitGemmV4R1Fwd_FP16,
                         testing::Combine(testing::Values(GetTestParamsFP16()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipImplicitGemmV4R1Fwd_BFP16,
                         testing::Combine(testing::Values(GetTestParamsBFP16()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipImplicitGemmV4R1Fwd_FP32,
                         testing::Combine(testing::Values(GetTestParamsFP32()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverHipImplicitGemmV4R1FwdDevApplicabilityFwd_FP16,
                         testing::Combine(testing::Values(GetTestParamsFP16()),
                                          testing::Values(GetConvTestCases(miopenHalf)[0])));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverHipImplicitGemmV4R1FwdDevApplicabilityFwd_FP32,
                         testing::Combine(testing::Values(GetTestParamsFP32()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));
