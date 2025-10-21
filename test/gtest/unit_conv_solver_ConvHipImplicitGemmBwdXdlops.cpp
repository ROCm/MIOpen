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

#define WORKAROUND_SWDEV_522871 1

#if WORKAROUND_SWDEV_522871
#define SOLVER_NAME_DEV_APP DISABLED_ConvHipImplicitGemmBwdXdlops
#else
#define SOLVER_NAME_DEV_APP ConvHipImplicitGemmBwdXdlops
#endif

namespace {

// Non-deterministic test cases (for GPU smoke tests)
auto GetConvSmokeTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{datatype, miopenTensorNHWC, {1, 32, 8, 8}},
                 {datatype, miopenTensorNHWC, {32, 32, 1, 1}},
                 datatype, {{0, 0}, {1, 1}, {1, 1}}},
        // clang-format on
    };
}

// Deterministic test case (for CPU deterministic applicability test)
auto GetConvDeterministicTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{datatype, miopenTensorNHWC, {1, 32, 8, 8}},
                 {datatype, miopenTensorNHWC, {32, 32, 1, 1}},
                 datatype, {{0, 0}, {1, 1}, {1, 1}, 1, true}},
        // clang-format on
    };
}

auto GetConvFullTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{datatype, miopenTensorNHWC, {1, 32, 8, 8}},
                 {datatype, miopenTensorNHWC, {32, 32, 3, 3}},
                 datatype, {{1, 1}, {1, 1}, {1, 1}}}, // non-zero padding
        TestCase{{datatype, miopenTensorNHWC, {1, 64, 24, 48}},
                 {datatype, miopenTensorNHWC, {96, 64, 1, 1}},
                 datatype, {{0, 0}, {2, 2}, {1, 1}}}, // stride > 1
        TestCase{{datatype, miopenTensorNHWC, {1, 32, 8, 8}},
                 {datatype, miopenTensorNHWC, {32, 32, 3, 3}},
                 datatype, {{0, 0}, {1, 1}, {3, 3}}}, // dilation > 1
        TestCase{{datatype, miopenTensorNHWC, {1, 64, 24, 48}},
                 {datatype, miopenTensorNHWC, {96, 64, 1, 1}},
                 datatype, {{0, 0}, {1, 1}, {1, 1}}}, // some different NCHW and k parameters
        // clang-format on
    };
}

auto GetTestParams(miopenDataType_t datatype)
{
// If MIOpen is built without CK these tests will fail, skip them to avoid failing
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    Gpu supportedDevices = Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950;
#else
    Gpu supportedDevices = Gpu::None;
#endif
    auto params = miopen::unit_tests::UnitTestConvSolverParams(supportedDevices);
    params.Tunable(5);
    if(datatype == miopenHalf)
    {
        // Enable the backward solver on MI200 for fp16 by disabling the alternate implementation
        params.SetConvAttrFp16Alt(0);
    }

    return params;
}

} // namespace

using GPU_UnitTestConvSolverImplicitGemmBwdXdlops_FP16  = GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverImplicitGemmBwdXdlops_BFP16 = GPU_UnitTestConvSolverBwd_BFP16;
using GPU_UnitTestConvSolverImplicitGemmBwdXdlops_FP32  = GPU_UnitTestConvSolverBwd_FP32;
using CPU_UnitTestConvSolverImplicitGemmBwdXdlopsDevApplicability_FP16 =
    CPU_UnitTestConvSolverDevApplicabilityBwd_NONE;
using CPU_UnitTestConvSolverImplicitGemmBwdXdlopsDeterministicApplicability_NONE =
    CPU_UnitTestConvSolverDevApplicabilityBwd_NONE;

TEST_P(GPU_UnitTestConvSolverImplicitGemmBwdXdlops_FP16, ConvHipImplicitGemmBwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmBwdXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmBwdXdlops_BFP16, ConvHipImplicitGemmBwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmBwdXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmBwdXdlops_FP32, ConvHipImplicitGemmBwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmBwdXdlops{});
};

TEST_P(CPU_UnitTestConvSolverImplicitGemmBwdXdlopsDevApplicability_FP16, SOLVER_NAME_DEV_APP)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmBwdXdlops{});
};

TEST_P(CPU_UnitTestConvSolverImplicitGemmBwdXdlopsDeterministicApplicability_NONE,
       ConvHipImplicitGemmBwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmBwdXdlops{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverImplicitGemmBwdXdlops_FP16,
                         testing::Combine(testing::Values(GetTestParams(miopenHalf)),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvSmokeTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverImplicitGemmBwdXdlops_BFP16,
    testing::Combine(testing::Values(GetTestParams(miopenBFloat16)),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetConvSmokeTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverImplicitGemmBwdXdlops_FP32,
                         testing::Combine(testing::Values(GetTestParams(miopenFloat)),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvSmokeTestCases(miopenFloat))));

// Full tests
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverImplicitGemmBwdXdlops_FP16,
                         testing::Combine(testing::Values(GetTestParams(miopenHalf)),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvFullTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverImplicitGemmBwdXdlops_BFP16,
                         testing::Combine(testing::Values(GetTestParams(miopenBFloat16)),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvFullTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverImplicitGemmBwdXdlops_FP32,
                         testing::Combine(testing::Values(GetTestParams(miopenFloat)),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvFullTestCases(miopenFloat))));

// Device applicability tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverImplicitGemmBwdXdlopsDevApplicability_FP16,
                         testing::Combine(testing::Values(GetTestParams(miopenHalf)),
                                          testing::Values(GetConvSmokeTestCases(miopenHalf)[0])));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_UnitTestConvSolverImplicitGemmBwdXdlopsDeterministicApplicability_NONE,
    testing::Combine(testing::Values(Gpu::None),
                     testing::Values(GetConvDeterministicTestCases(miopenHalf)[0])));
