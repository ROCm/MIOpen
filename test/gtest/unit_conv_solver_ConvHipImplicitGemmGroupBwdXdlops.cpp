// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "unit_conv_solver_group_xdlops.hpp"

namespace {

using TestCase = miopen::unit_tests::GroupXdlopsNumericData;

// Non-deterministic test cases (for GPU smoke tests)
auto GetConvSmokeTestCases()
{
    static std::vector<TestCase> test_cases = {
        // clang-format off
        TestCase{{1, 32, 8, 8}, {32, 32, 1, 1}, {0, 0}, {1, 1}, {1, 1}, 1}
        // clang-format on
    };

    return test_cases;
}

auto GetConvFullTestCases()
{
    static std::vector<TestCase> test_cases = {
        // clang-format off
        TestCase{{1, 32, 8, 8}, {32, 32, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 1}, // non-zero padding
        TestCase{{1, 64, 24, 48}, {96, 64, 1, 1}, {0, 0}, {2, 2}, {1, 1}, 1}, // stride > 1
        TestCase{{1, 32, 8, 8}, {32, 32, 3, 3}, {0, 0}, {1, 1}, {3, 3}, 1}, // dilation > 1
        TestCase{{1, 64, 24, 48}, {96, 64, 1, 1}, {0, 0}, {1, 1}, {1, 1}, 1},
        // Group count = 2 and 4
        TestCase{{1, 32, 8, 8}, {32, 16, 3, 3}, {0, 0}, {1, 1}, {3, 3}, 2}, // dilation > 1
        TestCase{{1, 64, 24, 48}, {96, 16, 1, 1}, {0, 0}, {1, 1}, {1, 1}, 4},
        // clang-format on
    };

    return test_cases;
}

auto GetDevApplicabilityConvCase()
{
    // For device applicability checks
    return GetConvTestForGroupXdlops<miopenHalf>(miopenTensorNHWC,
                                                 std::move(GetConvSmokeTestCases()[0]));
}

// Deterministic test case (for CPU deterministic applicability test)
auto GetDeterministicConvCase()
{
    TestCase test_case = {
        // clang-format off
        TestCase{{1, 32, 8, 8}, {32, 32, 1, 1}, {0, 0}, {1, 1}, {1, 1}, 1, true},
        // clang-format on
    };

    return GetConvTestForGroupXdlops<miopenHalf>(miopenTensorNHWC, std::move(test_case));
}

auto GetTestParams()
{
// If MIOpen is built without CK these tests will fail, skip them to avoid failing
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    Gpu supportedDevices = Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950;
#else
    Gpu supportedDevices = Gpu::None;
#endif
    auto params = miopen::unit_tests::UnitTestConvSolverParams(supportedDevices);
    params.Tunable(5);

    return params;
}

} // namespace

using GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_FP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardData,
                                                      miopenHalf>;
using GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_BFP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardData,
                                                      miopenBFloat16>;
using GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_FP32 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardData,
                                                      miopenFloat>;
using CPU_UnitTestConvSolverImplicitGemmGroupBwdXdlopsDevApplicability_FP16 =
    CPU_UnitTestConvSolverDevApplicabilityBwd_NONE;
using CPU_UnitTestConvSolverImplicitGemmGroupBwdXdlopsDeterministicApplicability_NONE =
    CPU_UnitTestConvSolverDevApplicabilityBwd_NONE;

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_FP16, ConvHipImplicitGemmGroupBwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupBwdXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_BFP16, ConvHipImplicitGemmGroupBwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupBwdXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_FP32, ConvHipImplicitGemmGroupBwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupBwdXdlops{});
};

TEST_P(CPU_UnitTestConvSolverImplicitGemmGroupBwdXdlopsDevApplicability_FP16,
       ConvHipImplicitGemmGroupBwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupBwdXdlops{});
};

TEST_P(CPU_UnitTestConvSolverImplicitGemmGroupBwdXdlopsDeterministicApplicability_NONE,
       ConvHipImplicitGemmGroupBwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupBwdXdlops{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                                          testing::ValuesIn(GetConvSmokeTestCases())));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                                          testing::ValuesIn(GetConvSmokeTestCases())));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                                          testing::ValuesIn(GetConvSmokeTestCases())));

// Full tests
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                                          testing::ValuesIn(GetConvFullTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                                          testing::ValuesIn(GetConvFullTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                                          testing::ValuesIn(GetConvFullTestCases())));

// Device applicability tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverImplicitGemmGroupBwdXdlopsDevApplicability_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetDevApplicabilityConvCase())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_UnitTestConvSolverImplicitGemmGroupBwdXdlopsDeterministicApplicability_NONE,
    testing::Combine(testing::Values(Gpu::None), testing::Values(GetDeterministicConvCase())));
