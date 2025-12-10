// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "unit_conv_solver_group_xdlops.hpp"

namespace {

// numeric part of test case
using TestCase = miopen::unit_tests::GroupXdlopsNumericData;

auto GetConvSmokeTestCases()
{
    std::vector<TestCase> test_cases = {
        // clang-format off
        TestCase{{1, 64, 8, 8}, {96, 64, 1, 1}, {0, 0}, {1, 1}, {1, 1}, 1}
        // clang-format on
    };

    return test_cases;
}

auto GetConvFullTestCases()
{
    std::vector<TestCase> test_cases = {
        // clang-format off
        TestCase{{1, 32, 8, 8}, {48, 32, 1, 1}, {1, 1}, {1, 1}, {1, 1}, 1}, // non-zero padding
        TestCase{{1, 32, 8, 8}, {48, 32, 1, 1}, {0, 0}, {2, 2}, {1, 1}, 1}, // stride > 1
        TestCase{{1, 32, 8, 8}, {48, 32, 1, 1}, {0, 0}, {1, 1}, {2, 2}, 1}, // dilation > 1
        TestCase{{1, 32, 24, 48}, {192, 32, 1, 1}, {0, 0}, {1, 1}, {1, 1}, 1},
        // Group count = 2 and 4
        TestCase{{1, 32, 8, 8}, {48, 16, 1, 1}, {0, 0}, {1, 1}, {2, 2}, 2}, // dilation > 1
        TestCase{{1, 32, 24, 48}, {96, 8, 1, 1}, {0, 0}, {1, 1}, {1, 1}, 4},
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
        TestCase{{1, 64, 8, 8}, {96, 64, 1, 1}, {0, 0}, {1, 1}, {1, 1}, 1, true}
        // clang-format on
    };

    return GetConvTestForGroupXdlops<miopenHalf>(miopenTensorNHWC, std::move(test_case));
}

const auto& GetTestParams()
{
    static const auto params = [] {
// If MIOpen is built without CK these tests will fail, skip them to avoid failing
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
        Gpu supportedDevices = Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950;
#else
        Gpu supportedDevices = Gpu::None;
#endif
        auto p = miopen::unit_tests::UnitTestConvSolverParams(supportedDevices);
        p.Tunable(5);
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_I8 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::Forward, miopenInt8>;
using GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_FP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::Forward, miopenHalf>;
using GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_BFP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::Forward,
                                                      miopenBFloat16>;
using GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_FP32 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::Forward,
                                                      miopenFloat>;
using CPU_UnitTestConvSolverImplicitGemmGroupFwdXdlopsDevApplicability_FP16 =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;
using CPU_UnitTestConvSolverImplicitGemmGroupFwdXdlopsDeterministicApplicability_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_I8, ConvHipImplicitGemmGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_FP16, ConvHipImplicitGemmGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_BFP16, ConvHipImplicitGemmGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_FP32, ConvHipImplicitGemmGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops{});
};

TEST_P(CPU_UnitTestConvSolverImplicitGemmGroupFwdXdlopsDevApplicability_FP16,
       ConvHipImplicitGemmGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops{});
};

TEST_P(CPU_UnitTestConvSolverImplicitGemmGroupFwdXdlopsDeterministicApplicability_NONE,
       ConvHipImplicitGemmGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_I8,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                                          testing::ValuesIn(GetConvSmokeTestCases())));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                                          testing::ValuesIn(GetConvSmokeTestCases())));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                                          testing::ValuesIn(GetConvSmokeTestCases())));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                                          testing::ValuesIn(GetConvSmokeTestCases())));

// Full tests
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_I8,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                                          testing::ValuesIn(GetConvFullTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                                          testing::ValuesIn(GetConvFullTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                                          testing::ValuesIn(GetConvFullTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverImplicitGemmGroupFwdXdlops_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                                          testing::ValuesIn(GetConvFullTestCases())));

// Device applicability tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverImplicitGemmGroupFwdXdlopsDevApplicability_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetDevApplicabilityConvCase())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_UnitTestConvSolverImplicitGemmGroupFwdXdlopsDeterministicApplicability_NONE,
    testing::Combine(testing::Values(Gpu::None), testing::Values(GetDeterministicConvCase())));
