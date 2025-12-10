// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "unit_conv_solver_group_xdlops.hpp"

namespace {

// numeric part of test case
using TestCase = miopen::unit_tests::GroupXdlopsNumericData;

auto GetConvSmokeTestCases()
{
    static std::vector<TestCase> test_cases = {
        // clang-format off
        TestCase{{1, 64, 8, 8}, {96, 64, 1, 1}, {0, 0}, {1, 1}, {1, 1}, 1}
        // clang-format on
    };

    return test_cases;
}

auto GetConvFullTestCases()
{
    static std::vector<TestCase> test_cases = {
        // clang-format off
        TestCase{{1, 64, 8, 8}, {96, 64, 1, 1}, {1, 1}, {1, 1}, {1, 1}, 1}, // non-zero padding
        TestCase{{1, 64, 8, 8}, {96, 64, 1, 1}, {0, 0}, {2, 2}, {1, 1}, 1}, // stride > 1

        // Group count = 2 and 4
        TestCase{{1, 64, 8, 8}, {96, 32, 1, 1}, {0, 0}, {1, 1}, {2, 2}, 2}, // dilation > 1
        TestCase{{1, 64, 8, 8}, {96, 16, 1, 1}, {0, 0}, {2, 2}, {1, 1}, 4}, // stride > 1
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

template <miopenDataType_t datatype>
const auto& GetTestParams()
{
    static const auto params = [] {
// If MIOpen is built without CK these tests will fail, skip them to avoid failing
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
        Gpu supportedDevices = Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950;
        if(datatype == miopenBFloat16)
            supportedDevices = Gpu::gfx94X | Gpu::gfx950;
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

// Solver itself supports I8 in isApplicable, but CK returns 0 compatible kernels

using GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_FP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardWeights,
                                                      miopenHalf>;
;
using GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_BFP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardWeights,
                                                      miopenBFloat16>;
;
using GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_FP32 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardWeights,
                                                      miopenFloat>;
;
using CPU_UnitTestConvSolverImplicitGemmGroupWrwXdlopsDevApplicability_FP16 =
    CPU_UnitTestConvSolverDevApplicabilityWrw_NONE;
using CPU_UnitTestConvSolverImplicitGemmGroupWrwXdlopsDeterministicApplicability_NONE =
    CPU_UnitTestConvSolverDevApplicabilityWrw_NONE;

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_FP16, ConvHipImplicitGemmGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_BFP16, ConvHipImplicitGemmGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_FP32, ConvHipImplicitGemmGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops{});
};

TEST_P(CPU_UnitTestConvSolverImplicitGemmGroupWrwXdlopsDevApplicability_FP16,
       ConvHipImplicitGemmGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops{});
};

TEST_P(CPU_UnitTestConvSolverImplicitGemmGroupWrwXdlopsDeterministicApplicability_NONE,
       ConvHipImplicitGemmGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_FP16,
                         testing::Combine(testing::Values(GetTestParams<miopenHalf>()),
                                          testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                                          testing::ValuesIn(GetConvSmokeTestCases())));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_BFP16,
                         testing::Combine(testing::Values(GetTestParams<miopenBFloat16>()),
                                          testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                                          testing::ValuesIn(GetConvSmokeTestCases())));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_FP32,
                         testing::Combine(testing::Values(GetTestParams<miopenFloat>()),
                                          testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                                          testing::ValuesIn(GetConvSmokeTestCases())));

// Full tests

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_FP16,
                         testing::Combine(testing::Values(GetTestParams<miopenHalf>()),
                                          testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                                          testing::ValuesIn(GetConvFullTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_BFP16,
                         testing::Combine(testing::Values(GetTestParams<miopenBFloat16>()),
                                          testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                                          testing::ValuesIn(GetConvFullTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_FP32,
                         testing::Combine(testing::Values(GetTestParams<miopenFloat>()),
                                          testing::Values(miopenTensorNHWC, miopenTensorNCHW),
                                          testing::ValuesIn(GetConvFullTestCases())));

// Device applicability tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverImplicitGemmGroupWrwXdlopsDevApplicability_FP16,
                         testing::Combine(testing::Values(GetTestParams<miopenHalf>()),
                                          testing::Values(GetDevApplicabilityConvCase())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_UnitTestConvSolverImplicitGemmGroupWrwXdlopsDeterministicApplicability_NONE,
    testing::Combine(testing::Values(Gpu::None), testing::Values(GetDeterministicConvCase())));
