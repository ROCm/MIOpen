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
        TestCase {{1, 4, 8, 28, 28}, {4, 4, 3, 3, 3}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, 1}
        // clang-format on
    };

    return test_cases;
}

auto GetConvFullTestCases()
{
    std::vector<TestCase> test_cases = {
        // clang-format off
        // Group Count = 1
        TestCase {{1, 1, 8, 8, 8}, {1, 1, 2, 2, 2}, {0, 0, 0}, {2, 2, 2}, {1, 1, 1}, 1},
        TestCase {{6, 448, 3, 118, 182}, {896, 448, 1, 1, 1}, {0, 0, 0}, {1, 2, 2}, {1, 1, 1}, 1},

        // Group Count > 1 (2, 3, 4)
        TestCase {{128, 2, 28, 28, 28}, {2, 1, 3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, 2},
        TestCase {{128, 2, 28, 28, 28}, {2, 1, 3, 3, 3}, {1, 1, 1}, {2, 2, 2}, {1, 1, 1}, 2},
        TestCase {{256, 9, 2, 14, 14}, {27, 3, 2, 14, 14}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 3},
        TestCase {{128, 4, 28, 28, 28}, {8, 1, 3, 3, 3}, {1, 1, 1}, {2, 2, 2}, {1, 1, 1}, 4}
        // clang-format on  
    };

    return test_cases;
}

auto GetDevApplicabilityConvCase()
{
    // For device applicability checks
    return GetConvTestForGroupXdlops<miopenHalf>(miopenTensorNDHWC,
                                                 std::move(GetConvSmokeTestCases()[0]));
}

// Deterministic test case (for CPU deterministic applicability test)
auto GetDeterministicConvCase()
{
    TestCase test_case = {
        // clang-format off
        TestCase {{1, 4, 8, 28, 28}, {4, 4, 3, 3, 3}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, 1, true}
        // clang-format on
    };

    return GetConvTestForGroupXdlops<miopenHalf>(miopenTensorNDHWC, std::move(test_case));
}

template <miopenDataType_t datatype>
const auto& GetTestParams()
{
    static const auto params = [] {
// If MIOpen is built without CK these tests will fail, skip them to avoid failing
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL && !defined(_WIN32)
        Gpu supportedDevices = Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950;

        if(datatype == miopenBFloat16)
            supportedDevices = Gpu::gfx94X | Gpu::gfx950;
#else
        Gpu supportedDevices = Gpu::None;
#endif
        auto p = miopen::unit_tests::UnitTestConvSolverParams(supportedDevices);
        p.Tunable(5);

        // Increased tolerance because of tolerance failures
        p.SetTolerance(supportedDevices, miopenFloat, 20.0f);

        return p;
    }();
    return params;
}

} // namespace

// For I8 datatype we get "Empty code object path", so it requires additional
// investigation/debugging

using GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_FP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardWeights,
                                                      miopenHalf>;
using GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_BFP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardWeights,
                                                      miopenBFloat16>;
using GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_FP32 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardWeights,
                                                      miopenFloat>;
using CPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsDevApplicability_FP16 =
    CPU_UnitTestConvSolverDevApplicabilityWrw_NONE;
using CPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsDeterministicApplicability_NONE =
    CPU_UnitTestConvSolverDevApplicabilityWrw_NONE;

TEST_P(GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_FP16, ConvHipImplicitGemm3DGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_BFP16,
       ConvHipImplicitGemm3DGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_FP32, ConvHipImplicitGemm3DGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{});
};

TEST_P(CPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsDevApplicability_FP16,
       ConvHipImplicitGemm3DGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{});
};

TEST_P(CPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsDeterministicApplicability_NONE,
       ConvHipImplicitGemm3DGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_FP16,
                         testing::Combine(testing::Values(GetTestParams<miopenHalf>()),
                                          testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                                          testing::ValuesIn(GetConvSmokeTestCases())));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_BFP16,
                         testing::Combine(testing::Values(GetTestParams<miopenBFloat16>()),
                                          testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                                          testing::ValuesIn(GetConvSmokeTestCases())));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_FP32,
                         testing::Combine(testing::Values(GetTestParams<miopenFloat>()),
                                          testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                                          testing::ValuesIn(GetConvSmokeTestCases())));

// Full tests
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_FP16,
                         testing::Combine(testing::Values(GetTestParams<miopenHalf>()),
                                          testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                                          testing::ValuesIn(GetConvFullTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_BFP16,
                         testing::Combine(testing::Values(GetTestParams<miopenBFloat16>()),
                                          testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                                          testing::ValuesIn(GetConvFullTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_FP32,
                         testing::Combine(testing::Values(GetTestParams<miopenFloat>()),
                                          testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                                          testing::ValuesIn(GetConvFullTestCases())));

// Device applicability tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsDevApplicability_FP16,
                         testing::Combine(testing::Values(GetTestParams<miopenHalf>()),
                                          testing::Values(GetDevApplicabilityConvCase())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsDeterministicApplicability_NONE,
    testing::Combine(testing::Values(Gpu::None), testing::Values(GetDeterministicConvCase())));
