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
        //TestCase {{1, 4, 14, 28, 28}, {4, 4, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 1}
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
        TestCase {{128, 2, 28, 28, 28}, {2, 1, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 2},
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

// CK Applicability checks fail on I8 type
// ie for this case: TestCase {{1, 4, 8, 28, 28}, {4, 4, 3, 3, 3}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, 1}
// using GPU_UnitTestConvSolverImplicitGemm3DGroupBwdXdlops_I8 =
// miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardData, miopenInt8>;

using GPU_UnitTestConvSolverImplicitGemm3DGroupBwdXdlops_FP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardData, miopenHalf>;
using GPU_UnitTestConvSolverImplicitGemm3DGroupBwdXdlops_BFP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardData,
                                                      miopenBFloat16>;
using GPU_UnitTestConvSolverImplicitGemm3DGroupBwdXdlops_FP32 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardData,
                                                      miopenFloat>;
using CPU_UnitTestConvSolverImplicitGemm3DGroupBwdXdlopsDevApplicability_NONE =
    CPU_UnitTestConvSolverDevApplicabilityBwd_NONE;

TEST_P(GPU_UnitTestConvSolverImplicitGemm3DGroupBwdXdlops_FP16, ConvHipImplicitGemm3DGroupBwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupBwdXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemm3DGroupBwdXdlops_BFP16,
       ConvHipImplicitGemm3DGroupBwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupBwdXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemm3DGroupBwdXdlops_FP32, ConvHipImplicitGemm3DGroupBwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupBwdXdlops{});
};

TEST_P(CPU_UnitTestConvSolverImplicitGemm3DGroupBwdXdlopsDevApplicability_NONE,
       ConvHipImplicitGemm3DGroupBwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupBwdXdlops{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverImplicitGemm3DGroupBwdXdlops_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                                          testing::ValuesIn(GetConvSmokeTestCases())));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverImplicitGemm3DGroupBwdXdlops_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                                          testing::ValuesIn(GetConvSmokeTestCases())));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverImplicitGemm3DGroupBwdXdlops_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                                          testing::ValuesIn(GetConvSmokeTestCases())));

// Full tests
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverImplicitGemm3DGroupBwdXdlops_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                                          testing::ValuesIn(GetConvFullTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverImplicitGemm3DGroupBwdXdlops_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                                          testing::ValuesIn(GetConvFullTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverImplicitGemm3DGroupBwdXdlops_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                                          testing::ValuesIn(GetConvFullTestCases())));

// Device applicability tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverImplicitGemm3DGroupBwdXdlopsDevApplicability_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetDevApplicabilityConvCase())));
