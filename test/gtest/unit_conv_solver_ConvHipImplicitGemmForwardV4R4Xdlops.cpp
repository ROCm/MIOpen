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

#define SOLVER_NAME ConvHipImplicitGemmForwardV4R4Xdlops

namespace {

auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{128, 48, 13, 13}, {192, 48, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        // clang-format on
    };
}

auto GetConvTestCasesFull(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    auto cases = std::vector<TestCase>{};

    if(datatype == miopenHalf)
    {
        // clang-format off
        // Regression test for MIOpen-internal issue #4 (MI200)
        cases.emplace_back(TestCase{{120, 64, 75, 75}, {128, 64, 1, 1}, {0, 0}, {2, 2}, {1, 1}, miopenHalf});
        // clang-format on
    }

    return cases;
}

const auto& GetTestParams()
{
    static const auto params = [] {
        Gpu supported_gpus = Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx94X;
        auto p             = miopen::unit_tests::UnitTestConvSolverParams(supported_gpus);
        p.EnableDeprecatedSolvers();
        p.Tunable(5);
        p.SetConvAttrFp16Alt(0);
        return p;
    }();
    return params;
}

const auto& GetTestParamsFull()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::gfx90A);
        p.EnableDeprecatedSolvers();
        p.Tunable(1000);
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverHipImplicitGemmForwardV4R4XdlopsFwd_FP16 =
    GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverHipImplicitGemmForwardV4R4XdlopsFwd_BFP16 =
    GPU_UnitTestConvSolverFwd_BFP16;
using GPU_UnitTestConvSolverHipImplicitGemmForwardV4R4XdlopsFwd_FP32 =
    GPU_UnitTestConvSolverFwd_FP32;
using CPU_UnitTestConvSolverHipImplicitGemmForwardV4R4XdlopsDevApplicabilityFwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverHipImplicitGemmForwardV4R4XdlopsFwd_FP16, SOLVER_NAME)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmForwardV4R4Xdlops{});
};

TEST_P(GPU_UnitTestConvSolverHipImplicitGemmForwardV4R4XdlopsFwd_BFP16, SOLVER_NAME)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmForwardV4R4Xdlops{});
};

TEST_P(GPU_UnitTestConvSolverHipImplicitGemmForwardV4R4XdlopsFwd_FP32, SOLVER_NAME)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmForwardV4R4Xdlops{});
};

TEST_P(CPU_UnitTestConvSolverHipImplicitGemmForwardV4R4XdlopsDevApplicabilityFwd_NONE, SOLVER_NAME)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmForwardV4R4Xdlops{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipImplicitGemmForwardV4R4XdlopsFwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipImplicitGemmForwardV4R4XdlopsFwd_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverHipImplicitGemmForwardV4R4XdlopsFwd_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_UnitTestConvSolverHipImplicitGemmForwardV4R4XdlopsDevApplicabilityFwd_NONE,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(GetConvTestCases(miopenFloat)[0])));

// Full tests
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverHipImplicitGemmForwardV4R4XdlopsFwd_FP16,
                         testing::Combine(testing::Values(GetTestParamsFull()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCasesFull(miopenHalf))));
