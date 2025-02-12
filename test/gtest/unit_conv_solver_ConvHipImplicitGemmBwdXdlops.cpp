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

namespace {

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

auto GetTestParams(miopenDataType_t datatype)
{
    Gpu supportedDevices = Gpu::gfx908 | Gpu::gfx94X;
    if(datatype != miopenHalf)
    {
        supportedDevices = supportedDevices | Gpu::gfx90A;
    }
    auto params = miopen::unit_tests::UnitTestConvSolverParams(supportedDevices);
    params.Tunable(5);

    return params;
}

} // namespace

using GPU_UnitTestConvSolverImplicitGemmBwdXdlops_FP16  = GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverImplicitGemmBwdXdlops_BFP16 = GPU_UnitTestConvSolverBwd_BFP16;
using GPU_UnitTestConvSolverImplicitGemmBwdXdlops_FP32  = GPU_UnitTestConvSolverBwd_FP32;
using CPU_UnitTestConvSolverImplicitGemmBwdXdlopsDevApplicability_FP16 =
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

TEST_P(CPU_UnitTestConvSolverImplicitGemmBwdXdlopsDevApplicability_FP16,
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

// Device applicability tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverImplicitGemmBwdXdlopsDevApplicability_FP16,
                         testing::Combine(testing::Values(GetTestParams(miopenHalf)),
                                          testing::Values(GetConvSmokeTestCases(miopenHalf)[0])));
