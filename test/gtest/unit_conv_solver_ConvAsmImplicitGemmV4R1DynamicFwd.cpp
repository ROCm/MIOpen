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

auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{16, 16, 16, 16}, {16, 16, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        // clang-format on
    };
}

auto GetConvTestCasesFull(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        // https://github.com/ROCm/MIOpen/pull/350
        TestCase{{ 16,   16, 56, 56}, { 64,   16, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        TestCase{{ 16,   64, 34, 34}, { 64,   64, 3, 3}, {0, 0}, {1, 1}, {1, 1}, datatype},
        TestCase{{ 32,   32, 17, 17}, { 32,   32, 1, 7}, {0, 3}, {1, 1}, {1, 1}, datatype},
        // https://github.com/ROCm/MIOpen/pull/166
        TestCase{{ 64,   64, 56, 56}, {256,   64, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        TestCase{{ 64,  256, 34, 34}, {256,  256, 3, 3}, {0, 0}, {1, 1}, {1, 1}, datatype},
        TestCase{{128,  128, 35, 35}, {128,  128, 3, 3}, {0, 0}, {2, 2}, {1, 1}, datatype},
        TestCase{{ 64, 1536,  8,  8}, {256, 1536, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        TestCase{{128,   48,  7,  7}, {128,   48, 5, 5}, {2, 2}, {1, 1}, {1, 1}, datatype},
        TestCase{{128,  128, 17, 17}, {128,  128, 1, 7}, {0, 3}, {1, 1}, {1, 1}, datatype},
        // Regression tests for wo=1/ho=1 stride clamping bug  
        TestCase{{32, 3, 1, 1}, {64, 3,  3,  5}, {1, 2}, {100, 7}, {1, 1}, datatype},
        TestCase{{32, 3, 2, 2}, {64, 3,  3,  5}, {2, 4}, {  4, 6}, {1, 1}, datatype},
        TestCase{{32, 3, 7, 9}, {64, 3, 10, 12}, {3, 3}, {  4, 5}, {1, 1}, datatype},
        // clang-format on
    };
}

const auto& GetTestParams()
{
    static const auto params = [] {
        Gpu supported_gpus = Gpu::gfx900 | Gpu::gfx906;
        auto p             = miopen::unit_tests::UnitTestConvSolverParams(supported_gpus);
        p.CheckXnackDisabled();
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverAsmImplicitGemmV4R1DynamicFwd_FP32 = GPU_UnitTestConvSolverFwd_FP32;
using CPU_UnitTestConvSolverAsmImplicitGemmV4R1DynamicDevApplicabilityFwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverAsmImplicitGemmV4R1DynamicFwd_FP32, ConvAsmImplicitGemmV4R1DynamicFwd)
{
    this->RunTest(miopen::solver::conv::ConvAsmImplicitGemmV4R1DynamicFwd{});
};

TEST_P(CPU_UnitTestConvSolverAsmImplicitGemmV4R1DynamicDevApplicabilityFwd_NONE,
       ConvAsmImplicitGemmV4R1DynamicFwd)
{
    this->RunTest(miopen::solver::conv::ConvAsmImplicitGemmV4R1DynamicFwd{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverAsmImplicitGemmV4R1DynamicFwd_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverAsmImplicitGemmV4R1DynamicDevApplicabilityFwd_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));

// Full tests
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverAsmImplicitGemmV4R1DynamicFwd_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCasesFull(miopenFloat))));
