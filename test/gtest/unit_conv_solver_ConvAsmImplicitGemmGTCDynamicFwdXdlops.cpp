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
        TestCase{{64, 128, 7, 7}, {128, 128, 3, 3}, {1, 1}, {1, 1}, {1, 1}, datatype},
        // clang-format on
    };
}

auto GetConvTestCasesFull(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    auto cases = std::vector{
        // clang-format off
        // Be careful to add testings for (x=1, y=1, c % 8 != 0) due to WORKAROUND_SWDEV_306318
        TestCase{{ 64, 1024,  14,  14}, {1024, 1024, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        TestCase{{ 64,  256,  56,  56}, { 512,  256, 1, 1}, {0, 0}, {2, 2}, {1, 1}, datatype},
        TestCase{{ 64, 2048,   7,   7}, {2048, 2048, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        TestCase{{128,  128,  17,  17}, { 128,  128, 7, 1}, {3, 0}, {1, 1}, {1, 1}, datatype},
        TestCase{{128,  128,  17,  17}, { 128,  128, 1, 7}, {0, 3}, {1, 1}, {1, 1}, datatype},
        TestCase{{128,  192,  17,  17}, { 320,  192, 3, 3}, {0, 0}, {2, 2}, {1, 1}, datatype},
        TestCase{{128,  256,  35,  35}, {  64,  256, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        TestCase{{128,   48,  35,  35}, {  64,   48, 5, 5}, {2, 2}, {1, 1}, {1, 1}, datatype},
        TestCase{{ 64,  512,   7,   7}, { 512,  512, 3, 3}, {1, 1}, {1, 1}, {1, 1}, datatype},
        TestCase{{ 32, 1024,  14,  14}, {2048, 1024, 1, 1}, {0, 0}, {2, 2}, {1, 1}, datatype},
        TestCase{{  2,  256, 100, 104}, {  12,  256, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        TestCase{{  1,  256,  28,  28}, {  80,  256, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        // ho=wo=1 stride=2
        TestCase{{256, 2048,   2,   2}, {1024, 2048, 1, 1}, {0, 0}, {2, 2}, {1, 1}, datatype},
        // Regression tests for wo=1/ho=1 stride clamping bug  
        TestCase{{32, 3, 1, 1}, {64, 3,  3,  5}, {1, 2}, {100, 7}, {1, 1}, datatype},
        TestCase{{32, 3, 2, 2}, {64, 3,  3,  5}, {2, 4}, {  4, 6}, {1, 1}, datatype},
        TestCase{{32, 3, 7, 9}, {64, 3, 10, 12}, {3, 3}, {  4, 5}, {1, 1}, datatype},
        // clang-format on
    };

    if(datatype == miopenHalf)
    {
        // clang-format off
        cases.emplace_back(TestCase{{64, 3, 224, 224}, {64, 3, 7, 7}, {3, 3}, {2, 2}, {1, 1}, datatype});
        cases.emplace_back(TestCase{{64, 3, 230, 230}, {64, 3, 7, 7}, {0, 0}, {2, 2}, {1, 1}, datatype});
        // clang-format on
    }

    return cases;
}

const auto& GetTestParams()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::gfx908);
        p.CheckXnackDisabled();
        p.SetTolerance(Gpu::gfx908, miopenFloat, 2.0f);
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicXdlopsFwd_FP16 =
    GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicXdlopsFwd_FP32 =
    GPU_UnitTestConvSolverFwd_FP32;
using CPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicXdlopsDevApplicabilityFwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicXdlopsFwd_FP16,
       ConvAsmImplicitGemmGTCDynamicFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvAsmImplicitGemmGTCDynamicFwdXdlops{});
};

TEST_P(GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicXdlopsFwd_FP32,
       ConvAsmImplicitGemmGTCDynamicFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvAsmImplicitGemmGTCDynamicFwdXdlops{});
};

TEST_P(CPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicXdlopsDevApplicabilityFwd_NONE,
       ConvAsmImplicitGemmGTCDynamicFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvAsmImplicitGemmGTCDynamicFwdXdlops{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicXdlopsFwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicXdlopsFwd_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicXdlopsDevApplicabilityFwd_NONE,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(GetConvTestCases(miopenFloat)[0])));

// Full tests
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicXdlopsFwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCasesFull(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicXdlopsFwd_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCasesFull(miopenFloat))));
