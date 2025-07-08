/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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
        // rage v4.6
        TestCase{{1, 16, 135, 240}, {16, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, datatype},
        TestCase{{2,  4,  64,  64}, {16,  4, 3, 3}, {1, 1}, {1, 1}, {1, 1}, datatype},
        // rage v4.7
        TestCase{{1, 16, 135, 240}, {16, 16, 5, 5}, {2, 2}, {1, 1}, {1, 1}, datatype},
        TestCase{{2,  4,  64,  64}, {16,  4, 5, 5}, {2, 2}, {1, 1}, {1, 1}, datatype},
        // group convs
        TestCase{{2, 15,  28,  28}, {15,  3, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 5, datatype},
        TestCase{{2, 15,  28,  28}, {15,  3, 5, 5}, {2, 2}, {1, 1}, {1, 1}, 5, datatype},
        // clang-format on
    };
}

auto GetConvTestCasesWrw(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{1, 16,  5,  5}, {16, 16, 3, 3}, {0, 0}, {1, 1}, {1, 1}, datatype},
        TestCase{{1, 32,  7,  7}, { 4, 32, 3, 3}, {0, 0}, {1, 1}, {1, 1}, datatype},
        // group convs
        TestCase{{1, 16,  5,  5}, {16,  1, 3, 3}, {0, 0}, {1, 1}, {1, 1}, 16, datatype},
        TestCase{{2, 16, 28, 28}, {16,  1, 5, 5}, {2, 2}, {1, 1}, {1, 1}, 16, datatype},
        // clang-format on
    };
}

const auto& GetTestParams()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::gfx94X);
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverWinoRage2x3Fwd_FP16 = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverWinoRage2x3Bwd_FP16 = GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverWinoRage2x3Wrw_FP16 = GPU_UnitTestConvSolverWrw_FP16;
using CPU_UnitTestConvSolverWinoRage2x3DevApplicabilityFwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverWinoRage2x3Fwd_FP16, ConvWinoRageRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvWinoRageRxS<2, 3>{});
};

TEST_P(GPU_UnitTestConvSolverWinoRage2x3Bwd_FP16, ConvWinoRageRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvWinoRageRxS<2, 3>{});
};

TEST_P(GPU_UnitTestConvSolverWinoRage2x3Wrw_FP16, ConvWinoRageRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvWinoRageRxS<2, 3>{});
};

TEST_P(CPU_UnitTestConvSolverWinoRage2x3DevApplicabilityFwd_NONE, ConvWinoRageRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvWinoRageRxS<2, 3>{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverWinoRage2x3Fwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverWinoRage2x3Bwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverWinoRage2x3Wrw_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCasesWrw(miopenHalf))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverWinoRage2x3DevApplicabilityFwd_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenHalf)[0])));
