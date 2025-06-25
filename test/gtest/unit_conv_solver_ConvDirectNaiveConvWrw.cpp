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
        TestCase{{1, 16, 14, 14}, {48, 16, 5, 5}, {2, 2}, {1, 1}, {1, 1}, datatype},
        // clang-format on
    };
}

auto GetConvTestCasesFull(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    auto cases = std::vector<TestCase>{};

    // clang-format off
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{2, 2}, {1, 1}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{0, 0}, {2, 2}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{2, 2}, {2, 2}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{0, 0}, {3, 3}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{1, 1}, {3, 3}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{2, 2}, {3, 3}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{0, 0}, {1, 1}, {2, 2}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{1, 1}, {1, 1}, {2, 2}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{2, 2}, {1, 1}, {2, 2}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{0, 0}, {2, 2}, {2, 2}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{1, 1}, {2, 2}, {2, 2}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{2, 2}, {2, 2}, {2, 2}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{0, 0}, {3, 3}, {2, 2}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{1, 1}, {3, 3}, {2, 2}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{2, 2}, {3, 3}, {2, 2}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{0, 0}, {1, 1}, {3, 3}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{1, 1}, {1, 1}, {3, 3}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{2, 2}, {1, 1}, {3, 3}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{0, 0}, {2, 2}, {3, 3}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{1, 1}, {2, 2}, {3, 3}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{2, 2}, {2, 2}, {3, 3}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{0, 0}, {3, 3}, {3, 3}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{1, 1}, {3, 3}, {3, 3}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{2, 2}, {3, 3}, {3, 3}}});

    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{2, 2}, {1, 1}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{0, 0}, {2, 2}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{2, 2}, {2, 2}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{0, 0}, {3, 3}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{1, 1}, {3, 3}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{2, 2}, {3, 3}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{0, 0}, {1, 1}, {2, 2}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{1, 1}, {1, 1}, {2, 2}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{2, 2}, {1, 1}, {2, 2}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{0, 0}, {2, 2}, {2, 2}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{1, 1}, {2, 2}, {2, 2}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{2, 2}, {2, 2}, {2, 2}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{0, 0}, {3, 3}, {2, 2}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{1, 1}, {3, 3}, {2, 2}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{2, 2}, {3, 3}, {2, 2}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{0, 0}, {1, 1}, {3, 3}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{1, 1}, {1, 1}, {3, 3}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{2, 2}, {1, 1}, {3, 3}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{0, 0}, {2, 2}, {3, 3}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{1, 1}, {2, 2}, {3, 3}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{2, 2}, {2, 2}, {3, 3}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{0, 0}, {3, 3}, {3, 3}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{1, 1}, {3, 3}, {3, 3}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{2, 2}, {3, 3}, {3, 3}}});

    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 1, 1}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 127, 127}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {1, 1, 129, 129}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}}});

    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 1, 1}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 127, 127}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 129, 129}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}}});

    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {64, 1, 32, 32}}, {datatype, miopenTensorNCHW, {1, 1, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {64, 1, 32, 32}}, {datatype, miopenTensorNCHW, {2, 1, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {64, 1, 32, 32}}, {datatype, miopenTensorNCHW, {4, 1, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {64, 1, 32, 32}}, {datatype, miopenTensorNCHW, {8, 1, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {64, 1, 32, 32}}, {datatype, miopenTensorNCHW, {16, 1, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {64, 16, 32, 32}}, {datatype, miopenTensorNCHW, {16, 1, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}, 16}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {64, 16, 32, 32}}, {datatype, miopenTensorNCHW, {4, 4, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}, 4}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {64, 16, 32, 32}}, {datatype, miopenTensorNCHW, {1, 16, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {64, 16, 32, 32}}, {datatype, miopenTensorNCHW, {8, 4, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}, 4}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {64, 16, 32, 32}}, {datatype, miopenTensorNCHW, {8, 16, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNCHW, {64, 16, 32, 32}}, {datatype, miopenTensorNCHW, {64, 16, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}}});

    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {64, 1, 32, 32}}, {datatype, miopenTensorNHWC, {1, 1, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {64, 1, 32, 32}}, {datatype, miopenTensorNHWC, {2, 1, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {64, 1, 32, 32}}, {datatype, miopenTensorNHWC, {4, 1, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {64, 1, 32, 32}}, {datatype, miopenTensorNHWC, {8, 1, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {64, 1, 32, 32}}, {datatype, miopenTensorNHWC, {16, 1, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {64, 16, 32, 32}}, {datatype, miopenTensorNHWC, {16, 1, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}, 16}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {64, 16, 32, 32}}, {datatype, miopenTensorNHWC, {4, 4, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}, 4}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {64, 16, 32, 32}}, {datatype, miopenTensorNHWC, {1, 16, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {64, 16, 32, 32}}, {datatype, miopenTensorNHWC, {8, 4, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}, 4}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {64, 16, 32, 32}}, {datatype, miopenTensorNHWC, {8, 16, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}}});
    cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {64, 16, 32, 32}}, {datatype, miopenTensorNHWC, {64, 16, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}}});
    // clang-format on

    return cases;
}

const auto& GetTestParams()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::All);
        p.UseCpuRef(); // CPU verification
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverDirectNaiveWrw_FP16  = GPU_UnitTestConvSolverWrw_FP16;
using GPU_UnitTestConvSolverDirectNaiveWrw_BFP16 = GPU_UnitTestConvSolverWrw_BFP16;
using GPU_UnitTestConvSolverDirectNaiveWrw_FP32  = GPU_UnitTestConvSolverWrw_FP32;
using CPU_UnitTestConvSolverDirectNaiveDevApplicabilityWrw_NONE =
    CPU_UnitTestConvSolverDevApplicabilityWrw_NONE;

TEST_P(GPU_UnitTestConvSolverDirectNaiveWrw_FP16, ConvDirectNaiveConvWrw)
{
    this->RunTest(miopen::solver::conv::ConvDirectNaiveConvWrw{});
};

TEST_P(GPU_UnitTestConvSolverDirectNaiveWrw_BFP16, ConvDirectNaiveConvWrw)
{
    this->RunTest(miopen::solver::conv::ConvDirectNaiveConvWrw{});
};

TEST_P(GPU_UnitTestConvSolverDirectNaiveWrw_FP32, ConvDirectNaiveConvWrw)
{
    this->RunTest(miopen::solver::conv::ConvDirectNaiveConvWrw{});
};

TEST_P(CPU_UnitTestConvSolverDirectNaiveDevApplicabilityWrw_NONE, ConvDirectNaiveConvWrw)
{
    this->RunTest(miopen::solver::conv::ConvDirectNaiveConvWrw{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverDirectNaiveWrw_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverDirectNaiveWrw_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverDirectNaiveWrw_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverDirectNaiveDevApplicabilityWrw_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));

// Full tests
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverDirectNaiveWrw_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCasesFull(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverDirectNaiveWrw_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCasesFull(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverDirectNaiveWrw_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCasesFull(miopenFloat))));
