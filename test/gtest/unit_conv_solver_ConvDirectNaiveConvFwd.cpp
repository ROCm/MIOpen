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

    auto type_x = datatype;
    auto type_w = datatype;
    auto type_y = (datatype == miopenInt8) ? miopenInt32 : datatype;

    return std::vector{
        // clang-format off
        TestCase{{1, 16, 14, 14}, {48, 16, 5, 5}, {2, 2}, {1, 1}, {1, 1}, type_x, type_w, type_y},
        // clang-format on
    };
}

auto GetConvTestCasesFull(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    auto cases = std::vector<TestCase>{};

    if(datatype == miopenInt8)
    {
        auto type_x = datatype;
        auto type_w = datatype;
        auto type_y = miopenInt32;

        // clang-format off
        // Regression test for int8, issue unknown
        cases.emplace_back(TestCase{{256, 1024, 14, 14}, {256, 1024, 1, 1}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y});
        // clang-format on
    }

    if(datatype == miopenHalf || datatype == miopenFloat)
    {
        // clang-format off
        // Regression test for https://github.com/ROCm/MIOpen/issues/3279
        cases.emplace_back(TestCase{{datatype, miopenTensorNHWC, {1, 1, 7, 7}}, {datatype, {1, 1, 1, 1}}, datatype, {{0, 0}, {1, 1}, {1, 1}}});
        // clang-format on
    }

    return cases;
}

Gpu GetSupportedDevices() { return Gpu::All; }

} // namespace

TEST_P(GPU_UnitTestConvSolverFwd_I8, ConvDirectNaiveConvFwd)
{
    this->RunTest(miopen::solver::conv::ConvDirectNaiveConvFwd{}, true); // CPU verification
};

TEST_P(GPU_UnitTestConvSolverFwd_FP16, ConvDirectNaiveConvFwd)
{
    this->RunTest(miopen::solver::conv::ConvDirectNaiveConvFwd{}, true); // CPU verification
};

TEST_P(GPU_UnitTestConvSolverFwd_BFP16, ConvDirectNaiveConvFwd)
{
    this->RunTest(miopen::solver::conv::ConvDirectNaiveConvFwd{}, true); // CPU verification
};

TEST_P(GPU_UnitTestConvSolverFwd_FP32, ConvDirectNaiveConvFwd)
{
    this->RunTest(miopen::solver::conv::ConvDirectNaiveConvFwd{}, true); // CPU verification
};

TEST_P(CPU_UnitTestConvSolverDevApplicabilityFwd_NONE, ConvDirectNaiveConvFwd)
{
    this->RunTest(miopen::solver::conv::ConvDirectNaiveConvFwd{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverFwd_I8,
                         testing::Combine(testing::Values(GetSupportedDevices()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenInt8))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverFwd_FP16,
                         testing::Combine(testing::Values(GetSupportedDevices()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverFwd_BFP16,
                         testing::Combine(testing::Values(GetSupportedDevices()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverFwd_FP32,
                         testing::Combine(testing::Values(GetSupportedDevices()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverDevApplicabilityFwd_NONE,
                         testing::Combine(testing::Values(GetSupportedDevices()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));

// Full tests
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverFwd_I8,
                         testing::Combine(testing::Values(GetSupportedDevices()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCasesFull(miopenInt8))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverFwd_FP16,
                         testing::Combine(testing::Values(GetSupportedDevices()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCasesFull(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverFwd_FP32,
                         testing::Combine(testing::Values(GetSupportedDevices()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCasesFull(miopenFloat))));
