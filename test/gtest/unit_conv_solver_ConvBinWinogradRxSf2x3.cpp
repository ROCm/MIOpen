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
        TestCase{
            {datatype, {1, 40, 20, 20}},
            {datatype, {20, 20, 3, 3}},
            datatype,
            {{1, 1}, {1, 1}, {1, 1}, 2}
        },
        // clang-format on
    };
}

template <miopenDataType_t datatype>
const auto& GetTestParams()
{
    static const auto params = [] {
        Gpu supported_gpus = Gpu::gfx906 | Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950 |
                             Gpu::gfx103X | Gpu::gfx110X | Gpu::gfx115X;
        if constexpr(datatype == miopenFloat)
        {
            supported_gpus = supported_gpus | Gpu::gfx900;
        }
        auto p = miopen::unit_tests::UnitTestConvSolverParams(supported_gpus);
        p.Tunable(5);
        p.CheckXnackDisabled();
        p.SetConvAttrFp16Alt(0);
        return p;
    }();
    return params;
}

const auto& GetTestParamsHalf() { return GetTestParams<miopenHalf>(); }

const auto& GetTestParamsFloat() { return GetTestParams<miopenFloat>(); }

} // namespace

using GPU_UnitTestConvSolverBinWinogradRxSf2x3Fwd_FP16 = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverBinWinogradRxSf2x3Bwd_FP16 = GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverBinWinogradRxSf2x3Wrw_FP16 = GPU_UnitTestConvSolverWrw_FP16;
using GPU_UnitTestConvSolverBinWinogradRxSf2x3Fwd_FP32 = GPU_UnitTestConvSolverFwd_FP32;
using GPU_UnitTestConvSolverBinWinogradRxSf2x3Bwd_FP32 = GPU_UnitTestConvSolverBwd_FP32;
using GPU_UnitTestConvSolverBinWinogradRxSf2x3Wrw_FP32 = GPU_UnitTestConvSolverWrw_FP32;
using CPU_UnitTestConvSolverBinWinogradRxSf2x3DevApplicabilityFwd_FP16 =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;
using CPU_UnitTestConvSolverBinWinogradRxSf2x3DevApplicabilityFwd_FP32 =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverBinWinogradRxSf2x3Fwd_FP16, ConvBinWinogradRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvBinWinoRxS<2, 3>{});
};

TEST_P(GPU_UnitTestConvSolverBinWinogradRxSf2x3Bwd_FP16, ConvBinWinogradRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvBinWinoRxS<2, 3>{});
};

TEST_P(GPU_UnitTestConvSolverBinWinogradRxSf2x3Wrw_FP16, ConvBinWinogradRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvBinWinoRxS<2, 3>{});
};

TEST_P(GPU_UnitTestConvSolverBinWinogradRxSf2x3Fwd_FP32, ConvBinWinogradRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvBinWinoRxS<2, 3>{});
};

TEST_P(GPU_UnitTestConvSolverBinWinogradRxSf2x3Bwd_FP32, ConvBinWinogradRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvBinWinoRxS<2, 3>{});
};

TEST_P(GPU_UnitTestConvSolverBinWinogradRxSf2x3Wrw_FP32, ConvBinWinogradRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvBinWinoRxS<2, 3>{});
};

TEST_P(CPU_UnitTestConvSolverBinWinogradRxSf2x3DevApplicabilityFwd_FP16, ConvBinWinogradRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvBinWinoRxS<2, 3>{});
};

TEST_P(CPU_UnitTestConvSolverBinWinogradRxSf2x3DevApplicabilityFwd_FP32, ConvBinWinogradRxSf2x3)
{
    this->RunTest(miopen::solver::conv::ConvBinWinoRxS<2, 3>{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverBinWinogradRxSf2x3Fwd_FP16,
                         testing::Combine(testing::Values(GetTestParamsHalf()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverBinWinogradRxSf2x3Bwd_FP16,
                         testing::Combine(testing::Values(GetTestParamsHalf()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverBinWinogradRxSf2x3Wrw_FP16,
                         testing::Combine(testing::Values(GetTestParamsHalf()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverBinWinogradRxSf2x3Fwd_FP32,
                         testing::Combine(testing::Values(GetTestParamsFloat()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverBinWinogradRxSf2x3Bwd_FP32,
                         testing::Combine(testing::Values(GetTestParamsFloat()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverBinWinogradRxSf2x3Wrw_FP32,
                         testing::Combine(testing::Values(GetTestParamsFloat()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverBinWinogradRxSf2x3DevApplicabilityFwd_FP16,
                         testing::Combine(testing::Values(GetTestParamsHalf()),
                                          testing::Values(GetConvTestCases(miopenHalf)[0])));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverBinWinogradRxSf2x3DevApplicabilityFwd_FP32,
                         testing::Combine(testing::Values(GetTestParamsFloat()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));
