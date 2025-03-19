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
        TestCase{{1, 16, 24, 24}, {16, 16, 7, 1}, {3, 0}, {1, 1}, {1, 1}, datatype},
        // clang-format on
    };
}

template <miopenDataType_t datatype>
const auto& GetTestParams()
{
    static const auto params = [] {
        Gpu supported_gpus = Gpu::gfx900 | Gpu::gfx906 | Gpu::gfx908 | Gpu::gfx90A;
        auto p             = miopen::unit_tests::UnitTestConvSolverParams(supported_gpus);
        p.CheckXnackDisabled();
        if constexpr(datatype == miopenHalf)
        {
            p.SetConvAttrFp16Alt(0);
        }
        return p;
    }();
    return params;
}

const auto& GetTestParamsFP16() { return GetTestParams<miopenHalf>(); }
const auto& GetTestParamsBFP16() { return GetTestParams<miopenBFloat16>(); }
const auto& GetTestParamsFP32() { return GetTestParams<miopenFloat>(); }

} // namespace

using GPU_UnitTestConvSolverWinograd3x3MultipassF7x2x1x1Wrw_FP16  = GPU_UnitTestConvSolverWrw_FP16;
using GPU_UnitTestConvSolverWinograd3x3MultipassF7x2x1x1Wrw_BFP16 = GPU_UnitTestConvSolverWrw_BFP16;
using GPU_UnitTestConvSolverWinograd3x3MultipassF7x2x1x1Wrw_FP32  = GPU_UnitTestConvSolverWrw_FP32;

using CPU_UnitTestConvSolverWinograd3x3MultipassF7x2x1x1DevApplicabilityWrw_NONE =
    CPU_UnitTestConvSolverDevApplicabilityWrw_NONE;

TEST_P(GPU_UnitTestConvSolverWinograd3x3MultipassF7x2x1x1Wrw_FP16,
       ConvWinograd3x3MultipassWrWF7x2x1x1)
{
    this->RunTest(miopen::solver::conv::ConvWinograd3x3MultipassWrW<7, 2, 1, 1>{});
};

TEST_P(GPU_UnitTestConvSolverWinograd3x3MultipassF7x2x1x1Wrw_BFP16,
       ConvWinograd3x3MultipassWrWF7x2x1x1)
{
    this->RunTest(miopen::solver::conv::ConvWinograd3x3MultipassWrW<7, 2, 1, 1>{});
};

TEST_P(GPU_UnitTestConvSolverWinograd3x3MultipassF7x2x1x1Wrw_FP32,
       ConvWinograd3x3MultipassWrWF7x2x1x1)
{
    this->RunTest(miopen::solver::conv::ConvWinograd3x3MultipassWrW<7, 2, 1, 1>{});
};

TEST_P(CPU_UnitTestConvSolverWinograd3x3MultipassF7x2x1x1DevApplicabilityWrw_NONE,
       ConvWinograd3x3MultipassWrWF7x2x1x1)
{
    this->RunTest(miopen::solver::conv::ConvWinograd3x3MultipassWrW<7, 2, 1, 1>{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverWinograd3x3MultipassF7x2x1x1Wrw_FP16,
                         testing::Combine(testing::Values(GetTestParamsFP16()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverWinograd3x3MultipassF7x2x1x1Wrw_BFP16,
                         testing::Combine(testing::Values(GetTestParamsBFP16()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverWinograd3x3MultipassF7x2x1x1Wrw_FP32,
                         testing::Combine(testing::Values(GetTestParamsFP32()),
                                          testing::Values(miopenConvolutionAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverWinograd3x3MultipassF7x2x1x1DevApplicabilityWrw_NONE,
                         testing::Combine(testing::Values(GetTestParamsFP32()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));
