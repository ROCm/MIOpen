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

#define WORKAROUND_SWDEV_330460 1 // ConvAsmBwdWrw3x3 has precision issues on MI200

namespace {

auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{2, 4, 3, 3}, {4, 4, 3, 3}, {1, 1}, {1, 1}, {1, 1}, datatype},
        // clang-format on
    };
}

const auto& GetTestParams()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::gfx900 | Gpu::gfx906 |
                                                              Gpu::gfx908 | Gpu::gfx90A);
        p.Tunable(5);
        p.SetConvAttrFp16Alt(0);
        p.CheckXnackDisabled();
        p.EnableDeprecatedSolvers();
        return p;
    }();
    return params;
}

#if WORKAROUND_SWDEV_330460
const auto& GetTestParamsFp32()
{
    static const auto params = [] {
        auto p =
            miopen::unit_tests::UnitTestConvSolverParams(Gpu::gfx900 | Gpu::gfx906 | Gpu::gfx908);
        p.Tunable(5);
        p.CheckXnackDisabled();
        p.EnableDeprecatedSolvers();
        return p;
    }();
    return params;
}
#endif

} // namespace

using GPU_UnitTestConvSolverAsmBwdWrW3x3Wrw_FP16 = GPU_UnitTestConvSolverWrw_FP16;
using GPU_UnitTestConvSolverAsmBwdWrW3x3Wrw_FP32 = GPU_UnitTestConvSolverWrw_FP32;

#if WORKAROUND_SWDEV_330460
using CPU_UnitTestConvSolverAsmBwdWrW3x3DevApplicabilityWrw_FP16 =
    CPU_UnitTestConvSolverDevApplicabilityWrw_NONE;
using CPU_UnitTestConvSolverAsmBwdWrW3x3DevApplicabilityWrw_FP32 =
    CPU_UnitTestConvSolverDevApplicabilityWrw_NONE;

TEST_P(CPU_UnitTestConvSolverAsmBwdWrW3x3DevApplicabilityWrw_FP16, ConvAsmBwdWrW3x3)
{
    this->RunTest(miopen::solver::conv::ConvAsmBwdWrW3x3{});
};

TEST_P(CPU_UnitTestConvSolverAsmBwdWrW3x3DevApplicabilityWrw_FP32, ConvAsmBwdWrW3x3)
{
    this->RunTest(miopen::solver::conv::ConvAsmBwdWrW3x3{});
};
#else
using CPU_UnitTestConvSolverAsmBwdWrW3x3DevApplicabilityWrw_NONE =
    CPU_UnitTestConvSolverDevApplicabilityWrw_NONE;

TEST_P(CPU_UnitTestConvSolverAsmBwdWrW3x3DevApplicabilityWrw_NONE, ConvAsmBwdWrW3x3)
{
    this->RunTest(miopen::solver::conv::ConvAsmBwdWrW3x3{});
};
#endif

TEST_P(GPU_UnitTestConvSolverAsmBwdWrW3x3Wrw_FP16, ConvAsmBwdWrW3x3)
{
    this->RunTest(miopen::solver::conv::ConvAsmBwdWrW3x3{});
};

TEST_P(GPU_UnitTestConvSolverAsmBwdWrW3x3Wrw_FP32, ConvAsmBwdWrW3x3)
{
    this->RunTest(miopen::solver::conv::ConvAsmBwdWrW3x3{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverAsmBwdWrW3x3Wrw_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

#if WORKAROUND_SWDEV_330460
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverAsmBwdWrW3x3Wrw_FP32,
                         testing::Combine(testing::Values(GetTestParamsFp32()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverAsmBwdWrW3x3DevApplicabilityWrw_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenHalf)[0])));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverAsmBwdWrW3x3DevApplicabilityWrw_FP32,
                         testing::Combine(testing::Values(GetTestParamsFp32()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));
#else
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverAsmBwdWrW3x3Wrw_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverAsmBwdWrW3x3DevApplicabilityWrw_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));
#endif
