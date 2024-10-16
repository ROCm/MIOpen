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

class WA_SWDEV_271887_ScopedDisabler
{
public:
    WA_SWDEV_271887_ScopedDisabler() noexcept
    {
        prev                                   = miopen::debug::disable_wa_swdev_271887;
        miopen::debug::disable_wa_swdev_271887 = true;
    }

    ~WA_SWDEV_271887_ScopedDisabler() noexcept
    {
        if(prev)
            miopen::debug::disable_wa_swdev_271887 = prev.value();
    }

private:
    std::optional<bool> prev;
};

auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{16, 16, 16, 16}, {16, 16, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        // clang-format on
    };
}

auto GetConvTestCasesFull(miopenDataType_t datatype, miopen::conv::Direction direction)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    auto cases = std::vector<TestCase>{};

    if(datatype == miopenHalf && direction == miopen::conv::Direction::Forward)
    {
        // clang-format off
        // Regression test for https://github.com/ROCm/MIOpen/issues/894
        cases.emplace_back(TestCase{{1, 16, 7, 7}, {16, 16, 1, 1}, {0, 0}, {1, 1}, {1, 1}, miopenHalf});
        // clang-format on
    }

    return cases;
}

const auto& GetTestParams()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::All);
        p.EnableDeprecatedSolvers();
        p.Tunable(5);
        return p;
    }();
    return params;
}

} // namespace

TEST_P(GPU_UnitTestConvSolverFwd_FP16, ConvOclDirectFwd1x1)
{
    WA_SWDEV_271887_ScopedDisabler wa_swdev_271887_disabler;
    this->RunTest(miopen::solver::conv::ConvOclDirectFwd1x1{});
};

TEST_P(GPU_UnitTestConvSolverBwd_FP16, ConvOclDirectFwd1x1)
{
    WA_SWDEV_271887_ScopedDisabler wa_swdev_271887_disabler;
    this->RunTest(miopen::solver::conv::ConvOclDirectFwd1x1{});
};

TEST_P(GPU_UnitTestConvSolverFwd_BFP16, ConvOclDirectFwd1x1)
{
    WA_SWDEV_271887_ScopedDisabler wa_swdev_271887_disabler;
    this->RunTest(miopen::solver::conv::ConvOclDirectFwd1x1{});
};

TEST_P(GPU_UnitTestConvSolverBwd_BFP16, ConvOclDirectFwd1x1)
{
    WA_SWDEV_271887_ScopedDisabler wa_swdev_271887_disabler;
    this->RunTest(miopen::solver::conv::ConvOclDirectFwd1x1{});
};

TEST_P(GPU_UnitTestConvSolverFwd_FP32, ConvOclDirectFwd1x1)
{
    WA_SWDEV_271887_ScopedDisabler wa_swdev_271887_disabler;
    this->RunTest(miopen::solver::conv::ConvOclDirectFwd1x1{});
};

TEST_P(GPU_UnitTestConvSolverBwd_FP32, ConvOclDirectFwd1x1)
{
    WA_SWDEV_271887_ScopedDisabler wa_swdev_271887_disabler;
    this->RunTest(miopen::solver::conv::ConvOclDirectFwd1x1{});
};

TEST_P(CPU_UnitTestConvSolverDevApplicabilityFwd_NONE, ConvOclDirectFwd1x1)
{
    WA_SWDEV_271887_ScopedDisabler wa_swdev_271887_disabler;
    this->RunTest(miopen::solver::conv::ConvOclDirectFwd1x1{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverFwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverBwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverFwd_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverBwd_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverFwd_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverBwd_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverDevApplicabilityFwd_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));

// Full tests
// clang-format off
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverFwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCasesFull(miopenHalf, miopen::conv::Direction::Forward))));
// clang-format on
