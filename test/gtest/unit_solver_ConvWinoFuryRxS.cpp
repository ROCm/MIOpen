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

auto GetConvTestCases()
{
    using TestCase = ConvTestCaseBase;

    return std::vector{
        // clang-format off
        TestCase{1, 16, 16, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
        // clang-format on
    };
}

auto GetConvTestCasesWrw()
{
    using TestCase = ConvTestCaseBase;

    return std::vector{
        // clang-format off
        TestCase{1, 16, 5, 5, 16, 3, 3, 0, 0, 1, 1, 1, 1, miopenConvolution},
        // clang-format on
    };
}

Gpu GetSupportedDevices() { return static_cast<Gpu>(enabled<Gpu::gfx110X>::val); }

} // namespace

TEST_P(UnitTestConvSolverFwdHalf, Unit_ConvWinoFuryRxS_Fwd_Half)
{
    auto solver = miopen::solver::conv::ConvWinoFuryRxS<2, 3>{};
    this->RunTest(solver, GetSupportedDevices());
};

TEST_P(UnitTestConvSolverBwdHalf, Unit_ConvWinoFuryRxS_Bwd_Half)
{
    auto solver = miopen::solver::conv::ConvWinoFuryRxS<2, 3>{};
    this->RunTest(solver, GetSupportedDevices());
};

TEST_P(UnitTestConvSolverWrwHalf, Unit_ConvWinoFuryRxS_Wrw_Half)
{
    auto solver = miopen::solver::conv::ConvWinoFuryRxS<2, 3>{};
    this->RunTest(solver, GetSupportedDevices());
};

INSTANTIATE_TEST_SUITE_P(UnitSolverConvWinoFuryRxS,
                         UnitTestConvSolverFwdHalf,
                         testing::Combine(testing::Values(miopenConvolutionFwdAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCases()),
                                          testing::Values(miopenTensorNCHW)));

INSTANTIATE_TEST_SUITE_P(UnitSolverConvWinoFuryRxS,
                         UnitTestConvSolverBwdHalf,
                         testing::Combine(testing::Values(miopenConvolutionFwdAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCases())));

INSTANTIATE_TEST_SUITE_P(UnitSolverConvWinoFuryRxS,
                         UnitTestConvSolverWrwHalf,
                         testing::Combine(testing::Values(miopenConvolutionFwdAlgoWinograd),
                                          testing::ValuesIn(GetConvTestCasesWrw())));
