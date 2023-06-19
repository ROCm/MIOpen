/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include "solver_fwd.hpp"
struct ConvFwdFp8 : ConvFwdSolverTest<float8, float>
{
};

struct ConvFwdFp8Naive : ConvFwdSolverTest<float8, float, true>
{
};

#if 0
TEST_P(ConvFwdFp8, Gemm1x1x0x1)
{
    SolverFwd<miopen::solver::GemmFwd1x1_0_1>(input.desc,
                                              in_dev.get(),
                                              weights.desc,
                                              wei_dev.get(),
                                              output.desc,
                                              out_dev.get(),
                                              conv_desc,
                                              conv_config,
                                              test_skipped);
}
#endif
TEST_P(ConvFwdFp8Naive, Fwd)
{
    miopen::solver::ConvDirectNaiveConvFwd solv{};
    SolverFwd<miopen::solver::ConvDirectNaiveConvFwd>(solv);
}
// INSTANTIATE_TEST_SUITE_P(ConvFwdTest,
//                          ConvFwdGemmTestFp8,
//                          testing::Combine(testing::Values(miopenConvolutionAlgoGEMM),
//                                           testing::ValuesIn(ConvTestConfigs())));
// Since NaiveConv is verified against the CPU, we are conservative in the number and type
// of test cases we instantiate
INSTANTIATE_TEST_SUITE_P(ConvFwdTest,
                         ConvFwdFp8Naive,
                         testing::Combine(testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(ConvTestConfigs())));
