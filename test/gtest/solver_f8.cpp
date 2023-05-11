/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#include "solver.hpp"
struct ConvFwdGemmTestFp8 : ConvFwdSolverTest<float8>
{
};
template <typename Solver>
void SolverFwd(const miopen::TensorDescriptor& inputDesc,
               ConstData_t input,
               const miopen::TensorDescriptor& wDesc,
               ConstData_t weight,
               const miopen::TensorDescriptor& outputDesc,
               Data_t output,
               const miopen::ConvolutionDescriptor& convDesc,
               const ConvTestCase& conv_config,
               bool& test_skipped)
{
    auto&& handle = get_handle();

    // const auto naive_conv_id = miopen::solver::Id{"ConvDirectNaiveConvFwd"};
    // const auto naive_solver  = naive_conv_id.GetSolver();

    Solver solv{};

    const auto tensors =
        miopen::ConvFwdTensors{inputDesc, input, wDesc, weight, outputDesc, output};
    auto ctx = miopen::ConvolutionContext{
        inputDesc, wDesc, outputDesc, convDesc, miopen::conv::Direction::Forward};
    ctx.SetStream(&handle);
    ctx.DetectRocm();
    if(!solv.IsApplicable(ctx))
    {
        test_skipped = true;
        GTEST_SKIP() << /*solv.SolverDbId() <<*/ "GemmFwd1x1_0_1 Not Applicable for this problem"
                     << conv_config;
    }

    const auto invoke_params = miopen::conv::DataInvokeParams{
        tensors, nullptr, 0, convDesc.attribute.gfx90aFp16alt.GetFwd()};

    // const auto invoker = miopen::LoadOrPrepareInvoker(
    //    handle, ctx, naive_conv_id.Value(), miopen::conv::Direction::Forward);
    // invoker(handle, invoke_params);
    // rout.data = handle.Read<Tout>(out_dev, rout.data.size());

    ASSERT_TRUE(solv.IsApplicable(ctx));
    auto sol = solv.GetSolution(ctx);
    ASSERT_TRUE(sol.Succeeded());
    ASSERT_TRUE(sol.invoker_factory);
    const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
    (invoker)(handle, invoke_params);
    handle.Finish();
}

TEST_P(ConvFwdGemmTestFp8, GemmFwdFp8)
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
code

    INSTANTIATE_TEST_SUITE_P(ConvFwdTest,
                             ConvFwdGemmTestFp8,
                             testing::Combine(testing::Values(miopenConvolutionAlgoGEMM),
                                              testing::ValuesIn(ConvTestConfigs())));
