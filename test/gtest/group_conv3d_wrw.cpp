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
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/solver_id.hpp>
#include <serialize.hpp>

#include "tensor_util.hpp"
#include "get_handle.hpp"
#include "group_conv3d_wrw.hpp"

struct ConvWrwSolverTest3D : ConvWrwSolverTest<float>
{
};

template <typename Solver>
void SolverWrw(const miopen::TensorDescriptor& inputDesc,
               ConstData_t input, // x
               const miopen::TensorDescriptor& wDesc,
               Data_t weight, // w
               const miopen::TensorDescriptor& outputDesc,
               ConstData_t output, // dy
               const miopen::ConvolutionDescriptor& convDesc,
               const ConvTestCase& conv_config,
               bool& test_skipped)
{

    auto&& handle = get_handle();

    Solver solv{};

    const auto tensors =
        miopen::ConvWrwTensors{outputDesc, output, inputDesc, input, wDesc, weight};

    const auto problem = miopen::conv::ProblemDescription{
        inputDesc, wDesc, outputDesc, convDesc, miopen::conv::Direction::BackwardWeights};
    auto ctx = miopen::ExecutionContext{};

    ctx.SetStream(&handle);

    if(!solv.IsApplicable(ctx, problem))
    {
        test_skipped = true;
        GTEST_SKIP() << solv.SolverDbId()
                     << "ConvHipImplicitGemm3DGroupWrwXdlops Not Applicable for this problem"
                     << conv_config;
    }
    const auto invoke_params = miopen::conv::WrWInvokeParams{tensors, nullptr, 0, false};
    ASSERT_TRUE(solv.IsApplicable(ctx, problem));
    auto sol = solv.GetSolution(ctx, problem, solv.GetDefaultPerformanceConfig(ctx, problem));
    ASSERT_TRUE(sol.Succeeded());
    ASSERT_TRUE(sol.invoker_factory);
    const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
    (invoker)(handle, invoke_params);
    handle.Finish();
}

TEST_P(ConvWrwSolverTest3D, CKGroupConvWrw3D)
{
    SolverWrw<miopen::solver::ConvHipImplicitGemm3DGroupWrwXdlops>(input.desc,
                                                                   in_dev.get(),
                                                                   weights.desc,
                                                                   wei_dev.get(),
                                                                   output.desc,
                                                                   out_dev.get(),
                                                                   conv_desc,
                                                                   conv_config,
                                                                   test_skipped);
}

INSTANTIATE_TEST_SUITE_P(
    ConvWrwTest,
    ConvWrwSolverTest3D,
    testing::Combine(testing::Values(miopenConvolutionBwdWeightsAlgoImplicitGEMM),
                     testing::ValuesIn(ConvTestConfigs()),
                     testing::Values(miopenTensorNDHWC)));
