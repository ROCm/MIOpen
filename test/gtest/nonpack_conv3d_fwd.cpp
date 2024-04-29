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
#include "nonpack_conv3d_fwd.hpp"

struct ConvNonpackFwdSolverTest3DHalf : ConvNonpackFwdSolverTest3D<half_float::half>
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
               const NonPackTestCase& conv_config,
               bool& test_skipped,
               const miopen::Scalar& alpha = {1.0},
               const miopen::Scalar& beta  = {0.0})
{
    auto&& handle = get_handle();

    Solver solv{};

    const auto tensors =
        miopen::ConvFwdTensors{inputDesc, input, wDesc, weight, outputDesc, output};

    const auto problem = miopen::conv::ProblemDescription{
        inputDesc, wDesc, outputDesc, convDesc, miopen::conv::Direction::Forward, 0, alpha, beta};
    auto ctx = miopen::ExecutionContext{};

    ctx.SetStream(&handle);

    if(!solv.IsApplicable(ctx, problem))
    {
        test_skipped = true;
        GTEST_SKIP() << solv.SolverDbId()
                     << "ConvHipImplicitGemm3DGroupFwdXdlops Not Applicable for this problem"
                     << conv_config;
    }
    const auto invoke_params =
        miopen::conv::DataInvokeParams{tensors, nullptr, 0, false, alpha, beta};

    ASSERT_TRUE(solv.IsApplicable(ctx, problem));
    auto sol = solv.GetSolution(ctx, problem, solv.GetDefaultPerformanceConfig(ctx, problem));
    ASSERT_TRUE(sol.Succeeded());
    ASSERT_TRUE(sol.invoker_factory);
    const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
    (invoker)(handle, invoke_params);
    handle.Finish();
}

TEST_P(ConvNonpackFwdSolverTest3DHalf, CKNonPackConvFwd3D)
{
    SolverFwd<miopen::solver::conv::ConvHipImplicitGemm3DGroupFwdXdlops>(input.desc,
                                                                         in_dev.get(),
                                                                         weights.desc,
                                                                         wei_dev.get(),
                                                                         output.desc,
                                                                         out_dev.get(),
                                                                         conv_desc,
                                                                         conv_config,
                                                                         test_skipped,
                                                                         miopen::Scalar(alpha_val),
                                                                         miopen::Scalar(beta_val));
}

INSTANTIATE_TEST_SUITE_P(ConvFwdTest,
                         ConvNonpackFwdSolverTest3DHalf,
                         testing::Combine(testing::Values(miopenConvolutionFwdAlgoImplicitGEMM),
                                          testing::ValuesIn(ConvTestConfigs<NonPackTestCase>()),
                                          testing::ValuesIn({1.0f}), // alpha
                                          testing::ValuesIn({0.0f}), // beta
                                          testing::Values(miopenTensorNDHWC)));
