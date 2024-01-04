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

namespace group_conv3d_wrw {

std::vector<Conv3DTestCase> ConvTestConfigs()
{ // g   n   c   d    h   w   k   z  y  x pad_x pad_y pad_z stri_x stri_y stri_z dia_x dia_y dia_z
    return {{1, 1, 4, 14, 28, 28, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {1, 1, 1, 1, 4, 4, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {1, 1, 1, 8, 8, 8, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {1, 1, 1, 8, 8, 8, 1, 2, 2, 2, 0, 0, 0, 2, 2, 2, 1, 1, 1, miopenConvolution},
            {1, 64, 5, 28, 28, 28, 9, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {2, 128, 4, 28, 28, 28, 10, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {16, 128, 32, 28, 28, 28, 32, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {8, 64, 32, 28, 28, 28, 16, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {16, 64, 16, 28, 28, 28, 32, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {3, 48, 48, 28, 28, 28, 48, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {3, 48, 39, 28, 28, 28, 12, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {5, 120, 10, 28, 28, 28, 30, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution}};
}

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
               const Conv3DTestCase& conv_config,
               bool& test_skipped)
{

    auto&& handle = get_handle();

    Solver solv{};

    const auto tensors =
        miopen::ConvWrwTensors{outputDesc, output, inputDesc, input, wDesc, weight};

    // order for bwd weights pass is dy, dw, x
    const auto problem = miopen::conv::ProblemDescription{
        outputDesc, wDesc, inputDesc, convDesc, miopen::conv::Direction::BackwardWeights};
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

} // namespace group_conv3d_wrw
using namespace group_conv3d_wrw;

TEST_P(ConvWrwSolverTest3D, CKGroupConvWrw3D)
{
    SolverWrw<miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops>(input.desc,
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
