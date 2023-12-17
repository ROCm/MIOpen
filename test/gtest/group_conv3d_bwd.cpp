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
#include "group_conv3d_bwd.hpp"

namespace group_conv3d_bwd {

std::vector<Conv3DTestCase> ConvTestConfigs()
{ // g    n   c   d    h   w   k   z  y  x pad_x pad_y pad_z stri_x stri_y stri_z dia_x dia_y dia_z
    return {{1, 128, 64, 14, 28, 28, 64, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {1, 64, 32, 28, 28, 28, 32, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {32, 128, 32, 28, 28, 28, 32, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {16, 128, 16, 28, 28, 28, 16, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {8, 128, 8, 28, 28, 28, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {4, 128, 4, 28, 28, 28, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {2, 128, 2, 28, 28, 28, 2, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution}};
}

struct ConvBwdSolverTest3D : ConvBwdSolverTest<float>
{
};

template <typename Solver>
void SolverBwd(const miopen::TensorDescriptor& inputDesc,
               Data_t input,
               const miopen::TensorDescriptor& wDesc,
               ConstData_t weight,
               const miopen::TensorDescriptor& outputDesc,
               ConstData_t output,
               const miopen::ConvolutionDescriptor& convDesc,
               const Conv3DTestCase& conv_config,
               bool& test_skipped)
{
    auto&& handle = get_handle();

    Solver solv{};

    const auto tensors =
        miopen::ConvBwdTensors{outputDesc, output, wDesc, weight, inputDesc, input};

    const auto problem = miopen::conv::ProblemDescription{
        inputDesc, wDesc, outputDesc, convDesc, miopen::conv::Direction::BackwardData};
    auto ctx = miopen::ExecutionContext{};

    ctx.SetStream(&handle);

    if(!solv.IsApplicable(ctx, problem))
    {
        test_skipped = true;
        GTEST_SKIP() << solv.SolverDbId()
                     << "ConvHipImplicitGemm3DGroupBwdXdlops Not Applicable for this problem"
                     << conv_config;
    }
    const auto invoke_params = miopen::conv::DataInvokeParams{tensors, nullptr, 0, false};

    ASSERT_TRUE(solv.IsApplicable(ctx, problem));
    auto sol = solv.GetSolution(ctx, problem, solv.GetDefaultPerformanceConfig(ctx, problem));
    ASSERT_TRUE(sol.Succeeded());
    ASSERT_TRUE(sol.invoker_factory);
    const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
    (invoker)(handle, invoke_params);
    handle.Finish();
}

} // namespace group_conv3d_bwd
using namespace group_conv3d_bwd;

TEST_P(ConvBwdSolverTest3D, CKGroupConvBwd3D)
{
    SolverBwd<miopen::solver::conv::ConvHipImplicitGemm3DGroupBwdXdlops>(input.desc,
                                                                         in_dev.get(),
                                                                         weights.desc,
                                                                         wei_dev.get(),
                                                                         output.desc,
                                                                         out_dev.get(),
                                                                         conv_desc,
                                                                         conv_config,
                                                                         test_skipped);
}

INSTANTIATE_TEST_SUITE_P(ConvBwdTest,
                         ConvBwdSolverTest3D,
                         testing::Combine(testing::Values(miopenConvolutionBwdDataAlgoImplicitGEMM),
                                          testing::ValuesIn(ConvTestConfigs()),
                                          testing::Values(miopenTensorNDHWC)));
