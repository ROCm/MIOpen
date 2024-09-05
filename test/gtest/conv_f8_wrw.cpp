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
#include <miopen/conv/solvers.hpp>
#include <miopen/miopen.h>
#include <miopen/solver_id.hpp>
#include <serialize.hpp>
#include "tensor_util.hpp"
#include "get_handle.hpp"
#include "f8_cast_util.hpp"
#include "conv3d_test_case.hpp"

namespace conv_f8_wrw {

std::vector<Conv3DTestCase> ConvTestConfigs()
{ // g   n   c   k   image   filter   pad   stride   dilation
    // clang-format off
    return {{1, 16, 16, 16, {1, 14, 14}, {1, 3, 3}, {0, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenConvolution},
            {1, 64, 128, 128, {1, 28, 3},  {1, 3, 3}, {0, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenConvolution},
            {1, 64, 64, 64, {1, 28, 3}, {1, 3, 3}, {0, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenConvolution},
            {1, 32, 64, 64, {1, 14, 14}, {1, 3, 3}, {0, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenConvolution},
            {1, 32, 32, 32, {1, 14, 14}, {1, 3, 3}, {0, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenConvolution},
            {1, 64, 32, 32, {1, 14, 14}, {1, 3, 3}, {0, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenConvolution},
            {1, 128, 64, 64, {1, 7, 7}, {1, 3, 3}, {0, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenConvolution},
            {1, 128, 32, 32, {1, 7, 7}, {1, 3, 3}, {0, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenConvolution}};
    // clang-format on
}

template <typename T = float>
struct ConvWrwSolverTest
    : public ::testing::TestWithParam<
          std::tuple<miopenConvBwdWeightsAlgorithm_t, Conv3DTestCase, miopenTensorLayout_t>>
{
protected:
    void SetUp() override
    {
        test_skipped = false;

        std::tie(algo, conv_config, tensor_layout) = GetParam();
        input   = tensor<T>{tensor_layout, conv_config.GetInput()};
        weights = tensor<T>{tensor_layout, conv_config.GetWeights()};

        auto gen_value = [](auto...) {
            return prng::gen_A_to_B(static_cast<T>(-3.0), static_cast<T>(3.0));
        };
        input.generate(gen_value);

        std::fill(weights.begin(), weights.end(), 0);
        conv_desc = conv_config.GetConv();

        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<T>{});
        output = tensor<T>{tensor_layout, output_desc.GetLengths()};
        output.generate(gen_value);
        auto&& handle = get_handle();
        input.desc.SetCastType(miopenFloat8);
        output.desc.SetCastType(miopenBFloat8);

        in_dev  = handle.Write(input.data);
        wei_dev = handle.Write(weights.data);
        out_dev = handle.Write(output.data);
    }
    void TearDown() override
    {
        if(test_skipped)
            return;

        auto&& handle = get_handle();

        ref_wei     = tensor<T>{tensor_layout, weights.desc.GetLengths()};
        using FI    = Bf8Cast<T, T>;
        using FO    = Fp8Cast<T, T>;
        FI in_func  = {0, true};
        FO out_func = {0, true};
        cpu_convolution_backward_weight<T, T, T, decltype(conv_desc.GetConvPads()), float, FI, FO>(
            conv_desc.GetSpatialDimension(),
            input,
            ref_wei,
            output,
            conv_desc.GetConvPads(),
            conv_desc.GetConvStrides(),
            conv_desc.GetConvDilations(),
            conv_desc.GetGroupCount(),
            in_func,
            out_func);
        weights.data = handle.Read<T>(wei_dev, weights.data.size());
        EXPECT_FALSE(miopen::range_zero(ref_wei)) << "Cpu data is all zeros";
        EXPECT_FALSE(miopen::range_zero(weights)) << "Gpu data is all zeros";
        EXPECT_FALSE(miopen::find_idx(ref_wei, miopen::not_finite) >= 0)
            << "Non finite number found in the CPU data";
        EXPECT_FALSE(miopen::find_idx(weights, miopen::not_finite) >= 0)
            << "Non finite number found in the CK GPU data";
        EXPECT_TRUE(miopen::range_distance(ref_wei) == miopen::range_distance(weights));

        const double tolerance = 80;
        double threshold       = 1e-3 * tolerance;
        auto error             = miopen::rms_range(ref_wei, weights);

        EXPECT_TRUE(error < threshold)
            << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;
    }
    Conv3DTestCase conv_config;
    miopen::ConvolutionDescriptor conv_desc;
    tensor<T> input;
    tensor<T> weights;
    tensor<T> output;
    tensor<T> ref_wei;
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr wei_dev;
    miopen::Allocator::ManageDataPtr out_dev;
    miopenConvBwdWeightsAlgorithm_t algo = miopenConvolutionBwdWeightsAlgoImplicitGEMM;
    bool test_skipped                    = false;
    miopenTensorLayout_t tensor_layout;
};

struct GPU_ConvWrwSolver_FP8 : ConvWrwSolverTest<half_float::half>
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

    const auto problem = miopen::conv::ProblemDescription{
        inputDesc, wDesc, outputDesc, convDesc, miopen::conv::Direction::BackwardWeights};
    auto ctx = miopen::ExecutionContext{};

    ctx.SetStream(&handle);

    if(!solv.IsApplicable(ctx, problem))
    {
        test_skipped = true;
        GTEST_SKIP() << solv.SolverDbId()
                     << "ConvHipImplicitGemmF16F8F16WrwXdlops Not Applicable for this problem"
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
} // namespace conv_f8_wrw
using namespace conv_f8_wrw;

TEST_P(GPU_ConvWrwSolver_FP8, CKConvF8Wrw)
{
    SolverWrw<miopen::solver::conv::ConvHipImplicitGemmF16F8F16WrwXdlops>(input.desc,
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
    Full,
    GPU_ConvWrwSolver_FP8,
    testing::Combine(testing::Values(miopenConvolutionBwdWeightsAlgoImplicitGEMM),
                     testing::ValuesIn(ConvTestConfigs()),
                     testing::Values(miopenTensorNDHWC)));
