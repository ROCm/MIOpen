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

namespace conv_f8_fwd {

std::vector<Conv3DTestCase> ConvTestConfigs()
{ // g   n   c   k   image   filter   pad   stride   dilation
    // clang-format off
    return {{1, 16, 16, 16, {1, 14, 14}, {1, 3, 3}, {0, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenConvolution},
            {1, 64, 64, 64, {1, 14, 14}, {1, 3, 3}, {0, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenConvolution},
            {1, 64, 32, 32, {1, 28, 28}, {1, 3, 3}, {0, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenConvolution},
            {2, 128, 32, 32, {1, 28, 28}, {1, 3, 3}, {0, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenConvolution},
            {32, 128, 32, 32, {1, 28, 28}, {1, 3, 3}, {0, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenConvolution},
            {5, 120, 60, 60, {1, 28, 28}, {1, 3, 3}, {0, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenConvolution}};
    // clang-format on
}

template <typename T = float>
struct ConvFwdSolverTest
    : public ::testing::TestWithParam<
          std::tuple<miopenConvFwdAlgorithm_t, Conv3DTestCase, miopenTensorLayout_t>>
{
protected:
    void SetUp() override
    {
        test_skipped = false;

        std::tie(algo, conv_config, tensor_layout) = GetParam();
        input          = tensor<T>{tensor_layout, conv_config.GetInput()};
        weights        = tensor<T>{tensor_layout, conv_config.GetWeights()};
        auto gen_value = [](auto...) {
            return prng::gen_A_to_B(static_cast<T>(-3.0), static_cast<T>(3.0));
        };
        input.generate(gen_value);
        weights.generate(gen_value);
        conv_desc = conv_config.GetConv();
        input.desc.SetCastType(miopenFloat8);
        weights.desc.SetCastType(miopenFloat8);

        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<T>{});
        output = tensor<T>{tensor_layout, output_desc.GetLengths()};
        std::fill(output.begin(), output.end(), std::numeric_limits<double>::quiet_NaN());
        auto&& handle = get_handle();
        in_dev        = handle.Write(input.data);
        wei_dev       = handle.Write(weights.data);
        out_dev       = handle.Write(output.data);
    }
    void TearDown() override
    {
        if(test_skipped)
            return;

        auto&& handle = get_handle();

        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<T>{});
        ref_out        = tensor<T>{tensor_layout, output_desc.GetLengths()};
        using FI       = Fp8Cast<T, T>;
        using FW       = Fp8Cast<T, T>;
        FI in_func     = {0, true};
        FW weight_func = {0, true};
        cpu_convolution_forward<T, T, T, decltype(conv_desc.GetConvPads()), float, FI, FW>(
            conv_desc.GetSpatialDimension(),
            input,
            weights,
            ref_out,
            conv_desc.GetConvPads(),
            conv_desc.GetConvStrides(),
            conv_desc.GetConvDilations(),
            conv_desc.GetGroupCount(),
            in_func,
            weight_func);
        output.data = handle.Read<T>(out_dev, output.data.size());
        EXPECT_FALSE(miopen::range_zero(ref_out)) << "Cpu data is all zeros";
        EXPECT_FALSE(miopen::range_zero(output)) << "Gpu data is all zeros";
        EXPECT_TRUE(miopen::range_distance(ref_out) == miopen::range_distance(output));

        const double tolerance = 80;
        double threshold       = std::numeric_limits<T>::epsilon() * tolerance;
        auto error             = miopen::rms_range(ref_out, output);

        EXPECT_FALSE(miopen::find_idx(ref_out, miopen::not_finite) >= 0)
            << "Non finite number found in the CPU data";

        EXPECT_TRUE(error < threshold)
            << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;
    }
    Conv3DTestCase conv_config;
    miopen::ConvolutionDescriptor conv_desc;
    tensor<T> input;
    tensor<T> weights;
    tensor<T> output;
    tensor<T> ref_out;
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr wei_dev;
    miopen::Allocator::ManageDataPtr out_dev;
    miopenConvFwdAlgorithm_t algo = miopenConvolutionFwdAlgoImplicitGEMM;
    bool test_skipped             = false;
    miopenTensorLayout_t tensor_layout;
};

struct GPU_ConvFwdSolver_FP8 : ConvFwdSolverTest<half_float::half>
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
               const Conv3DTestCase& conv_config,
               bool& test_skipped)
{
    auto&& handle = get_handle();

    Solver solv{};

    const auto tensors =
        miopen::ConvFwdTensors{inputDesc, input, wDesc, weight, outputDesc, output};

    const auto problem = miopen::conv::ProblemDescription{
        inputDesc, wDesc, outputDesc, convDesc, miopen::conv::Direction::Forward};
    auto ctx = miopen::ExecutionContext{};

    ctx.SetStream(&handle);

    if(!solv.IsApplicable(ctx, problem))
    {
        test_skipped = true;
        GTEST_SKIP() << solv.SolverDbId()
                     << "ConvHipImplicitGemmF16F8F16FwdXdlops Not Applicable for this problem"
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

} // namespace conv_f8_fwd
using namespace conv_f8_fwd;

TEST_P(GPU_ConvFwdSolver_FP8, CKConvF8Fwd)
{
    SolverFwd<miopen::solver::conv::ConvHipImplicitGemmF16F8F16FwdXdlops>(input.desc,
                                                                          in_dev.get(),
                                                                          weights.desc,
                                                                          wei_dev.get(),
                                                                          output.desc,
                                                                          out_dev.get(),
                                                                          conv_desc,
                                                                          conv_config,
                                                                          test_skipped);
}

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_ConvFwdSolver_FP8,
                         testing::Combine(testing::Values(miopenConvolutionFwdAlgoImplicitGEMM),
                                          testing::ValuesIn(ConvTestConfigs()),
                                          testing::Values(miopenTensorNDHWC)));
