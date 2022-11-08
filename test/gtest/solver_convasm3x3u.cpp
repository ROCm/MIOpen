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
#include <gtest/gtest.h>
#include <miopen/miopen.h>
//#include <miopen/fusion.hpp>
#include <miopen/invoke_params.hpp>
//#include <miopen/fusion/fusion_invoke_params.hpp>
#include <miopen/solver_id.hpp>
#include <serialize.hpp>
#include <fusionHost.hpp>

#include "tensor_util.hpp"
#include "get_handle.hpp"
#include <miopen/convolution.hpp>
#include <miopen/conv/tensors.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include "cpu_conv.hpp"

struct ConvTestCase
{
    size_t N;
    size_t C;
    size_t H;
    size_t W;
    size_t k;
    size_t y;
    size_t x;
    size_t pad_x;
    size_t pad_y;
    size_t stride_x;
    size_t stride_y;
    size_t dialtion_x;
    size_t dilation_y;
    miopenConvolutionMode_t conv_mode;
    friend std::ostream& operator<<(std::ostream& os, const ConvTestCase& tc)
    {
        return os << "N: " << tc.N << " C:" << tc.C << " H:" << tc.H << " W:" << tc.W
                  << " k: " << tc.k << " y:" << tc.y << " x:" << tc.x << " pad_y:" << tc.pad_y
                  << " pad_x:" << tc.pad_x << " stride_y:" << tc.stride_y
                  << " dilation_y:" << tc.dilation_y << " conv_mode:" << tc.conv_mode;
    }
};

struct CBAFwdSolverTest
    : public ::testing::TestWithParam<std::tuple<miopenConvFwdAlgorithm_t, ConvTestCase>>
{
protected:
    void SetUp() override
    {
        std::tie(algo, cba_config) = GetParam();
        input   = tensor<float>{cba_config.N, cba_config.C, cba_config.H, cba_config.W};
        weights = tensor<float>{1, cba_config.k, cba_config.x, cba_config.y};
        input.generate(tensor_elem_gen_integer{17});
        weights.generate(tensor_elem_gen_integer{17});

        miopenCreateConvolutionDescriptor(&conv_desc);
        miopenCreateActivationDescriptor(&activ_desc);

        miopenInitConvolutionDescriptor(conv_desc,
                                        cba_config.conv_mode,
                                        cba_config.pad_y,
                                        cba_config.pad_x,
                                        cba_config.stride_y,
                                        cba_config.stride_x,
                                        cba_config.dilation_y,
                                        cba_config.dialtion_x);

        // miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, &input.desc);

        int n, c, h, w;
        miopenGetConvolutionForwardOutputDim(conv_desc, &input.desc, &weights.desc, &n, &c, &h, &w);

        output  = tensor<float>{static_cast<size_t>(n),
                               static_cast<size_t>(c),
                               static_cast<size_t>(h),
                               static_cast<size_t>(w)};
        ref_out = tensor<float>{static_cast<size_t>(n),
                                static_cast<size_t>(c),
                                static_cast<size_t>(h),
                                static_cast<size_t>(w)};

        std::fill(output.begin(), output.end(), std::numeric_limits<double>::quiet_NaN());
        std::fill(ref_out.begin(), ref_out.end(), std::numeric_limits<double>::quiet_NaN());

        // bias    = tensor<float>{1, static_cast<size_t>(c), 1, 1};
        // std::fill(bias.begin(), bias.end(), std::numeric_limits<double>::quiet_NaN());

        auto&& handle = get_handle();
        in_dev        = handle.Write(input.data);
        wei_dev       = handle.Write(weights.data);
        out_dev       = handle.Write(output.data);
    }
    void TearDown() override
    {
        auto&& handle = get_handle();

        miopen::TensorDescriptor output_desc =
            miopen::deref(conv_desc).GetForwardOutputTensor(input.desc, weights.desc, miopenFloat);
        ref_out = tensor<float>{output_desc.GetLengths()};
        // ref_out = ref_conv_fwd(input, weights, output, conv_desc);
        cpu_convolution_forward(miopen::deref(conv_desc).GetSpatialDimension(),
                                input,
                                weights,
                                ref_out,
                                miopen::deref(conv_desc).GetConvPads(),
                                miopen::deref(conv_desc).GetConvStrides(),
                                miopen::deref(conv_desc).GetConvDilations(),
                                miopen::deref(conv_desc).GetGroupCount());

        output.data = handle.Read<float>(out_dev, output.data.size());
        EXPECT_FALSE(miopen::range_zero(ref_out)) << "Cpu data is all zeros";
        EXPECT_FALSE(miopen::range_zero(output)) << "Gpu data is all zeros";
        EXPECT_TRUE(miopen::range_distance(ref_out) == miopen::range_distance(output));

        // const auto mxdiff = miopen::max_diff(output, ref_out);
        // std::ignore       = mxdiff;
        // auto idx          = miopen::mismatch_idx(ref_out, output, miopen::float_equal);
        // EXPECT_FALSE(idx < miopen::range_distance(ref_out));

        const double tolerance = 80;
        double threshold       = std::numeric_limits<float>::epsilon() * tolerance;
        auto error             = miopen::rms_range(ref_out, output);

        EXPECT_FALSE(miopen::find_idx(ref_out, miopen::not_finite) >= 0)
            << "Non finite number found in the CPU data";

        EXPECT_TRUE(error < threshold)
            << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;

        miopenDestroyConvolutionDescriptor(conv_desc);

        // miopenDestroyFusionPlan(fusePlanDesc);
    }
    ConvTestCase cba_config;
    miopenConvolutionDescriptor_t conv_desc;
    tensor<float> input;
    tensor<float> weights;
    tensor<float> output;
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr wei_dev;
    miopen::Allocator::ManageDataPtr out_dev;

    // miopenFusionPlanDescriptor_t /*miopen::FusionPlanDescriptor*/ fusePlanDesc;
    miopenActivationDescriptor_t activ_desc;

    tensor<float> ref_out;

    miopenConvFwdAlgorithm_t algo = miopenConvolutionFwdAlgoDirect;

    // Remove
    // tensor<float> bias;
    // miopen::Allocator::ManageDataPtr bias_dev;
    miopen::fusion::FusionInvokeParams plan_params;
};

TEST_P(CBAFwdSolverTest, ConvASM3x3UFwd)
{
    auto&& handle = get_handle();

    // const auto naive_conv_id = miopen::solver::Id{"ConvDirectNaiveConvFwd"};
    // const auto naive_solver  = naive_conv_id.GetSolver();

    miopen::solver::ConvAsm3x3U convAsm3x3Solv{};

    const auto tensors = miopen::ConvFwdTensors{
        input.desc, in_dev.get(), weights.desc, wei_dev.get(), output.desc, out_dev.get()};

    auto ctx = miopen::ConvolutionContext{input.desc,
                                          weights.desc,
                                          output.desc,
                                          miopen::deref(conv_desc),
                                          miopen::conv::Direction::Forward};

    ctx.SetStream(&handle);
    ctx.DetectRocm();

    if(!convAsm3x3Solv.IsApplicable(ctx))
    {
        GTEST_FAIL() << convAsm3x3Solv.SolverDbId() << "ConvAsm3x3U not for this problem";
    }

    const auto invoke_params = miopen::conv::DataInvokeParams{
        tensors, nullptr, 0, miopen::deref(conv_desc).attribute.gfx90aFp16alt.GetFwd()};

    // const auto invoker = miopen::LoadOrPrepareInvoker(
    //    handle, ctx, naive_conv_id.Value(), miopen::conv::Direction::Forward);
    // invoker(handle, invoke_params);
    // rout.data = handle.Read<Tout>(out_dev, rout.data.size());

    ASSERT_TRUE(convAsm3x3Solv.IsApplicable(ctx));
    auto sol = convAsm3x3Solv.GetSolution(ctx, convAsm3x3Solv.GetDefaultPerformanceConfig(ctx));
    ASSERT_TRUE(sol.Succeeded());
    ASSERT_TRUE(sol.invoker_factory);
    const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
    (invoker)(handle, invoke_params);
    handle.Finish();
}

INSTANTIATE_TEST_SUITE_P(
    CBAFwdTest,
    CBAFwdSolverTest,
    testing::Combine(testing::Values(miopenConvolutionFwdAlgoDirect),
                     testing::Values(ConvTestCase{
                         16, 128, 16, 16, 128, 3, 3, 0, 0, 1, 1, 1, 1, miopenConvolution})));
