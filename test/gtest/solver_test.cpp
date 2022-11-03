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
#include <miopen/fusion.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/fusion/fusion_invoke_params.hpp>
#include <miopen/solver_id.hpp>
#include <serialize.hpp>
#include <fusionHost.hpp>

#include "tensor_util.hpp"
#include "get_handle.hpp"
#include <miopen/miopen.h>

struct CBATestCase
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
    miopenActivationMode_t activ_mode;
    miopenConvolutionMode_t conv_mode;
    friend std::ostream& operator<<(std::ostream& os, const CBATestCase& tc)
    {
        return os << "N: " << tc.N << " C:" << tc.C << " H:" << tc.H << " W:" << tc.W
                  << " k: " << tc.k << " y:" << tc.y << " x:" << tc.x << " pad_y:" << tc.pad_y
                  << " pad_x:" << tc.pad_x << " stride_y:" << tc.stride_y
                  << " dilation_y:" << tc.dilation_y << " activ_mode:" << tc.activ_mode
                  << " conv_mode:" << tc.conv_mode;
    }
};

struct CBAFwdSolverTest
    : public ::testing::TestWithParam<std::tuple<miopenConvFwdAlgorithm_t, CBATestCase>>
{
protected:
    void SetUp() override
    {
        std::tie(algo, cba_config) = GetParam();
        const double double_zero   = 0.0f;
        input   = tensor<float>{cba_config.N, cba_config.C, cba_config.H, cba_config.W};
        weights = tensor<float>{1, cba_config.k, cba_config.x, cba_config.y};
        input.generate(tensor_elem_gen_integer{17});
        weights.generate(tensor_elem_gen_integer{17});

        miopenCreateConvolutionDescriptor(&conv_desc);
        miopenCreateActivationDescriptor(&activ_desc);
        miopenSetActivationDescriptor(
            activ_desc, cba_config.activ_mode, double_zero, double_zero, double_zero);
        miopenInitConvolutionDescriptor(conv_desc,
                                        cba_config.conv_mode,
                                        cba_config.pad_y,
                                        cba_config.pad_x,
                                        cba_config.stride_y,
                                        cba_config.stride_x,
                                        cba_config.dilation_y,
                                        cba_config.dialtion_x);

        miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, &input.desc);

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
        bias    = tensor<float>{1, static_cast<size_t>(c), 1, 1};
        bias.generate(tensor_elem_gen_integer{17});
        std::fill(output.begin(), output.end(), 0.0f);
        std::fill(ref_out.begin(), ref_out.end(), 0.0f);
        std::fill(bias.begin(), bias.end(), 0.0f);
        int bias_mode = 1; // zero disables bias
        convHostForward(input, ref_out, weights, bias_mode, bias, conv_desc);
        activationHostInfer(cba_config.activ_mode,
                            double_zero,
                            double_zero,
                            double_zero,
                            ref_out.data,
                            ref_out.data);
        auto&& handle = get_handle();
        in_dev        = handle.Write(input.data);
        wei_dev       = handle.Write(weights.data);
        out_dev       = handle.Write(output.data);
        bias_dev      = handle.Write(bias.data);
    }
    void TearDown() override
    {
        auto&& handle = get_handle();
        output.data   = handle.Read<float>(out_dev, output.data.size());
        EXPECT_FALSE(miopen::range_zero(ref_out)) << "Cpu data is all zeros";
        EXPECT_FALSE(miopen::range_zero(output)) << "Gpu data is all zeros";
        const auto mxdiff = miopen::max_diff(output, ref_out);
        std::ignore       = mxdiff;
        auto idx          = miopen::mismatch_idx(ref_out, output, miopen::float_equal);
        EXPECT_FALSE(miopen::find_idx(ref_out, miopen::not_finite) >= 0)
            << "Non finite number found in the CPU data";
        EXPECT_FALSE(idx < miopen::range_distance(ref_out));
        miopenDestroyConvolutionDescriptor(conv_desc);
        miopenDestroyActivationDescriptor(activ_desc);
        miopenDestroyFusionPlan(fusePlanDesc);
    }
    CBATestCase cba_config;
    miopenFusionPlanDescriptor_t /*miopen::FusionPlanDescriptor*/ fusePlanDesc;
    miopenConvolutionDescriptor_t conv_desc;
    miopenActivationDescriptor_t activ_desc;
    tensor<float> input;
    tensor<float> weights;
    tensor<float> output;
    tensor<float> bias;
    tensor<float> ref_out;
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr wei_dev;
    miopen::Allocator::ManageDataPtr out_dev;
    miopen::Allocator::ManageDataPtr bias_dev;
    miopenConvFwdAlgorithm_t algo = miopenConvolutionFwdAlgoDirect;
    miopen::fusion::FusionInvokeParams plan_params;
};

TEST_P(CBAFwdSolverTest, ConvASM3x3U)
{

    auto& handle            = get_handle();
    const float alpha       = static_cast<float>(1.0f);
    const float beta        = static_cast<float>(0);
    const float activ_alpha = static_cast<float>(0.5f);
    const float activ_beta  = static_cast<float>(0.5f);
    const float activ_gamma = static_cast<float>(0.5f);

    // 1. Create a fusion plan, moved to setup()
    // miopen::FusionPlanDescriptor fusePlanDesc{miopenVerticalFusion, input.desc};

    miopen::OperatorArgs fusionArgs;
    // 2. Create the convolution, bias and activation operators
    auto convoOp =
        std::make_shared<miopen::ConvForwardOpDescriptor>(miopen::deref(conv_desc), weights.desc);
    auto biasOp  = std::make_shared<miopen::BiasFusionOpDescriptor>(bias.desc);
    auto activOp = std::make_shared<miopen::ActivFwdFusionOpDescriptor>(
        miopen::deref(activ_desc).GetMode()); // cba_config.activ_mode); //activ_desc->GetMode()

    // 3. Add the operators
    EXPECT_EQ(miopen::deref(fusePlanDesc).AddOp(convoOp), miopenStatusSuccess);
    EXPECT_EQ(miopen::deref(fusePlanDesc).SetConvAlgo(algo), miopenStatusSuccess);
    EXPECT_EQ(miopen::deref(fusePlanDesc).AddOp(biasOp), miopenStatusSuccess);
    EXPECT_EQ(miopen::deref(fusePlanDesc).AddOp(activOp), miopenStatusSuccess);

    // 4. Compile the plan, find solver here, to replace with user-specified solver?
    // EXPECT_EQ(fusePlanDesc.Compile(handle), miopenStatusSuccess);

    // 5. Set the Args for each operator, same as
    // miopenSetOpArgsConvForward
    // miopenSetOpArgsActivForward
    // miopenSetOpArgsBiasForward

    EXPECT_EQ(activOp->SetArgs(fusionArgs, &alpha, &beta, activ_alpha, activ_beta, activ_gamma),
              miopenStatusSuccess);
    EXPECT_EQ(biasOp->SetArgs(fusionArgs, &alpha, &beta, bias_dev.get()), miopenStatusSuccess);
    EXPECT_EQ(convoOp->SetArgs(fusionArgs, &alpha, &beta, wei_dev.get()), miopenStatusSuccess);

    // 6. Execute the fusion plan
    // EXPECT_EQ(fusePlanDesc.Execute(handle, xDesc, x, yDesc, y, fusionArgs),miopenStatusSuccess);

    // Setup the params
    /*
        std::vector<std::shared_ptr<miopen::FusionOpDescriptor>> params;
        for(const auto& op : miopen::deref(fusePlanDesc).op_map)
            params.push_back(op->GetArgs());

        plan_params = FusionInvokeParams{
            params, input.desc, in_dev.get(), output.desc, out_dev.get(), false};
    */

    /*
       std::vector<std::shared_ptr<miopen::fusion::FusionOpInvokeParamBase>> params;
            for(const auto& op : miopen::deref(fusePlanDesc).op_map)
                params.push_back(op->GetArgs());

         miopen::fusion::FusionInvokeParams plan_params  = miopen::fusion::FusionInvokeParams{
                params, input.desc, in_dev.get(), output.desc, out_dev.get(), false};
    */
    // create Problem. Default int bias_ = 0, is this an issue?
    miopen::solver::ConvAsm3x3U convAsm3x3Solv{};
    auto ctx = miopen::ConvolutionContext{input.desc,
                                          weights.desc,
                                          output.desc,
                                          miopen::deref(conv_desc),
                                          miopen::conv::Direction::Forward}; // bias_dev.get()
    ctx.DetectRocm();

    if(!convAsm3x3Solv.IsApplicable(ctx))
    {
        GTEST_FAIL() << "ConvAsm3x3U not for this problem";
    }

    // From solution.cpp

    const auto solution =
        convAsm3x3Solv.GetSolution(ctx, convAsm3x3Solv.GetDefaultPerformanceConfig(ctx));

    // ASSERT_TRUE(solution.succeed())<<"Failed to get solution";
    ASSERT_TRUE(solution.invoker_factory) << "Invalid solution.invoker_factory";

    // invoker to replace Execution
    EXPECT_EQ(miopenExecuteFusionPlan(&handle,
                                      fusePlanDesc,
                                      &input.desc,
                                      in_dev.get(),
                                      &output.desc,
                                      out_dev.get(),
                                      &fusionArgs),
              miopenStatusSuccess);
    /*
    using Invoker = std::function<void(const Handle&, const AnyInvokeParams& primitive_parameters)>;
    using InvokerFactory = std::function<Invoker(const std::vector<Kernel>&)>;


            //invoker(handle, invoke_ctx);

            (invoker)(handle,plan_params);  //Create plan paramters
            handle.Finish();


    */
    // EXPECT_EQ(status, miopenStatusSuccess);
}

INSTANTIATE_TEST_SUITE_P(CBAFwdTest,
                         CBAFwdSolverTest,
                         testing::Combine(testing::Values(miopenConvolutionFwdAlgoDirect),
                                          testing::Values(CBATestCase{16,
                                                                      128,
                                                                      16,
                                                                      16,
                                                                      128,
                                                                      3,
                                                                      3,
                                                                      0,
                                                                      0,
                                                                      1,
                                                                      1,
                                                                      1,
                                                                      1,
                                                                      miopenActivationRELU,
                                                                      miopenConvolution})));
