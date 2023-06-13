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
#include <miopen/tensor.hpp>
#include <miopen/handle.hpp>

#include "tensor_util.hpp"
#include "get_handle.hpp"
#include "verify.hpp"
#include "cpu_conv.hpp"
#include "cpu_bias.hpp"
#include "solver.hpp"

#if MIOPEN_BACKEND_HIP

template <typename T = float>
class FusionTestApi : public ::testing::Test
{
protected:
    void SetUp() override
    {
        test_skipped = false;

        dims_i = {1, 64, 7, 7};
        dims_o = {1, 64, 7, 7};
        dims_f = {64, 64, 1, 1};
        dims_b = {64, 1, 1, 1};
        pads   = {0, 0, 0, 0};
        ones   = {1, 1, 1, 1};

        h_input   = tensor<T>(dims_i);
        h_output  = tensor<T>(dims_o);
        h_filter1 = tensor<T>(dims_f);
        h_filter2 = tensor<T>(dims_f);
        h_bias    = tensor<T>(dims_b);
        std::fill(h_output.begin(), h_output.end(), std::numeric_limits<double>::quiet_NaN());

        activ_mode = miopenActivationRELU;

        miopenCreateConvolutionDescriptor(&convDesc);
        miopenCreateActivationDescriptor(&activDesc);

        miopenInitConvolutionNdDescriptor(
            convDesc, 2, pads.data(), ones.data(), ones.data(), miopenConvolution);

        miopenSetConvolutionGroupCount(convDesc, 1);
        miopenSetActivationDescriptor(activDesc, activ_mode, activ_alpha, activ_beta, activ_gamma);

        // Prepare fusion plan.
        miopenCreateFusionPlan(&fusion_plan, miopenVerticalFusion, &h_input.desc);
        miopenCreateOpConvForward(fusion_plan, &conv_op, convDesc, &h_filter1.desc);
        miopenCreateOpBiasForward(fusion_plan, &bias_op, &h_bias.desc);
        miopenCreateOpActivationForward(fusion_plan, &activ_op, activ_mode);
        miopenCreateOperatorArgs(&fusion_args);

        auto&& handle = get_handle();
        h_input.generate(tensor_elem_gen_integer{17});
        d_input = handle.Write(h_input.data);

        h_filter1.generate(tensor_elem_gen_integer{17});
        d_filter1 = handle.Write(h_filter1.data);

        h_filter2.generate(tensor_elem_gen_integer{17});
        d_filter2 = handle.Write(h_filter2.data);

        h_bias.generate(tensor_elem_gen_integer{17});
        d_bias = handle.Write(h_bias.data);

        d_output = handle.Write(h_output.data);
    }

    void TearDown() override
    {
        if(test_skipped)
            return;

        miopen::TensorDescriptor conv_output_desc =
            conv_desc.GetForwardOutputTensor(h_input.desc, h_filter2.desc, GetDataType<T>());
        ref_out = tensor<T>{conv_output_desc.GetLengths()};
        cpu_convolution_forward(conv_desc.GetSpatialDimension(),
                                h_input,
                                h_filter2,
                                ref_out,
                                conv_desc.GetConvPads(),
                                conv_desc.GetConvStrides(),
                                conv_desc.GetConvDilations(),
                                conv_desc.GetGroupCount());
        cpu_bias_forward(ref_out, h_bias);
        activationHostInfer(
            activ_mode, activ_gamma, activ_beta, activ_alpha, ref_out.data, ref_out.data);

        auto&& handle = get_handle();
        h_output.data = handle.Read<float>(d_output, h_output.data.size());
        EXPECT_FALSE(miopen::range_zero(ref_out)) << "Cpu data is all zeros";
        EXPECT_FALSE(miopen::range_zero(h_output)) << "Gpu data is all zeros";
        EXPECT_TRUE(miopen::range_distance(ref_out) == miopen::range_distance(h_output));

        auto error             = miopen::rms_range(ref_out, h_output);
        const double tolerance = 80;
        double threshold       = std::numeric_limits<T>::epsilon() * tolerance;

        EXPECT_FALSE(miopen::find_idx(ref_out, miopen::not_finite) >= 0)
            << "Non finite number found in the CPU data";

        EXPECT_TRUE(error < threshold)
            << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;

        miopenDestroyConvolutionDescriptor(convDesc);
        miopenDestroyFusionPlan(fusion_plan);
        miopenDestroyOperatorArgs(fusion_args);
    }

    miopenFusionPlanDescriptor_t fusion_plan;
    miopenOperatorArgs_t fusion_args;
    miopenConvolutionDescriptor_t convDesc;
    miopenActivationDescriptor_t activDesc;
    miopenActivationMode_t activ_mode;

    std::vector<int> dims_i;
    std::vector<int> dims_o;
    std::vector<int> dims_f;
    std::vector<int> dims_b;
    std::vector<int> pads;
    std::vector<int> ones;

    miopenFusionOpDescriptor_t conv_op, bias_op, activ_op;

    double alpha = 1.0, beta = 0.0;
    double activ_alpha = 0.5f;
    double activ_beta  = 0.5f;
    double activ_gamma = 0.5f;

    tensor<T> h_input;
    tensor<T> h_filter1;
    tensor<T> h_filter2;
    tensor<T> h_bias;
    tensor<T> h_output;
    tensor<T> ref_out;
    miopen::Allocator::ManageDataPtr d_input;
    miopen::Allocator::ManageDataPtr d_filter1;
    miopen::Allocator::ManageDataPtr d_filter2;
    miopen::Allocator::ManageDataPtr d_bias;
    miopen::Allocator::ManageDataPtr d_output;
    miopen::ConvolutionDescriptor conv_desc;

    bool test_skipped = false;
};

struct FusionTestApiFloat : FusionTestApi<float>
{
};

TEST_F(FusionTestApiFloat, TestFusionPlanCompilation)
{
    auto&& handle = get_handle();
    EXPECT_EQ(miopenCompileFusionPlan(&handle, fusion_plan), 0);
    EXPECT_EQ(miopenSetOpArgsConvForward(fusion_args, conv_op, &alpha, &beta, d_filter1.get()), 0);
    EXPECT_EQ(miopenSetOpArgsBiasForward(fusion_args, bias_op, &alpha, &beta, d_bias.get()), 0);
    EXPECT_EQ(miopenSetOpArgsActivForward(
                  fusion_args, activ_op, &alpha, &beta, activ_alpha, activ_beta, activ_gamma),
              0);
    EXPECT_EQ(miopenExecuteFusionPlan(&handle,
                                      fusion_plan,
                                      &h_input.desc,
                                      d_input.get(),
                                      &h_output.desc,
                                      d_output.get(),
                                      fusion_args),
              0);
    h_output.data = handle.Read<float>(d_output, h_output.data.size());
    handle.Finish();

    miopen::OperatorArgs* fus_args = reinterpret_cast<miopen::OperatorArgs*>(fusion_args);
    using ConvParam                = miopen::fusion::ConvolutionOpInvokeParam;
    using BiasParam                = miopen::fusion::BiasOpInvokeParam;

    ConvParam* conv_param = dynamic_cast<ConvParam*>(fus_args->params[0].get());
    BiasParam* bias_param = dynamic_cast<BiasParam*>(fus_args->params[1].get());

    ASSERT_EQ(conv_param->weights, d_filter1.get());
    ASSERT_EQ(bias_param->bdata, d_bias.get());

    // Change fusion parameters (filter), see if it still works properly.
    EXPECT_EQ(miopenSetOpArgsConvForward(fusion_args, conv_op, &alpha, &beta, d_filter2.get()), 0);
    EXPECT_EQ(miopenExecuteFusionPlan(&handle,
                                      fusion_plan,
                                      &h_input.desc,
                                      d_input.get(),
                                      &h_output.desc,
                                      d_output.get(),
                                      fusion_args),
              0);
    h_output.data = handle.Read<float>(d_output, h_output.data.size());
    handle.Finish();

    conv_param = dynamic_cast<ConvParam*>(fus_args->params[0].get());
    bias_param = dynamic_cast<BiasParam*>(fus_args->params[1].get());

    ASSERT_EQ(conv_param->weights, d_filter2.get());
}

#endif
