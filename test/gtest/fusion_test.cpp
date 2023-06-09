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

        miopenCreateConvolutionDescriptor(&conv);
        miopenInitConvolutionNdDescriptor(
            conv, 2, pads.data(), ones.data(), ones.data(), miopenConvolution);
        miopenSetConvolutionGroupCount(conv, 1);

        // Prepare fusion plan.
        miopenCreateFusionPlan(&fusion_plan, miopenVerticalFusion, &h_input.desc);
        miopenCreateOpConvForward(fusion_plan, &conv_op, conv, &h_filter1.desc);
        miopenCreateOpBiasForward(fusion_plan, &bias_op, &h_bias.desc);
        miopenCreateOperatorArgs(&fusion_args);

        auto&& handle = get_handle();
        std::fill(h_input.begin(), h_input.end(), 1.0);
        // h_input.generate(tensor_elem_gen_integer{17});     // tolerance is off for 3 decimal
        // points
        d_input = handle.Write(h_input.data);

        std::fill(h_filter1.begin(), h_filter1.end(), 1.0);
        // h_filter1.generate(tensor_elem_gen_integer{17});
        d_filter1 = handle.Write(h_filter1.data);

        std::fill(h_filter2.begin(), h_filter2.end(), 2.0);
        // h_filter2.generate(tensor_elem_gen_integer{17});
        d_filter2 = handle.Write(h_filter2.data);

        std::fill(h_bias.begin(), h_bias.end(), 0.0);
        // h_bias.generate(tensor_elem_gen_integer{17});
        d_bias = handle.Write(h_bias.data);

        d_output = handle.Write(h_output.data);
    }

    void TearDown() override
    {
        if(test_skipped)
            return;

        auto&& handle = get_handle();
        EXPECT_EQ(miopenCompileFusionPlan(&handle, fusion_plan), 0);
        EXPECT_EQ(miopenSetOpArgsConvForward(fusion_args, conv_op, &alpha, &beta, d_filter1.get()),
                  0);
        EXPECT_EQ(miopenSetOpArgsBiasForward(fusion_args, bias_op, &alpha, &beta, d_bias.get()), 0);
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

        miopen::TensorDescriptor conv_output_desc =
            conv_desc.GetForwardOutputTensor(h_input.desc, h_filter1.desc, GetDataType<T>());
        ref_out1 = tensor<T>{conv_output_desc.GetLengths()};
        cpu_convolution_forward(conv_desc.GetSpatialDimension(),
                                h_input,
                                h_filter1,
                                ref_out1,
                                conv_desc.GetConvPads(),
                                conv_desc.GetConvStrides(),
                                conv_desc.GetConvDilations(),
                                conv_desc.GetGroupCount());

        EXPECT_FALSE(miopen::range_zero(ref_out1)) << "Cpu data is all zeros";
        EXPECT_FALSE(miopen::range_zero(h_output)) << "Gpu data is all zeros";
        EXPECT_TRUE(miopen::range_distance(ref_out1) == miopen::range_distance(h_output));

        const double tolerance = 80;
        double threshold       = std::numeric_limits<T>::epsilon() * tolerance;
        auto error             = miopen::rms_range(ref_out1, h_output);

        EXPECT_FALSE(miopen::find_idx(ref_out1, miopen::not_finite) >= 0)
            << "Non finite number found in the CPU data";

        EXPECT_TRUE(error < threshold)
            << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;

        // Change fusion parameters (filter), see if it still works properly.
        EXPECT_EQ(miopenSetOpArgsConvForward(fusion_args, conv_op, &alpha, &beta, d_filter2.get()),
                  0);
        EXPECT_EQ(miopenSetOpArgsBiasForward(fusion_args, bias_op, &alpha, &beta, d_bias.get()), 0);
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

        conv_output_desc =
            conv_desc.GetForwardOutputTensor(h_input.desc, h_filter2.desc, GetDataType<T>());
        ref_out2 = tensor<T>{conv_output_desc.GetLengths()};
        cpu_convolution_forward(conv_desc.GetSpatialDimension(),
                                h_input,
                                h_filter2,
                                ref_out2,
                                conv_desc.GetConvPads(),
                                conv_desc.GetConvStrides(),
                                conv_desc.GetConvDilations(),
                                conv_desc.GetGroupCount());

        EXPECT_FALSE(miopen::range_zero(ref_out2)) << "Cpu data is all zeros";
        EXPECT_FALSE(miopen::range_zero(h_output)) << "Gpu data is all zeros";
        EXPECT_TRUE(miopen::range_distance(ref_out2) == miopen::range_distance(h_output));

        error = miopen::rms_range(ref_out2, h_output);

        EXPECT_FALSE(miopen::find_idx(ref_out2, miopen::not_finite) >= 0)
            << "Non finite number found in the CPU data";

        EXPECT_TRUE(error < threshold)
            << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;

        miopenDestroyConvolutionDescriptor(conv);
        miopenDestroyFusionPlan(fusion_plan);
        miopenDestroyOperatorArgs(fusion_args);
    }

    miopenFusionPlanDescriptor_t fusion_plan;
    miopenOperatorArgs_t fusion_args;
    miopenConvolutionDescriptor_t conv;

    std::vector<int> dims_i;
    std::vector<int> dims_o;
    std::vector<int> dims_f;
    std::vector<int> dims_b;
    std::vector<int> pads;
    std::vector<int> ones;

    miopenFusionOpDescriptor_t conv_op, bias_op;

    float alpha = 1.0, beta = 0.0;

    tensor<T> h_input;
    miopen::Allocator::ManageDataPtr d_input;

    tensor<T> h_filter1;
    miopen::Allocator::ManageDataPtr d_filter1;

    tensor<T> h_filter2;
    miopen::Allocator::ManageDataPtr d_filter2;

    tensor<T> h_bias;
    miopen::Allocator::ManageDataPtr d_bias;

    tensor<T> h_output;
    miopen::Allocator::ManageDataPtr d_output;

    tensor<T> ref_out1;
    tensor<T> ref_out2;
    miopen::ConvolutionDescriptor conv_desc;

    bool test_skipped = false;
};

struct FusionTestApiFloat : FusionTestApi<float>
{
};

TEST_F(FusionTestApiFloat, TestFusionPlanCompilation)
{
    // ---- DISCLAIMER ----
    // I KEEP THIS SECTION IN CASE WE WANT TO COMPATMENTALIZE TEST CASE FROM TEAR DOWN CHECKS
    // ---- DISCLAIMER ----

    // auto&& handle = get_handle();
    // EXPECT_EQ(miopenCompileFusionPlan(&handle, fusion_plan), 0);
    // EXPECT_EQ(miopenSetOpArgsConvForward(fusion_args, conv_op, &alpha, &beta, d_filter1.get()),
    // 0); EXPECT_EQ(miopenSetOpArgsBiasForward(fusion_args, bias_op, &alpha, &beta, d_bias.get()),
    // 0); EXPECT_EQ(miopenExecuteFusionPlan(&handle,
    //                                   fusion_plan,
    //                                   &h_input.desc,
    //                                   d_input.get(),
    //                                   &h_output.desc,
    //                                   d_output.get(),
    //                                   fusion_args),
    //           0);
    // h_output.data = handle.Read<float>(d_output, h_output.data.size());
    // handle.Finish();

    // miopen::OperatorArgs* fus_args  = reinterpret_cast<miopen::OperatorArgs*>(fusion_args);
    // using ConvParam = miopen::fusion::ConvolutionOpInvokeParam;
    // using BiasParam = miopen::fusion::BiasOpInvokeParam;

    // ConvParam* conv_param = dynamic_cast<ConvParam*>(fus_args->params[0].get());
    // BiasParam* bias_param = dynamic_cast<BiasParam*>(fus_args->params[1].get());

    // ASSERT_EQ(conv_param->weights, d_filter1.get());
    // ASSERT_EQ(bias_param->bdata, d_bias.get());

    // // Change fusion parameters (filter), see if it still works properly.
    // EXPECT_EQ(miopenSetOpArgsConvForward(fusion_args, conv_op, &alpha, &beta, d_filter2.get()),
    // 0); EXPECT_EQ(miopenSetOpArgsBiasForward(fusion_args, bias_op, &alpha, &beta, d_bias.get()),
    // 0); EXPECT_EQ(miopenExecuteFusionPlan(&handle,
    //                                   fusion_plan,
    //                                   &h_input.desc,
    //                                   d_input.get(),
    //                                   &h_output.desc,
    //                                   d_output.get(),
    //                                   fusion_args),
    //           0);
    // h_output.data = handle.Read<float>(d_output, h_output.data.size());
    // handle.Finish();

    // conv_param = dynamic_cast<ConvParam*>(fus_args->params[0].get());
    // bias_param = dynamic_cast<BiasParam*>(fus_args->params[1].get());

    // ASSERT_EQ(conv_param->weights, d_filter2.get());
}

#endif
