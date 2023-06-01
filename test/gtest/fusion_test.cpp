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

#if MIOPEN_BACKEND_HIP

std::vector<int> strides(std::vector<int> dims)
{
    std::vector<int> strides_array(4);
    int d = 1;
    for(int i = 3; i >= 0; --i)
    {
        strides_array[i] = d;
        d *= dims[i];
    }
    return strides_array;
}

class FusionTestApi : public ::testing::Test
{
protected:
    void SetUp() override
    {
        test_skipped = false;

        dims_f = {64, 64, 1, 1};
        dims_i = {1, 64, 7, 7};
        dims_o = {1, 64, 7, 7};
        dims_b = {64, 1, 1, 1};
        zeros = {0, 0, 0, 0};
        ones = {1, 1, 1, 1};

        h_filter1 = tensor<float>{dims_f, strides(dims_f)};
        h_filter2 = tensor<float>{dims_f, strides(dims_f)};
        h_bias    = tensor<float>{dims_b, strides(dims_b)};
        h_input   = tensor<float>{dims_i, strides(dims_i)};
        h_output  = tensor<float>{dims_o, strides(dims_o)};

        miopenCreateConvolutionDescriptor(&conv);
        miopenInitConvolutionNdDescriptor(
            conv, 2, zeros.data(), ones.data(), ones.data(), miopenConvolution);
        miopenSetConvolutionGroupCount(conv, 1);

        // Prepare fusion plan.
        miopenCreateFusionPlan(&fusion_plan, miopenVerticalFusion, &h_input.desc);
        miopenCreateOpConvForward(fusion_plan, &conv_op, conv, &h_filter1.desc);
        miopenCreateOpBiasForward(fusion_plan, &bias_op, &h_bias.desc);
        miopenCreateOperatorArgs(&fusion_args);

        auto&& handle = get_handle();
        for(auto& x : h_filter1.data)
            x = 1.0;
        d_filter1 = handle.Write(h_filter1.data);

        for(auto& x : h_filter2.data)
            x = 2.0;
        d_filter2 = handle.Write(h_filter2.data);

        for(auto& x : h_bias.data)
            x = 0.0;
        d_bias = handle.Write(h_bias.data);

        for(auto& x : h_input.data)
            x = 1.0;
        d_input = handle.Write(h_input.data);

        d_output = handle.Write(h_output.data); // not sure
    }

    // void TearDown() override
    // {
    //     if(test_skipped)
    //         return;

    //     auto&& handle = get_handle();
    //     h_output.data = handle.Read<float>(d_output, h_output.data.size());
    //     ref_out = tensor<float>{h_output.desc.GetLengths()};

    //     EXPECT_FALSE(miopen::range_zero(ref_out)) << "Cpu data is all zeros";
    //     EXPECT_FALSE(miopen::range_zero(h_output)) << "Gpu data is all zeros";
    //     EXPECT_TRUE(miopen::range_distance(ref_out) == miopen::range_distance(h_output));

    //     const double tolerance = 80;
    //     double threshold       = std::numeric_limits<float>::epsilon() * tolerance;
    //     auto error             = miopen::rms_range(ref_out, h_output);

    //     EXPECT_FALSE(miopen::find_idx(ref_out, miopen::not_finite) >= 0)
    //         << "Non finite number found in the CPU data";

    //     EXPECT_TRUE(error < threshold)
    //         << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;
    // }
    
    miopenFusionPlanDescriptor_t fusion_plan;
    miopenOperatorArgs_t fusion_args;
    miopenConvolutionDescriptor_t conv;

    std::vector<int> dims_f;
    std::vector<int> dims_i;
    std::vector<int> dims_o;
    std::vector<int> dims_b;
    std::vector<int> zeros;
    std::vector<int> ones;

    miopenFusionOpDescriptor_t conv_op, bias_op;

    float alpha = 1.0, beta = 0.0;

    tensor<float> h_filter1;
    miopen::Allocator::ManageDataPtr d_filter1;

    tensor<float> h_filter2;
    miopen::Allocator::ManageDataPtr d_filter2;

    tensor<float> h_bias;
    miopen::Allocator::ManageDataPtr d_bias;

    tensor<float> h_input;
    miopen::Allocator::ManageDataPtr d_input;

    tensor<float> h_output;
    miopen::Allocator::ManageDataPtr d_output;

    tensor<float> ref_out;
    bool test_skipped = false;
};

TEST_F(FusionTestApi, TestFusionPlanCompilation)
{
    auto&& handle = get_handle();
    EXPECT_EQ(miopenCompileFusionPlan(&handle, fusion_plan), 0);
    EXPECT_EQ(miopenSetOpArgsConvForward(fusion_args, conv_op, &alpha, &beta, d_filter1.get()), 0);
    EXPECT_EQ(miopenSetOpArgsBiasForward(fusion_args, bias_op, &alpha, &beta, d_bias.get()), 0);
    EXPECT_EQ(miopenExecuteFusionPlan(&handle, fusion_plan, &h_input.desc, d_input.get(), &h_output.desc, d_output.get(), fusion_args),
              0);
    hipMemcpy(&h_output.data[0], d_output.get(), h_output.data.size() * 4, hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
    EXPECT_EQ(h_output.data[0], 64);

    // Change fusion parameters (filter), see if it still works properly.
    EXPECT_EQ(miopenSetOpArgsConvForward(fusion_args, conv_op, &alpha, &beta, d_filter2.get()), 0);
    EXPECT_EQ(miopenSetOpArgsBiasForward(fusion_args, bias_op, &alpha, &beta, d_bias.get()), 0);
    EXPECT_EQ(miopenExecuteFusionPlan(&handle, fusion_plan, &h_input.desc, d_input.get(), &h_output.desc, d_output.get(), fusion_args),
              0);
    hipMemcpy(&h_output.data[0], d_output.get(), h_output.data.size() * 4, hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
    EXPECT_EQ(h_output.data[0], 128);
}

#endif
