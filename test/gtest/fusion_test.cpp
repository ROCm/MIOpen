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

#if MIOPEN_BACKEND_HIP // equiv to hipInit(0)?

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

void upload(float*& pd, const std::vector<float>& v)
{
    hipMalloc(&pd, v.size() * 4);
    hipMemcpy(pd, &v[0], v.size() * 4, hipMemcpyHostToDevice);
}

class FusionTestApi : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // hipInit(0);

        miopenCreateConvolutionDescriptor(&conv);
        miopenInitConvolutionNdDescriptor(
            conv, 2, zeros.data(), ones.data(), ones.data(), miopenConvolution);
        miopenSetConvolutionGroupCount(conv, 1);

        // Prepare fusion plan.
        miopenCreateFusionPlan(&fusion_plan, miopenVerticalFusion, &in);
        miopenCreateOpConvForward(fusion_plan, &conv_op, conv, &filter);
        miopenCreateOpBiasForward(fusion_plan, &bias_op, &bias);
        miopenCreateOperatorArgs(&fusion_args);

        for(auto& x : h_filter1)
            x = 1.0;
        upload(d_filter1, h_filter1);
        // d_filter1 = handle.Write(h_filter1.data());

        for(auto& x : h_filter2)
            x = 2.0;
        upload(d_filter2, h_filter2);

        for(auto& x : h_bias)
            x = 0.0;
        upload(d_bias, h_bias);

        for(auto& x : h_input)
            x = 1.0;
        upload(d_input, h_input);

        hipMalloc(&d_output, dims_o[0] * dims_o[1] * dims_o[2] * dims_o[3] * 4);
    }
    
    miopen::Handle handle{nullptr};

    miopenFusionPlanDescriptor_t fusion_plan;
    miopenOperatorArgs_t fusion_args;
    miopenConvolutionDescriptor_t conv;

    std::vector<int> dims_f{64, 64, 1, 1};
    std::vector<int> dims_i{1, 64, 7, 7};
    std::vector<int> dims_o{1, 64, 7, 7};
    std::vector<int> dims_b{64, 1, 1, 1};
    std::vector<int> zeros{0, 0, 0, 0};
    std::vector<int> ones{1, 1, 1, 1};

    miopen::TensorDescriptor filter = miopen::TensorDescriptor(miopenFloat, dims_f, strides(dims_f));
    miopen::TensorDescriptor bias   = miopen::TensorDescriptor(miopenFloat, dims_b, strides(dims_b));
    miopen::TensorDescriptor in     = miopen::TensorDescriptor(miopenFloat, dims_i, strides(dims_i));
    miopen::TensorDescriptor out    = miopen::TensorDescriptor(miopenFloat, dims_o, strides(dims_o));

    miopenFusionOpDescriptor_t conv_op, bias_op;

    float alpha = 1.0, beta = 0.0;
    float* d_filter1;
    std::vector<float> h_filter1 = std::vector<float>(dims_f[0] * dims_f[1]);

    float* d_filter2;
    std::vector<float> h_filter2 = std::vector<float>(dims_f[0] * dims_f[1]);

    float* d_bias;
    std::vector<float> h_bias = std::vector<float>(dims_b[0] * dims_b[1]);

    float* d_input;
    std::vector<float> h_input = std::vector<float>(dims_i[0] * dims_i[1] * dims_i[2] * dims_i[3]);

    float* d_output;
    std::vector<float> h_output = std::vector<float>(dims_o[0] * dims_o[1] * dims_o[2] * dims_o[3]);
};

TEST_F(FusionTestApi, TestFusionPlanCompilation)
{
    EXPECT_EQ(miopenCompileFusionPlan(&handle, fusion_plan), 0);
    EXPECT_EQ(miopenSetOpArgsConvForward(fusion_args, conv_op, &alpha, &beta, d_filter1), 0);
    EXPECT_EQ(miopenSetOpArgsBiasForward(fusion_args, bias_op, &alpha, &beta, d_bias), 0);
    EXPECT_EQ(miopenExecuteFusionPlan(&handle, fusion_plan, &in, d_input, &out, d_output, fusion_args),
              0);
    hipMemcpy(&h_output[0], d_output, h_output.size() * 4, hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
    EXPECT_EQ(h_output[0], 64);

    // Change fusion parameters (filter), see if it still works properly.
    EXPECT_EQ(miopenSetOpArgsConvForward(fusion_args, conv_op, &alpha, &beta, d_filter2), 0);
    EXPECT_EQ(miopenSetOpArgsBiasForward(fusion_args, bias_op, &alpha, &beta, d_bias), 0);
    EXPECT_EQ(miopenExecuteFusionPlan(&handle, fusion_plan, &in, d_input, &out, d_output, fusion_args),
              0);
    hipMemcpy(&h_output[0], d_output, h_output.size() * 4, hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
    EXPECT_EQ(h_output[0], 128);
}

#endif
