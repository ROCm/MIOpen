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

#if MIOPEN_BACKEND_HIP // equiv to hipInit(0)?

std::vector<int> strides(std::vector<int> v) 
{
    std::vector<int> v2(4);
    int d=1;
    for(int i=3; i>=0; i--) {
        v2[i] = d;
        d *= v[i];
    }
    return v2;
}

void upload(float*& pd, const std::vector<float>& v)
{
    hipMalloc(&pd, v.size()*4);
    hipMemcpy(pd, &v[0], v.size()*4, hipMemcpyHostToDevice);
}

class FusionTestApi : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // hipInit(0);

        miopenCreateWithStream(&handle, (hipStream_t)(nullptr));
        miopenSetStream(handle, nullptr);

        miopenCreateTensorDescriptor(&filter);
        miopenCreateTensorDescriptor(&bias);
        miopenCreateTensorDescriptor(&in);
        miopenCreateTensorDescriptor(&out);
       

        miopenSetTensorDescriptor(filter, miopenFloat, 4, dims_f.data(), strides(dims_f).data());
        miopenSetTensorDescriptor(bias, miopenFloat, 4, dims_b.data(), strides(dims_b).data());
        miopenSetTensorDescriptor(in, miopenFloat, 4, dims_i.data(), strides(dims_i).data());
        miopenSetTensorDescriptor(out, miopenFloat, 4, dims_o.data(), strides(dims_o).data());
        miopenCreateConvolutionDescriptor(&conv);
        miopenInitConvolutionNdDescriptor(conv,
            2, zeros.data(), ones.data(),
            ones.data(), miopenConvolution);
        miopenSetConvolutionGroupCount(conv, 1);

        miopenCreateFusionPlan(&fusion_plan, miopenVerticalFusion, in);
        miopenCreateOpConvForward(fusion_plan, &conv_op, conv, filter);
        miopenCreateOpBiasForward(fusion_plan, &bias_op, bias);
        miopenCreateOperatorArgs(&fusion_args);
       
        h_filter.resize(dims_f[0]*dims_f[1]);
        for(auto& x: h_filter)
            x = 1.0;
        upload(d_filter, h_filter);


        h_filter2.resize(dims_f[0]*dims_f[1]);
        for(auto& x: h_filter2)
            x = 2.0;
        upload(d_filter2, h_filter2);

        h_bias.resize(dims_b[0]*dims_b[1]);
        for(auto& x: h_bias)
            x = 0.0;
        upload(d_bias, h_bias);

        h_input.resize(dims_i[0]*dims_i[1]*dims_i[2]*dims_i[3]);
        for(auto& x: h_input)
            x = 1.0;
        upload(d_input, h_input);

        hipMalloc(&d_output, dims_o[0]*dims_o[1]*dims_o[2]*dims_o[3]*4);
        h_output.resize(dims_o[0]*dims_o[1]*dims_o[2]*dims_o[3]);

    }

    miopenHandle_t handle = nullptr;

    miopenFusionPlanDescriptor_t fusion_plan;
    miopenOperatorArgs_t fusion_args;
    miopenTensorDescriptor_t filter, bias, in, out;
    miopenConvolutionDescriptor_t conv;

    std::vector<int> dims_f{ 64, 64, 1, 1};
    std::vector<int> dims_i{ 1, 64, 7, 7 };
    std::vector<int> dims_o{ 1, 64, 7, 7 };
    std::vector<int> dims_b{ 64, 1, 1, 1 };
    std::vector<int> zeros{ 0, 0, 0, 0};
    std::vector<int> ones{ 1, 1, 1, 1};

    miopenFusionOpDescriptor_t conv_op, bias_op;

    float alpha = 1.0, beta = 0.0;
    float* d_filter;
    std::vector<float> h_filter;

    float* d_filter2;
    std::vector<float> h_filter2;

    float* d_bias;
    std::vector<float> h_bias;

    float* d_input;
    std::vector<float> h_input;

    float* d_output;
    std::vector<float> h_output;

};

TEST_F(FusionTestApi, TestFusionPlanCompilation)
{
    EXPECT_EQ(miopenCompileFusionPlan(handle, fusion_plan), 0);
    EXPECT_EQ(miopenSetOpArgsConvForward(fusion_args, conv_op, &alpha, &beta, d_filter), 0);
    EXPECT_EQ(miopenSetOpArgsBiasForward(fusion_args, bias_op, &alpha, &beta, d_bias), 0);
    EXPECT_EQ(miopenExecuteFusionPlan(handle, fusion_plan, 
        in, d_input, out, d_output, fusion_args), 0);
    hipMemcpy(&h_output[0], d_output, h_output.size()*4, hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
    EXPECT_EQ(h_output[0], 64);

    EXPECT_EQ(miopenSetOpArgsConvForward(fusion_args, conv_op, &alpha, &beta, d_filter2), 0);
    EXPECT_EQ(miopenSetOpArgsBiasForward(fusion_args, bias_op, &alpha, &beta, d_bias), 0);
    EXPECT_EQ(miopenExecuteFusionPlan(handle, fusion_plan, 
        in, d_input, out, d_output, fusion_args), 0);
    hipMemcpy(&h_output[0], d_output, h_output.size()*4, hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
    EXPECT_EQ(h_output[0], 128);

}


#endif
