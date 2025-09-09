/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <miopen/miopen.h>

#include "../utils.hpp"

/**
 * @brief Get the total number of elements in a tensor
 * @param tensor The tensor descriptor
 * @return Total number of elements
 */
size_t GetTensorElementCount(const miopenTensorDescriptor_t& tensor)
{
    int n_dims;
    MIOPEN_CHECK(miopenGetTensorDescriptorSize(tensor, &n_dims));

    std::vector<int> dims(n_dims);
    miopenDataType_t tensor_dt;
    MIOPEN_CHECK(miopenGetTensorDescriptor(tensor, &tensor_dt, dims.data(), nullptr));

    return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
}

/**
 * @brief Convert float to bfloat16 format
 * @param f Input float value
 * @return bfloat16 representation as unsigned short
 */
unsigned short float_to_bfloat16(float f)
{
    unsigned int i;
    std::memcpy(&i, &f, sizeof(float));
    return (i >> 16);
}

/**
 * @brief Convert bfloat16 to float format
 * @param b Input bfloat16 value as unsigned short
 * @return Float representation
 */
float bfloat16_to_float(unsigned short b)
{
    float f;
    unsigned int i = b << 16;
    std::memcpy(&f, &i, sizeof(float));
    return f;
}

int main()
{
    // Approximate bfloat16 as unsigned short for simplicity
    using bfloat16 = unsigned short;

    //==========================================================================
    // STEP 1: INITIALIZE MIOpen AND HIP RESOURCES
    //==========================================================================

    miopenHandle_t handle;
    MIOPEN_CHECK(miopenCreate(&handle));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    MIOPEN_CHECK(miopenSetStream(handle, stream));

    //==========================================================================
    // STEP 2: DECLARE DESCRIPTORS AND FUSION OBJECTS
    //==========================================================================

    miopenConvolutionMode_t mode     = miopenConvolution;
    miopenActivationMode_t activMode = miopenActivationRELU;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t weightTensor;
    miopenTensorDescriptor_t biasScaleTensor;
    miopenTensorDescriptor_t biasTensor;
    miopenTensorDescriptor_t outputTensor;

    miopenConvolutionDescriptor_t convDesc;
    miopenActivationDescriptor_t activDesc;
    miopenFusionPlanDescriptor_t fusePlanDesc;

    miopenFusionOpDescriptor_t convoOp;
    miopenFusionOpDescriptor_t biasOp;
    miopenFusionOpDescriptor_t activOp;

    miopenOperatorArgs_t fusionArgs;

    MIOPEN_CHECK(miopenCreateTensorDescriptor(&inputTensor));
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&weightTensor));
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&outputTensor));
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&biasTensor));
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&biasScaleTensor));

    MIOPEN_CHECK(miopenCreateConvolutionDescriptor(&convDesc));
    MIOPEN_CHECK(miopenCreateActivationDescriptor(&activDesc));
    MIOPEN_CHECK(miopenCreateOperatorArgs(&fusionArgs));

    //==========================================================================
    // STEP 3: DEFINE TENSOR DIMENSIONS AND CONVOLUTION PARAMETERS
    //==========================================================================

    const int spatial_dim = 2; // 2D convolution

    std::vector<int> in_spatial_lens(spatial_dim);
    std::vector<int> wei_spatial_lens(spatial_dim);
    std::vector<int> pads(spatial_dim);
    std::vector<int> strides(spatial_dim);
    std::vector<int> dilations(spatial_dim);
    std::vector<int> trans_output_pads(spatial_dim);

    const int out_c       = 16; // output channels
    const int in_c        = 8;  // input channels
    const int group_count = 4;  // group count
    const int in_n        = 4;  // batch size

    // Input spatial dimensions: [height, width]
    in_spatial_lens[0] = 20;
    in_spatial_lens[1] = 20;

    // Filter/kernel spatial dimensions: [height, width]
    wei_spatial_lens[0] = 3;
    wei_spatial_lens[1] = 3;

    // Remaining conv params
    pads[0]      = 0;
    pads[1]      = 0;
    strides[0]   = 2;
    strides[1]   = 2;
    dilations[0] = 1;
    dilations[1] = 1;

    // Compute "same" padding mode where output spatial dimensions are input_size / stride
    for(int i = 0; i < spatial_dim; ++i)
    {
        pads[i] = (in_spatial_lens[i] % strides[i] == 0)
                      ? (std::max((wei_spatial_lens[i] - strides[i]), 0))
                      : (std::max((wei_spatial_lens[i] - (in_spatial_lens[i] % strides[i])), 0));
        pads[i] /= 2;
    }

    // For ReLU: f(x) = max(0, x), these parameters don't affect the output
    double activ_alpha = 0.0;
    double activ_beta  = 0.0;
    double activ_gamma = 0.0;

    // Input tensor dimensions: [batch, channels, height, width]
    std::vector<int> in_len = {in_n, in_c, in_spatial_lens[0], in_spatial_lens[1]};

    // Weight tensor dimensions: [output_channels, input_channels_per_group, kernel_height,
    // kernel_width]
    std::vector<int> wei_len = {
        out_c, in_c / group_count, wei_spatial_lens[0], wei_spatial_lens[1]};

    int ndim = in_len.size();
    std::vector<int> out_len(ndim);

    //==========================================================================
    // STEP 4: CONFIGURE DESCRIPTORS
    //==========================================================================

    MIOPEN_CHECK(miopenInitConvolutionNdDescriptor(
        convDesc, spatial_dim, pads.data(), strides.data(), dilations.data(), mode));

    MIOPEN_CHECK(miopenSetConvolutionGroupCount(convDesc, group_count));

    MIOPEN_CHECK(
        miopenSetActivationDescriptor(activDesc, activMode, activ_alpha, activ_beta, activ_gamma));

    // Set tensor descriptors with NHWC layout (batch, height, width, channels)
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        inputTensor, miopenBFloat16, miopenTensorNHWC, in_len.data(), in_len.size()));

    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        weightTensor, miopenBFloat16, miopenTensorNHWC, wei_len.data(), wei_len.size()));

    MIOPEN_CHECK(miopenGetConvolutionNdForwardOutputDim(
        convDesc, inputTensor, weightTensor, &ndim, out_len.data()));

    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        outputTensor, miopenBFloat16, miopenTensorNHWC, out_len.data(), out_len.size()));

    // Bias tensor dimensions: [1, output_channels, 1, 1]
    // This allows broadcasting across spatial dimensions and batch
    std::vector<int> bias_len = {1, out_len[1], 1, 1};

    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        biasTensor, miopenBFloat16, miopenTensorNHWC, bias_len.data(), bias_len.size()));

    //==========================================================================
    // STEP 5: ALLOCATE MEMORY AND INITIALIZE DATA
    //==========================================================================

    size_t in_sz_elements   = GetTensorElementCount(inputTensor);
    size_t wei_sz_elements  = GetTensorElementCount(weightTensor);
    size_t out_sz_elements  = GetTensorElementCount(outputTensor);
    size_t bias_sz_elements = GetTensorElementCount(biasTensor);

    size_t in_sz_bytes   = in_sz_elements * sizeof(bfloat16);
    size_t wei_sz_bytes  = wei_sz_elements * sizeof(bfloat16);
    size_t out_sz_bytes  = out_sz_elements * sizeof(bfloat16);
    size_t bias_sz_bytes = bias_sz_elements * sizeof(bfloat16);

    // Host memory
    std::vector<bfloat16> in_host(in_sz_elements);
    std::vector<bfloat16> wei_host(wei_sz_elements);
    std::vector<bfloat16> bias_host(bias_sz_elements);
    std::vector<bfloat16> out_host(out_sz_elements, 0.f);

    // Device memory
    void *in_dev, *wei_dev, *out_dev, *bias_dev;
    HIP_CHECK(hipMalloc(&in_dev, in_sz_bytes));
    HIP_CHECK(hipMalloc(&wei_dev, wei_sz_bytes));
    HIP_CHECK(hipMalloc(&out_dev, out_sz_bytes));
    HIP_CHECK(hipMalloc(&bias_dev, bias_sz_bytes));

    for(size_t i = 0; i < in_sz_elements; i++)
    {
        float val  = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        in_host[i] = float_to_bfloat16(val);
    }

    for(size_t i = 0; i < wei_sz_elements; i++)
    {
        float val   = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        wei_host[i] = float_to_bfloat16(val);
    }

    for(size_t i = 0; i < bias_sz_elements; i++)
    {
        float val    = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        bias_host[i] = float_to_bfloat16(val);
    }

    std::cout << "=== INPUT DATA VERIFICATION ===" << std::endl;
    std::cout << "First 5 input values:" << std::endl;
    for(size_t i = 0; i < 5 && i < in_host.size(); ++i)
    {
        std::cout << "  in_host[" << i << "] = " << bfloat16_to_float(in_host[i]) << std::endl;
    }

    std::cout << "First 5 weight values:" << std::endl;
    for(size_t i = 0; i < 5 && i < wei_host.size(); ++i)
    {
        std::cout << "  wei_host[" << i << "] = " << bfloat16_to_float(wei_host[i]) << std::endl;
    }

    std::cout << "First 5 bias values:" << std::endl;
    for(size_t i = 0; i < 5 && i < bias_host.size(); ++i)
    {
        std::cout << "  bias_host[" << i << "] = " << bfloat16_to_float(bias_host[i]) << std::endl;
    }

    HIP_CHECK(hipMemcpy(in_dev, in_host.data(), in_sz_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(wei_dev, wei_host.data(), wei_sz_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(bias_dev, bias_host.data(), bias_sz_bytes, hipMemcpyHostToDevice));

    //==========================================================================
    // STEP 6: CREATE AND EXECUTE FUSION PLAN
    //==========================================================================

    // Create fusion plan for vertical fusion (sequential operation execution)
    MIOPEN_CHECK(miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, inputTensor));

    // Create fusion operations in order: conv -> bias -> activ
    MIOPEN_CHECK(miopenCreateOpConvForward(fusePlanDesc, &convoOp, convDesc, weightTensor));
    MIOPEN_CHECK(miopenCreateOpBiasForward(fusePlanDesc, &biasOp, biasTensor));
    MIOPEN_CHECK(miopenCreateOpActivationForward(fusePlanDesc, &activOp, activMode));

    // Op scaling factors
    float conv_alpha = 1.0f,
          conv_beta  = 0.0f; // alpha * conv_result + beta * existing (writes conv output)
    float bias_alpha = 1.0f, bias_beta = 1.0f; // alpha * bias_result + beta * existing (adds bias)
    float activ_op_alpha = 1.0f,
          activ_op_beta  = 0.0f; // alpha * activation_result + beta * existing (applies activation)

    MIOPEN_CHECK(miopenSetOpArgsConvForward(fusionArgs, convoOp, &conv_alpha, &conv_beta, wei_dev));
    MIOPEN_CHECK(miopenSetOpArgsBiasForward(fusionArgs, biasOp, &bias_alpha, &bias_beta, bias_dev));
    MIOPEN_CHECK(miopenSetOpArgsActivForward(fusionArgs,
                                             activOp,
                                             &activ_op_alpha,
                                             &activ_op_beta,
                                             activ_alpha,
                                             activ_beta,
                                             activ_gamma));

    MIOPEN_CHECK(miopenCompileFusionPlan(handle, fusePlanDesc));

    // No layout transformation workspace needed: Composable Kernel (the fusion provider) natively
    // supports NHWC layout
    MIOPEN_CHECK(miopenExecuteFusionPlan_v2(
        handle, fusePlanDesc, inputTensor, in_dev, outputTensor, out_dev, fusionArgs, nullptr, 0));

    HIP_CHECK(hipStreamSynchronize(stream));

    //==========================================================================
    // STEP 7: RETRIEVE AND DISPLAY RESULTS
    //==========================================================================

    HIP_CHECK(hipMemcpy(out_host.data(), out_dev, out_sz_bytes, hipMemcpyDeviceToHost));

    std::cout << "\n=== EXECUTION RESULTS ===" << std::endl;
    std::cout << "Fusion execution complete. First 5 output values:" << std::endl;
    for(int i = 0; i < 5 && i < out_host.size(); ++i)
    {
        std::cout << "  out_host[" << i << "] = " << bfloat16_to_float(out_host[i]) << std::endl;
    }

    //==========================================================================
    // STEP 8: CLEANUP RESOURCES
    //==========================================================================

    HIP_CHECK(hipFree(in_dev));
    HIP_CHECK(hipFree(wei_dev));
    HIP_CHECK(hipFree(out_dev));
    HIP_CHECK(hipFree(bias_dev));

    miopenDestroyTensorDescriptor(inputTensor);
    miopenDestroyTensorDescriptor(weightTensor);
    miopenDestroyTensorDescriptor(outputTensor);
    miopenDestroyTensorDescriptor(biasTensor);
    miopenDestroyTensorDescriptor(biasScaleTensor);

    miopenDestroyConvolutionDescriptor(convDesc);
    miopenDestroyActivationDescriptor(activDesc);
    miopenDestroyOperatorArgs(fusionArgs);

    MIOPEN_CHECK(miopenDestroy(handle));
    HIP_CHECK(hipStreamDestroy(stream));

    std::cout << "\n=== CLEANUP COMPLETE ===" << std::endl;
    std::cout << "All resources have been successfully released." << std::endl;

    return 0;
}
