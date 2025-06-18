#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <miopen/miopen.h>

#include "utils.hpp"



size_t GetTensorElementCount(const miopenTensorDescriptor_t& tensor)
{
    int n_dims;
    MIOPEN_CHECK(miopenGetTensorDescriptorSize(tensor, &n_dims));

    std::vector<int> dims(n_dims);
    miopenDataType_t tensor_dt; 
    MIOPEN_CHECK(miopenGetTensorDescriptor(tensor, &tensor_dt, dims.data(), nullptr));

    return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
}

// Function to convert a float to a bfloat16 (represented as unsigned short)
unsigned short float_to_bfloat16(float f) {
    unsigned int i;
    std::memcpy(&i, &f, sizeof(float));
    return (i >> 16);
}

// Function to convert a bfloat16 (represented as unsigned short) back to a float
float bfloat16_to_float(unsigned short b) {
    float f;
    unsigned int i = b << 16;
    std::memcpy(&f, &i, sizeof(float));
    return f;
}

auto make_unique_tensor_desc() {
    miopenTensorDescriptor_t desc;
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&desc));
    auto deleter = [](miopenTensorDescriptor_t* ptr) {
        if (ptr) MIOPEN_CHECK(miopenDestroyTensorDescriptor(*ptr));
        delete ptr;
    };
    return std::unique_ptr<miopenTensorDescriptor_t, decltype(deleter)>(new miopenTensorDescriptor_t(desc), deleter);
}

auto make_unique_conv_desc() {
    miopenConvolutionDescriptor_t desc;
    MIOPEN_CHECK(miopenCreateConvolutionDescriptor(&desc));
    auto deleter = [](miopenConvolutionDescriptor_t* ptr) {
        if (ptr) MIOPEN_CHECK(miopenDestroyConvolutionDescriptor(*ptr));
        delete ptr;
    };
    return std::unique_ptr<miopenConvolutionDescriptor_t, decltype(deleter)>(new miopenConvolutionDescriptor_t(desc), deleter);
}

auto make_unique_act_desc() {
    miopenActivationDescriptor_t desc;
    MIOPEN_CHECK(miopenCreateActivationDescriptor(&desc));
    auto deleter = [](miopenActivationDescriptor_t* ptr) {
        if (ptr) MIOPEN_CHECK(miopenDestroyActivationDescriptor(*ptr));
        delete ptr;
    };
    return std::unique_ptr<miopenActivationDescriptor_t, decltype(deleter)>(new miopenActivationDescriptor_t(desc), deleter);
}

auto make_unique_fusion_plan_desc() {
    miopenFusionPlanDescriptor_t desc;
    auto deleter = [](miopenFusionPlanDescriptor_t* ptr) {
        if (ptr) MIOPEN_CHECK(miopenDestroyFusionPlan(*ptr));
        delete ptr;
    };
    return std::unique_ptr<miopenFusionPlanDescriptor_t, decltype(deleter)>(new miopenFusionPlanDescriptor_t(), deleter);
}

auto make_unique_fusion_args() {
    miopenOperatorArgs_t args;
    MIOPEN_CHECK(miopenCreateOperatorArgs(&args));
    auto deleter = [](miopenOperatorArgs_t* ptr) {
        if (ptr) MIOPEN_CHECK(miopenDestroyOperatorArgs(*ptr));
        delete ptr;
    };
    return std::unique_ptr<miopenOperatorArgs_t, decltype(deleter)>(new miopenOperatorArgs_t(args), deleter);
}


int main() {

    /* INITIALIZE VARIABLES AND SET DESCRIPTORS */
    
    using bfloat16 = unsigned short;

    miopenHandle_t handle;
    MIOPEN_CHECK(miopenCreate(&handle));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    MIOPEN_CHECK(miopenSetStream(handle, stream));

    miopenConvolutionMode_t mode = miopenConvolution;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t weightTensor;
    miopenTensorDescriptor_t biasScaleTensor;
    miopenTensorDescriptor_t biasTensor;
    miopenTensorDescriptor_t outputTensor;

    miopenActivationDescriptor_t activDesc;
    miopenConvolutionDescriptor_t convDesc;

    miopenOperatorArgs_t fusionArgs;

    // miopenActivationMode_t activMode;

    miopenFusionPlanDescriptor_t fusePlanDesc;

    miopenCreateTensorDescriptor(&inputTensor);
    miopenCreateTensorDescriptor(&weightTensor);
    miopenCreateTensorDescriptor(&outputTensor);
    miopenCreateTensorDescriptor(&biasTensor);
    miopenCreateTensorDescriptor(&biasScaleTensor);
    
    miopenCreateConvolutionDescriptor(&convDesc);
    miopenCreateActivationDescriptor(&activDesc);
    miopenCreateOperatorArgs(&fusionArgs);

    int spatial_dim = 2; // For 2D convolution

    std::vector<int> in_spatial_lens(spatial_dim);
    std::vector<int> wei_spatial_lens(spatial_dim);
    std::vector<int> pads(spatial_dim);
    std::vector<int> strides(spatial_dim);
    std::vector<int> dilations(spatial_dim);
    std::vector<int> trans_output_pads(spatial_dim);

    const int out_c = 16;
    const int in_c = 8;
    const int group_count = 4;
    const int in_n = 4;

    in_spatial_lens[0]  = 20;
    in_spatial_lens[1]  = 20;
    wei_spatial_lens[0] = 3;
    wei_spatial_lens[1] = 3;
    pads[0]             = 0;
    pads[1]             = 0;
    strides[0]          = 2;
    strides[1]          = 2;
    dilations[0]        = 1;
    dilations[1]        = 1;

    if(group_count > 1)
    {
        if(in_c % group_count != 0 || out_c % group_count != 0 || group_count > in_c ||
           group_count > out_c)
        {
            printf("Invalid group number\n");
            exit(0); // NOLINT (concurrency-mt-unsafe)
        }
    }

    for(int i = 0; i < spatial_dim; ++i)
    {
        pads[i] =
            (in_spatial_lens[i] % strides[i] == 0)
                ? (std::max((wei_spatial_lens[i] - strides[i]), 0))
                : (std::max((wei_spatial_lens[i] - (in_spatial_lens[i] % strides[i])), 0));
        pads[i] /= 2;
    }

    // after these should set conv descriptor with default pmode
    MIOPEN_CHECK(miopenCreateConvolutionDescriptor(&convDesc));
    MIOPEN_CHECK(miopenInitConvolutionNdDescriptor(
        convDesc, spatial_dim, pads.data(), strides.data(), dilations.data(), mode));

    MIOPEN_CHECK(miopenSetConvolutionGroupCount(convDesc, group_count));
    
    miopenActivationMode_t activMode = miopenActivationCLIPPEDRELU;
    double activ_alpha = 1.0;
    double activ_beta  = 1.0;
    double activ_gamma = 0.0;
    // activMode         = static_cast<miopenActivationMode_t>(7);
    MIOPEN_CHECK(miopenSetActivationDescriptor(activDesc, activMode, activ_alpha, activ_beta, activ_gamma));

    std::vector<int> in_len = {in_n, in_c, in_spatial_lens[0], in_spatial_lens[1]};
    std::vector<int> wei_len = {out_c, in_c / group_count, wei_spatial_lens[0],  wei_spatial_lens[0]}; // K_out = 16, R_wei = 3, S_wei = 3

    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(inputTensor, miopenBFloat16, miopenTensorNHWC, in_len.data(), in_len.size()));

    MIOPEN_CHECK(miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, inputTensor));

    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(weightTensor, miopenBFloat16, miopenTensorNHWC, wei_len.data(), wei_len.size()));

    int ndim = 4;
    std::vector<int> out_len(ndim);

    MIOPEN_CHECK(miopenGetConvolutionNdForwardOutputDim(
        convDesc, inputTensor, weightTensor, &ndim, out_len.data()));

    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(outputTensor, miopenBFloat16, miopenTensorNHWC, out_len.data(), out_len.size()));

    std::vector<int> bias_len = {1, out_len[1], 1, 1};
    
    MIOPEN_CHECK(miopenSetTensorDescriptor(biasTensor, miopenBFloat16, bias_len.size(), bias_len.data(), nullptr));

    

    /* BUFFER ALLLOCATION */

    size_t in_sz_elements   = GetTensorElementCount(inputTensor);
    size_t wei_sz_elements  = GetTensorElementCount(weightTensor);
    size_t out_sz_elements  = GetTensorElementCount(outputTensor);
    size_t bias_sz_elements = GetTensorElementCount(biasTensor);

    size_t in_sz_bytes = in_sz_elements * sizeof(bfloat16);
    size_t wei_sz_bytes = wei_sz_elements * sizeof(bfloat16);
    size_t out_sz_bytes = out_sz_elements * sizeof(bfloat16);
    size_t bias_sz_bytes = bias_sz_elements * sizeof(bfloat16);

    std::vector<bfloat16> in_host(in_sz_elements);
    std::vector<bfloat16> wei_host(wei_sz_elements);
    std::vector<bfloat16> bias_host(bias_sz_elements);
    std::vector<bfloat16> out_host(out_sz_elements, 0.f); 

    void *in_dev, *wei_dev, *out_dev, *bias_dev;
    HIP_CHECK(hipMalloc(&in_dev, in_sz_bytes));
    HIP_CHECK(hipMalloc(&wei_dev, wei_sz_bytes));
    HIP_CHECK(hipMalloc(&out_dev, out_sz_bytes));
    HIP_CHECK(hipMalloc(&bias_dev, bias_sz_bytes));

    for(size_t i = 0; i < in_sz_elements; i++) {
        in_host[i] = static_cast<bfloat16>(rand()) / static_cast<bfloat16>(RAND_MAX);
    }
    for(size_t i = 0; i < wei_sz_elements; i++) {
        wei_host[i] = static_cast<bfloat16>(rand()) / static_cast<bfloat16>(RAND_MAX);
    }
    for(size_t i = 0; i < bias_sz_elements; i++) {
        bias_host[i] = static_cast<bfloat16>(rand()) / static_cast<bfloat16>(RAND_MAX);
    }

    HIP_CHECK(hipMemcpy(in_dev, in_host.data(), in_sz_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(wei_dev, wei_host.data(), wei_sz_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(bias_dev, bias_host.data(), bias_sz_bytes, hipMemcpyHostToDevice));

    /* END OF BUFFER ALLOCATION */

    /* COMPILE AND EXECUTE FUSION */

    MIOPEN_CHECK(miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, inputTensor));

    miopenFusionOpDescriptor_t convoOp;
    miopenFusionOpDescriptor_t biasOp;
    miopenFusionOpDescriptor_t activOp;

    MIOPEN_CHECK(miopenCreateOpConvForward(fusePlanDesc, &convoOp, convDesc, weightTensor));
    MIOPEN_CHECK(miopenCreateOpBiasForward(fusePlanDesc, &biasOp, biasTensor));
    MIOPEN_CHECK(miopenCreateOpActivationForward(fusePlanDesc, &activOp, activMode));

    float alpha = 1.0f, beta = 0.0f;
    MIOPEN_CHECK(miopenSetOpArgsConvForward(fusionArgs, convoOp, &alpha, &beta, wei_dev));
    MIOPEN_CHECK(miopenSetOpArgsBiasForward(fusionArgs, biasOp, &alpha, &beta, bias_dev));
    MIOPEN_CHECK(miopenSetOpArgsActivForward(fusionArgs, activOp, &alpha, &beta, activ_alpha, activ_beta, activ_gamma));

    MIOPEN_CHECK(miopenCompileFusionPlan(handle, fusePlanDesc));

    std::cout << "Executing fusion plan..." << std::endl;
    MIOPEN_CHECK(miopenExecuteFusionPlan(handle,
                                     fusePlanDesc,
                                     inputTensor,
                                     in_dev,
                                     outputTensor,
                                     out_dev,
                                     fusionArgs));

    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipMemcpy(out_host.data(), out_dev, out_sz_bytes, hipMemcpyDeviceToHost));
    
    std::cout << "Execution complete. First 5 output values:" << std::endl;
    for(int i = 0; i < 5 && i < out_host.size(); ++i) {
        std::cout << "out_host[" << i << "] = " << out_host[i] << std::endl;
    }
  
    /* END OF FUSION */
    
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

    std::cout << "Cleaned up resources." << std::endl;

    return 0;
}
