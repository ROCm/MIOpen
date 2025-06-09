#include <miopen/miopen.h>  
#include <iostream>  
#include <vector>  
  
int main() {  
    miopenHandle_t handle;  
    miopenTensorDescriptor_t inputDesc, outputDesc, filterDesc;  
    miopenConvolutionDescriptor_t convDesc;  
  
    // Initialize MIOpen handle  
    miopenCreate(&handle);  
  
    // Input tensor descriptor setup  
    miopenCreateTensorDescriptor(&inputDesc);  
    std::array<int, 4> inputDims    = {128, 480, 28, 28};  
    std::array<int, 4> inputStrides = {376320, 784, 28, 1};  
    miopenSetTensorDescriptor(  
        inputDesc, miopenFloat, 4, inputDims.data(), inputStrides.data());  
  
    // Filter tensor descriptor setup  
    miopenCreateTensorDescriptor(&filterDesc);  
    std::array<int, 4> filterDims = {480, 1, 5, 5};  
    std::array<int, 4> filterStrides = {25, 25, 5, 1};  
    miopenSetTensorDescriptor(  
        filterDesc, miopenFloat, 4, filterDims.data(), filterStrides.data());  
  
    // Convolution descriptor setup  
    miopenCreateConvolutionDescriptor(&convDesc);  
    int pads[] = {2, 2};  
    int strides[] = {1, 1};  
    int dilations[] = {1, 1};  
    miopenInitConvolutionNdDescriptor(  
        convDesc, 2, pads, strides, dilations, miopenConvolution);  
    int ndim = 4;
    std::vector<int> out_lens(ndim);

    miopenGetConvolutionNdForwardOutputDim(
        convDesc, inputDesc, filterDesc, &ndim, out_lens.data());


    // Clean up  
    miopenDestroyTensorDescriptor(inputDesc);  
    miopenDestroyTensorDescriptor(filterDesc);  
    miopenDestroyConvolutionDescriptor(convDesc);  
    miopenDestroy(handle);  
  
    return 0;  
}  
