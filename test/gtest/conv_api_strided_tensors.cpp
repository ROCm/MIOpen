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

// Test Suite for convolution with strided tensor descriptors

#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include "platform.hpp"
#include "../workspace.hpp"

#define MIOPEN_CHECK_RET(val) ASSERT_EQ(val, miopenStatusSuccess)

class GPU_ConvStridedTensors_FP32 : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // MIOpen handle
        MIOPEN_CHECK_RET(miopenCreate(&handle));

        // Tensor descriptors
        MIOPEN_CHECK_RET(miopenCreateTensorDescriptor(&input_descr));
        MIOPEN_CHECK_RET(miopenSetTensorDescriptor(
            input_descr, miopenFloat, input_dims.size(), input_dims.data(), input_strides.data()));

        MIOPEN_CHECK_RET(miopenCreateTensorDescriptor(&filter_descr));
        MIOPEN_CHECK_RET(miopenSetTensorDescriptor(filter_descr,
                                                   miopenFloat,
                                                   filter_dims.size(),
                                                   filter_dims.data(),
                                                   filter_strides.data()));

        MIOPEN_CHECK_RET(miopenCreateTensorDescriptor(&output_descr));
        MIOPEN_CHECK_RET(miopenSetTensorDescriptor(output_descr,
                                                   miopenFloat,
                                                   output_dims.size(),
                                                   output_dims.data(),
                                                   output_strides.data()));

        // Convolution descriptor
        MIOPEN_CHECK_RET(miopenCreateConvolutionDescriptor(&conv_descr));
        MIOPEN_CHECK_RET(miopenInitConvolutionNdDescriptor(
            conv_descr, pad.size(), pad.data(), stride.data(), dilation.data(), miopenConvolution));
        MIOPEN_CHECK_RET(miopenSetConvolutionGroupCount(conv_descr, 1));

        // Workspace
        size_t sz = 0;
        MIOPEN_CHECK_RET(miopenConvolutionForwardGetWorkSpaceSize(
            handle, filter_descr, input_descr, conv_descr, output_descr, &sz));

        // Data
        wspace.resize(sz);
        h_input.resize(input_size);
        h_filter.resize(filter_size);
        h_output.resize(output_size);
    }

    void TearDown() override
    {
        // Convolution descriptor
        if(conv_descr != nullptr)
        {
            MIOPEN_CHECK_RET(miopenDestroyConvolutionDescriptor(conv_descr));
        }

        // Tensor descriptors
        if(output_descr != nullptr)
        {
            MIOPEN_CHECK_RET(miopenDestroyTensorDescriptor(output_descr));
        }
        if(filter_descr != nullptr)
        {
            MIOPEN_CHECK_RET(miopenDestroyTensorDescriptor(filter_descr));
        }
        if(input_descr != nullptr)
        {
            MIOPEN_CHECK_RET(miopenDestroyTensorDescriptor(input_descr));
        }

        // MIOpen handle
        if(handle != nullptr)
        {
            MIOPEN_CHECK_RET(miopenDestroy(handle));
        }
    }

    // MIOpen handle
    miopenHandle_t handle = nullptr;

    // Tensor descriptors
    miopenTensorDescriptor_t input_descr  = nullptr;
    miopenTensorDescriptor_t filter_descr = nullptr;
    miopenTensorDescriptor_t output_descr = nullptr;
    std::vector<int> input_dims           = {4, 4, 16, 9, 16};
    std::vector<int> input_strides        = {10240, 2560, 160, 16, 1};
    std::vector<int> filter_dims          = {8, 4, 3, 3, 3};
    std::vector<int> filter_strides       = {108, 27, 9, 3, 1};
    std::vector<int> output_dims          = {4, 8, 8, 4, 8};
    std::vector<int> output_strides       = {2048, 256, 32, 8, 1};

    // Convolution descriptor
    miopenConvolutionDescriptor_t conv_descr = nullptr;
    std::vector<int> pad                     = {1, 0, 1};
    std::vector<int> stride                  = {2, 2, 2};
    std::vector<int> dilation                = {1, 1, 1};

    // Workspace
    Workspace wspace{};

    // Data
    const size_t input_size   = input_dims[0] * input_strides[0];
    const size_t filter_size  = filter_dims[0] * filter_strides[0];
    const size_t output_size  = output_dims[0] * output_strides[0];
    const size_t input_bytes  = input_size * sizeof(float);
    const size_t filter_bytes = filter_size * sizeof(float);
    const size_t output_bytes = output_size * sizeof(float);
    std::vector<float> h_input;
    std::vector<float> h_filter;
    std::vector<float> h_output;
};

TEST_F(GPU_ConvStridedTensors_FP32, ConvStridedTensorsNotImplemented)
{
    auto device = Device(handle);

    auto d_input  = device.Malloc(input_bytes);
    auto d_filter = device.Malloc(filter_bytes);
    auto d_output = device.Malloc(output_bytes);

    std::fill_n(h_input.begin(), h_input.size(), 1.f);
    ASSERT_TRUE(d_input.CopyToDevice(h_input.data(), input_bytes));

    std::fill_n(h_filter.begin(), h_filter.size(), 1.f);
    ASSERT_TRUE(d_filter.CopyToDevice(h_filter.data(), filter_bytes));

    miopenConvAlgoPerf_t perf_results[10];
    int perf_results_count;

    ASSERT_EQ(miopenFindConvolutionForwardAlgorithm(handle,
                                                    input_descr,
                                                    d_input.Data(),
                                                    filter_descr,
                                                    d_filter.Data(),
                                                    conv_descr,
                                                    output_descr,
                                                    d_output.Data(),
                                                    sizeof(perf_results) / sizeof(perf_results[0]),
                                                    &perf_results_count,
                                                    perf_results,
                                                    wspace.ptr(),
                                                    wspace.size(),
                                                    true),
              miopenStatusSuccess);
    ASSERT_GT(perf_results_count, 0);

    const float alpha = 1.f;
    const float beta  = 0.f;

    ASSERT_TRUE(device.Synchronize());
    ASSERT_EQ(miopenConvolutionForward(handle,
                                       &alpha,
                                       input_descr,
                                       d_input.Data(),
                                       filter_descr,
                                       d_filter.Data(),
                                       conv_descr,
                                       perf_results[0].fwd_algo,
                                       &beta,
                                       output_descr,
                                       d_output.Data(),
                                       wspace.ptr(),
                                       wspace.size()),
              miopenStatusSuccess);
    ASSERT_TRUE(device.Synchronize());
}
