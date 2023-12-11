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
#pragma once

#include <miopen/miopen.h>
#include <iostream>

#include "conv_common.hpp"
#include "conv_tensor_gen.hpp"

template <typename T>
extern miopenDataType_t GetDataType();

struct ConvTestCaseBase
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
    size_t dilation_x;
    size_t dilation_y;
    miopenConvolutionMode_t conv_mode;

    friend std::ostream& operator<<(std::ostream& os, const ConvTestCaseBase& tc)
    {
        return os << "(N: " << tc.N << " C:" << tc.C << " H:" << tc.H << " W:" << tc.W
                  << " k: " << tc.k << " y:" << tc.y << " x:" << tc.x << " pad_y:" << tc.pad_y
                  << " pad_x:" << tc.pad_x << " stride_y:" << tc.stride_y
                  << " stride_x:" << tc.stride_x << " dilation_y:" << tc.dilation_y
                  << " dilation_x:" << tc.dilation_x << " conv_mode:" << tc.conv_mode << " )";
    }
    const std::vector<size_t> GetInput() { return {N, C, H, W}; }
    const std::vector<size_t> GetWeights() { return {k, C, y, x}; }
    const miopen::ConvolutionDescriptor GetConv()
    {
        return miopen::ConvolutionDescriptor{
            {static_cast<int>(pad_y), static_cast<int>(pad_x)},
            {static_cast<int>(stride_y), static_cast<int>(stride_x)},
            {static_cast<int>(dilation_y), static_cast<int>(dilation_x)}};
    }
};

template<typename T>
extern std::vector<T> GetNetworkForFusionCompileStepTest();

template<typename T>
extern std::vector<T> GetNetwork1();

template<typename T>
extern std::vector<T> ConvTestConfigs();

template <typename T, typename Tref = float, bool use_cpu_ref = false>
struct ConvFwdSolverTestBase
{
protected:
    void SetUpImpl(ConvTestCaseBase conv_config, miopenTensorLayout_t tensor_layout);

    void TearDownConv();

    void ThresholdChecks();

    miopen::ConvolutionDescriptor conv_desc;
    tensor<T> input;
    tensor<T> weights;
    tensor<T> output;
    tensor<T> ref_out;
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr wei_dev;
    miopen::Allocator::ManageDataPtr out_dev;
};
