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

#include "tensor_holder.hpp"
#include "conv_common.hpp"
#include "conv_tensor_gen.hpp"

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
    const std::vector<size_t> GetInput() const { return {N, C, H, W}; }
    const std::vector<size_t> GetWeights() const { return {k, C, y, x}; }
    const miopen::ConvolutionDescriptor GetConv() const
    {
        return miopen::ConvolutionDescriptor{
            {static_cast<int>(pad_y), static_cast<int>(pad_x)},
            {static_cast<int>(stride_y), static_cast<int>(stride_x)},
            {static_cast<int>(dilation_y), static_cast<int>(dilation_x)}};
    }
};

template <typename T>
std::vector<T> GetNetworkForFusionCompileStepTest();

template <>
inline std::vector<ConvTestCaseBase> GetNetworkForFusionCompileStepTest()
{
    return {{1, 64, 56, 56, 64, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {1, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution}};
}

template <typename T>
std::vector<T> GetNetwork1();

template <>
inline std::vector<ConvTestCaseBase> GetNetwork1()
{
    // pyt_mlperf_resnet50v1.5
    return {{64, 1024, 14, 14, 2048, 1, 1, 0, 0, 2, 2, 1, 1, miopenConvolution},
            {64, 1024, 14, 14, 256, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {64, 1024, 14, 14, 512, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {64, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {64, 128, 28, 28, 512, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {64, 128, 56, 56, 128, 3, 3, 1, 1, 2, 2, 1, 1, miopenConvolution},
            {64, 2048, 7, 7, 512, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {64, 256, 14, 14, 1024, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {64, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {64, 256, 28, 28, 256, 3, 3, 1, 1, 2, 2, 1, 1, miopenConvolution},
            {64, 256, 56, 56, 128, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {64, 256, 56, 56, 512, 1, 1, 0, 0, 2, 2, 1, 1, miopenConvolution},
            {64, 256, 56, 56, 64, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {64, 3, 224, 224, 64, 7, 7, 3, 3, 2, 2, 1, 1, miopenConvolution},
            {64, 512, 14, 14, 512, 3, 3, 1, 1, 2, 2, 1, 1, miopenConvolution},
            {64, 512, 28, 28, 1024, 1, 1, 0, 0, 2, 2, 1, 1, miopenConvolution},
            {64, 512, 28, 28, 128, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {64, 512, 28, 28, 256, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {64, 512, 7, 7, 2048, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {64, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {64, 64, 56, 56, 256, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {64, 64, 56, 56, 64, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {64, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution}};
}

template <typename T>
std::vector<T> ConvTestConfigs();

template <>
inline std::vector<ConvTestCaseBase> ConvTestConfigs()
{ // n  c   h   w   k   y  x pad_x pad_y stri_x stri_y dia_x dia_y
    return {{16, 128, 16, 16, 128, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {64, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {64, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {64, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {64, 1024, 14, 14, 1024, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution}};
}

template <typename T, typename Tref = float, bool use_cpu_ref = false>
struct ConvFwdSolverTestBase
{
protected:
    void SetUpImpl(ConvTestCaseBase conv_config, miopenTensorLayout_t tensor_layout)
    {
        input   = tensor<T>{tensor_layout, conv_config.GetInput()};
        weights = tensor<T>{tensor_layout, conv_config.GetWeights()};
        input.generate(GenData<T>{});
        weights.generate(GenWeights<T>{});

        conv_desc = conv_config.GetConv();

        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<T>{});

        output = tensor<T>{tensor_layout, output_desc.GetLengths()};
        std::fill(output.begin(), output.end(), T(0));

        auto&& handle = get_handle();
        in_dev        = handle.Write(input.data);
        wei_dev       = handle.Write(weights.data);
        out_dev       = handle.Write(output.data);
    }

    void TearDownConv()
    {
        ref_out = tensor<Tref>{output.desc.GetLayout_t(), output.desc.GetLengths()};
        if(use_cpu_ref)
        {
            cpu_convolution_forward(conv_desc.GetSpatialDimension(),
                                    input,
                                    weights,
                                    ref_out,
                                    conv_desc.GetConvPads(),
                                    conv_desc.GetConvStrides(),
                                    conv_desc.GetConvDilations(),
                                    conv_desc.GetGroupCount());
        }
        else
        {
            ref_out = ref_conv_fwd(input, weights, ref_out, conv_desc);
        }
    }

    void ThresholdChecks()
    {
        auto&& handle = get_handle();
        output.data   = handle.Read<T>(out_dev, output.data.size());

        ASSERT_FALSE(miopen::range_zero(ref_out)) << "Cpu data is all zeros";
        ASSERT_FALSE(miopen::range_zero(output)) << "Gpu data is all zeros";
        ASSERT_EQ(miopen::range_distance(ref_out), miopen::range_distance(output));

        const double tolerance = 80;
        double threshold       = std::numeric_limits<T>::epsilon() * tolerance;
        auto error             = miopen::rms_range(ref_out, output);

        ASSERT_LT(miopen::find_idx(ref_out, miopen::not_finite), 0)
            << "Non finite number found in the CPU data";

        ASSERT_LT(error, threshold) << "Error beyond tolerance";
    }

    miopen::ConvolutionDescriptor conv_desc;
    tensor<T> input;
    tensor<T> weights;
    tensor<T> output;
    tensor<Tref> ref_out;
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr wei_dev;
    miopen::Allocator::ManageDataPtr out_dev;
};
