/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include <gtest/gtest.h>
#include "cpu_conv.hpp"
#include "get_handle.hpp"
#include "tensor_util.hpp"
#include <fusionHost.hpp>
#include <miopen/conv/data_invoke_params.hpp>

struct ConvTestCase
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
    size_t dialtion_x;
    size_t dilation_y;
    miopenConvolutionMode_t conv_mode;
    friend std::ostream& operator<<(std::ostream& os, const ConvTestCase& tc)
    {
        return os << "N: " << tc.N << " C:" << tc.C << " H:" << tc.H << " W:" << tc.W
                  << " k: " << tc.k << " y:" << tc.y << " x:" << tc.x << " pad_y:" << tc.pad_y
                  << " pad_x:" << tc.pad_x << " stride_y:" << tc.stride_y
                  << " dilation_y:" << tc.dilation_y << " conv_mode:" << tc.conv_mode;
    }
};

std::vector<ConvTestCase> ConvTestConfigs()
{ // n  c   h   w   k   y  x pad_x pad_y stri_x stri_y dia_x dia_y
    return {{16, 128, 16, 16, 128, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {64, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {64, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {64, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {64, 1024, 14, 14, 1024, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution}};
}

template <typename T = float>
struct ConvFwdSolverTest
    : public ::testing::TestWithParam<std::tuple<miopenConvFwdAlgorithm_t, ConvTestCase>>
{
protected:
    void SetUp() override
    {
        std::tie(algo, conv_config) = GetParam();
        input   = tensor<T>{conv_config.N, conv_config.C, conv_config.H, conv_config.W};
        weights = tensor<T>{1, conv_config.k, conv_config.x, conv_config.y};
        input.generate(tensor_elem_gen_integer{17});
        weights.generate(tensor_elem_gen_integer{17});

        miopenCreateConvolutionDescriptor(&conv_desc);

        miopenInitConvolutionDescriptor(conv_desc,
                                        conv_config.conv_mode,
                                        conv_config.pad_y,
                                        conv_config.pad_x,
                                        conv_config.stride_y,
                                        conv_config.stride_x,
                                        conv_config.dilation_y,
                                        conv_config.dialtion_x);

        int n, c, h, w;
        miopenGetConvolutionForwardOutputDim(conv_desc, &input.desc, &weights.desc, &n, &c, &h, &w);

        output  = tensor<T>{static_cast<size_t>(n),
                           static_cast<size_t>(c),
                           static_cast<size_t>(h),
                           static_cast<size_t>(w)};
        ref_out = tensor<T>{static_cast<size_t>(n),
                            static_cast<size_t>(c),
                            static_cast<size_t>(h),
                            static_cast<size_t>(w)};

        std::fill(output.begin(), output.end(), std::numeric_limits<double>::quiet_NaN());
        std::fill(ref_out.begin(), ref_out.end(), std::numeric_limits<double>::quiet_NaN());

        auto&& handle = get_handle();
        in_dev        = handle.Write(input.data);
        wei_dev       = handle.Write(weights.data);
        out_dev       = handle.Write(output.data);
    }
    void TearDown() override
    {
        auto&& handle = get_handle();

        miopen::TensorDescriptor output_desc =
            miopen::deref(conv_desc).GetForwardOutputTensor(input.desc, weights.desc, miopenFloat);
        ref_out = tensor<T>{output_desc.GetLengths()};
        // ref_out = ref_conv_fwd(input, weights, output, conv_desc);
        cpu_convolution_forward(miopen::deref(conv_desc).GetSpatialDimension(),
                                input,
                                weights,
                                ref_out,
                                miopen::deref(conv_desc).GetConvPads(),
                                miopen::deref(conv_desc).GetConvStrides(),
                                miopen::deref(conv_desc).GetConvDilations(),
                                miopen::deref(conv_desc).GetGroupCount());

        output.data = handle.Read<T>(out_dev, output.data.size());
        EXPECT_FALSE(miopen::range_zero(ref_out)) << "Cpu data is all zeros";
        EXPECT_FALSE(miopen::range_zero(output)) << "Gpu data is all zeros";
        EXPECT_TRUE(miopen::range_distance(ref_out) == miopen::range_distance(output));

        const double tolerance = 80;
        double threshold       = std::numeric_limits<T>::epsilon() * tolerance;
        auto error             = miopen::rms_range(ref_out, output);

        EXPECT_FALSE(miopen::find_idx(ref_out, miopen::not_finite) >= 0)
            << "Non finite number found in the CPU data";

        EXPECT_TRUE(error < threshold)
            << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;

        miopenDestroyConvolutionDescriptor(conv_desc);
    }
    ConvTestCase conv_config;
    miopenConvolutionDescriptor_t conv_desc;
    tensor<T> input;
    tensor<T> weights;
    tensor<T> output;
    tensor<T> ref_out;
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr wei_dev;
    miopen::Allocator::ManageDataPtr out_dev;

    miopenActivationDescriptor_t activ_desc;
    miopenConvFwdAlgorithm_t algo = miopenConvolutionFwdAlgoDirect;
    bool test_skipped             = false;
};
