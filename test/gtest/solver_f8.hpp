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
#include "conv_common.hpp"
#include <miopen/hip_float8.hpp>
#include "verify.hpp"
using float8  = miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>;
using bfloat8 = miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>;

template <typename T>
miopenDataType_t GetDataType();

template <>
miopenDataType_t GetDataType<float8>()
{
    return miopenFloat8;
}

template <>
miopenDataType_t GetDataType<bfloat8>()
{
    return miopenBFloat8;
}

template <>
miopenDataType_t GetDataType<float>()
{
    return miopenFloat;
}

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

    miopen::ConvolutionDescriptor GetConv()
    {
        return miopen::ConvolutionDescriptor{
            {static_cast<int>(pad_y), static_cast<int>(pad_x)},
            {static_cast<int>(stride_y), static_cast<int>(stride_x)},
            {static_cast<int>(dilation_y), static_cast<int>(dilation_y)}};
    }
};

std::vector<ConvTestCase> ConvTestConfigs()
{           // n  c   h   w   k   y  x pad_x pad_y stri_x stri_y dia_x dia_y
    return {// New tests begin
            {1, 32, 4, 4, 16, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {2, 32, 4, 4, 16, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {4, 32, 4, 4, 16, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {8, 32, 4, 4, 16, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {16, 32, 4, 4, 16, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {16, 128, 16, 16, 128, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {64, 128, 28, 28, 128, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {64, 128, 64, 64, 64, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {64, 128, 128, 64, 64, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {64, 128, 128, 128, 64, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {64, 128, 128, 128, 128, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {64, 256, 128, 128, 128, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {64, 256, 256, 128, 128, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {64, 256, 256, 256, 128, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {64, 256, 256, 256, 256, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {128, 256, 256, 256, 256, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {128, 256, 512, 256, 256, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {128, 256, 512, 512, 256, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {128, 256, 1024, 512, 256, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {128, 256, 1024, 1024, 256, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {128, 512, 1024, 1024, 256, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {128, 512, 1024, 1024, 512, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {128, 1024, 1024, 1024, 512, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {128, 1024, 1024, 1024, 1024, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {256, 1024, 1024, 1024, 1024, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {1024, 1024, 1024, 1024, 1024, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {1024, 2048, 2048, 2048, 2048, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            // New tests end
            {16, 128, 16, 16, 128, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {64, 128, 28, 28, 128, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {64, 256, 14, 14, 256, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {64, 512, 7, 7, 512, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {64, 1024, 14, 14, 1024, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution}};
}

template <typename U, typename V>
struct Fp8Cast
{
    uint64_t seed = 1234;
    bool is_stoch = true;
    V operator()(U x)
    {
        if(is_stoch)
        {
            auto tmp =
                float8(static_cast<float>(x), miopen_f8::hip_f8_rounding_mode::stochastic, seed);
            return static_cast<V>(tmp);
        }
        else
        {
            auto tmp = float8(static_cast<float>(x));
            return static_cast<V>(tmp);
        }
    }
};

template <typename T, typename Tout = T, typename Tacc = float>
struct ConvFwdSolverTest
    : public ::testing::TestWithParam<std::tuple<miopenConvFwdAlgorithm_t, ConvTestCase>>
{
protected:
    void SetUp() override
    {
        test_skipped                = false;
        std::tie(algo, conv_config) = GetParam();
        input   = tensor<T>{conv_config.N, conv_config.C, conv_config.H, conv_config.W};
        weights = tensor<T>{conv_config.k, conv_config.C, conv_config.x, conv_config.y};

        auto gen_fp8_value = [=](auto...) {
            const auto tmp = float8(scalar_gen_random_float{-0.5, 0.5}());
            return tmp;
        };

        input.generate(gen_fp8_value);
        weights.generate(gen_fp8_value);

        conv_desc = conv_config.GetConv();

        miopen::TensorDescriptor output_desc = conv_desc.GetForwardOutputTensor(
            input.desc, weights.desc, GetDataType<Tout>()); // Tgpu Datatype?

        output = tensor<Tout>{output_desc.GetLengths()}; // half_float::half instead?

        std::fill(output.begin(), output.end(), std::numeric_limits<Tout>::quiet_NaN());

        auto&& handle = get_handle();
        in_dev        = handle.Write(input.data);
        wei_dev       = handle.Write(weights.data);
        out_dev       = handle.Write(output.data);
    }
    void TearDown() override
    {
        if(test_skipped)
            return;

        auto&& handle = get_handle();

        miopen::TensorDescriptor output_desc = conv_desc.GetForwardOutputTensor(
            input.desc, weights.desc, GetDataType<Tout>()); // miopenFloat or GetDataType<Tgpu>() ?
        ref_out = tensor<Tout>{output_desc.GetLengths()};

        using FI       = Fp8Cast<T, T>;
        using FW       = Fp8Cast<T, T>;
        FI in_func     = {0, true};
        FW weight_func = {0, true};

        cpu_convolution_forward<T, T, Tout, decltype(conv_desc.GetConvPads()), Tacc, FW, FI>(
            conv_desc.GetSpatialDimension(),
            input,
            weights,
            ref_out,
            conv_desc.GetConvPads(),
            conv_desc.GetConvStrides(),
            conv_desc.GetConvDilations(),
            conv_desc.GetGroupCount(),
            in_func,
            weight_func);

        output.data = handle.Read<Tout>(out_dev, output.data.size());
        EXPECT_FALSE(miopen::f8_range_zero(ref_out)) << "Cpu data is all zeros";
        EXPECT_FALSE(miopen::f8_range_zero(output)) << "Gpu data is all zeros";
        EXPECT_TRUE(miopen::range_distance(ref_out) == miopen::range_distance(output));

        const float tolerance = 80.0;
        auto threshold        = (static_cast<float>(std::numeric_limits<Tout>::epsilon()) *
                          static_cast<float>(tolerance));

        auto error = miopen::rms_range(ref_out, output);

        bool refOutNan = false;
        for(auto refOutElem : ref_out.data)
        {
            if(refOutElem.is_nan())
            {
                refOutNan = true;
                break;
            }
        }

        bool outputNan = false;
        for(auto outputElem : output.data)
        {
            if(outputElem.is_nan())
            {
                outputNan = true;
                break;
            }
        }

        EXPECT_FALSE(refOutNan) << "NAN found in CPU data";
        EXPECT_FALSE(outputNan) << "NAN found in GPU data";

        EXPECT_TRUE(error < threshold)
            << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;
    }
    ConvTestCase conv_config;
    miopen::ConvolutionDescriptor conv_desc;
    tensor<T> input;
    tensor<T> weights;
    tensor<Tout> output; // Or T?
    tensor<Tout> ref_out;
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr wei_dev;
    miopen::Allocator::ManageDataPtr out_dev;
    miopenConvFwdAlgorithm_t algo = miopenConvolutionFwdAlgoGEMM;
    bool test_skipped             = false;
};
