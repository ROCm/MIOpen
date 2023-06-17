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

#include <miopen/hip_float8.h>

#include "conv_test_case.hpp"
#include "conv_tensor_gen.hpp"

template <typename T = float, typename Tref = double>
struct ConvFwdSolverTest
    : public ::testing::TestWithParam<std::tuple<miopenConvFwdAlgorithm_t, ConvTestCase>>
{
protected:
    void SetUp() override
    {
        test_skipped                = false;
        std::tie(algo, conv_config) = GetParam();
        input   = tensor<T>{conv_config.N, conv_config.C, conv_config.H, conv_config.W};
        weights = tensor<T>{conv_config.k, conv_config.C, conv_config.y, conv_config.x};
        input.generate(GenData<T>{});
        weights.generate(GenWeights<T>{});

        conv_desc = conv_config.GetConv();

        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, GetDataType<T>());

        output = tensor<T>{output_desc.GetLengths()};

        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

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

        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, GetDataType<T>());
        ref_out = tensor<Tref>{output_desc.GetLengths()};
        cpu_convolution_forward(conv_desc.GetSpatialDimension(),
                                input,
                                weights,
                                ref_out,
                                conv_desc.GetConvPads(),
                                conv_desc.GetConvStrides(),
                                conv_desc.GetConvDilations(),
                                conv_desc.GetGroupCount());

        output.data         = handle.Read<T>(out_dev, output.data.size());
        const auto zero_chk = [](T x) { return static_cast<T>(x) == static_cast<T>(0.0); };
        EXPECT_FALSE(std::all_of(ref_out.begin(), ref_out.end(), [](float x) { return x == 0.0f; }))
            << "Cpu data is all zeros";
        EXPECT_FALSE(std::all_of(output.begin(), output.end(), zero_chk))
            << "Gpu data is all zeros";
        EXPECT_TRUE(miopen::range_distance(ref_out) == miopen::range_distance(output));

        const double tolerance = 80;
        double threshold       = static_cast<float>(std::numeric_limits<T>::epsilon()) * tolerance;
        auto error             = miopen::rms_range(ref_out, output);

        EXPECT_FALSE(miopen::find_idx(ref_out, miopen::not_finite) >= 0)
            << "Non finite number found in the CPU data";

        EXPECT_TRUE(error < threshold)
            << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;
    }
    ConvTestCase conv_config;
    miopen::ConvolutionDescriptor conv_desc;
    tensor<T> input;
    tensor<T> weights;
    tensor<T> output;
    tensor<Tref> ref_out;
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr wei_dev;
    miopen::Allocator::ManageDataPtr out_dev;
    miopenConvFwdAlgorithm_t algo = miopenConvolutionFwdAlgoDirect;
    bool test_skipped             = false;
};
