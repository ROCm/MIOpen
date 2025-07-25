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

#include <miopen/bfloat16.hpp>
#include <miopen/miopen.h>
#include <iostream>

#include "tensor_holder.hpp"
#include "conv_common.hpp"
#include "conv_tensor_gen.hpp"

using Direction = miopen::conv::Direction;

struct GroupConvTestConfigBase
{
    virtual ~GroupConvTestConfigBase() = default;
};

template <unsigned NDIM>
struct GroupConvTestConfig : GroupConvTestConfigBase
{
};

template <>
struct GroupConvTestConfig<2u> : GroupConvTestConfigBase
{

    struct Size2D
    {
        size_t y;
        size_t x;
    };

    size_t G;
    size_t N;
    size_t C;
    size_t k;

    Size2D img;
    Size2D filter;
    Size2D pad;
    Size2D stride;
    Size2D dilation;

    GroupConvTestConfig() = default;

    GroupConvTestConfig(size_t g_,
                        size_t n_,
                        size_t c_,
                        size_t k_,
                        Size2D img_,
                        Size2D filter_,
                        Size2D pad_,
                        Size2D stride_,
                        Size2D dilation_)
        : G(g_),
          N(n_),
          C(c_),
          k(k_),
          img(img_),
          filter(filter_),
          pad(pad_),
          stride(stride_),
          dilation(dilation_)
    {
    }

    friend std::ostream& operator<<(std::ostream& os, const GroupConvTestConfig& tc)
    {
        return os << " G:" << tc.G << " N:" << tc.N << " C:" << tc.C << " K:" << tc.k
                  << " H:" << tc.img.y << " W:" << tc.img.x << " y:" << tc.filter.y
                  << " x:" << tc.filter.x << " pad.y:" << tc.pad.y << " pad.x:" << tc.pad.x
                  << " stride.y:" << tc.stride.y << " stride.x" << tc.stride.x
                  << " dilation.y:" << tc.dilation.y << " dilation.x" << tc.dilation.x;
    }

    std::vector<size_t> GetInput() { return {N, C, img.y, img.x}; }
    std::vector<size_t> GetWeights()
    {
        EXPECT_EQUAL(C % G, 0);
        return {k, C / G, filter.y, filter.x};
    }

    miopen::ConvolutionDescriptor GetConv()
    {
        return miopen::ConvolutionDescriptor{
            2,
            miopenConvolution,
            miopenPaddingDefault,
            {static_cast<int>(pad.y), static_cast<int>(pad.x)},
            {static_cast<int>(stride.y), static_cast<int>(stride.x)},
            {static_cast<int>(dilation.y), static_cast<int>(dilation.x)},
            {0, 0},
            static_cast<int>(G),
            1.0};
    }

    template <Direction DIR>
    static std::vector<GroupConvTestConfig> GetSmokeConfigs()
    {
        if constexpr(DIR == Direction::Forward)
        {

            return {
                // clang-format off
            // g   n    C    K    img         filter    pad     stride  dilation
            {1,   32,   64,  128, {28, 28},   {3, 3},   {0, 1}, {1, 2}, {2, 1}},
            {32,  16,   32,   64,  {7, 7},    {3, 3},   {1, 1}, {1, 1}, {1, 1}},
            {1,   16,   32,   64, {16, 16},   {2, 2},   {0, 0}, {3, 3}, {1, 1}},
            {4,    8,   16,   32, {32, 4},    {3, 1},   {1, 0}, {1, 1}, {1, 1}},
                // clang-format on
            };
        }
    }

    template <Direction DIR>
    static std::vector<GroupConvTestConfig> GetConfigs()
    {
        if constexpr(DIR == Direction::Forward)
        {
            // clang-format off
        return {
            // g   n     C     K      img       filter   pad    stride  dilation
              {1,  64,  1024, 2048, {14, 14},   {1, 1}, {0, 0}, {2, 2}, {1, 1}},
              {1,  256, 192,  192,  {28, 28},   {3, 3}, {1, 1}, {1, 1}, {1, 1}},
              {8,  256, 192,  192,  {28, 28},   {3, 3}, {1, 1}, {1, 1}, {1, 1}},
              {8,  256, 384,  384,  {28, 28},   {3, 3}, {1, 1}, {1, 1}, {1, 1}},
              {32, 256, 1024, 2048, {28, 28},   {3, 3}, {1, 1}, {1, 1}, {1, 1}},
              {4,  256, 192,  192,  {28, 28},   {3, 3}, {1, 1}, {2, 2}, {1, 1}},
              {8,  256, 384,  384,  {28, 28},   {3, 3}, {1, 1}, {2, 2}, {1, 1}},
              {32, 256, 1024, 2048, {28, 28},   {3, 3}, {1, 1}, {2, 2}, {1, 1}},
              {1,  6,   448,  896,  {118, 182}, {3, 3}, {0, 0}, {2, 2}, {1, 1}},
              {4,  256, 192,  192,  {28, 28},   {1, 1}, {1, 1}, {2, 2}, {1, 1}},
              {8,  256, 384,  384,  {28, 28},   {1, 1}, {1, 1}, {2, 2}, {1, 1}},
              {32, 256, 1024, 2048, {28, 28},   {1, 1}, {1, 1}, {2, 2}, {1, 1}},
              {1,  6,   448,  896,  {118, 182}, {1, 1}, {0, 0}, {2, 2}, {1, 1}},
              {4,  16,  224,  224,  {469, 724}, {3, 3}, {1, 1}, {2, 2}, {1, 1}},
        };
            // clang-format on
        }
    }
};

template <>
struct GroupConvTestConfig<3u>
{

    struct Size3D
    {
        size_t z;
        size_t y;
        size_t x;
    };

    size_t G;
    size_t N;
    size_t C;
    size_t k;

    Size3D img;
    Size3D filter;
    Size3D pad;
    Size3D stride;
    Size3D dilation;

    friend std::ostream& operator<<(std::ostream& os, const GroupConvTestConfig<3u>& tc)
    {
        return os << " G:" << tc.G << " N:" << tc.N << " C:" << tc.C << " K:" << tc.k
                  << " D:" << tc.img.z << " H:" << tc.img.y << " W:" << tc.img.x
                  << " z:" << tc.filter.z << " y:" << tc.filter.y << " x:" << tc.filter.x
                  << " pad.z:" << tc.pad.z << " pad.y:" << tc.pad.y << " pad.x:" << tc.pad.x
                  << " stride.z:" << tc.stride.z << " stride.y:" << tc.stride.y
                  << " stride.x:" << tc.stride.x << " dilation.z:" << tc.dilation.z
                  << " dilation.y:" << tc.dilation.y << " dilation.x:" << tc.dilation.x;
    }

    std::vector<size_t> GetInput() { return {N, C, img.z, img.y, img.x}; }
    std::vector<size_t> GetWeights()
    {
        EXPECT_EQUAL(C % G, 0);
        return {k, C / G, filter.z, filter.y, filter.x};
    }

    miopen::ConvolutionDescriptor GetConv()
    {
        return miopen::ConvolutionDescriptor{
            3,
            miopenConvolution,
            miopenPaddingDefault,
            {static_cast<int>(pad.z), static_cast<int>(pad.y), static_cast<int>(pad.x)},
            {static_cast<int>(stride.z), static_cast<int>(stride.y), static_cast<int>(stride.x)},
            {static_cast<int>(dilation.z),
             static_cast<int>(dilation.y),
             static_cast<int>(dilation.x)},
            {0, 0, 0},
            static_cast<int>(G),
            1.0};
    }

    template <Direction DIR>
    static std::vector<GroupConvTestConfig> GetSmokeConfigs()
    {
        if constexpr(DIR == Direction::Forward)
        {

            return {
                // clang-format off
            // g   n    C    K    img         filter    pad     stride  dilation
            {1,   32,   64,  128, {28, 28, 28},   {3, 3, 3},   {0, 1, 1}, {1, 2, 2}, {2, 1, 1}},
            {32,  16,   32,   64,  {7, 7, 7},    {3, 3, 3},   {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
            {1,   16,   32,   64, {16, 16, 16},   {2, 2, 2},   {0, 0, 0}, {3, 3, 3}, {1, 1, 1}},
            {4,    8,   16,   32, {32, 4, 4},    {3, 1, 1},   {1, 0, 0}, {1, 1, 1}, {1, 1, 1}},
                // clang-format on
            };
        }
    }

    template <Direction DIR>
    static std::vector<GroupConvTestConfig> GetConfigs()
    {

        if constexpr(DIR == Direction::Forward)
        {
            // clang-format off
            return {
              // g   n   C    K      img         filter      pad        stride    dilation
                {1 , 128, 64 , 64 , {14, 28, 28} , {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {1 , 64 , 32 , 32 , {28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {2 , 128, 32 , 32 , {28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {8 , 128, 32 , 32 , {28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {2 , 128, 32 , 32 , {28, 28, 28} , {3, 3, 3}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}},
                {8 , 64 , 32 , 32 , {28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {16, 64 , 32 , 32 , {28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {2 , 128, 32 , 32 , {28, 28, 28} , {3, 3, 3}, {0, 0, 0}, {2, 2, 2}, {1, 1, 1}},
                {8 , 64 , 32 , 32 , {28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {2, 2, 2}, {1, 1, 1}},
                {16, 64 , 32 , 32 , {28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {2, 2, 2}, {1, 1, 1}},
                {3 , 48 , 48 , 48 , {28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {3 , 48 , 39 , 39 , {28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {5 , 120, 60 , 60 , {28, 28, 28} , {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
                {1 , 6  , 448, 896, {3, 118, 182}, {1, 1, 1}, {0, 0, 0}, {1, 2, 2}, {1, 1, 1}},
            };
            // clang-format on
        }
    }
};

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
    // N, C, H, W, k, y, x, pad_x, pad_y, stride_x, stride_y, dilation_x, dilation_y, conv_mode
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

template <typename T,
          typename Tref    = float,
          typename TConfig = ConvTestCaseBase,
          bool use_cpu_ref = false,
          unsigned NDIM    = 2>
struct ConvFwdSolverTestBase
{
protected:
    void SetUpImpl(TConfig conv_config, miopenTensorLayout_t tensor_layout)
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

        double tolerance = 80;
        if constexpr(std::is_same_v<T, bfloat16>)
        {
            tolerance = 4;
        }

        double threshold = std::numeric_limits<T>::epsilon() * tolerance;
        auto error       = miopen::rms_range(ref_out, output);

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
