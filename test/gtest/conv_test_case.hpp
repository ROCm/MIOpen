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

#include <gtest/gtest.h>
#include <miopen/miopen.h>

template <typename T>
miopenDataType_t GetDataType();

template <>
miopenDataType_t GetDataType<float>()
{
    return miopenFloat;
}

template <>
miopenDataType_t GetDataType<half_float::half>()
{
    return miopenHalf;
}

template <>
miopenDataType_t GetDataType<miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>>()
{
    return miopenFloat8;
}

template <>
miopenDataType_t GetDataType<miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>>()
{
    return miopenBFloat8;
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
    size_t dilation_x;
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
    std::vector<size_t> GetInput() { return {N, C, H, W}; }
    std::vector<size_t> GetWeights() { return {k, C, y, x}; }
};

std::vector<ConvTestCase> ConvTestConfigs()
{ // n  c   h   w   k   y  x pad_x pad_y stri_x stri_y dia_x dia_y
    return {{16, 128, 16, 16, 128, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {64, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {64, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {64, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {64, 1024, 14, 14, 1024, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution}};
}

std::vector<ConvTestCase> GetNetworkForFusionCompileStepTest()
{
    return {{1, 64, 56, 56, 64, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {1, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution}};
}

std::vector<ConvTestCase> GetNetwork1()
{
    // pyt_mlperf_resnet50v1.5
    const std::vector<ConvTestCase> res{
        {64, 1024, 14, 14, 2048, 1, 1, 0, 0, 2, 2, 1, 1, miopenConvolution},
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
    return res;
}
