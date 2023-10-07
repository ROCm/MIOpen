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

#include <random>

#include "get_handle.hpp"
#include <miopen/conv/data_invoke_params.hpp>

#include "../driver/tensor_driver.hpp"
#include "conv_common.hpp"

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
miopenDataType_t GetDataType<int8_t>()
{
    return miopenInt8;
}

struct Conv3DTestCase
{
    size_t G;
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    size_t k;
    size_t z;
    size_t y;
    size_t x;
    size_t pad_x;
    size_t pad_y;
    size_t pad_z;
    size_t stride_x;
    size_t stride_y;
    size_t stride_z;
    size_t dilation_x;
    size_t dilation_y;
    size_t dilation_z;
    miopenConvolutionMode_t conv_mode;
    friend std::ostream& operator<<(std::ostream& os, const Conv3DTestCase& tc)
    {
        return os << " G:" << tc.G << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D
                  << " H:" << tc.H << " W:" << tc.W << " k:" << tc.k << " z:" << tc.z
                  << " y:" << tc.y << " x:" << tc.x << " pad_z:" << tc.pad_z
                  << " pad_y:" << tc.pad_y << " pad_x:" << tc.pad_x << " stride_z:" << tc.stride_z
                  << " stride_y:" << tc.stride_y << " stride_x:" << tc.stride_x
                  << " dilation_z:" << tc.dilation_z << " dilation_y:" << tc.dilation_y
                  << " dilation_x:" << tc.dilation_x << " conv_mode:" << tc.conv_mode;
    }

    std::vector<size_t> GetInput() { return {N, C, D, H, W}; }
    std::vector<size_t> GetWeights()
    {
        EXPECT_EQUAL(C % G, 0);
        return {k, C / G, z, y, x};
    }

    miopen::ConvolutionDescriptor GetConv()
    {
        return miopen::ConvolutionDescriptor{
            3,
            miopenConvolution,
            miopenPaddingDefault,
            {static_cast<int>(pad_z), static_cast<int>(pad_y), static_cast<int>(pad_x)},
            {static_cast<int>(stride_z), static_cast<int>(stride_y), static_cast<int>(stride_x)},
            {static_cast<int>(dilation_z),
             static_cast<int>(dilation_y),
             static_cast<int>(dilation_x)},
            {0, 0, 0},
            static_cast<int>(G),
            1.0};
    }
};
