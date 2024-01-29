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
#include "conv_test_base.hpp"

struct Conv2DTestCase
{
    size_t G;
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
    friend std::ostream& operator<<(std::ostream& os, const Conv2DTestCase& tc)
    {
        return os << " G:" << tc.G << " N:" << tc.N << " C:" << tc.C << " H:" << tc.H
                  << " W:" << tc.W << " k:" << tc.k << " y:" << tc.y << " x:" << tc.x
                  << " pad_y:" << tc.pad_y << " pad_x:" << tc.pad_x << " stride_y:" << tc.stride_y
                  << " dilation_y:" << tc.dilation_y << " conv_mode:" << tc.conv_mode;
    }

    std::vector<size_t> GetInput() { return {N, C, H, W}; }
    std::vector<size_t> GetWeights()
    {
        EXPECT_EQUAL(C % G, 0);
        return {k, C / G, y, x};
    }

    miopen::ConvolutionDescriptor GetConv()
    {
        return miopen::ConvolutionDescriptor{
            2,
            miopenConvolution,
            miopenPaddingDefault,
            {static_cast<int>(pad_y), static_cast<int>(pad_x)},
            {static_cast<int>(stride_y), static_cast<int>(stride_x)},
            {static_cast<int>(dilation_y), static_cast<int>(dilation_x)},
            {0, 0},
            static_cast<int>(G),
            1.0};
    }
};
