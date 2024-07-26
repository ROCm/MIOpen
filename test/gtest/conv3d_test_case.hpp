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

struct Conv3DTestCase
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
    size_t K;
    Size3D img;
    Size3D filter;
    Size3D pad;
    Size3D stride;
    Size3D dilation;
    miopenConvolutionMode_t conv_mode;

    friend std::ostream& operator<<(std::ostream& os, const Conv3DTestCase& tc)
    {
        return os << " G:" << tc.G << " N:" << tc.N << " C:" << tc.C << " K:" << tc.K
                  << " D:" << tc.img.z << " H:" << tc.img.y << " W:" << tc.img.x
                  << " Z:" << tc.filter.z << " Y:" << tc.filter.y << " X:" << tc.filter.x
                  << " pad.z:" << tc.pad.z << " pad.y:" << tc.pad.y << " pad.x:" << tc.pad.x
                  << " stride.z:" << tc.stride.z << " stride.y:" << tc.stride.y
                  << " stride.x:" << tc.stride.x << " dilation.z:" << tc.dilation.z
                  << " dilation.y:" << tc.dilation.y << " dilation.x:" << tc.dilation.x
                  << " conv_mode:" << tc.conv_mode;
    }

    std::vector<size_t> GetInput() { return {N, C, img.z, img.y, img.x}; }
    std::vector<size_t> GetWeights()
    {
        EXPECT_EQUAL(C % G, 0);
        return {K, C / G, filter.z, filter.y, filter.x};
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
};
