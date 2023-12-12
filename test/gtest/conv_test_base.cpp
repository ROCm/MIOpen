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
#include "conv_test_base.hpp"

template<>
std::vector<ConvTestCaseBase> GetNetworkForFusionCompileStepTest()
{
    return {{1, 64, 56, 56, 64, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
            {1, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution}};
}

template<>
std::vector<ConvTestCaseBase> GetNetwork1()
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

template<>
std::vector<ConvTestCaseBase> ConvTestConfigs()
{ // n  c   h   w   k   y  x pad_x pad_y stri_x stri_y dia_x dia_y
    return {{16, 128, 16, 16, 128, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {64, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {64, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {64, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {64, 1024, 14, 14, 1024, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution}};
}
