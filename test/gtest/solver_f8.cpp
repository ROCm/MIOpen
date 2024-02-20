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
#include "solver_f8.hpp"

template <>
std::vector<ConvTestCaseF8> ConvTestConfigs()
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
