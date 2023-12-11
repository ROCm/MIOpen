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

#include "group_solver.hpp"


template<>
std::vector<ConvTestCaseGroup> ConvTestConfigs()
{ // g  n  c   h   w   k   y  x pad_x pad_y stri_x stri_y dia_x dia_y
    return {{1, 256, 192, 28, 28, 192, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {1, 256, 12, 28, 28, 12, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {4, 256, 192, 28, 28, 192, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {8, 256, 192, 28, 28, 192, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {8, 256, 384, 28, 28, 384, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {32, 256, 1024, 28, 28, 2048, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution}};
}

