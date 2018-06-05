/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_DIRECT_CONV_OCL_HPP_
#define GUARD_MIOPEN_DIRECT_CONV_OCL_HPP_

#include <chrono>
#include <cmath>
#include <miopen/common.hpp>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>

namespace miopen {

void DirectConvInference(Handle& handle,
                         const void* alpha,
                         const TensorDescriptor& xDesc,
                         ConstData_t x,
                         const TensorDescriptor& wDesc,
                         ConstData_t w,
                         const void* beta,
                         const TensorDescriptor& yDesc,
                         Data_t y,
                         int pad_h,
                         int pad_w,
                         int u,
                         int v,
                         int dilation_h,
                         int dilation_w,
                         int bias_mode,
                         ConstData_t convBias);
}

#endif
