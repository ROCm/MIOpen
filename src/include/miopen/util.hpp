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
#ifndef MIOPEN_UTIL_HPP_
#define MIOPEN_UTIL_HPP_

#include "miopen/common.hpp"
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>

namespace miopen {

float Im2ColGPU(Handle& handle,
                int data_size,
                ConstData_t im,
                size_t im_offset,
                int c,
                int h,
                int w,
                int wei_h,
                int wei_w,
                int out_h,
                int out_w,
                int pad_h,
                int pad_w,
                int stride_h,
                int stride_w,
                Data_t col);

float Col2ImGPU(Handle& handle,
                ConstData_t col,
                int col_h,
                int col_w,
                int wei_h,
                int wei_w,
                int pad_h,
                int pad_w,
                int stride_h,
                int stride_w,
                int c,
                int h,
                int w,
                Data_t im,
                size_t im_offset);

} // namespace miopen
#endif // _MIOPEN_UTIL_HPP_
