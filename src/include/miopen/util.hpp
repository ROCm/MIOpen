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

#include <miopen/common.hpp>
#include <miopen/miopen.h>

namespace miopen {

struct Handle;

float Im2ColGPU(Handle& handle,
                int data_size,
                ConstData_t im,
                int im_offset,
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
                int dilation_h,
                int dilation_w,
                Data_t col,
                miopenDataType_t type);

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
                int dilation_h,
                int dilation_w,
                int c,
                int h,
                int w,
                Data_t im,
                int im_offset,
                miopenDataType_t type);

float transpose_NCHW2CNHW(Handle& handle,
                          int n,
                          int c,
                          int h_in,
                          int w_in,
                          int h_out,
                          int w_out,
                          ConstData_t in,
                          Data_t out,
                          int in_offset,
                          int out_offset,
                          int h_stride,
                          int w_stride,
                          miopenDataType_t type);

float transpose_CNHW2NCHW(Handle& handle,
                          int n,
                          int c,
                          int h_out,
                          int w_out,
                          int h_in,
                          int w_in,
                          ConstData_t in,
                          Data_t out,
                          int in_offset,
                          int out_offset,
                          int h_stride,
                          int w_stride,
                          miopenDataType_t type);

float transpose_NCHW2Vec(Handle& handle,
                         int n,
                         int c,
                         int h,
                         int w,
                         ConstData_t in,
                         Data_t out,
                         int vec_size,
                         bool trans,
                         bool forward);

float transpose_packed_MN2NM(Handle& handle,
                             int m,
                             int n,
                             int in_offset,
                             int out_offset,
                             ConstData_t in,
                             Data_t out,
                             miopenDataType_t type);
} // namespace miopen

#endif // _MIOPEN_UTIL_HPP_
