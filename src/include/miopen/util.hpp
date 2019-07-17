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

#include <boost/range/adaptors.hpp>

namespace miopen {

struct Handle;

float Im2ColGPU(
    Handle& handle,
    std::size_t spatial_dim,
    ConstData_t im,
    std::size_t im_offset,
    std::size_t in_c,
    const decltype(boost::adaptors::slice(std::vector<std::size_t>(), 0, 1))& in_spatial,
    const decltype(boost::adaptors::slice(std::vector<std::size_t>(), 0, 1))& wei_spatial,
    const decltype(boost::adaptors::slice(std::vector<std::size_t>(), 0, 1))& out_spatial,
    const std::vector<int>& pad_spatial,
    const std::vector<int>& stride_spatial,
    const std::vector<int>& dilation_spatial,
    Data_t col,
    miopenDataType_t type);

float Col2ImGPU(
    Handle& handle,
    std::size_t spatial_dim,
    ConstData_t col,
    const decltype(boost::adaptors::slice(std::vector<std::size_t>(), 0, 1))& out_spatial,
    const decltype(boost::adaptors::slice(std::vector<std::size_t>(), 0, 1))& wei_spatial,
    const std::vector<int>& pad_spatial,
    const std::vector<int>& stride_spatial,
    const std::vector<int>& dilation_spatial,
    std::size_t in_c,
    const decltype(boost::adaptors::slice(std::vector<std::size_t>(), 0, 1))& in_spatial,
    Data_t im,
    std::size_t im_offset,
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
                         const std::vector<std::size_t>& lens,
                         ConstData_t in,
                         Data_t out,
                         std::size_t vec_size,
                         bool trans,
                         bool forward,
                         const void* alpha,
                         const void* beta);

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
