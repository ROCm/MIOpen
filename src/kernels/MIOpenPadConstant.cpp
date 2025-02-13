/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#include <cstddef>
#include <cstdio>
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#endif

#include "float_types.h"
#include "tensor_view_5d.hpp"

template <typename TI, typename TO>
__device__ void padconstantfwd(const TI* __restrict__ x,
                               TO* __restrict__ y,
                               const tensor_view_5d_t x_tv,
                               const tensor_view_5d_t y_tv,
                               const padding_5d_t padding,
                               const size_t output_size,
                               TO value)
{
    TO padding_value   = value;
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= output_size)
        return;

    int64_t o[5];
    getNCDHW(o, gid, y_tv.size);
    bool flag = true;

    for(int i = 0; i < 5; ++i)
    {
        o[i] = o[i] - padding.val[2 * i];
        flag *= (o[i] >= 0) && (o[i] < x_tv.size[i]);
    }

    TO val = flag ? get5DValueAt<TO>(x, x_tv.stride, o[0], o[1], o[2], o[3], o[4]) : padding_value;
    set5DValueAt<TO>(y, y_tv, gid, val);
}

template <typename TI, typename TO>
__device__ void padconstantfwdcontiguous(const TI* __restrict__ x,
                                         TO* __restrict__ y,
                                         const tensor_view_5d_t x_tv,
                                         const tensor_view_5d_t y_tv,
                                         const padding_5d_t padding,
                                         const size_t output_size,
                                         TO value)
{
    TO padding_value   = value;
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= output_size)
        return;

    int64_t o[5];
    getNCDHW(o, gid, y_tv.size);
    bool flag = true;

    for(int i = 0; i < 5; ++i)
    {
        o[i] = o[i] - padding.val[2 * i];
        flag *= (o[i] >= 0) && (o[i] < x_tv.size[i]);
    }

    y[gid] = flag ? get5DValueAt<TO>(x, x_tv.stride, o[0], o[1], o[2], o[3], o[4]) : padding_value;
}

template <typename TI, typename TO>
__device__ void padconstantbwd(TI* __restrict__ dx,
                               const TO* __restrict__ y_grad,
                               const tensor_view_5d_t dx_tv,
                               const tensor_view_5d_t y_grad_tv,
                               const padding_5d_t padding,
                               const size_t input_size)

{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= input_size)
        return;

    int64_t o[5];
    getNCDHW(o, gid, dx_tv.size);
    bool flag = true;

    for(int i = 0; i < 5; ++i)
    {
        o[i] = o[i] + padding.val[2 * i];
        flag *= (o[i] >= 0) && (o[i] < y_grad_tv.size[i]);
    }

    TI val = flag ? get5DValueAt<TI>(y_grad, y_grad_tv.stride, o[0], o[1], o[2], o[3], o[4])
                  : static_cast<TI>(0);
    set5DValueAt<TI>(dx, dx_tv, gid, val);
}

template <typename TI, typename TO>
__device__ void padconstantbwdcontiguous(TI* __restrict__ dx,
                                         const TO* __restrict__ y_grad,
                                         const tensor_view_5d_t dx_tv,
                                         const tensor_view_5d_t y_grad_tv,
                                         const padding_5d_t padding,
                                         const size_t input_size)

{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= input_size)
        return;

    int64_t o[5];
    getNCDHW(o, gid, dx_tv.size);
    bool flag = true;

    for(int i = 0; i < 5; ++i)
    {
        o[i] = o[i] + padding.val[2 * i];
        flag *= (o[i] >= 0) && (o[i] < y_grad_tv.size[i]);
    }

    dx[gid] = flag ? get5DValueAt<TI>(y_grad, y_grad_tv.stride, o[0], o[1], o[2], o[3], o[4])
                   : static_cast<TI>(0);
}

extern "C" __global__ void PadConstantFwd(const INPUT_TYPE* __restrict__ x,
                                          OUTPUT_TYPE* __restrict__ y,
                                          const tensor_view_5d_t x_tv,
                                          const tensor_view_5d_t y_tv,
                                          const padding_5d_t padding,
                                          const size_t output_size,
                                          FLOAT_ACCUM value)
{
    padconstantfwd<INPUT_TYPE, OUTPUT_TYPE>(x, y, x_tv, y_tv, padding, output_size, value);
}

extern "C" __global__ void PadConstantFwdContiguous(const INPUT_TYPE* __restrict__ x,
                                                    OUTPUT_TYPE* __restrict__ y,
                                                    const tensor_view_5d_t x_tv,
                                                    const tensor_view_5d_t y_tv,
                                                    const padding_5d_t padding,
                                                    const size_t output_size,
                                                    FLOAT_ACCUM value)
{
    padconstantfwdcontiguous<INPUT_TYPE, OUTPUT_TYPE>(
        x, y, x_tv, y_tv, padding, output_size, value);
}

extern "C" __global__ void PadConstantBwd(OUTPUT_TYPE* __restrict__ dx,
                                          const INPUT_TYPE* __restrict__ y_grad,
                                          const tensor_view_5d_t dx_tv,
                                          const tensor_view_5d_t y_grad_tv,
                                          const padding_5d_t padding,
                                          const size_t input_size)
{
    padconstantbwd<OUTPUT_TYPE, INPUT_TYPE>(dx, y_grad, dx_tv, y_grad_tv, padding, input_size);
}

extern "C" __global__ void PadConstantBwdContiguous(OUTPUT_TYPE* __restrict__ dx,
                                                    const INPUT_TYPE* __restrict__ y_grad,
                                                    const tensor_view_5d_t dx_tv,
                                                    const tensor_view_5d_t y_grad_tv,
                                                    const padding_5d_t padding,
                                                    const size_t input_size)
{
    padconstantbwdcontiguous<OUTPUT_TYPE, INPUT_TYPE>(
        dx, y_grad, dx_tv, y_grad_tv, padding, input_size);
}
