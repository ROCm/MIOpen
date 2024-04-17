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
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

template <typename TI, typename TO>
__device__ void padReflection2dFwdContiguous(const TI* __restrict__ input,
                                             TO* __restrict__ output,
                                             uint64_t output_size,
                                             long padding_left,
                                             long padding_top,
                                             const size_t in_H,
                                             const size_t in_W,
                                             const size_t output_size_1,
                                             const size_t output_size_2,
                                             const size_t output_size_3,
                                             const size_t input_stride_0,
                                             const size_t input_stride_1,
                                             const size_t input_stride_2,
                                             const size_t input_stride_3)
{
    const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= output_size)
        return;

    long n, c, h, w;
    ulong nch = (gid) / output_size_3;
    w         = (gid) % output_size_3;
    ulong nc  = nch / output_size_2;
    h         = nch % output_size_2;
    n         = nc / output_size_1;
    c         = nc % output_size_1;

    long in_start_x  = max(0L, -padding_left);
    long in_start_y  = max(0L, -padding_top);
    long out_start_x = max(0L, padding_left);
    long out_start_y = max(0L, padding_top);

    if(w < padding_left)
    {
        w = padding_left * 2 - w;
    }
    else if(!(padding_left <= w && w < in_W + padding_left))
    {
        w = (in_W + padding_left - 1) * 2 - w;
    }
    w = w - out_start_x + in_start_x;

    if(h < padding_top)
    {
        h = padding_top * 2 - h;
    }
    else if(!(padding_top <= h && h < in_H + padding_top))
    {
        h = (in_H + padding_top - 1) * 2 - h;
    }
    h = h - out_start_y + in_start_y;

    output[gid] = input[(input_stride_3 * (w)) + (input_stride_2 * (h)) + (input_stride_1 * (c)) +
                        (input_stride_0 * (n)) + 0];
}

extern "C" __global__ void PadReflection2dFwdContiguous(const INPUT_TYPE* __restrict__ input,
                                                        OUTPUT_TYPE* __restrict__ output,
                                                        uint64_t output_size,
                                                        long padding_left,
                                                        long padding_top,
                                                        const size_t in_H,
                                                        const size_t in_W,
                                                        const size_t output_size_1,
                                                        const size_t output_size_2,
                                                        const size_t output_size_3,
                                                        const size_t input_stride_0,
                                                        const size_t input_stride_1,
                                                        const size_t input_stride_2,
                                                        const size_t input_stride_3)
{
    padReflection2dFwdContiguous<INPUT_TYPE, OUTPUT_TYPE>(input,
                                                          output,
                                                          output_size,
                                                          padding_left,
                                                          padding_top,
                                                          in_H,
                                                          in_W,
                                                          output_size_1,
                                                          output_size_2,
                                                          output_size_3,
                                                          input_stride_0,
                                                          input_stride_1,
                                                          input_stride_2,
                                                          input_stride_3);
}