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
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

template <typename TI, typename TO>
__device__ void ropefwdcontiguous(const TI* __restrict__ x,
                                  const TI* __restrict__ cos,
                                  const TI* __restrict__ sin,
                                  TO* __restrict__ y,
                                  uint64_t output_numel,
                                  uint64_t rotary_numel)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= output_numel)
        return;

    FLOAT_ACCUM input = CVT_FLOAT2ACCUM(x[gid]);
    uint64_t freqs_id = gid % rotary_numel;
    FLOAT_ACCUM input_rotate_half =
        (gid % 2 == 0) ? CVT_FLOAT2ACCUM(-x[(gid + 1)]) : CVT_FLOAT2ACCUM(x[gid - 1]);

    FLOAT_ACCUM cos_val = CVT_FLOAT2ACCUM(cos[freqs_id]);
    FLOAT_ACCUM sin_val = CVT_FLOAT2ACCUM(sin[freqs_id]);

    FLOAT_ACCUM val = (input * cos_val) + (input_rotate_half * sin_val);

    y[gid] = CVT_ACCUM2FLOAT(val);
}

template <typename TI, typename TO>
__device__ void ropebwdcontiguous(const TI* __restrict__ dy,
                                  const TI* __restrict__ cos,
                                  const TI* __restrict__ sin,
                                  TO* __restrict__ dx,
                                  uint64_t output_numel,
                                  uint64_t rotary_numel)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= output_numel)
        return;

    uint64_t freqs_id = gid % rotary_numel;

    FLOAT_ACCUM output_grad = CVT_FLOAT2ACCUM(dy[gid]);
    FLOAT_ACCUM output_grad_rotate_half =
        (gid % 2 == 0) ? CVT_FLOAT2ACCUM(dy[(gid + 1)]) : CVT_FLOAT2ACCUM(-dy[(gid - 1)]);

    FLOAT_ACCUM cos_val = CVT_FLOAT2ACCUM(cos[freqs_id]);
    FLOAT_ACCUM sin_val = (freqs_id % 2 == 0) ? CVT_FLOAT2ACCUM(sin[freqs_id + 1])
                                              : CVT_FLOAT2ACCUM(sin[freqs_id - 1]);

    FLOAT_ACCUM val = (output_grad * cos_val) + (output_grad_rotate_half * sin_val);

    dx[gid] = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void RoPEFwdContiguous(const INPUT_TYPE* __restrict__ x,
                                             const INPUT_TYPE* __restrict__ cos,
                                             const INPUT_TYPE* __restrict__ sin,
                                             OUTPUT_TYPE* __restrict__ y,
                                             uint64_t output_numel,
                                             uint64_t rotary_numel)
{
    // instantiate the kernel
    ropefwdcontiguous<INPUT_TYPE, OUTPUT_TYPE>(x, cos, sin, y, output_numel, rotary_numel);
}

extern "C" __global__ void RoPEBwdContiguous(const INPUT_TYPE* __restrict__ dy,
                                             const INPUT_TYPE* __restrict__ cos,
                                             const INPUT_TYPE* __restrict__ sin,
                                             OUTPUT_TYPE* __restrict__ dx,
                                             uint64_t output_numel,
                                             uint64_t rotary_numel)
{
    // instantiate the kernel
    ropebwdcontiguous<INPUT_TYPE, OUTPUT_TYPE>(dy, cos, sin, dx, output_numel, rotary_numel);
}
