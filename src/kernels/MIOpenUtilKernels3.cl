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
#define PPCAT_NX(A, B) A##B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define TWO 2
#define FOUR 4
#define EIGHT 8

#if MIOPEN_USE_FP16 == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT half
#ifndef HALF_MAX
#define MAX_VAL 65504 /* max value */
#else
#define MAX_VAL HALF_MAX
#endif
#endif
#if MIOPEN_USE_FP32 == 1
#define _FLOAT float
#ifndef FLT_MAX
#define MAX_VAL 3.402823466e+38F /* max value */
#else
#define MAX_VAL FLT_MAX
#endif
#endif

#define _FLOAT2 PPCAT(_FLOAT, TWO)
#define _FLOAT4 PPCAT(_FLOAT, FOUR)
#define _FLOAT8 PPCAT(_FLOAT, EIGHT)

#include "math_ops.h"

#define MLO_OUT_CHANNEL_STRIDE_ALIGNED (MLO_OUT_CHANNEL_STRIDE / MLO_WRITE_UNIT)
#define MLO_OUT_STRIDE_ALIGNED (MLO_OUT_STRIDE / MLO_WRITE_UNIT)

#define MLO_IN_CHANNEL_STRIDE_ALIGNED (MLO_IN0_CHANNEL_STRIDE / MLO_WRITE_UNIT)
#define MLO_IN_STRIDE_ALIGNED (MLO_IN0_STRIDE / MLO_WRITE_UNIT)

#ifndef DATA_TYPE
#define DATA_TYPE _FLOAT
#endif

__attribute__((reqd_work_group_size(MLO_GRP0_SZ0, MLO_GRP0_SZ1, MLO_GRP0_SZ2))) __kernel void
SubSample(const __global DATA_TYPE* __restrict in, __global DATA_TYPE* __restrict out)
{
    uint stack_pos = get_global_id(0);
    uint batch_id  = get_global_id(1);
    uint map_id    = iDiv(stack_pos, MLO_OUT_CHANNEL_STRIDE_ALIGNED);
    uint pix_pos   = iMod(stack_pos, map_id, MLO_OUT_CHANNEL_STRIDE_ALIGNED);
    uint out_y     = iDiv(pix_pos, MLO_OUT_STRIDE_ALIGNED);
    uint out_x     = iMod(pix_pos, out_y, MLO_OUT_STRIDE_ALIGNED) * MLO_WRITE_UNIT;

    uint out_off = batch_id * MLO_IN_BATCH_STRIDE + stack_pos * MLO_WRITE_UNIT;
    uint in_y    = out_y * MLO_FILTER0_STRIDE1;
    uint in_x    = out_x * MLO_FILTER0_STRIDE0;
    uint in_off  = batch_id * MLO_IN0_BATCH_STRIDE + map_id * MLO_IN0_CHANNEL_STRIDE +
                  in_y * MLO_IN0_STRIDE + in_x;

    const __global DATA_TYPE* in_ptr = &in[in_off];
    __global DATA_TYPE* out_ptr      = &out[out_off];

    for(uint i = 0; i < MLO_WRITE_UNIT; ++i, in_ptr += MLO_FILTER0_STRIDE0, out_ptr++)
    {
        *out_ptr = *in_ptr;
    }
}

__attribute__((reqd_work_group_size(MLO_GRP0_SZ0, MLO_GRP0_SZ1, MLO_GRP0_SZ2))) __kernel void
UpSample(const __global DATA_TYPE* __restrict in, __global DATA_TYPE* __restrict out)
{
    uint stack_pos = get_global_id(0);
    uint batch_id  = get_global_id(1);
    uint map_id    = iDiv(stack_pos, MLO_IN_CHANNEL_STRIDE_ALIGNED);
    uint pix_pos   = iMod(stack_pos, map_id, MLO_IN_CHANNEL_STRIDE_ALIGNED);
    uint in_y      = iDiv(pix_pos, MLO_IN_STRIDE_ALIGNED);
    uint in_x      = iMod(pix_pos, in_y, MLO_IN_STRIDE_ALIGNED) * MLO_WRITE_UNIT;

    uint in_off  = batch_id * MLO_IN_BATCH_STRIDE + stack_pos * MLO_WRITE_UNIT;
    uint out_y   = in_y * MLO_FILTER0_STRIDE1;
    uint out_x   = in_x * MLO_FILTER0_STRIDE0;
    uint out_off = batch_id * MLO_IN0_BATCH_STRIDE + map_id * MLO_OUT_CHANNEL_STRIDE +
                   out_y * MLO_OUT_STRIDE + out_x;

    const __global DATA_TYPE* in_ptr = &in[in_off];
    __global DATA_TYPE* out_ptr      = &out[out_off];

    for(uint i = 0; i < MLO_WRITE_UNIT; ++i, in_ptr++, out_ptr += MLO_FILTER0_STRIDE0)
    {
        *out_ptr = *in_ptr;
    }
}
