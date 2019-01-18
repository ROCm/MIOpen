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
#endif
#if MIOPEN_USE_FP32 == 1
#define _FLOAT float
#endif

#define _FLOAT2 PPCAT(_FLOAT, TWO)
#define _FLOAT4 PPCAT(_FLOAT, FOUR)
#define _FLOAT8 PPCAT(_FLOAT, EIGHT)

#include "math_ops.h"

static inline void ReduceKernel(__local float* lcl_mem, int sum_stride, int unit_id, int unit_len)
{
    float sum      = 0.0f;
    int lcl_offset = unit_id * unit_len;
    for(int i = 0; i < unit_len; i += sum_stride)
    {
        sum += lcl_mem[lcl_offset + i];
    }
    lcl_mem[lcl_offset] = sum;
}

__attribute__((reqd_work_group_size(MLO_CONVBWD_GROUP_SZ0, MLO_CONVBWD_GROUP_SZ1, 1))) __kernel void
MIOpenConvBwdB(const __global _FLOAT* top_df, __global _FLOAT* bias_df)
{
    int lid        = (int)get_local_id(0);
    int output_map = get_group_id(1);

    __local float lcl_sum[MLO_CONVBWDB_LCL_MEMSZ];
    float sum = 0.0f;

    for(int j = lid; j < MLO_WK_SIZE * MLO_OUT_BATCH_SZ; j += MLO_CONVBWD_GROUP_SZ0)
    {
        int map_id            = iDiv(j, MLO_WK_SIZE);
        int read_id           = iMod(j, map_id, MLO_WK_SIZE);
        int glb_top_df_offset = output_map * MLO_OUT_CHANNEL_STRIDE +
                                (map_id * MLO_OUT_BATCH_STRIDE) + (read_id * MLO_CONVBWDB_UNITSIZE);
#if MLO_N_PIX_OFF
        if(read_id == MLO_WK_SIZE - 1)
        {
            for(int k = 0; k < MLO_N_PIX_OFF; k++)
                sum += (float)(top_df[glb_top_df_offset + k]);
        }
        else
        {
            for(int k = 0; k < MLO_CONVBWDB_UNITSIZE; k++)
                sum += (float)(top_df[glb_top_df_offset + k]);
        }
#else
        for(int k = 0; k < MLO_CONVBWDB_UNITSIZE; k++)
            sum += (float)(top_df[glb_top_df_offset + k]);
#endif
    }
    lcl_sum[lid] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(lid < (MLO_CONVBWD_GROUP_SZ0 >> 2))
    {
        ReduceKernel(lcl_sum, 1, lid, 4);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(lid < (MLO_CONVBWD_GROUP_SZ0 >> 4))
    {
        ReduceKernel(lcl_sum, 4, lid, 16);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(lid == 0)
    {
        ReduceKernel(lcl_sum, 16, lid, 256);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    bias_df[output_map] = (_FLOAT)(lcl_sum[0]);
}
