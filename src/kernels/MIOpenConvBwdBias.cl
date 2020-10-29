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
#include "float_types.h"
#include "math_ops.h"

static inline void
ReduceKernel(__local _FLOAT_ACCUM* lcl_mem, int sum_stride, int unit_id, int unit_len)
{
    _FLOAT_ACCUM sum = 0.0f;
    int lcl_offset   = unit_id * unit_len;
    for(int i = 0; i < unit_len; i += sum_stride)
    {
        sum += lcl_mem[lcl_offset + i];
    }
    lcl_mem[lcl_offset] = sum;
}

__attribute__((reqd_work_group_size(MLO_CONVBWD_GROUP_SZ0, MLO_CONVBWD_GROUP_SZ1, 1))) __kernel void
MIOpenConvBwdB(const __global _FLOAT* top_df,
               __global _FLOAT* bias_df,
               uint bias_c,
               uint top_str_c,
               uint top_str_b,
               uint num_spatial_work,
               uint off_pix,
               uint total_work)
{
    int lid = (int)get_local_id(0);
    __local _FLOAT_ACCUM lcl_sum[MLO_CONVBWDB_LCL_MEMSZ];

    int gid = get_group_id(1);
    if(gid < bias_c)
    {
        _FLOAT_ACCUM sum = 0.0f;

        for(int j = lid; j < total_work; j += MLO_CONVBWD_GROUP_SZ0)
        {
            int map_id  = iDiv(j, num_spatial_work);
            int read_id = iMod(j, map_id, num_spatial_work);
            int glb_top_df_offset =
                gid * top_str_c + (map_id * top_str_b) + (read_id * MLO_CONVBWDB_UNITSIZE);

            int upper_bound =
                off_pix > 0 && read_id == num_spatial_work - 1 ? off_pix : MLO_CONVBWDB_UNITSIZE;
            for(int k = 0; k < upper_bound; k++)
                sum += CVT_FLOAT2ACCUM(top_df[glb_top_df_offset + k]);
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

        bias_df[gid] = CVT_ACCUM2FLOAT(lcl_sum[0]);
    }
    gid += get_global_size(1);
    for(; gid < bias_c; gid += get_global_size(1))
    {
        _FLOAT_ACCUM sum = 0.0f;

        for(int j = lid; j < total_work; j += MLO_CONVBWD_GROUP_SZ0)
        {
            int map_id  = iDiv(j, num_spatial_work);
            int read_id = iMod(j, map_id, num_spatial_work);
            int glb_top_df_offset =
                gid * top_str_c + (map_id * top_str_b) + (read_id * MLO_CONVBWDB_UNITSIZE);

            int upper_bound =
                off_pix > 0 && read_id == num_spatial_work - 1 ? off_pix : MLO_CONVBWDB_UNITSIZE;
            for(int k = 0; k < upper_bound; k++)
                sum += CVT_FLOAT2ACCUM(top_df[glb_top_df_offset + k]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
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

        bias_df[gid] = CVT_ACCUM2FLOAT(lcl_sum[0]);
    }
}
