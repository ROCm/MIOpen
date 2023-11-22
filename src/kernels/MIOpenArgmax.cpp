/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

extern "C" __global__ void ArgmaxFwdContiguous(const FLOAT* __restrict__ x,
                                               int32_t* __restrict__ y,
                                               uint64_t output_numel,
                                               int32_t reduce_size,
                                               uint64_t inner_size,
                                               int32_t dim)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= output_numel)
        return;

    uint64_t input_idx = (gid / inner_size) * inner_size * reduce_size + gid % inner_size;

    int32_t max_idx = 0;
    FLOAT max       = x[input_idx];

    for(int32_t k = 1; k < reduce_size; ++k)
    {
        input_idx += inner_size;
        FLOAT val = x[input_idx];
        if(max < val)
        {
            max     = val;
            max_idx = k;
        }
    }

    y[gid] = max_idx;
}

extern "C" __device__ void LocalReduceMaxAndIndex(volatile FLOAT* data,
                                                  volatile int32_t* index,
                                                  int32_t local_idx,
                                                  int32_t reduce_size)
{
    int32_t end_idx = reduce_size;
    for(int32_t i = (reduce_size / 2) + reduce_size % 2; i > 1; i = i / 2 + i % 2)
    {
        if(local_idx < i && local_idx + i < end_idx)
        {
            if(data[local_idx] < data[local_idx + i])
            {
                data[local_idx]  = data[local_idx + i];
                index[local_idx] = index[local_idx + i];
            }
        }
        end_idx = i;
        __syncthreads();
    }
    if(local_idx == 0)
    {
        if(data[0] < data[1])
        {
            data[0]  = data[1];
            index[0] = index[1];
        }
    }
    __syncthreads();
}

extern "C" __global__ void ArgmaxFwdContiguousLastDim(const FLOAT* __restrict__ x,
                                                      int32_t* __restrict__ y,
                                                      uint64_t output_numel,
                                                      int32_t reduce_size,
                                                      uint64_t inner_size,
                                                      int32_t dim)
{
    uint64_t gid       = blockIdx.x;
    const uint64_t lid = threadIdx.x;

    int32_t loop_end = (reduce_size / LOCAL_SIZE) + 1;
    int32_t loop_idx = static_cast<int32_t>(lid);

    __shared__ FLOAT ltmp1[LOCAL_SIZE];
    __shared__ int32_t ltmp2[LOCAL_SIZE];

    while(gid < output_numel && lid < reduce_size)
    {
        uint64_t input_idx = gid * reduce_size + lid;

        ltmp1[lid] = x[input_idx];
        input_idx += LOCAL_SIZE;
        loop_idx += LOCAL_SIZE;
        for(int32_t k = 1; k < loop_end && loop_idx < reduce_size;
            k++, input_idx += LOCAL_SIZE, loop_idx += LOCAL_SIZE)
        {
            FLOAT val = x[input_idx];
            if(ltmp1[lid] < val)
            {
                ltmp1[lid] = val;
                ltmp2[lid] = loop_idx;
            }
        }

        __syncthreads();
        LocalReduceMaxAndIndex(ltmp1, ltmp2, lid, LOCAL_SIZE);

        if(lid == 0)
        {
            FLOAT max_val = ltmp1[0];
            y[gid]        = ltmp2[0];
        }
        gid += gridDim.x;

        ltmp1[lid] = -MAX_VAL;
        ltmp2[lid] = static_cast<int32_t>(lid);
        loop_idx   = static_cast<int32_t>(lid);
    }
}
