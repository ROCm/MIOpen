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
#include "MIOpenCumulativeReduction.hpp"

template <CumulativeReductionOp_t OP, uint64_t LOCAL_SIZE, typename... Ts>
__device__ inline void CumulativeReductionScan(const bool& reverse,
                                               const uint64_t& lid,
                                               FLOAT_ACCUM* __restrict__ a,
                                               Ts* __restrict__... b)
{
    // reduction
    uint64_t stride = 1;
    while(stride <= LOCAL_SIZE)
    {
        uint64_t idx = (lid + 1) * stride * 2 - 1;
        if(idx < LOCAL_SIZE)
            reduce_func<OP, FLOAT_ACCUM, Ts...>{}.calculate(
                !reverse, a[idx], a[idx - stride], b[idx]..., b[idx - stride]...);
        stride *= 2;
        __syncthreads();
    }

    // post scan
    stride = LOCAL_SIZE / 2;
    while(stride > 0)
    {
        uint64_t idx = (lid + 1) * stride * 2 - 1;
        if((idx + stride) < LOCAL_SIZE)
            reduce_func<OP, FLOAT_ACCUM, Ts...>{}.calculate(
                !reverse, a[idx + stride], a[idx], b[idx + stride]..., b[idx]...);
        stride /= 2;
        __syncthreads();
    }
}

template <typename TI, typename TO, CumulativeReductionOp_t OP, uint64_t LOCAL_SIZE>
__device__ void CumulativeReductionForwardContiguousLastDim(const TI* __restrict__ input,
                                                            TO* __restrict__ output,
                                                            int64_t* __restrict__ indices,
                                                            const uint64_t reduce_size,
                                                            const bool exclusive,
                                                            const bool reverse)
{
    /*
     * input = packed tensor with stride[last_dim]=1, output: the same as input, indices: the same
as input
     * reduce_size = input.size[last_dim]
     * exclusive: TRUE to exclude input[i] when calculate output[i]
     * reverse: reverse the operating order
     *
     * cumulative dimension = last dim
     * blockSize = {1, LOCAL_SIZE}
     * gridSize = {Number of input elements / input.size[last_dim], input.size[last_dim]}
     */

    __shared__ FLOAT_ACCUM otmp[LOCAL_SIZE];
    int64_t* itmp = nullptr;
    if(indices)
    {
        __shared__ int64_t _itmp[LOCAL_SIZE];
        itmp = _itmp;
    }

    uint64_t lid = threadIdx.y;

    uint64_t xid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t yid = blockIdx.y * blockDim.y + threadIdx.y;

    if(exclusive <= yid && yid < reduce_size)
    {
        int64_t idx = (reverse ? reduce_size - static_cast<int64_t>(yid) + exclusive - 1
                               : static_cast<int64_t>(yid) - exclusive);
        otmp[lid]   = CVT_FLOAT2ACCUM(input[xid * reduce_size + idx]);
        if(indices)
            itmp[lid] = idx;
    }
    else
    {
        otmp[lid] = reduce_func<OP, FLOAT_ACCUM>{}.START_VAL;
        if(indices)
            itmp[lid] = (reverse ? reduce_size - static_cast<int64_t>(yid) + exclusive - 1
                                 : static_cast<int64_t>(yid) - exclusive);
    }
    __syncthreads();

    if(indices)
        CumulativeReductionScan<OP, LOCAL_SIZE, int64_t>(reverse, lid, otmp, itmp);
    else
        CumulativeReductionScan<OP, LOCAL_SIZE>(reverse, lid, otmp);

    if(yid < reduce_size)
    {
        int64_t idx =
            (reverse ? reduce_size - static_cast<int64_t>(yid) - 1 : static_cast<int64_t>(yid));
        if(output)
            output[xid * reduce_size + idx] = CVT_ACCUM2FLOAT(otmp[lid]);
        if(indices)
            indices[xid * reduce_size + idx] = itmp[lid];
    }
}

extern "C" __global__ void CumulativeReductionForwardContiguousLastDim(const INPUT_TYPE* input,
                                                                       OUTPUT_TYPE* output,
                                                                       int64_t* indices,
                                                                       const uint64_t reduce_size,
                                                                       const bool exclusive,
                                                                       const bool reverse)
{
    // instantiate the kernel
    CumulativeReductionForwardContiguousLastDim<INPUT_TYPE,
                                                OUTPUT_TYPE,
                                                (CumulativeReductionOp_t)OP_TYPE,
                                                REDUCE_SIZE>(
        input, output, indices, reduce_size, exclusive, reverse);
}
