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
                                               const int& lid,
                                               FLOAT_ACCUM* __restrict__ a,
                                               Ts* __restrict__... b)
{
    // reduction
    int stride = 1;
    while(stride <= LOCAL_SIZE)
    {
        int idx = (lid + 1) * stride * 2 - 1;
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
        int idx = (lid + 1) * stride * 2 - 1;
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
                                                            int* __restrict__ indices,
                                                            const uint64_t reduce_size,
                                                            const bool exclusive,
                                                            const bool reverse)
{
    __shared__ FLOAT_ACCUM otmp[LOCAL_SIZE];
    int* itmp = nullptr;
    if(indices)
    {
        __shared__ int _itmp[LOCAL_SIZE];
        itmp = _itmp;
    }

    int lid = threadIdx.y;

    auto xid = blockIdx.x * blockDim.x + threadIdx.x;
    auto yid = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = yid - exclusive;
    if(0 <= idx && idx < reduce_size - exclusive && idx < reduce_size)
    {
        idx       = (reverse ? reduce_size - idx - 1 : idx);
        otmp[lid] = CVT_FLOAT2ACCUM(input[xid * reduce_size + idx]);
        if(indices)
            itmp[lid] = idx;
    }
    else
    {
        otmp[lid] = reduce_func<OP, FLOAT_ACCUM>{}.START_VAL;
        if(indices)
            itmp[lid] = (reverse ? reduce_size - idx - 1 : idx);
    }
    __syncthreads();

    if(indices)
        CumulativeReductionScan<OP, LOCAL_SIZE, int>(reverse, lid, otmp, itmp);
    else
        CumulativeReductionScan<OP, LOCAL_SIZE>(reverse, lid, otmp);

    idx = yid;
    if(idx < reduce_size)
    {
        idx = (reverse ? reduce_size - idx - 1 : idx);
        if(output)
            output[xid * reduce_size + idx] = CVT_ACCUM2FLOAT(otmp[lid]);
        if(indices)
            indices[xid * reduce_size + idx] = itmp[lid];
    }
}

extern "C" __global__ void CumulativeReductionForwardContiguousLastDim(const INPUT_TYPE* input,
                                                                       OUTPUT_TYPE* output,
                                                                       int* indices,
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
