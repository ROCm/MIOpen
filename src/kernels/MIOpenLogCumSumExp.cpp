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

template <uint64_t LOCAL_SIZE>
__device__ inline void CumulativeReductionSumScan(const int& lid, FLOAT_ACCUM* cumsum)
{
    // reduction
    int stride = 1;
    while(stride <= LOCAL_SIZE)
    {
        int idx = (lid + 1) * stride * 2 - 1;
        if(idx < LOCAL_SIZE)
            cumsum[idx] += cumsum[idx - stride];
        stride *= 2;
        __syncthreads();
    }

    // post scan
    stride = LOCAL_SIZE / 2;
    while(stride > 0)
    {
        int idx = (lid + 1) * stride * 2 - 1;
        if((idx + stride) < LOCAL_SIZE)
            cumsum[idx + stride] += cumsum[idx];
        stride /= 2;
        __syncthreads();
    }
}

template <typename DTYPE, uint64_t LOCAL_SIZE>
__device__ void LogCumSumExpForwardContiguousSmallLastDim(const DTYPE* __restrict__ input,
                                                          DTYPE* __restrict__ output,
                                                          const uint64_t reduce_size,
                                                          const bool exclusive,
                                                          const bool reverse)
{
    /*
     * input: packed tensor with stride[last_dim]=1
     * output: the same as input
     * reduce_size = input.size[last_dim]
     * exclusive: TRUE to exclude input[i] when calculate output[i]
     * reverse: reverse the operating order
     *
     * cumulative dimension = last dim
     * blockSize = {1, LOCAL_SIZE}
     * gridSize = {Number of input elements / input.size[last_dim], input.size[last_dim]}
     */

    __shared__ FLOAT_ACCUM otmp[LOCAL_SIZE];

    int lid = threadIdx.y;

    auto xid = blockIdx.x * blockDim.x + threadIdx.x;
    auto yid = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = yid - exclusive;
    if(0 <= idx && idx < reduce_size - exclusive)
    {
        idx       = (reverse ? reduce_size - idx - 1 : idx);
        otmp[lid] = exp(CVT_FLOAT2ACCUM(input[xid * reduce_size + idx]));
    }
    else
    {
        otmp[lid] = 0;
    }
    __syncthreads();

    CumulativeReductionSumScan<LOCAL_SIZE>(lid, otmp);

    idx = yid;
    if(idx < reduce_size)
    {
        idx                             = (reverse ? reduce_size - idx - 1 : idx);
        output[xid * reduce_size + idx] = CVT_ACCUM2FLOAT(log(otmp[lid]));
    }
}

extern "C" __global__ void LogCumSumExpForwardContiguousSmallLastDim(const FLOAT* input,
                                                                     FLOAT* output,
                                                                     const uint64_t reduce_size,
                                                                     const bool exclusive,
                                                                     const bool reverse)
{
    // instantiate the kernel
    LogCumSumExpForwardContiguousSmallLastDim<FLOAT, REDUCE_SIZE>(
        input, output, reduce_size, exclusive, reverse);
}

extern "C" __global__ void InitLogGradContiguous(const FLOAT* output_grad,
                                                 const FLOAT* output,
                                                 FLOAT_ACCUM* log_grad_positive,
                                                 FLOAT_ACCUM* log_grad_negative,
                                                 const uint64_t N)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= N)
        return;

    auto output_grad_v = CVT_FLOAT2ACCUM(output_grad[gid]);
    auto output_v      = CVT_FLOAT2ACCUM(output[gid]);

    log_grad_positive[gid] = (output_grad_v > 0 ? log(output_grad_v) - output_v : log(0));
    log_grad_negative[gid] = (output_grad_v < 0 ? log(-output_grad_v) - output_v : log(0));
}

extern "C" __global__ void
LogCumSumExp1dBackwardStep2Contiguous(const FLOAT_ACCUM* pos_reverse_logcumsumexp,
                                      const FLOAT_ACCUM* neg_reverse_logcumsumexp,
                                      const FLOAT* input,
                                      FLOAT* input_grad,
                                      const uint64_t N,
                                      const uint64_t reduce_size,
                                      const bool exclusive)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= N)
        return;
    if(gid % reduce_size + exclusive >= reduce_size)
    {
        input_grad[gid] = CVT_FP32_2FLOAT(0.0f);
        return;
    }

    FLOAT_ACCUM input_v = CVT_FLOAT2ACCUM(input[gid]);

    FLOAT_ACCUM output_pos = exp(pos_reverse_logcumsumexp[gid + exclusive] + input_v);
    FLOAT_ACCUM output_neg = exp(neg_reverse_logcumsumexp[gid + exclusive] + input_v);

    input_grad[gid] = CVT_ACCUM2FLOAT(output_pos - output_neg);
}

template <typename DTYPE, uint64_t LOCAL_SIZE>
__device__ void LogCumSumExpBackwardContiguousSmallLastDim(const DTYPE* __restrict__ input,
                                                           const DTYPE* __restrict__ output,
                                                           const DTYPE* __restrict__ output_grad,
                                                           DTYPE* __restrict__ input_grad,
                                                           const uint64_t reduce_size,
                                                           const bool exclusive,
                                                           const bool reverse)
{
    /*
     * input_grad: packed tensor with stride[last_dim]=1
     * input, output, output_grad: the same as input
     * reduce_size = input.size[last_dim]
     * exclusive: TRUE to exclude input[i] when calculate output[i]
     * reverse: reverse the operating order
     *
     * cumulative dimension = last dim
     * blockSize = {1, LOCAL_SIZE}
     * gridSize = {Number of input elements / input.size[last_dim], input.size[last_dim]}
     */

    __shared__ FLOAT_ACCUM otmp[LOCAL_SIZE];
    int lid  = threadIdx.y;
    auto xid = blockIdx.x * blockDim.x + threadIdx.x;
    auto yid = blockIdx.y * blockDim.y + threadIdx.y;

    FLOAT_ACCUM output_v, output_grad_v = 0;
    if(exclusive <= yid && yid < reduce_size)
    {
        auto idx      = (reverse ? reduce_size - yid - 1 : yid);
        output_grad_v = CVT_FLOAT2ACCUM(output_grad[xid * reduce_size + idx]);
        output_v      = CVT_FLOAT2ACCUM(output[xid * reduce_size + idx]);
    }

    // LogCumSumExp pos_reverse_logcumsumexp
    otmp[lid] = 0;
    if(output_grad_v > 0)
        otmp[lid] = exp(log(output_grad_v) - output_v);
    __syncthreads();
    CumulativeReductionSumScan<LOCAL_SIZE>(lid, otmp);
    auto pos_reverse_logcumsumexp =
        ((reverse ? exclusive <= lid : lid + exclusive < reduce_size)
             ? log(otmp[reverse ? reduce_size - (lid - exclusive) - 1 : (lid + exclusive)])
             : 0.0f);
    //------------------------------------------------------------------------------------

    __syncthreads();

    // LogCumSumExp neg_reverse_logcumsumexp
    otmp[lid] = 0;
    if(output_grad_v < 0)
        otmp[lid] = exp(log(-output_grad_v) - output_v);
    __syncthreads();
    CumulativeReductionSumScan<LOCAL_SIZE>(lid, otmp);
    auto neg_reverse_logcumsumexp =
        ((reverse ? exclusive <= lid : lid + exclusive < reduce_size)
             ? log(otmp[reverse ? reduce_size - (lid - exclusive) - 1 : (lid + exclusive)])
             : 0.0f);
    //------------------------------------------------------------------------------------

    // Calculate Input Gradient
    if(yid < reduce_size)
    {
        auto idx               = yid;
        auto input_v           = CVT_FLOAT2ACCUM(input[xid * reduce_size + idx]);
        FLOAT_ACCUM output_pos = exp(pos_reverse_logcumsumexp + input_v);
        FLOAT_ACCUM output_neg = exp(neg_reverse_logcumsumexp + input_v);

        input_grad[xid * reduce_size + idx] = CVT_ACCUM2FLOAT(output_pos - output_neg);
    }
    //------------------------------------------------------------------------------------
}

extern "C" __global__ void LogCumSumExpBackwardContiguousSmallLastDim(const FLOAT* input,
                                                                      const FLOAT* output,
                                                                      const FLOAT* output_grad,
                                                                      FLOAT* input_grad,
                                                                      const uint64_t reduce_size,
                                                                      const bool exclusive,
                                                                      const bool reverse)
{
    LogCumSumExpBackwardContiguousSmallLastDim<FLOAT, REDUCE_SIZE>(
        input, output, output_grad, input_grad, reduce_size, exclusive, reverse);
}
