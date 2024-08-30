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
#include "tensor_view.hpp"
#include "block_reduce.hpp"

template <typename TO>
__device__ void
ReduceSum(const FLOAT_ACCUM* input, TO* output, uint64_t N, tensor_view_t<1> output_tv)
{
    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    FLOAT_ACCUM val = gid < N ? input[gid] : CVT_FP32_2ACCUM(0.0f);
    val             = block_reduce<BinaryOp_t::Add, REDUCE_SIZE, ReduceThreadDim::X>(val);

    if(threadIdx.x == 0)
        output[output_tv.get_tensor_view_idx({blockIdx.x})] = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void ReduceSum(const FLOAT_ACCUM* __restrict__ input,
                                     FLOAT* __restrict__ output,
                                     uint64_t N,
                                     tensor_view_t<1> output_tv)
{
    // instantiate the kernel
    ReduceSum<FLOAT>(input, output, N, output_tv);
}

extern "C" __global__ void ReduceSumFLOATACCUM(const FLOAT_ACCUM* __restrict__ input,
                                               FLOAT_ACCUM* __restrict__ output,
                                               uint64_t N)
{
    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    FLOAT_ACCUM val = gid < N ? input[gid] : 0.0f;
    val             = block_reduce<BinaryOp_t::Add, REDUCE_SIZE, ReduceThreadDim::X>(val);

    if(threadIdx.x == 0)
        output[blockIdx.x] = val;
}

template <typename TO>
__device__ void Reduce1dSum(const FLOAT_ACCUM* __restrict__ input,
                            TO* __restrict__ output,
                            uint64_t output_numel,
                            uint64_t inner_size,
                            uint64_t outer_size,
                            tensor_view_t<1> output_tv)
{
    uint64_t tid  = threadIdx.x;
    uint64_t oidx = blockIdx.x;

    // use double instead of FLOAT_ACCUM for better precision
    double sum_double = 0.0;
    for(uint64_t i = tid; i < outer_size * inner_size; i += blockDim.x)
        sum_double += static_cast<double>(
            input[i / inner_size * output_numel * inner_size + oidx * inner_size + i % inner_size]);

    FLOAT_ACCUM sum = static_cast<FLOAT_ACCUM>(sum_double);
    sum             = block_reduce<BinaryOp_t::Add, REDUCE_SIZE, ReduceThreadDim::X>(sum);

    if(tid == 0)
        output[output_tv.get_tensor_view_idx({oidx})] = CVT_ACCUM2FLOAT(sum);
}

extern "C" __global__ void Reduce1dSum(const FLOAT_ACCUM* __restrict__ input,
                                       FLOAT* __restrict__ output,
                                       uint64_t output_numel,
                                       uint64_t inner_size,
                                       uint64_t outer_size,
                                       tensor_view_t<1> output_tv)
{
    // instantiate the kernel
    Reduce1dSum<FLOAT>(input, output, output_numel, inner_size, outer_size, output_tv);
}
