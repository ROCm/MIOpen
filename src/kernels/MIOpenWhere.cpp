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

template <typename TIO>
__device__ void WhereBroadcastedContiguousBackward_Kernel(const TIO* condition,
                                                          const TIO* output_grad,
                                                          TIO* input_grad,
                                                          TIO* other_grad,
                                                          size_t size,
                                                          size_t condition_size,
                                                          size_t input_size,
                                                          size_t other_size)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if(gid >= size)
        return;

    if(input_grad && gid < input_size)
    {
        FLOAT_ACCUM outgrad = CVT_FLOAT2ACCUM(output_grad[gid]);
        FLOAT_ACCUM cond    = CVT_FLOAT2ACCUM(condition[gid % condition_size]);
        input_grad[gid]     = CVT_ACCUM2FLOAT(outgrad * cond);
    }
    if(other_grad && gid < other_size)
    {
        FLOAT_ACCUM outgrad = CVT_FLOAT2ACCUM(output_grad[gid]);
        FLOAT_ACCUM cond    = CVT_FLOAT2ACCUM(condition[gid % condition_size]);
        other_grad[gid]     = CVT_ACCUM2FLOAT(outgrad * (1 - cond));
    }
}

extern "C" __global__ void WhereBroadcastedContiguousBackward(const IO_TYPE* condition,
                                                              const IO_TYPE* output_grad,
                                                              IO_TYPE* input_grad,
                                                              IO_TYPE* other_grad,
                                                              size_t size,
                                                              size_t condition_size,
                                                              size_t input_size,
                                                              size_t other_size)
{
    WhereBroadcastedContiguousBackward_Kernel<IO_TYPE>(condition,
                                                       output_grad,
                                                       input_grad,
                                                       other_grad,
                                                       size,
                                                       condition_size,
                                                       input_size,
                                                       other_size);
}

template <typename TIO>
__device__ void WhereConditionBroadcastedContiguousBackward_Kernel(const TIO* condition,
                                                                   const TIO* output_grad,
                                                                   TIO* input_grad,
                                                                   TIO* other_grad,
                                                                   size_t size,
                                                                   size_t condition_size,
                                                                   size_t input_size,
                                                                   size_t other_size)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= condition_size)
        return;

    FLOAT_ACCUM cond = CVT_FLOAT2ACCUM(condition[gid]);

    if(input_grad)
    {
        for(int idx = gid; idx < input_size; idx += condition_size)
        {
            FLOAT_ACCUM outgrad = CVT_FLOAT2ACCUM(output_grad[idx % size]);
            input_grad[idx]     = CVT_ACCUM2FLOAT(outgrad * cond);
        }
    }
    if(other_grad)
    {
        for(int idx = gid; idx < other_size; idx += condition_size)
        {
            FLOAT_ACCUM outgrad = CVT_FLOAT2ACCUM(output_grad[idx % size]);
            other_grad[idx]     = CVT_ACCUM2FLOAT(outgrad * (1 - cond));
        }
    }
}

extern "C" __global__ void WhereConditionBroadcastedContiguousBackward(const IO_TYPE* condition,
                                                                       const IO_TYPE* output_grad,
                                                                       IO_TYPE* input_grad,
                                                                       IO_TYPE* other_grad,
                                                                       size_t size,
                                                                       size_t condition_size,
                                                                       size_t input_size,
                                                                       size_t other_size)
{
    WhereConditionBroadcastedContiguousBackward_Kernel<IO_TYPE>(condition,
                                                                output_grad,
                                                                input_grad,
                                                                other_grad,
                                                                size,
                                                                condition_size,
                                                                input_size,
                                                                other_size);
}

template <typename TIO>
__device__ void WhereContiguousBackward_Kernel(
    const TIO* condition, const TIO* output_grad, TIO* input_grad, TIO* other_grad, size_t size)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if(gid >= size)
        return;

    if(input_grad)
    {
        FLOAT_ACCUM outgrad = CVT_FLOAT2ACCUM(output_grad[gid]);
        FLOAT_ACCUM cond    = CVT_FLOAT2ACCUM(condition[gid]);
        input_grad[gid]     = CVT_ACCUM2FLOAT(outgrad * cond);
    }
    if(other_grad)
    {
        FLOAT_ACCUM outgrad = CVT_FLOAT2ACCUM(output_grad[gid]);
        FLOAT_ACCUM cond    = CVT_FLOAT2ACCUM(condition[gid]);
        other_grad[gid]     = CVT_ACCUM2FLOAT(outgrad * (1 - cond));
    }
}

extern "C" __global__ void WhereContiguousBackward(const IO_TYPE* condition,
                                                   const IO_TYPE* output_grad,
                                                   IO_TYPE* input_grad,
                                                   IO_TYPE* other_grad,
                                                   size_t size)
{
    WhereContiguousBackward_Kernel<IO_TYPE>(condition, output_grad, input_grad, other_grad, size);
}
