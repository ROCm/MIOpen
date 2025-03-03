/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
 * IMPLIED, INCLUDING BUT NOT LIMITED T THE WARRANTIES OF MERCHANTABILITY,
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

template <typename T>
__device__ void softmaxcrossentropywithlogitsForwardContiguous(const T* __restrict__ input,
                                                               const T* __restrict__ target,
                                                               T* __restrict__ output,
                                                               T* __restrict__ backprop,
                                                               size_t num_class)
{
    uint64_t gid = blockIdx.x;
    uint64_t lid = threadIdx.x;

    __shared__ FLOAT_ACCUM lmax[LOCAL_SIZE], lsum[LOCAL_SIZE], lloss[LOCAL_SIZE];
    lmax[lid]           = -INFINITY;
    lsum[lid]           = 0.0f;
    lloss[lid]          = 0.0f;
    size_t batch_offset = gid * num_class;

    for(size_t i = lid; i < num_class; i += LOCAL_SIZE)
    {
        FLOAT_ACCUM val = CVT_FLOAT2ACCUM(input[i + batch_offset]);
        lmax[lid]       = max(lmax[lid], val);
    }
    __syncthreads();

    for(int i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            lmax[lid] = max(lmax[lid], lmax[lid + i]);
        }
        __syncthreads();
    }

    for(size_t i = lid; i < num_class; i += LOCAL_SIZE)
    {
        FLOAT_ACCUM val = CVT_FLOAT2ACCUM(input[i + batch_offset]);
        lsum[lid] += exp(val - lmax[0]);
    }
    __syncthreads();

    for(int i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            lsum[lid] += lsum[lid + i];
        }
        __syncthreads();
    }

    FLOAT_ACCUM log_val = log(lsum[0]);
    for(size_t i = lid; i < num_class; i += LOCAL_SIZE)
    {
        FLOAT_ACCUM val   = CVT_FLOAT2ACCUM(input[i + batch_offset]);
        FLOAT_ACCUM label = CVT_FLOAT2ACCUM(target[i + batch_offset]);
        lloss[lid] += label * (log_val - val + lmax[0]);
    }
    __syncthreads();

    for(int i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            lloss[lid] += lloss[lid + i];
        }
        __syncthreads();
    }

    if(lid == 0)
    {
        output[gid] = CVT_ACCUM2FLOAT(lloss[0]);
    }

    for(size_t i = lid; i < num_class; i += LOCAL_SIZE)
    {
        FLOAT_ACCUM val            = CVT_FLOAT2ACCUM(input[i + batch_offset]);
        FLOAT_ACCUM label          = CVT_FLOAT2ACCUM(target[i + batch_offset]);
        FLOAT_ACCUM backprop_val   = exp(val - lmax[0]) / lsum[0] - label;
        backprop[i + batch_offset] = CVT_ACCUM2FLOAT(backprop_val);
    }
}

extern "C" __global__ void
SoftmaxCrossEntropyWithLogitsForwardContiguous(const D_TYPE* __restrict__ input,
                                               const D_TYPE* __restrict__ target,
                                               D_TYPE* __restrict__ output,
                                               D_TYPE* __restrict__ backprop,
                                               size_t num_class)
{
    softmaxcrossentropywithlogitsForwardContiguous<D_TYPE>(
        input, target, output, backprop, num_class);
}

template <typename T>
__device__ void softmaxcrossentropywithlogitsBackwardContiguous(const T* __restrict__ output_grad,
                                                                const T* __restrict__ backprop,
                                                                const T* __restrict__ input,
                                                                T* __restrict__ input_grad,
                                                                T* __restrict__ target_grad,
                                                                size_t num_class)
{
    uint64_t gid = blockIdx.x;
    uint64_t lid = threadIdx.x;

    size_t batch_offset = gid * num_class;

    __shared__ FLOAT_ACCUM lmax[LOCAL_SIZE], lsum[LOCAL_SIZE];
    lmax[lid] = -INFINITY;
    lsum[lid] = 0.0f;

    __shared__ FLOAT_ACCUM output_grad_val;
    if(lid == 0)
    {
        output_grad_val = CVT_FLOAT2ACCUM(output_grad[gid]);
    }
    __syncthreads();

    if(input_grad)
    {
        for(size_t i = lid; i < num_class; i += LOCAL_SIZE)
        {
            FLOAT_ACCUM backprop_val     = CVT_FLOAT2ACCUM(backprop[i + batch_offset]);
            input_grad[i + batch_offset] = CVT_ACCUM2FLOAT(output_grad_val * backprop_val);
        }
    }

    if(target_grad)
    {
        for(size_t i = lid; i < num_class; i += LOCAL_SIZE)
        {
            FLOAT_ACCUM val = CVT_FLOAT2ACCUM(input[i + batch_offset]);
            lmax[lid]       = max(lmax[lid], val);
        }
        __syncthreads();

        for(int i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
        {
            if(lid < i)
            {
                lmax[lid] = max(lmax[lid], lmax[lid + i]);
            }
            __syncthreads();
        }

        for(size_t i = lid; i < num_class; i += LOCAL_SIZE)
        {
            FLOAT_ACCUM val = CVT_FLOAT2ACCUM(input[i + batch_offset]);
            lsum[lid] += exp(val - lmax[0]);
        }
        __syncthreads();

        for(int i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
        {
            if(lid < i)
            {
                lsum[lid] += lsum[lid + i];
            }
            __syncthreads();
        }

        FLOAT_ACCUM log_val = log(lsum[0]);
        for(size_t i = lid; i < num_class; i += LOCAL_SIZE)
        {
            FLOAT_ACCUM logit_val = CVT_FLOAT2ACCUM(input[i + batch_offset]);
            target_grad[i + batch_offset] =
                CVT_ACCUM2FLOAT((lmax[0] + log_val - logit_val) * output_grad_val);
        }
    }
}

extern "C" __global__ void
SoftmaxCrossEntropyWithLogitsBackwardContiguous(const D_TYPE* __restrict__ output_grad,
                                                const D_TYPE* __restrict__ backprop,
                                                const D_TYPE* __restrict__ input,
                                                D_TYPE* __restrict__ input_grad,
                                                D_TYPE* __restrict__ target_grad,
                                                size_t num_class)
{
    softmaxcrossentropywithlogitsBackwardContiguous<D_TYPE>(
        output_grad, backprop, input, input_grad, target_grad, num_class);
}
