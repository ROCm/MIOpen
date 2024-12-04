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

#ifndef T_TYPE
#define T_TYPE int32_t
#endif

template <typename T, typename Ta>
__device__ void sparseSoftmaxCrossEntropyWithLogitsForward(const T* input,
                                                           const Ta* target,
                                                           T* output,
                                                           T* backprop,
                                                           uint64_t num_class,
                                                           tensor_view_t<2> input_tv,
                                                           tensor_view_t<1> target_tv,
                                                           tensor_view_t<1> output_tv,
                                                           tensor_view_t<2> backprop_tv)
{
    uint64_t gid = blockIdx.x;
    uint64_t lid = threadIdx.x;

    __shared__ uint64_t label;
    FLOAT_ACCUM lmax = log(0.0f), lsum = 0.0f;

    if(lid == 0)
        label = static_cast<uint64_t>(target[target_tv.get_tensor_view_idx({gid})]);

    for(uint64_t i = lid; i < num_class; i += LOCAL_SIZE)
    {
        FLOAT_ACCUM val = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({gid, i})]);
        lmax            = max(lmax, val);
    }
    lmax = block_reduce<BinaryOp_t::Max, LOCAL_SIZE, ReduceThreadDim::X>(lmax);

    for(uint64_t i = lid; i < num_class; i += LOCAL_SIZE)
    {
        FLOAT_ACCUM val =
            exp(CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({gid, i})]) - lmax);
        lsum += val;
    }
    lsum = block_reduce<BinaryOp_t::Add, LOCAL_SIZE, ReduceThreadDim::X>(lsum);

    if(lid == 0)
    {
        FLOAT_ACCUM val = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({gid, label})]);
        output[gid]     = CVT_ACCUM2FLOAT(log(lsum) - val + lmax);
    }

    for(uint64_t i = lid; i < num_class; i += LOCAL_SIZE)
    {
        FLOAT_ACCUM val = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({gid, i})]);
        FLOAT_ACCUM backprop_val =
            (i == label) ? exp(val - lmax) / lsum - 1.0f : exp(val - lmax) / lsum;

        backprop[backprop_tv.get_tensor_view_idx({gid, i})] = CVT_ACCUM2FLOAT(backprop_val);
    }
}

extern "C" __global__ void SparseSoftmaxCrossEntropyWithLogitsForward(const D_TYPE* input,
                                                                      const T_TYPE* target,
                                                                      D_TYPE* output,
                                                                      D_TYPE* backprop,
                                                                      uint64_t num_class,
                                                                      tensor_view_t<2> input_tv,
                                                                      tensor_view_t<1> target_tv,
                                                                      tensor_view_t<1> output_tv,
                                                                      tensor_view_t<2> backprop_tv)
{
    sparseSoftmaxCrossEntropyWithLogitsForward<D_TYPE, T_TYPE>(
        input, target, output, backprop, num_class, input_tv, target_tv, output_tv, backprop_tv);
}

template <typename T, typename Ta>
__device__ void sparseSoftmaxCrossEntropyWithLogitsForwardContiguous(
    const T* input, const Ta* target, T* output, T* backprop, uint64_t num_class)
{
    uint64_t gid          = blockIdx.x;
    uint64_t lid          = threadIdx.x;
    uint64_t batch_offset = gid * num_class;

    __shared__ uint64_t label;

    FLOAT_ACCUM lmax = log(0.0f), lsum = 0.0f;

    if(lid == 0)
        label = static_cast<uint64_t>(target[gid]);

    for(uint64_t i = lid; i < num_class; i += LOCAL_SIZE)
    {
        FLOAT_ACCUM val = CVT_FLOAT2ACCUM(input[i + batch_offset]);
        lmax            = max(lmax, val);
    }
    lmax = block_reduce<BinaryOp_t::Max, LOCAL_SIZE, ReduceThreadDim::X>(lmax);

    for(uint64_t i = lid; i < num_class; i += LOCAL_SIZE)
    {
        FLOAT_ACCUM val = exp(CVT_FLOAT2ACCUM(input[i + batch_offset]) - lmax);
        lsum += val;
    }
    lsum = block_reduce<BinaryOp_t::Add, LOCAL_SIZE, ReduceThreadDim::X>(lsum);

    if(lid == 0)
    {
        FLOAT_ACCUM val = CVT_FLOAT2ACCUM(input[label + batch_offset]);
        output[gid]     = CVT_ACCUM2FLOAT(log(lsum) - val + lmax);
    }

    for(uint64_t i = lid; i < num_class; i += LOCAL_SIZE)
    {
        FLOAT_ACCUM val = CVT_FLOAT2ACCUM(input[i + batch_offset]);
        FLOAT_ACCUM backprop_val =
            (i == label) ? exp(val - lmax) / lsum - 1.0f : exp(val - lmax) / lsum;

        backprop[i + batch_offset] = CVT_ACCUM2FLOAT(backprop_val);
    }
}

extern "C" __global__ void SparseSoftmaxCrossEntropyWithLogitsForwardContiguous(
    const D_TYPE* input, const T_TYPE* target, D_TYPE* output, D_TYPE* backprop, uint64_t num_class)
{
    sparseSoftmaxCrossEntropyWithLogitsForwardContiguous<D_TYPE, T_TYPE>(
        input, target, output, backprop, num_class);
}

template <typename T>
__device__ void sparseSoftmaxCrossEntropyWithLogitsBackward(const T* output_grad,
                                                            const T* backprop,
                                                            T* input_grad,
                                                            uint64_t num_class,
                                                            tensor_view_t<1> output_grad_tv,
                                                            tensor_view_t<2> backprop_tv,
                                                            tensor_view_t<2> input_grad_tv)
{
    uint64_t gid = blockIdx.x;
    uint64_t lid = threadIdx.x;

    FLOAT_ACCUM output_grad_val =
        CVT_FLOAT2ACCUM(output_grad[output_grad_tv.get_tensor_view_idx({gid})]);

    for(uint64_t i = lid; i < num_class; i += LOCAL_SIZE)
    {
        FLOAT_ACCUM backprop_val =
            CVT_FLOAT2ACCUM(backprop[backprop_tv.get_tensor_view_idx({gid, i})]);
        input_grad[input_grad_tv.get_tensor_view_idx({gid, i})] =
            CVT_ACCUM2FLOAT(output_grad_val * backprop_val);
    }
}

extern "C" __global__ void
SparseSoftmaxCrossEntropyWithLogitsBackward(const D_TYPE* output_grad,
                                            const D_TYPE* backprop,
                                            D_TYPE* input_grad,
                                            uint64_t num_class,
                                            tensor_view_t<1> output_grad_tv,
                                            tensor_view_t<2> backprop_tv,
                                            tensor_view_t<2> input_grad_tv)
{
    sparseSoftmaxCrossEntropyWithLogitsBackward<D_TYPE>(
        output_grad, backprop, input_grad, num_class, output_grad_tv, backprop_tv, input_grad_tv);
}

template <typename T>
__device__ void sparseSoftmaxCrossEntropyWithLogitsBackwardContiguous(const T* output_grad,
                                                                      const T* backprop,
                                                                      T* input_grad,
                                                                      uint64_t num_class)
{
    uint64_t gid = blockIdx.x;
    uint64_t lid = threadIdx.x;

    uint64_t batch_offset = gid * num_class;

    FLOAT_ACCUM output_grad_val = CVT_FLOAT2ACCUM(output_grad[gid]);

    for(uint64_t i = lid; i < num_class; i += LOCAL_SIZE)
    {
        FLOAT_ACCUM backprop_val     = CVT_FLOAT2ACCUM(backprop[i + batch_offset]);
        input_grad[i + batch_offset] = CVT_ACCUM2FLOAT(output_grad_val * backprop_val);
    }
}

extern "C" __global__ void SparseSoftmaxCrossEntropyWithLogitsBackwardContiguous(
    const D_TYPE* output_grad, const D_TYPE* backprop, D_TYPE* input_grad, uint64_t num_class)
{
    sparseSoftmaxCrossEntropyWithLogitsBackwardContiguous<D_TYPE>(
        output_grad, backprop, input_grad, num_class);
}
