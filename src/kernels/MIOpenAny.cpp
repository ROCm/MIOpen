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
#include <hip/hip_runtime.h>
#include <cstdio>
#endif

#include <float_types.h>
#include <tensor_view.hpp>

#ifndef MIOPEN_USE_INT8
#define MIOPEN_USE_INT8 0
#endif

#if MIOPEN_USE_INT8
using INPUT_TYPE = signed char;
#endif

template <typename INPUT_TYPE>
__device__ void any_forward(const INPUT_TYPE* __restrict__ input,
                            unsigned char* __restrict__ output,
                            uint64_t N,
                            uint64_t K,
                            uint64_t st,
                            uint64_t reduce_dim,
                            tensor_view_t<5> input_tv,
                            tensor_view_t<5> output_tv)
{
    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if(gid >= N)
        return;

    size_t idx = (gid / st) * st * K + gid % st;

    auto i_tl      = tensor_layout_t<5>(input_tv, idx);
    auto input_idx = input_tv.get_tensor_view_idx(i_tl);

    unsigned char any = 0;
    for(size_t k = 0; k < K; ++k)
    {
#if MIOPEN_USE_FP32 || MIOPEN_USE_FP16 || MIOPEN_USE_BFP16
        FLOAT_ACCUM val = CVT_FLOAT2ACCUM(input[input_idx]);
#else
        auto val = input[input_idx];
#endif
        any = (any || val) != 0;
        input_idx += input_tv.stride[reduce_dim];
    }

    auto o_tl       = tensor_layout_t<5>(output_tv, gid);
    auto output_idx = output_tv.get_tensor_view_idx(o_tl);

    output[output_idx] = any;
}

template <typename INPUT_TYPE>
__device__ void reduce_any(INPUT_TYPE* __restrict__ input,
                           unsigned char* __restrict__ output,
                           unsigned char* local_mem,
                           uint64_t N,
                           tensor_view_t<5> input_tv,
                           tensor_view_t<5> output_tv)
{
    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t lid = threadIdx.x;

    auto i_tl      = tensor_layout_t(input_tv, gid);
    auto input_idx = input_tv.get_tensor_view_idx(i_tl);

    if(gid < N)
    {
#if MIOPEN_USE_FP32 || MIOPEN_USE_FP16 || MIOPEN_USE_BFP16
        FLOAT_ACCUM val = CVT_FLOAT2ACCUM(input[input_idx]);
#else
        auto val = input[input_idx];
#endif

        local_mem[lid] = (val != 0);
    }
    else
    {
        local_mem[lid] = 0;
    }

    __syncthreads();

    for(size_t i = blockDim.x / 2; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            local_mem[lid] = local_mem[lid] || local_mem[lid + i];
        }
        __syncthreads();
    }

    if(lid == 0)
    {
        auto o_tl          = tensor_layout_t<5>(output_tv, blockIdx.x);
        auto output_idx    = output_tv.get_tensor_view_idx(o_tl);
        output[output_idx] = local_mem[0];
    }
}

extern "C" __global__ void AnyForward(const INPUT_TYPE* __restrict__ input,
                                      unsigned char* __restrict__ output,
                                      uint64_t N,
                                      uint64_t K,
                                      uint64_t st,
                                      uint64_t reduce_dim,
                                      tensor_view_t<5> input_tv,
                                      tensor_view_t<5> output_tv)
{
    any_forward<INPUT_TYPE>(input, output, N, K, st, reduce_dim, input_tv, output_tv);
}

extern "C" __global__ void ReduceAny(INPUT_TYPE* __restrict__ input,
                                     unsigned char* __restrict__ output,
                                     unsigned char* local_mem,
                                     uint64_t N,
                                     tensor_view_t<5> input_tv,
                                     tensor_view_t<5> output_tv)
{
    reduce_any<INPUT_TYPE>(input, output, local_mem, N, input_tv, output_tv);
}