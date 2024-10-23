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
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "tensor_view.hpp"

// template <typename TI, typename TO>
// __device__ void AnyForward(const TI* __restrict__ input,
//                            const TO* __restrict__ output,
//                            uint64_t N,
//                            uint64_t K,
//                            uint64_t st,
//                            uint64_t reduce_dim,
//                            tensor_view_t<5> input_tv,
//                            tensor_view_t<5> output_tv)
// {
//     uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;

//     if(gid >= N)
//         return;

//     size_t idx       = (gid / st) * st * K + gid % st;
//     size_t input_idx = input_tv.get_tensor_view_idx({idx});

//     TO any = 0;
//     for(size_t k = 0; k < K; ++k)
//     {
//         any = any || input[input_idx];
//         input_idx += input_tv.stride[reduce_dim];
//     }

//     output[output_tv.get_tensor_view_idx({gid})] = any;
// }

extern "C" __global__ void AnyForward(const INPUT_TYPE* __restrict__ input,
                                      OUTPUT_TYPE* __restrict__ output,
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

    size_t idx       = (gid / st) * st * K + gid % st;
    size_t input_idx = input_tv.get_tensor_view_idx({idx});

    OUTPUT_TYPE any = 0;
    for(size_t k = 0; k < K; ++k)
    {
        any = any || input[input_idx];
        input_idx += input_tv.stride[reduce_dim];
    }

    output[output_tv.get_tensor_view_idx({gid})] = any;
}

extern "C" __global__ void ReduceAny(const INPUT_TYPE* __restrict__ input,
                                     OUTPUT_TYPE* __restrict__ output,
                                     OUTPUT_TYPE* local_mem,
                                     uint64_t N,
                                     tensor_view_t<5> input_tv,
                                     tensor_view_t<5> output_tv)
{
    // printf("Kernel function\n");
    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t lid = threadIdx.x;

    auto idx = input_tv.get_tensor_view_idx({gid});
    // printf("idx: %d\n", idx);
    // printf
    // printf("N: %d, gid: %d, idx: %d\n", N, gid, idx);

    // printf("Hit here 0");

    // printf("test: %d\n", local_mem[0]);

    local_mem[lid] = (gid < N) ? input[input_tv.get_tensor_view_idx({gid})] : 0;

    __syncthreads();

    for(size_t i = blockDim.x / 2; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            local_mem[lid] = local_mem[lid] || local_mem[lid + i];
            // printf("Hit here 2");
        }
        __syncthreads();
    }

    // printf("lid: %d\n", lid);

    if(lid == 0)
    {
        printf("Notice");
        output[output_tv.get_tensor_view_idx({blockIdx.x})] = local_mem[0];
        // printf("Hit here 3");
        printf("gid: %d, idx: %d, blockIdx.x: %d, output: %d",
               gid,
               idx,
               blockIdx.x,
               output[output_tv.get_tensor_view_idx({blockIdx.x})]);
        // printf("output: %d\n", output[output_tv.get_tensor_view_idx({blockIdx.x})]);
    }

    // size_t input_idx = input_tv.get_tensor_view_idx({gid});

    // OUTPUT_TYPE any = 0;
    // for(size_t k = 0; k < input_tv.sizes[4]; ++k)
    // {
    //     any = any || input[input_idx];
    //     input_idx += input_tv.stride[4];
    // }

    // local_mem[lid] = any;
    // __syncthreads();

    // for(uint64_t s = blockDim.x / 2; s > 0; s >>= 1)
    // {
    //     if(lid < s)
    //     {
    //         local_mem[lid] = local_mem[lid] || local_mem[lid + s];
    //     }
    //     __syncthreads();
    // }

    // if(lid == 0)
    // {
    //     output[output_tv.get_tensor_view_idx({blockIdx.x})] = local_mem[0];
    // }
}
