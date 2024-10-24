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
#ifndef MIOPEN_USE_FP32
#define MIOPEN_USE_FP32 0
#endif

#ifndef MIOPEN_USE_FP16
#define MIOPEN_USE_FP16 0
#endif

#ifndef MIOPEN_USE_BFP16
#define MIOPEN_USE_BFP16 0
#endif

#ifndef MIOPEN_USE_INT8
#define MIOPEN_USE_INT8 0
#endif

#ifndef MIOPEN_USE_INT32
#define MIOPEN_USE_INT32 0
#endif

// #ifndef MIOPEN_USE_FP8
// #define MIOPEN_USE_FP8 0
// #endif

// #ifndef MIOPEN_USE_BFP8
// #define MIOPEN_USE_BFP8 0
// #endif

#if MIOPEN_USE_INT8
typedef char INPUT_TYPE;
#elif MIOPEN_USE_INT32
typedef int INPUT_TYPE;
#elif(MIOPEN_USE_FP16 || MIOPEN_USE_BFP16)
// As the half type degrades the performance, use short instead of half in
// transpose kernels, which have no match op. May change back to half when
// compile can deliver equal performance as short
typedef short INPUT_TYPE;
#elif MIOPEN_USE_FP32
typedef float INPUT_TYPE;
#endif

using OUTPUT_TYPE = unsigned char;

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_runtime.h>
#include <cstdio>
#endif

// #include "float_types.h"
#include "tensor_view.hpp"

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

extern "C" __global__ void ReduceAny(INPUT_TYPE* __restrict__ input,
                                     OUTPUT_TYPE* __restrict__ output,
                                     OUTPUT_TYPE* local_mem,
                                     uint64_t N,
                                     tensor_view_t<5> input_tv,
                                     tensor_view_t<5> output_tv)
{
    // printf("Running kernel\n");
    // uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t lid = threadIdx.x;

    // tensor_layout_t input_layout = tensor_layout_t(input_tv, gid);

    // tensor layout
    auto i_tl      = tensor_layout_t(input_tv, gid);
    auto input_idx = input_tv.get_tensor_view_idx(i_tl);

    local_mem[lid] = (gid < N) ? input[input_idx] : 0;

    // local_mem[lid] = (gid < N) ? input[tensor_layout] : 0;
    // local_mem[lid] = ()
    // local_mem[lid] = (gid < N) ? input[gid] : 0;

    __syncthreads();

    // printf("local_mem[0]: %u\n", (unsigned char)local_mem[0]);

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
        // auto o_tl = tensor_layout_t(output_tv, blockIdx.x);
        auto o_tl          = tensor_layout_t<5>(output_tv, blockIdx.x);
        auto output_idx    = output_tv.get_tensor_view_idx(o_tl);
        output[output_idx] = static_cast<OUTPUT_TYPE>(local_mem[0]);
        // printf("gid: %d, lid: %d, local_mem[0]: %d\n", gid, lid, local_mem[0]);
        // printf("Hit here;\n");

        // printf("Hit here\n");
        // printf("[Set output] blockDim.x: %d, blockIdx: %d, threadIdx: %d, gid: %d, lid: %d\n",
        //    blockDim.x,
        //    blockIdx.x,
        //    threadIdx.x,
        //    gid,
        //    lid);
        // printf("[Set output] blockDim.x: %d, blockIdx: %d, threadIdx: %d, gid: %d, lid: %d, "
        //        "local_mem[0]: %u\n",
        //        blockDim.x,
        //        blockIdx.x,
        //        threadIdx.x,
        //        gid,
        //        lid,
        //        (unsigned char)local_mem[0]);
        // OUTPUT_TYPE val                                     =
        // static_cast<OUTPUT_TYPE>(local_mem[0]); output[blockIdx.x] = val;
        // output[output_tv.get_tensor_view_idx({blockIdx.x})] = val;
        // printf("local_mem[0]: %d", local_mem[0]);
    }
}
