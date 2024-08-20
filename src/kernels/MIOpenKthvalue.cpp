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
 * LIABILITY, WHETHER IN AN ACDTYPEN OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECDTYPEN WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <cstddef>
#include <cstdint>
// #ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
// #endif

#include "float_types.h"
#include "radix.hpp"
#include "tensor_view.hpp"

#ifndef IN_OUT_TYPE
#define IN_OUT_TYPE float
#endif

#ifndef CVT_ACCUM2FLOAT
#define CVT_ACCUM2FLOAT(x) (float_to_bfloat16(x))
#endif

#ifndef CVT_FLOAT2ACCUM
#define CVT_FLOAT2ACCUM(x) (bfloat16_to_float(x))
#endif

#ifndef LOCAL_SIZE
#define LOCAL_SIZE 256
#endif

#ifndef RADIX_BITS
#define RADIX_BITS 2
#endif

#ifndef RADIX_SIZE
#define RADIX_SIZE (1 << RADIX_BITS)
#endif

#ifndef RADIX_MASK
#define RADIX_MASK (RADIX_SIZE - 1)
#endif

template <typename DTYPE>
__device__ void kthvalueFwd(const DTYPE* input,
                            DTYPE* output,
                            size_t* indices,
                            size_t k,
                            size_t dim_size,
                            size_t dim_stride,
                            tensor_view_t<4> input_tv)
{
    /*
     * Example)
     * input : {A, B, C, D, E}
     * output/indices : {A, B, 1, D, E} or {A, B, D, E}
     * dim = 2 (C)
     * => gws = {LOCAL_SIZE, A * B * D * E}, lws = {LOCAL_SIZE, 1}
     */

    size_t lid = threadIdx.x;
    size_t gid = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ size_t lsum[LOCAL_SIZE][RADIX_SIZE];
    __shared__ DTYPE lval;
    __shared__ long lidx;
    size_t counts[RADIX_SIZE];
    RADIX_TYPE desired_mask = 0;
    RADIX_TYPE desired      = 0;

    tensor_layout_t<4> layout(input_tv, gid);
    auto idx = input_tv.get_tensor_view_idx(layout);

    for(size_t pos = sizeof(RADIX_TYPE) * 8 - RADIX_BITS; pos >= 0; pos -= RADIX_BITS)
    {
        for(size_t i = 0; i < RADIX_SIZE; ++i)
        {
            counts[i] = 0;
        }

        for(size_t i = lid; i < dim_size; i += LOCAL_SIZE)
        {
            size_t input_idx = idx + i * dim_stride;
            RADIX_TYPE val   = ENCODE(input[input_idx]);
            if((val & desired_mask) == desired)
            {
                ++counts[GetBitField(val, pos, RADIX_BITS)];
            }
        }

        for(size_t i = 0; i < RADIX_SIZE; ++i)
        {
            lsum[lid][i] = counts[i];
        }
        __syncthreads();
        // warp shuffle
        for(size_t i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
        {
            if(lid < i)
            {
                for(size_t j = 0; j < RADIX_SIZE; ++j)
                {
                    lsum[lid][j] += lsum[lid + i][j];
                }
            }
            __syncthreads();
        }
        // remove use share mem
        for(size_t i = 0; i < RADIX_SIZE; ++i)
        {
            counts[i] = lsum[0][i];
        }
        __syncthreads();

        bool found = false;
        // Process in ascending order
        for(size_t j = 0; j < RADIX_SIZE; ++j)
        {
            if(counts[j] >= k)
            {
                // Answer is inside this count
                if(counts[j] == 1 || pos == 0)
                {
                    // 1. counts[j] == 1
                    // We found an unique answer.
                    // 2. pos == 0
                    // There are multiple answers so we return any of them
                    for(size_t i = lid; i < dim_size; i += LOCAL_SIZE)
                    {
                        size_t input_idx = idx + i * dim_stride;
                        DTYPE val_ori    = input[input_idx];
                        RADIX_TYPE val   = ENCODE(val_ori);
                        if((val & desired_mask) == desired &&
                           GetBitField(val, pos, RADIX_BITS) == j)
                        {
                            // For case 2, this will be non-deterministic.
                            lval = val_ori;
                            lidx = i;
                        }
                    }
                    found = true;
                    break;
                }
                desired      = SetBitField(desired, j, pos, RADIX_BITS);
                desired_mask = SetBitField(desired_mask, RADIX_MASK, pos, RADIX_BITS);
                break;
            }
            k -= counts[j];
        }
        if(found)
            break;
    }

    __syncthreads();
    if(lid == 0)
    {
        output[gid]  = lval;
        indices[gid] = lidx;
    }
}

extern "C" __global__ void KthvalueUnreducedFwd(const IN_OUT_TYPE* input,
                                                IN_OUT_TYPE* output,
                                                size_t* indices,
                                                size_t k,
                                                size_t dim_size,
                                                size_t dim_stride,
                                                tensor_view_t<4> input_tv)
{
    kthvalueFwd<IN_OUT_TYPE>(input, output, indices, k, dim_size, dim_stride, input_tv);
}
