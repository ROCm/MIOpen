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
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "tensor_view.hpp"
#include "radix.hpp"

#ifndef IN_OUT_TYPE
#define IN_OUT_TYPE float
#endif

#ifndef LOCAL_SIZE
#define LOCAL_SIZE 256
#endif

template <typename DTYPE>
__device__ void kthvalueFwd(const DTYPE* input,
                            DTYPE* output,
                            size_t* indices,
                            size_t k,
                            size_t dim_size,
                            size_t dim_stride,
                            size_t output_size,
                            tensor_view_t<4> input_tv,
                            tensor_view_t<5> output_tv,
                            tensor_view_t<5> indices_tv)
{
    /*
     * Input : {N, C, D, H, W}. Select dim: 2(D)
     * Output/indices : {N, C, H, W} or {N, C, 1, H, W} (if keepDim param in miopen.h = True)
     * Each lws handle dim_size elements to find the kth value.
     * Lws = {256 or 512, 1, 1}
     * Gws = {A * B * D * E * lws.x, 1, 1},
     */
    using RADIX_TYPE = typename RadixType<DTYPE>::type;

    const int RADIX_BITS = 2;
    const int RADIX_SIZE = 1 << RADIX_BITS;
    const int RADIX_MASK = RADIX_SIZE - 1;

    size_t lid = threadIdx.x;
    size_t gid = blockIdx.x;
    if(gid >= output_size)
    {
        return;
    }

    __shared__ size_t lsum[LOCAL_SIZE][RADIX_SIZE];
    __shared__ DTYPE lval;
    __shared__ size_t lidx;
    size_t counts[RADIX_SIZE];
    RADIX_TYPE desired_mask = 0;
    RADIX_TYPE desired      = 0;

    tensor_layout_t<4> layout(input_tv, gid);
    auto idx = input_tv.get_tensor_view_idx(layout);

    for(int pos = sizeof(RADIX_TYPE) * 8 - RADIX_BITS; pos >= 0; pos -= RADIX_BITS)
    {
        for(size_t& count : counts)
        {
            count = 0;
        }

        for(size_t i = lid; i < dim_size; i += LOCAL_SIZE)
        {
            size_t input_idx = idx + i * dim_stride;
            RADIX_TYPE val   = encode<DTYPE>(input[input_idx]);
            if((val & desired_mask) == desired)
            {
                ++counts[GetBitFieldImpl<RADIX_BITS>(val, pos)];
            }
        }

        for(size_t i = 0; i < RADIX_SIZE; ++i)
        {
            lsum[lid][i] = counts[i];
        }
        __syncthreads();
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
        for(size_t i = 0; i < RADIX_SIZE; ++i)
        {
            counts[i] = lsum[0][i];
        }
        __syncthreads();

        bool found = false;
        // Process in ascending order
        for(size_t j = 0; j < RADIX_SIZE; ++j)
        {
            if(counts[j] < k)
            {
                k -= counts[j];
                continue;
            }
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
                    RADIX_TYPE val   = encode<DTYPE>(val_ori);
                    if((val & desired_mask) == desired &&
                       GetBitFieldImpl<RADIX_BITS>(val, pos) == j)
                    {
                        // For case 2, this will be non-deterministic.
                        lval = val_ori;
                        lidx = i;
                    }
                }
                found = true;
                break;
            }
            desired      = SetBitFieldImpl<RADIX_TYPE>(desired, j, pos);
            desired_mask = SetBitFieldImpl<RADIX_TYPE>(desired_mask, RADIX_MASK, pos);
            break;
        }
        if(found)
            break;
    }

    __syncthreads();
    if(lid == 0)
    {
        auto output_layout                                   = tensor_layout_t<5>(output_tv, gid);
        auto indices_layout                                  = tensor_layout_t<5>(indices_tv, gid);
        output[output_tv.get_tensor_view_idx(output_layout)] = lval;
        indices[indices_tv.get_tensor_view_idx(indices_layout)] = lidx;
    }
}

extern "C" __global__ void KthvalueFwd(const IN_OUT_TYPE* input,
                                       IN_OUT_TYPE* output,
                                       size_t* indices,
                                       size_t k,
                                       size_t dim_size,
                                       size_t dim_stride,
                                       size_t output_size,
                                       tensor_view_t<4> input_tv,
                                       tensor_view_t<5> output_tv,
                                       tensor_view_t<5> indices_tv)
{
    kthvalueFwd<IN_OUT_TYPE>(input,
                             output,
                             indices,
                             k,
                             dim_size,
                             dim_stride,
                             output_size,
                             input_tv,
                             output_tv,
                             indices_tv);
}
