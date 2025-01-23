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
#endif // MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS

#if MIOPEN_USE_FP32 == 1
#define HIP_FLOAT float
#define __float2T(x) (x)
#define custom_pow(x, y) (powf(x, y))
#elif MIOPEN_USE_FP16 == 1
#include <hip/hip_fp16.h>
#define HIP_FLOAT __half
#define __float2T(x) (__float2half(x))
#define custom_pow(x, y) (hexp(hlog(x) * y))
#endif

#include "float_types.h"
#include "tensor_view.hpp"
#include "block_reduce.hpp"

extern "C" __global__ void NormalizeReduceContiguous(const FLOAT* __restrict__ input,
                                                     const FLOAT* __restrict__ output_grad,
                                                     FLOAT* __restrict__ output,
                                                     const uint64_t inner_size)
{
    uint64_t lid   = threadIdx.y;
    uint64_t gid_0 = blockIdx.x * blockDim.x + threadIdx.x;

    FLOAT_ACCUM sum = 0.0;
    for(uint64_t inner_idx = lid; inner_idx < inner_size; inner_idx += LOCAL_SIZE)
    {
        uint64_t input_idx = gid_0 * inner_size + inner_idx;
        sum += CVT_FLOAT2ACCUM(input[input_idx]) * CVT_FLOAT2ACCUM(output_grad[input_idx]);
    }
    __syncthreads();
    sum = block_reduce<BinaryOp_t::Add, LOCAL_SIZE, ReduceThreadDim::Y>(sum);

    if(lid == 0)
        output[gid_0] = CVT_ACCUM2FLOAT(sum);
}

template <typename T>
__device__ void normalizebackwardopt(const T* __restrict__ input,
                                     const T* __restrict__ divisor,
                                     const T* __restrict__ output_grad,
                                     T* __restrict__ input_grad,
                                     const T* __restrict__ reduce,
                                     const T p,
                                     const T eps,
                                     const uint64_t num_elem,
                                     const uint32_t dim,
                                     tensor_view_t<5> input_tv,
                                     tensor_view_t<5> divisor_tv)
{
    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= num_elem)
        return;

    T x  = input[gid];
    T dy = output_grad[gid];

    // In this case, all elements that used similar 'divisor' and 'reduce' are in the same
    // block. Therefore, we can use shared memory to read it faster
    static __shared__ T div, red;
    static __shared__ tensor_layout_t<5> divisor_layout;
    if(threadIdx.x == 0)
    {
        divisor_layout             = tensor_layout_t<5>(input_tv, gid);
        divisor_layout.layout[dim] = 0;
        // divisor_ix = reduce_idx
        uint64_t divisor_idx = divisor_tv.get_tensor_view_idx(divisor_layout);
        div                  = divisor[divisor_idx];
        red                  = reduce[divisor_idx];
    }
    __syncthreads();

    if(eps == div)
    {
        input_grad[gid] = dy / eps;
    }
    else
    {
        T abs_coef = (x < __float2T(0.0f) ? __float2T(-1.0f) : __float2T(1.0f));
        T tmp      = __float2T(-1.0f) / div / custom_pow(div, p) *
                custom_pow(abs_coef * x, (p - __float2T(1.0f))) * abs_coef;
        input_grad[gid] = red * tmp + dy / div;
    }
}

extern "C" __global__ void NormalizeBackwardOpt(const HIP_FLOAT* __restrict__ input,
                                                const HIP_FLOAT* __restrict__ divisor,
                                                const HIP_FLOAT* __restrict__ output_grad,
                                                HIP_FLOAT* __restrict__ input_grad,
                                                const HIP_FLOAT* __restrict__ reduce,
                                                const float p,
                                                const float eps,
                                                const uint64_t num_elem,
                                                const uint32_t dim,
                                                tensor_view_t<5> input_tv,
                                                tensor_view_t<5> divisor_tv)
{
    normalizebackwardopt<HIP_FLOAT>(input,
                                    divisor,
                                    output_grad,
                                    input_grad,
                                    reduce,
                                    __float2T(p),
                                    __float2T(eps),
                                    num_elem,
                                    dim,
                                    input_tv,
                                    divisor_tv);
}
