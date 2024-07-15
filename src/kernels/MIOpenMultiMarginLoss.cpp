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

template <typename DTYPE>
__device__ void multimarginlossunreducedforward2d(const DTYPE* __restrict__ I,
                                                  const uint64_t* __restrict__ T,
                                                  const DTYPE* __restrict__ W,
                                                  DTYPE* __restrict__ O,
                                                  long p,
                                                  float margin,
                                                  tensor_view_t<2> I_tv,
                                                  tensor_view_t<1> T_tv,
                                                  tensor_view_t<1> W_tv,
                                                  tensor_view_t<1> O_tv)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t N = I_tv.size[0], C = I_tv.size[1];
    size_t n = gid;
    if(n >= N)
        return;

    FLOAT_ACCUM loss = 0;
    uint64_t y       = T[T_tv.get_tensor_view_idx({n})];
    if(y >= C)
    {
        // TODO: need to handle invalid target index value
        return;
    }

    for(size_t c = 0; c < C; c++)
    {
        if(y == c)
            continue;
        FLOAT_ACCUM t = margin - CVT_FLOAT2ACCUM(I[I_tv.get_tensor_view_idx({n, y})]) +
                        CVT_FLOAT2ACCUM(I[I_tv.get_tensor_view_idx({n, c})]);
        if(t < 0)
            continue;
        if(p == 2)
            t = t * t;
        t = CVT_FLOAT2ACCUM(W[W_tv.get_tensor_view_idx({y})]) * t;
        loss += t / (float)C;
    }
    O[O_tv.get_tensor_view_idx({n})] = CVT_ACCUM2FLOAT(loss);
}

extern "C" __global__ void MultiMarginLossUnreducedForward2d(const FLOAT* __restrict__ I,
                                                             const uint64_t* __restrict__ T,
                                                             const FLOAT* __restrict__ W,
                                                             FLOAT* __restrict__ O,
                                                             long p,
                                                             float margin,
                                                             tensor_view_t<2> I_tv,
                                                             tensor_view_t<1> T_tv,
                                                             tensor_view_t<1> W_tv,
                                                             tensor_view_t<1> O_tv)
{
    // instantiate the kernel
    multimarginlossunreducedforward2d<FLOAT>(I, T, W, O, p, margin, I_tv, T_tv, W_tv, O_tv);
}

template <typename DTYPE>
__device__ void multimarginlossforward2d(const DTYPE* __restrict__ I,
                                         const uint64_t* __restrict__ T,
                                         const DTYPE* __restrict__ W,
                                         DTYPE* __restrict__ lsum,
                                         long p,
                                         float margin,
                                         const float divisor,
                                         tensor_view_t<2> I_tv,
                                         tensor_view_t<1> T_tv,
                                         tensor_view_t<1> W_tv)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t N = I_tv.size[0], C = I_tv.size[1];
    size_t n = gid;
    if(n >= N)
        return;

    FLOAT_ACCUM loss = 0;
    long y           = T[T_tv.get_tensor_view_idx({n})];
    if(y >= C)
    {
        // TODO: need to handle invalid target index value
        return;
    }

    for(size_t c = 0; c < C; c++)
    {
        if(y == c)
            continue;
        FLOAT_ACCUM t = margin - CVT_FLOAT2ACCUM(I[I_tv.get_tensor_view_idx({n, y})]) +
                        CVT_FLOAT2ACCUM(I[I_tv.get_tensor_view_idx({n, c})]);
        if(t < 0)
            continue;
        if(p == 2)
            t = t * t;
        t = CVT_FLOAT2ACCUM(W[W_tv.get_tensor_view_idx({y})]) * t;
        loss += t / (float)C;
    }

    lsum[n] = CVT_ACCUM2FLOAT(loss / divisor);
}

extern "C" __global__ void MultiMarginLossForward2d(const FLOAT* __restrict__ I,
                                                    const uint64_t* __restrict__ T,
                                                    const FLOAT* __restrict__ W,
                                                    FLOAT* __restrict__ lsum,
                                                    long p,
                                                    float margin,
                                                    const float divisor,
                                                    tensor_view_t<2> I_tv,
                                                    tensor_view_t<1> T_tv,
                                                    tensor_view_t<1> W_tv)
{
    // instantiate the kernel
    multimarginlossforward2d<FLOAT>(I, T, W, lsum, p, margin, divisor, I_tv, T_tv, W_tv);
}
