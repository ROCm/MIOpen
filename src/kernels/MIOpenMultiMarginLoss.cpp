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

template <typename DTYPE, int REDUCTION_T>
__device__ void multimarginlossforward2d(const DTYPE* __restrict__ I,
                                         const uint64_t* __restrict__ T,
                                         const DTYPE* __restrict__ W,
                                         void* __restrict__ O,
                                         const long p,
                                         const float margin,
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
    size_t y         = T[T_tv.get_tensor_view_idx({n})];
    if(y >= C)
    {
        // TODO: need to handle invalid target index value
        return;
    }

    FLOAT_ACCUM Iny = CVT_FLOAT2ACCUM(I[I_tv.get_tensor_view_idx({n, y})]);
    FLOAT_ACCUM Wy  = CVT_FLOAT2ACCUM(W[W_tv.get_tensor_view_idx({y})]);

    for(size_t c = 0; c < C; c++)
    {
        if(y == c)
            continue;
        FLOAT_ACCUM t = margin - Iny + CVT_FLOAT2ACCUM(I[I_tv.get_tensor_view_idx({n, c})]);
        if(t < 0)
            continue;
        if(p == 2)
            t = t * t;
        t = Wy * t;
        loss += t;
    }
    loss /= C;
    switch(REDUCTION_T)
    {
    case 0: static_cast<DTYPE*>(O)[O_tv.get_tensor_view_idx({n})] = CVT_ACCUM2FLOAT(loss); break;
    case 1: static_cast<FLOAT_ACCUM*>(O)[n] = loss; break;
    case 2: static_cast<FLOAT_ACCUM*>(O)[n] = loss / N; break;
    default: break;
    }
}

extern "C" __global__ void MultiMarginLossForward2d(const FLOAT* __restrict__ I,
                                                    const uint64_t* __restrict__ T,
                                                    const FLOAT* __restrict__ W,
                                                    void* __restrict__ O,
                                                    const long p,
                                                    const float margin,
                                                    tensor_view_t<2> I_tv,
                                                    tensor_view_t<1> T_tv,
                                                    tensor_view_t<1> W_tv,
                                                    tensor_view_t<1> O_tv)
{
    // instantiate the kernel
    multimarginlossforward2d<FLOAT, REDUCTION_TYPE>(I, T, W, O, p, margin, I_tv, T_tv, W_tv, O_tv);
}
