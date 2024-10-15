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
__device__ void softmarginlossforward5d(const DTYPE* __restrict__ I,
                                        const DTYPE* __restrict__ T,
                                        void* __restrict__ O,
                                        const size_t num_elem,
                                        tensor_view_t<5> I_tv,
                                        tensor_view_t<5> T_tv,
                                        tensor_view_t<5> O_tv)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    tensor_layout_t<5> idx(I_tv, gid);
    if(idx.layout[0] >= I_tv.size[0])
        return;

    FLOAT_ACCUM i    = CVT_FLOAT2ACCUM(I[I_tv.get_tensor_view_idx(idx)]);
    FLOAT_ACCUM t    = CVT_FLOAT2ACCUM(T[T_tv.get_tensor_view_idx(idx)]);
    FLOAT_ACCUM loss = log1p(exp(-i * t));
    switch(REDUCTION_T)
    {
    // If reduction = None, O is DTYPE*
    case 0: static_cast<DTYPE*>(O)[O_tv.get_tensor_view_idx(idx)] = CVT_ACCUM2FLOAT(loss); break;
    // If reduction = Sum, O is FLOAT_ACCUM* and then all elements will be sum up in the next
    // kernel
    case 1: static_cast<FLOAT_ACCUM*>(O)[gid] = loss; break;
    // If reduction = Mean, same as Sum but O will be divided by num_elem, then the next kernel sum
    // up will return mean of all elements
    case 2: static_cast<FLOAT_ACCUM*>(O)[gid] = loss / num_elem; break;
    default: break;
    }
}

extern "C" __global__ void SoftMarginLossForward5d(const FLOAT* __restrict__ I,
                                                   const FLOAT* __restrict__ T,
                                                   void* __restrict__ O,
                                                   const size_t num_elem,
                                                   tensor_view_t<5> I_tv,
                                                   tensor_view_t<5> T_tv,
                                                   tensor_view_t<5> O_tv)
{
    // instantiate the kernel
    softmarginlossforward5d<FLOAT, REDUCTION_TYPE>(I, T, O, num_elem, I_tv, T_tv, O_tv);
}

template <typename DTYPE, int REDUCTION_T>
__device__ void softmarginlossbackward5d(const DTYPE* __restrict__ I,
                                         const DTYPE* __restrict__ T,
                                         const DTYPE* __restrict__ dO,
                                         DTYPE* __restrict__ dI,
                                         const size_t num_elem,
                                         tensor_view_t<5> I_tv,
                                         tensor_view_t<5> T_tv,
                                         tensor_view_t<5> dO_tv,
                                         tensor_view_t<5> dI_tv)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    tensor_layout_t<5> idx(I_tv, gid);
    if(idx.layout[0] >= I_tv.size[0])
        return;

    FLOAT_ACCUM i        = CVT_FLOAT2ACCUM(I[I_tv.get_tensor_view_idx(idx)]);
    FLOAT_ACCUM t        = CVT_FLOAT2ACCUM(T[T_tv.get_tensor_view_idx(idx)]);
    FLOAT_ACCUM dO_accum = CVT_FLOAT2ACCUM(dO[dO_tv.get_tensor_view_idx(idx)]);
    FLOAT_ACCUM loss     = -t / (exp(i * t) + 1) * dO_accum;
    switch(REDUCTION_T)
    {
    case 0:
    case 1: dI[dI_tv.get_tensor_view_idx(idx)] = CVT_ACCUM2FLOAT(loss); break;
    case 2: dI[dI_tv.get_tensor_view_idx(idx)] = CVT_ACCUM2FLOAT(loss / num_elem); break;
    default: break;
    }
}

extern "C" __global__ void SoftMarginLossBackward5d(const FLOAT* __restrict__ I,
                                                    const FLOAT* __restrict__ T,
                                                    const FLOAT* __restrict__ dO,
                                                    FLOAT* __restrict__ dI,
                                                    const size_t num_elem,
                                                    tensor_view_t<5> I_tv,
                                                    tensor_view_t<5> T_tv,
                                                    tensor_view_t<5> dO_tv,
                                                    tensor_view_t<5> dI_tv)
{
    // instantiate the kernel
    softmarginlossbackward5d<FLOAT, REDUCTION_TYPE>(
        I, T, dO, dI, num_elem, I_tv, T_tv, dO_tv, dI_tv);
}
