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
#include "MIOpenLossReductionMode.hpp"

template <typename TIO, uint32_t NDIM, LossReductionMode_t REDUCTION_T>
__device__ void MSELossForward(const TIO* __restrict__ I,
                               const TIO* __restrict__ T,
                               void* O,
                               const uint64_t size,
                               tensor_view_t<NDIM> I_tv,
                               tensor_view_t<NDIM> T_tv,
                               tensor_view_t<NDIM> O_tv)
{
    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    tensor_layout_t<5> tensor_layout(I_tv, gid);
    if(tensor_layout.layout[0] >= I_tv.size[0])
        return;

    FLOAT_ACCUM i = CVT_FLOAT2ACCUM(I[I_tv.get_tensor_view_idx(tensor_layout)]);
    FLOAT_ACCUM t = CVT_FLOAT2ACCUM(T[T_tv.get_tensor_view_idx(tensor_layout)]);

    FLOAT_ACCUM loss = (i - t) * (i - t);

    switch(REDUCTION_T)
    {
    case LossReductionMode_t::NONE:
        static_cast<TIO*>(O)[O_tv.get_tensor_view_idx(tensor_layout)] = CVT_ACCUM2FLOAT(loss);
        break;
    case LossReductionMode_t::SUM: static_cast<FLOAT_ACCUM*>(O)[gid] = loss; break;
    case LossReductionMode_t::MEAN: static_cast<FLOAT_ACCUM*>(O)[gid] = loss / size; break;
    default: break;
    }
}

extern "C" __global__ void MSELossForward(const FLOAT* __restrict__ I,
                                          const FLOAT* __restrict__ T,
                                          void* __restrict__ O,
                                          const uint64_t size,
                                          tensor_view_t<VIEW_DIMS> I_tv,
                                          tensor_view_t<VIEW_DIMS> T_tv,
                                          tensor_view_t<VIEW_DIMS> O_tv)

{
    // instantiate the kernel
    MSELossForward<FLOAT, VIEW_DIMS, static_cast<LossReductionMode_t>(REDUCTION_TYPE)>(
        I, T, O, size, I_tv, T_tv, O_tv);
}

template <typename TIO, uint32_t NDIM, LossReductionMode_t REDUCTION_T>
__device__ void MSELossBackward(const TIO* __restrict__ I,
                                const TIO* __restrict__ T,
                                const TIO* __restrict__ dO,
                                TIO* __restrict__ dI,
                                TIO* __restrict__ dT,
                                const uint64_t size,
                                tensor_view_t<5> I_tv,
                                tensor_view_t<5> T_tv,
                                tensor_view_t<5> dO_tv,
                                tensor_view_t<5> dI_tv,
                                tensor_view_t<5> dT_tv)
{
    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    tensor_layout_t<5> tensor_layout(I_tv, gid);
    if(tensor_layout.layout[0] >= I_tv.size[0])
        return;

    FLOAT_ACCUM o_grad;
    switch(REDUCTION_T)
    {
    case LossReductionMode_t::NONE:
        o_grad = CVT_FLOAT2ACCUM(dO[dO_tv.get_tensor_view_idx(tensor_layout)]);
        break;
    case LossReductionMode_t::SUM: o_grad = CVT_FLOAT2ACCUM(dO[0]); break;
    case LossReductionMode_t::MEAN: o_grad = CVT_FLOAT2ACCUM(dO[0]) / size; break;
    default: break;
    }

    FLOAT_ACCUM i    = CVT_FLOAT2ACCUM(I[I_tv.get_tensor_view_idx(tensor_layout)]);
    FLOAT_ACCUM t    = CVT_FLOAT2ACCUM(T[T_tv.get_tensor_view_idx(tensor_layout)]);
    FLOAT_ACCUM grad = 2.0f * (i - t) * o_grad;

    if(dI)
        dI[dI_tv.get_tensor_view_idx(tensor_layout)] = CVT_ACCUM2FLOAT(grad);
    if(dT)
        dT[dT_tv.get_tensor_view_idx(tensor_layout)] = CVT_ACCUM2FLOAT(-grad);
}

extern "C" __global__ void MSELossBackward(const FLOAT* __restrict__ I,
                                           const FLOAT* __restrict__ T,
                                           const FLOAT* __restrict__ dO,
                                           FLOAT* __restrict__ dI,
                                           FLOAT* __restrict__ dT,
                                           const uint64_t size,
                                           tensor_view_t<5> I_tv,
                                           tensor_view_t<5> T_tv,
                                           tensor_view_t<5> dO_tv,
                                           tensor_view_t<5> dI_tv,
                                           tensor_view_t<5> dT_tv)
{
    // instantiate the kernel
    MSELossBackward<FLOAT, VIEW_DIMS, static_cast<LossReductionMode_t>(REDUCTION_TYPE)>(
        I, T, dO, dI, dT, size, I_tv, T_tv, dO_tv, dI_tv, dT_tv);
}
