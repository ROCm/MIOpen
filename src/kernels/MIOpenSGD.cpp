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

template <typename TIO>
__device__ void sgdFwd(const TIO* __restrict__ param_in,
                       TIO* __restrict__ param_out,
                       const TIO* __restrict__ grad,
                       const TIO* __restrict__ momentum_buffer_in,
                       TIO* __restrict__ momentum_buffer_out,
                       double lr,
                       double momentum,
                       double dampening,
                       double weight_decay,
                       bool nesterov,
                       bool momentum_initialized,
                       uint64_t param_size,
                       tensor_view_t<4> param_in_tv,
                       tensor_view_t<4> param_out_tv,
                       tensor_view_t<4> grad_tv,
                       tensor_view_t<4> momentum_buffer_in_tv,
                       tensor_view_t<4> momentum_buffer_out_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= param_size)
        return;

    auto tensor_layout = tensor_layout_t<4>(param_in_tv, gid);
    FLOAT_ACCUM param  = CVT_FLOAT2ACCUM(param_in[param_in_tv.get_tensor_view_idx(tensor_layout)]);
    FLOAT_ACCUM d_p    = CVT_FLOAT2ACCUM(grad[grad_tv.get_tensor_view_idx(tensor_layout)]);

    if(weight_decay)
    {
        d_p += param * static_cast<FLOAT_ACCUM>(weight_decay);
    }

    if(momentum)
    {
        FLOAT_ACCUM momentum_v;
        if(momentum_initialized != 0)
        {
            momentum_v = CVT_FLOAT2ACCUM(
                momentum_buffer_in[momentum_buffer_in_tv.get_tensor_view_idx(tensor_layout)]);
            momentum_v = momentum_v * static_cast<FLOAT_ACCUM>(momentum) +
                         d_p * static_cast<FLOAT_ACCUM>(1 - dampening);
        }
        else
        {
            momentum_v = d_p;
        }
        momentum_buffer_out[momentum_buffer_out_tv.get_tensor_view_idx(tensor_layout)] =
            CVT_ACCUM2FLOAT(momentum_v);

        if(nesterov != 0)
        {
            d_p = d_p + momentum_v * static_cast<FLOAT_ACCUM>(momentum);
        }
        else
        {
            d_p = momentum_v;
        }
    }

    param_out[param_out_tv.get_tensor_view_idx(tensor_layout)] =
        CVT_ACCUM2FLOAT(param - static_cast<FLOAT_ACCUM>(lr) * d_p);
}

extern "C" __global__ void SGDFwd(const D_TYPE* __restrict__ param_in,
                                  D_TYPE* __restrict__ param_out,
                                  const D_TYPE* __restrict__ grad,
                                  const D_TYPE* __restrict__ momentum_buffer_in,
                                  D_TYPE* __restrict__ momentum_buffer_out,
                                  double lr,
                                  double momentum,
                                  double dampening,
                                  double weight_decay,
                                  bool nesterov,
                                  bool momentum_initialized,
                                  uint64_t param_size,
                                  tensor_view_t<4> param_in_tv,
                                  tensor_view_t<4> param_out_tv,
                                  tensor_view_t<4> grad_tv,
                                  tensor_view_t<4> momentum_buffer_in_tv,
                                  tensor_view_t<4> momentum_buffer_out_tv)
{
    sgdFwd<D_TYPE>(param_in,
                   param_out,
                   grad,
                   momentum_buffer_in,
                   momentum_buffer_out,
                   lr,
                   momentum,
                   dampening,
                   weight_decay,
                   nesterov,
                   momentum_initialized,
                   param_size,
                   param_in_tv,
                   param_out_tv,
                   grad_tv,
                   momentum_buffer_in_tv,
                   momentum_buffer_out_tv);
}

template <typename TIO>
__device__ void sgdFwdContiguous(const TIO* __restrict__ param_in,
                                 TIO* __restrict__ param_out,
                                 const TIO* __restrict__ grad,
                                 const TIO* __restrict__ momentum_buffer_in,
                                 TIO* __restrict__ momentum_buffer_out,
                                 double lr,
                                 double momentum,
                                 double dampening,
                                 double weight_decay,
                                 bool nesterov,
                                 bool momentum_initialized,
                                 uint64_t param_size)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= param_size)
        return;

    FLOAT_ACCUM param = CVT_FLOAT2ACCUM(param_in[gid]);
    FLOAT_ACCUM d_p   = CVT_FLOAT2ACCUM(grad[gid]);

    if(weight_decay != 0)
    {
        d_p += param * static_cast<FLOAT_ACCUM>(weight_decay);
    }

    if(momentum != 0)
    {
        FLOAT_ACCUM momentum_v;
        if(momentum_initialized)
        {
            momentum_v = CVT_FLOAT2ACCUM(momentum_buffer_in[gid]);
            momentum_v = momentum_v * static_cast<FLOAT_ACCUM>(momentum) +
                         d_p * static_cast<FLOAT_ACCUM>(1 - dampening);
        }
        else
        {
            momentum_v = d_p;
        }
        momentum_buffer_out[gid] = CVT_ACCUM2FLOAT(momentum_v);

        if(nesterov)
        {
            d_p = d_p + momentum_v * static_cast<FLOAT_ACCUM>(momentum);
        }
        else
        {
            d_p = momentum_v;
        }
    }

    param_out[gid] = CVT_ACCUM2FLOAT(param - static_cast<FLOAT_ACCUM>(lr) * d_p);
}

extern "C" __global__ void SGDFwdContiguous(const D_TYPE* __restrict__ param_in,
                                            D_TYPE* __restrict__ param_out,
                                            const D_TYPE* __restrict__ grad,
                                            const D_TYPE* __restrict__ momentum_buffer_in,
                                            D_TYPE* __restrict__ momentum_buffer_out,
                                            double lr,
                                            double momentum,
                                            double dampening,
                                            double weight_decay,
                                            bool nesterov,
                                            bool momentum_initialized,
                                            uint64_t param_size)
{
    sgdFwdContiguous<D_TYPE>(param_in,
                             param_out,
                             grad,
                             momentum_buffer_in,
                             momentum_buffer_out,
                             lr,
                             momentum,
                             dampening,
                             weight_decay,
                             nesterov,
                             momentum_initialized,
                             param_size);
}
