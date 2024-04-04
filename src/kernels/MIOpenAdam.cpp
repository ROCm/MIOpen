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
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

template <typename T1, typename T2>
inline __device__ void AdamInternal(T1* param_in,
                                    T1* param_out,
                                    T1* exp_avg_in,
                                    T1* exp_avg_out,
                                    T1* exp_avg_sq_in,
                                    T1* exp_avg_sq_out,
                                    T1* max_exp_avg_sq_in,
                                    T1* max_exp_avg_sq_out,
                                    T2 grad,
                                    T2 lr,
                                    T2 beta1,
                                    T2 beta2,
                                    T2 weight_decay,
                                    T2 eps,
                                    int step,
                                    bool amsgrad,
                                    bool maximize,
                                    size_t gid)
{
    T2 param      = static_cast<T2>(param_in[gid]);
    T2 exp_avg    = static_cast<T2>(exp_avg_in[gid]);
    T2 exp_avg_sq = static_cast<T2>(exp_avg_sq_in[gid]);

    __builtin_assume(exp_avg_sq >= 0 && exp_avg_sq <= 1);
    __builtin_assume(step >= 0);
    __builtin_assume(beta1 >= 0);
    __builtin_assume(beta2 >= 0);

    T2 bias_correction1 = 1 - pow(beta1, step);
    T2 bias_correction2 = 1 - pow(beta2, step);

    if(maximize)
        grad *= -1;
    if(weight_decay != 0)
        grad += param * weight_decay;

    exp_avg    = exp_avg * beta1 + grad * (1 - beta1);
    exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1 - beta2);

    T2 denom;
    if(amsgrad)
    {
        T2 max_exp_avg_sq = static_cast<T2>(max_exp_avg_sq_in[gid]);
        __builtin_assume(max_exp_avg_sq >= 0 && max_exp_avg_sq <= 1);
        max_exp_avg_sq          = max(max_exp_avg_sq, exp_avg_sq);
        max_exp_avg_sq_out[gid] = static_cast<T1>(max_exp_avg_sq);
        denom                   = sqrt(max_exp_avg_sq) / sqrt(bias_correction2) + eps;
    }
    else
    {
        denom = sqrt(exp_avg_sq) / sqrt(bias_correction2) + eps;
    }

    T2 step_size = lr / bias_correction1;
    param        = param - step_size * exp_avg / denom;

    param_out[gid]      = static_cast<T1>(param);
    exp_avg_out[gid]    = static_cast<T1>(exp_avg);
    exp_avg_sq_out[gid] = static_cast<T1>(exp_avg_sq);
}

extern "C" __global__ void AdamPacked(PTYPE* param_in,
                                      PTYPE* param_out,
                                      PTYPE* grad_in,
                                      PTYPE* exp_avg_in,
                                      PTYPE* exp_avg_out,
                                      PTYPE* exp_avg_sq_in,
                                      PTYPE* exp_avg_sq_out,
                                      PTYPE* max_exp_avg_sq_in,
                                      PTYPE* max_exp_avg_sq_out,
                                      float lr,
                                      float beta1,
                                      float beta2,
                                      float weight_decay,
                                      float eps,
                                      int step,
                                      bool amsgrad,
                                      bool maximize,
                                      size_t input_size)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gsz = gridDim.x * blockDim.x;

    for(; gid < input_size; gid += gsz)
    {
        CTYPE grad = static_cast<CTYPE>(grad_in[gid]);

        AdamInternal<PTYPE, CTYPE>(param_in,
                                   param_out,
                                   exp_avg_in,
                                   exp_avg_out,
                                   exp_avg_sq_in,
                                   exp_avg_sq_out,
                                   max_exp_avg_sq_in,
                                   max_exp_avg_sq_out,
                                   grad,
                                   lr,
                                   beta1,
                                   beta2,
                                   weight_decay,
                                   eps,
                                   step,
                                   amsgrad,
                                   maximize,
                                   gid);
    }
}

extern "C" __global__ void AmpAdamPacked(PTYPE* param_in,
                                         PTYPE* param_out,
                                         half* param_out_fp16,
                                         GTYPE* grad_in,
                                         PTYPE* exp_avg_in,
                                         PTYPE* exp_avg_out,
                                         PTYPE* exp_avg_sq_in,
                                         PTYPE* exp_avg_sq_out,
                                         PTYPE* max_exp_avg_sq_in,
                                         PTYPE* max_exp_avg_sq_out,
                                         int32_t* grad_scale,
                                         bool* found_inf,
                                         int* step,
                                         float lr,
                                         float beta1,
                                         float beta2,
                                         float weight_decay,
                                         float eps,
                                         bool amsgrad,
                                         bool maximize,
                                         size_t input_size)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gsz = gridDim.x * blockDim.x;

    if(gid >= input_size)
        return;

    if(found_inf && *found_inf)
        return;

    CTYPE scale_factor = (grad_scale) ? static_cast<CTYPE>(*grad_scale) : 1.0f;
    int step_val       = *step + 1;

    for(; gid < input_size; gid += gsz)
    {
        CTYPE grad = static_cast<CTYPE>(grad_in[gid]);
        grad /= scale_factor;

        AdamInternal<PTYPE, CTYPE>(param_in,
                                   param_out,
                                   exp_avg_in,
                                   exp_avg_out,
                                   exp_avg_sq_in,
                                   exp_avg_sq_out,
                                   max_exp_avg_sq_in,
                                   max_exp_avg_sq_out,
                                   grad,
                                   lr,
                                   beta1,
                                   beta2,
                                   weight_decay,
                                   eps,
                                   step_val,
                                   amsgrad,
                                   maximize,
                                   gid);

        if(param_out_fp16)
            param_out_fp16[gid] = static_cast<half>(param_out[gid]);
    }
}

extern "C" __global__ void AdamUpdateStep(bool* found_inf, int* step_in, int* step_out)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if(gid != 0)
        return;

    if(found_inf && *found_inf)
        return;

    *step_out = *step_in + 1;
}
