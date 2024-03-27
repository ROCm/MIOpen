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
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

template <typename T>
inline __device__ void AdamInternal(size_t gid,
                                    T* param_in,
                                    T grad,
                                    T* exp_avg_in,
                                    T* exp_avg_sq_in,
                                    T* max_exp_avg_sq_in,
                                    double lr,
                                    double beta1,
                                    double beta2,
                                    double weight_decay,
                                    double eps,
                                    int step,
                                    bool amsgrad,
                                    bool maximize,
                                    T* param_out,
                                    half* param_out_fp16,
                                    T* exp_avg_out,
                                    T* exp_avg_sq_out,
                                    T* max_exp_avg_sq_out)
{
    T param      = param_in[gid];
    T exp_avg    = exp_avg_in[gid];
    T exp_avg_sq = exp_avg_sq_in[gid];

    float bias_correction1 = 1 - pow(beta1, step);
    float bias_correction2 = 1 - pow(beta2, step);

    if(maximize)
        grad *= -1;
    if(weight_decay != 0)
        grad += param * weight_decay;

    exp_avg    = exp_avg * beta1 + grad * (1 - beta1);
    exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1 - beta2);

    T denom;
    if(amsgrad)
    {
        T max_exp_avg_sq = max_exp_avg_sq_in[gid];
        if(exp_avg_sq > max_exp_avg_sq)
        {
            max_exp_avg_sq          = exp_avg_sq;
            max_exp_avg_sq_out[gid] = max_exp_avg_sq;
        }

        denom = sqrt(max_exp_avg_sq) / sqrt(bias_correction2) + eps;
    }
    else
    {
        denom = sqrt(exp_avg_sq) / sqrt(bias_correction2) + eps;
    }

    float step_size = lr / bias_correction1;
    param           = param - step_size * exp_avg / denom;

    if(param_out_fp16)
        param_out_fp16[gid] = (half)param;
    param_out[gid]      = param;
    exp_avg_out[gid]    = exp_avg;
    exp_avg_sq_out[gid] = exp_avg_sq;
}

extern "C" __global__ void AdamPacked(FLOAT* params_in,
                                      FLOAT* grad_in,
                                      FLOAT* exp_avg_in,
                                      FLOAT* exp_avg_sq_in,
                                      FLOAT* max_exp_avg_sq_in,
                                      int step,
                                      double lr,
                                      double beta1,
                                      double beta2,
                                      double weight_decay,
                                      double eps,
                                      bool amsgrad,
                                      bool maximize,
                                      FLOAT* param_out,
                                      FLOAT* exp_avg_out,
                                      FLOAT* exp_avg_sq_out,
                                      FLOAT* max_exp_avg_sq_out,
                                      long input_size)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if(gid >= input_size)
        return;

    AdamInternal<FLOAT>(gid,
                        params_in,
                        grad_in[gid],
                        exp_avg_in,
                        exp_avg_sq_in,
                        max_exp_avg_sq_in,
                        lr,
                        beta1,
                        beta2,
                        weight_decay,
                        eps,
                        step,
                        amsgrad,
                        maximize,
                        param_out,
                        nullptr,
                        exp_avg_out,
                        exp_avg_sq_out,
                        max_exp_avg_sq_out);
}

extern "C" __global__ void AmpAdamPacked(FLOAT* param_in,
                                         FLOAT* grad_in,
                                         FLOAT* exp_avg_in,
                                         FLOAT* exp_avg_sq_in,
                                         FLOAT* max_exp_avg_sq_in,
                                         int32_t* grad_scale,
                                         bool* found_inf,
                                         int* step,
                                         double lr,
                                         double beta1,
                                         double beta2,
                                         double weight_decay,
                                         double eps,
                                         bool amsgrad,
                                         bool maximize,
                                         FLOAT* param_out,
                                         half* param_out_fp16,
                                         FLOAT* exp_avg_out,
                                         FLOAT* exp_avg_sq_out,
                                         FLOAT* max_exp_avg_sq_out,
                                         long input_size)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t lid = threadIdx.x;

    if(gid >= input_size)
        return;

    __shared__ int step_val;
    __shared__ bool skip;
    __shared__ float scale_factor;

    if(lid == 0)
    {
        skip         = (found_inf) ? *found_inf : false;
        scale_factor = (grad_scale) ? *grad_scale : 1.0f;
        step_val     = *step + 1;
    }
    __syncthreads();

    if(skip)
        return;

    FLOAT grad = grad_in[gid] / scale_factor;

    AdamInternal<FLOAT>(gid,
                        param_in,
                        grad,
                        exp_avg_in,
                        exp_avg_sq_in,
                        max_exp_avg_sq_in,
                        lr,
                        beta1,
                        beta2,
                        weight_decay,
                        eps,
                        step_val,
                        amsgrad,
                        maximize,
                        param_out,
                        param_out_fp16,
                        exp_avg_out,
                        exp_avg_sq_out,
                        max_exp_avg_sq_out);
}

extern "C" __global__ void AdamUpdateStep(bool* found_inf, int* step)
{
    if(found_inf && *found_inf)
        return;

    *step += 1;
}
