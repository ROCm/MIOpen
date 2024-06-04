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

#ifndef IN_OUT_TYPE
#define IN_OUT_TYPE float
#endif

#ifndef TARGET_TYPE
#define TARGET_TYPE int
#endif

template <typename TIO, typename TT>
__device__ void multilabelMarginLossForward2d(const TIO* __restrict__ I,
                                                const TT* __restrict__ T,
                                                TIO* __restrict__ lsum,
                                                char * ws,
                                                long ws_offset,
                                                const float divisor,
                                                const size_t I_size_0,
                                                const size_t I_size_1,
                                                const size_t T_size_0,
                                                const size_t T_size_1,
                                                const size_t I_stride_0,
                                                const size_t I_stride_1,
                                                const size_t T_stride_0,
                                                const size_t T_stride_1)
{
    const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t N = I_size_0, C = I_size_1;
    size_t n = gid;

    if (!(n < I_size_0)) return;

    ws = ws + ws_offset;
    for (size_t c = 0; c < C; c++) 
    {
        ws[n * C + c] = 0;
    }

    for (size_t c = 0; c < C; c++) 
    {
        int is_target_idx = 0;
        for (size_t i = 0; i < C; i++)
        {
            size_t T_at_n_i = T[I_stride_1 * i + T_stride_0 * n];
            if (T_at_n_i == -1) break;
            if (T_at_n_i == c) 
            {
                is_target_idx = 1;
                break;
            }
        }
        if (is_target_idx)
        {
            ws[n * C + c] = 1;
        }
    }

    FLOAT_ACCUM loss = CVT_FLOAT2ACCUM(0.0f);

    for (size_t ct = 0; ct < C; ct++)
    {
        size_t T_at_n_ct = T[T_stride_1 * ct + T_stride_0 * n];
        if (T_at_n_ct == -1) break;
        for (size_t ci = 0; ci < C; ci++)
        {
            if (ws[n * C + ci] == 0)
            {
                FLOAT_ACCUM t = CVT_FLOAT2ACCUM(1.0f) - CVT_FLOAT2ACCUM(I[I_stride_1 * T_at_n_ct + I_stride_0 * n]) - CVT_FLOAT2ACCUM(I[I_stride_1 * ci + I_stride_0 * n]);
                t /= C;
                loss += t >= 0 ? t : CVT_FLOAT2ACCUM(0.0f);
            }
        }
    }

    lsum[n] = CVT_ACCUM2FLOAT(loss / divisor);
}

extern "C" __global__ void MultilabelMarginLossForward2d(const IN_OUT_TYPE* __restrict__ I,
                                                        const TARGET_TYPE* __restrict__ T,
                                                        IN_OUT_TYPE* __restrict__ lsum,
                                                        char * ws,
                                                        long ws_offset,
                                                        const float divisor,
                                                        const size_t I_size_0,
                                                        const size_t I_size_1,
                                                        const size_t T_size_0,
                                                        const size_t T_size_1,
                                                        const size_t I_stride_0,
                                                        const size_t I_stride_1,
                                                        const size_t T_stride_0,
                                                        const size_t T_stride_1)
{
    multilabelMarginLossForward2d<IN_OUT_TYPE, TARGET_TYPE>(I,
                                                T,
                                                lsum,
                                                ws,
                                                ws_offset,
                                                divisor,
                                                I_size_0,
                                                I_size_1,
                                                T_size_0,
                                                T_size_1,
                                                I_stride_0,
                                                I_stride_1,
                                                T_stride_0,
                                                T_stride_1);
}

template <typename TIO, typename TT>
__device__ void multilabelMarginLossBackward2d(const TIO* __restrict__ I,
                                                const TT* __restrict__ T,
                                                const TIO* __restrict__ dO,
                                                TIO* __restrict__ dI,
                                                char * ws,
                                                const float divisor,
                                                const size_t I_size_0,
                                                const size_t I_size_1,
                                                const size_t T_size_0,
                                                const size_t T_size_1,
                                                const size_t I_stride_0,
                                                const size_t I_stride_1,
                                                const size_t T_stride_0,
                                                const size_t T_stride_1,
                                                const size_t dI_stride_0,
                                                const size_t dI_stride_1,
                                                const size_t dO_stride_0)
{
    const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t N = I_size_0, C = I_size_1;
    size_t n = gid;

    if (!(n < I_size_0)) return;

    for (size_t c = 0; c < C; c++) 
    {
        ws[n * C + c] = 0;
        dI[(dI_stride_1 * c) + (dI_stride_0 * n)] = 0.0f;
    }

    for (size_t c = 0; c < C; c++) 
    {
        int is_target_idx = 0;
        for (size_t i = 0; i < C; i++)
        {
            size_t T_at_n_i = T[I_stride_1 * i + T_stride_0 * n];
            if (T_at_n_i == -1) break;
            if (T_at_n_i == c) 
            {
                is_target_idx = 1;
                break;
            }
        }
        if (is_target_idx)
        {
            ws[n * C + c] = 1;
        }
    }

    FLOAT_ACCUM out_grad = CVT_FLOAT2ACCUM(dO[dO_stride_0 * 0]);
    FLOAT_ACCUM delta = 1.0f / C * out_grad / divisor;
    for (size_t ct = 0; ct < C; ct++)
    {
        size_t T_at_n_ct = T[T_stride_1 * ct + T_stride_0 * n];
        if (T_at_n_ct == -1) break;
        for (size_t ci = 0; ci < C; ci++)
        {
            if (ws[n * C + ci] == 0)
            {
                FLOAT_ACCUM t = CVT_FLOAT2ACCUM(1.0f) - CVT_FLOAT2ACCUM(I[I_stride_1 * T_at_n_ct + I_stride_0 * n]) - CVT_FLOAT2ACCUM(I[I_stride_1 * ci + I_stride_0 * n]);
                if (t >= 0)
                {
                    FLOAT_ACCUM x = CVT_FLOAT2ACCUM(dI[(dI_stride_1 * ci) + (dI_stride_0 * n)]) + delta;
                    dI[(dI_stride_1 * ci) + (dI_stride_0 * n)] = CVT_ACCUM2FLOAT(x);
                    FLOAT_ACCUM y = CVT_FLOAT2ACCUM(dI[(dI_stride_1 * T_at_n_ct) + (dI_stride_0 * n)]) - delta;
                    dI[(dI_stride_1 * T_at_n_ct) + (dI_stride_0 * n)] = CVT_ACCUM2FLOAT(y);
                }
            }
        }
    }
}

extern "C" __global__ void MultilabelMarginLossBackward2d(const IN_OUT_TYPE* __restrict__ I,
                                                        const TARGET_TYPE* __restrict__ T,
                                                        const IN_OUT_TYPE* __restrict__ dO,
                                                        IN_OUT_TYPE* __restrict__ dI,
                                                        char * ws,
                                                        const float divisor,
                                                        const size_t I_size_0,
                                                        const size_t I_size_1,
                                                        const size_t T_size_0,
                                                        const size_t T_size_1,
                                                        const size_t I_stride_0,
                                                        const size_t I_stride_1,
                                                        const size_t T_stride_0,
                                                        const size_t T_stride_1,
                                                        const size_t dI_stride_0,
                                                        const size_t dI_stride_1,
                                                        const size_t dO_stride_0)
{
    multilabelMarginLossBackward2d<IN_OUT_TYPE, TARGET_TYPE>(I,
                                                T,
                                                dO,
                                                dI,
                                                ws,
                                                divisor,
                                                I_size_0,
                                                I_size_1,
                                                T_size_0,
                                                T_size_1,
                                                I_stride_0,
                                                I_stride_1,
                                                T_stride_0,
                                                T_stride_1,
                                                dI_stride_0,
                                                dI_stride_1,
                                                dO_stride_0);
}

template <typename TIO, typename TT>
__device__ void multilabelMarginLossUnreducedForward2d(const TIO* __restrict__ I,
                                                const TT* __restrict__ T,
                                                TIO* __restrict__ O,
                                                char * ws,
                                                const size_t I_size_0,
                                                const size_t I_size_1,
                                                const size_t T_size_0,
                                                const size_t T_size_1,
                                                const size_t I_stride_0,
                                                const size_t I_stride_1,
                                                const size_t T_stride_0,
                                                const size_t T_stride_1,
                                                const size_t O_stride_0)
{
    const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t N = I_size_0, C = I_size_1;
    size_t n = gid;

    if (!(n < I_size_0)) return;

    for (size_t c = 0; c < C; c++)
    {
        ws[n * C + c] = 0;
    }

    /* For each input, determine if it is the target */
    for (size_t c = 0; c < C; c++) 
    {
        int is_target_idx = 0;
        for (size_t i = 0; i < C; i++)
        {
            size_t T_at_n_i = T[I_stride_1 * i + T_stride_0 * n];
            if (T_at_n_i == -1) break;
            if (T_at_n_i == c) 
            {
                is_target_idx = 1;
                break;
            }
        }
        if (is_target_idx)
        {
            ws[n * C + c] = 1;
        }
    }

    FLOAT_ACCUM loss = CVT_FLOAT2ACCUM(0.0f);

    for (size_t ct = 0; ct < C; ct++)
    {
        size_t T_at_n_ct = T[T_stride_1 * ct + T_stride_0 * n];
        if (T_at_n_ct == -1) break;
        for (size_t ci = 0; ci < C; ci++)
        {
            if (ws[n * C + ci] == 0)
            {
                FLOAT_ACCUM t = CVT_FLOAT2ACCUM(1.0f) - CVT_FLOAT2ACCUM(I[I_stride_1 * T_at_n_ct + I_stride_0 * n]) - CVT_FLOAT2ACCUM(I[I_stride_1 * ci + I_stride_0 * n]);
                t /= C;
                loss += t >= 0 ? t : CVT_FLOAT2ACCUM(0.0f);
            }
        }
    }

    O[O_stride_0 * n] = CVT_ACCUM2FLOAT(loss);
}

extern "C" __global__ void MultilabelMarginLossUnreducedForward2d(const IN_OUT_TYPE* __restrict__ I,
                                                        const TARGET_TYPE* __restrict__ T,
                                                        IN_OUT_TYPE* __restrict__ O,
                                                        char * ws,
                                                        const size_t I_size_0,
                                                        const size_t I_size_1,
                                                        const size_t T_size_0,
                                                        const size_t T_size_1,
                                                        const size_t I_stride_0,
                                                        const size_t I_stride_1,
                                                        const size_t T_stride_0,
                                                        const size_t T_stride_1,
                                                        const size_t O_stride_0)
{
    multilabelMarginLossUnreducedForward2d<IN_OUT_TYPE, TARGET_TYPE>(I,
                                                T,
                                                O,
                                                ws,
                                                I_size_0,
                                                I_size_1,
                                                T_size_0,
                                                T_size_1,
                                                I_stride_0,
                                                I_stride_1,
                                                T_stride_0,
                                                T_stride_1,
                                                O_stride_0);
}

template <typename TIO, typename TT>
__device__ void multilabelMarginLossUnreducedBackward2d(const TIO* __restrict__ I,
                                                const TT* __restrict__ T,
                                                const TIO* __restrict__ dO,
                                                TIO* __restrict__ dI,
                                                char * ws,
                                                const size_t I_size_0,
                                                const size_t I_size_1,
                                                const size_t T_size_0,
                                                const size_t T_size_1,
                                                const size_t I_stride_0,
                                                const size_t I_stride_1,
                                                const size_t T_stride_0,
                                                const size_t T_stride_1,
                                                const size_t dI_stride_0,
                                                const size_t dI_stride_1,
                                                const size_t dO_stride_0)
{
    const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t N = I_size_0, C = I_size_1;
    size_t n = gid;

    if (!(n < I_size_0)) return;

    for (size_t c = 0; c < C; c++) 
    {
        ws[n * C + c] = 0;
        dI[(dI_stride_1 * c) + (dI_stride_0 * n)] = 0.0f;
    }

    for (size_t c = 0; c < C; c++) 
    {
        int is_target_idx = 0;
        for (size_t i = 0; i < C; i++)
        {
            size_t T_at_n_i = T[I_stride_1 * i + T_stride_0 * n];
            if (T_at_n_i == -1) break;
            if (T_at_n_i == c) 
            {
                is_target_idx = 1;
                break;
            }
        }
        if (is_target_idx)
        {
            ws[n * C + c] = 1;
        }
    }

    FLOAT_ACCUM out_grad = CVT_FLOAT2ACCUM(dO[dO_stride_0 * n]);
    FLOAT_ACCUM delta = 1.0f / C * out_grad;
    for (size_t ct = 0; ct < C; ct++)
    {
        size_t T_at_n_ct = T[T_stride_1 * ct + T_stride_0 * n];
        if (T_at_n_ct == -1) break;
        for (size_t ci = 0; ci < C; ci++)
        {
            if (ws[n * C + ci] == 0)
            {
                FLOAT_ACCUM t = CVT_FLOAT2ACCUM(1.0f) - CVT_FLOAT2ACCUM(I[I_stride_1 * T_at_n_ct + I_stride_0 * n]) - CVT_FLOAT2ACCUM(I[I_stride_1 * ci + I_stride_0 * n]);
                if (t >= 0)
                {
                    FLOAT_ACCUM x = CVT_FLOAT2ACCUM(dI[(dI_stride_1 * ci) + (dI_stride_0 * n)]) + delta;
                    dI[(dI_stride_1 * ci) + (dI_stride_0 * n)] = CVT_ACCUM2FLOAT(x);
                    FLOAT_ACCUM y = CVT_FLOAT2ACCUM(dI[(dI_stride_1 * T_at_n_ct) + (dI_stride_0 * n)]) - delta;
                    dI[(dI_stride_1 * T_at_n_ct) + (dI_stride_0 * n)] = CVT_ACCUM2FLOAT(y);
                }
            }
        }
    }
}

extern "C" __global__ void MultilabelMarginLossUnreducedBackward2d(const IN_OUT_TYPE* __restrict__ I,
                                                        const TARGET_TYPE* __restrict__ T,
                                                        const IN_OUT_TYPE* __restrict__ dO,
                                                        IN_OUT_TYPE* __restrict__ dI,
                                                        char * ws,
                                                        const size_t I_size_0,
                                                        const size_t I_size_1,
                                                        const size_t T_size_0,
                                                        const size_t T_size_1,
                                                        const size_t I_stride_0,
                                                        const size_t I_stride_1,
                                                        const size_t T_stride_0,
                                                        const size_t T_stride_1,
                                                        const size_t dI_stride_0,
                                                        const size_t dI_stride_1,
                                                        const size_t dO_stride_0)
{
    multilabelMarginLossUnreducedBackward2d<IN_OUT_TYPE, TARGET_TYPE>(I,
                                                T,
                                                dO,
                                                dI,
                                                ws,
                                                I_size_0,
                                                I_size_1,
                                                T_size_0,
                                                T_size_1,
                                                I_stride_0,
                                                I_stride_1,
                                                T_stride_0,
                                                T_stride_1,
                                                dI_stride_0,
                                                dI_stride_1,
                                                dO_stride_0);
}