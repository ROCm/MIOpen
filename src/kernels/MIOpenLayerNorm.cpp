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
#include "miopen_cstdint.hpp"

#include "float_types.h"

template <typename TI, typename TO>
__device__ void layernormfwd(const TI* __restrict__ x,
                             const TI* __restrict__ weight,
                             const TI* __restrict__ bias,
                             TO* __restrict__ y,
                             TO* __restrict__ mean,
                             TO* __restrict__ rstd,
                             const float eps,
                             const int32_t mode)
{
    /*
     * Each group works on a single channel.
     * Example)
     * x dim = {N, C, L}, normalized shape = {C, L}, layout = NCHW or NHWC
     * outer_size = N, inner_size = C * L, stride = 1
     *
     * Example2)
     * x dim = {N, C, L}, normalized shape = {L}, layout = NCHW
     * outer_size = N * C, inner_size = L, stride = 1
     *
     * Example3)
     * x dim = {N, C, L}, normalized shape = {L}, layout = NHWC
     * outer_size = N, inner_size = L, stride = C
     *
     * => gws = {outer_size * stride * LOCAL_SIZE}, lws = {LOCAL_SIZE}
     */

    /*
     * Reduction to calculate mean and rstd
     */

    const uint64_t gid = blockIdx.x;
    const uint64_t lid = threadIdx.x;
    const uint64_t o   = gid / STRIDE;
    const uint64_t s   = gid % STRIDE;

    FLOAT_ACCUM pmean = static_cast<FLOAT_ACCUM>(0);
    FLOAT_ACCUM pvar  = static_cast<FLOAT_ACCUM>(0);
    __shared__ FLOAT_ACCUM ltmp1[LOCAL_SIZE];
    __shared__ FLOAT_ACCUM ltmp2[LOCAL_SIZE];

    // reduce sum for mean and var
    for(uint64_t i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t x_idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;

        FLOAT_ACCUM tmp = CVT_FLOAT2ACCUM(x[x_idx]);
        pmean += tmp;
        pvar += tmp * tmp;
    }

    ltmp1[lid] = pmean;
    ltmp2[lid] = pvar;
    __syncthreads();
    for(uint32_t i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            ltmp1[lid] += ltmp1[lid + i];
            ltmp2[lid] += ltmp2[lid + i];
        }
        __syncthreads();
    }
    pmean             = ltmp1[0] / INNER_SIZE;
    pvar              = ltmp2[0] / INNER_SIZE - pmean * pmean;
    FLOAT_ACCUM prstd = rsqrt(pvar + FLOAT_ACCUM(eps));

    if(lid == 0)
    {
        if(mean)
            mean[gid] = CVT_ACCUM2FLOAT(pmean);
        if(rstd)
            rstd[gid] = CVT_ACCUM2FLOAT(prstd);
    }

    // forward calculation
    for(uint64_t i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;

        FLOAT_ACCUM pweight;
        FLOAT_ACCUM pbias;

        pweight = (mode == MIOPEN_ELEMENTWISE_AFFINE) ? CVT_FP32_2ACCUM(1.0f)
                                                      : CVT_FLOAT2ACCUM(weight[i]);
        pbias =
            (mode == MIOPEN_ELEMENTWISE_AFFINE) ? static_cast<FLOAT>(0) : CVT_FLOAT2ACCUM(bias[i]);

        FLOAT_ACCUM val = (CVT_FLOAT2ACCUM(x[idx]) - pmean) * prstd * pweight + pbias;
        y[idx]          = CVT_ACCUM2FLOAT(val);
    }
}

template <typename TI, typename TO>
__device__ void layernormbwd(const TI* __restrict__ dy,
                             const TI* __restrict__ x,
                             const TI* __restrict__ weight,
                             const TI* __restrict__ mean,
                             const TI* __restrict__ rstd,
                             TO* __restrict__ dx,
                             const int32_t mode)
{
    const uint64_t gid = blockIdx.x;
    const uint64_t lid = threadIdx.x;
    const uint64_t o   = gid / STRIDE;
    const uint64_t s   = gid % STRIDE;

    __shared__ FLOAT_ACCUM ltmp1[LOCAL_SIZE];
    __shared__ FLOAT_ACCUM ltmp2[LOCAL_SIZE];
    FLOAT_ACCUM sum_dy_weight   = 0;
    FLOAT_ACCUM sum_dy_weight_x = 0;

    // Reduce sums
    if(dy)
    {
        for(uint64_t i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
        {
            size_t x_idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;

            FLOAT_ACCUM pdy_pweight =
                CVT_FLOAT2ACCUM(dy[x_idx]) * ((mode == MIOPEN_ELEMENTWISE_AFFINE)
                                                  ? CVT_FP32_2ACCUM(1.0f)
                                                  : CVT_FLOAT2ACCUM(weight[i]));

            sum_dy_weight += pdy_pweight;
            sum_dy_weight_x += pdy_pweight * CVT_FLOAT2ACCUM(x[x_idx]);
        }
    }

    ltmp1[lid] = sum_dy_weight;
    ltmp2[lid] = sum_dy_weight_x;
    __syncthreads();
    for(uint32_t i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            ltmp1[lid] += ltmp1[lid + i];
            ltmp2[lid] += ltmp2[lid + i];
        }
        __syncthreads();
    }

    sum_dy_weight     = ltmp1[0];
    sum_dy_weight_x   = ltmp2[0];
    FLOAT_ACCUM scale = 1.0f / INNER_SIZE;
    FLOAT_ACCUM prstd = CVT_FLOAT2ACCUM(rstd[gid]);
    FLOAT_ACCUM pmean = CVT_FLOAT2ACCUM(mean[gid]);
    FLOAT_ACCUM a     = prstd * prstd * prstd * scale * (sum_dy_weight_x - sum_dy_weight * pmean);
    FLOAT_ACCUM b     = prstd * sum_dy_weight * scale - a * pmean;

    // Backward calculation
    for(uint64_t i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;

        FLOAT_ACCUM pdy     = dy ? CVT_FLOAT2ACCUM(dy[idx]) : 0;
        FLOAT_ACCUM pweight = (mode == MIOPEN_ELEMENTWISE_AFFINE) ? CVT_FP32_2ACCUM(1.0f)
                                                                  : CVT_FLOAT2ACCUM(weight[i]);

        FLOAT_ACCUM val = prstd * pdy * pweight - a * CVT_FLOAT2ACCUM(x[idx]) - b;
        dx[idx]         = CVT_ACCUM2FLOAT(val);
    }
}

template <typename TI, typename TO>
__device__ void layernormbwdweightbias(const TI* __restrict__ dy,
                                       const TI* __restrict__ x,
                                       const TI* __restrict__ mean,
                                       const TI* __restrict__ rstd,
                                       TO* __restrict__ dw,
                                       TO* __restrict__ db)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(dw || db)
    {
        FLOAT_ACCUM sum_dw = 0;
        FLOAT_ACCUM sum_db = 0;

        // Backward calculation
        for(uint64_t o = 0; o < OUTER_SIZE; ++o)
        {
            for(uint64_t s = 0; s < STRIDE; ++s)
            {
                uint64_t input_idx = o * INNER_SIZE * STRIDE + gid * STRIDE + s;

                FLOAT_ACCUM prstd = CVT_FLOAT2ACCUM(rstd[o * STRIDE + s]);
                FLOAT_ACCUM pmean = CVT_FLOAT2ACCUM(mean[o * STRIDE + s]);
                FLOAT_ACCUM pdy   = dy ? CVT_FLOAT2ACCUM(dy[input_idx]) : 0;

                sum_dw += prstd * pdy * (CVT_FLOAT2ACCUM(x[input_idx]) - pmean);
                sum_db += pdy;
            }
        }

        if(dw)
        {
            dw[gid] = CVT_ACCUM2FLOAT(sum_dw);
        }
        if(db)
        {
            db[gid] = CVT_ACCUM2FLOAT(sum_db);
        }
    }
}

template <typename TI, typename TO>
__device__ void layernormbwdweightbiasparallel(const TI* __restrict__ dy,
                                               const TI* __restrict__ x,
                                               const TI* __restrict__ mean,
                                               const TI* __restrict__ rstd,
                                               TO* __restrict__ workspace)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(gid >= INNER_SIZE * PARALLEL_SIZE)
        return;

    uint64_t pid   = gid / INNER_SIZE;
    uint64_t s_lid = (gid % INNER_SIZE) * STRIDE;

    FLOAT_ACCUM sum_dw = 0;
    FLOAT_ACCUM sum_db = 0;

    if(dy)
    {
        // Backward calculation
        for(uint64_t i = pid; i < OUTER_SIZE * STRIDE; i += PARALLEL_SIZE)
        {
            uint64_t o         = i / STRIDE;
            uint64_t s         = i % STRIDE;
            uint64_t input_idx = o * INNER_SIZE * STRIDE + s_lid + s;

            FLOAT_ACCUM prstd = CVT_FLOAT2ACCUM(rstd[i]);
            FLOAT_ACCUM pmean = CVT_FLOAT2ACCUM(mean[i]);
            FLOAT_ACCUM pdy   = CVT_FLOAT2ACCUM(dy[input_idx]);

            sum_dw += pdy * prstd * (CVT_FLOAT2ACCUM(x[input_idx]) - pmean);
            sum_db += pdy;
        }
    }

    workspace[gid]                              = CVT_ACCUM2FLOAT(sum_dw);
    workspace[gid + PARALLEL_SIZE * INNER_SIZE] = CVT_ACCUM2FLOAT(sum_db);
}

template <typename TI, typename TO>
__device__ void
layernormbwdreducesum(const TI* __restrict__ workspace, TO* __restrict__ dw, TO* __restrict__ db)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(gid >= INNER_SIZE)
        return;

    if(dw || db)
    {
        FLOAT_ACCUM sum_dw = 0;
        FLOAT_ACCUM sum_db = 0;

        for(uint64_t i = 0; i < PARALLEL_SIZE; ++i)
        {
            uint64_t input_idx = i * INNER_SIZE + gid;
            sum_dw += CVT_FLOAT2ACCUM(workspace[input_idx]);
            sum_db += CVT_FLOAT2ACCUM(workspace[input_idx + PARALLEL_SIZE * INNER_SIZE]);
        }

        if(dw)
        {
            dw[gid] = CVT_ACCUM2FLOAT(sum_dw);
        }
        if(db)
        {
            db[gid] = CVT_ACCUM2FLOAT(sum_db);
        }
    }
}

extern "C" __global__ void LayernormFwd(const INPUT_TYPE* __restrict__ x,
                                        const INPUT_TYPE* __restrict__ weight,
                                        const INPUT_TYPE* __restrict__ bias,
                                        OUTPUT_TYPE* __restrict__ y,
                                        OUTPUT_TYPE* __restrict__ mean,
                                        OUTPUT_TYPE* __restrict__ rstd,
                                        const float eps,
                                        const int32_t mode)
{
    // instantiate the kernel
    layernormfwd<INPUT_TYPE, OUTPUT_TYPE>(x, weight, bias, y, mean, rstd, eps, mode);
}

extern "C" __global__ void LayernormBwd(const INPUT_TYPE* __restrict__ dy,
                                        const INPUT_TYPE* __restrict__ x,
                                        const INPUT_TYPE* __restrict__ weight,
                                        const INPUT_TYPE* __restrict__ mean,
                                        const INPUT_TYPE* __restrict__ rstd,
                                        OUTPUT_TYPE* __restrict__ dx,
                                        const int32_t mode)
{
    // instantiate the kernel
    layernormbwd<INPUT_TYPE, OUTPUT_TYPE>(dy, x, weight, mean, rstd, dx, mode);
}

extern "C" __global__ void LayernormBwdWeightBias(const INPUT_TYPE* __restrict__ dy,
                                                  const INPUT_TYPE* __restrict__ x,
                                                  const INPUT_TYPE* __restrict__ mean,
                                                  const INPUT_TYPE* __restrict__ rstd,
                                                  OUTPUT_TYPE* __restrict__ dw,
                                                  OUTPUT_TYPE* __restrict__ db)
{
    layernormbwdweightbias<INPUT_TYPE, OUTPUT_TYPE>(dy, x, mean, rstd, dw, db);
}

extern "C" __global__ void LayernormBwdWeightBiasParallel(const INPUT_TYPE* __restrict__ dy,
                                                          const INPUT_TYPE* __restrict__ x,
                                                          const INPUT_TYPE* __restrict__ mean,
                                                          const INPUT_TYPE* __restrict__ rstd,
                                                          OUTPUT_TYPE* __restrict__ workspace)
{
    layernormbwdweightbiasparallel<INPUT_TYPE, OUTPUT_TYPE>(dy, x, mean, rstd, workspace);
}

extern "C" __global__ void LayernormBwdReduceSum(const INPUT_TYPE* __restrict__ workspace,
                                                 OUTPUT_TYPE* __restrict__ dw,
                                                 OUTPUT_TYPE* __restrict__ db)
{
    layernormbwdreducesum<INPUT_TYPE, OUTPUT_TYPE>(workspace, dw, db);
}
