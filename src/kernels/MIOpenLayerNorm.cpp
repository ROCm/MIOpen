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
#ifdef MIOPEN_BETA_API

#include "float_types.h"

//#if MIOPEN_USE_BFP16 == 1
//#undef FLOAT
//#define FLOAT hip_bfloat16
//#endif

extern "C" __global__ void LayernormFwdContiguous(const FLOAT* __restrict__ x,
                                                  FLOAT* __restrict__ y,
                                                  const FLOAT* __restrict__ weight,
                                                  const FLOAT* __restrict__ bias,
                                                  FLOAT* __restrict__ mean,
                                                  FLOAT* __restrict__ rstd,
                                                  float eps,
                                                  uint64_t inner_size,
                                                  bool mode)
{
    /*
     * Each group works on a single channel.
     * Example)
     * x dim = {N, C, L}, normalized shape = {C, L}
     * outer_size = N, inner_size = C * L
     *
     * Example2)
     * x dim = {N, C, L}, normalized shape = {L}
     * outer_size = N * C, inner_size = L
     *
     * => gws = {outer_size * LOCAL_SIZE}, lws = {LOCAL_SIZE}
     */

    /*
     * Reduction to calculate mean and rstd
     */

    const uint64_t gid = blockIdx.x;
    const uint64_t lid = threadIdx.x;

    FLOAT pmean = FLOAT(0);
    FLOAT pvar  = FLOAT(0);
    __shared__ FLOAT ltmp1[LOCAL_SIZE];
    __shared__ FLOAT ltmp2[LOCAL_SIZE];

    // reduce sum for mean and var
    for(uint64_t i = lid; i < inner_size; i += LOCAL_SIZE)
    {
        uint64_t x_idx = gid * inner_size + i;

        FLOAT tmp = x[x_idx];
        pmean += tmp;
        pvar += tmp * tmp;
    }

    ltmp1[lid] = pmean;
    ltmp2[lid] = pvar;
    __syncthreads();
    for(uint64_t i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            ltmp1[lid] += ltmp1[lid + i];
            ltmp2[lid] += ltmp2[lid + i];
        }
        __syncthreads();
    }
    pmean       = FLOAT(ltmp1[0] / FLOAT(inner_size));
    pvar        = FLOAT(ltmp2[0] / FLOAT(inner_size)) - pmean * pmean;
    FLOAT prstd = FLOAT(rsqrt(pvar + FLOAT(eps)));

    if(lid == 0)
    {
        if(mean)
            mean[gid] = pmean;
        if(rstd)
            rstd[gid] = prstd;
    }

    // forward calculation
    for(uint64_t i = lid; i < inner_size; i += LOCAL_SIZE)
    {
        uint64_t idx = gid * inner_size + i;

        FLOAT pweight;
        FLOAT pbias;

        pweight = mode ? FLOAT(1) : weight[i];
        pbias   = mode ? FLOAT(0) : bias[i];

        FLOAT val = (x[idx] - pmean) * prstd * pweight + pbias;
        y[idx]    = val;
    }
}

extern "C" __global__ void LayerNormBwdContiguous(const FLOAT* __restrict__ x,
                                                  const FLOAT* __restrict__ dy,
                                                  const FLOAT* __restrict__ weight,
                                                  const FLOAT* __restrict__ mean,
                                                  const FLOAT* __restrict__ rstd,
                                                  FLOAT* __restrict__ dx,
                                                  uint64_t inner_size,
                                                  bool mode)
{

    const uint64_t gid = blockIdx.x;
    const uint64_t lid = threadIdx.x;

    // reduce sum for sum1, sum2
    FLOAT sum1 = FLOAT(0);
    FLOAT sum2 = FLOAT(0);

    __shared__ FLOAT ltmp1[LOCAL_SIZE];
    __shared__ FLOAT ltmp2[LOCAL_SIZE];

    for(size_t i = lid; i < inner_size; i += LOCAL_SIZE)
    {
        size_t idx = gid * inner_size + i;

        FLOAT weight_v = mode ? FLOAT(1) : weight[i];
        FLOAT dy_v     = dy[idx];

        sum1 += dy_v * x[idx] * weight_v;
        sum2 += dy_v * weight_v;
    }

    ltmp1[lid] = sum1;
    ltmp2[lid] = sum2;
    __syncthreads();
    for(size_t i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            ltmp1[lid] += ltmp1[lid + i];
            ltmp2[lid] += ltmp2[lid + i];
        }
        __syncthreads();
    }

    FLOAT ds = ltmp1[0];
    FLOAT db = ltmp2[0];

    FLOAT s = FLOAT(1.0f / inner_size);

    FLOAT mean_v = mean[gid];
    FLOAT rstd_v = rstd[gid];

    FLOAT a  = (db * mean_v - ds) * rstd_v * rstd_v * rstd_v * s;
    FLOAT c2 = -(a * mean_v + db * rstd_v * s);

    for(size_t i = lid; i < inner_size; i += LOCAL_SIZE)
    {
        size_t idx = gid * inner_size + i;

        FLOAT weight_v = mode ? FLOAT(1) : weight[i];
        FLOAT dy_v     = dy[idx];

        FLOAT val = rstd_v * dy_v * weight_v + a * x[idx] + c2;
        dx[idx]   = val;
    }
}

extern "C" __global__ void LayernormBwdWeightBiasContiguous(const FLOAT* __restrict__ x,
                                                            const FLOAT* __restrict__ dy,
                                                            const FLOAT* __restrict__ mean,
                                                            const FLOAT* __restrict__ rstd,
                                                            FLOAT* __restrict__ dw,
                                                            FLOAT* __restrict__ db,
                                                            uint64_t outer_size,
                                                            uint64_t inner_size)
{

    const uint64_t gid = blockIdx.x;
    const uint64_t lid = threadIdx.x;

    __shared__ FLOAT ltmp1[LOCAL_SIZE];
    __shared__ FLOAT ltmp2[LOCAL_SIZE];

    FLOAT sum1 = FLOAT(0);
    FLOAT sum2 = FLOAT(0);

    for(size_t i = lid; i < outer_size; i += LOCAL_SIZE)
    {
        size_t idx = i * (inner_size) + gid;

        FLOAT dy_v = dy[idx];

        sum1 += dy_v * (x[idx] - mean[i]) * rstd[i];
        sum2 += dy_v;
    }

    ltmp1[lid] = sum1;
    ltmp2[lid] = sum2;
    __syncthreads();
    for(size_t i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            ltmp1[lid] += ltmp1[lid + i];
            ltmp2[lid] += ltmp2[lid + i];
        }
        __syncthreads();
    }

    sum1 = ltmp1[0];
    sum2 = ltmp2[0];

    if(lid == 0)
    {
        if(dw)
        {
            dw[gid] = sum1;
        }
        if(db)
        {
            db[gid] = sum2;
        }
    }
}
#endif
