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

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

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

    FLOAT_ACCUM pmean = CVT_FLOAT2ACCUM(0);
    FLOAT_ACCUM pvar  = CVT_FLOAT2ACCUM(0);
    __shared__ FLOAT_ACCUM ltmp1[LOCAL_SIZE];
    __shared__ FLOAT_ACCUM ltmp2[LOCAL_SIZE];

    // reduce sum for mean and var
    for(uint64_t i = lid; i < inner_size; i += LOCAL_SIZE)
    {
        uint64_t x_idx = gid * inner_size + i;

        FLOAT_ACCUM tmp = CVT_FLOAT2ACCUM(x[x_idx]);
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
    pmean             = ltmp1[0] / inner_size;
    pvar              = ltmp2[0] / inner_size - pmean * pmean;
    FLOAT_ACCUM prstd = rsqrt(pvar + FLOAT_ACCUM(eps));

    if(lid == 0)
    {
        if(mean)
            mean[gid] = CVT_ACCUM2FLOAT(pmean);
        if(rstd)
            rstd[gid] = CVT_ACCUM2FLOAT(prstd);
    }

    // forward calculation
    for(uint64_t i = lid; i < inner_size; i += LOCAL_SIZE)
    {
        uint64_t idx = gid * inner_size + i;

        FLOAT_ACCUM pweight;
        FLOAT_ACCUM pbias;

        pweight = mode ? CVT_FLOAT2ACCUM(1) : CVT_FLOAT2ACCUM(weight[i]);
        pbias   = mode ? CVT_FLOAT2ACCUM(0) : CVT_FLOAT2ACCUM(bias[i]);

        FLOAT_ACCUM val = (CVT_FLOAT2ACCUM(x[idx]) - pmean) * prstd * pweight + pbias;
        y[idx]          = CVT_ACCUM2FLOAT(val);
    }
}
#endif
