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

template <typename TI, typename TO>
__device__ void groupnormfwdcontiguous(const TI* __restrict__ x,
                                       const TI* __restrict__ weight,
                                       const TI* __restrict__ bias,
                                       TO* __restrict__ y,
                                       TO* __restrict__ mean,
                                       TO* __restrict__ rstd,
                                       float eps,
                                       uint64_t num_groups,
                                       uint64_t num_channels,
                                       uint64_t numel_per_channel,
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

    const size_t inner_size = numel_per_channel * num_channels / num_groups;

    FLOAT_ACCUM pmean = static_cast<FLOAT_ACCUM>(0);
    FLOAT_ACCUM pvar  = static_cast<FLOAT_ACCUM>(0);
    __shared__ FLOAT_ACCUM ltmp1[LOCAL_SIZE];
    __shared__ FLOAT_ACCUM ltmp2[LOCAL_SIZE];

    // reduce sum for mean and var
    for(uint64_t i = lid; i < inner_size; i += LOCAL_SIZE)
    {
        size_t x_idx = gid * inner_size + i;

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
        size_t idx = gid * inner_size + i;

        FLOAT_ACCUM pweight;
        FLOAT_ACCUM pbias;

        size_t c = mode ? (idx / numel_per_channel) % num_channels : 0;
        pweight  = mode ? CVT_FLOAT2ACCUM(weight[c]) : CVT_FP32_2ACCUM(1.0f);
        pbias    = mode ? CVT_FLOAT2ACCUM(bias[c]) : static_cast<FLOAT>(0);

        FLOAT_ACCUM val = (CVT_FLOAT2ACCUM(x[idx]) - pmean) * prstd * pweight + pbias;
        y[idx]          = CVT_ACCUM2FLOAT(val);
    }
}

extern "C" __global__ void GroupNormFwdContiguous(const INPUT_TYPE* __restrict__ x,
                                                  const INPUT_TYPE* __restrict__ weight,
                                                  const INPUT_TYPE* __restrict__ bias,
                                                  OUTPUT_TYPE* __restrict__ y,
                                                  OUTPUT_TYPE* __restrict__ mean,
                                                  OUTPUT_TYPE* __restrict__ rstd,
                                                  float eps,
                                                  uint64_t num_groups,
                                                  uint64_t num_channels,
                                                  uint64_t numel_per_channel,
                                                  bool mode)
{
    // instantiate the kernel
    groupnormfwdcontiguous<INPUT_TYPE, OUTPUT_TYPE>(
        x, weight, bias, y, mean, rstd, eps, num_groups, num_channels, numel_per_channel, mode);
}
