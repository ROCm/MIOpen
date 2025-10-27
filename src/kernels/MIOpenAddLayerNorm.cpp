// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "miopen_cstdint.hpp"
#include "float_types.h"

template <typename TI, typename TO>
__device__ void addlayernormfwdcontiguous(const TI* __restrict__ x,
                                          const TI* __restrict__ x2,
                                          const TI* __restrict__ weight,
                                          const TI* __restrict__ bias,
                                          TO* __restrict__ y,
                                          TO* __restrict__ mean,
                                          TO* __restrict__ rstd,
                                          float eps,
                                          uint64_t inner_size,
                                          int32_t mode)
{
    const uint64_t gid = blockIdx.x;
    const uint64_t lid = threadIdx.x;

    FLOAT_ACCUM pmean = static_cast<FLOAT_ACCUM>(0);
    FLOAT_ACCUM pvar  = static_cast<FLOAT_ACCUM>(0);
    __shared__ FLOAT_ACCUM ltmp1[LOCAL_SIZE];
    __shared__ FLOAT_ACCUM ltmp2[LOCAL_SIZE];

    // reduce sum for mean and var
    for(uint64_t i = lid; i < inner_size; i += LOCAL_SIZE)
    {
        size_t x_idx = gid * inner_size + i;

        FLOAT_ACCUM tmp = CVT_FLOAT2ACCUM(x[x_idx]) + CVT_FLOAT2ACCUM(x2[x_idx]);
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

        pweight = (mode == MIOPEN_ELEMENTWISE_AFFINE_FUSED_ADD) ? CVT_FP32_2ACCUM(1.0f)
                                                                : CVT_FLOAT2ACCUM(weight[i]);
        pbias   = (mode == MIOPEN_ELEMENTWISE_AFFINE_FUSED_ADD) ? static_cast<FLOAT>(0)
                                                                : CVT_FLOAT2ACCUM(bias[i]);

        FLOAT_ACCUM val =
            (CVT_FLOAT2ACCUM(x[idx]) + CVT_FLOAT2ACCUM(x2[idx]) - pmean) * prstd * pweight + pbias;
        y[idx] = CVT_ACCUM2FLOAT(val);
    }
}

extern "C" __global__ void AddLayernormFwdContiguous(const INPUT_TYPE* __restrict__ x,
                                                     const INPUT_TYPE* __restrict__ x2,
                                                     const INPUT_TYPE* __restrict__ weight,
                                                     const INPUT_TYPE* __restrict__ bias,
                                                     OUTPUT_TYPE* __restrict__ y,
                                                     OUTPUT_TYPE* __restrict__ mean,
                                                     OUTPUT_TYPE* __restrict__ rstd,
                                                     float eps,
                                                     uint64_t inner_size,
                                                     int32_t mode)
{
    // instantiate the kernel
    addlayernormfwdcontiguous<INPUT_TYPE, OUTPUT_TYPE>(
        x, x2, weight, bias, y, mean, rstd, eps, inner_size, mode);
}