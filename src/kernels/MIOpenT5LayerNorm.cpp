// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "miopen_cstdint.hpp"
#include "float_types.h"

template <typename TI, typename TO>
__device__ void t5layernormfwdcontiguous(const TI* __restrict__ x,
                                         const TI* __restrict__ weight,
                                         TO* __restrict__ y,
                                         TO* __restrict__ rstd,
                                         float eps,
                                         uint64_t inner_size,
                                         int32_t mode)
{
    const uint64_t gid = blockIdx.x;
    const uint64_t lid = threadIdx.x;

    FLOAT_ACCUM pvar = static_cast<FLOAT_ACCUM>(0);
    __shared__ FLOAT_ACCUM ltmp[LOCAL_SIZE];

    // reduce sum
    for(uint64_t i = lid; i < inner_size; i += LOCAL_SIZE)
    {
        size_t x_idx = gid * inner_size + i;

        FLOAT_ACCUM tmp = CVT_FLOAT2ACCUM(x[x_idx]);
        pvar += tmp * tmp;
    }

    ltmp[lid] = pvar;
    __syncthreads();
    for(uint32_t i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            ltmp[lid] += ltmp[lid + i];
        }
        __syncthreads();
    }
    pvar              = ltmp[0] / inner_size;
    FLOAT_ACCUM prstd = rsqrt(pvar + FLOAT_ACCUM(eps));

    if(lid == 0)
    {
        if(rstd)
            rstd[gid] = CVT_ACCUM2FLOAT(prstd);
    }

    // forward calculation
    for(uint64_t i = lid; i < inner_size; i += LOCAL_SIZE)
    {
        size_t idx = gid * inner_size + i;

        FLOAT_ACCUM pweight;

        pweight = (mode == MIOPEN_ELEMENTWISE_AFFINE_T5) ? CVT_FP32_2ACCUM(1.0f)
                                                         : CVT_FLOAT2ACCUM(weight[i]);

        FLOAT_ACCUM val = (CVT_FLOAT2ACCUM(x[idx])) * prstd * pweight;
        y[idx]          = CVT_ACCUM2FLOAT(val);
    }
}

template <typename TI, typename TO>
__device__ void t5layernormbwdcontiguous(const TI* __restrict__ dy,
                                         const TI* __restrict__ x,
                                         const TI* __restrict__ weight,
                                         const TI* __restrict__ rstd,
                                         TO* __restrict__ dx,
                                         uint64_t inner_size,
                                         int32_t mode)
{
    const uint64_t gid = blockIdx.x;
    const uint64_t lid = threadIdx.x;

    __shared__ FLOAT_ACCUM ltmp[LOCAL_SIZE];

    // reduce sum
    FLOAT_ACCUM sum = 0;

    for(uint64_t i = lid; i < inner_size; i += LOCAL_SIZE)
    {
        size_t x_idx = gid * inner_size + i;

        FLOAT_ACCUM pweight = (mode == MIOPEN_ELEMENTWISE_AFFINE_T5) ? CVT_FP32_2ACCUM(1.0f)
                                                                     : CVT_FLOAT2ACCUM(weight[i]);

        FLOAT_ACCUM pdy = dy ? CVT_FLOAT2ACCUM(dy[x_idx]) : 0;
        sum += pdy * CVT_FLOAT2ACCUM(x[x_idx]) * pweight;
    }

    ltmp[lid] = sum;
    __syncthreads();
    for(uint32_t i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            ltmp[lid] += ltmp[lid + i];
        }
        __syncthreads();
    }

    FLOAT_ACCUM ds    = ltmp[0];
    FLOAT_ACCUM s     = 1.0f / inner_size;
    FLOAT_ACCUM prstd = CVT_FLOAT2ACCUM(rstd[gid]);
    FLOAT_ACCUM a     = ds * prstd * prstd * prstd * s;

    for(uint64_t i = lid; i < inner_size; i += LOCAL_SIZE)
    {
        size_t idx = gid * inner_size + i;

        FLOAT_ACCUM pweight = (mode == MIOPEN_ELEMENTWISE_AFFINE_T5) ? CVT_FP32_2ACCUM(1.0f)
                                                                     : CVT_FLOAT2ACCUM(weight[i]);
        FLOAT_ACCUM pdy     = dy ? CVT_FLOAT2ACCUM(dy[idx]) : 0;

        FLOAT_ACCUM val = prstd * pdy * pweight - a * CVT_FLOAT2ACCUM(x[idx]);
        dx[idx]         = CVT_ACCUM2FLOAT(val);
    }
}

template <typename TI, typename TO>
__device__ void t5layernormbwdweightcontiguous(const TI* __restrict__ dy,
                                               const TI* __restrict__ x,
                                               const TI* __restrict__ rstd,
                                               TO* __restrict__ dw,
                                               uint64_t outer_size,
                                               uint64_t inner_size)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    FLOAT_ACCUM sum = static_cast<FLOAT_ACCUM>(0);
    for(uint64_t i = 0; i < outer_size; ++i)
    {
        uint64_t input_idx = i * inner_size + gid;

        FLOAT_ACCUM prstd = CVT_FLOAT2ACCUM(rstd[i]);
        FLOAT_ACCUM pdy   = dy ? CVT_FLOAT2ACCUM(dy[input_idx]) : 0;

        sum += pdy * CVT_FLOAT2ACCUM(x[input_idx]) * prstd;
    }

    if(dw)
    {
        dw[gid] = CVT_ACCUM2FLOAT(sum);
    }
}

template <typename TI, typename TO>
__device__ void t5layernormbwdweightcontiguousparallel(const TI* __restrict__ dy,
                                                       const TI* __restrict__ x,
                                                       const TI* __restrict__ rstd,
                                                       TO* __restrict__ workspace,
                                                       uint64_t outer_size,
                                                       uint64_t inner_size,
                                                       uint64_t parallel_size)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(gid >= inner_size * parallel_size)
        return;

    uint64_t pid = gid / inner_size;

    uint64_t input_idx = gid;

    FLOAT_ACCUM sum = static_cast<FLOAT_ACCUM>(0);

    if(dy)
    {
        for(uint64_t i = pid; i < outer_size; i += parallel_size)
        {
            FLOAT_ACCUM prstd = CVT_FLOAT2ACCUM(rstd[i]);
            FLOAT_ACCUM pdy   = CVT_FLOAT2ACCUM(dy[input_idx]);

            sum += pdy * CVT_FLOAT2ACCUM(x[input_idx]) * prstd;
            input_idx += inner_size * parallel_size;
        }
    }

    workspace[gid] = CVT_ACCUM2FLOAT(sum);
}

template <typename TI, typename TO>
__device__ void t5layernormbwdcontiguousreduceSum(const TI* __restrict__ workspace,
                                                  TO* __restrict__ dw,
                                                  uint64_t inner_size,
                                                  uint64_t parallel_size)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(gid >= inner_size)
        return;

    FLOAT_ACCUM sum = static_cast<FLOAT_ACCUM>(0);
    for(uint64_t i = 0; i < parallel_size; ++i)
    {
        uint64_t input_idx = i * inner_size + gid;
        sum += CVT_FLOAT2ACCUM(workspace[input_idx]);
    }

    if(dw)
    {
        dw[gid] = CVT_ACCUM2FLOAT(sum);
    }
}

extern "C" __global__ void T5LayernormFwdContiguous(const INPUT_TYPE* __restrict__ x,
                                                    const INPUT_TYPE* __restrict__ weight,
                                                    OUTPUT_TYPE* __restrict__ y,
                                                    OUTPUT_TYPE* __restrict__ rstd,
                                                    float eps,
                                                    uint64_t inner_size,
                                                    int32_t mode)
{
    // instantiate the kernel
    t5layernormfwdcontiguous<INPUT_TYPE, OUTPUT_TYPE>(x, weight, y, rstd, eps, inner_size, mode);
}

extern "C" __global__ void T5LayernormBwdContiguous(const INPUT_TYPE* __restrict__ dy,
                                                    const INPUT_TYPE* __restrict__ x,
                                                    const INPUT_TYPE* __restrict__ weight,
                                                    const INPUT_TYPE* __restrict__ rstd,
                                                    OUTPUT_TYPE* __restrict__ dx,
                                                    uint64_t inner_size,
                                                    int32_t mode)
{
    // instantiate the kernel
    t5layernormbwdcontiguous<INPUT_TYPE, OUTPUT_TYPE>(dy, x, weight, rstd, dx, inner_size, mode);
}

extern "C" __global__ void T5LayernormBwdWeightContiguous(const INPUT_TYPE* __restrict__ dy,
                                                          const INPUT_TYPE* __restrict__ x,
                                                          const INPUT_TYPE* __restrict__ rstd,
                                                          OUTPUT_TYPE* __restrict__ dw,
                                                          uint64_t outer_size,
                                                          uint64_t inner_size)
{
    // instantiate the kernel
    t5layernormbwdweightcontiguous<INPUT_TYPE, OUTPUT_TYPE>(
        dy, x, rstd, dw, outer_size, inner_size);
}

extern "C" __global__ void
T5LayernormBwdWeightContiguousParallel(const INPUT_TYPE* __restrict__ dy,
                                       const INPUT_TYPE* __restrict__ x,
                                       const INPUT_TYPE* __restrict__ rstd,
                                       OUTPUT_TYPE* __restrict__ workspace,
                                       uint64_t outer_size,
                                       uint64_t inner_size,
                                       uint64_t parallel_size)
{
    // instantiate the kernel
    t5layernormbwdweightcontiguousparallel<INPUT_TYPE, OUTPUT_TYPE>(
        dy, x, rstd, workspace, outer_size, inner_size, parallel_size);
}

extern "C" __global__ void
T5LayernormBwdContiguousReduceSum(const INPUT_TYPE* __restrict__ workspace,
                                  OUTPUT_TYPE* __restrict__ dw,
                                  uint64_t inner_size,
                                  uint64_t parallel_size)
{
    // instantiate the kernel
    t5layernormbwdcontiguousreduceSum<INPUT_TYPE, OUTPUT_TYPE>(
        workspace, dw, inner_size, parallel_size);
}
