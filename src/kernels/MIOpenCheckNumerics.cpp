/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023  Advanced Micro Devices, Inc.
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
#include <hip/hip_bfloat16.h>
#endif

#define MIOPEN_ENABLE_F8_DEVICE_CODE 1
#include "hip_float8.hpp"

#include "miopen_limits.hpp"

struct Numerics
{
    float sum;
    float absSum;
    float min;
    float max;
};

struct CheckNumericsResult
{
    Numerics n;

    int hasZero;
    int hasNan;
    int hasInf;
};

__device__ void thread_redux(Numerics* stats, size_t wid)
{
    const auto lid = threadIdx.x;
    if(lid < wid)
    {
        stats[lid].sum += stats[lid + wid].sum;
        stats[lid].absSum += stats[lid + wid].absSum;
        stats[lid].min = fmin(stats[lid].min, stats[lid + wid].min);
        stats[lid].max = fmax(stats[lid].max, stats[lid + wid].max);
    }
}

template <typename T, typename U>
__device__ void
check_numerics(const T* C_d, size_t sz, CheckNumericsResult* abnormal, bool computeStats)
{
    __shared__ Numerics stats[256];
    U sum    = 0;
    U absSum = 0;
    T minV   = std::numeric_limits<T>::max();
    T maxV   = std::numeric_limits<T>::min();

    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    for(size_t i = offset; i < sz; i += stride)
    {
        T val = C_d[i];
        sum += static_cast<U>(val);
        const auto abs_val = fabs(static_cast<U>(val));
        absSum += abs_val;
        minV = min(minV, val);
        maxV = max(maxV, val);
        if(abs_val <= static_cast<U>(0.0f))
            abnormal->hasZero = 1;
        if(isnan(static_cast<U>(val)))
            abnormal->hasNan = 1;
        if(isinf(static_cast<U>(val)))
            abnormal->hasInf = 1;
    }
    if(computeStats)
    {
        stats[threadIdx.x].sum    = static_cast<float>(sum);
        stats[threadIdx.x].absSum = static_cast<float>(absSum);
        stats[threadIdx.x].min    = static_cast<float>(minV);
        stats[threadIdx.x].max    = static_cast<float>(maxV);
        __syncthreads();
        for(int idx = 128; idx > 0; idx = idx >> 1)
        {
            thread_redux(stats, idx);
            __syncthreads();
        }
        if(threadIdx.x == 0)
        {
            atomicAdd(&abnormal->n.sum, stats[0].sum);
            atomicAdd(&abnormal->n.absSum, stats[0].absSum);
            atomicMin(&abnormal->n.min, stats[0].min);
            atomicMax(&abnormal->n.max, stats[0].max);
        }
    }
}

extern "C" __global__ void check_numerics_fp32(const void* __restrict__ C_d,
                                               size_t sz,
                                               CheckNumericsResult* __restrict__ abnormal,
                                               bool computeStats)
{
    check_numerics<float, float>(reinterpret_cast<const float*>(C_d), sz, abnormal, computeStats);
}

extern "C" __global__ void check_numerics_fp16(const void* __restrict__ C_d,
                                               size_t sz,
                                               CheckNumericsResult* __restrict__ abnormal,
                                               bool computeStats)
{
    check_numerics<_Float16, float>(
        reinterpret_cast<const _Float16*>(C_d), sz, abnormal, computeStats);
}

extern "C" __global__ void check_numerics_bf16(const void* __restrict__ C_d,
                                               size_t sz,
                                               CheckNumericsResult* __restrict__ abnormal,
                                               bool computeStats)
{
    check_numerics<hip_bfloat16, float>(
        reinterpret_cast<const hip_bfloat16*>(C_d), sz, abnormal, computeStats);
}

extern "C" __global__ void check_numerics_fp8(const void* __restrict__ C_d,
                                              size_t sz,
                                              CheckNumericsResult* __restrict__ abnormal,
                                              bool computeStats)
{
    check_numerics<miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>, float>(
        reinterpret_cast<const miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>*>(C_d),
        sz,
        abnormal,
        computeStats);
}

extern "C" __global__ void check_numerics_bf8(const void* __restrict__ C_d,
                                              size_t sz,
                                              CheckNumericsResult* __restrict__ abnormal,
                                              bool computeStats)
{
    check_numerics<miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>, float>(
        reinterpret_cast<const miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>*>(C_d),
        sz,
        abnormal,
        computeStats);
}
