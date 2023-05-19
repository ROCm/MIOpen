#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "hip_float8.h"

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

    bool hasZero;
    bool hasNan;
    bool hasInf;
};

__device__ void thread_redux(Numerics* stats, size_t wid)
{
    const auto lid = threadIdx.x;
    if(lid < wid)
    {
        stats[lid].sum += stats[lid + wid].sum;
        stats[lid].absSum += stats[lid + wid].absSum;
        stats[lid].min = std::fmin(stats[lid].min, stats[lid + wid].min);
        stats[lid].max = std::fmax(stats[lid].max, stats[lid + wid].max);
    }
}

__device__ void atomicMax(float* __restrict__ target, float val)
{
    float current, expected, next;

    current = *target;
    do
    {
        expected = current;
        next     = std::fmax(current, val);
        if(next == current)
            break;
        const auto i_expected = *(reinterpret_cast<unsigned int*>(&expected));
        const auto i_next     = *(reinterpret_cast<unsigned int*>(&next));
        auto i_current = atomicCAS(reinterpret_cast<unsigned int*>(target), i_expected, i_next);
        current        = *(reinterpret_cast<float*>(&i_current));
    } while(current != expected);
}

__device__ void atomicMin(float* __restrict__ target, float val)
{
    float current, expected, next;

    current = *target;
    do
    {
        expected = current;
        next     = std::fmin(current, val);
        if(next == current)
            break;
        const auto i_expected = *(reinterpret_cast<unsigned int*>(&expected));
        const auto i_next     = *(reinterpret_cast<unsigned int*>(&next));
        auto i_current = atomicCAS(reinterpret_cast<unsigned int*>(target), i_expected, i_next);
        current        = *(reinterpret_cast<float*>(&i_current));
    } while(current != expected);
}

template <typename T, typename U>
__global__ void check_numerics(T* C_d, size_t sz, CheckNumericsResult* abnormal, bool computeStats)
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
        minV = std::min(minV, val);
        maxV = std::max(maxV, val);
        if(abs_val <= static_cast<U>(0.0f))
            abnormal->hasZero = true;
        if(std::isnan(static_cast<U>(val)))
            abnormal->hasNan = true;
        if(std::isinf(static_cast<U>(val)))
            abnormal->hasInf = true;
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
    check_numerics<half, float>(reinterpret_cast<const half*>(C_d), sz, abnormal, computeStats);
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
