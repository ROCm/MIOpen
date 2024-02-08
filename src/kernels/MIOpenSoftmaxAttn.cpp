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
#include <hip/hip_runtime.h>

#include "miopen_limits.hpp"
#include "miopen_cstdint.hpp"

#ifndef THREADS
#define THREADS 64
#endif

namespace {
template <typename DST, typename SRC>
constexpr DST bitcast(SRC val)
{
    static_assert(sizeof(DST) == sizeof(SRC));
    DST tmp;
    /// TODO: wait for C++20 and use std::bit_cast
    /// until then it's the true way of type puning
    __builtin_memcpy(&tmp, &val, sizeof(DST));
    /// unions can work for C99 and later but not for C++,
    /// though compilers widely and unoficially support it
    return tmp;
}

constexpr float max_op(float a, float b) { return a > b ? a : b; };
constexpr float max_abs_op(float a, float b) { return max_op(abs(a), abs(b)); };
constexpr float plus_op(float a, float b) { return a + b; };
constexpr float identety_op(float a) { return a; };
} // namespace

template <uint32_t WARP_SIZE, typename Op, uint32_t SWIZZLE_SIZE = WARP_SIZE>
__device__ float reductionFullWarp(float reduced_val, uint32_t laneId, Op op)
{
    static_assert(WARP_SIZE != 0, "WARP_SIZEmust not be 0");
    static_assert((SWIZZLE_SIZE & (SWIZZLE_SIZE - 1)) == 0,
                  "WARP_SIZE and SWIZZLE must be a power of 2");

    if constexpr(SWIZZLE_SIZE == 1)
        return reduced_val;

    reduced_val = reductionFullWarp<WARP_SIZE, Op, (SWIZZLE_SIZE >> 1)>(reduced_val, laneId, op);

    constexpr uint32_t warp_msk = (WARP_SIZE - 1);

    float tmp;
    if constexpr(SWIZZLE_SIZE >= 64)
    {
        // swizzle can handle only 32 lanes, switching to bpermute
        uint32_t idx = laneId ^ (SWIZZLE_SIZE >> 1);

        idx = idx >= ((laneId + WARP_SIZE) & ~warp_msk) ? laneId : idx;
        int itmp =
            __builtin_amdgcn_ds_bpermute(static_cast<int>(idx << 2), bitcast<int>(reduced_val));
        tmp = bitcast<float>(itmp);
    }
    else
    {
        // butterfly reduction based on __shfl_xor
        // swizzle <xor_mask[14:10], or_mask[9:5], and_mask[4:0]>()
        constexpr uint32_t xor_off = 10;
        constexpr uint32_t or_off  = 5;
        constexpr uint32_t and_off = 0;

        constexpr uint32_t field_msk = 0x1f;

        constexpr uint32_t and_msk = warp_msk & field_msk;
        constexpr uint32_t or_msk  = 0;
        constexpr uint32_t xor_msk = (SWIZZLE_SIZE >> 1) & field_msk;

        constexpr uint32_t swizzle_op =
            (xor_msk << xor_off) | (or_msk << or_off) | (and_msk << and_off);

        tmp = __hip_ds_swizzlef_N<swizzle_op>(reduced_val);
    }

    return op(tmp, reduced_val);
};

template <uint32_t NumWarps, typename Op>
__device__ float
reductionBlock(float local_val, Op op, uint32_t lid, uint32_t laneId, uint32_t warpId)
{
    __shared__ float reduction_tmp[NumWarps];

    float reduced_val = reductionFullWarp<warpSize>(local_val, laneId, op);
    if(laneId == 0)
        reduction_tmp[warpId] = reduced_val;
    __syncthreads();

    if(lid < NumWarps)
    {
        reduced_val = reductionFullWarp<NumWarps>(reduction_tmp[lid], laneId, op);
        if(lid == 0)
        {
            reduction_tmp[0] = reduced_val;
        }
    }
    __syncthreads();

    return reduction_tmp[0];
};

template <uint32_t NumWarps, typename ReductionOp, typename ElementOp>
__device__ float reductionCommon(const float* __restrict__ line,
                                 const float init_value,
                                 const uint32_t seq_len,
                                 ReductionOp&& op,
                                 ElementOp&& eop,
                                 uint32_t lid,
                                 uint32_t laneId,
                                 uint32_t warpId)
{
    float reduced_val = (lid < seq_len) ? eop(line[lid]) : init_value;

    for(uint32_t loop_lid = lid + blockDim.x; loop_lid < seq_len; loop_lid += blockDim.x)
        reduced_val = op(eop(line[loop_lid]), reduced_val);

    return reductionBlock<NumWarps>(reduced_val, op, lid, laneId, warpId);
};

extern "C" __global__ void MaxAbsReductionWarp(float* __restrict__ val, uint32_t len)
{
    const uint32_t lid    = threadIdx.x;
    const uint32_t laneId = threadIdx.x % warpSize;

    float local_val = (lid < len) ? (*val) : 0;
    float r_max     = reductionFullWarp<warpSize>(local_val, laneId, max_abs_op);

    if(lid == 0)
        val[0] = r_max;
}

extern "C" __global__ void MaxAbsReductionBlock(float* __restrict__ val, uint32_t len)
{
    constexpr uint32_t NumWarps = THREADS / warpSize;
    const uint32_t lid          = threadIdx.x;
    const uint32_t laneId       = threadIdx.x % warpSize;
    const uint32_t warpId       = threadIdx.x / warpSize;

    float local_val = (lid < len) ? (*val) : 0;
    float r_max     = reductionBlock<NumWarps>(local_val, max_abs_op, lid, laneId, warpId);

    if(lid == 0)
        val[0] = r_max;
}

extern "C" __global__ void MaxAbsReductionCommon(float* __restrict__ val, uint32_t len)
{
    constexpr uint32_t NumWarps = THREADS / warpSize;
    const uint32_t lid          = threadIdx.x;
    const uint32_t laneId       = threadIdx.x % warpSize;
    const uint32_t warpId       = threadIdx.x / warpSize;

    float r_max =
        reductionCommon<NumWarps>(val, 0, len, max_abs_op, identety_op, lid, laneId, warpId);

    if(lid == 0)
        val[0] = r_max;
}

extern "C" __global__ void SoftMaxWarp(const float* in,
                                       float* out,
                                       float* __restrict__ M,
                                       float* __restrict__ Z,
                                       float* __restrict__ AmaxWorkspace,
                                       const float* __restrict__ descale_Q,
                                       const float* __restrict__ descale_K,
                                       uint32_t seq_len,
                                       uint64_t nhs)
{
    const uint32_t NumWarps = THREADS / warpSize;
    const uint32_t lid      = threadIdx.x;
    const uint32_t laneId   = lid % warpSize;
    const uint32_t warpId   = lid / warpSize;
    const float descaler    = (descale_Q ? *descale_Q : 1.f) * (descale_K ? *descale_K : 1.f);
    const bool save_stats   = M && Z && laneId == 0;

    float r_Amax = 0;

    for(uint64_t gid = blockIdx.x * NumWarps + warpId; gid < nhs; gid += gridDim.x * NumWarps)
    {
        const float* line = in + gid * seq_len + laneId;
        float* res        = out + gid * seq_len + laneId;

        float local_val =
            (laneId < seq_len) ? (*line) * descaler : std::numeric_limits<float>::lowest();

        float r_max = reductionFullWarp<warpSize>(local_val, laneId, max_op);

        local_val = (laneId < seq_len) ? exp(local_val - r_max) : 0;

        float r_sum = 1.f / reductionFullWarp<warpSize>(local_val, laneId, plus_op);

        local_val = (laneId < seq_len) ? local_val * r_sum : 0;
        if(laneId < seq_len)
        {
            *res = local_val;
        }

        r_Amax = max_abs_op(r_Amax, local_val);
        if(save_stats)
        {
            M[gid] = r_max;
            Z[gid] = r_sum;
        }
    }

    if(AmaxWorkspace)
    {
        r_Amax = reductionBlock<NumWarps>(r_Amax, max_abs_op, lid, laneId, warpId);
        if(lid == 0)
        {
            AmaxWorkspace[blockIdx.x] = r_Amax;
        }
    }
}

extern "C" __global__ void SoftMaxBlock(const float* in,
                                        float* out,
                                        float* __restrict__ M,
                                        float* __restrict__ Z,
                                        float* __restrict__ AmaxWorkspace,
                                        const float* __restrict__ descale_Q,
                                        const float* __restrict__ descale_K,
                                        uint32_t seq_len,
                                        uint64_t nhs)
{
    constexpr uint32_t NumWarps = THREADS / warpSize;
    const uint32_t lid          = threadIdx.x;
    const uint32_t laneId       = lid % warpSize;
    const uint32_t warpId       = lid / warpSize;
    const float descaler        = (descale_Q ? *descale_Q : 1.f) * (descale_K ? *descale_K : 1.f);
    const bool save_stats       = M && Z && lid == 0;

    float r_Amax = 0;

    for(uint64_t gid = blockIdx.x; gid < nhs; gid += gridDim.x)
    {
        const float* line = in + gid * seq_len + lid;
        float* res        = out + gid * seq_len + lid;

        float local_val =
            (lid < seq_len) ? (*line) * descaler : std::numeric_limits<float>::lowest();
        float r_max = reductionBlock<NumWarps>(local_val, max_op, lid, laneId, warpId);

        local_val = (lid < seq_len) ? exp(local_val - r_max) : 0;

        float r_sum = 1.f / reductionBlock<NumWarps>(local_val, plus_op, lid, laneId, warpId);

        local_val = (lid < seq_len) ? local_val * r_sum : 0;
        if(lid < seq_len)
        {
            *res = local_val;
        }

        r_Amax = max_abs_op(r_Amax, local_val);

        if(save_stats)
        {
            M[gid] = r_max;
            Z[gid] = r_sum;
        }
    }

    if(AmaxWorkspace)
    {
        r_Amax = reductionBlock<NumWarps>(r_Amax, max_abs_op, lid, laneId, warpId);
        if(lid == 0)
        {
            AmaxWorkspace[blockIdx.x] = r_Amax;
        }
    }
}

extern "C" __global__ void SoftMaxCommon(const float* in,
                                         float* out,
                                         float* __restrict__ M,
                                         float* __restrict__ Z,
                                         float* __restrict__ AmaxWorkspace,
                                         const float* __restrict__ descale_Q,
                                         const float* __restrict__ descale_K,
                                         uint32_t seq_len,
                                         uint64_t nhs)
{
    constexpr uint32_t NumWarps = THREADS / warpSize;
    const uint32_t lid          = threadIdx.x;
    const uint32_t laneId       = lid % warpSize;
    const uint32_t warpId       = lid / warpSize;
    const float descaler        = (descale_Q ? *descale_Q : 1.f) * (descale_K ? *descale_K : 1.f);
    const bool save_stats       = M && Z && lid == 0;

    float r_Amax = 0;

    for(uint64_t gid = blockIdx.x; gid < nhs; gid += gridDim.x)
    {
        const float* line = in + gid * seq_len;
        float* res        = out + gid * seq_len;

        float r_max = reductionCommon<NumWarps>(
            line,
            std::numeric_limits<float>::lowest(),
            seq_len,
            max_op,
            [descaler](float x) { return x * descaler; },
            lid,
            laneId,
            warpId);

        float r_sum = 1.f / reductionCommon<NumWarps>(
                                line,
                                0,
                                seq_len,
                                plus_op,
                                [r_max](float x) { return exp(x - r_max); },
                                lid,
                                laneId,
                                warpId);

        for(uint32_t loop_lid = lid; loop_lid < seq_len; loop_lid += blockDim.x)
        {
            float local_val = exp(line[loop_lid] - r_max) * r_sum;
            res[loop_lid]   = local_val;
            r_Amax          = max_abs_op(r_Amax, local_val);
        }

        if(save_stats)
        {
            M[gid] = r_max;
            Z[gid] = r_sum;
        }
    }

    if(AmaxWorkspace)
    {
        r_Amax = reductionBlock<NumWarps>(r_Amax, max_abs_op, lid, laneId, warpId);
        if(lid == 0)
        {
            AmaxWorkspace[blockIdx.x] = r_Amax;
        }
    }
}
