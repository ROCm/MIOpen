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

// rocblas operates with non-ieee FP8
#define MIOPEN_FP8_IEEE_EXPONENT_BIAS 0

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#endif

#include "miopen_cstdint.hpp"
#include "miopen_limits.hpp"
#include "miopen_rocrand.hpp"

#define MIOPEN_ENABLE_F8_DEVICE_CODE 1
#include <hip_float8.hpp>

// Some versions of amd-clang treats MIOPEN_ENABLE_F8_DEVICE_CODE as unused despite the fact that
// it's used in the subsequent include. Some other versions amd-clang failed to compile everything
// without MIOPEN_ENABLE_F8_DEVICE_CODE explicitely defined.
// Fake-using it.
#ifndef MIOPEN_ENABLE_F8_DEVICE_CODE
#message "MIOPEN_ENABLE_F8_DEVICE_CODE must be defined"
#endif

#ifndef THREADS
#define THREADS 64
#endif

#ifndef OUT_TYPE
#define OUT_TYPE float
#endif

#ifndef dO_TYPE
#define dO_TYPE float
#endif

namespace {
constexpr float plus_op(float a, float b) { return a + b; };
constexpr float fmaxf_op(float a, float b) { return fmaxf(a, b); };

/// Atomically calculates maximum of non-negative ordered values.
/// Produces wrong results for negatve values or nans,
/// but it is a final amax reducton step and we expect only non-negative ordered values.
__forceinline__ __device__ float atomicMaxOfNonNegative(float* addr, float value)
{
    // ordered non-negatve and even infinity values can be compared as integers
    // NOLINTBEGIN
    // cppcheck-suppress invalidPointerCast
    return __int_as_float(atomicMax(reinterpret_cast<int32_t*>(addr), __float_as_int(value)));
    // NOLINTEND
}

template <uint32_t WARP_SIZE, typename Op, uint32_t SWIZZLE_SIZE = WARP_SIZE>
__forceinline__ __device__ float reductionFullWarp(float reduced_val, uint32_t laneId, Op op)
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
            __builtin_amdgcn_ds_bpermute(static_cast<int>(idx << 2), __float_as_int(reduced_val));
        tmp = __int_as_float(itmp);
    }
    else
    {
        // butterfly reduction based on __shfl_xor
        // swizzle <xor_mask[14:10], or_mask[9:5], and_mask[4:0]>()
        constexpr uint32_t xor_off = 10;
        // constexpr uint32_t or_off  = 5;
        constexpr uint32_t and_off = 0;

        constexpr uint32_t field_msk = 0x1f;

        constexpr uint32_t and_msk = warp_msk & field_msk;
        // constexpr uint32_t or_msk  = 0;
        constexpr uint32_t xor_msk = (SWIZZLE_SIZE >> 1) & field_msk;

        // clang tidy does not like that (or_msk << or_off) is zero
        // and cliams that it's redundant, but it's required for
        // __hip_ds_swizzlef_N reference. Menawhile swizzle_op generation
        // must be a part of hip intrinsics, because it depends on ISA
        // like __hip_ds_swizzlef_N<xor_mask, or_mask, and_mask>
        // For some reason NILINT doesn't work.
        // NOLINTBEGIN
        constexpr uint32_t swizzle_op =
            (xor_msk << xor_off) /* | (or_msk << or_off) */ | (and_msk << and_off);
        // NOLINTEND

        tmp = __hip_ds_swizzlef_N<swizzle_op>(reduced_val);
    }

    return op(tmp, reduced_val);
};

template <uint32_t NumWarps, typename Op>
__forceinline__ __device__ float
reductionBlock(float local_val, Op op, uint32_t lid, uint32_t laneId, uint32_t warpId)
{
    static_assert(NumWarps <= warpSize);
    static_assert((NumWarps & (NumWarps - 1)) == 0, "NumWarps must be a power of 2");
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
__forceinline__ __device__ float reductionCommon(const float* __restrict__ line,
                                                 const float* __restrict__ bias_line,
                                                 const float init_value,
                                                 const uint32_t seq_len,
                                                 ReductionOp&& op,
                                                 ElementOp&& eop,
                                                 uint32_t lid,
                                                 uint32_t laneId,
                                                 uint32_t warpId)
{
    float bias        = (bias_line && lid < seq_len) ? bias_line[lid] : 0.0f;
    float reduced_val = (lid < seq_len) ? eop(line[lid] - bias) : init_value;

    if(bias_line)
    {
        for(uint32_t loop_lid = lid + blockDim.x; loop_lid < seq_len; loop_lid += blockDim.x)
            reduced_val = op(eop(line[loop_lid] - bias_line[loop_lid]), reduced_val);
    }
    else
    {
        for(uint32_t loop_lid = lid + blockDim.x; loop_lid < seq_len; loop_lid += blockDim.x)
            reduced_val = op(eop(line[loop_lid]), reduced_val);
    }

    return reductionBlock<NumWarps>(reduced_val, op, lid, laneId, warpId);
};

__forceinline__ __device__ bool doDropout(float dropout, rocrand_device::xorwow_engine* state)
{
    return (dropout > 0.0f && prng::xorwow_uniform(state) < dropout);
}
} // namespace

extern "C" __global__ void __launch_bounds__(THREADS)
    SoftMaxWarp(const float* in,
                OUT_TYPE* out,
                float* __restrict__ M,
                float* __restrict__ Z,
                float* __restrict__ bias,
                float* __restrict__ Amax,
                const float* __restrict__ descale_Q,
                const float* __restrict__ descale_K,
                const float* __restrict__ scale_S,
                const uint64_t* __restrict__ seed,
                const uint64_t* __restrict__ offset,
                const float* __restrict__ dropout_P,
                uint32_t seq_len,
                uint64_t nhs)
{
    static_assert(THREADS % warpSize == 0);
    constexpr uint32_t NumWarps = THREADS / warpSize;
    const uint32_t lid          = threadIdx.x;
    const uint32_t laneId       = lid % warpSize;
    const uint32_t warpId       = lid / warpSize;
    const float descaler        = (descale_Q ? *descale_Q : 1.0f) * (descale_K ? *descale_K : 1.0f);
    const float dropout         = (dropout_P && seed && offset) ? (*dropout_P) : 0.0f;
    const float scaler          = (scale_S ? *scale_S : 1.0f) / (1.0f - dropout);
    const bool save_stats       = M && Z && (laneId == 0);

    rocrand_state_xorwow rng;
    if(dropout > 0.0f)
    {
        const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        rocrand_init(prng::hash(*seed + idx), 0, *offset, &rng);
    }

    float r_Amax = 0;

    for(uint64_t gid = blockIdx.x * NumWarps + warpId; gid < nhs; gid += gridDim.x * NumWarps)
    {
        const float* line = in + gid * seq_len + laneId;
        auto res          = out + gid * seq_len + laneId;

        float local_val =
            (laneId < seq_len) ? (*line) * descaler : std::numeric_limits<float>::lowest();

        if(bias)
            local_val -= *(bias + gid * seq_len + laneId);

        float r_max = reductionFullWarp<warpSize>(local_val, laneId, fmaxf_op);

        local_val = (laneId < seq_len) ? expf(local_val - r_max) : 0;

        float r_sum = 1.0f / reductionFullWarp<warpSize>(local_val, laneId, plus_op);

        local_val *= r_sum;

        // It is supposed to be maximum of absolute values,
        // however we do not need abs() because expf() above produces
        // non-negative value. Plain max() is enough.
        r_Amax = fmaxf_op(r_Amax, local_val);

        if(laneId < seq_len)
        {
            *res = static_cast<OUT_TYPE>(doDropout(dropout, &rng) ? 0.0f : local_val * scaler);
        }

        if(save_stats)
        {
            M[gid] = r_max;
            Z[gid] = r_sum;
        }
    }

    if(Amax)
    {
        r_Amax = reductionBlock<NumWarps>(r_Amax, fmaxf_op, lid, laneId, warpId);
        if(lid == 0)
        {
            atomicMaxOfNonNegative(Amax, r_Amax);
        }
    }
}

extern "C" __global__ void __launch_bounds__(THREADS)
    SoftMaxBlock(const float* in,
                 OUT_TYPE* out,
                 float* __restrict__ M,
                 float* __restrict__ Z,
                 float* __restrict__ bias,
                 float* __restrict__ Amax,
                 const float* __restrict__ descale_Q,
                 const float* __restrict__ descale_K,
                 const float* __restrict__ scale_S,
                 const uint64_t* __restrict__ seed,
                 const uint64_t* __restrict__ offset,
                 const float* __restrict__ dropout_P,
                 uint32_t seq_len,
                 uint64_t nhs)
{
    static_assert(THREADS % warpSize == 0);
    constexpr uint32_t NumWarps = THREADS / warpSize;
    const uint32_t lid          = threadIdx.x;
    const uint32_t laneId       = lid % warpSize;
    const uint32_t warpId       = lid / warpSize;
    const float descaler        = (descale_Q ? *descale_Q : 1.0f) * (descale_K ? *descale_K : 1.0f);
    const float dropout         = (dropout_P && seed && offset) ? (*dropout_P) : 0.0f;
    const float scaler          = (scale_S ? *scale_S : 1.0f) / (1.0f - dropout);
    const bool save_stats       = M && Z && (lid == 0);

    rocrand_state_xorwow rng;
    if(dropout > 0.0f)
    {
        const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        rocrand_init(prng::hash(*seed + idx), 0, *offset, &rng);
    }

    float r_Amax = 0;

    for(uint64_t gid = blockIdx.x; gid < nhs; gid += gridDim.x)
    {
        const float* line = in + gid * seq_len + lid;
        auto res          = out + gid * seq_len + lid;

        float local_val =
            (lid < seq_len) ? (*line) * descaler : std::numeric_limits<float>::lowest();

        if(bias)
            local_val -= *(bias + gid * seq_len + lid);

        float r_max = reductionBlock<NumWarps>(local_val, fmaxf_op, lid, laneId, warpId);

        local_val = (lid < seq_len) ? expf(local_val - r_max) : 0;

        float r_sum = 1.0f / reductionBlock<NumWarps>(local_val, plus_op, lid, laneId, warpId);

        local_val *= r_sum;

        // It is supposed to be maximum of absolute values,
        // however we do not need abs() because expf() above produces
        // non-negative value. Plain max() is enough.
        r_Amax = fmaxf_op(r_Amax, local_val);

        if(lid < seq_len)
        {
            *res = static_cast<OUT_TYPE>(doDropout(dropout, &rng) ? 0.0f : local_val * scaler);
        }

        if(save_stats)
        {
            M[gid] = r_max;
            Z[gid] = r_sum;
        }
    }

    if(Amax)
    {
        r_Amax = reductionBlock<NumWarps>(r_Amax, fmaxf_op, lid, laneId, warpId);
        if(lid == 0)
        {
            atomicMaxOfNonNegative(Amax, r_Amax);
        }
    }
}

extern "C" __global__ void __launch_bounds__(THREADS)
    SoftMaxCommon(const float* in,
                  OUT_TYPE* out,
                  float* __restrict__ M,
                  float* __restrict__ Z,
                  float* __restrict__ bias,
                  float* __restrict__ Amax,
                  const float* __restrict__ descale_Q,
                  const float* __restrict__ descale_K,
                  const float* __restrict__ scale_S,
                  const uint64_t* __restrict__ seed,
                  const uint64_t* __restrict__ offset,
                  const float* __restrict__ dropout_P,
                  uint32_t seq_len,
                  uint64_t nhs)
{
    static_assert(THREADS % warpSize == 0);
    constexpr uint32_t NumWarps = THREADS / warpSize;
    const uint32_t lid          = threadIdx.x;
    const uint32_t laneId       = lid % warpSize;
    const uint32_t warpId       = lid / warpSize;
    const float descaler        = (descale_Q ? *descale_Q : 1.0f) * (descale_K ? *descale_K : 1.0f);
    const float dropout         = (dropout_P && seed && offset) ? (*dropout_P) : 0.0f;
    const float scaler          = (scale_S ? *scale_S : 1.0f) / (1.0f - dropout);
    const bool save_stats       = M && Z && (lid == 0);

    rocrand_state_xorwow rng;
    if(dropout > 0.0f)
    {
        const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        rocrand_init(prng::hash(*seed + idx), 0, *offset, &rng);
    }

    float r_Amax = 0;

    for(uint64_t gid = blockIdx.x; gid < nhs; gid += gridDim.x)
    {
        const float* line      = in + gid * seq_len;
        const float* bias_line = bias + gid * seq_len;
        auto res               = out + gid * seq_len;

        float r_max = reductionCommon<NumWarps>(
            line,
            bias_line,
            std::numeric_limits<float>::lowest(),
            seq_len,
            fmaxf_op,
            [descaler](float x) { return x * descaler; },
            lid,
            laneId,
            warpId);

        float r_sum = 1.0f / reductionCommon<NumWarps>(
                                 line,
                                 nullptr,
                                 0,
                                 seq_len,
                                 plus_op,
                                 [r_max, descaler](float x) { return expf(x * descaler - r_max); },
                                 lid,
                                 laneId,
                                 warpId);

        for(uint32_t loop_lid = lid; loop_lid < seq_len; loop_lid += blockDim.x)
        {
            float local_val = expf(line[loop_lid] * descaler - r_max) * r_sum;

            // It is supposed to be maximum of absolute values,
            // however we do not need abs() because expf() above produces
            // non-negative value. Plain max() is enough.
            r_Amax = fmaxf_op(r_Amax, local_val);

            res[loop_lid] =
                static_cast<OUT_TYPE>(doDropout(dropout, &rng) ? 0.0f : local_val * scaler);
        }

        if(save_stats)
        {
            M[gid] = r_max;
            Z[gid] = r_sum;
        }
    }

    if(Amax)
    {
        r_Amax = reductionBlock<NumWarps>(r_Amax, fmaxf_op, lid, laneId, warpId);
        if(lid == 0)
        {
            atomicMaxOfNonNegative(Amax, r_Amax);
        }
    }
}

extern "C" __global__ void __launch_bounds__(THREADS)
    ScaleReduce(const float* __restrict__ in,
                OUT_TYPE* __restrict__ out,
                float* __restrict__ Amax,
                const float* __restrict__ descale_S,
                const float* __restrict__ descale_V,
                const float* __restrict__ scale_O,
                uint64_t nhsd)
{
    const float descaler = (*descale_S) * (*descale_V);
    const float scaler   = (*scale_O);

    const auto gid  = blockIdx.x * blockDim.x + threadIdx.x;
    const auto step = gridDim.x * blockDim.x;

    auto in_ptr    = in + gid;
    auto out_ptr   = out + gid;
    const auto end = in + nhsd;

    float r_Amax = 0;

    while(in_ptr < end)
    {
        const auto res = *in_ptr * descaler;

        r_Amax = fmaxf_op(r_Amax, fabsf(res));

        *out_ptr = static_cast<OUT_TYPE>(res * scaler);

        in_ptr += step;
        out_ptr += step;
    }

    constexpr uint32_t NumWarps = THREADS / warpSize;
    const uint32_t lid          = threadIdx.x;
    const uint32_t laneId       = lid % warpSize;
    const uint32_t warpId       = lid / warpSize;

    r_Amax = reductionBlock<NumWarps>(r_Amax, fmaxf_op, lid, laneId, warpId);
    if(lid == 0)
    {
        atomicMaxOfNonNegative(Amax, r_Amax);
    }
}

extern "C" __global__ void __launch_bounds__(THREADS)
    ScaleRowReduceWarp(const dO_TYPE* __restrict__ dO,
                       const OUT_TYPE* __restrict__ O,
                       float* __restrict__ out,
                       const float* __restrict__ descale_dO,
                       const float* __restrict__ descale_O,
                       const float* __restrict__ dropout_P,
                       uint32_t d,
                       uint64_t nhs)
{
    static_assert(THREADS % warpSize == 0);
    constexpr uint32_t NumWarps = THREADS / warpSize;
    const uint32_t lid          = threadIdx.x;
    const uint32_t laneId       = lid % warpSize;
    const uint32_t warpId       = lid / warpSize;
    const float scaler          = (*descale_dO) * (*descale_O) * (1.0f - (*dropout_P));

    for(uint64_t gid = blockIdx.x * NumWarps + warpId; gid < nhs; gid += gridDim.x * NumWarps)
    {
        float local_val = 0.0f;
        if(laneId < d)
        {
            const auto dO_ptr = dO + gid * d + laneId;
            const auto O_ptr  = O + gid * d + laneId;

            local_val = static_cast<float>(*dO_ptr) * static_cast<float>(*O_ptr) * scaler;
        }

        local_val = reductionFullWarp<warpSize>(local_val, laneId, plus_op);

        if(laneId == 0)
        {
            out[gid] = local_val;
        }
    }
}

extern "C" __global__ void __launch_bounds__(THREADS)
    ScaleRowReduceBlock(const dO_TYPE* __restrict__ dO,
                        const OUT_TYPE* __restrict__ O,
                        float* __restrict__ out,
                        const float* __restrict__ descale_dO,
                        const float* __restrict__ descale_O,
                        const float* __restrict__ dropout_P,
                        uint32_t d,
                        uint64_t nhs)
{
    static_assert(THREADS % warpSize == 0);
    constexpr uint32_t NumWarps = THREADS / warpSize;
    const uint32_t lid          = threadIdx.x;
    const uint32_t laneId       = lid % warpSize;
    const uint32_t warpId       = lid / warpSize;
    const float scaler          = (*descale_dO) * (*descale_O) * (1.0f - (*dropout_P));

    for(uint64_t gid = blockIdx.x; gid < nhs; gid += gridDim.x)
    {
        const auto dO_ptr = dO + gid * d + lid;
        const auto O_ptr  = O + gid * d + lid;

        float local_val = 0.0f;
        if(lid < d)
        {
            local_val = static_cast<float>(*dO_ptr) * static_cast<float>(*O_ptr) * scaler;
        }

        local_val = reductionBlock<NumWarps>(local_val, plus_op, lid, laneId, warpId);

        if(lid == 0)
        {
            out[gid] = local_val;
        }
    }
}

extern "C" __global__ void __launch_bounds__(THREADS)
    ScaleRowReduceCommon(const dO_TYPE* __restrict__ dO,
                         const OUT_TYPE* __restrict__ O,
                         float* __restrict__ out,
                         const float* __restrict__ descale_dO,
                         const float* __restrict__ descale_O,
                         const float* __restrict__ dropout_P,
                         uint32_t d,
                         uint64_t nhs)
{
    static_assert(THREADS % warpSize == 0);
    constexpr uint32_t NumWarps = THREADS / warpSize;
    const uint32_t lid          = threadIdx.x;
    const uint32_t laneId       = lid % warpSize;
    const uint32_t warpId       = lid / warpSize;
    const float scaler          = (*descale_dO) * (*descale_O) * (1.0f - (*dropout_P));

    for(uint64_t gid = blockIdx.x; gid < nhs; gid += gridDim.x)
    {
        const auto dO_ptr = dO + gid * d;
        const auto O_ptr  = O + gid * d;

        float local_val =
            (lid < d) ? static_cast<float>(dO_ptr[lid]) * static_cast<float>(O_ptr[lid]) * scaler
                      : 0.0f;

        for(uint32_t loop_lid = lid + blockDim.x; loop_lid < d; loop_lid += blockDim.x)
            local_val +=
                static_cast<float>(dO_ptr[loop_lid]) * static_cast<float>(O_ptr[loop_lid]) * scaler;

        local_val = reductionBlock<NumWarps>(local_val, plus_op, lid, laneId, warpId);

        if(lid == 0)
        {
            out[gid] = local_val;
        }
    }
}

extern "C" __global__ void __launch_bounds__(THREADS)
    BwdAttentionWarp(float* __restrict__ QxK_S,
                     const float* dOxV,
                     OUT_TYPE* dS, // may overlap with dOxV
                     const float* __restrict__ M,
                     const float* __restrict__ Zinv,
                     const float* __restrict__ dOxO,
                     float* __restrict__ Amax,
                     const float* __restrict__ descale_Q,
                     const float* __restrict__ descale_K,
                     const float* __restrict__ descale_dO,
                     const float* __restrict__ descale_V,
                     const float* __restrict__ scale_S,
                     const float* __restrict__ scale_dS,
                     const uint64_t* __restrict__ seed,
                     const uint64_t* __restrict__ offset,
                     const float* __restrict__ dropout_P,
                     float scale,
                     uint32_t seq_len,
                     uint64_t nhs)
{
    static_assert(THREADS % warpSize == 0);
    constexpr uint32_t NumWarps = THREADS / warpSize;
    const uint32_t lid          = threadIdx.x;
    const uint32_t laneId       = lid % warpSize;
    const uint32_t warpId       = lid / warpSize;

    const float dropout            = (dropout_P && seed && offset) ? (*dropout_P) : 0.0f;
    const float scaler_dropout     = 1.0f - dropout;
    const float scaler_inv_dropout = 1.0f / scaler_dropout;

    const float descaler_QxK  = (*descale_Q) * (*descale_K);
    const float descaler_dOxV = (*descale_dO) * (*descale_V) * scaler_dropout;

    const float scaler_S  = (*scale_S);
    const float scaler_dS = (*scale_dS);

    rocrand_state_xorwow rng;
    if(dropout > 0.0f)
    {
        const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        rocrand_init(prng::hash(*seed + idx), 0, *offset, &rng);
    }

    float r_Amax = 0;

    for(uint64_t gid = blockIdx.x * NumWarps + warpId; gid < nhs && laneId < seq_len;
        gid += gridDim.x * NumWarps)
    {
        const float M_val    = M[gid];
        const float Zinv_val = Zinv[gid];
        const float dOxO_val = dOxO[gid];

        const size_t idx = gid * seq_len + laneId;

        const float QxK_val = doDropout(dropout, &rng) ? 0.0f
                                                       : expf(QxK_S[idx] * descaler_QxK - M_val) *
                                                             Zinv_val * scaler_inv_dropout;

        QxK_S[idx] = QxK_val * scaler_S;

        const float dOxV_val = (dOxV[idx] * descaler_dOxV - dOxO_val) * scale * QxK_val;

        dS[idx] = static_cast<OUT_TYPE>(dOxV_val * scaler_dS);

        r_Amax = fmaxf_op(r_Amax, fabsf(dOxV_val));
    }

    r_Amax = reductionBlock<NumWarps>(r_Amax, fmaxf_op, lid, laneId, warpId);
    if(lid == 0)
    {
        atomicMaxOfNonNegative(Amax, r_Amax);
    }
}

extern "C" __global__ void __launch_bounds__(THREADS)
    BwdAttentionBlock(float* __restrict__ QxK_S,
                      const float* dOxV,
                      OUT_TYPE* dS, // may overlap with dOxV
                      const float* __restrict__ M,
                      const float* __restrict__ Zinv,
                      const float* __restrict__ dOxO,
                      float* __restrict__ Amax,
                      const float* __restrict__ descale_Q,
                      const float* __restrict__ descale_K,
                      const float* __restrict__ descale_dO,
                      const float* __restrict__ descale_V,
                      const float* __restrict__ scale_S,
                      const float* __restrict__ scale_dS,
                      const uint64_t* __restrict__ seed,
                      const uint64_t* __restrict__ offset,
                      const float* __restrict__ dropout_P,
                      float scale,
                      uint32_t seq_len,
                      uint64_t nhs)
{
    static_assert(THREADS % warpSize == 0);
    constexpr uint32_t NumWarps = THREADS / warpSize;
    const uint32_t lid          = threadIdx.x;
    const uint32_t laneId       = lid % warpSize;
    const uint32_t warpId       = lid / warpSize;

    const float dropout            = (dropout_P && seed && offset) ? (*dropout_P) : 0.0f;
    const float scaler_dropout     = 1.0f - dropout;
    const float scaler_inv_dropout = 1.0f / scaler_dropout;

    const float descaler_QxK  = (*descale_Q) * (*descale_K);
    const float descaler_dOxV = (*descale_dO) * (*descale_V) * scaler_dropout;

    const float scaler_S  = (*scale_S);
    const float scaler_dS = (*scale_dS);

    rocrand_state_xorwow rng;
    if(dropout > 0.0f)
    {
        const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        rocrand_init(prng::hash(*seed + idx), 0, *offset, &rng);
    }

    float r_Amax = 0;

    for(uint64_t gid = blockIdx.x; gid < nhs && lid < seq_len; gid += gridDim.x)
    {
        const float M_val    = M[gid];
        const float Zinv_val = Zinv[gid];
        const float dOxO_val = dOxO[gid];

        const size_t idx = gid * seq_len + lid;

        const float QxK_val = doDropout(dropout, &rng) ? 0.0f
                                                       : expf(QxK_S[idx] * descaler_QxK - M_val) *
                                                             Zinv_val * scaler_inv_dropout;

        QxK_S[idx] = QxK_val * scaler_S;

        const float dOxV_val = (dOxV[idx] * descaler_dOxV - dOxO_val) * scale * QxK_val;

        dS[idx] = static_cast<OUT_TYPE>(dOxV_val * scaler_dS);

        r_Amax = fmaxf_op(r_Amax, fabsf(dOxV_val));
    }

    r_Amax = reductionBlock<NumWarps>(r_Amax, fmaxf_op, lid, laneId, warpId);
    if(lid == 0)
    {
        atomicMaxOfNonNegative(Amax, r_Amax);
    }
}

extern "C" __global__ void __launch_bounds__(THREADS)
    BwdAttentionCommon(float* __restrict__ QxK_S,
                       const float* dOxV,
                       OUT_TYPE* dS, // may overlap with dOxV
                       const float* __restrict__ M,
                       const float* __restrict__ Zinv,
                       const float* __restrict__ dOxO,
                       float* __restrict__ Amax,
                       const float* __restrict__ descale_Q,
                       const float* __restrict__ descale_K,
                       const float* __restrict__ descale_dO,
                       const float* __restrict__ descale_V,
                       const float* __restrict__ scale_S,
                       const float* __restrict__ scale_dS,
                       const uint64_t* __restrict__ seed,
                       const uint64_t* __restrict__ offset,
                       const float* __restrict__ dropout_P,
                       float scale,
                       uint32_t seq_len,
                       uint64_t nhs)
{
    static_assert(THREADS % warpSize == 0);
    constexpr uint32_t NumWarps = THREADS / warpSize;
    const uint32_t lid          = threadIdx.x;
    const uint32_t laneId       = lid % warpSize;
    const uint32_t warpId       = lid / warpSize;

    const float dropout            = (dropout_P && seed && offset) ? (*dropout_P) : 0.0f;
    const float scaler_dropout     = 1.0f - dropout;
    const float scaler_inv_dropout = 1.0f / scaler_dropout;

    const float descaler_QxK  = (*descale_Q) * (*descale_K);
    const float descaler_dOxV = (*descale_dO) * (*descale_V) * scaler_dropout;

    const float scaler_S  = (*scale_S);
    const float scaler_dS = (*scale_dS);

    rocrand_state_xorwow rng;
    if(dropout > 0.0f)
    {
        const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        rocrand_init(prng::hash(*seed + idx), 0, *offset, &rng);
    }

    float r_Amax = 0;

    for(uint64_t gid = blockIdx.x; gid < nhs; gid += gridDim.x)
    {
        const float M_val    = M[gid];
        const float Zinv_val = Zinv[gid];
        const float dOxO_val = dOxO[gid];

        float* QxK_S_ptr      = QxK_S + gid * seq_len;
        const float* dOxV_ptr = dOxV + gid * seq_len;
        OUT_TYPE* dS_ptr      = dS + gid * seq_len;

        for(uint32_t loop_lid = lid; loop_lid < seq_len; loop_lid += blockDim.x)
        {
            const float QxK_val = doDropout(dropout, &rng)
                                      ? 0.0f
                                      : expf(QxK_S_ptr[loop_lid] * descaler_QxK - M_val) *
                                            Zinv_val * scaler_inv_dropout;

            QxK_S_ptr[loop_lid] = QxK_val * scaler_S;

            const float dOxV_val =
                (dOxV_ptr[loop_lid] * descaler_dOxV - dOxO_val) * scale * QxK_val;

            dS_ptr[loop_lid] = static_cast<OUT_TYPE>(dOxV_val * scaler_dS);

            r_Amax = fmaxf_op(r_Amax, fabsf(dOxV_val));
        }
    }

    r_Amax = reductionBlock<NumWarps>(r_Amax, fmaxf_op, lid, laneId, warpId);
    if(lid == 0)
    {
        atomicMaxOfNonNegative(Amax, r_Amax);
    }
}
