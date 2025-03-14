/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020-2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
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
//#include <half.hpp>
#include <hip/hip_bfloat16.h>
#endif

#include "miopen_cstdint.hpp"
#include "miopen_type_traits.hpp"
#include "miopen_limits.hpp"

#define MIOPEN_ENABLE_F8_DEVICE_CODE 1
#include "hip_float8.hpp"

#include "fp8_kern_types.h"

using float8_fnuz  = miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>;
using bfloat8_fnuz = miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>;

template <typename T>
inline __device__ uint32_t draft_rng(T x, uint32_t seed)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    typedef typename std::conditional<sizeof(T) == 2, uint16_t, uint32_t>::type IT;
    IT tmp             = *(reinterpret_cast<IT*>(&x));
    uint32_t drop_bits = uint32_t(tmp) & 0xFFFFu;
    if(sizeof(tmp) == 4)
        drop_bits ^= tmp >> 16;
    drop_bits = ((drop_bits & 31) << 11) | (drop_bits >> 5);
    drop_bits *= 0x7000149;
    uint32_t rng = (drop_bits ^ 0x13371337 ^ (i * 229791) ^ seed);
    return rng;
}

template <typename TI, typename TO>
inline __device__ TO cast_number(const TI input, miopen_f8::hip_f8_rounding_mode mode, uint32_t rng)
{
    if(std::is_same<TI, TO>::value)
    {
        return input;
    }
    if(sizeof(TI) == sizeof(TO))
    {
        const auto tmp = static_cast<float>(input);
        return TO{tmp, mode, rng};
    }
    else if(sizeof(TO) > sizeof(TI))
    {
        return static_cast<TO>(input);
    }
    else
    {
        return TO{input, mode, rng};
    }
}

template <typename TI,
          typename TW,
          typename TO,
          typename in_cast_type,
          typename wei_cast_type,
          typename TACC = double>
inline __device__ void naive_conv_fwd_nchw(const TI* __restrict__ p_in,
                                           const TW* __restrict__ p_wei,
                                           TO* __restrict__ p_out,
                                           const int hi,
                                           const int wi,
                                           const int n,
                                           const int k_per_group,
                                           const int c_per_group,
                                           const int ho,
                                           const int wo,
                                           const int sy,
                                           const int sx,
                                           const int dy,
                                           const int dx,
                                           const int py,
                                           const int px,
                                           const int fy,
                                           const int fx,
                                           const int group,
                                           bool stoch,
                                           uint32_t seed)
{
    /*
     *  need to compute total output pixel: `group * n * k_per_group * ho * wo`.
     *  to distribute this workload, let one workgroup compute `ho * wo` pixel,
     *  hence need `group * n * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = ho * wo;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int in            = (bid / k_per_group) % n;
    int ig            = bid / (n * k_per_group);

    p_in += static_cast<size_t>(in) * c * hi * wi + static_cast<size_t>(ig) * c_per_group * hi * wi;
    p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fy * fx +
             static_cast<size_t>(ik) * c_per_group * fy * fx;
    p_out += static_cast<size_t>(in) * k * ho * wo +
             static_cast<size_t>(ig) * k_per_group * ho * wo + static_cast<size_t>(ik) * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int iho = tid / wo;
        int iwo = tid % wo;

        TACC value = .0f;

        for(int ic = 0; ic < c_per_group; ic++)
        {
            for(int iy = 0; iy < fy; iy++)
            {
                int valid_h = 1;
                int cur_h   = sy * iho - py + dy * iy;
                if(cur_h < 0 || cur_h >= hi)
                    valid_h &= 0;
                for(int ix = 0; ix < fx; ix++)
                {
                    int valid_w = 1;
                    int cur_w   = sx * iwo - px + dx * ix;
                    if(cur_w < 0 || cur_w >= wi)
                        valid_w &= 0;

                    if(valid_w & valid_h)
                    {
                        size_t i_idx = static_cast<size_t>(ic) * hi * wi +
                                       static_cast<size_t>(cur_h) * wi + static_cast<size_t>(cur_w);
                        size_t f_idx = static_cast<size_t>(ic) * fy * fx +
                                       static_cast<size_t>(iy) * fx + static_cast<size_t>(ix);
                        uint32_t rng1 = 0;
                        uint32_t rng2 = 0;
                        auto rnd_mode = miopen_f8::hip_f8_rounding_mode::standard;
                        if(stoch)
                        {
                            rng1     = draft_rng(p_in[i_idx], seed);
                            rng2     = draft_rng(p_in[f_idx], seed);
                            rnd_mode = miopen_f8::hip_f8_rounding_mode::stochastic;
                        }
                        const auto item_in  = in_cast_type(p_in[i_idx], rnd_mode, rng1);
                        const auto item_wei = wei_cast_type(p_wei[f_idx], rnd_mode, rng2);
                        value += static_cast<TACC>(item_in) * static_cast<TACC>(item_wei);
                    }
                }
            }
        }
        size_t o_idx = static_cast<size_t>(iho) * wo + static_cast<size_t>(iwo);
        // p_out[o_idx] = __float2half(static_cast<float>(value));
        p_out[o_idx] = static_cast<TO>(value);
    }
}

extern "C" __global__ void FWD_KERNEL_NAME(const INPUT_TYPE* __restrict__ p_in,

                                           const WEIGHTS_TYPE* __restrict__ p_wei,
                                           OUTPUT_TYPE* __restrict__ p_out,
                                           int hi,
                                           int wi,
                                           int n,
                                           int k_per_group,
                                           int c_per_group,
                                           int ho,
                                           int wo,
                                           int sy,
                                           int sx,
                                           int dy,
                                           int dx,
                                           int py,
                                           int px,
                                           int fy,
                                           int fx,
                                           int group,
                                           bool stochastic,
                                           uint32_t seed)
{
    // instantiate the kernel
    naive_conv_fwd_nchw<INPUT_TYPE,
                        WEIGHTS_TYPE,
                        OUTPUT_TYPE,
                        INPUT_CAST_TYPE,
                        WEIGHTS_CAST_TYPE,
                        ACCUMULATOR_TYPE>(p_in,
                                          p_wei,
                                          p_out,
                                          hi,
                                          wi,
                                          n,
                                          k_per_group,
                                          c_per_group,
                                          ho,
                                          wo,
                                          sy,
                                          sx,
                                          dy,
                                          dx,
                                          py,
                                          px,
                                          fy,
                                          fx,
                                          group,
                                          stochastic,
                                          seed);
}

template <typename TI,
          typename TW,
          typename TO,
          typename wei_cast_type,
          typename out_cast_type,
          typename TACC = double>
inline __device__ void naive_conv_bwd_nchw(TI* __restrict__ p_in,
                                           const TW* __restrict__ p_wei,
                                           const TO* __restrict__ p_out,
                                           int hi,
                                           int wi,
                                           int n,
                                           int k_per_group,
                                           int c_per_group,
                                           int ho,
                                           int wo,
                                           int sy,
                                           int sx,
                                           int dy,
                                           int dx,
                                           int py,
                                           int px,
                                           int fy,
                                           int fx,
                                           int group,
                                           bool stoch,
                                           uint32_t seed)
{
    /*
     *  need to compute total input pixel: `group * n * c_per_group * hi * wi`.
     *  to distribute this workload, let one workgroup compute `hi * wi` pixel,
     *  hence need `group * n * c_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = hi * wi;
    int bid           = blockIdx.x;
    int ic            = bid % c_per_group;
    int in            = (bid / c_per_group) % n;
    int ig            = bid / (n * c_per_group);

    p_in += static_cast<size_t>(in) * c * hi * wi +
            static_cast<size_t>(ig) * c_per_group * hi * wi + static_cast<size_t>(ic) * hi * wi;
    p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fy * fx +
             static_cast<size_t>(ic) * fy * fx;
    p_out +=
        static_cast<size_t>(in) * k * ho * wo + static_cast<size_t>(ig) * k_per_group * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ihi = tid / wi;
        int iwi = tid % wi;

        TACC value = .0f;

        for(int ik = 0; ik < k_per_group; ik++)
        {
            for(int iy = 0; iy < fy; iy++)
            {
                int valid_h = 1;
                int cur_ho  = ihi + py - dy * iy; // cur_h = sy*iho-py+dy*iy;
                if(cur_ho < 0 || cur_ho % sy)
                    valid_h &= 0;
                cur_ho /= sy;
                if(cur_ho >= ho)
                    valid_h &= 0;
                for(int ix = 0; ix < fx; ix++)
                {
                    int valid_w = 1;
                    int cur_wo  = iwi + px - dx * ix; // cur_w = sx*iwo-px+dx*ix;
                    if(cur_wo < 0 || cur_wo % sx)
                        valid_w &= 0;
                    cur_wo /= sx;
                    if(cur_wo >= wo)
                        valid_w &= 0;

                    if(valid_h & valid_w)
                    {
                        size_t o_idx = static_cast<size_t>(ik) * ho * wo +
                                       static_cast<size_t>(cur_ho) * wo +
                                       static_cast<size_t>(cur_wo);
                        size_t f_idx = static_cast<size_t>(ik) * c_per_group * fy * fx +
                                       static_cast<size_t>(iy) * fx + static_cast<size_t>(ix);
                        uint32_t rng1 = 0;
                        uint32_t rng2 = 0;
                        auto rnd_mode = miopen_f8::hip_f8_rounding_mode::standard;
                        if(stoch)
                        {
                            rng1     = draft_rng(p_out[o_idx], seed);
                            rng2     = draft_rng(p_wei[f_idx], seed);
                            rnd_mode = miopen_f8::hip_f8_rounding_mode::stochastic;
                        }
                        const auto item_out = out_cast_type(p_out[o_idx], rnd_mode, rng1);
                        const auto item_wei = wei_cast_type(p_wei[f_idx], rnd_mode, rng2);
                        value += static_cast<TACC>(item_out) * static_cast<TACC>(item_wei);
                    }
                }
            }
        }
        size_t i_idx = static_cast<size_t>(ihi) * wi + static_cast<size_t>(iwi);
        p_in[i_idx]  = static_cast<TI>(value);
    }
}

extern "C" __global__ void BWD_KERNEL_NAME(INPUT_TYPE* __restrict__ p_in,
                                           const WEIGHTS_TYPE* __restrict__ p_wei,
                                           const OUTPUT_TYPE* __restrict__ p_out,
                                           int hi,
                                           int wi,
                                           int n,
                                           int k_per_group,
                                           int c_per_group,
                                           int ho,
                                           int wo,
                                           int sy,
                                           int sx,
                                           int dy,
                                           int dx,
                                           int py,
                                           int px,
                                           int fy,
                                           int fx,
                                           int group,
                                           bool stochastic,
                                           uint32_t seed)
{
    // instantiate template
    naive_conv_bwd_nchw<INPUT_TYPE,
                        WEIGHTS_TYPE,
                        OUTPUT_TYPE,
                        WEIGHTS_CAST_TYPE,
                        OUTPUT_CAST_TYPE,
                        ACCUMULATOR_TYPE>(p_in,
                                          p_wei,
                                          p_out,
                                          hi,
                                          wi,
                                          n,
                                          k_per_group,
                                          c_per_group,
                                          ho,
                                          wo,
                                          sy,
                                          sx,
                                          dy,
                                          dx,
                                          py,
                                          px,
                                          fy,
                                          fx,
                                          group,
                                          stochastic,
                                          seed);
}

template <typename TI,
          typename TW,
          typename TO,
          typename in_cast_type,
          typename out_cast_type,
          typename TACC = double>
inline __device__ void naive_conv_wrw_nchw(const TI* __restrict__ p_in,
                                           TW* __restrict__ p_wei,
                                           const TO* __restrict__ p_out,
                                           int hi,
                                           int wi,
                                           int n,
                                           int k_per_group,
                                           int c_per_group,
                                           int ho,
                                           int wo,
                                           int sy,
                                           int sx,
                                           int dy,
                                           int dx,
                                           int py,
                                           int px,
                                           int fy,
                                           int fx,
                                           int group,
                                           bool stoch,
                                           uint32_t seed)
{
    /*
     *  need to compute total filter pixel: `group * k_per_group * c_per_group *
     * fy * fx`.
     *  to distribute this workload, let one workgroup compute `c_per_group * fy
     * * fx` pixel,
     *  hence need `group * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = c_per_group * fy * fx;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int ig            = bid / k_per_group;

    p_in += static_cast<size_t>(ig) * c_per_group * hi * wi;
    p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fy * fx +
             static_cast<size_t>(ik) * c_per_group * fy * fx;
    p_out += static_cast<size_t>(ig) * k_per_group * ho * wo + static_cast<size_t>(ik) * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ix = tid % fx;
        int iy = (tid / fx) % fy;
        int ic = tid / (fx * fy);

        TACC value = .0f;

        for(int in = 0; in < n; in++)
        {
            for(int iho = 0; iho < ho; iho++)
            {
                int valid_h = 1;
                int cur_h   = sy * iho - py + dy * iy;
                if(cur_h < 0 || cur_h >= hi)
                    valid_h &= 0;
                for(int iwo = 0; iwo < wo; iwo++)
                {
                    int valid_w = 1;
                    int cur_w   = sx * iwo - px + dx * ix;
                    if(cur_w < 0 || cur_w >= wi)
                        valid_w &= 0;

                    if(valid_h & valid_w)
                    {
                        size_t i_idx = static_cast<size_t>(in) * c * hi * wi +
                                       static_cast<size_t>(ic) * hi * wi +
                                       static_cast<size_t>(cur_h) * wi + static_cast<size_t>(cur_w);
                        size_t o_idx = static_cast<size_t>(in) * k * ho * wo +
                                       static_cast<size_t>(iho) * wo + static_cast<size_t>(iwo);
                        uint32_t rng1 = 0;
                        uint32_t rng2 = 0;
                        auto rnd_mode = miopen_f8::hip_f8_rounding_mode::standard;
                        if(stoch)
                        {
                            rng1     = draft_rng(p_in[i_idx], seed);
                            rng2     = draft_rng(p_out[o_idx], seed);
                            rnd_mode = miopen_f8::hip_f8_rounding_mode::stochastic;
                        }
                        const auto item_in  = in_cast_type(p_in[i_idx], rnd_mode, rng1);
                        const auto item_out = out_cast_type(p_out[o_idx], rnd_mode, rng2);
                        value += static_cast<TACC>(item_in) * static_cast<TACC>(item_out);
                    }
                }
            }
        }
        size_t f_idx = static_cast<size_t>(ic) * fy * fx + static_cast<size_t>(iy) * fx +
                       static_cast<size_t>(ix);
        p_wei[f_idx] = static_cast<TW>(value);
    }
}

extern "C" __global__ void WRW_KERNEL_NAME(const INPUT_TYPE* __restrict__ p_in,
                                           WEIGHTS_TYPE* __restrict__ p_wei,
                                           const OUTPUT_TYPE* __restrict__ p_out,
                                           int hi,
                                           int wi,
                                           int n,
                                           int k_per_group,
                                           int c_per_group,
                                           int ho,
                                           int wo,
                                           int sy,
                                           int sx,
                                           int dy,
                                           int dx,
                                           int py,
                                           int px,
                                           int fy,
                                           int fx,
                                           int group,
                                           bool stochastic,
                                           uint32_t seed)
{
    naive_conv_wrw_nchw<INPUT_TYPE,
                        WEIGHTS_TYPE,
                        OUTPUT_TYPE,
                        INPUT_CAST_TYPE,
                        OUTPUT_CAST_TYPE,
                        ACCUMULATOR_TYPE>(p_in,
                                          p_wei,
                                          p_out,
                                          hi,
                                          wi,
                                          n,
                                          k_per_group,
                                          c_per_group,
                                          ho,
                                          wo,
                                          sy,
                                          sx,
                                          dy,
                                          dx,
                                          py,
                                          px,
                                          fy,
                                          fx,
                                          group,
                                          stochastic,
                                          seed);
}
