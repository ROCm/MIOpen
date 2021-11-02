/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// hcc seems need __device__ __host__ together to compile, and no extern "C"
typedef union _cvt_bf16_fp32
{
    uint u32;
    ushort2 ushortx2;
    ushort ushortvec[2];
    float f32;
} _cvt_bf16_fp32_t;

inline __device__ __host__ float __bfloat16_to_float(ushort src_val)
{
    _cvt_bf16_fp32_t target_val;
    target_val.ushortx2 = make_ushort2(0, src_val);
    return target_val.f32;
}

inline __device__ __host__ ushort __float_to_bfloat16(float src_val)
{
    _cvt_bf16_fp32_t target_val;
    target_val.f32 = src_val;

    if((~target_val.u32 & 0x7f800000) == 0) // Inf or NaN
    {
        if((target_val.u32 & 0xffff) != 0)
        {
            target_val.u32 |= 0x10000; // Preserve signaling NaN
        }
    }
    else
    {
#ifdef MIOPEN_USE_RNE_BFLOAT16
        target_val.u32 += (0x7fff + (target_val.ushortvec[1] & 1));
#endif // MIOPEN_USE_RNE_BFLOAT16
    }
    return target_val.ushortvec[1];
}

// design block_size 256
extern "C" __global__ void naive_conv_fwd_nchw_fp32(const float* __restrict__ p_in,
                                                    const float* __restrict__ p_wei,
                                                    float* __restrict__ p_out,
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
                                                    int group)
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

        double value = .0f;

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
                        value +=
                            static_cast<double>(p_in[i_idx]) * static_cast<double>(p_wei[f_idx]);
                    }
                }
            }
        }
        size_t o_idx = static_cast<size_t>(iho) * wo + static_cast<size_t>(iwo);
        p_out[o_idx] = static_cast<float>(value);
    }
}

extern "C" __global__ void naive_conv_bwd_nchw_fp32(float* __restrict__ p_in,
                                                    const float* __restrict__ p_wei,
                                                    const float* __restrict__ p_out,
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
                                                    int group)
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

        double value = .0f;

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
                        value +=
                            static_cast<double>(p_out[o_idx]) * static_cast<double>(p_wei[f_idx]);
                    }
                }
            }
        }
        size_t i_idx = static_cast<size_t>(ihi) * wi + static_cast<size_t>(iwi);
        p_in[i_idx]  = static_cast<float>(value);
    }
}

extern "C" __global__ void naive_conv_wrw_nchw_fp32(const float* __restrict__ p_in,
                                                    float* __restrict__ p_wei,
                                                    const float* __restrict__ p_out,
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
                                                    int group)
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

        double value = .0f;

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
                        value +=
                            static_cast<double>(p_in[i_idx]) * static_cast<double>(p_out[o_idx]);
                    }
                }
            }
        }
        size_t f_idx = static_cast<size_t>(ic) * fy * fx + static_cast<size_t>(iy) * fx +
                       static_cast<size_t>(ix);
        p_wei[f_idx] = static_cast<float>(value);
    }
}

// design block_size 256
extern "C" __global__ void naive_conv_fwd_ncdhw_fp32(const float* __restrict__ p_in,
                                                     const float* __restrict__ p_wei,
                                                     float* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total output pixel: `group * n * k_per_group * do_ * ho
     * * wo`.
     *  to distribute this workload, let one workgroup compute `do_ * ho * wo`
     * pixel,
     *  hence need `group * n * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = do_ * ho * wo;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int in            = (bid / k_per_group) % n;
    int ig            = bid / (n * k_per_group);

    p_in += static_cast<size_t>(in) * c * di * hi * wi +
            static_cast<size_t>(ig) * c_per_group * di * hi * wi;
    p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fz * fy * fx +
             static_cast<size_t>(ik) * c_per_group * fz * fy * fx;
    p_out += static_cast<size_t>(in) * k * do_ * ho * wo +
             static_cast<size_t>(ig) * k_per_group * do_ * ho * wo +
             static_cast<size_t>(ik) * do_ * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int iwo = tid % wo;
        int iho = (tid / wo) % ho;
        int ido = tid / (ho * wo);

        double value = .0f;

        for(int ic = 0; ic < c_per_group; ic++)
        {
            for(int iz = 0; iz < fz; iz++)
            {
                int valid_d = 1;
                int cur_d   = sz * ido - pz + dz * iz;
                if(cur_d < 0 || cur_d >= di)
                    valid_d &= 0;
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

                        if(valid_d & valid_w & valid_h)
                        {
                            size_t i_idx = static_cast<size_t>(ic) * di * hi * wi +
                                           static_cast<size_t>(cur_d) * hi * wi +
                                           static_cast<size_t>(cur_h) * wi +
                                           static_cast<size_t>(cur_w);
                            size_t f_idx = static_cast<size_t>(ic) * fz * fy * fx +
                                           static_cast<size_t>(iz) * fy * fx +
                                           static_cast<size_t>(iy) * fx + static_cast<size_t>(ix);
                            value += static_cast<double>(p_in[i_idx]) *
                                     static_cast<double>(p_wei[f_idx]);
                        }
                    }
                }
            }
        }
        size_t o_idx = static_cast<size_t>(ido) * ho * wo + static_cast<size_t>(iho) * wo +
                       static_cast<size_t>(iwo);
        p_out[o_idx] = static_cast<float>(value);
    }
}

extern "C" __global__ void naive_conv_bwd_ncdhw_fp32(float* __restrict__ p_in,
                                                     const float* __restrict__ p_wei,
                                                     const float* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total input pixel: `group * n * c_per_group * di * hi *
     * wi`.
     *  to distribute this workload, let one workgroup compute `di * hi * wi`
     * pixel,
     *  hence need `group * n * c_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = di * hi * wi;
    int bid           = blockIdx.x;
    int ic            = bid % c_per_group;
    int in            = (bid / c_per_group) % n;
    int ig            = bid / (n * c_per_group);

    p_in += static_cast<size_t>(in) * c * di * hi * wi +
            static_cast<size_t>(ig) * c_per_group * di * hi * wi +
            static_cast<size_t>(ic) * di * hi * wi;
    p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fz * fy * fx +
             static_cast<size_t>(ic) * fz * fy * fx;
    p_out += static_cast<size_t>(in) * k * do_ * ho * wo +
             static_cast<size_t>(ig) * k_per_group * do_ * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int iwi = tid % wi;
        int ihi = (tid / wi) % hi;
        int idi = tid / (hi * wi);

        double value = .0f;

        for(int ik = 0; ik < k_per_group; ik++)
        {
            for(int iz = 0; iz < fz; iz++)
            {
                int valid_d = 1;
                int cur_do  = idi + pz - dz * iz;
                if(cur_do < 0 || cur_do % sz)
                    valid_d &= 0;
                cur_do /= sz;
                if(cur_do >= do_)
                    valid_d &= 0;
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

                        if(valid_d & valid_h & valid_w)
                        {
                            size_t o_idx = static_cast<size_t>(ik) * do_ * ho * wo +
                                           static_cast<size_t>(cur_do) * ho * wo +
                                           static_cast<size_t>(cur_ho) * wo +
                                           static_cast<size_t>(cur_wo);
                            size_t f_idx = static_cast<size_t>(ik) * c_per_group * fz * fy * fx +
                                           static_cast<size_t>(iz) * fy * fx +
                                           static_cast<size_t>(iy) * fx + static_cast<size_t>(ix);
                            value += static_cast<double>(p_out[o_idx]) *
                                     static_cast<double>(p_wei[f_idx]);
                        }
                    }
                }
            }
        }
        size_t i_idx = static_cast<size_t>(idi) * hi * wi + static_cast<size_t>(ihi) * wi +
                       static_cast<size_t>(iwi);
        p_in[i_idx] = static_cast<float>(value);
    }
}

extern "C" __global__ void naive_conv_wrw_ncdhw_fp32(const float* __restrict__ p_in,
                                                     float* __restrict__ p_wei,
                                                     const float* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total filter pixel: `group * k_per_group * c_per_group *
     * fz * fy * fx`.
     *  to distribute this workload, let one workgroup compute `c_per_group * fz
     * * fy * fx` pixel,
     *  hence need `group * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = c_per_group * fz * fy * fx;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int ig            = bid / k_per_group;

    p_in += static_cast<size_t>(ig) * c_per_group * di * hi * wi;
    p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fz * fy * fx +
             static_cast<size_t>(ik) * c_per_group * fz * fy * fx;
    p_out += static_cast<size_t>(ig) * k_per_group * do_ * ho * wo +
             static_cast<size_t>(ik) * do_ * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ix = tid % fx;
        int iy = (tid / fx) % fy;
        int iz = (tid / (fx * fy)) % fz;
        int ic = tid / (fx * fy * fz);

        double value = .0f;

        for(int in = 0; in < n; in++)
        {
            for(int ido = 0; ido < do_; ido++)
            {
                int valid_d = 1;
                int cur_d   = sz * ido - pz + dz * iz;
                if(cur_d < 0 || cur_d >= di)
                    valid_d &= 0;
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

                        if(valid_d & valid_h & valid_w)
                        {
                            size_t i_idx = static_cast<size_t>(in) * c * di * hi * wi +
                                           static_cast<size_t>(ic) * di * hi * wi +
                                           static_cast<size_t>(cur_d) * hi * wi +
                                           static_cast<size_t>(cur_h) * wi +
                                           static_cast<size_t>(cur_w);
                            size_t o_idx = static_cast<size_t>(in) * k * do_ * ho * wo +
                                           static_cast<size_t>(ido) * ho * wo +
                                           static_cast<size_t>(iho) * wo + static_cast<size_t>(iwo);
                            value += static_cast<double>(p_in[i_idx]) *
                                     static_cast<double>(p_out[o_idx]);
                        }
                    }
                }
            }
        }
        size_t f_idx = static_cast<size_t>(ic) * fz * fy * fx + static_cast<size_t>(iz) * fy * fx +
                       static_cast<size_t>(iy) * fx + static_cast<size_t>(ix);
        p_wei[f_idx] = static_cast<float>(value);
    }
}

extern "C" __global__ void naive_conv_fwd_nchw_fp16(const half* __restrict__ p_in,
                                                    const half* __restrict__ p_wei,
                                                    half* __restrict__ p_out,
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
                                                    int group)
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

        double value = .0f;

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
                        value += static_cast<double>(__half2float(p_in[i_idx])) *
                                 static_cast<double>(__half2float(p_wei[f_idx]));
                    }
                }
            }
        }
        size_t o_idx = static_cast<size_t>(iho) * wo + static_cast<size_t>(iwo);
        p_out[o_idx] = __float2half(static_cast<float>(value));
    }
}

extern "C" __global__ void naive_conv_bwd_nchw_fp16(half* __restrict__ p_in,
                                                    const half* __restrict__ p_wei,
                                                    const half* __restrict__ p_out,
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
                                                    int group)
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

        double value = .0f;

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
                        value += static_cast<double>(__half2float(p_out[o_idx])) *
                                 static_cast<double>(__half2float(p_wei[f_idx]));
                    }
                }
            }
        }
        size_t i_idx = static_cast<size_t>(ihi) * wi + static_cast<size_t>(iwi);
        p_in[i_idx]  = __float2half(static_cast<float>(value));
    }
}

extern "C" __global__ void naive_conv_wrw_nchw_fp16(const half* __restrict__ p_in,
                                                    half* __restrict__ p_wei,
                                                    const half* __restrict__ p_out,
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
                                                    int group)
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

        double value = .0f;

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
                        value += static_cast<double>(__half2float(p_in[i_idx])) *
                                 static_cast<double>(__half2float(p_out[o_idx]));
                    }
                }
            }
        }
        size_t f_idx = static_cast<size_t>(ic) * fy * fx + static_cast<size_t>(iy) * fx +
                       static_cast<size_t>(ix);
        p_wei[f_idx] = __float2half(static_cast<float>(value));
    }
}

// design block_size 256
extern "C" __global__ void naive_conv_fwd_ncdhw_fp16(const half* __restrict__ p_in,
                                                     const half* __restrict__ p_wei,
                                                     half* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total output pixel: `group * n * k_per_group * do_ * ho
     * * wo`.
     *  to distribute this workload, let one workgroup compute `do_ * ho * wo`
     * pixel,
     *  hence need `group * n * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = do_ * ho * wo;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int in            = (bid / k_per_group) % n;
    int ig            = bid / (n * k_per_group);

    p_in += static_cast<size_t>(in) * c * di * hi * wi +
            static_cast<size_t>(ig) * c_per_group * di * hi * wi;
    p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fz * fy * fx +
             static_cast<size_t>(ik) * c_per_group * fz * fy * fx;
    p_out += static_cast<size_t>(in) * k * do_ * ho * wo +
             static_cast<size_t>(ig) * k_per_group * do_ * ho * wo +
             static_cast<size_t>(ik) * do_ * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int iwo = tid % wo;
        int iho = (tid / wo) % ho;
        int ido = tid / (ho * wo);

        double value = .0f;

        for(int ic = 0; ic < c_per_group; ic++)
        {
            for(int iz = 0; iz < fz; iz++)
            {
                int valid_d = 1;
                int cur_d   = sz * ido - pz + dz * iz;
                if(cur_d < 0 || cur_d >= di)
                    valid_d &= 0;
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

                        if(valid_d & valid_w & valid_h)
                        {
                            size_t i_idx = static_cast<size_t>(ic) * di * hi * wi +
                                           static_cast<size_t>(cur_d) * hi * wi +
                                           static_cast<size_t>(cur_h) * wi +
                                           static_cast<size_t>(cur_w);
                            size_t f_idx = static_cast<size_t>(ic) * fz * fy * fx +
                                           static_cast<size_t>(iz) * fy * fx +
                                           static_cast<size_t>(iy) * fx + static_cast<size_t>(ix);
                            value += static_cast<double>(__half2float(p_in[i_idx])) *
                                     static_cast<double>(__half2float(p_wei[f_idx]));
                        }
                    }
                }
            }
        }
        size_t o_idx = static_cast<size_t>(ido) * ho * wo + static_cast<size_t>(iho) * wo +
                       static_cast<size_t>(iwo);
        p_out[o_idx] = __float2half(static_cast<float>(value));
    }
}

extern "C" __global__ void naive_conv_bwd_ncdhw_fp16(half* __restrict__ p_in,
                                                     const half* __restrict__ p_wei,
                                                     const half* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total input pixel: `group * n * c_per_group * di * hi *
     * wi`.
     *  to distribute this workload, let one workgroup compute `di * hi * wi`
     * pixel,
     *  hence need `group * n * c_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = di * hi * wi;
    int bid           = blockIdx.x;
    int ic            = bid % c_per_group;
    int in            = (bid / c_per_group) % n;
    int ig            = bid / (n * c_per_group);

    p_in += static_cast<size_t>(in) * c * di * hi * wi +
            static_cast<size_t>(ig) * c_per_group * di * hi * wi +
            static_cast<size_t>(ic) * di * hi * wi;
    p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fz * fy * fx +
             static_cast<size_t>(ic) * fz * fy * fx;
    p_out += static_cast<size_t>(in) * k * do_ * ho * wo +
             static_cast<size_t>(ig) * k_per_group * do_ * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int iwi = tid % wi;
        int ihi = (tid / wi) % hi;
        int idi = tid / (hi * wi);

        double value = .0f;

        for(int ik = 0; ik < k_per_group; ik++)
        {
            for(int iz = 0; iz < fz; iz++)
            {
                int valid_d = 1;
                int cur_do  = idi + pz - dz * iz;
                if(cur_do < 0 || cur_do % sz)
                    valid_d &= 0;
                cur_do /= sz;
                if(cur_do >= do_)
                    valid_d &= 0;
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

                        if(valid_d & valid_h & valid_w)
                        {
                            size_t o_idx = static_cast<size_t>(ik) * do_ * ho * wo +
                                           static_cast<size_t>(cur_do) * ho * wo +
                                           static_cast<size_t>(cur_ho) * wo +
                                           static_cast<size_t>(cur_wo);
                            size_t f_idx = static_cast<size_t>(ik) * c_per_group * fz * fy * fx +
                                           static_cast<size_t>(iz) * fy * fx +
                                           static_cast<size_t>(iy) * fx + static_cast<size_t>(ix);
                            value += static_cast<double>(__half2float(p_out[o_idx])) *
                                     static_cast<double>(__half2float(p_wei[f_idx]));
                        }
                    }
                }
            }
        }
        size_t i_idx = static_cast<size_t>(idi) * hi * wi + static_cast<size_t>(ihi) * wi +
                       static_cast<size_t>(iwi);
        p_in[i_idx] = __float2half(static_cast<float>(value));
    }
}

extern "C" __global__ void naive_conv_wrw_ncdhw_fp16(const half* __restrict__ p_in,
                                                     half* __restrict__ p_wei,
                                                     const half* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total filter pixel: `group * k_per_group * c_per_group *
     * fz * fy * fx`.
     *  to distribute this workload, let one workgroup compute `c_per_group * fz
     * * fy * fx` pixel,
     *  hence need `group * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = c_per_group * fz * fy * fx;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int ig            = bid / k_per_group;

    p_in += static_cast<size_t>(ig) * c_per_group * di * hi * wi;
    p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fz * fy * fx +
             static_cast<size_t>(ik) * c_per_group * fz * fy * fx;
    p_out += static_cast<size_t>(ig) * k_per_group * do_ * ho * wo +
             static_cast<size_t>(ik) * do_ * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ix = tid % fx;
        int iy = (tid / fx) % fy;
        int iz = (tid / (fx * fy)) % fz;
        int ic = tid / (fx * fy * fz);

        double value = .0f;

        for(int in = 0; in < n; in++)
        {
            for(int ido = 0; ido < do_; ido++)
            {
                int valid_d = 1;
                int cur_d   = sz * ido - pz + dz * iz;
                if(cur_d < 0 || cur_d >= di)
                    valid_d &= 0;
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

                        if(valid_d & valid_h & valid_w)
                        {
                            size_t i_idx = static_cast<size_t>(in) * c * di * hi * wi +
                                           static_cast<size_t>(ic) * di * hi * wi +
                                           static_cast<size_t>(cur_d) * hi * wi +
                                           static_cast<size_t>(cur_h) * wi +
                                           static_cast<size_t>(cur_w);
                            size_t o_idx = static_cast<size_t>(in) * k * do_ * ho * wo +
                                           static_cast<size_t>(ido) * ho * wo +
                                           static_cast<size_t>(iho) * wo + static_cast<size_t>(iwo);
                            value += static_cast<double>(__half2float(p_in[i_idx])) *
                                     static_cast<double>(__half2float(p_out[o_idx]));
                        }
                    }
                }
            }
        }
        size_t f_idx = static_cast<size_t>(ic) * fz * fy * fx + static_cast<size_t>(iz) * fy * fx +
                       static_cast<size_t>(iy) * fx + static_cast<size_t>(ix);
        p_wei[f_idx] = __float2half(static_cast<float>(value));
    }
}

extern "C" __global__ void naive_conv_fwd_nchw_bf16(const ushort* __restrict__ p_in,
                                                    const ushort* __restrict__ p_wei,
                                                    ushort* __restrict__ p_out,
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
                                                    int group)
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

        double value = .0f;

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
                        value += static_cast<double>(__bfloat16_to_float(p_in[i_idx])) *
                                 static_cast<double>(__bfloat16_to_float(p_wei[f_idx]));
                    }
                }
            }
        }
        size_t o_idx = static_cast<size_t>(iho) * wo + static_cast<size_t>(iwo);
        p_out[o_idx] = __float_to_bfloat16(static_cast<float>(value));
    }
}

extern "C" __global__ void naive_conv_bwd_nchw_bf16(ushort* __restrict__ p_in,
                                                    const ushort* __restrict__ p_wei,
                                                    const ushort* __restrict__ p_out,
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
                                                    int group)
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

        double value = .0f;

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
                        value += static_cast<double>(__bfloat16_to_float(p_out[o_idx])) *
                                 static_cast<double>(__bfloat16_to_float(p_wei[f_idx]));
                    }
                }
            }
        }
        size_t i_idx = static_cast<size_t>(ihi) * wi + static_cast<size_t>(iwi);
        p_in[i_idx]  = __float_to_bfloat16(static_cast<float>(value));
    }
}

extern "C" __global__ void naive_conv_wrw_nchw_bf16(const ushort* __restrict__ p_in,
                                                    ushort* __restrict__ p_wei,
                                                    const ushort* __restrict__ p_out,
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
                                                    int group)
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

        double value = .0f;

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
                        value += static_cast<double>(__bfloat16_to_float(p_in[i_idx])) *
                                 static_cast<double>(__bfloat16_to_float(p_out[o_idx]));
                    }
                }
            }
        }
        size_t f_idx = static_cast<size_t>(ic) * fy * fx + static_cast<size_t>(iy) * fx +
                       static_cast<size_t>(ix);
        p_wei[f_idx] = __float_to_bfloat16(static_cast<float>(value));
    }
}

// design block_size 256
extern "C" __global__ void naive_conv_fwd_ncdhw_bf16(const ushort* __restrict__ p_in,
                                                     const ushort* __restrict__ p_wei,
                                                     ushort* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total output pixel: `group * n * k_per_group * do_ * ho
     * * wo`.
     *  to distribute this workload, let one workgroup compute `do_ * ho * wo`
     * pixel,
     *  hence need `group * n * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = do_ * ho * wo;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int in            = (bid / k_per_group) % n;
    int ig            = bid / (n * k_per_group);

    p_in += static_cast<size_t>(in) * c * di * hi * wi +
            static_cast<size_t>(ig) * c_per_group * di * hi * wi;
    p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fz * fy * fx +
             static_cast<size_t>(ik) * c_per_group * fz * fy * fx;
    p_out += static_cast<size_t>(in) * k * do_ * ho * wo +
             static_cast<size_t>(ig) * k_per_group * do_ * ho * wo +
             static_cast<size_t>(ik) * do_ * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int iwo = tid % wo;
        int iho = (tid / wo) % ho;
        int ido = tid / (ho * wo);

        double value = .0f;

        for(int ic = 0; ic < c_per_group; ic++)
        {
            for(int iz = 0; iz < fz; iz++)
            {
                int valid_d = 1;
                int cur_d   = sz * ido - pz + dz * iz;
                if(cur_d < 0 || cur_d >= di)
                    valid_d &= 0;
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

                        if(valid_d & valid_w & valid_h)
                        {
                            size_t i_idx = static_cast<size_t>(ic) * di * hi * wi +
                                           static_cast<size_t>(cur_d) * hi * wi +
                                           static_cast<size_t>(cur_h) * wi +
                                           static_cast<size_t>(cur_w);
                            size_t f_idx = static_cast<size_t>(ic) * fz * fy * fx +
                                           static_cast<size_t>(iz) * fy * fx +
                                           static_cast<size_t>(iy) * fx + static_cast<size_t>(ix);
                            value += static_cast<double>(__bfloat16_to_float(p_in[i_idx])) *
                                     static_cast<double>(__bfloat16_to_float(p_wei[f_idx]));
                        }
                    }
                }
            }
        }
        size_t o_idx = static_cast<size_t>(ido) * ho * wo + static_cast<size_t>(iho) * wo +
                       static_cast<size_t>(iwo);
        p_out[o_idx] = __float_to_bfloat16(static_cast<float>(value));
    }
}

extern "C" __global__ void naive_conv_bwd_ncdhw_bf16(ushort* __restrict__ p_in,
                                                     const ushort* __restrict__ p_wei,
                                                     const ushort* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total input pixel: `group * n * c_per_group * di * hi *
     * wi`.
     *  to distribute this workload, let one workgroup compute `di * hi * wi`
     * pixel,
     *  hence need `group * n * c_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = di * hi * wi;
    int bid           = blockIdx.x;
    int ic            = bid % c_per_group;
    int in            = (bid / c_per_group) % n;
    int ig            = bid / (n * c_per_group);

    p_in += static_cast<size_t>(in) * c * di * hi * wi +
            static_cast<size_t>(ig) * c_per_group * di * hi * wi +
            static_cast<size_t>(ic) * di * hi * wi;
    p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fz * fy * fx +
             static_cast<size_t>(ic) * fz * fy * fx;
    p_out += static_cast<size_t>(in) * k * do_ * ho * wo +
             static_cast<size_t>(ig) * k_per_group * do_ * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int iwi = tid % wi;
        int ihi = (tid / wi) % hi;
        int idi = tid / (hi * wi);

        double value = .0f;

        for(int ik = 0; ik < k_per_group; ik++)
        {
            for(int iz = 0; iz < fz; iz++)
            {
                int valid_d = 1;
                int cur_do  = idi + pz - dz * iz;
                if(cur_do < 0 || cur_do % sz)
                    valid_d &= 0;
                cur_do /= sz;
                if(cur_do >= do_)
                    valid_d &= 0;
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

                        if(valid_d & valid_h & valid_w)
                        {
                            size_t o_idx = static_cast<size_t>(ik) * do_ * ho * wo +
                                           static_cast<size_t>(cur_do) * ho * wo +
                                           static_cast<size_t>(cur_ho) * wo +
                                           static_cast<size_t>(cur_wo);
                            size_t f_idx = static_cast<size_t>(ik) * c_per_group * fz * fy * fx +
                                           static_cast<size_t>(iz) * fy * fx +
                                           static_cast<size_t>(iy) * fx + static_cast<size_t>(ix);
                            value += static_cast<double>(__bfloat16_to_float(p_out[o_idx])) *
                                     static_cast<double>(__bfloat16_to_float(p_wei[f_idx]));
                        }
                    }
                }
            }
        }
        size_t i_idx = static_cast<size_t>(idi) * hi * wi + static_cast<size_t>(ihi) * wi +
                       static_cast<size_t>(iwi);
        p_in[i_idx] = __float_to_bfloat16(static_cast<float>(value));
    }
}

extern "C" __global__ void naive_conv_wrw_ncdhw_bf16(const ushort* __restrict__ p_in,
                                                     ushort* __restrict__ p_wei,
                                                     const ushort* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total filter pixel: `group * k_per_group * c_per_group *
     * fz * fy * fx`.
     *  to distribute this workload, let one workgroup compute `c_per_group * fz
     * * fy * fx` pixel,
     *  hence need `group * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = c_per_group * fz * fy * fx;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int ig            = bid / k_per_group;

    p_in += static_cast<size_t>(ig) * c_per_group * di * hi * wi;
    p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fz * fy * fx +
             static_cast<size_t>(ik) * c_per_group * fz * fy * fx;
    p_out += static_cast<size_t>(ig) * k_per_group * do_ * ho * wo +
             static_cast<size_t>(ik) * do_ * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ix = tid % fx;
        int iy = (tid / fx) % fy;
        int iz = (tid / (fx * fy)) % fz;
        int ic = tid / (fx * fy * fz);

        double value = .0f;

        for(int in = 0; in < n; in++)
        {
            for(int ido = 0; ido < do_; ido++)
            {
                int valid_d = 1;
                int cur_d   = sz * ido - pz + dz * iz;
                if(cur_d < 0 || cur_d >= di)
                    valid_d &= 0;
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

                        if(valid_d & valid_h & valid_w)
                        {
                            size_t i_idx = static_cast<size_t>(in) * c * di * hi * wi +
                                           static_cast<size_t>(ic) * di * hi * wi +
                                           static_cast<size_t>(cur_d) * hi * wi +
                                           static_cast<size_t>(cur_h) * wi +
                                           static_cast<size_t>(cur_w);
                            size_t o_idx = static_cast<size_t>(in) * k * do_ * ho * wo +
                                           static_cast<size_t>(ido) * ho * wo +
                                           static_cast<size_t>(iho) * wo + static_cast<size_t>(iwo);
                            value += static_cast<double>(__bfloat16_to_float(p_in[i_idx])) *
                                     static_cast<double>(__bfloat16_to_float(p_out[o_idx]));
                        }
                    }
                }
            }
        }
        size_t f_idx = static_cast<size_t>(ic) * fz * fy * fx + static_cast<size_t>(iz) * fy * fx +
                       static_cast<size_t>(iy) * fx + static_cast<size_t>(ix);
        p_wei[f_idx] = __float_to_bfloat16(static_cast<float>(value));
    }
}

/***************************** nhwc *****************************/
// design block_size 256
extern "C" __global__ void naive_conv_fwd_nhwc_fp32(const float* __restrict__ p_in,
                                                    const float* __restrict__ p_wei,
                                                    float* __restrict__ p_out,
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
                                                    int group)
{
    /*
     *  need to compute total output pixel: `group * n * ho * wo * k_per_group`.
     *  to distribute this workload, let one workgroup compute `wo *
     * k_per_group` pixel,
     *  hence need `group * n * ho` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = wo * k_per_group;
    int bid           = blockIdx.x;
    int iho           = bid % ho;
    int in            = (bid / ho) % n;
    int ig            = bid / (n * ho);

    p_in += static_cast<size_t>(in) * hi * wi * c + static_cast<size_t>(ig) * c_per_group;
    p_wei += static_cast<size_t>(ig) * k_per_group * fy * fx * c_per_group;
    p_out += static_cast<size_t>(in) * ho * wo * k + static_cast<size_t>(ig) * k_per_group +
             static_cast<size_t>(iho) * wo * k;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int iwo = tid / k_per_group;
        int ik  = tid % k_per_group;

        double value = .0f;

        for(int iy = 0; iy < fy; iy++)
        {
            int valid_h = 1;
            int cur_h   = sy * iho - py + dy * iy;
            if(cur_h < 0 || cur_h >= hi)
                valid_h &= 0;
            for(int ix = 0; ix < fx; ix++)
            {
                for(int ic = 0; ic < c_per_group; ic++)
                {
                    int valid_w = 1;
                    int cur_w   = sx * iwo - px + dx * ix;
                    if(cur_w < 0 || cur_w >= wi)
                        valid_w &= 0;

                    if(valid_w & valid_h)
                    {
                        size_t i_idx = static_cast<size_t>(cur_h) * wi * c +
                                       static_cast<size_t>(cur_w) * c + static_cast<size_t>(ic);
                        size_t f_idx = static_cast<size_t>(ik) * fy * fx * c_per_group +
                                       static_cast<size_t>(iy) * fx * c_per_group +
                                       static_cast<size_t>(ix) * c_per_group +
                                       static_cast<size_t>(ic);
                        value +=
                            static_cast<double>(p_in[i_idx]) * static_cast<double>(p_wei[f_idx]);
                    }
                }
            }
        }
        size_t o_idx = static_cast<size_t>(iwo) * k + static_cast<size_t>(ik);
        p_out[o_idx] = static_cast<float>(value);
    }
}

extern "C" __global__ void naive_conv_bwd_nhwc_fp32(float* __restrict__ p_in,
                                                    const float* __restrict__ p_wei,
                                                    const float* __restrict__ p_out,
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
                                                    int group)
{
    /*
     *  need to compute total input pixel: `group * n * hi * wi * c_per_group`.
     *  to distribute this workload, let one workgroup compute `wi *
     * c_per_group` pixel,
     *  hence need `group * n * hi` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = wi * c_per_group;
    int bid           = blockIdx.x;
    int ihi           = bid % hi;
    int in            = (bid / hi) % n;
    int ig            = bid / (n * hi);

    p_in += static_cast<size_t>(in) * hi * wi * c + static_cast<size_t>(ihi) * wi * c +
            static_cast<size_t>(ig) * c_per_group;
    p_wei += static_cast<size_t>(ig) * k_per_group * fy * fx * c_per_group;
    p_out += static_cast<size_t>(in) * ho * wo * k + static_cast<size_t>(ig) * k_per_group;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int iwi = tid / c_per_group;
        int ic  = tid % c_per_group;

        double value = .0f;

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
                for(int ik = 0; ik < k_per_group; ik++)
                {

                    if(valid_h & valid_w)
                    {
                        size_t o_idx = static_cast<size_t>(cur_ho) * wo * k +
                                       static_cast<size_t>(cur_wo) * k + static_cast<size_t>(ik);
                        size_t f_idx = static_cast<size_t>(ik) * fy * fx * c_per_group +
                                       static_cast<size_t>(iy) * fx * c_per_group +
                                       static_cast<size_t>(ix) * c_per_group +
                                       static_cast<size_t>(ic);
                        value +=
                            static_cast<double>(p_out[o_idx]) * static_cast<double>(p_wei[f_idx]);
                    }
                }
            }
        }
        size_t i_idx = static_cast<size_t>(iwi) * c + static_cast<size_t>(ic);
        p_in[i_idx]  = static_cast<float>(value);
    }
}

extern "C" __global__ void naive_conv_wrw_nhwc_fp32(const float* __restrict__ p_in,
                                                    float* __restrict__ p_wei,
                                                    const float* __restrict__ p_out,
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
                                                    int group)
{
    /*
     *  need to compute total filter pixel: `group * k_per_group * fy * fx *
     * c_per_group`.
     *  to distribute this workload, let one workgroup compute `fy * fx *
     * c_per_group` pixel,
     *  hence need `group * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = c_per_group * fy * fx;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int ig            = bid / k_per_group;

    p_in += static_cast<size_t>(ig) * c_per_group;
    p_wei += static_cast<size_t>(ig) * k_per_group * fy * fx * c_per_group +
             static_cast<size_t>(ik) * fy * fx * c_per_group;
    p_out += static_cast<size_t>(ig) * k_per_group + static_cast<size_t>(ik);

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ic = tid % c_per_group;
        int ix = (tid / c_per_group) % fx;
        int iy = tid / (c_per_group * fx);

        double value = .0f;

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
                        size_t i_idx = static_cast<size_t>(in) * hi * wi * c +
                                       static_cast<size_t>(cur_h) * wi * c +
                                       static_cast<size_t>(cur_w) * c + static_cast<size_t>(ic);
                        size_t o_idx = static_cast<size_t>(in) * ho * wo * k +
                                       static_cast<size_t>(iho) * wo * k +
                                       static_cast<size_t>(iwo) * k;
                        value +=
                            static_cast<double>(p_in[i_idx]) * static_cast<double>(p_out[o_idx]);
                    }
                }
            }
        }
        size_t f_idx = static_cast<size_t>(iy) * fx * c_per_group +
                       static_cast<size_t>(ix) * c_per_group + static_cast<size_t>(ic);
        p_wei[f_idx] = static_cast<float>(value);
    }
}

// design block_size 256
extern "C" __global__ void naive_conv_fwd_ndhwc_fp32(const float* __restrict__ p_in,
                                                     const float* __restrict__ p_wei,
                                                     float* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total output pixel: `group * n * do_ * ho * wo *
     * k_per_group`.
     *  to distribute this workload, let one workgroup compute `ho * wo *
     * k_per_group` pixel,
     *  hence need `group * n * do_` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = ho * wo * k_per_group;
    int bid           = blockIdx.x;
    int ido           = bid % do_;
    int in            = (bid / do_) % n;
    int ig            = bid / (n * do_);

    p_in += static_cast<size_t>(in) * di * hi * wi * c + static_cast<size_t>(ig) * c_per_group;
    p_wei += static_cast<size_t>(ig) * k_per_group * fz * fy * fx * c_per_group;
    p_out += static_cast<size_t>(in) * do_ * ho * wo * k + static_cast<size_t>(ido) * ho * wo * k +
             static_cast<size_t>(ig) * k_per_group;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ik  = tid % k_per_group;
        int iwo = (tid / k_per_group) % wo;
        int iho = tid / (k_per_group * wo);

        double value = .0f;

        for(int iz = 0; iz < fz; iz++)
        {
            int valid_d = 1;
            int cur_d   = sz * ido - pz + dz * iz;
            if(cur_d < 0 || cur_d >= di)
                valid_d &= 0;
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
                    for(int ic = 0; ic < c_per_group; ic++)
                    {
                        if(valid_d & valid_w & valid_h)
                        {
                            size_t i_idx = static_cast<size_t>(cur_d) * hi * wi * c +
                                           static_cast<size_t>(cur_h) * wi * c +
                                           static_cast<size_t>(cur_w) * c + static_cast<size_t>(ic);
                            size_t f_idx = static_cast<size_t>(ik) * fz * fy * fx * c_per_group +
                                           static_cast<size_t>(iz) * fy * fx * c_per_group +
                                           static_cast<size_t>(iy) * fx * c_per_group +
                                           static_cast<size_t>(ix) * c_per_group +
                                           static_cast<size_t>(ic);
                            value += static_cast<double>(p_in[i_idx]) *
                                     static_cast<double>(p_wei[f_idx]);
                        }
                    }
                }
            }
        }
        size_t o_idx = static_cast<size_t>(iho) * wo * k + static_cast<size_t>(iwo) * k +
                       static_cast<size_t>(ik);
        p_out[o_idx] = static_cast<float>(value);
    }
}

extern "C" __global__ void naive_conv_bwd_ndhwc_fp32(float* __restrict__ p_in,
                                                     const float* __restrict__ p_wei,
                                                     const float* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total input pixel: `group * n * di * hi * wi *
     * c_per_group`.
     *  to distribute this workload, let one workgroup compute `hi * wi *
     * c_per_group` pixel,
     *  hence need `group * n * di` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = hi * wi * c_per_group;
    int bid           = blockIdx.x;
    int idi           = bid % di;
    int in            = (bid / di) % n;
    int ig            = bid / (n * di);

    p_in += static_cast<size_t>(in) * di * hi * wi * c + static_cast<size_t>(idi) * hi * wi * c +
            static_cast<size_t>(ig) * c_per_group;
    p_wei += static_cast<size_t>(ig) * k_per_group * fz * fy * fx * c_per_group;
    p_out += static_cast<size_t>(in) * do_ * ho * wo * k + static_cast<size_t>(ig) * k_per_group;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ic  = tid % c_per_group;
        int iwi = (tid / c_per_group) % wi;
        int ihi = (tid / (c_per_group * wi));

        double value = .0f;

        for(int iz = 0; iz < fz; iz++)
        {
            int valid_d = 1;
            int cur_do  = idi + pz - dz * iz;
            if(cur_do < 0 || cur_do % sz)
                valid_d &= 0;
            cur_do /= sz;
            if(cur_do >= do_)
                valid_d &= 0;
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
                    for(int ik = 0; ik < k_per_group; ik++)
                    {
                        if(valid_d & valid_h & valid_w)
                        {
                            size_t o_idx = static_cast<size_t>(cur_do) * ho * wo * k +
                                           static_cast<size_t>(cur_ho) * wo * k +
                                           static_cast<size_t>(cur_wo) * k +
                                           static_cast<size_t>(ik);
                            size_t f_idx = static_cast<size_t>(ik) * fz * fy * fx * c_per_group +
                                           static_cast<size_t>(iz) * fy * fx * c_per_group +
                                           static_cast<size_t>(iy) * fx * c_per_group +
                                           static_cast<size_t>(ix) * c_per_group +
                                           static_cast<size_t>(ic);
                            value += static_cast<double>(p_out[o_idx]) *
                                     static_cast<double>(p_wei[f_idx]);
                        }
                    }
                }
            }
        }
        size_t i_idx = static_cast<size_t>(ihi) * wi * c + static_cast<size_t>(iwi) * c +
                       static_cast<size_t>(ic);
        p_in[i_idx] = static_cast<float>(value);
    }
}

extern "C" __global__ void naive_conv_wrw_ndhwc_fp32(const float* __restrict__ p_in,
                                                     float* __restrict__ p_wei,
                                                     const float* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total filter pixel: `group * k_per_group * fz * fy * fx
     * * c_per_group`.
     *  to distribute this workload, let one workgroup compute `fz * fy * fx *
     * c_per_group` pixel,
     *  hence need `group * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = fz * fy * fx * c_per_group;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int ig            = bid / k_per_group;

    p_in += static_cast<size_t>(ig) * c_per_group;
    p_wei += static_cast<size_t>(ig) * k_per_group * fz * fy * fx * c_per_group +
             static_cast<size_t>(ik) * fz * fy * fx * c_per_group;
    p_out += static_cast<size_t>(ig) * k_per_group + static_cast<size_t>(ik);

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ic = tid % c_per_group;
        int ix = (tid / c_per_group) % fx;
        int iy = (tid / (c_per_group * fx)) % fy;
        int iz = (tid / (c_per_group * fx * fy));

        double value = .0f;

        for(int in = 0; in < n; in++)
        {
            for(int ido = 0; ido < do_; ido++)
            {
                int valid_d = 1;
                int cur_d   = sz * ido - pz + dz * iz;
                if(cur_d < 0 || cur_d >= di)
                    valid_d &= 0;
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

                        if(valid_d & valid_h & valid_w)
                        {
                            size_t i_idx = static_cast<size_t>(in) * di * hi * wi * c +
                                           static_cast<size_t>(cur_d) * hi * wi * c +
                                           static_cast<size_t>(cur_h) * wi * c +
                                           static_cast<size_t>(cur_w) * c + static_cast<size_t>(ic);
                            size_t o_idx = static_cast<size_t>(in) * do_ * ho * wo * k +
                                           static_cast<size_t>(ido) * ho * wo * k +
                                           static_cast<size_t>(iho) * wo * k +
                                           static_cast<size_t>(iwo) * k;
                            value += static_cast<double>(p_in[i_idx]) *
                                     static_cast<double>(p_out[o_idx]);
                        }
                    }
                }
            }
        }
        size_t f_idx = static_cast<size_t>(iz) * fy * fx * c_per_group +
                       static_cast<size_t>(iy) * fx * c_per_group +
                       static_cast<size_t>(ix) * c_per_group + static_cast<size_t>(ic);
        p_wei[f_idx] = static_cast<float>(value);
    }
}

// design block_size 256
extern "C" __global__ void naive_conv_fwd_nhwc_fp16(const half* __restrict__ p_in,
                                                    const half* __restrict__ p_wei,
                                                    half* __restrict__ p_out,
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
                                                    int group)
{
    /*
     *  need to compute total output pixel: `group * n * ho * wo * k_per_group`.
     *  to distribute this workload, let one workgroup compute `wo *
     * k_per_group` pixel,
     *  hence need `group * n * ho` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = wo * k_per_group;
    int bid           = blockIdx.x;
    int iho           = bid % ho;
    int in            = (bid / ho) % n;
    int ig            = bid / (n * ho);

    p_in += static_cast<size_t>(in) * hi * wi * c + static_cast<size_t>(ig) * c_per_group;
    p_wei += static_cast<size_t>(ig) * k_per_group * fy * fx * c_per_group;
    p_out += static_cast<size_t>(in) * ho * wo * k + static_cast<size_t>(ig) * k_per_group +
             static_cast<size_t>(iho) * wo * k;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int iwo = tid / k_per_group;
        int ik  = tid % k_per_group;

        double value = .0f;

        for(int iy = 0; iy < fy; iy++)
        {
            int valid_h = 1;
            int cur_h   = sy * iho - py + dy * iy;
            if(cur_h < 0 || cur_h >= hi)
                valid_h &= 0;
            for(int ix = 0; ix < fx; ix++)
            {
                for(int ic = 0; ic < c_per_group; ic++)
                {
                    int valid_w = 1;
                    int cur_w   = sx * iwo - px + dx * ix;
                    if(cur_w < 0 || cur_w >= wi)
                        valid_w &= 0;

                    if(valid_w & valid_h)
                    {
                        size_t i_idx = static_cast<size_t>(cur_h) * wi * c +
                                       static_cast<size_t>(cur_w) * c + static_cast<size_t>(ic);
                        size_t f_idx = static_cast<size_t>(ik) * fy * fx * c_per_group +
                                       static_cast<size_t>(iy) * fx * c_per_group +
                                       static_cast<size_t>(ix) * c_per_group +
                                       static_cast<size_t>(ic);
                        value += static_cast<double>(__half2float(p_in[i_idx])) *
                                 static_cast<double>(__half2float(p_wei[f_idx]));
                    }
                }
            }
        }
        size_t o_idx = static_cast<size_t>(iwo) * k + static_cast<size_t>(ik);
        p_out[o_idx] = __float2half(static_cast<float>(value));
    }
}

extern "C" __global__ void naive_conv_bwd_nhwc_fp16(half* __restrict__ p_in,
                                                    const half* __restrict__ p_wei,
                                                    const half* __restrict__ p_out,
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
                                                    int group)
{
    /*
     *  need to compute total input pixel: `group * n * hi * wi * c_per_group`.
     *  to distribute this workload, let one workgroup compute `wi *
     * c_per_group` pixel,
     *  hence need `group * n * hi` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = wi * c_per_group;
    int bid           = blockIdx.x;
    int ihi           = bid % hi;
    int in            = (bid / hi) % n;
    int ig            = bid / (n * hi);

    p_in += static_cast<size_t>(in) * hi * wi * c + static_cast<size_t>(ihi) * wi * c +
            static_cast<size_t>(ig) * c_per_group;
    p_wei += static_cast<size_t>(ig) * k_per_group * fy * fx * c_per_group;
    p_out += static_cast<size_t>(in) * ho * wo * k + static_cast<size_t>(ig) * k_per_group;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int iwi = tid / c_per_group;
        int ic  = tid % c_per_group;

        double value = .0f;

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
                for(int ik = 0; ik < k_per_group; ik++)
                {

                    if(valid_h & valid_w)
                    {
                        size_t o_idx = static_cast<size_t>(cur_ho) * wo * k +
                                       static_cast<size_t>(cur_wo) * k + static_cast<size_t>(ik);
                        size_t f_idx = static_cast<size_t>(ik) * fy * fx * c_per_group +
                                       static_cast<size_t>(iy) * fx * c_per_group +
                                       static_cast<size_t>(ix) * c_per_group +
                                       static_cast<size_t>(ic);
                        value += static_cast<double>(__half2float(p_out[o_idx])) *
                                 static_cast<double>(__half2float(p_wei[f_idx]));
                    }
                }
            }
        }
        size_t i_idx = static_cast<size_t>(iwi) * c + static_cast<size_t>(ic);
        p_in[i_idx]  = __float2half(static_cast<float>(value));
    }
}

extern "C" __global__ void naive_conv_wrw_nhwc_fp16(const half* __restrict__ p_in,
                                                    half* __restrict__ p_wei,
                                                    const half* __restrict__ p_out,
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
                                                    int group)
{
    /*
     *  need to compute total filter pixel: `group * k_per_group * fy * fx *
     * c_per_group`.
     *  to distribute this workload, let one workgroup compute `fy * fx *
     * c_per_group` pixel,
     *  hence need `group * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = c_per_group * fy * fx;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int ig            = bid / k_per_group;

    p_in += static_cast<size_t>(ig) * c_per_group;
    p_wei += static_cast<size_t>(ig) * k_per_group * fy * fx * c_per_group +
             static_cast<size_t>(ik) * fy * fx * c_per_group;
    p_out += static_cast<size_t>(ig) * k_per_group + static_cast<size_t>(ik);

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ic = tid % c_per_group;
        int ix = (tid / c_per_group) % fx;
        int iy = tid / (c_per_group * fx);

        double value = .0f;

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
                        size_t i_idx = static_cast<size_t>(in) * hi * wi * c +
                                       static_cast<size_t>(cur_h) * wi * c +
                                       static_cast<size_t>(cur_w) * c + static_cast<size_t>(ic);
                        size_t o_idx = static_cast<size_t>(in) * ho * wo * k +
                                       static_cast<size_t>(iho) * wo * k +
                                       static_cast<size_t>(iwo) * k;
                        value += static_cast<double>(__half2float(p_in[i_idx])) *
                                 static_cast<double>(__half2float(p_out[o_idx]));
                    }
                }
            }
        }
        size_t f_idx = static_cast<size_t>(iy) * fx * c_per_group +
                       static_cast<size_t>(ix) * c_per_group + static_cast<size_t>(ic);
        p_wei[f_idx] = __float2half(static_cast<float>(value));
    }
}

// design block_size 256
extern "C" __global__ void naive_conv_fwd_ndhwc_fp16(const half* __restrict__ p_in,
                                                     const half* __restrict__ p_wei,
                                                     half* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total output pixel: `group * n * do_ * ho * wo *
     * k_per_group`.
     *  to distribute this workload, let one workgroup compute `ho * wo *
     * k_per_group` pixel,
     *  hence need `group * n * do_` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = ho * wo * k_per_group;
    int bid           = blockIdx.x;
    int ido           = bid % do_;
    int in            = (bid / do_) % n;
    int ig            = bid / (n * do_);

    p_in += static_cast<size_t>(in) * di * hi * wi * c + static_cast<size_t>(ig) * c_per_group;
    p_wei += static_cast<size_t>(ig) * k_per_group * fz * fy * fx * c_per_group;
    p_out += static_cast<size_t>(in) * do_ * ho * wo * k + static_cast<size_t>(ido) * ho * wo * k +
             static_cast<size_t>(ig) * k_per_group;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ik  = tid % k_per_group;
        int iwo = (tid / k_per_group) % wo;
        int iho = tid / (k_per_group * wo);

        double value = .0f;

        for(int iz = 0; iz < fz; iz++)
        {
            int valid_d = 1;
            int cur_d   = sz * ido - pz + dz * iz;
            if(cur_d < 0 || cur_d >= di)
                valid_d &= 0;
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
                    for(int ic = 0; ic < c_per_group; ic++)
                    {
                        if(valid_d & valid_w & valid_h)
                        {
                            size_t i_idx = static_cast<size_t>(cur_d) * hi * wi * c +
                                           static_cast<size_t>(cur_h) * wi * c +
                                           static_cast<size_t>(cur_w) * c + static_cast<size_t>(ic);
                            size_t f_idx = static_cast<size_t>(ik) * fz * fy * fx * c_per_group +
                                           static_cast<size_t>(iz) * fy * fx * c_per_group +
                                           static_cast<size_t>(iy) * fx * c_per_group +
                                           static_cast<size_t>(ix) * c_per_group +
                                           static_cast<size_t>(ic);
                            value += static_cast<double>(__half2float(p_in[i_idx])) *
                                     static_cast<double>(__half2float(p_wei[f_idx]));
                        }
                    }
                }
            }
        }
        size_t o_idx = static_cast<size_t>(iho) * wo * k + static_cast<size_t>(iwo) * k +
                       static_cast<size_t>(ik);
        p_out[o_idx] = __float2half(static_cast<float>(value));
    }
}

extern "C" __global__ void naive_conv_bwd_ndhwc_fp16(half* __restrict__ p_in,
                                                     const half* __restrict__ p_wei,
                                                     const half* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total input pixel: `group * n * di * hi * wi *
     * c_per_group`.
     *  to distribute this workload, let one workgroup compute `hi * wi *
     * c_per_group` pixel,
     *  hence need `group * n * di` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = hi * wi * c_per_group;
    int bid           = blockIdx.x;
    int idi           = bid % di;
    int in            = (bid / di) % n;
    int ig            = bid / (n * di);

    p_in += static_cast<size_t>(in) * di * hi * wi * c + static_cast<size_t>(idi) * hi * wi * c +
            static_cast<size_t>(ig) * c_per_group;
    p_wei += static_cast<size_t>(ig) * k_per_group * fz * fy * fx * c_per_group;
    p_out += static_cast<size_t>(in) * do_ * ho * wo * k + static_cast<size_t>(ig) * k_per_group;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ic  = tid % c_per_group;
        int iwi = (tid / c_per_group) % wi;
        int ihi = (tid / (c_per_group * wi));

        double value = .0f;

        for(int iz = 0; iz < fz; iz++)
        {
            int valid_d = 1;
            int cur_do  = idi + pz - dz * iz;
            if(cur_do < 0 || cur_do % sz)
                valid_d &= 0;
            cur_do /= sz;
            if(cur_do >= do_)
                valid_d &= 0;
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
                    for(int ik = 0; ik < k_per_group; ik++)
                    {
                        if(valid_d & valid_h & valid_w)
                        {
                            size_t o_idx = static_cast<size_t>(cur_do) * ho * wo * k +
                                           static_cast<size_t>(cur_ho) * wo * k +
                                           static_cast<size_t>(cur_wo) * k +
                                           static_cast<size_t>(ik);
                            size_t f_idx = static_cast<size_t>(ik) * fz * fy * fx * c_per_group +
                                           static_cast<size_t>(iz) * fy * fx * c_per_group +
                                           static_cast<size_t>(iy) * fx * c_per_group +
                                           static_cast<size_t>(ix) * c_per_group +
                                           static_cast<size_t>(ic);
                            value += static_cast<double>(__half2float(p_out[o_idx])) *
                                     static_cast<double>(__half2float(p_wei[f_idx]));
                        }
                    }
                }
            }
        }
        size_t i_idx = static_cast<size_t>(ihi) * wi * c + static_cast<size_t>(iwi) * c +
                       static_cast<size_t>(ic);
        p_in[i_idx] = __float2half(static_cast<float>(value));
    }
}

extern "C" __global__ void naive_conv_wrw_ndhwc_fp16(const half* __restrict__ p_in,
                                                     half* __restrict__ p_wei,
                                                     const half* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total filter pixel: `group * k_per_group * fz * fy * fx
     * * c_per_group`.
     *  to distribute this workload, let one workgroup compute `fz * fy * fx *
     * c_per_group` pixel,
     *  hence need `group * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = fz * fy * fx * c_per_group;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int ig            = bid / k_per_group;

    p_in += static_cast<size_t>(ig) * c_per_group;
    p_wei += static_cast<size_t>(ig) * k_per_group * fz * fy * fx * c_per_group +
             static_cast<size_t>(ik) * fz * fy * fx * c_per_group;
    p_out += static_cast<size_t>(ig) * k_per_group + static_cast<size_t>(ik);

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ic = tid % c_per_group;
        int ix = (tid / c_per_group) % fx;
        int iy = (tid / (c_per_group * fx)) % fy;
        int iz = (tid / (c_per_group * fx * fy));

        double value = .0f;

        for(int in = 0; in < n; in++)
        {
            for(int ido = 0; ido < do_; ido++)
            {
                int valid_d = 1;
                int cur_d   = sz * ido - pz + dz * iz;
                if(cur_d < 0 || cur_d >= di)
                    valid_d &= 0;
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

                        if(valid_d & valid_h & valid_w)
                        {
                            size_t i_idx = static_cast<size_t>(in) * di * hi * wi * c +
                                           static_cast<size_t>(cur_d) * hi * wi * c +
                                           static_cast<size_t>(cur_h) * wi * c +
                                           static_cast<size_t>(cur_w) * c + static_cast<size_t>(ic);
                            size_t o_idx = static_cast<size_t>(in) * do_ * ho * wo * k +
                                           static_cast<size_t>(ido) * ho * wo * k +
                                           static_cast<size_t>(iho) * wo * k +
                                           static_cast<size_t>(iwo) * k;
                            value += static_cast<double>(__half2float(p_in[i_idx])) *
                                     static_cast<double>(__half2float(p_out[o_idx]));
                        }
                    }
                }
            }
        }
        size_t f_idx = static_cast<size_t>(iz) * fy * fx * c_per_group +
                       static_cast<size_t>(iy) * fx * c_per_group +
                       static_cast<size_t>(ix) * c_per_group + static_cast<size_t>(ic);
        p_wei[f_idx] = __float2half(static_cast<float>(value));
    }
}

// design block_size 256
extern "C" __global__ void naive_conv_fwd_nhwc_bf16(const ushort* __restrict__ p_in,
                                                    const ushort* __restrict__ p_wei,
                                                    ushort* __restrict__ p_out,
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
                                                    int group)
{
    /*
     *  need to compute total output pixel: `group * n * ho * wo * k_per_group`.
     *  to distribute this workload, let one workgroup compute `wo *
     * k_per_group` pixel,
     *  hence need `group * n * ho` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = wo * k_per_group;
    int bid           = blockIdx.x;
    int iho           = bid % ho;
    int in            = (bid / ho) % n;
    int ig            = bid / (n * ho);

    p_in += static_cast<size_t>(in) * hi * wi * c + static_cast<size_t>(ig) * c_per_group;
    p_wei += static_cast<size_t>(ig) * k_per_group * fy * fx * c_per_group;
    p_out += static_cast<size_t>(in) * ho * wo * k + static_cast<size_t>(ig) * k_per_group +
             static_cast<size_t>(iho) * wo * k;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int iwo = tid / k_per_group;
        int ik  = tid % k_per_group;

        double value = .0f;

        for(int iy = 0; iy < fy; iy++)
        {
            int valid_h = 1;
            int cur_h   = sy * iho - py + dy * iy;
            if(cur_h < 0 || cur_h >= hi)
                valid_h &= 0;
            for(int ix = 0; ix < fx; ix++)
            {
                for(int ic = 0; ic < c_per_group; ic++)
                {
                    int valid_w = 1;
                    int cur_w   = sx * iwo - px + dx * ix;
                    if(cur_w < 0 || cur_w >= wi)
                        valid_w &= 0;

                    if(valid_w & valid_h)
                    {
                        size_t i_idx = static_cast<size_t>(cur_h) * wi * c +
                                       static_cast<size_t>(cur_w) * c + static_cast<size_t>(ic);
                        size_t f_idx = static_cast<size_t>(ik) * fy * fx * c_per_group +
                                       static_cast<size_t>(iy) * fx * c_per_group +
                                       static_cast<size_t>(ix) * c_per_group +
                                       static_cast<size_t>(ic);
                        value += static_cast<double>(__bfloat16_to_float(p_in[i_idx])) *
                                 static_cast<double>(__bfloat16_to_float(p_wei[f_idx]));
                    }
                }
            }
        }
        size_t o_idx = static_cast<size_t>(iwo) * k + static_cast<size_t>(ik);
        p_out[o_idx] = __float_to_bfloat16(static_cast<float>(value));
    }
}

extern "C" __global__ void naive_conv_bwd_nhwc_bf16(ushort* __restrict__ p_in,
                                                    const ushort* __restrict__ p_wei,
                                                    const ushort* __restrict__ p_out,
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
                                                    int group)
{
    /*
     *  need to compute total input pixel: `group * n * hi * wi * c_per_group`.
     *  to distribute this workload, let one workgroup compute `wi *
     * c_per_group` pixel,
     *  hence need `group * n * hi` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = wi * c_per_group;
    int bid           = blockIdx.x;
    int ihi           = bid % hi;
    int in            = (bid / hi) % n;
    int ig            = bid / (n * hi);

    p_in += static_cast<size_t>(in) * hi * wi * c + static_cast<size_t>(ihi) * wi * c +
            static_cast<size_t>(ig) * c_per_group;
    p_wei += static_cast<size_t>(ig) * k_per_group * fy * fx * c_per_group;
    p_out += static_cast<size_t>(in) * ho * wo * k + static_cast<size_t>(ig) * k_per_group;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int iwi = tid / c_per_group;
        int ic  = tid % c_per_group;

        double value = .0f;

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
                for(int ik = 0; ik < k_per_group; ik++)
                {

                    if(valid_h & valid_w)
                    {
                        size_t o_idx = static_cast<size_t>(cur_ho) * wo * k +
                                       static_cast<size_t>(cur_wo) * k + static_cast<size_t>(ik);
                        size_t f_idx = static_cast<size_t>(ik) * fy * fx * c_per_group +
                                       static_cast<size_t>(iy) * fx * c_per_group +
                                       static_cast<size_t>(ix) * c_per_group +
                                       static_cast<size_t>(ic);
                        value += static_cast<double>(__bfloat16_to_float(p_out[o_idx])) *
                                 static_cast<double>(__bfloat16_to_float(p_wei[f_idx]));
                    }
                }
            }
        }
        size_t i_idx = static_cast<size_t>(iwi) * c + static_cast<size_t>(ic);
        p_in[i_idx]  = __float_to_bfloat16(static_cast<float>(value));
    }
}

extern "C" __global__ void naive_conv_wrw_nhwc_bf16(const ushort* __restrict__ p_in,
                                                    ushort* __restrict__ p_wei,
                                                    const ushort* __restrict__ p_out,
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
                                                    int group)
{
    /*
     *  need to compute total filter pixel: `group * k_per_group * fy * fx *
     * c_per_group`.
     *  to distribute this workload, let one workgroup compute `fy * fx *
     * c_per_group` pixel,
     *  hence need `group * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = c_per_group * fy * fx;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int ig            = bid / k_per_group;

    p_in += static_cast<size_t>(ig) * c_per_group;
    p_wei += static_cast<size_t>(ig) * k_per_group * fy * fx * c_per_group +
             static_cast<size_t>(ik) * fy * fx * c_per_group;
    p_out += static_cast<size_t>(ig) * k_per_group + static_cast<size_t>(ik);

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ic = tid % c_per_group;
        int ix = (tid / c_per_group) % fx;
        int iy = tid / (c_per_group * fx);

        double value = .0f;

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
                        size_t i_idx = static_cast<size_t>(in) * hi * wi * c +
                                       static_cast<size_t>(cur_h) * wi * c +
                                       static_cast<size_t>(cur_w) * c + static_cast<size_t>(ic);
                        size_t o_idx = static_cast<size_t>(in) * ho * wo * k +
                                       static_cast<size_t>(iho) * wo * k +
                                       static_cast<size_t>(iwo) * k;
                        value += static_cast<double>(__bfloat16_to_float(p_in[i_idx])) *
                                 static_cast<double>(__bfloat16_to_float(p_out[o_idx]));
                    }
                }
            }
        }
        size_t f_idx = static_cast<size_t>(iy) * fx * c_per_group +
                       static_cast<size_t>(ix) * c_per_group + static_cast<size_t>(ic);
        p_wei[f_idx] = __float_to_bfloat16(static_cast<float>(value));
    }
}

// design block_size 256
extern "C" __global__ void naive_conv_fwd_ndhwc_bf16(const ushort* __restrict__ p_in,
                                                     const ushort* __restrict__ p_wei,
                                                     ushort* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total output pixel: `group * n * do_ * ho * wo *
     * k_per_group`.
     *  to distribute this workload, let one workgroup compute `ho * wo *
     * k_per_group` pixel,
     *  hence need `group * n * do_` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = ho * wo * k_per_group;
    int bid           = blockIdx.x;
    int ido           = bid % do_;
    int in            = (bid / do_) % n;
    int ig            = bid / (n * do_);

    p_in += static_cast<size_t>(in) * di * hi * wi * c + static_cast<size_t>(ig) * c_per_group;
    p_wei += static_cast<size_t>(ig) * k_per_group * fz * fy * fx * c_per_group;
    p_out += static_cast<size_t>(in) * do_ * ho * wo * k + static_cast<size_t>(ido) * ho * wo * k +
             static_cast<size_t>(ig) * k_per_group;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ik  = tid % k_per_group;
        int iwo = (tid / k_per_group) % wo;
        int iho = tid / (k_per_group * wo);

        double value = .0f;

        for(int iz = 0; iz < fz; iz++)
        {
            int valid_d = 1;
            int cur_d   = sz * ido - pz + dz * iz;
            if(cur_d < 0 || cur_d >= di)
                valid_d &= 0;
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
                    for(int ic = 0; ic < c_per_group; ic++)
                    {
                        if(valid_d & valid_w & valid_h)
                        {
                            size_t i_idx = static_cast<size_t>(cur_d) * hi * wi * c +
                                           static_cast<size_t>(cur_h) * wi * c +
                                           static_cast<size_t>(cur_w) * c + static_cast<size_t>(ic);
                            size_t f_idx = static_cast<size_t>(ik) * fz * fy * fx * c_per_group +
                                           static_cast<size_t>(iz) * fy * fx * c_per_group +
                                           static_cast<size_t>(iy) * fx * c_per_group +
                                           static_cast<size_t>(ix) * c_per_group +
                                           static_cast<size_t>(ic);
                            value += static_cast<double>(__bfloat16_to_float(p_in[i_idx])) *
                                     static_cast<double>(__bfloat16_to_float(p_wei[f_idx]));
                        }
                    }
                }
            }
        }
        size_t o_idx = static_cast<size_t>(iho) * wo * k + static_cast<size_t>(iwo) * k +
                       static_cast<size_t>(ik);
        p_out[o_idx] = __float_to_bfloat16(static_cast<float>(value));
    }
}

extern "C" __global__ void naive_conv_bwd_ndhwc_bf16(ushort* __restrict__ p_in,
                                                     const ushort* __restrict__ p_wei,
                                                     const ushort* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total input pixel: `group * n * di * hi * wi *
     * c_per_group`.
     *  to distribute this workload, let one workgroup compute `hi * wi *
     * c_per_group` pixel,
     *  hence need `group * n * di` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = hi * wi * c_per_group;
    int bid           = blockIdx.x;
    int idi           = bid % di;
    int in            = (bid / di) % n;
    int ig            = bid / (n * di);

    p_in += static_cast<size_t>(in) * di * hi * wi * c + static_cast<size_t>(idi) * hi * wi * c +
            static_cast<size_t>(ig) * c_per_group;
    p_wei += static_cast<size_t>(ig) * k_per_group * fz * fy * fx * c_per_group;
    p_out += static_cast<size_t>(in) * do_ * ho * wo * k + static_cast<size_t>(ig) * k_per_group;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ic  = tid % c_per_group;
        int iwi = (tid / c_per_group) % wi;
        int ihi = (tid / (c_per_group * wi));

        double value = .0f;

        for(int iz = 0; iz < fz; iz++)
        {
            int valid_d = 1;
            int cur_do  = idi + pz - dz * iz;
            if(cur_do < 0 || cur_do % sz)
                valid_d &= 0;
            cur_do /= sz;
            if(cur_do >= do_)
                valid_d &= 0;
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
                    for(int ik = 0; ik < k_per_group; ik++)
                    {
                        if(valid_d & valid_h & valid_w)
                        {
                            size_t o_idx = static_cast<size_t>(cur_do) * ho * wo * k +
                                           static_cast<size_t>(cur_ho) * wo * k +
                                           static_cast<size_t>(cur_wo) * k +
                                           static_cast<size_t>(ik);
                            size_t f_idx = static_cast<size_t>(ik) * fz * fy * fx * c_per_group +
                                           static_cast<size_t>(iz) * fy * fx * c_per_group +
                                           static_cast<size_t>(iy) * fx * c_per_group +
                                           static_cast<size_t>(ix) * c_per_group +
                                           static_cast<size_t>(ic);
                            value += static_cast<double>(__bfloat16_to_float(p_out[o_idx])) *
                                     static_cast<double>(__bfloat16_to_float(p_wei[f_idx]));
                        }
                    }
                }
            }
        }
        size_t i_idx = static_cast<size_t>(ihi) * wi * c + static_cast<size_t>(iwi) * c +
                       static_cast<size_t>(ic);
        p_in[i_idx] = __float_to_bfloat16(static_cast<float>(value));
    }
}

extern "C" __global__ void naive_conv_wrw_ndhwc_bf16(const ushort* __restrict__ p_in,
                                                     ushort* __restrict__ p_wei,
                                                     const ushort* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total filter pixel: `group * k_per_group * fz * fy * fx
     * * c_per_group`.
     *  to distribute this workload, let one workgroup compute `fz * fy * fx *
     * c_per_group` pixel,
     *  hence need `group * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = fz * fy * fx * c_per_group;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int ig            = bid / k_per_group;

    p_in += static_cast<size_t>(ig) * c_per_group;
    p_wei += static_cast<size_t>(ig) * k_per_group * fz * fy * fx * c_per_group +
             static_cast<size_t>(ik) * fz * fy * fx * c_per_group;
    p_out += static_cast<size_t>(ig) * k_per_group + static_cast<size_t>(ik);

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ic = tid % c_per_group;
        int ix = (tid / c_per_group) % fx;
        int iy = (tid / (c_per_group * fx)) % fy;
        int iz = (tid / (c_per_group * fx * fy));

        double value = .0f;

        for(int in = 0; in < n; in++)
        {
            for(int ido = 0; ido < do_; ido++)
            {
                int valid_d = 1;
                int cur_d   = sz * ido - pz + dz * iz;
                if(cur_d < 0 || cur_d >= di)
                    valid_d &= 0;
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

                        if(valid_d & valid_h & valid_w)
                        {
                            size_t i_idx = static_cast<size_t>(in) * di * hi * wi * c +
                                           static_cast<size_t>(cur_d) * hi * wi * c +
                                           static_cast<size_t>(cur_h) * wi * c +
                                           static_cast<size_t>(cur_w) * c + static_cast<size_t>(ic);
                            size_t o_idx = static_cast<size_t>(in) * do_ * ho * wo * k +
                                           static_cast<size_t>(ido) * ho * wo * k +
                                           static_cast<size_t>(iho) * wo * k +
                                           static_cast<size_t>(iwo) * k;
                            value += static_cast<double>(__bfloat16_to_float(p_in[i_idx])) *
                                     static_cast<double>(__bfloat16_to_float(p_out[o_idx]));
                        }
                    }
                }
            }
        }
        size_t f_idx = static_cast<size_t>(iz) * fy * fx * c_per_group +
                       static_cast<size_t>(iy) * fx * c_per_group +
                       static_cast<size_t>(ix) * c_per_group + static_cast<size_t>(ic);
        p_wei[f_idx] = __float_to_bfloat16(static_cast<float>(value));
    }
}
