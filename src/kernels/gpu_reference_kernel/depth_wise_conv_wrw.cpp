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
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#endif

#include "miopen_cstdint.hpp"
#include "miopen_limits.hpp"
// #include "miopen/conv/solvers.hpp"
// #include "../include/miopen/solver/implicitgemm_util.hpp"

#include "stride_array.hpp"

// hcc seems need __device__ __host__ together to compile, and no extern "C"
typedef union value_bf16_fp32_t
{
    uint u32;
    ushort2 ushortx2;
    ushort ushortvec[2];
    float f32;
} value_bf16_fp32_t;

inline __device__ __host__ float convert_bf16_to_fp32(ushort src_val)
{
    value_bf16_fp32_t target_val;
    target_val.ushortx2 = make_ushort2(0, src_val);
    return target_val.f32;
}

inline __device__ __host__ ushort convert_fp32_to_bf16(float src_val)
{
    value_bf16_fp32_t target_val;
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

template <typename src_data_t, typename dst_data_t>
inline __device__ __host__ dst_data_t cast_to(const src_data_t& val)
{
    return static_cast<dst_data_t>(val);
}
template <>
inline __device__ __host__ ushort cast_to(const double& val)
{
    return convert_fp32_to_bf16(static_cast<float>(val));
}
template <>
inline __device__ __host__ double cast_to(const ushort& val)
{
    return static_cast<double>(convert_bf16_to_fp32(val));
}
template <>
inline __device__ __host__ half cast_to(const double& val)
{
    return __float2half(static_cast<float>(val));
}
template <>
inline __device__ __host__ double cast_to(const half& val)
{
    return static_cast<double>(__half2float(val));
}
template <>
inline __device__ __host__ int8_t cast_to(const int32_t& val)
{
    return static_cast<int8_t>(val & 0xff);
}

template <typename X, typename Y>
__host__ __device__ constexpr auto integer_divide_ceil(X x, Y y)
{
    return (x + y - 1) / y;
}

inline __device__ __host__ bool IsZero(double val) { return val == 0.0; }

inline __device__ __host__ bool IsOne(double val) { return val == 1.0; }

template <typename dst_data_t, typename acc_data_t>
inline __device__ void applyalphaBetaUpdate(dst_data_t* __restrict__ p_array,
                                            const acc_data_t value,
                                            double alpha,
                                            double beta,
                                            size_t index)
{
    if(IsOne(alpha) && IsZero(beta))
    {
        p_array[index] = cast_to<acc_data_t, dst_data_t>(value);
        return;
    }
    // cast_to<src, dst>
    acc_data_t val_alpha_beta =
        cast_to<double, acc_data_t>(alpha) * value +
        cast_to<dst_data_t, acc_data_t>(p_array[index]) * cast_to<double, acc_data_t>(beta);
    p_array[index] = cast_to<acc_data_t, dst_data_t>(val_alpha_beta);
}

/// \todo remove template parameter 'bool ASSUME_PACKED' in a follow up PR
/// --amberhassaan
/// Notes (Amber):
/// - The following code used to assume that group (G) is an implicit
/// dimension, i.e. c= c_per_group * group and k = k_per_group * group. This is not
/// true for non-packed case because group (G) dimension needs to have its stride
/// explicitly specified for address math to make sense. This is also how
/// composable_kernel (CK) treats G dimension. Which is why nchw should be ngchw,
/// and nhwc should be nhwgc. Same follows for the 3D case.
///
/// - strides here are stored right to left, i.e., for NHWC, stride for N is
/// at index 3 while stride for C is at index 0. This is different from how the
/// tensor descriptors store strides, which is always NCHW order, left-to-right.

/// alpha and beta are double to ensure high precision.

template <bool ASSUME_PACKED, typename src_data_t, typename acc_data_t, typename dst_data_t>
inline __device__ void naive_conv_fwd_nchw(const src_data_t* __restrict__ p_in,
                                           const src_data_t* __restrict__ p_wei,
                                           const double alpha,
                                           const double beta,
                                           dst_data_t* __restrict__ p_out,
                                           Strides5D in_strides,
                                           Strides5D wei_strides,
                                           Strides5D out_strides,
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

    if constexpr(ASSUME_PACKED)
    {
        p_in +=
            static_cast<size_t>(in) * c * hi * wi + static_cast<size_t>(ig) * c_per_group * hi * wi;

        p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fy * fx +
                 static_cast<size_t>(ik) * c_per_group * fy * fx;

        p_out += static_cast<size_t>(in) * k * ho * wo +
                 static_cast<size_t>(ig) * k_per_group * ho * wo +
                 static_cast<size_t>(ik) * ho * wo;
    }
    else
    {
        p_in += static_cast<size_t>(in) * in_strides[4] + static_cast<size_t>(ig) * in_strides[3];

        p_wei +=
            static_cast<size_t>(ig) * wei_strides[4] + static_cast<size_t>(ik) * wei_strides[3];

        p_out += static_cast<size_t>(in) * out_strides[4] +
                 static_cast<size_t>(ig) * out_strides[3] +
                 static_cast<size_t>(ik) * out_strides[2];
    }

    for(int tid = threadIdx.x; tid < thread_length; tid += blockDim.x)
    {
        int iho = tid / wo;
        int iwo = tid % wo;

        acc_data_t value = 0;

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
                        if constexpr(ASSUME_PACKED)
                        {
                            size_t i_idx = static_cast<size_t>(ic) * hi * wi +
                                           static_cast<size_t>(cur_h) * wi +
                                           static_cast<size_t>(cur_w);

                            size_t f_idx = static_cast<size_t>(ic) * fy * fx +
                                           static_cast<size_t>(iy) * fx + static_cast<size_t>(ix);

                            value += cast_to<src_data_t, acc_data_t>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t>(p_wei[f_idx]);
                        }
                        else
                        {
                            size_t i_idx = static_cast<size_t>(ic) * in_strides[2] +
                                           static_cast<size_t>(cur_h) * in_strides[1] +
                                           static_cast<size_t>(cur_w) * in_strides[0];

                            size_t f_idx = static_cast<size_t>(ic) * wei_strides[2] +
                                           static_cast<size_t>(iy) * wei_strides[1] +
                                           static_cast<size_t>(ix) * wei_strides[0];

                            value += cast_to<src_data_t, acc_data_t>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t>(p_wei[f_idx]);
                        }
                    }
                }
            }
        }
        if constexpr(ASSUME_PACKED)
        {
            size_t o_idx = static_cast<size_t>(iho) * wo + static_cast<size_t>(iwo);
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_out, value, alpha, beta, o_idx);
        }
        else
        {
            size_t o_idx = static_cast<size_t>(iho) * out_strides[1] +
                           static_cast<size_t>(iwo) * out_strides[0];
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_out, value, alpha, beta, o_idx);
        }
    }
}

template <bool ASSUME_PACKED, typename src_data_t, typename acc_data_t, typename dst_data_t>
inline __device__ void naive_conv_bwd_nchw(dst_data_t* __restrict__ p_in,
                                           const src_data_t* __restrict__ p_wei,
                                           const double alpha,
                                           const double beta,
                                           const src_data_t* __restrict__ p_out,
                                           Strides5D in_strides,
                                           Strides5D wei_strides,
                                           Strides5D out_strides,
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

    if constexpr(ASSUME_PACKED)
    {
        p_in += static_cast<size_t>(in) * c * hi * wi +
                static_cast<size_t>(ig) * c_per_group * hi * wi + static_cast<size_t>(ic) * hi * wi;

        p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fy * fx +
                 static_cast<size_t>(ic) * fy * fx;

        p_out +=
            static_cast<size_t>(in) * k * ho * wo + static_cast<size_t>(ig) * k_per_group * ho * wo;
    }
    else
    {
        p_in += static_cast<size_t>(in) * in_strides[4] + static_cast<size_t>(ig) * in_strides[3] +
                static_cast<size_t>(ic) * in_strides[2];

        p_wei +=
            static_cast<size_t>(ig) * wei_strides[4] + static_cast<size_t>(ic) * wei_strides[2];

        p_out +=
            static_cast<size_t>(in) * out_strides[4] + static_cast<size_t>(ig) * out_strides[3];
    }

    for(int tid = threadIdx.x; tid < thread_length; tid += blockDim.x)
    {
        int ihi = tid / wi;
        int iwi = tid % wi;

        acc_data_t value = 0;

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
                        if constexpr(ASSUME_PACKED)
                        {
                            size_t o_idx = static_cast<size_t>(ik) * ho * wo +
                                           static_cast<size_t>(cur_ho) * wo +
                                           static_cast<size_t>(cur_wo);

                            size_t f_idx = static_cast<size_t>(ik) * c_per_group * fy * fx +
                                           static_cast<size_t>(iy) * fx + static_cast<size_t>(ix);

                            value += cast_to<src_data_t, acc_data_t>(p_out[o_idx]) *
                                     cast_to<src_data_t, acc_data_t>(p_wei[f_idx]);
                        }
                        else
                        {
                            size_t o_idx = static_cast<size_t>(ik) * out_strides[2] +
                                           static_cast<size_t>(cur_ho) * out_strides[1] +
                                           static_cast<size_t>(cur_wo) * out_strides[0];

                            size_t f_idx = static_cast<size_t>(ik) * wei_strides[3] +
                                           static_cast<size_t>(iy) * wei_strides[1] +
                                           static_cast<size_t>(ix) * wei_strides[0];

                            value += cast_to<src_data_t, acc_data_t>(p_out[o_idx]) *
                                     cast_to<src_data_t, acc_data_t>(p_wei[f_idx]);
                        }
                    }
                }
            }
        }

        if constexpr(ASSUME_PACKED)
        {
            size_t i_idx = static_cast<size_t>(ihi) * wi + static_cast<size_t>(iwi);
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_in, value, alpha, beta, i_idx);
        }
        else
        {
            size_t i_idx =
                static_cast<size_t>(ihi) * in_strides[1] + static_cast<size_t>(iwi) * in_strides[0];
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_in, value, alpha, beta, i_idx);
        }
    }
}

using index_t      = int32_t;
const index_t BatchPerWave = 1;
const index_t NumWavePerTile = 1;
const index_t TileIn_Pack_H = 1;
const index_t TileOut_Pack_H = 1;
const index_t InScalarPerVector = 1;
const index_t OutScalarPerVector = 1;
const index_t NumVectorPerPixel = 1;
const index_t NumTilePerBlock = 1;
const index_t NBatch = 1;
const index_t BlockTileSize = 1;
const index_t Tile_H = 1;
const index_t TileOut_H = 1;
static constexpr index_t wave_size = 64;

template <bool ASSUME_PACKED, typename src_data_t, typename acc_data_t, typename dst_data_t>
inline __device__ void naive_conv_wrw_nchw(const src_data_t* __restrict__ p_in,
                                           dst_data_t* __restrict__ p_wei,
                                           const double alpha,
                                           const double beta,
                                           const src_data_t* __restrict__ p_out,
                                           Strides5D in_strides,
                                           Strides5D wei_strides,
                                           Strides5D out_strides,
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
    __shared__ char
        p_share_in[GridwiseConvBwdWeight::ShareMemInSize * GridwiseConvBwdWeight::NumTilePerBlock];
    __shared__ char p_share_out[GridwiseConvBwdWeight::ShareMemOutSize *
                                GridwiseConvBwdWeight::NumTilePerBlock];

    const index_t g_idx   = __builtin_amdgcn_readfirstlane(blockIdx.x);
    const index_t wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / wave_size);
    const index_t tile_id = wave_id / NumWavePerTile;
    const index_t lane_id = __lane_id();

    constexpr index_t ThreadPerBatch = wave_size / BatchPerWave;
    // Debug<Sequence<TileIn_Pack_H, TileOut_Pack_H>> xx3;
    static_assert(Tile_H % NumWavePerTile == 0);
    static_assert(TileOut_H % NumWavePerTile == 0);
    index_t tmp_in[integer_divide_ceil(TileIn_Pack_H, NumWavePerTile) *
                        NumVectorPerPixel * InScalarPerVector]    = {};
    index_t tmp_out[integer_divide_ceil(TileOut_Pack_H, NumWavePerTile) *
                            NumVectorPerPixel * OutScalarPerVector] = {};

    static_assert(NumTilePerBlock == 1 || NumWavePerTile == 1);

    static constexpr index_t spatial_offset = 3;
    // const index_t n                         = arg.in_g_n_c_wis_lengths_[1];
    index_t num_loop                        = n / NumTilePerBlock / NBatch - 1;
    index_t n_idx                           = n / NumTilePerBlock * tile_id;
    if constexpr(NumTilePerBlock > 1)
    {
        if(tile_id == NumTilePerBlock - 1)
        {
            n_idx    = n / NumTilePerBlock * (NumTilePerBlock - 1);
            num_loop = (n - n_idx) / NBatch - 1;
        }
    }

    // In
    // const index_t hi = arg.in_g_n_c_wis_lengths_[spatial_offset + 0];
    // const index_t wi = arg.in_g_n_c_wis_lengths_[spatial_offset + 1];

    // const index_t hi_stride   = arg.in_g_n_c_wis_strides_[spatial_offset + 0];
    // const index_t wi_stride   = arg.in_g_n_c_wis_strides_[spatial_offset + 1];
    // const index_t in_g_stride = arg.in_g_n_c_wis_strides_[0];
    // const index_t in_n_stride = arg.in_g_n_c_wis_strides_[1];

    const index_t hi_stride   = in_strides[1];
    const index_t wi_stride   = in_strides[0];
    const index_t in_g_stride = in_strides[3];
    const index_t in_n_stride = in_strides[4];

    // Out
    // const index_t ho = arg.out_g_n_k_wos_lengths_[spatial_offset + 0];
    // const index_t wo = arg.out_g_n_k_wos_lengths_[spatial_offset + 1];

    // const index_t ho_stride    = arg.out_g_n_k_wos_strides_[spatial_offset + 0];
    // const index_t wo_stride    = arg.out_g_n_k_wos_strides_[spatial_offset + 1];
    // const index_t out_g_stride = arg.out_g_n_k_wos_strides_[0];
    // const index_t out_n_stride = arg.out_g_n_k_wos_strides_[1];

    const index_t ho_stride    = out_strides[1];
    const index_t wo_stride    = out_strides[0];
    const index_t out_g_stride = out_strides[3];
    const index_t out_n_stride = out_strides[4];

    // Wei
    // auto* p_in  = arg.p_in_grid_ + g_idx * in_g_stride + n_idx * in_n_stride;
    // auto* p_out = arg.p_out_grid_ + g_idx * out_g_stride + n_idx * out_n_stride;

    p_in  = p_in + g_idx * in_g_stride + n_idx * in_n_stride;
    p_out = p_out + g_idx * out_g_stride + n_idx * out_n_stride;

    constexpr index_t Copy_Tile_H    = Tile_H / NumWavePerTile;
    constexpr index_t Copy_TileOut_H = TileOut_H / NumWavePerTile;
    if constexpr(NumWavePerTile > 1)
    {
        // static_assert(RequirePadding == false);
        p_in += Copy_Tile_H * hi_stride * (wave_id % NumWavePerTile);
        p_out += Copy_TileOut_H * ho_stride * (wave_id % NumWavePerTile);
    }

//     InDataVector* share_in   = reinterpret_cast<InDataVector*>(p_share_in);
//     OutDataVector* share_out = reinterpret_cast<OutDataVector*>(p_share_out);
//     if constexpr(NumTilePerBlock > 1)
//     {
//         share_in  = reinterpret_cast<InDataVector*>(p_share_in + ShareMemInSize * wave_id);
//         share_out = reinterpret_cast<InDataVector*>(p_share_out + ShareMemOutSize * wave_id);
//     }

//     // init lds 0
//     index_t cluster_id = threadIdx.x % (wave_size * NumWavePerTile);
//     auto init_pading   = [&](auto* share_vec, auto count) {
//         static_for<0, math::integer_divide_ceil(count, wave_size * NumWavePerTile), 1>{}(
//             [&](auto i) {
//                 if(cluster_id + i * wave_size * NumWavePerTile < count)
//                 {
//                     share_vec[cluster_id + i * wave_size * NumWavePerTile] = {};
//                 }
//             });
//     };
//     auto init_array_pading = [&](auto* share_vec,
//                                     auto element_count,
//                                     auto array_count,
//                                     index_t stride) {
//         static_for<0, math::integer_divide_ceil(array_count, wave_size * NumWavePerTile), 1>{}(
//             [&](auto i) {
//                 static_for<0, element_count, 1>{}([&](auto j) {
//                     if(cluster_id + i * wave_size * NumWavePerTile < array_count)
//                     {
//                         auto p = share_vec +
//                                     (cluster_id + i * wave_size * NumWavePerTile) * stride + j;
//                         *p = {};
//                     }
//                 });
//             });
//     };
//     constexpr index_t TopPadingSize = Pad_H * TileIn_Align_W * NumVectorPerPixel;
//     constexpr index_t TileInEnd     = (Tile_H + Pad_H) * TileIn_Align_W;
//     constexpr index_t ButtomPaddingSize =
//         (ShareMemInSize / (sizeof(InDataType) * NBatch) - TileInEnd) * NumVectorPerPixel;
//     if constexpr(Pad_W > 0)
//     {
//         static_assert(ButtomPaddingSize >= 0);
//         init_array_pading(share_in + TopPadingSize,
//                             Number<Pad_W * NumVectorPerPixel>{},
//                             Number<Tile_H>{},
//                             TileIn_Align_W * NumVectorPerPixel);
//         init_array_pading(share_in + TopPadingSize + (Tile_W + Pad_W) * NumVectorPerPixel,
//                             Number<Pad_W * NumVectorPerPixel>{},
//                             Number<Tile_H>{},
//                             TileIn_Align_W * NumVectorPerPixel);
//     }
//     if constexpr(Pad_H > 0)
//     {
//         init_pading(share_in, Number<TopPadingSize>{});
//         init_pading(share_in + TileInEnd * NumVectorPerPixel, Number<ButtomPaddingSize>{});
//     }
//     constexpr index_t TileOutEnd = TileOut_H * TileOut_Align_W;
//     constexpr index_t OutButtomPaddingSize =
//         (ShareMemOutSize / (sizeof(OutDataType) * NBatch) - TileOutEnd) * NumVectorPerPixel;
//     init_pading(share_out + TileOutEnd * NumVectorPerPixel, Number<OutButtomPaddingSize>{});

//     if constexpr(NumWavePerTile > 1)
//     {
//         block_sync_lds();
//     }

//     const index_t in_x         = lane_id % TileIn_Pack_W;
//     const index_t in_y_offset  = lane_id / TileIn_Pack_W;
//     const index_t out_x        = lane_id % TileOut_Pack_W;
//     const index_t out_y_offset = lane_id / TileOut_Pack_W;

//     // prefetch 0
//     if(in_x < (Tile_W / InScalarPerVector))
//     {
//         load_data_from_global<Copy_Tile_H, Tile_W, InScalarPerVector>(
//             p_in, in_x, in_y_offset, in_n_stride, hi, wi, hi_stride, wi_stride, tmp_in);
//     }
//     if(out_x < (TileOut_W / OutScalarPerVector))
//     {
//         load_data_from_global<Copy_TileOut_H, TileOut_W, OutScalarPerVector>(
//             p_out, out_x, out_y_offset, out_n_stride, ho, wo, ho_stride, wo_stride, tmp_out);
//     }
//     p_in += NBatch * in_n_stride;
//     p_out += NBatch * out_n_stride;

//     auto share_in_base  = share_in;
//     auto share_out_base = share_out;
//     // adjust share memory offset for copy
//     if constexpr(NumWavePerTile > 1)
//     {
//         // static_assert(RequirePadding == false);
//         share_in += Copy_Tile_H * TileIn_Align_W * NumVectorPerPixel * wave_id;
//         share_out += Copy_TileOut_H * TileOut_Align_W * NumVectorPerPixel * wave_id;
//     }

//     share_in += (TileIn_Align_W * Pad_H + Pad_W) * NumVectorPerPixel;
//     if(in_x < (Tile_W / InScalarPerVector))
//     {
//         write_data_to_lds<Copy_Tile_H, Tile_W, TileIn_Align_W, InScalarPerVector>(
//             in_x, in_y_offset, tmp_in, share_in);
//     }
//     if(out_x < (TileOut_W / OutScalarPerVector))
//     {
//         write_data_to_lds<Copy_TileOut_H, TileOut_W, TileOut_Align_W, OutScalarPerVector>(
//             out_x, out_y_offset, tmp_out, share_out);
//     }

//     constexpr index_t TileOut_H_batch = math::integer_divide_ceil(Copy_TileOut_H, BatchPerWave);
//     index_t hout_base                 = lane_id / ThreadPerBatch * TileOut_H_batch;
//     if constexpr(NumWavePerTile > 1)
//     {
//         hout_base += Copy_TileOut_H * wave_id;
//     }
//     index_t x = (lane_id % ThreadPerBatch) % Filter_X;
//     index_t y = (lane_id % ThreadPerBatch) / Filter_X;
//     float acc = 0;

//     // if (lane_id == 0)
//     //{
//     //     dump_lds(tmp_in, sizeof(tmp_in)/sizeof(InDataVector),
//     //     sizeof(tmp_in)/sizeof(InDataVector));
//     //    dump_lds(tmp_out, sizeof(tmp_out)/sizeof(OutDataVector),
//     //    sizeof(tmp_out)/sizeof(OutDataVector));
//     //}
//     // block_sync_lds();
//     // if (threadIdx.x == 0)
//     //{
//     //   dump_lds(reinterpret_cast<InDataVector*>(p_share_in),
//     //   ShareMemInSize/sizeof(InDataVector), TileIn_Align_W * NumVectorPerPixel);
//     //  dump_lds(reinterpret_cast<OutDataVector*>(p_share_out),
//     //  ShareMemOutSize/sizeof(OutDataVector), TileOut_Align_W * NumVectorPerPixel);
//     //}
//     // block_sync_lds();

// #if defined(ENABLE_PIPELINE_V2)
//     while(num_loop > 0)
//     {
//         // do conv_bwd on 0
//         if constexpr(NumWavePerTile > 1)
//         {
//             block_sync_lds();
//         }
//         run_conv_bwd_weight<TileOut_H_batch>(
//             x, y, ho, wo, hout_base, share_in_base, share_out_base, acc);
//         if constexpr(NumWavePerTile > 1)
//         {
//             block_sync_lds();
//         }
//         if(in_x < (Tile_W / InScalarPerVector))
//         {
//             load_data_from_global<Copy_Tile_H, Tile_W, InScalarPerVector>(
//                 p_in, in_x, in_y_offset, in_n_stride, hi, wi, hi_stride, wi_stride, tmp_in);

//             write_data_to_lds<Copy_Tile_H, Tile_W, TileIn_Align_W, InScalarPerVector>(
//                 in_x, in_y_offset, tmp_in, share_in);
//         }

//         if(out_x < (TileOut_W / OutScalarPerVector))
//         {
//             load_data_from_global<Copy_TileOut_H, TileOut_W, OutScalarPerVector>(p_out,
//                                                                                     out_x,
//                                                                                     out_y_offset,
//                                                                                     out_n_stride,
//                                                                                     ho,
//                                                                                     wo,
//                                                                                     ho_stride,
//                                                                                     wo_stride,
//                                                                                     tmp_out);
//             write_data_to_lds<Copy_TileOut_H, TileOut_W, TileOut_Align_W, OutScalarPerVector>(
//                 out_x, out_y_offset, tmp_out, share_out);
//         }

//         p_in += NBatch * in_n_stride;
//         p_out += NBatch * out_n_stride;
//         num_loop--;
//     };
// #else
//     while(num_loop > 0)
//     {
//         if(in_x < (Tile_W / InScalarPerVector))
//         {
//             load_data_from_global<Copy_Tile_H, Tile_W, InScalarPerVector>(
//                 p_in, in_x, in_y_offset, in_n_stride, hi, wi, hi_stride, wi_stride, tmp_in);
//         }
//         if(out_x < (TileOut_W / OutScalarPerVector))
//         {
//             load_data_from_global<Copy_TileOut_H, TileOut_W, OutScalarPerVector>(p_out,
//                                                                                     out_x,
//                                                                                     out_y_offset,
//                                                                                     out_n_stride,
//                                                                                     ho,
//                                                                                     wo,
//                                                                                     ho_stride,
//                                                                                     wo_stride,
//                                                                                     tmp_out);
//         }
//         p_in += NBatch * in_n_stride;
//         p_out += NBatch * out_n_stride;

//         // do conv_bwd on 0
//         if constexpr(NumWavePerTile > 1)
//         {
//             block_sync_lds();
//         }
//         run_conv_bwd_weight<TileOut_H_batch>(
//             x, y, ho, wo, hout_base, share_in_base, share_out_base, acc);
//         if constexpr(NumWavePerTile > 1)
//         {
//             block_sync_lds();
//         }
//         // write 0
//         if(in_x < (Tile_W / InScalarPerVector))
//         {
//             write_data_to_lds<Copy_Tile_H, Tile_W, TileIn_Align_W, InScalarPerVector>(
//                 in_x, in_y_offset, tmp_in, share_in);
//         }
//         if(out_x < (TileOut_W / OutScalarPerVector))
//         {
//             write_data_to_lds<Copy_TileOut_H, TileOut_W, TileOut_Align_W, OutScalarPerVector>(
//                 out_x, out_y_offset, tmp_out, share_out);
//         }
//         num_loop--;
//     };
// #endif
//     // tail
//     {
//         if constexpr(NumWavePerTile > 1)
//         {
//             block_sync_lds();
//         }
//         run_conv_bwd_weight<TileOut_H_batch>(
//             x, y, ho, wo, hout_base, share_in_base, share_out_base, acc);
//     }

//     if constexpr(ThreadPerBatch == 32)
//     {
//         float acc_2 = warp_shuffle_down(acc, ThreadPerBatch);
//         acc += acc_2;
//     }
//     else if constexpr(ThreadPerBatch == 9)
//     {
//         // todo optimization reduce operation.
//         float acc_2 = warp_shuffle_down(acc, ThreadPerBatch);
//         float acc_3 = warp_shuffle_down(acc, 2 * ThreadPerBatch);
//         float acc_4 = warp_shuffle_down(acc, 3 * ThreadPerBatch);
//         float acc_5 = warp_shuffle_down(acc, 4 * ThreadPerBatch);
//         float acc_6 = warp_shuffle_down(acc, 5 * ThreadPerBatch);
//         float acc_7 = warp_shuffle_down(acc, 6 * ThreadPerBatch);
//         acc += acc_2 + acc_3 + acc_4 + acc_5 + acc_6 + acc_7;
//     }
//     if constexpr(NumTilePerBlock == 1 && NumWavePerTile == 1)
//     {
//         if(hout_base == 0)
//         {
//             write_output(arg, g_idx, y, x, acc);
//         }
//     }
//     else
//     {
//         uint32_t* p_share_acc = reinterpret_cast<uint32_t*>(p_share_in);
//         block_sync_lds();
//         p_share_acc[threadIdx.x] = bit_cast<uint32_t>(acc);
//         block_sync_lds();
//         if(hout_base == 0 && wave_id == 0)
//         {
//             for(int i = 1; i < NumTilePerBlock * NumWavePerTile; i++)
//             {
//                 acc += bit_cast<float>(p_share_acc[i * wave_size + lane_id]);
//             }
//             write_output(arg, g_idx, y, x, type_convert<WeiDataType>(acc));
//         }
//     }
}

// design block_size 256
template <bool ASSUME_PACKED, typename src_data_t, typename acc_data_t, typename dst_data_t>
inline __device__ void naive_conv_fwd_ncdhw(const src_data_t* __restrict__ p_in,
                                            const src_data_t* __restrict__ p_wei,
                                            const double alpha,
                                            const double beta,
                                            dst_data_t* __restrict__ p_out,
                                            Strides6D in_strides,
                                            Strides6D wei_strides,
                                            Strides6D out_strides,
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

    if constexpr(ASSUME_PACKED)
    {
        p_in += static_cast<size_t>(in) * c * di * hi * wi +
                static_cast<size_t>(ig) * c_per_group * di * hi * wi;

        p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fz * fy * fx +
                 static_cast<size_t>(ik) * c_per_group * fz * fy * fx;

        p_out += static_cast<size_t>(in) * k * do_ * ho * wo +
                 static_cast<size_t>(ig) * k_per_group * do_ * ho * wo +
                 static_cast<size_t>(ik) * do_ * ho * wo;
    }
    else
    {
        p_in += static_cast<size_t>(in) * in_strides[5] + static_cast<size_t>(ig) * in_strides[4];

        p_wei +=
            static_cast<size_t>(ig) * wei_strides[5] + static_cast<size_t>(ik) * wei_strides[4];

        p_out += static_cast<size_t>(in) * out_strides[5] +
                 static_cast<size_t>(ig) * out_strides[4] +
                 static_cast<size_t>(ik) * out_strides[3];
    }

    for(int tid = threadIdx.x; tid < thread_length; tid += blockDim.x)
    {
        int iwo = tid % wo;
        int iho = (tid / wo) % ho;
        int ido = tid / (ho * wo);

        acc_data_t value = 0;

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
                            if constexpr(ASSUME_PACKED)
                            {
                                size_t i_idx = static_cast<size_t>(ic) * di * hi * wi +
                                               static_cast<size_t>(cur_d) * hi * wi +
                                               static_cast<size_t>(cur_h) * wi +
                                               static_cast<size_t>(cur_w);

                                size_t f_idx = static_cast<size_t>(ic) * fz * fy * fx +
                                               static_cast<size_t>(iz) * fy * fx +
                                               static_cast<size_t>(iy) * fx +
                                               static_cast<size_t>(ix);

                                value += cast_to<src_data_t, acc_data_t>(p_in[i_idx]) *
                                         cast_to<src_data_t, acc_data_t>(p_wei[f_idx]);
                            }
                            else
                            {
                                size_t i_idx = static_cast<size_t>(ic) * in_strides[3] +
                                               static_cast<size_t>(cur_d) * in_strides[2] +
                                               static_cast<size_t>(cur_h) * in_strides[1] +
                                               static_cast<size_t>(cur_w) * in_strides[0];

                                size_t f_idx = static_cast<size_t>(ic) * wei_strides[3] +
                                               static_cast<size_t>(iz) * wei_strides[2] +
                                               static_cast<size_t>(iy) * wei_strides[1] +
                                               static_cast<size_t>(ix) * wei_strides[0];

                                value += cast_to<src_data_t, acc_data_t>(p_in[i_idx]) *
                                         cast_to<src_data_t, acc_data_t>(p_wei[f_idx]);
                            }
                        }
                    }
                }
            }
        }

        if constexpr(ASSUME_PACKED)
        {
            size_t o_idx = static_cast<size_t>(ido) * ho * wo + static_cast<size_t>(iho) * wo +
                           static_cast<size_t>(iwo);
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_out, value, alpha, beta, o_idx);
        }
        else
        {
            size_t o_idx = static_cast<size_t>(ido) * out_strides[2] +
                           static_cast<size_t>(iho) * out_strides[1] +
                           static_cast<size_t>(iwo) * out_strides[0];
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_out, value, alpha, beta, o_idx);
        }
    }
}

template <bool ASSUME_PACKED, typename src_data_t, typename acc_data_t, typename dst_data_t>
inline __device__ void naive_conv_bwd_ncdhw(dst_data_t* __restrict__ p_in,
                                            const src_data_t* __restrict__ p_wei,
                                            const double alpha,
                                            const double beta,
                                            const src_data_t* __restrict__ p_out,
                                            Strides6D in_strides,
                                            Strides6D wei_strides,
                                            Strides6D out_strides,
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

    if constexpr(ASSUME_PACKED)
    {
        p_in += static_cast<size_t>(in) * c * di * hi * wi +
                static_cast<size_t>(ig) * c_per_group * di * hi * wi +
                static_cast<size_t>(ic) * di * hi * wi;

        p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fz * fy * fx +
                 static_cast<size_t>(ic) * fz * fy * fx;

        p_out += static_cast<size_t>(in) * k * do_ * ho * wo +
                 static_cast<size_t>(ig) * k_per_group * do_ * ho * wo;
    }
    else
    {
        p_in += static_cast<size_t>(in) * in_strides[5] + static_cast<size_t>(ig) * in_strides[4] +
                static_cast<size_t>(ic) * in_strides[3];

        p_wei +=
            static_cast<size_t>(ig) * wei_strides[5] + static_cast<size_t>(ic) * wei_strides[3];

        p_out +=
            static_cast<size_t>(in) * out_strides[5] + static_cast<size_t>(ig) * out_strides[4];
    }

    for(int tid = threadIdx.x; tid < thread_length; tid += blockDim.x)
    {
        int iwi = tid % wi;
        int ihi = (tid / wi) % hi;
        int idi = tid / (hi * wi);

        acc_data_t value = 0;

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
                            if constexpr(ASSUME_PACKED)
                            {
                                size_t o_idx = static_cast<size_t>(ik) * do_ * ho * wo +
                                               static_cast<size_t>(cur_do) * ho * wo +
                                               static_cast<size_t>(cur_ho) * wo +
                                               static_cast<size_t>(cur_wo);

                                size_t f_idx =
                                    static_cast<size_t>(ik) * c_per_group * fz * fy * fx +
                                    static_cast<size_t>(iz) * fy * fx +
                                    static_cast<size_t>(iy) * fx + static_cast<size_t>(ix);

                                value += cast_to<src_data_t, acc_data_t>(p_out[o_idx]) *
                                         cast_to<src_data_t, acc_data_t>(p_wei[f_idx]);
                            }
                            else
                            {
                                size_t o_idx = static_cast<size_t>(ik) * out_strides[3] +
                                               static_cast<size_t>(cur_do) * out_strides[2] +
                                               static_cast<size_t>(cur_ho) * out_strides[1] +
                                               static_cast<size_t>(cur_wo) * out_strides[0];

                                size_t f_idx = static_cast<size_t>(ik) * wei_strides[4] +
                                               static_cast<size_t>(iz) * wei_strides[2] +
                                               static_cast<size_t>(iy) * wei_strides[1] +
                                               static_cast<size_t>(ix) * wei_strides[0];

                                value += cast_to<src_data_t, acc_data_t>(p_out[o_idx]) *
                                         cast_to<src_data_t, acc_data_t>(p_wei[f_idx]);
                            }
                        }
                    }
                }
            }
        }

        if constexpr(ASSUME_PACKED)
        {
            size_t i_idx = static_cast<size_t>(idi) * hi * wi + static_cast<size_t>(ihi) * wi +
                           static_cast<size_t>(iwi);
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_in, value, alpha, beta, i_idx);
        }
        else
        {
            size_t i_idx = static_cast<size_t>(idi) * in_strides[2] +
                           static_cast<size_t>(ihi) * in_strides[1] +
                           static_cast<size_t>(iwi) * in_strides[0];
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_in, value, alpha, beta, i_idx);
        }
    }
}

template <bool ASSUME_PACKED, typename src_data_t, typename acc_data_t, typename dst_data_t>
inline __device__ void naive_conv_wrw_ncdhw(const src_data_t* __restrict__ p_in,
                                            dst_data_t* __restrict__ p_wei,
                                            const double alpha,
                                            const double beta,
                                            const src_data_t* __restrict__ p_out,
                                            Strides6D in_strides,
                                            Strides6D wei_strides,
                                            Strides6D out_strides,
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

    if constexpr(ASSUME_PACKED)
    {
        p_in += static_cast<size_t>(ig) * c_per_group * di * hi * wi;

        p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fz * fy * fx +
                 static_cast<size_t>(ik) * c_per_group * fz * fy * fx;

        p_out += static_cast<size_t>(ig) * k_per_group * do_ * ho * wo +
                 static_cast<size_t>(ik) * do_ * ho * wo;
    }
    else
    {
        p_in += static_cast<size_t>(ig) * in_strides[4];

        p_wei +=
            static_cast<size_t>(ig) * wei_strides[5] + static_cast<size_t>(ik) * wei_strides[4];

        p_out +=
            static_cast<size_t>(ig) * out_strides[4] + static_cast<size_t>(ik) * out_strides[3];
    }

    for(int tid = threadIdx.x; tid < thread_length; tid += blockDim.x)
    {
        int ix = tid % fx;
        int iy = (tid / fx) % fy;
        int iz = (tid / (fx * fy)) % fz;
        int ic = tid / (fx * fy * fz);

        acc_data_t value = 0;

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
                            if constexpr(ASSUME_PACKED)
                            {
                                size_t i_idx = static_cast<size_t>(in) * c * di * hi * wi +
                                               static_cast<size_t>(ic) * di * hi * wi +
                                               static_cast<size_t>(cur_d) * hi * wi +
                                               static_cast<size_t>(cur_h) * wi +
                                               static_cast<size_t>(cur_w);

                                size_t o_idx = static_cast<size_t>(in) * k * do_ * ho * wo +
                                               static_cast<size_t>(ido) * ho * wo +
                                               static_cast<size_t>(iho) * wo +
                                               static_cast<size_t>(iwo);

                                value += cast_to<src_data_t, acc_data_t>(p_in[i_idx]) *
                                         cast_to<src_data_t, acc_data_t>(p_out[o_idx]);
                            }
                            else
                            {
                                size_t i_idx = static_cast<size_t>(in) * in_strides[5] +
                                               static_cast<size_t>(ic) * in_strides[3] +
                                               static_cast<size_t>(cur_d) * in_strides[2] +
                                               static_cast<size_t>(cur_h) * in_strides[1] +
                                               static_cast<size_t>(cur_w) * in_strides[0];

                                size_t o_idx = static_cast<size_t>(in) * out_strides[5] +
                                               static_cast<size_t>(ido) * out_strides[2] +
                                               static_cast<size_t>(iho) * out_strides[1] +
                                               static_cast<size_t>(iwo) * out_strides[0];

                                value += cast_to<src_data_t, acc_data_t>(p_in[i_idx]) *
                                         cast_to<src_data_t, acc_data_t>(p_out[o_idx]);
                            }
                        }
                    }
                }
            }
        }

        if constexpr(ASSUME_PACKED)
        {
            size_t f_idx = static_cast<size_t>(ic) * fz * fy * fx +
                           static_cast<size_t>(iz) * fy * fx + static_cast<size_t>(iy) * fx +
                           static_cast<size_t>(ix);
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, value, alpha, beta, f_idx);
        }
        else
        {
            size_t f_idx = static_cast<size_t>(ic) * wei_strides[3] +
                           static_cast<size_t>(iz) * wei_strides[2] +
                           static_cast<size_t>(iy) * wei_strides[1] +
                           static_cast<size_t>(ix) * wei_strides[0];
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, value, alpha, beta, f_idx);
        }
    }
}

/***************************** nhwc *****************************/
// design block_size 256
template <bool ASSUME_PACKED, typename src_data_t, typename acc_data_t, typename dst_data_t>
inline __device__ void naive_conv_fwd_nhwc(const src_data_t* __restrict__ p_in,
                                           const src_data_t* __restrict__ p_wei,
                                           const double alpha,
                                           const double beta,
                                           dst_data_t* __restrict__ p_out,
                                           Strides5D in_strides,
                                           Strides5D wei_strides,
                                           Strides5D out_strides,
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

    if constexpr(ASSUME_PACKED)
    {
        p_in += static_cast<size_t>(in) * hi * wi * c + static_cast<size_t>(ig) * c_per_group;

        p_wei += static_cast<size_t>(ig) * k_per_group * fy * fx * c_per_group;

        p_out += static_cast<size_t>(in) * ho * wo * k + static_cast<size_t>(iho) * wo * k +
                 static_cast<size_t>(ig) * k_per_group;
    }
    else
    {
        p_in += static_cast<size_t>(in) * in_strides[4] + static_cast<size_t>(ig) * in_strides[1];

        p_wei += static_cast<size_t>(ig) * wei_strides[4];

        p_out += static_cast<size_t>(in) * out_strides[4] +
                 static_cast<size_t>(iho) * out_strides[3] +
                 static_cast<size_t>(ig) * out_strides[1];
    }

    for(int tid = threadIdx.x; tid < thread_length; tid += blockDim.x)
    {
        int iwo = tid / k_per_group;
        int ik  = tid % k_per_group;

        acc_data_t value = 0;

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
                        if constexpr(ASSUME_PACKED)
                        {
                            size_t i_idx = static_cast<size_t>(cur_h) * wi * c +
                                           static_cast<size_t>(cur_w) * c + static_cast<size_t>(ic);

                            size_t f_idx = static_cast<size_t>(ik) * fy * fx * c_per_group +
                                           static_cast<size_t>(iy) * fx * c_per_group +
                                           static_cast<size_t>(ix) * c_per_group +
                                           static_cast<size_t>(ic);

                            value += cast_to<src_data_t, acc_data_t>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t>(p_wei[f_idx]);
                        }
                        else
                        {
                            size_t i_idx = static_cast<size_t>(cur_h) * in_strides[3] +
                                           static_cast<size_t>(cur_w) * in_strides[2] +
                                           static_cast<size_t>(ic) * in_strides[0];

                            size_t f_idx = static_cast<size_t>(ik) * wei_strides[3] +
                                           static_cast<size_t>(iy) * wei_strides[2] +
                                           static_cast<size_t>(ix) * wei_strides[1] +
                                           static_cast<size_t>(ic) * wei_strides[0];

                            value += cast_to<src_data_t, acc_data_t>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t>(p_wei[f_idx]);
                        }
                    }
                }
            }
        }

        if constexpr(ASSUME_PACKED)
        {
            size_t o_idx = static_cast<size_t>(iwo) * k + static_cast<size_t>(ik);
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_out, value, alpha, beta, o_idx);
        }
        else
        {
            size_t o_idx = static_cast<size_t>(iwo) * out_strides[2] +
                           static_cast<size_t>(ik) * out_strides[0];
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_out, value, alpha, beta, o_idx);
        }
    }
}

template <bool ASSUME_PACKED, typename src_data_t, typename acc_data_t, typename dst_data_t>
inline __device__ void naive_conv_bwd_nhwc(dst_data_t* __restrict__ p_in,
                                           const src_data_t* __restrict__ p_wei,
                                           const double alpha,
                                           const double beta,
                                           const src_data_t* __restrict__ p_out,
                                           Strides5D in_strides,
                                           Strides5D wei_strides,
                                           Strides5D out_strides,
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

    if constexpr(ASSUME_PACKED)
    {
        p_in += static_cast<size_t>(in) * hi * wi * c + static_cast<size_t>(ihi) * wi * c +
                static_cast<size_t>(ig) * c_per_group;

        p_wei += static_cast<size_t>(ig) * k_per_group * fy * fx * c_per_group;

        p_out += static_cast<size_t>(in) * ho * wo * k + static_cast<size_t>(ig) * k_per_group;
    }
    else
    {
        p_in += static_cast<size_t>(in) * in_strides[4] + static_cast<size_t>(ihi) * in_strides[3] +
                static_cast<size_t>(ig) * in_strides[1];

        p_wei += static_cast<size_t>(ig) * wei_strides[4];

        p_out +=
            static_cast<size_t>(in) * out_strides[4] + static_cast<size_t>(ig) * out_strides[1];
    }

    for(int tid = threadIdx.x; tid < thread_length; tid += blockDim.x)
    {
        int iwi = tid / c_per_group;
        int ic  = tid % c_per_group;

        acc_data_t value = 0;

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
                        if constexpr(ASSUME_PACKED)
                        {
                            size_t o_idx = static_cast<size_t>(cur_ho) * wo * k +
                                           static_cast<size_t>(cur_wo) * k +
                                           static_cast<size_t>(ik);

                            size_t f_idx = static_cast<size_t>(ik) * fy * fx * c_per_group +
                                           static_cast<size_t>(iy) * fx * c_per_group +
                                           static_cast<size_t>(ix) * c_per_group +
                                           static_cast<size_t>(ic);

                            value += cast_to<src_data_t, acc_data_t>(p_out[o_idx]) *
                                     cast_to<src_data_t, acc_data_t>(p_wei[f_idx]);
                        }
                        else
                        {
                            size_t o_idx = static_cast<size_t>(cur_ho) * out_strides[3] +
                                           static_cast<size_t>(cur_wo) * out_strides[2] +
                                           static_cast<size_t>(ik) * out_strides[0];

                            size_t f_idx = static_cast<size_t>(ik) * wei_strides[3] +
                                           static_cast<size_t>(iy) * wei_strides[2] +
                                           static_cast<size_t>(ix) * wei_strides[1] +
                                           static_cast<size_t>(ic) * wei_strides[0];

                            value += cast_to<src_data_t, acc_data_t>(p_out[o_idx]) *
                                     cast_to<src_data_t, acc_data_t>(p_wei[f_idx]);
                        }
                    }
                }
            }
        }

        if constexpr(ASSUME_PACKED)
        {
            size_t i_idx = static_cast<size_t>(iwi) * c + static_cast<size_t>(ic);
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_in, value, alpha, beta, i_idx);
        }
        else
        {
            size_t i_idx =
                static_cast<size_t>(iwi) * in_strides[2] + static_cast<size_t>(ic) * in_strides[0];
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_in, value, alpha, beta, i_idx);
        }
    }
}

template <bool ASSUME_PACKED, typename src_data_t, typename acc_data_t, typename dst_data_t>
inline __device__ void naive_conv_wrw_nhwc(const src_data_t* __restrict__ p_in,
                                           dst_data_t* __restrict__ p_wei,
                                           const double alpha,
                                           const double beta,
                                           const src_data_t* __restrict__ p_out,
                                           Strides5D in_strides,
                                           Strides5D wei_strides,
                                           Strides5D out_strides,
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

    if constexpr(ASSUME_PACKED)
    {
        p_in += static_cast<size_t>(ig) * c_per_group;

        p_wei += static_cast<size_t>(ig) * k_per_group * fy * fx * c_per_group +
                 static_cast<size_t>(ik) * fy * fx * c_per_group;

        p_out += static_cast<size_t>(ig) * k_per_group + static_cast<size_t>(ik);
    }
    else
    {
        p_in += static_cast<size_t>(ig) * in_strides[1];

        p_wei +=
            static_cast<size_t>(ig) * wei_strides[4] + static_cast<size_t>(ik) * wei_strides[3];

        p_out +=
            static_cast<size_t>(ig) * out_strides[1] + static_cast<size_t>(ik) * out_strides[0];
    }

    for(int tid = threadIdx.x; tid < thread_length; tid += blockDim.x)
    {
        int ic = tid % c_per_group;
        int ix = (tid / c_per_group) % fx;
        int iy = tid / (c_per_group * fx);

        acc_data_t value = 0;

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

                        if constexpr(ASSUME_PACKED)
                        {
                            size_t i_idx = static_cast<size_t>(in) * hi * wi * c +
                                           static_cast<size_t>(cur_h) * wi * c +
                                           static_cast<size_t>(cur_w) * c + static_cast<size_t>(ic);

                            size_t o_idx = static_cast<size_t>(in) * ho * wo * k +
                                           static_cast<size_t>(iho) * wo * k +
                                           static_cast<size_t>(iwo) * k;

                            value += cast_to<src_data_t, acc_data_t>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t>(p_out[o_idx]);
                        }
                        else
                        {
                            size_t i_idx = static_cast<size_t>(in) * in_strides[4] +
                                           static_cast<size_t>(cur_h) * in_strides[3] +
                                           static_cast<size_t>(cur_w) * in_strides[2] +
                                           static_cast<size_t>(ic) * in_strides[0];

                            size_t o_idx = static_cast<size_t>(in) * out_strides[4] +
                                           static_cast<size_t>(iho) * out_strides[3] +
                                           static_cast<size_t>(iwo) * out_strides[2];

                            value += cast_to<src_data_t, acc_data_t>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t>(p_out[o_idx]);
                        }
                    }
                }
            }
        }

        if constexpr(ASSUME_PACKED)
        {
            size_t f_idx = static_cast<size_t>(iy) * fx * c_per_group +
                           static_cast<size_t>(ix) * c_per_group + static_cast<size_t>(ic);
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, value, alpha, beta, f_idx);
        }
        else
        {
            size_t f_idx = static_cast<size_t>(iy) * wei_strides[2] +
                           static_cast<size_t>(ix) * wei_strides[1] +
                           static_cast<size_t>(ic) * wei_strides[0];
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, value, alpha, beta, f_idx);
        }
    }
}

// design block_size 256
template <bool ASSUME_PACKED, typename src_data_t, typename acc_data_t, typename dst_data_t>
inline __device__ void naive_conv_fwd_ndhwc(const src_data_t* __restrict__ p_in,
                                            const src_data_t* __restrict__ p_wei,
                                            const double alpha,
                                            const double beta,
                                            dst_data_t* __restrict__ p_out,
                                            Strides6D in_strides,
                                            Strides6D wei_strides,
                                            Strides6D out_strides,
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

    if constexpr(ASSUME_PACKED)
    {
        p_in += static_cast<size_t>(in) * di * hi * wi * c + static_cast<size_t>(ig) * c_per_group;

        p_wei += static_cast<size_t>(ig) * k_per_group * fz * fy * fx * c_per_group;

        p_out += static_cast<size_t>(in) * do_ * ho * wo * k +
                 static_cast<size_t>(ido) * ho * wo * k + static_cast<size_t>(ig) * k_per_group;
    }
    else
    {
        // dim order NDHWGC
        // replace C and K with G * C_per_G and G * K_per_G
        p_in += static_cast<size_t>(in) * in_strides[5] + static_cast<size_t>(ig) * in_strides[1];

        // Assumes that group G is the highest dimension in the layout
        p_wei += static_cast<size_t>(ig) * wei_strides[5];

        p_out += static_cast<size_t>(in) * out_strides[5] +
                 static_cast<size_t>(ido) * out_strides[4] +
                 static_cast<size_t>(ig) * out_strides[1];
    }

    for(int tid = threadIdx.x; tid < thread_length; tid += blockDim.x)
    {
        int ik  = tid % k_per_group;
        int iwo = (tid / k_per_group) % wo;
        int iho = tid / (k_per_group * wo);

        acc_data_t value = 0;

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
                            if constexpr(ASSUME_PACKED)
                            {
                                size_t i_idx = static_cast<size_t>(cur_d) * hi * wi * c +
                                               static_cast<size_t>(cur_h) * wi * c +
                                               static_cast<size_t>(cur_w) * c +
                                               static_cast<size_t>(ic);

                                size_t f_idx =
                                    static_cast<size_t>(ik) * fz * fy * fx * c_per_group +
                                    static_cast<size_t>(iz) * fy * fx * c_per_group +
                                    static_cast<size_t>(iy) * fx * c_per_group +
                                    static_cast<size_t>(ix) * c_per_group + static_cast<size_t>(ic);

                                value += cast_to<src_data_t, acc_data_t>(p_in[i_idx]) *
                                         cast_to<src_data_t, acc_data_t>(p_wei[f_idx]);
                            }
                            else
                            {
                                size_t i_idx = static_cast<size_t>(cur_d) * in_strides[4] +
                                               static_cast<size_t>(cur_h) * in_strides[3] +
                                               static_cast<size_t>(cur_w) * in_strides[2] +
                                               static_cast<size_t>(ic) * in_strides[0];

                                size_t f_idx = static_cast<size_t>(ik) * wei_strides[4] +
                                               static_cast<size_t>(iz) * wei_strides[3] +
                                               static_cast<size_t>(iy) * wei_strides[2] +
                                               static_cast<size_t>(ix) * wei_strides[1] +
                                               static_cast<size_t>(ic) * wei_strides[0];

                                value += cast_to<src_data_t, acc_data_t>(p_in[i_idx]) *
                                         cast_to<src_data_t, acc_data_t>(p_wei[f_idx]);
                            }
                        }
                    }
                }
            }
        }

        if constexpr(ASSUME_PACKED)
        {
            size_t o_idx = static_cast<size_t>(iho) * wo * k + static_cast<size_t>(iwo) * k +
                           static_cast<size_t>(ik);
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_out, value, alpha, beta, o_idx);
        }
        else
        {
            size_t o_idx = static_cast<size_t>(iho) * out_strides[3] +
                           static_cast<size_t>(iwo) * out_strides[2] +
                           static_cast<size_t>(ik) * out_strides[0];
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_out, value, alpha, beta, o_idx);
        }
    }
}

template <bool ASSUME_PACKED, typename src_data_t, typename acc_data_t, typename dst_data_t>
inline __device__ void naive_conv_bwd_ndhwc(dst_data_t* __restrict__ p_in,
                                            const src_data_t* __restrict__ p_wei,
                                            const double alpha,
                                            const double beta,
                                            const src_data_t* __restrict__ p_out,
                                            Strides6D in_strides,
                                            Strides6D wei_strides,
                                            Strides6D out_strides,
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

    if constexpr(ASSUME_PACKED)
    {
        p_in += static_cast<size_t>(in) * di * hi * wi * c +
                static_cast<size_t>(idi) * hi * wi * c + static_cast<size_t>(ig) * c_per_group;

        p_wei += static_cast<size_t>(ig) * k_per_group * fz * fy * fx * c_per_group;

        p_out +=
            static_cast<size_t>(in) * do_ * ho * wo * k + static_cast<size_t>(ig) * k_per_group;
    }
    else
    {
        p_in += static_cast<size_t>(in) * in_strides[5] + static_cast<size_t>(idi) * in_strides[4] +
                static_cast<size_t>(ig) * in_strides[1];

        p_wei += static_cast<size_t>(ig) * wei_strides[5];

        p_out +=
            static_cast<size_t>(in) * out_strides[5] + static_cast<size_t>(ig) * out_strides[1];
    }

    for(int tid = threadIdx.x; tid < thread_length; tid += blockDim.x)
    {
        int ic  = tid % c_per_group;
        int iwi = (tid / c_per_group) % wi;
        int ihi = (tid / (c_per_group * wi));

        acc_data_t value = 0;

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
                            if constexpr(ASSUME_PACKED)
                            {
                                size_t o_idx = static_cast<size_t>(cur_do) * ho * wo * k +
                                               static_cast<size_t>(cur_ho) * wo * k +
                                               static_cast<size_t>(cur_wo) * k +
                                               static_cast<size_t>(ik);

                                size_t f_idx =
                                    static_cast<size_t>(ik) * fz * fy * fx * c_per_group +
                                    static_cast<size_t>(iz) * fy * fx * c_per_group +
                                    static_cast<size_t>(iy) * fx * c_per_group +
                                    static_cast<size_t>(ix) * c_per_group + static_cast<size_t>(ic);

                                value += cast_to<src_data_t, acc_data_t>(p_out[o_idx]) *
                                         cast_to<src_data_t, acc_data_t>(p_wei[f_idx]);
                            }
                            else
                            {
                                size_t o_idx = static_cast<size_t>(cur_do) * out_strides[4] +
                                               static_cast<size_t>(cur_ho) * out_strides[3] +
                                               static_cast<size_t>(cur_wo) * out_strides[2] +
                                               static_cast<size_t>(ik) * out_strides[0];

                                size_t f_idx = static_cast<size_t>(ik) * wei_strides[4] +
                                               static_cast<size_t>(iz) * wei_strides[3] +
                                               static_cast<size_t>(iy) * wei_strides[2] +
                                               static_cast<size_t>(ix) * wei_strides[1] +
                                               static_cast<size_t>(ic) * wei_strides[0];

                                value += cast_to<src_data_t, acc_data_t>(p_out[o_idx]) *
                                         cast_to<src_data_t, acc_data_t>(p_wei[f_idx]);
                            }
                        }
                    }
                }
            }
        }

        if constexpr(ASSUME_PACKED)
        {
            size_t i_idx = static_cast<size_t>(ihi) * wi * c + static_cast<size_t>(iwi) * c +
                           static_cast<size_t>(ic);
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_in, value, alpha, beta, i_idx);
        }
        else
        {
            size_t i_idx = static_cast<size_t>(ihi) * in_strides[3] +
                           static_cast<size_t>(iwi) * in_strides[2] +
                           static_cast<size_t>(ic) * in_strides[0];
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_in, value, alpha, beta, i_idx);
        }
    }
}

template <bool ASSUME_PACKED, typename src_data_t, typename acc_data_t, typename dst_data_t>
inline __device__ void naive_conv_wrw_ndhwc(const src_data_t* __restrict__ p_in,
                                            dst_data_t* __restrict__ p_wei,
                                            const double alpha,
                                            const double beta,
                                            const src_data_t* __restrict__ p_out,
                                            Strides6D in_strides,
                                            Strides6D wei_strides,
                                            Strides6D out_strides,
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

    if constexpr(ASSUME_PACKED)
    {
        p_in += static_cast<size_t>(ig) * c_per_group;

        p_wei += static_cast<size_t>(ig) * k_per_group * fz * fy * fx * c_per_group +
                 static_cast<size_t>(ik) * fz * fy * fx * c_per_group;

        p_out += static_cast<size_t>(ig) * k_per_group + static_cast<size_t>(ik);
    }
    else
    {
        p_in += static_cast<size_t>(ig) * in_strides[1];

        p_wei +=
            static_cast<size_t>(ig) * wei_strides[5] + static_cast<size_t>(ik) * wei_strides[4];

        p_out +=
            static_cast<size_t>(ig) * out_strides[1] + static_cast<size_t>(ik) * out_strides[0];
    }

    for(int tid = threadIdx.x; tid < thread_length; tid += blockDim.x)
    {
        int ic = tid % c_per_group;
        int ix = (tid / c_per_group) % fx;
        int iy = (tid / (c_per_group * fx)) % fy;
        int iz = (tid / (c_per_group * fx * fy));

        acc_data_t value = 0;

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

                            if constexpr(ASSUME_PACKED)
                            {
                                size_t i_idx = static_cast<size_t>(in) * di * hi * wi * c +
                                               static_cast<size_t>(cur_d) * hi * wi * c +
                                               static_cast<size_t>(cur_h) * wi * c +
                                               static_cast<size_t>(cur_w) * c +
                                               static_cast<size_t>(ic);

                                size_t o_idx = static_cast<size_t>(in) * do_ * ho * wo * k +
                                               static_cast<size_t>(ido) * ho * wo * k +
                                               static_cast<size_t>(iho) * wo * k +
                                               static_cast<size_t>(iwo) * k;

                                value += cast_to<src_data_t, acc_data_t>(p_in[i_idx]) *
                                         cast_to<src_data_t, acc_data_t>(p_out[o_idx]);
                            }
                            else
                            {

                                size_t i_idx = static_cast<size_t>(in) * in_strides[5] +
                                               static_cast<size_t>(cur_d) * in_strides[4] +
                                               static_cast<size_t>(cur_h) * in_strides[3] +
                                               static_cast<size_t>(cur_w) * in_strides[2] +
                                               static_cast<size_t>(ic) * in_strides[0];

                                size_t o_idx = static_cast<size_t>(in) * out_strides[5] +
                                               static_cast<size_t>(ido) * out_strides[4] +
                                               static_cast<size_t>(iho) * out_strides[3] +
                                               static_cast<size_t>(iwo) * out_strides[2];

                                value += cast_to<src_data_t, acc_data_t>(p_in[i_idx]) *
                                         cast_to<src_data_t, acc_data_t>(p_out[o_idx]);
                            }
                        }
                    }
                }
            }
        }

        if constexpr(ASSUME_PACKED)
        {
            size_t f_idx = static_cast<size_t>(iz) * fy * fx * c_per_group +
                           static_cast<size_t>(iy) * fx * c_per_group +
                           static_cast<size_t>(ix) * c_per_group + static_cast<size_t>(ic);
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, value, alpha, beta, f_idx);
        }
        else
        {
            size_t f_idx = static_cast<size_t>(iz) * wei_strides[3] +
                           static_cast<size_t>(iy) * wei_strides[2] +
                           static_cast<size_t>(ix) * wei_strides[1] +
                           static_cast<size_t>(ic) * wei_strides[0];
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, value, alpha, beta, f_idx);
        }
    }
}

#define DEFINE_2D_NAIVE_CONV_KERNEL(direction, tensor_layout, src_data_t, acc_data_t, dst_data_t)           \
    extern "C" __global__ void                                                                              \
        naive_conv_ab_packed_##direction##_##tensor_layout##_##src_data_t##_##acc_data_t##_##dst_data_t(    \
            src_data_t* __restrict__ p_in,                                                                  \
            src_data_t* __restrict__ p_wei,                                                                 \
            double alpha,                                                                                   \
            double beta,                                                                                    \
            dst_data_t* __restrict__ p_out,                                                                 \
            Strides5D in_strides,                                                                           \
            Strides5D wei_strides,                                                                          \
            Strides5D out_strides,                                                                          \
            int hi,                                                                                         \
            int wi,                                                                                         \
            int n,                                                                                          \
            int k_per_group,                                                                                \
            int c_per_group,                                                                                \
            int ho,                                                                                         \
            int wo,                                                                                         \
            int sy,                                                                                         \
            int sx,                                                                                         \
            int dy,                                                                                         \
            int dx,                                                                                         \
            int py,                                                                                         \
            int px,                                                                                         \
            int fy,                                                                                         \
            int fx,                                                                                         \
            int group)                                                                                      \
    {                                                                                                       \
        naive_conv_##direction##_##tensor_layout<true, src_data_t, acc_data_t, dst_data_t>(                 \
            p_in,                                                                                           \
            p_wei,                                                                                          \
            alpha,                                                                                          \
            beta,                                                                                           \
            p_out,                                                                                          \
            in_strides,                                                                                     \
            wei_strides,                                                                                    \
            out_strides,                                                                                    \
            hi,                                                                                             \
            wi,                                                                                             \
            n,                                                                                              \
            k_per_group,                                                                                    \
            c_per_group,                                                                                    \
            ho,                                                                                             \
            wo,                                                                                             \
            sy,                                                                                             \
            sx,                                                                                             \
            dy,                                                                                             \
            dx,                                                                                             \
            py,                                                                                             \
            px,                                                                                             \
            fy,                                                                                             \
            fx,                                                                                             \
            group);                                                                                         \
    }                                                                                                       \
    extern "C" __global__ void                                                                              \
        naive_conv_ab_nonpacked_##direction##_##tensor_layout##_##src_data_t##_##acc_data_t##_##dst_data_t( \
            src_data_t* __restrict__ p_in,                                                                  \
            src_data_t* __restrict__ p_wei,                                                                 \
            double alpha,                                                                                   \
            double beta,                                                                                    \
            dst_data_t* __restrict__ p_out,                                                                 \
            Strides5D in_strides,                                                                           \
            Strides5D wei_strides,                                                                          \
            Strides5D out_strides,                                                                          \
            int hi,                                                                                         \
            int wi,                                                                                         \
            int n,                                                                                          \
            int k_per_group,                                                                                \
            int c_per_group,                                                                                \
            int ho,                                                                                         \
            int wo,                                                                                         \
            int sy,                                                                                         \
            int sx,                                                                                         \
            int dy,                                                                                         \
            int dx,                                                                                         \
            int py,                                                                                         \
            int px,                                                                                         \
            int fy,                                                                                         \
            int fx,                                                                                         \
            int group)                                                                                      \
    {                                                                                                       \
        naive_conv_##direction##_##tensor_layout<false, src_data_t, acc_data_t, dst_data_t>(                \
            p_in,                                                                                           \
            p_wei,                                                                                          \
            alpha,                                                                                          \
            beta,                                                                                           \
            p_out,                                                                                          \
            in_strides,                                                                                     \
            wei_strides,                                                                                    \
            out_strides,                                                                                    \
            hi,                                                                                             \
            wi,                                                                                             \
            n,                                                                                              \
            k_per_group,                                                                                    \
            c_per_group,                                                                                    \
            ho,                                                                                             \
            wo,                                                                                             \
            sy,                                                                                             \
            sx,                                                                                             \
            dy,                                                                                             \
            dx,                                                                                             \
            py,                                                                                             \
            px,                                                                                             \
            fy,                                                                                             \
            fx,                                                                                             \
            group);                                                                                         \
    }

#define DEFINE_3D_NAIVE_CONV_KERNEL(direction, tensor_layout, src_data_t, acc_data_t, dst_data_t)           \
    extern "C" __global__ void                                                                              \
        naive_conv_ab_packed_##direction##_##tensor_layout##_##src_data_t##_##acc_data_t##_##dst_data_t(    \
            src_data_t* __restrict__ p_in,                                                                  \
            src_data_t* __restrict__ p_wei,                                                                 \
            double alpha,                                                                                   \
            double beta,                                                                                    \
            dst_data_t* __restrict__ p_out,                                                                 \
            Strides6D in_strides,                                                                           \
            Strides6D wei_strides,                                                                          \
            Strides6D out_strides,                                                                          \
            int di,                                                                                         \
            int hi,                                                                                         \
            int wi,                                                                                         \
            int n,                                                                                          \
            int k_per_group,                                                                                \
            int c_per_group,                                                                                \
            int do_,                                                                                        \
            int ho,                                                                                         \
            int wo,                                                                                         \
            int sz,                                                                                         \
            int sy,                                                                                         \
            int sx,                                                                                         \
            int dz,                                                                                         \
            int dy,                                                                                         \
            int dx,                                                                                         \
            int pz,                                                                                         \
            int py,                                                                                         \
            int px,                                                                                         \
            int fz,                                                                                         \
            int fy,                                                                                         \
            int fx,                                                                                         \
            int group)                                                                                      \
    {                                                                                                       \
        naive_conv_##direction##_##tensor_layout<true, src_data_t, acc_data_t, dst_data_t>(                 \
            p_in,                                                                                           \
            p_wei,                                                                                          \
            alpha,                                                                                          \
            beta,                                                                                           \
            p_out,                                                                                          \
            in_strides,                                                                                     \
            wei_strides,                                                                                    \
            out_strides,                                                                                    \
            di,                                                                                             \
            hi,                                                                                             \
            wi,                                                                                             \
            n,                                                                                              \
            k_per_group,                                                                                    \
            c_per_group,                                                                                    \
            do_,                                                                                            \
            ho,                                                                                             \
            wo,                                                                                             \
            sz,                                                                                             \
            sy,                                                                                             \
            sx,                                                                                             \
            dz,                                                                                             \
            dy,                                                                                             \
            dx,                                                                                             \
            pz,                                                                                             \
            py,                                                                                             \
            px,                                                                                             \
            fz,                                                                                             \
            fy,                                                                                             \
            fx,                                                                                             \
            group);                                                                                         \
    }                                                                                                       \
    extern "C" __global__ void                                                                              \
        naive_conv_ab_nonpacked_##direction##_##tensor_layout##_##src_data_t##_##acc_data_t##_##dst_data_t( \
            src_data_t* __restrict__ p_in,                                                                  \
            src_data_t* __restrict__ p_wei,                                                                 \
            double alpha,                                                                                   \
            double beta,                                                                                    \
            dst_data_t* __restrict__ p_out,                                                                 \
            Strides6D in_strides,                                                                           \
            Strides6D wei_strides,                                                                          \
            Strides6D out_strides,                                                                          \
            int di,                                                                                         \
            int hi,                                                                                         \
            int wi,                                                                                         \
            int n,                                                                                          \
            int k_per_group,                                                                                \
            int c_per_group,                                                                                \
            int do_,                                                                                        \
            int ho,                                                                                         \
            int wo,                                                                                         \
            int sz,                                                                                         \
            int sy,                                                                                         \
            int sx,                                                                                         \
            int dz,                                                                                         \
            int dy,                                                                                         \
            int dx,                                                                                         \
            int pz,                                                                                         \
            int py,                                                                                         \
            int px,                                                                                         \
            int fz,                                                                                         \
            int fy,                                                                                         \
            int fx,                                                                                         \
            int group)                                                                                      \
    {                                                                                                       \
        naive_conv_##direction##_##tensor_layout<false, src_data_t, acc_data_t, dst_data_t>(                \
            p_in,                                                                                           \
            p_wei,                                                                                          \
            alpha,                                                                                          \
            beta,                                                                                           \
            p_out,                                                                                          \
            in_strides,                                                                                     \
            wei_strides,                                                                                    \
            out_strides,                                                                                    \
            di,                                                                                             \
            hi,                                                                                             \
            wi,                                                                                             \
            n,                                                                                              \
            k_per_group,                                                                                    \
            c_per_group,                                                                                    \
            do_,                                                                                            \
            ho,                                                                                             \
            wo,                                                                                             \
            sz,                                                                                             \
            sy,                                                                                             \
            sx,                                                                                             \
            dz,                                                                                             \
            dy,                                                                                             \
            dx,                                                                                             \
            pz,                                                                                             \
            py,                                                                                             \
            px,                                                                                             \
            fz,                                                                                             \
            fy,                                                                                             \
            fx,                                                                                             \
            group);                                                                                         \
    }

DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nchw, float, double, float)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nchw, half, double, half)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nchw, ushort, double, ushort)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nchw, int8_t, int32_t, int8_t)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nchw, int8_t, int32_t, int32_t)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nchw, int8_t, int32_t, float)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nhwc, float, double, float)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nhwc, half, double, half)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nhwc, ushort, double, ushort)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nhwc, int8_t, int32_t, int8_t)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nhwc, int8_t, int32_t, int32_t)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nhwc, int8_t, int32_t, float)

DEFINE_2D_NAIVE_CONV_KERNEL(bwd, nchw, float, double, float)
DEFINE_2D_NAIVE_CONV_KERNEL(bwd, nchw, half, double, half)
DEFINE_2D_NAIVE_CONV_KERNEL(bwd, nchw, ushort, double, ushort)
DEFINE_2D_NAIVE_CONV_KERNEL(bwd, nhwc, float, double, float)
DEFINE_2D_NAIVE_CONV_KERNEL(bwd, nhwc, half, double, half)
DEFINE_2D_NAIVE_CONV_KERNEL(bwd, nhwc, ushort, double, ushort)

DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nchw, float, double, float)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nchw, half, double, half)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nchw, ushort, double, ushort)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nhwc, float, double, float)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nhwc, half, double, half)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nhwc, ushort, double, ushort)

DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ncdhw, float, double, float)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ncdhw, half, double, half)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ncdhw, ushort, double, ushort)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ncdhw, int8_t, int32_t, int32_t)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ncdhw, int8_t, int32_t, float)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ndhwc, float, double, float)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ndhwc, half, double, half)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ndhwc, ushort, double, ushort)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ndhwc, int8_t, int32_t, int32_t)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ndhwc, int8_t, int32_t, float)

DEFINE_3D_NAIVE_CONV_KERNEL(bwd, ncdhw, float, double, float)
DEFINE_3D_NAIVE_CONV_KERNEL(bwd, ncdhw, half, double, half)
DEFINE_3D_NAIVE_CONV_KERNEL(bwd, ncdhw, ushort, double, ushort)
DEFINE_3D_NAIVE_CONV_KERNEL(bwd, ndhwc, float, double, float)
DEFINE_3D_NAIVE_CONV_KERNEL(bwd, ndhwc, half, double, half)
DEFINE_3D_NAIVE_CONV_KERNEL(bwd, ndhwc, ushort, double, ushort)

DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ncdhw, float, double, float)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ncdhw, half, double, half)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ncdhw, ushort, double, ushort)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ndhwc, float, double, float)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ndhwc, half, double, half)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ndhwc, ushort, double, ushort)

/// \todo discuss whether we should split the kernels into separate files, or
/// figure out a mechanism to compile each kernel separately to reduce hipRTC
/// compilation times.  --amberhassaan
