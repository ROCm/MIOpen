// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/utility/common_header.hpp"

#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_weight.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/host_utility/kernel_launch.hpp"

#define ENABLE_PIPELINE_V2 1

template <typename T > struct Debug;

namespace ck {
template <typename T>
__device__ T warp_shuffle_down(const T& v_local, uint32_t lane_delta)
{
#if 0
    return  __shfl_down(v_local, lane_delta);
#elif 1
    static_assert(sizeof(T) == sizeof(int32_t), "wrong!");

    const int32_t v_remote_tmp = __builtin_amdgcn_ds_bpermute(
        (__lane_id() << 2) + (lane_delta << 2), bit_cast<int32_t>(v_local));

    return bit_cast<T>(v_remote_tmp);
#endif
}
template <typename T>
void __device__ global_atomic_add(T* p, T val);

template <>
void __device__ global_atomic_add<float>(float* p, float v)
{
    __builtin_amdgcn_global_atomic_fadd_f32(p, v);
}
template <>
void __device__ global_atomic_add<half2_t>(half2_t* p, half2_t v)
{
#if defined(__gfx942__) || defined(__gfx950__)
    __builtin_amdgcn_global_atomic_fadd_v2f16(p, v);
#else
    ignore = p;
    ignore = v;
#endif
}
template <>
void __device__ global_atomic_add<bhalf2_t>(bhalf2_t* p, bhalf2_t v)
{
#if defined(__gfx942__) || defined(__gfx950__)
    __builtin_amdgcn_global_atomic_fadd_v2bf16(p, v);
#else
    ignore = p;
    ignore = v;
#endif
}

template <typename GridwiseConvBwdWeight, index_t BlockSize, index_t MinimumOccupancy = 1>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(BlockSize, MinimumOccupancy)
#endif
    kernel_grouped_conv_bwd_weight_dl_v4(typename GridwiseConvBwdWeight::Argument arg)
{
    __shared__ char
        p_share_in[GridwiseConvBwdWeight::ShareMemInSize * GridwiseConvBwdWeight::NumTilePerBlock];
    __shared__ char p_share_out[GridwiseConvBwdWeight::ShareMemOutSize *
                                GridwiseConvBwdWeight::NumTilePerBlock];

    GridwiseConvBwdWeight::template Run(arg, p_share_in, p_share_out);
}

template <typename WeiDataType, typename AccDataType, index_t FilterSize>
__global__ void kernel_grouped_conv_bwd_weight_elementwise(
    WeiDataType* p_wei_grid,
    const AccDataType* p_acc_grid,
    std::array<index_t, 2 + 3> wei_g_k_c_xs_strides,
    std::array<index_t, 2 + 3> acc_g_k_c_xs_strides)
{
    const index_t g_idx       = __builtin_amdgcn_readfirstlane(blockIdx.x);
    if (threadIdx.x < FilterSize * FilterSize)
    {
        const index_t x = threadIdx.x % FilterSize;
        const index_t y = threadIdx.x / FilterSize;
        auto* p_wei = p_wei_grid + g_idx * wei_g_k_c_xs_strides[0] + y * wei_g_k_c_xs_strides[3] + x * wei_g_k_c_xs_strides[4];
        auto* p_acc = p_acc_grid + g_idx * acc_g_k_c_xs_strides[0] + y * acc_g_k_c_xs_strides[3] + x * acc_g_k_c_xs_strides[4];
        *p_wei = type_convert<WeiDataType>(*p_acc);
    }
}

namespace tensor_operation {
namespace device {

template <index_t BlockSize,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AccDataType,
          typename BlockTileSize, // input, without padding
          index_t FilterSize,     //
          typename FilterParam,   // tuple<dilation, stride, padding>
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          index_t NBatch,
          index_t NumWavePerTile,
          index_t InScalarPerVector,
          index_t OutScalarPerVector,
          index_t DstScalarPerVector,
          bool RequirePadding   = false,
          index_t WSplit        = 1,
          typename ComputeTypeA = InDataType,
          typename ComputeTypeB = ComputeTypeA>
struct GridwiseGroupedConv2DBwdWeightDlV4
{
    static constexpr index_t
    GetConvOut(index_t length, index_t filter, index_t dilation, index_t pad, index_t stride)
    {
        return (length + pad + pad - ((filter - 1) * dilation + 1)) / stride + 1;
    }
    template <index_t W, index_t ScalarPerVector>
    static constexpr index_t GetAlignedPackW()
    {
        constexpr index_t pakced_w = W / ScalarPerVector;
        if constexpr(pakced_w == 7)
        {
            return 8;
        }
        else
        {
            return WaveSize / (WaveSize / pakced_w);
        }
    }
    static constexpr index_t GetBatchPerWave() { return WaveSize / (FilterSize * FilterSize); }

    static constexpr index_t NDimSpatial       = 2;
    static constexpr index_t NumVectorPerPixel = NBatch / DstScalarPerVector;
    static constexpr auto I0                   = Number<0>{};
    static constexpr auto I1                   = Number<1>{};
    static constexpr auto I2                   = Number<2>{};
    static constexpr auto I3                   = Number<3>{};
    static constexpr index_t WaveSize          = 64;
    static constexpr index_t Tile_H            = BlockTileSize{}.At(I0);
    static constexpr index_t Tile_W            = BlockTileSize{}.At(I1);

    // Only support pad left = pad right for now
    static constexpr index_t Pad_H = tuple_element_t<2, FilterParam>{}.At(I0);
    static constexpr index_t Pad_W = tuple_element_t<2, FilterParam>{}.At(I1);

    static constexpr index_t Stride_H = tuple_element_t<1, FilterParam>{}.At(I0);
    static constexpr index_t Stride_W = tuple_element_t<1, FilterParam>{}.At(I1);

    // Only support dilation = 1 for now
    static constexpr index_t Dilation_Y = tuple_element_t<0, FilterParam>{}.At(I0);
    static constexpr index_t Dilation_X = tuple_element_t<0, FilterParam>{}.At(I1);

    static constexpr index_t Filter_Y = FilterSize;
    static constexpr index_t Filter_X = FilterSize;

    static constexpr index_t TileIn_H = Tile_H + 2 * Pad_H;
    static constexpr index_t TileIn_W = Tile_W + 2 * Pad_W;

    static constexpr index_t TileOut_H =
        GetConvOut(Tile_H, FilterSize, Dilation_Y, Pad_H, Stride_H);
    static constexpr index_t TileOut_W =
        GetConvOut(Tile_W, FilterSize, Dilation_X, Pad_W, Stride_W);

    static_assert(Tile_W % InScalarPerVector == 0);
    static_assert(TileOut_W % (OutScalarPerVector * WSplit) == 0);

    static constexpr index_t SubTileOut_W = TileOut_W / WSplit;
    static constexpr index_t SubTileIn_Max_W = math::max(
        math::integer_least_multiple(SubTileOut_W * Stride_W + (Filter_X - 1) * Dilation_X - Pad_W,
                                     InScalarPerVector) +
            Pad_W,
        math::integer_least_multiple(SubTileOut_W* Stride_W + 2 * Pad_W, InScalarPerVector));

    static constexpr index_t SubTileIn_Stride  = SubTileIn_Max_W;
    static constexpr index_t SubTileOut_Stride = SubTileOut_W;

    static constexpr index_t SubTileIn_Pack_W  = (WSplit == 1) ?  GetAlignedPackW<Tile_W, InScalarPerVector>() :  GetAlignedPackW<SubTileIn_Max_W, InScalarPerVector>();
    static constexpr index_t TileIn_Pack_Group = WaveSize / SubTileIn_Pack_W;
    static constexpr index_t TileIn_Pack_H = math::integer_divide_ceil(Tile_H, TileIn_Pack_Group);
    static constexpr index_t TileIn_Align_H =
        math::max(TileIn_Pack_H * TileIn_Pack_Group + Pad_H, TileIn_H);

    static constexpr index_t SubTileOut_Pack_W =
        GetAlignedPackW<SubTileOut_W, OutScalarPerVector>();
    static constexpr index_t TileOut_Pack_Group = WaveSize / SubTileOut_Pack_W;
    static constexpr index_t TileOut_Pack_H =
        math::integer_divide_ceil(TileOut_H, TileOut_Pack_Group);

    static constexpr index_t BatchPerWave      = GetBatchPerWave();
    static constexpr index_t BatchPerTile      = BatchPerWave * NumWavePerTile;
    static constexpr index_t TileOut_HPerBatch = math::integer_divide_ceil(TileOut_H, BatchPerTile);
    static constexpr index_t TileOut_Align_H   = math::max(
        TileOut_Pack_H * TileOut_Pack_Group, TileOut_HPerBatch* BatchPerWave* NumWavePerTile);
    static constexpr index_t ShareMemInSize =
        TileIn_Align_H * SubTileIn_Stride * sizeof(InDataType) * NBatch;
    static constexpr index_t ShareMemOutSize =
        TileOut_Align_H * SubTileOut_Stride * sizeof(OutDataType) * NBatch;

    static constexpr index_t NumTilePerBlock = BlockSize / WaveSize / NumWavePerTile;

    using InDataVector  = typename vector_type<InDataType, DstScalarPerVector>::type;
    using OutDataVector = typename vector_type<OutDataType, DstScalarPerVector>::type;

    template <index_t TileH,
              index_t AlignedPackW,
              index_t ScalarPerVector,
              typename SrcType,
              typename DestVector>
    static void __device__ load_data_from_global(const SrcType* p,
                                                 index_t x,
                                                 index_t y_offset,
                                                 index_t n_stride,
                                                 index_t h,
                                                 index_t w,
                                                 index_t h_stride,
                                                 index_t w_stride,
                                                 DestVector* p_scratch)
    {
        ignore = h;
        ignore = w;

        using SrcVector = typename vector_type<SrcType, ScalarPerVector>::type;

        static_assert((AlignedPackW & (AlignedPackW - 1)) == 0, "aligned width is not power 2!");

        constexpr index_t NumGroup     = WaveSize / AlignedPackW;
        constexpr index_t AlignedPackH = math::integer_divide_ceil(TileH, NumGroup);
        constexpr index_t PackH        = TileH / NumGroup;

        auto get_offset = [&](index_t y_, index_t packed_x_, index_t n_) {
            return (y_ * h_stride + packed_x_ * ScalarPerVector * w_stride + n_ * n_stride) /
                   ScalarPerVector;
        };

        // todo: check with real width/height
        // and use OOB to avoid tynamic control flow.
        auto* p_base = reinterpret_cast<const SrcVector*>(p);

        static_for<0, PackH, 1>{}([&](auto i) {
            const index_t y = y_offset + i * NumGroup;
            // load data
            SrcVector tmp[NBatch];
            static_for<0, NBatch, 1>{}([&](auto n) {
                const index_t offset = get_offset(y, x, n);
                tmp[n]               = p_base[offset];
            });

            // interleave data
            auto* p_scratch_base = p_scratch + i * NumVectorPerPixel * ScalarPerVector;
            if constexpr(DstScalarPerVector == 1)
            {
                static_assert(NBatch == 1);
                static_for<0, ScalarPerVector, 1>{}(
                    [&](auto j) { p_scratch_base[j * NumVectorPerPixel] = tmp[0][j.value]; });
            }
            else if constexpr(ScalarPerVector == 1)
            {
                static_assert(DstScalarPerVector > 1);
                static_for<0, NBatch, 1>{}([&](auto n) {
                    p_scratch_base[n / DstScalarPerVector][n % DstScalarPerVector] = tmp[n];
                });
            }
            else
            {
                static_for<0, ScalarPerVector, 1>{}([&](auto j) {
                    static_for<0, NBatch, 1>{}([&](auto n) {
                        p_scratch_base[j * NumVectorPerPixel + n / DstScalarPerVector]
                                      [n % DstScalarPerVector] = tmp[n][j.value];
                    });
                });
            }
        });

        if constexpr(AlignedPackH != PackH)
        {
            if(y_offset < (TileH - NumGroup * PackH))
            {
                constexpr auto i = PackH;
                const index_t y  = y_offset + i * NumGroup;
                // load data
                SrcVector tmp[NBatch];
                static_for<0, NBatch, 1>{}([&](auto n) {
                    const index_t offset = get_offset(y, x, n);
                    tmp[n]               = p_base[offset];
                });

                // interleave data
                auto* p_scratch_base = p_scratch + i * NumVectorPerPixel * ScalarPerVector;
                if constexpr(DstScalarPerVector == 1)
                {
                    static_assert(NBatch == 1);
                    static_for<0, ScalarPerVector, 1>{}(
                        [&](auto j) { p_scratch_base[j * NumVectorPerPixel] = tmp[0][j.value]; });
                }
                else if constexpr(ScalarPerVector == 1)
                {
                    static_for<0, NBatch, 1>{}([&](auto n) {
                        p_scratch_base[n / DstScalarPerVector][n % DstScalarPerVector] = tmp[n];
                    });
                }
                else
                {
                    static_for<0, ScalarPerVector, 1>{}([&](auto j) {
                        static_for<0, NBatch, 1>{}([&](auto n) {
                            p_scratch_base[j * NumVectorPerPixel + n / DstScalarPerVector]
                                          [n % DstScalarPerVector] = tmp[n][j.value];
                        });
                    });
                }
            }
        }
    }

    // todo handle pading in p_sharemem
    template <index_t TileH,
              index_t AlignedPackW,
              index_t TileW_Stride,
              index_t ScalarPerVector,
              typename DestVector>
    static void __device__ write_data_to_lds(index_t x,
                                             index_t y_offset,
                                             const DestVector* p_scratch,
                                             DestVector* p_sharemem)
    {
        static_assert(AlignedPackW <= WaveSize);

        constexpr index_t NumGroup     = WaveSize / AlignedPackW;
        constexpr index_t AlignedPackH = math::integer_divide_ceil(TileH, NumGroup);
        constexpr index_t PackH        = TileH / NumGroup;

        auto get_offset = [&](index_t y_, index_t x_) {
            return y_ * TileW_Stride * NumVectorPerPixel + x_ * NumVectorPerPixel * ScalarPerVector;
        };

        static_for<0, PackH, 1>{}([&](auto i) {
            const index_t y      = y_offset + i * NumGroup;
            const index_t offset = get_offset(y, x);
            static_for<0, NumVectorPerPixel * ScalarPerVector, 1>{}([&](auto j) {
                p_sharemem[offset + j] = p_scratch[i * NumVectorPerPixel * ScalarPerVector + j];
            });
        });
        if constexpr(AlignedPackH != PackH)
        {
            if(y_offset < (TileH - NumGroup * PackH))
            {
                constexpr auto i     = PackH;
                const index_t y      = y_offset + i * NumGroup;
                const index_t offset = get_offset(y, x);
                static_for<0, NumVectorPerPixel * ScalarPerVector, 1>{}([&](auto j) {
                    p_sharemem[offset + j] = p_scratch[i * NumVectorPerPixel * ScalarPerVector + j];
                });
            }
        }
    }

    template <index_t TileH, index_t SubTileOut_W>
    static void __device__ run_conv_bwd_weight(index_t x,
                                               index_t y,
                                               index_t h,
                                               index_t w,
                                               index_t hout_base,
                                               InDataVector* p_share_in,
                                               OutDataVector* p_share_out,
                                               AccDataType& acc)
    {
        ignore      = h;
        ignore      = w;
        auto get_in = [&](index_t ho_, index_t wo_, index_t i_) {
            index_t hi = (ho_ + hout_base) * Stride_H + y * Dilation_Y;
            index_t wi = wo_ * Stride_W + x * Dilation_X;
            return p_share_in[(hi * SubTileIn_Stride + wi) * NumVectorPerPixel + i_];
        };
        auto get_out = [&](index_t ho_, index_t wo_, index_t i_) {
            return p_share_out[((ho_ + hout_base) * SubTileOut_Stride + wo_) * NumVectorPerPixel +
                               i_];
        };
        if(x < Filter_X && y < Filter_Y)
        {
            if constexpr(TileOut_W % 2 == 0 && DstScalarPerVector == 1)
            {
                static_assert(NumVectorPerPixel == 1);
                static_for<0, TileH, 1>{}([&](auto ho) {
                    static_for<0, SubTileOut_W, 2>{}([&](auto wo) {
                        typename vector_type<InDataType, 2>::type v_in;
                        typename vector_type<OutDataType, 2>::type v_out;
                        v_in[0]  = get_in(ho, wo, 0);
                        v_in[1]  = get_in(ho, wo + 1, 0);
                        v_out[0] = get_out(ho, wo, 0);
                        v_out[1] = get_out(ho, wo + 1, 0);
                        inner_product(v_in, v_out, acc);
                    });
                });
            }
            else
            {
                ///for (index_t wo = 0; wo < SubTileOut_W; wo ++) {
                 //   for (index_t ho = 0; ho < TileH; ho ++) {
                static_for<0, TileH, 1>{}([&](auto ho) {
                    static_for<0, SubTileOut_W, 1>{}([&](auto wo) {
                    
                        static_for<0, NumVectorPerPixel, 1>{}([&](auto i) {
                            auto v_in  = get_in(ho, wo, i);
                            auto v_out = get_out(ho, wo, i);
                            inner_product(v_in, v_out, acc);
#if 0
                             if (x == 0 && y == 0)
                             {
                                uint32_t * pin = reinterpret_cast<uint32_t*>(&v_in);
                                uint32_t * pout = reinterpret_cast<uint32_t*>(&v_out);
                               printf("threadIdx = %u, h w [%d %d] vin %08x vout %08x  acc = %f\n", threadIdx.x, ho+ hout_base, wo, pin[0],  pout[0], acc);
                            }
#endif                            
                        });
                    });
                });
            }
        }
    }

    template <typename Argument>
    static void __device__
    write_output(const Argument& arg, index_t g, index_t y, index_t x, AccDataType acc)
    {
        const index_t Wei_G_Stride = arg.wei_g_k_c_xs_strides_[0];
        const index_t Y_Stride     = arg.wei_g_k_c_xs_strides_[3];
        const index_t X_Stride     = arg.wei_g_k_c_xs_strides_[4];
        if(y < Filter_Y && x < Filter_X)
        {
            if(arg.enable_k_split_)
            {
                auto p_acc = arg.p_acc_grid_ + Wei_G_Stride * g + y * Y_Stride + x * X_Stride;
                global_atomic_add(p_acc, acc);
            }
            else
            {
                auto p_wei = arg.p_wei_grid_ + Wei_G_Stride * g + y * Y_Stride + x * X_Stride;
                *p_wei = type_convert<WeiDataType>(acc);
            }
        }
    }
    template <typename DstVector>
    static void __device__ dump_lds(DstVector* p, index_t totalcount, index_t length)
    {
        for(index_t i = 0; i < totalcount; i++)
        {
            if(i % length == 0)
            {
                printf("\n [%d]", i / length);
            }
            if constexpr(sizeof(DstVector) >= 4)
            {
                uint32_t* p1 = reinterpret_cast<uint32_t*>(&p[i]);
                static_for<0, sizeof(DstVector) / sizeof(uint32_t), 1>{}(
                    [&](auto j) { printf("%08x ", p1[j]); });
            }
            else
            {
                uint16_t* p1 = reinterpret_cast<uint16_t*>(&p[i]);
                static_for<0, sizeof(DstVector) / sizeof(uint16_t), 1>{}(
                    [&](auto j) { printf("%04x ", p1[j]); });
            }
        }
        printf("\n");
    }
    static constexpr index_t TotalLdsSize()
    {
        return (ShareMemInSize + ShareMemOutSize) * NumTilePerBlock;
    }

    template <typename Argument>
    static void __device__ Run(Argument arg, char* p_share_in, char* p_share_out)
    {
        const index_t g_idx       = __builtin_amdgcn_readfirstlane(blockIdx.x);
        const index_t k_split_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
        const index_t wave_id     = __builtin_amdgcn_readfirstlane(threadIdx.x / WaveSize);
        const index_t tile_id     = wave_id / NumWavePerTile;
        const index_t lane_id     = __lane_id();

        constexpr index_t ThreadPerBatch = WaveSize / BatchPerWave;

        static_assert(Tile_H % NumWavePerTile == 0);
        static_assert(TileOut_H % NumWavePerTile == 0);
        InDataVector tmp_in[math::integer_divide_ceil(TileIn_Pack_H, NumWavePerTile) *
                            NumVectorPerPixel * InScalarPerVector]    = {};
        OutDataVector tmp_out[math::integer_divide_ceil(TileOut_Pack_H, NumWavePerTile) *
                              NumVectorPerPixel * OutScalarPerVector] = {};

        static_assert(NumTilePerBlock == 1 || NumWavePerTile == 1);

        static constexpr index_t spatial_offset = 3;
        const index_t n                         = arg.in_g_n_c_wis_lengths_[1];
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
        n_idx += n * k_split_idx;

        // In
        const index_t hi = arg.in_g_n_c_wis_lengths_[spatial_offset + 0];
        const index_t wi = arg.in_g_n_c_wis_lengths_[spatial_offset + 1];

        const index_t hi_stride   = arg.in_g_n_c_wis_strides_[spatial_offset + 0];
        const index_t wi_stride   = arg.in_g_n_c_wis_strides_[spatial_offset + 1];
        const index_t in_g_stride = arg.in_g_n_c_wis_strides_[0];
        const index_t in_n_stride = arg.in_g_n_c_wis_strides_[1];

        // Out
        const index_t ho = arg.out_g_n_k_wos_lengths_[spatial_offset + 0];
        const index_t wo = arg.out_g_n_k_wos_lengths_[spatial_offset + 1];

        const index_t ho_stride    = arg.out_g_n_k_wos_strides_[spatial_offset + 0];
        const index_t wo_stride    = arg.out_g_n_k_wos_strides_[spatial_offset + 1];
        const index_t out_g_stride = arg.out_g_n_k_wos_strides_[0];
        const index_t out_n_stride = arg.out_g_n_k_wos_strides_[1];

        // Wei
        auto* p_in  = arg.p_in_grid_ + g_idx * in_g_stride + n_idx * in_n_stride;
        auto* p_out = arg.p_out_grid_ + g_idx * out_g_stride + n_idx * out_n_stride;

        constexpr index_t Copy_Tile_H    = Tile_H / NumWavePerTile;
        constexpr index_t Copy_TileOut_H = TileOut_H / NumWavePerTile;
        if constexpr(NumWavePerTile > 1)
        {
            static_assert(RequirePadding == false);
            p_in += Copy_Tile_H * hi_stride * (wave_id % NumWavePerTile);
            p_out += Copy_TileOut_H * ho_stride * (wave_id % NumWavePerTile);
        }

        InDataVector* share_in   = reinterpret_cast<InDataVector*>(p_share_in);
        OutDataVector* share_out = reinterpret_cast<OutDataVector*>(p_share_out);
        if constexpr(NumTilePerBlock > 1)
        {
            share_in  = reinterpret_cast<InDataVector*>(p_share_in + ShareMemInSize * wave_id);
            share_out = reinterpret_cast<InDataVector*>(p_share_out + ShareMemOutSize * wave_id);
        }
        auto share_in_base  = share_in;
        auto share_out_base = share_out;

        // init lds 0
        index_t cluster_id = threadIdx.x % (WaveSize * NumWavePerTile);
        auto init_pading   = [&](auto* share_vec, auto count) {
            static_for<0, math::integer_divide_ceil(count, WaveSize * NumWavePerTile), 1>{}(
                [&](auto i) {
                    if(cluster_id + i * WaveSize * NumWavePerTile < count)
                    {
                        share_vec[cluster_id + i * WaveSize * NumWavePerTile] = {};
                    }
                });
        };
        auto init_array_pading = [&](auto* share_vec,
                                     auto element_count,
                                     auto array_count,
                                     index_t stride) {
            static_for<0, math::integer_divide_ceil(array_count, WaveSize * NumWavePerTile), 1>{}(
                [&](auto i) {
                    static_for<0, element_count, 1>{}([&](auto j) {
                        if(cluster_id + i * WaveSize * NumWavePerTile < array_count)
                        {              
                            auto p = share_vec +
                                     (cluster_id + i * WaveSize * NumWavePerTile) * stride + j;
                            //printf("threadIdx %d %d %d\n", cluster_id, array_count.value, static_cast<index_t>(reinterpret_cast<char*>(p) - p_share_in));
                            *p = {};
                        }
                    });
                });
        };
        constexpr index_t TopPadingSize = Pad_H * SubTileIn_Stride * NumVectorPerPixel;
        constexpr index_t TileInEnd     = (Tile_H + Pad_H) * SubTileIn_Stride;
        constexpr index_t ButtomPaddingSize =
            (ShareMemInSize / (sizeof(InDataType) * NBatch) - TileInEnd) * NumVectorPerPixel;
        static_assert(ButtomPaddingSize >= 0);
        if constexpr(Pad_H > 0)
        {
            init_pading(share_in, Number<TopPadingSize>{});
            init_pading(share_in + TileInEnd * NumVectorPerPixel, Number<ButtomPaddingSize>{});
        }

        constexpr index_t TileOutEnd = TileOut_H * SubTileOut_Stride;
        constexpr index_t OutButtomPaddingSize =
            (ShareMemOutSize / (sizeof(OutDataType) * NBatch) - TileOutEnd) * NumVectorPerPixel;
        init_pading(share_out + TileOutEnd * NumVectorPerPixel, Number<OutButtomPaddingSize>{});

        if constexpr(NumWavePerTile > 1)
        {
            block_sync_lds();
        }

        const index_t in_x         = lane_id % SubTileIn_Pack_W;
        const index_t in_y_offset  = lane_id / SubTileIn_Pack_W;
        const index_t out_x        = lane_id % SubTileOut_Pack_W;
        const index_t out_y_offset = lane_id / SubTileOut_Pack_W;

        // adjust share memory offset for copy
        if constexpr(NumWavePerTile > 1)
        {
            static_assert(RequirePadding == false);
            share_in += Copy_Tile_H * SubTileIn_Stride * NumVectorPerPixel * wave_id;
            share_out += Copy_TileOut_H * SubTileOut_Stride * NumVectorPerPixel * wave_id;
        }
        share_in += (SubTileIn_Stride * Pad_H) * NumVectorPerPixel;

        constexpr index_t TileOut_H_batch = math::integer_divide_ceil(Copy_TileOut_H, BatchPerWave);
        index_t hout_base                 = lane_id / ThreadPerBatch * TileOut_H_batch;
        if constexpr(NumWavePerTile > 1)
        {
            hout_base += Copy_TileOut_H * wave_id;
        }
        index_t x = (lane_id % ThreadPerBatch) % Filter_X;
        index_t y = (lane_id % ThreadPerBatch) / Filter_X;
        if (lane_id/ThreadPerBatch >= BatchPerWave)
        {
            y = Filter_Y;
        }
        float acc = 0;
//
        auto p_in_base =  p_in;
        auto p_out_base = p_out;
        auto num_loop_base = num_loop;
        static_for<0, WSplit, 1>{}([&](auto subtile_idx) {
            p_in = p_in_base;
            p_out = p_out_base;
            num_loop = num_loop_base;
            constexpr index_t out_left  = SubTileOut_W * subtile_idx;
            constexpr index_t out_right = SubTileOut_W * (subtile_idx + 1);

            constexpr index_t in_left  = out_left * Stride_W;
            constexpr index_t in_right = out_right * Stride_W + (Filter_X - 1) * Dilation_X;
            static_assert(in_left % InScalarPerVector == 0);

            constexpr auto SubTileIn = [&]() {
                if constexpr(WSplit == 1)
                {
                    constexpr auto in_share_offset = Pad_W;
                    constexpr auto in_mem_offset   = 0;
                    constexpr auto in_subtile_w    = Tile_W;
                    constexpr auto in_share_base_offset = 0;
                    return Sequence<in_mem_offset, in_share_offset, in_subtile_w, in_share_base_offset>{};
                }
                else if constexpr(subtile_idx == 0)
                {
                    constexpr auto in_share_offset = Pad_W;
                    constexpr auto in_mem_offset   = 0;
                    constexpr auto in_subtile_w =
                        math::integer_least_multiple(in_right - Pad_W, InScalarPerVector);
                    constexpr auto in_share_base_offset = 0;
                    return Sequence<in_mem_offset, in_share_offset, in_subtile_w, in_share_base_offset>{};
                }
                else if constexpr(subtile_idx == WSplit - 1)
                {
                    constexpr auto in_subtile_w    = math::integer_least_multiple(in_right - in_left - Pad_W, InScalarPerVector);
                    constexpr auto in_left_pading  = Tile_W - in_subtile_w;
                    constexpr auto in_mem_offset   = in_left_pading;
                    constexpr auto in_share_offset = 0;
                    static_assert(in_right == TileIn_W);
                    static_assert(in_subtile_w % InScalarPerVector == 0);
                    constexpr auto in_share_base_offset = in_left - Pad_W - in_left_pading;
                    static_assert(in_share_base_offset >= 0);
                    //Debug<Sequence<in_left, in_left_pading, in_share_base_offset> > xx3;
                    return Sequence<in_mem_offset, in_share_offset, in_subtile_w, in_share_base_offset>{};
                }
                else
                {
                    constexpr auto in_share_offset = 0;
                    constexpr auto in_mem_offset   = in_left - Pad_W;
                    constexpr auto in_subtile_w =
                        math::integer_least_multiple(in_right - in_left, InScalarPerVector);
                     constexpr auto in_share_base_offset = 0;
                    return Sequence<in_mem_offset, in_share_offset, in_subtile_w, in_share_base_offset>{};
                }
            }();
            constexpr index_t in_mem_offset   = SubTileIn.At(I0);
            constexpr index_t in_share_offset = SubTileIn.At(I1);
            constexpr index_t in_subtile_w     = SubTileIn.At(I2);
            constexpr index_t in_share_base_offset     = SubTileIn.At(I3);
            if constexpr(subtile_idx == 0)
            {
                if constexpr(Pad_W > 0)
                {
                    init_array_pading(share_in_base + TopPadingSize,
                                      Number<Pad_W * NumVectorPerPixel>{},
                                      Number<Tile_H>{},
                                      SubTileIn_Stride * NumVectorPerPixel);
                     //block_sync_lds();     
                    // Debug<Number<ShareMemInSize>> xx2;            
                    //if (threadIdx.x == 0)
                    //{
                   //      dump_lds(reinterpret_cast<InDataVector*>(p_share_in), ShareMemInSize/sizeof(InDataVector), SubTileIn_Stride * NumVectorPerPixel);
                    //}
                }
            }

            if constexpr(subtile_idx == WSplit - 1)
            {
                if constexpr(Pad_W > 0)
                {
                    constexpr index_t right =
                        (subtile_idx == 0) ? in_subtile_w + Pad_W : in_subtile_w;
                    init_array_pading(share_in_base + TopPadingSize + right * NumVectorPerPixel,
                                      Number<Pad_W * NumVectorPerPixel>{},
                                      Number<Tile_H>{},
                                      SubTileIn_Stride * NumVectorPerPixel);
                }
            }

            // prefetch 0
            if(in_x < (in_subtile_w / InScalarPerVector))
            {
                load_data_from_global<Copy_Tile_H, SubTileIn_Pack_W, InScalarPerVector>(
                    p_in + in_mem_offset,
                    in_x,
                    in_y_offset,
                    in_n_stride,
                    hi,
                    wi,
                    hi_stride,
                    wi_stride,
                    tmp_in);
            }
            if(out_x < (SubTileOut_W / OutScalarPerVector))
            {
                load_data_from_global<Copy_TileOut_H, SubTileOut_Pack_W, OutScalarPerVector>(
                    p_out + out_left,
                    out_x,
                    out_y_offset,
                    out_n_stride,
                    ho,
                    wo,
                    ho_stride,
                    wo_stride,
                    tmp_out);
            }
            p_in += NBatch * in_n_stride;
            p_out += NBatch * out_n_stride;

            if(in_x < (in_subtile_w / InScalarPerVector))
            {
                write_data_to_lds<Copy_Tile_H,
                                  SubTileIn_Pack_W,
                                  SubTileIn_Stride,
                                  InScalarPerVector>(
                    in_x, in_y_offset, tmp_in, share_in + in_share_offset * NumVectorPerPixel);
            }
            if(out_x < (SubTileOut_W / OutScalarPerVector))
            {
                write_data_to_lds<Copy_TileOut_H,
                                  SubTileOut_Pack_W,
                                  SubTileOut_Stride,
                                  OutScalarPerVector>(
                    out_x, out_y_offset, tmp_out, share_out);
            }
#if 0
            if (lane_id == 0)
            {
                printf("share in %d share base %d  share out %d share out base %d\n", static_cast<index_t>(reinterpret_cast<char*>(share_in) - p_share_in),
                  static_cast<index_t>(reinterpret_cast<char*>(share_in_base) - p_share_in),  static_cast<index_t>(reinterpret_cast<char*>(share_out) - p_share_out),
                   static_cast<index_t>(reinterpret_cast<char*>(share_out_base) - p_share_out));
            //     dump_lds(tmp_in, sizeof(tmp_in)/sizeof(InDataVector), sizeof(tmp_in)/sizeof(InDataVector));
            //     dump_lds(tmp_out, sizeof(tmp_out)/sizeof(OutDataVector), sizeof(tmp_out)/sizeof(OutDataVector));
            }
            //block_sync_lds();
            
            if (threadIdx.x == 0)
            {
               printf("sub tile size %d %d \n", in_subtile_w, SubTileOut_W);
               dump_lds(reinterpret_cast<InDataVector*>(share_in_base), ShareMemInSize/sizeof(InDataVector), SubTileIn_Stride * NumVectorPerPixel);
               dump_lds(reinterpret_cast<OutDataVector*>(share_out_base), ShareMemOutSize/sizeof(OutDataVector), SubTileOut_Stride * NumVectorPerPixel);
            }
            block_sync_lds();
#endif

#if defined(ENABLE_PIPELINE_V2)
            while(num_loop > 0)
            {
                // do conv_bwd on 0
                if constexpr(NumWavePerTile > 1)
                {
                    block_sync_lds();
                }
                run_conv_bwd_weight<TileOut_H_batch, SubTileOut_W>(
                    x, y, ho, wo, hout_base, share_in_base + in_share_base_offset * NumVectorPerPixel, share_out_base, acc);
                if constexpr(NumWavePerTile > 1)
                {
                    block_sync_lds();
                }
                if(in_x < (in_subtile_w / InScalarPerVector))
                {
                    load_data_from_global<Copy_Tile_H, SubTileIn_Pack_W, InScalarPerVector>(
                        p_in + in_mem_offset,
                        in_x,
                        in_y_offset,
                        in_n_stride,
                        hi,
                        wi,
                        hi_stride,
                        wi_stride,
                        tmp_in);

                    write_data_to_lds<Copy_Tile_H,
                                      SubTileIn_Pack_W,
                                      SubTileIn_Stride,
                                      InScalarPerVector>(
                        in_x, in_y_offset, tmp_in, share_in + in_share_offset * NumVectorPerPixel);
                }

                if(out_x < (SubTileOut_W / OutScalarPerVector))
                {
                    load_data_from_global<Copy_TileOut_H, SubTileOut_Pack_W, OutScalarPerVector>(
                        p_out + out_left,
                        out_x,
                        out_y_offset,
                        out_n_stride,
                        ho,
                        wo,
                        ho_stride,
                        wo_stride,
                        tmp_out);
                    write_data_to_lds<Copy_TileOut_H,
                                      SubTileOut_Pack_W,
                                      SubTileOut_Stride,
                                      OutScalarPerVector>(
                        out_x, out_y_offset, tmp_out, share_out);
                }

                p_in += NBatch * in_n_stride;
                p_out += NBatch * out_n_stride;
                num_loop--;
            };
#else
            while(num_loop > 0)
            {
                if(in_x < (in_subtile_w / InScalarPerVector))
                {
                    load_data_from_global<Copy_Tile_H, SubTileIn_Pack_W, InScalarPerVector>(
                        p_in + in_mem_offset,
                        in_x,
                        in_y_offset,
                        in_n_stride,
                        hi,
                        wi,
                        hi_stride,
                        wi_stride,
                        tmp_in);
                }
                if(out_x < (SubTileOut_W / OutScalarPerVector))
                {
                    load_data_from_global<Copy_TileOut_H, SubTileOut_Pack_W, OutScalarPerVector>(
                        p_out + out_left,
                        out_x,
                        out_y_offset,
                        out_n_stride,
                        ho,
                        wo,
                        ho_stride,
                        wo_stride,
                        tmp_out);
                }
                p_in += NBatch * in_n_stride;
                p_out += NBatch * out_n_stride;

                // do conv_bwd on 0
                if constexpr(NumWavePerTile > 1)
                {
                    block_sync_lds();
                }
                run_conv_bwd_weight<TileOut_H_batch, SubTileOut_W>(
                    x, y, ho, wo, hout_base, share_in_base + in_share_base_offset * NumVectorPerPixel, share_out_base, acc);
                if constexpr(NumWavePerTile > 1)
                {
                    block_sync_lds();
                }
                // write 0
                if(in_x < (in_subtile_w / InScalarPerVector))
                {
                    write_data_to_lds<Copy_Tile_H,
                                      SubTileIn_Pack_W,
                                      SubTileIn_Stride,
                                      InScalarPerVector>(
                        in_x, in_y_offset, tmp_in, share_in + in_share_offset * NumVectorPerPixel);
                }
                if(out_x < (SubTileOut_W / OutScalarPerVector))
                {
                    write_data_to_lds<Copy_TileOut_H,
                                      SubTileOut_Pack_W,
                                      SubTileOut_Stride,
                                      OutScalarPerVector>(
                        out_x, out_y_offset, tmp_out, share_out);
                }
                num_loop--;
            };
#endif
            // tail
            {
                if constexpr(NumWavePerTile > 1)
                {
                    block_sync_lds();
                }
                run_conv_bwd_weight<TileOut_H_batch, SubTileOut_W>(
                    x, y, ho, wo, hout_base, share_in_base + in_share_base_offset * NumVectorPerPixel, share_out_base, acc);
                             if constexpr(NumWavePerTile > 1)
                {
                    block_sync_lds();
                }
            }
        });

        if constexpr(ThreadPerBatch == 32)
        {
            float acc_2 = warp_shuffle_down(acc, ThreadPerBatch);
        #if 0
           if (lane_id == 0)
            {
                printf("acc %f %f  \n", acc, acc_2);
            }
        #endif
            acc += acc_2;
        }
        else if constexpr(ThreadPerBatch == 9)
        {
            // todo optimization reduce operation.
            float acc_2 = warp_shuffle_down(acc, ThreadPerBatch);
            float acc_3 = warp_shuffle_down(acc, 2 * ThreadPerBatch);
            float acc_4 = warp_shuffle_down(acc, 3 * ThreadPerBatch);
            float acc_5 = warp_shuffle_down(acc, 4 * ThreadPerBatch);
            float acc_6 = warp_shuffle_down(acc, 5 * ThreadPerBatch);
            float acc_7 = warp_shuffle_down(acc, 6 * ThreadPerBatch);
            #if 0
            block_sync_lds();
            if (lane_id == 0)
            {
                printf("acc %f %f %f %f %f %f %f \n", acc, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7);
            }
            #endif
             acc += acc_2 + acc_3 + acc_4 + acc_5 + acc_6 + acc_7;
        }
        if constexpr(NumTilePerBlock == 1 && NumWavePerTile == 1)
        {
            if(hout_base == 0)
            {
                write_output(arg, g_idx, y, x, acc);
            }
        }
        else
        {
            uint32_t* p_share_acc = reinterpret_cast<uint32_t*>(p_share_in);
            block_sync_lds();
            p_share_acc[threadIdx.x] = bit_cast<uint32_t>(acc);
            block_sync_lds();
            if(hout_base == 0 && wave_id == 0)
            {
                for(int i = 1; i < NumTilePerBlock * NumWavePerTile; i++)
                {
                    acc += bit_cast<float>(p_share_acc[i * WaveSize + lane_id]);
                }
                write_output(arg, g_idx, y, x, acc);
            }
        }
    }
    struct Argument
    {
        Argument(const InDataType* p_in_grid,
                 WeiDataType* p_wei_grid,
                 const OutDataType* p_out_grid,
                 AccDataType* p_acc_grid,
                 const std::array<index_t, NDimSpatial + 3>& in_g_n_c_wis_lengths, // input
                 const std::array<index_t, NDimSpatial + 3>& in_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& wei_g_k_c_xs_lengths, // weight
                 const std::array<index_t, NDimSpatial + 3>& wei_g_k_c_xs_strides,
                 const std::array<index_t, NDimSpatial + 3>& out_g_n_k_wos_lengths, // output
                 const std::array<index_t, NDimSpatial + 3>& out_g_n_k_wos_strides,
                 index_t k_batch)
            : p_in_grid_{p_in_grid},
              p_wei_grid_{p_wei_grid},
              p_out_grid_{p_out_grid},
              p_acc_grid_{p_acc_grid},
              in_g_n_c_wis_lengths_(in_g_n_c_wis_lengths),
              in_g_n_c_wis_strides_(in_g_n_c_wis_strides),
              wei_g_k_c_xs_lengths_(wei_g_k_c_xs_lengths),
              wei_g_k_c_xs_strides_(wei_g_k_c_xs_strides),
              out_g_n_k_wos_lengths_(out_g_n_k_wos_lengths),
              out_g_n_k_wos_strides_(out_g_n_k_wos_strides),
              enable_k_split_(k_batch > 1)
        {
            if(enable_k_split_)
            {
                in_g_n_c_wis_lengths_[1] /= k_batch;
                out_g_n_k_wos_lengths_[1] /= k_batch;
            }
        }


        const InDataType* p_in_grid_;
        WeiDataType* p_wei_grid_;
        const OutDataType* p_out_grid_;
        AccDataType* p_acc_grid_;
        std::array<index_t, NDimSpatial + 3> in_g_n_c_wis_lengths_;
        std::array<index_t, NDimSpatial + 3> in_g_n_c_wis_strides_;
        std::array<index_t, NDimSpatial + 3> wei_g_k_c_xs_lengths_;
        std::array<index_t, NDimSpatial + 3> wei_g_k_c_xs_strides_;
        std::array<index_t, NDimSpatial + 3> out_g_n_k_wos_lengths_;
        std::array<index_t, NDimSpatial + 3> out_g_n_k_wos_strides_;
        bool enable_k_split_;
    };
};

template <index_t NDimSpatial,
          index_t BlockSize,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AccDataType,
          typename BlockTileSize, // input, without include pading
          index_t FilterSize,     // seqence<x, y, [z]>
          typename FilterParam,   // tuple<dilation, stride, padding>
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          index_t NBatch,
          index_t NumWavePerTile,
          index_t InScalarPerVector,
          index_t OutScalarPerVector,
          index_t DstScalarPerVector,
          bool RequirePadding,
          index_t WSplit        = 1,
          typename ComputeTypeA = InDataType,
          typename ComputeTypeB = ComputeTypeA>
struct DeviceGroupedConvBwdWeightDlV4 : public DeviceGroupedConvBwdWeight<NDimSpatial,
                                                                          InLayout,
                                                                          WeiLayout,
                                                                          OutLayout,
                                                                          InDataType,
                                                                          WeiDataType,
                                                                          OutDataType,
                                                                          InElementwiseOperation,
                                                                          WeiElementwiseOperation,
                                                                          OutElementwiseOperation,
                                                                          ComputeTypeA,
                                                                          ComputeTypeB>
{
    using DeviceOp           = DeviceGroupedConvBwdWeightDlV4;
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    static_assert(NDimSpatial == 2);
    static_assert(FilterSize * FilterSize < 64);
    static_assert(RequirePadding == false);
    static_assert(NBatch % DstScalarPerVector == 0);
    static_assert(is_same_v<InElementwiseOperation, element_wise::PassThrough>);
    static_assert(is_same_v<WeiElementwiseOperation, element_wise::PassThrough>);
    static_assert(is_same_v<OutElementwiseOperation, element_wise::PassThrough>);

    using GridwiseConvBwdWeight = GridwiseGroupedConv2DBwdWeightDlV4<BlockSize,
                                                                     InDataType,
                                                                     WeiDataType,
                                                                     OutDataType,
                                                                     AccDataType,
                                                                     BlockTileSize,
                                                                     FilterSize,
                                                                     FilterParam,
                                                                     InElementwiseOperation,
                                                                     WeiElementwiseOperation,
                                                                     OutElementwiseOperation,
                                                                     NBatch,
                                                                     NumWavePerTile,
                                                                     InScalarPerVector,
                                                                     OutScalarPerVector,
                                                                     DstScalarPerVector,
                                                                     RequirePadding,
                                                                     WSplit>;

    struct Argument : public BaseArgument
    {
        Argument(const InDataType* p_in_grid,
                 WeiDataType* p_wei_grid,
                 const OutDataType* p_out_grid,
                 const std::array<index_t, NDimSpatial + 3>& in_g_n_c_wis_lengths, // input
                 const std::array<index_t, NDimSpatial + 3>& in_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& wei_g_k_c_xs_lengths, // weight
                 const std::array<index_t, NDimSpatial + 3>& wei_g_k_c_xs_strides,
                 const std::array<index_t, NDimSpatial + 3>& out_g_n_k_wos_lengths, // output
                 const std::array<index_t, NDimSpatial + 3>& out_g_n_k_wos_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                 const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op,
                 ck::index_t split_k)
            : p_in_grid_{p_in_grid},
              p_wei_grid_{p_wei_grid},
              p_out_grid_{p_out_grid},
              out_element_op_{out_element_op},
              in_element_op_{in_element_op},
              wei_element_op_{wei_element_op},
              in_g_n_c_wis_lengths_(in_g_n_c_wis_lengths),
              in_g_n_c_wis_strides_(in_g_n_c_wis_strides),
              wei_g_k_c_xs_lengths_(wei_g_k_c_xs_lengths),
              wei_g_k_c_xs_strides_(wei_g_k_c_xs_strides),
              out_g_n_k_wos_lengths_(out_g_n_k_wos_lengths),
              out_g_n_k_wos_strides_(out_g_n_k_wos_strides),
              conv_filter_strides_(conv_filter_strides),
              conv_filter_dilations_(conv_filter_dilations),
              input_left_pads_(input_left_pads),
              input_right_pads_(input_right_pads),
              k_batch_{split_k}
        {
            // YX
            acc_g_k_c_xs_strides_[4] = 1;
            acc_g_k_c_xs_strides_[3] = wei_g_k_c_xs_lengths_[4];
            // GKC
            acc_g_k_c_xs_strides_[2] = acc_g_k_c_xs_strides_[3] * wei_g_k_c_xs_lengths_[3];
            acc_g_k_c_xs_strides_[1] = acc_g_k_c_xs_strides_[2] * wei_g_k_c_xs_lengths_[2];
            acc_g_k_c_xs_strides_[0] = acc_g_k_c_xs_strides_[1] * wei_g_k_c_xs_lengths_[1];
        }

        std::size_t GetWorkspaceSizeBytes() const
        {
            if(k_batch_ > 1)
            {
                return math::integer_least_multiple(
                    sizeof(AccDataType) * wei_g_k_c_xs_lengths_[0] * wei_g_k_c_xs_lengths_[1] *
                        wei_g_k_c_xs_lengths_[2] * wei_g_k_c_xs_lengths_[3] *
                        wei_g_k_c_xs_lengths_[4],
                    128);
            }
            else
            {
                return 0;
            }
        }

        const InDataType* p_in_grid_;
        WeiDataType* p_wei_grid_;
        const OutDataType* p_out_grid_;

        OutElementwiseOperation out_element_op_;
        InElementwiseOperation in_element_op_;
        WeiElementwiseOperation wei_element_op_;

        std::array<index_t, NDimSpatial + 3> in_g_n_c_wis_lengths_;
        std::array<index_t, NDimSpatial + 3> in_g_n_c_wis_strides_;
        std::array<index_t, NDimSpatial + 3> wei_g_k_c_xs_lengths_;
        std::array<index_t, NDimSpatial + 3> wei_g_k_c_xs_strides_;
        std::array<index_t, NDimSpatial + 3> out_g_n_k_wos_lengths_;
        std::array<index_t, NDimSpatial + 3> out_g_n_k_wos_strides_;
        std::array<index_t, NDimSpatial + 3> acc_g_k_c_xs_strides_;
        std::array<ck::index_t, NDimSpatial> conv_filter_strides_;
        std::array<ck::index_t, NDimSpatial> conv_filter_dilations_;
        std::array<ck::index_t, NDimSpatial> input_left_pads_;
        std::array<ck::index_t, NDimSpatial> input_right_pads_;
        const index_t k_batch_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        void ShowInfo(const Argument&) {}
        dim3 CalculateGridSize(const Argument& arg)
        {
            return dim3(arg.in_g_n_c_wis_lengths_[0], arg.k_batch_, 1);
        }

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            auto gdx = CalculateGridSize(arg);
            AccDataType* p_acc_grid = type_convert<AccDataType*>(arg.p_workspace_);

            float ave_time = 0;
            typename GridwiseConvBwdWeight::Argument conv_arg{arg.p_in_grid_,
                                                              arg.k_batch_ > 1 ? nullptr : arg.p_wei_grid_,
                                                              arg.p_out_grid_,
                                                              arg.k_batch_ > 1 ? p_acc_grid : nullptr,
                                                              arg.in_g_n_c_wis_lengths_,
                                                              arg.in_g_n_c_wis_strides_,
                                                              arg.wei_g_k_c_xs_lengths_,
                                                              arg.k_batch_ > 1 ? arg.acc_g_k_c_xs_strides_ : arg.wei_g_k_c_xs_strides_,
                                                              arg.out_g_n_k_wos_lengths_,
                                                              arg.out_g_n_k_wos_strides_,
                                                              arg.k_batch_};

            constexpr index_t minimum_occupancy =
                1; // GridwiseConvBwdWeight::TotalLdsSize() > (32 * 1024) ? 1 : 2;

            const auto clear_workspace = [&]() {
                hip_check_error(hipMemsetAsync(p_acc_grid,
                                               0,
                                               arg.GetWorkspaceSizeBytes(),
                                               stream_config.stream_id_));
            };

            const auto conv_kernel = kernel_grouped_conv_bwd_weight_dl_v4<GridwiseConvBwdWeight,
                                                                     BlockSize,
                                                                     minimum_occupancy>;
            auto elementwise_kernel = kernel_grouped_conv_bwd_weight_elementwise<WeiDataType, AccDataType, FilterSize>;
            if(arg.k_batch_ > 1)
            {
                ave_time += launch_and_time_kernel_with_preprocess(
                    stream_config, clear_workspace, conv_kernel, gdx, dim3(BlockSize), 0, conv_arg);
                    
                // copy result and convert type to wei type
                const index_t elementwise_gd = arg.in_g_n_c_wis_lengths_[0];
                const index_t elementwise_block = FilterSize * FilterSize;
                ave_time += launch_and_time_kernel(stream_config,
                                                   elementwise_kernel,
                                                   dim3(elementwise_gd),
                                                   dim3(elementwise_block),
                                                   0,
                                                   arg.p_wei_grid_,
                                                   p_acc_grid,
                                                   arg.wei_g_k_c_xs_strides_,
                                                   arg.acc_g_k_c_xs_strides_);
            }
            else
            {
                ave_time += launch_and_time_kernel(
                    stream_config, conv_kernel, gdx, dim3(BlockSize), 0, conv_arg);
            }

            return ave_time;
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        constexpr index_t spatial_offset = 3;
        // In
        const index_t hi        = arg.in_g_n_c_wis_lengths_[spatial_offset + 0];
        const index_t wi        = arg.in_g_n_c_wis_lengths_[spatial_offset + 1];
        const index_t wi_stride = arg.in_g_n_c_wis_strides_[spatial_offset + 1];
        const index_t n         = arg.in_g_n_c_wis_lengths_[1];
        // Out
        const index_t wo        = arg.out_g_n_k_wos_lengths_[spatial_offset + 1];
        const index_t wo_stride = arg.out_g_n_k_wos_strides_[spatial_offset + 1];
        // Wei
        const index_t filter_y = arg.wei_g_k_c_xs_lengths_[spatial_offset + 0];
        const index_t filter_x = arg.wei_g_k_c_xs_lengths_[spatial_offset + 1];
        const index_t filter_k = arg.wei_g_k_c_xs_lengths_[1];
        const index_t filter_c = arg.wei_g_k_c_xs_lengths_[2];

        static constexpr index_t Tile_H     = BlockTileSize{}.At(I0);
        static constexpr index_t Tile_W     = BlockTileSize{}.At(I1);
        static constexpr index_t Pad_H      = tuple_element_t<2, FilterParam>{}.At(I0);
        static constexpr index_t Pad_W      = tuple_element_t<2, FilterParam>{}.At(I1);
        static constexpr index_t Stride_H   = tuple_element_t<1, FilterParam>{}.At(I0);
        static constexpr index_t Stride_W   = tuple_element_t<1, FilterParam>{}.At(I1);
        static constexpr index_t Dilation_Y = tuple_element_t<0, FilterParam>{}.At(I0);
        static constexpr index_t Dilation_X = tuple_element_t<0, FilterParam>{}.At(I1);

        if(filter_k != 1 || filter_c != 1)
        {
            return false;
        }
        if(n % (arg.k_batch_ * NBatch * GridwiseConvBwdWeight::NumTilePerBlock) != 0)
        {
            return false;
        }
        if constexpr(RequirePadding == false)
        {
            if(hi != Tile_H || wi != Tile_W)
            {
                return false;
            }
        }
        if(filter_y != FilterSize || filter_x != FilterSize)
        {
            return false;
        }
        if(Pad_H != arg.input_left_pads_[0] || Pad_W != arg.input_left_pads_[1] ||
           Pad_H != arg.input_right_pads_[0] || Pad_W != arg.input_right_pads_[1])
        {
            return false;
        }
        if(Stride_H != arg.conv_filter_strides_[0] || Stride_W != arg.conv_filter_strides_[1])
        {
            return false;
        }
        if(Dilation_Y != arg.conv_filter_dilations_[0] ||
           Dilation_X != arg.conv_filter_dilations_[1])
        {
            return false;
        }
        if(InScalarPerVector > 1)
        {
            if(wi % InScalarPerVector != 0)
            {
                return false;
            }
            if(wi_stride != 1)
            {
                return false;
            }
        }
        if(OutScalarPerVector > 1)
        {
            if(wo % OutScalarPerVector != 0)
            {
                return false;
            }
            if(wo_stride != 1)
            {
                return false;
            }
        }
        return true;
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto
    MakeArgument(const InDataType* p_in_grid,
                 WeiDataType* p_wei_grid,
                 const OutDataType* p_out_grid,
                 const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths, // input
                 const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths, // weight
                 const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                 const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op,
                 const ck::index_t split_k)
    {
        return Argument{p_in_grid,
                        p_wei_grid,
                        p_out_grid,
                        b_g_n_c_wis_lengths, // input
                        b_g_n_c_wis_strides,
                        e_g_k_c_xs_lengths, // weight
                        e_g_k_c_xs_strides,
                        a_g_n_k_wos_lengths, // output
                        a_g_n_k_wos_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        in_element_op,
                        wei_element_op,
                        out_element_op,
                        split_k};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in_grid,
                        void* p_wei_grid,
                        const void* p_out_grid,
                        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths, // input
                        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
                        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths, // weight
                        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
                        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output
                        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
                        const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                        const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                        InElementwiseOperation in_element_op,
                        WeiElementwiseOperation wei_element_op,
                        OutElementwiseOperation out_element_op,
                        const ck::index_t split_k) override
    {
        return std::make_unique<Argument>(static_cast<const InDataType*>(p_in_grid),
                                          static_cast<WeiDataType*>(p_wei_grid),
                                          static_cast<const OutDataType*>(p_out_grid),
                                          b_g_n_c_wis_lengths, // input
                                          b_g_n_c_wis_strides,
                                          e_g_k_c_xs_lengths, // weight
                                          e_g_k_c_xs_strides,
                                          a_g_n_k_wos_lengths, // output
                                          a_g_n_k_wos_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          in_element_op,
                                          wei_element_op,
                                          out_element_op,
                                          split_k);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        index_t Pad_H = tuple_element_t<2, FilterParam>{}.At(I0);
        index_t Pad_W = tuple_element_t<2, FilterParam>{}.At(I1);

        index_t Stride_H = tuple_element_t<1, FilterParam>{}.At(I0);
        index_t Stride_W = tuple_element_t<1, FilterParam>{}.At(I1);

        index_t Dilation_Y = tuple_element_t<0, FilterParam>{}.At(I0);
        index_t Dilation_X = tuple_element_t<0, FilterParam>{}.At(I1);

        // clang-format off
        str << "DeviceGroupedConvBwdWeightDlV4<"
            << NDimSpatial << ", "
            << BlockSize << ", "
            << InLayout::name << ", "
            << WeiLayout::name << ", "
            << OutLayout::name << ", "
            << "BlockTileSize<" << BlockTileSize{}.At(I0) << ", " << BlockTileSize{}.At(I1) << ">, "
            << "FilterSize<" << FilterSize << ","<< FilterSize << ">, "
            << "Dilation<" << Dilation_Y << ", " << Dilation_X << ">, "
            << "Stride<" << Stride_H << ", " << Stride_W<< ">, "
            << "Pad<" << Pad_H << ", " << Pad_W<< ">, "
            << "NBatch: " << NBatch << ", "
            << "NumWavePerTile: " << NumWavePerTile << ", "
            << "InScalarPerVector: " << InScalarPerVector << ", "
            << "OutScalarPerVector: " << OutScalarPerVector << ", "
            << "DstScalarPerVector: " << DstScalarPerVector << ", "
            << "RequirePadding: " << RequirePadding << ", "
            << "WSplit: " << WSplit << ">"
            << std::endl;
        // clang-format on

        return str.str();
    }

    size_t GetWorkSpaceSize(const Argument* p_arg) const
    {
        return p_arg->GetWorkspaceSizeBytes();
    }

    // void SetWorkSpacePointer(Argument* p_arg,
    //                          void* p_workspace,
    //                          const StreamConfig& = StreamConfig{}) const
    // {
    //     p_arg->p_workspace_ = p_workspace;
    // }
    virtual size_t GetWorkSpaceSize(const BaseArgument*) const override { return 0; }


};
} // namespace device
} // namespace tensor_operation
} // namespace ck
