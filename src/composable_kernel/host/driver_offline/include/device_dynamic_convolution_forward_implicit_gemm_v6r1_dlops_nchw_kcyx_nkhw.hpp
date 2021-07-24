#pragma once
#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "transform_forward_convolution_into_gemm_v6r1_nchw_kcyx_nkhw.hpp"
#include "driver_dynamic_contraction_dlops_v1r2.hpp"

template <typename TInWei,
          typename TAcc,
          typename TOut,
          typename InLengths,
          typename WeiLengths,
          typename OutLengths,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void device_dynamic_convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw(
    const InLengths& in_n_c_hi_wi_lengths,
    const WeiLengths& wei_k_c_y_x_lengths,
    const OutLengths& out_n_k_ho_wo_lengths,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads,
    const Tensor<TInWei>& in_n_c_hi_wi,
    const Tensor<TInWei>& wei_k_c_y_x,
    Tensor<TOut>& out_n_k_ho_wo,
    ck::index_t nrepeat)
{
    using namespace ck;

    std::cout << __func__ << std::endl;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};

    DeviceMem in_n_c_hi_wi_device_buf(sizeof(TInWei) * in_n_c_hi_wi.mDesc.GetElementSpace());
    DeviceMem wei_k_c_y_x_device_buf(sizeof(TInWei) * wei_k_c_y_x.mDesc.GetElementSpace());
    DeviceMem out_n_k_ho_wo_device_buf(sizeof(TOut) * out_n_k_ho_wo.mDesc.GetElementSpace());

    in_n_c_hi_wi_device_buf.ToDevice(in_n_c_hi_wi.mData.data());
    wei_k_c_y_x_device_buf.ToDevice(wei_k_c_y_x.mData.data());
    out_n_k_ho_wo_device_buf.ToDevice(out_n_k_ho_wo.mData.data());

    const auto in_desc_n_c_hi_wi =
        make_dynamic_naive_tensor_descriptor_packed_v2(in_n_c_hi_wi_lengths);
    const auto wei_desc_k_c_y_x =
        make_dynamic_naive_tensor_descriptor_packed_v2(wei_k_c_y_x_lengths);
    const auto out_desc_n_k_ho_wo =
        make_dynamic_naive_tensor_descriptor_packed_v2(out_n_k_ho_wo_lengths);

#if 1
    // [8, 1, 128, 1] * [8, 4, 32, 1] = [1, 128, 4, 32] for fp32
    // cdata = 64, BlockSize = 256
    constexpr index_t BlockSize = 256;

    constexpr index_t GN0 = 4;
    constexpr index_t GK1 = 1;

    constexpr index_t GM1PerBlockGM11 = 128;
    constexpr index_t GN1PerBlockGN11 = 32;
    constexpr index_t GK0PerBlock     = 8;

    constexpr index_t BM1PerThreadBM11 = 4;
    constexpr index_t BN1PerThreadBN11 = 4;
    constexpr index_t BK0PerThread     = 1;

    using BM10BN10ThreadClusterBM10Xs = Sequence<8, 2>;
    using BM10BN10ThreadClusterBN10Xs = Sequence<8, 2>;

    using ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1   = Sequence<4, 1, 1, 1, 1>;
    using ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1 = Sequence<2, 1, 1, 128, 1>;

    using ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1 = Sequence<4, 1, 1, 1, 1>;
    using ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1 = Sequence<1, 1, 1, 1, 1>;

    using BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1   = Sequence<1, 4, 1, 1, 1>;
    using BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1 = Sequence<8, 1, 1, 32, 1>;

    using BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1 = Sequence<1, 1, 1, 1, 1>;
    using BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1 = Sequence<1, 1, 1, 1, 1>;

    constexpr index_t CThreadTransferDstScalarPerVector_BN1 = 1;
#elif 1
    // [8, 1, 128, 2] * [8, 4, 32, 2] = [1, 128, 4, 32] for fp16
    // cdata = 64, BlockSize = 256
    constexpr index_t BlockSize = 256;

    constexpr index_t GN0 = 4;
    constexpr index_t GK1 = 2;

    constexpr index_t GM1PerBlockGM11 = 128;
    constexpr index_t GN1PerBlockGN11 = 32;
    constexpr index_t GK0PerBlock     = 8;

    constexpr index_t BM1PerThreadBM11 = 4;
    constexpr index_t BN1PerThreadBN11 = 4;
    constexpr index_t BK0PerThread     = 1;

    using BM10BN10ThreadClusterBM10Xs = Sequence<8, 2>;
    using BM10BN10ThreadClusterBN10Xs = Sequence<8, 2>;

    using ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1   = Sequence<4, 1, 1, 1, 2>;
    using ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1 = Sequence<2, 1, 1, 128, 1>;

    using ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1 = Sequence<4, 1, 1, 1, 1>;
    using ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1 = Sequence<1, 1, 1, 1, 2>;

    using BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1   = Sequence<1, 4, 1, 1, 2>;
    using BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1 = Sequence<8, 1, 1, 32, 1>;

    using BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1 = Sequence<1, 1, 1, 1, 1>;
    using BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1 = Sequence<1, 1, 1, 1, 2>;

    constexpr index_t CThreadTransferDstScalarPerVector_BN1 = 1;
#endif

    const auto descs =
        transform_forward_convolution_into_contraction_v6r1_nchw_kcyx_nkhw_pad(wei_desc_k_c_y_x,
                                                                               in_desc_n_c_hi_wi,
                                                                               out_desc_n_k_ho_wo,
                                                                               conv_strides,
                                                                               conv_dilations,
                                                                               in_left_pads,
                                                                               in_right_pads,
                                                                               Number<GN0>{},
                                                                               Number<GK1>{});

    const auto wei_grid_desc_gk0_gm0_gm1_gk1 = descs[I0];
    const auto in_grid_desc_gk0_gn0_gn1_gk1  = descs[I1];
    const auto out_grid_desc_gm0_gm1_gn0_gn1 = descs[I2];

    // HACK: hacks that control index calculation when iterating over A, B, C matrix
    constexpr auto wei_grid_iterator_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0>{},   // 0+: GK0
                              Sequence<0, 0, 0, 0, 0, 0, 0>{},   // 1+: GM0
                              Sequence<0, 0, 0, 0, 0, 0, 0>{},   // 2+: GM10
                              Sequence<0, 0, 0, 0, 0, 0, 0>{},   // 3+: GM11
                              Sequence<0, 0, 0, 0, 0, 0, 0>{}),  // 4+: GK1
                   make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0>{},   // 0-: GK0
                              Sequence<0, 0, 0, 0, 0, 0, 0>{},   // 1-: GM0
                              Sequence<0, 0, 0, 0, 0, 0, 0>{},   // 2-: GM10
                              Sequence<0, 0, 0, 0, 0, 0, 0>{},   // 3-: GM11
                              Sequence<0, 0, 0, 0, 0, 0, 0>{})); // 4-: GK1

    constexpr auto in_grid_iterator_hacks = make_tuple(
        make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0>{},   // 0+: GK0
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0>{},   // 1+: GN0
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0>{},   // 2+: GN10
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0>{},   // 3+: GN11
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}),  // 4+: GK1
        make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0>{},   // 0-: GK0
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0>{},   // 1-: GN0
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0>{},   // 2-: GN10
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0>{},   // 3-: GN11
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{})); // 4-: GK1

    constexpr auto out_grid_iterator_hacks = make_tuple(
        make_tuple(
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},  // 0+: GM10
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0>{},  // 1+: BM0
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0>{},  // 2+: BM1
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},  // 3+: GN10
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0>{},  // 4+: BN0
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0>{}), // 5+: GN1
        make_tuple(
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 0-: GM10
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0>{},   // 1-: BM0
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0>{},   // 2-: BM1
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 3-: GN10
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0>{},   // 4-: BN0
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0>{})); // 5-: GN1

    constexpr auto wei_grid_move_slice_window_iterator_hacks = Sequence<0, 0, 0, 0, 0, 0, 0>{};

    constexpr auto in_grid_move_slice_window_iterator_hacks =
        Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0>{};

    for(index_t i = 0; i < 5; ++i)
    {
        float ave_time = driver_dynamic_contraction_dlops_v1r2<
            BlockSize,
            TInWei,
            TAcc,
            TOut,
            InMemoryDataOperation::Set,
            decltype(wei_grid_desc_gk0_gm0_gm1_gk1),
            decltype(in_grid_desc_gk0_gn0_gn1_gk1),
            decltype(out_grid_desc_gm0_gm1_gn0_gn1),
            GM1PerBlockGM11,
            GN1PerBlockGN11,
            GK0PerBlock,
            BM1PerThreadBM11,
            BN1PerThreadBN11,
            BK0PerThread,
            BM10BN10ThreadClusterBM10Xs,
            BM10BN10ThreadClusterBN10Xs,
            ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1,
            ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1,
            Sequence<1, 2, 3, 0, 4>, // ABlockTransferThreadClusterArrangeOrder
            Sequence<3, 2, 1, 0, 4>, // ABlockTransferSrcAccessOrder
            ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1,
            ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1,
            Sequence<0, 1, 2, 3, 4>, // ABlockTransferSrcVectorTensorContiguousDimOrder
            BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1,
            BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1,
            Sequence<0, 4, 1, 2, 3>, // BBlockTransferThreadClusterArrangeOrder
            Sequence<4, 3, 2, 0, 1>, // BBlockTransferSrcAccessOrder
            BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1,
            BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1,
            Sequence<0, 1, 2, 3, 4>,    // BBlockTransferSrcVectorTensorContiguousDimOrder
            Sequence<3, 4, 5, 0, 1, 2>, // CThreadTransferSrcDstAccessOrder
            5,                          // CThreadTransferSrcDstVectorDim
            CThreadTransferDstScalarPerVector_BN1,
            decltype(wei_grid_iterator_hacks),
            decltype(in_grid_iterator_hacks),
            decltype(out_grid_iterator_hacks),
            decltype(wei_grid_move_slice_window_iterator_hacks),
            decltype(in_grid_move_slice_window_iterator_hacks)>(
            static_cast<TInWei*>(wei_k_c_y_x_device_buf.GetDeviceBuffer()),
            static_cast<TInWei*>(in_n_c_hi_wi_device_buf.GetDeviceBuffer()),
            static_cast<TOut*>(out_n_k_ho_wo_device_buf.GetDeviceBuffer()),
            wei_grid_desc_gk0_gm0_gm1_gk1,
            in_grid_desc_gk0_gn0_gn1_gk1,
            out_grid_desc_gm0_gm1_gn0_gn1,
            wei_grid_iterator_hacks,
            in_grid_iterator_hacks,
            out_grid_iterator_hacks,
            wei_grid_move_slice_window_iterator_hacks,
            in_grid_move_slice_window_iterator_hacks,
            nrepeat);

        float perf = (float)calculate_convolution_flops(
                         in_desc_n_c_hi_wi, wei_desc_k_c_y_x, out_desc_n_k_ho_wo) /
                     (std::size_t(1000) * 1000 * 1000) / ave_time;

        std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s" << std::endl;
    }

    // copy result back to host
    out_n_k_ho_wo_device_buf.FromDevice(out_n_k_ho_wo.mData.data());
}
