#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "transform_backward_weight_convolution_into_gemm_v4r4r2_nchw_kcyx_nkhw.hpp"
#include "driver_gemm_xdlops_v2r3.hpp"

template <typename TIn,
          typename TWei,
          typename TAcc,
          typename TOut,
          typename InLengths,
          typename WeiLengths,
          typename OutLengths,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void device_convolution_backward_weight_implicit_gemm_v4r4r2_xdlops_nchw_kcyx_nkhw(
    const InLengths& in_n_c_hi_wi_lengths,
    const WeiLengths& wei_k_c_y_x_lengths,
    const OutLengths& out_n_k_ho_wo_lengths,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads,
    const Tensor<TIn>& in_n_c_hi_wi,
    Tensor<TWei>& wei_k_c_y_x,
    const Tensor<TOut>& out_n_k_ho_wo,
    ck::index_t nrepeat)
{
    using namespace ck;

    std::cout << __func__ << std::endl;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};

    DeviceMem in_n_c_hi_wi_device_buf(sizeof(TIn) * in_n_c_hi_wi.mDesc.GetElementSpace());
    DeviceMem wei_k_c_y_x_device_buf(sizeof(TWei) * wei_k_c_y_x.mDesc.GetElementSpace());
    DeviceMem out_n_k_ho_wo_device_buf(sizeof(TOut) * out_n_k_ho_wo.mDesc.GetElementSpace());

    in_n_c_hi_wi_device_buf.ToDevice(in_n_c_hi_wi.mData.data());
    wei_k_c_y_x_device_buf.ToDevice(wei_k_c_y_x.mData.data());
    out_n_k_ho_wo_device_buf.ToDevice(out_n_k_ho_wo.mData.data());

    const auto in_n_c_hi_wi_desc  = make_naive_tensor_descriptor_packed(in_n_c_hi_wi_lengths);
    const auto wei_k_c_y_x_desc   = make_naive_tensor_descriptor_packed(wei_k_c_y_x_lengths);
    const auto out_n_k_ho_wo_desc = make_naive_tensor_descriptor_packed(out_n_k_ho_wo_lengths);

#if 0
    // [M, N, K0, K1] = [128, 128, 4, 8] for fp16
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerWave = 32;
    constexpr index_t GemmNPerWave = 32;
    constexpr index_t GemmK1       = 8;

    constexpr index_t MRepeat = 2;
    constexpr index_t NRepeat = 2;

    using GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1   = Sequence<1, 2, 8>;
    using GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1 = Sequence<4, 64, 1>;
    // using vector load 4, so config's wo*ho  must be a multiple of 4
    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK1 = 4;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmK1 = 4;

    using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1   = Sequence<1, 2, 8>;
    using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN  = 1;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmK1 = 8;

    constexpr index_t GemmCThreadTransferDstScalarPerVector = 1;
#elif 1
    // [M, N, K0, K1] = [128, 128, 4, 8] for fp16
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 256;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerWave = 32;
    constexpr index_t GemmNPerWave = 32;
    constexpr index_t GemmK1       = 8;

    constexpr index_t MRepeat = 4;
    constexpr index_t NRepeat = 2;

    using GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1   = Sequence<1, 4, 8>;
    using GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1 = Sequence<4, 64, 1>;
    // using vector load 4, so config's wo*ho  must be a multiple of 4
    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK1 = 4;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmK1 = 4;

    using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1   = Sequence<1, 2, 8>;
    using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN  = 1;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmK1 = 8;

    constexpr index_t GemmCThreadTransferDstScalarPerVector = 1;
#endif

    const auto descs = transform_backward_weight_convolution_into_gemm_v4r4r2_nchw_kcyx_nkhw_pad(
        wei_k_c_y_x_desc,
        in_n_c_hi_wi_desc,
        out_n_k_ho_wo_desc,
        conv_strides,
        conv_dilations,
        in_left_pads,
        in_right_pads,
        Number<GemmK1>{});

    const auto out_gemmk0_gemmm_gemmk1_grid_desc = descs[I0];
    const auto in_gemmk0_gemmn_gemmk1_grid_desc  = descs[I1];
    const auto wei_gemmm_gemmn_grid_desc         = descs[I2];

    // HACK: hacks that control index calculation when iterating over A, B, C matrix
    constexpr auto out_gemmk0_gemmm_gemmk1_grid_step_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 1, 0, 0>{},   // 0+: GemmK0
                              Sequence<0, 0, 0, 0, 0>{},   // 1+: GemmM
                              Sequence<0, 0, 1, 0, 0>{}),  // 2+: GemmK1
                   make_tuple(Sequence<0, 0, 2, 0, 0>{},   // 0-: GemmK0
                              Sequence<0, 0, 0, 0, 0>{},   // 1-: GemmM
                              Sequence<0, 0, 2, 0, 0>{})); // 2-: GemmK1

    constexpr auto in_gemmk0_gemmn_gemmk1_grid_step_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0>{},   // 0+: GemmK0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{},   // 1+: GemmN
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0>{}),  // 2+: GemmK1
                   make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0>{},   // 0-: GemmK0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{},   // 1-: GemmN
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0>{})); // 2-: GemmK1

    constexpr auto wei_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 0+: M0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 1+: N0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 2+: M1
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 3+: N1
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 4+: M2
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 5+: M3
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 6+: M4
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}),  // 7+: N2
                   make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 0-: M0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 1-: N0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 2-: M1
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 3-: N1
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 4-: M2
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 5-: M3
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 6-: M4
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{})); // 7-: N2

    constexpr auto out_gemmk0_gemmm_gemmk1_grid_move_slice_window_step_hacks =
        Sequence<0, 0, 1, 0, 0>{};

    constexpr auto in_gemmk0_gemmn_gemmk1_grid_move_slice_window_step_hacks =
        Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0>{};

    for(index_t i = 0; i < 5; ++i)
    {
        float ave_time = driver_gemm_xdlops_v2r3<
            BlockSize,
            TIn,
            TAcc,
            TWei,
            InMemoryDataOperationEnum_t::Set,
            decltype(out_gemmk0_gemmm_gemmk1_grid_desc),
            decltype(in_gemmk0_gemmn_gemmk1_grid_desc),
            decltype(wei_gemmm_gemmn_grid_desc),
            GemmMPerBlock,
            GemmNPerBlock,
            GemmKPerBlock,
            GemmMPerWave,
            GemmNPerWave,
            GemmK1,
            MRepeat,
            NRepeat,
            GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1,
            GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1,
            Sequence<1, 0, 2>,
            Sequence<1, 0, 2>,
            2,
            GemmABlockTransferSrcScalarPerVector_GemmK1,
            GemmABlockTransferDstScalarPerVector_GemmK1,
            false, // don't move back src coordinate after threadwise copy
            GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1,
            GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1,
            Sequence<1, 0, 2>,
            Sequence<1, 0, 2>,
            2,
            GemmBBlockTransferSrcScalarPerVector_GemmN,
            GemmBBlockTransferDstScalarPerVector_GemmK1,
            false, // don't move back src coordinate after threadwise copy
            Sequence<3, 0, 1, 2, 7, 5, 4, 6>,
            7,
            GemmCThreadTransferDstScalarPerVector,
            decltype(out_gemmk0_gemmm_gemmk1_grid_step_hacks),
            decltype(in_gemmk0_gemmn_gemmk1_grid_step_hacks),
            decltype(wei_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks),
            decltype(out_gemmk0_gemmm_gemmk1_grid_move_slice_window_step_hacks),
            decltype(in_gemmk0_gemmn_gemmk1_grid_move_slice_window_step_hacks),
            false, // CAccessOrderMRepeatNRepeat
            true,  // ABlockLdsExtraM
            true   // BBlockLdsExtraN
            >(static_cast<TOut*>(out_n_k_ho_wo_device_buf.GetDeviceBuffer()),
              static_cast<TIn*>(in_n_c_hi_wi_device_buf.GetDeviceBuffer()),
              static_cast<TWei*>(wei_k_c_y_x_device_buf.GetDeviceBuffer()),
              out_gemmk0_gemmm_gemmk1_grid_desc,
              in_gemmk0_gemmn_gemmk1_grid_desc,
              wei_gemmm_gemmn_grid_desc,
              debug::debug_driver_gemm_xdlops_v2r3::M01,
              debug::debug_driver_gemm_xdlops_v2r3::N01,
              out_gemmk0_gemmm_gemmk1_grid_step_hacks,
              in_gemmk0_gemmn_gemmk1_grid_step_hacks,
              wei_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks,
              out_gemmk0_gemmm_gemmk1_grid_move_slice_window_step_hacks,
              in_gemmk0_gemmn_gemmk1_grid_move_slice_window_step_hacks,
              nrepeat);

        float perf = static_cast<float>(calculate_convolution_flops(
                         in_n_c_hi_wi_desc, wei_k_c_y_x_desc, out_n_k_ho_wo_desc)) /
                     (std::size_t(1000) * 1000 * 1000) / ave_time;

        std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s" << std::endl;
    }

    // copy result back to host
    wei_k_c_y_x_device_buf.FromDevice(wei_k_c_y_x.mData.data());
}
