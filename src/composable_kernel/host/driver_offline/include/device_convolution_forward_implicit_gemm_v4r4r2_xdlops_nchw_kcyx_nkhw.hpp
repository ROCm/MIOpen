#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "transform_forward_convolution_into_gemm_v4r4r2_nchw_kcyx_nkhw.hpp"
#include "driver_gemm_xdlops_v2r3.hpp"

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
void device_convolution_forward_implicit_gemm_v4r4r2_xdlops_nchw_kcyx_nkhw(
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

    const auto in_n_c_hi_wi_desc  = make_naive_tensor_descriptor_packed(in_n_c_hi_wi_lengths);
    const auto wei_k_c_y_x_desc   = make_naive_tensor_descriptor_packed(wei_k_c_y_x_lengths);
    const auto out_n_k_ho_wo_desc = make_naive_tensor_descriptor_packed(out_n_k_ho_wo_lengths);

#if 1
    // [M, N, K0, K1] = [256, 128, 4, 8] for fp16
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

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK1 = 8;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmK1 = 8;

    using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1   = Sequence<1, 2, 8>;
    using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN  = 1;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmK1 = 8;

    constexpr index_t GemmCThreadTransferDstScalarPerVector = 1;
#endif

    const auto descs =
        transform_forward_convolution_into_gemm_v4r4r2_nchw_kcyx_nkhw_pad(wei_k_c_y_x_desc,
                                                                          in_n_c_hi_wi_desc,
                                                                          out_n_k_ho_wo_desc,
                                                                          conv_strides,
                                                                          conv_dilations,
                                                                          in_left_pads,
                                                                          in_right_pads,
                                                                          Number<GemmK1>{});

    const auto wei_gemmk0_gemmm_gemmk1_grid_desc = descs[I0];
    const auto in_gemmk0_gemmn_gemmk1_grid_desc  = descs[I1];
    const auto out_gemmm_gemmn_grid_desc         = descs[I2];

    // HACK: hacks that control index calculation when iterating over A, B, C matrix
    constexpr auto wei_gemmk0_gemmm_gemmk1_grid_step_hacks = make_tuple(
        make_tuple(Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}),
        make_tuple(
            Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}));

    constexpr auto in_gemmk0_gemmn_gemmk1_grid_step_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{}));

    constexpr auto out_m0_m1_m2_n_grid_step_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 1, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 1, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 1, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 2, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 2, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 2, 0, 0>{}));

    constexpr auto wei_gemmk0_gemmm_gemmk1_grid_move_slice_window_step_hacks =
        Sequence<0, 0, 0, 0, 0>{};

    constexpr auto in_gemmk0_gemmn_gemmk1_grid_move_slice_window_step_hacks =
        Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0>{};

    for(index_t i = 0; i < 5; ++i)
    {
        float ave_time = driver_gemm_xdlops_v2r3<
            BlockSize,
            TInWei,
            TAcc,
            TOut,
            InMemoryDataOperationEnum_t::Set,
            decltype(wei_gemmk0_gemmm_gemmk1_grid_desc),
            decltype(in_gemmk0_gemmn_gemmk1_grid_desc),
            decltype(out_gemmm_gemmn_grid_desc),
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
            Sequence<0, 2, 1>,
            Sequence<1, 0, 2>,
            1,
            GemmBBlockTransferSrcScalarPerVector_GemmN,
            GemmBBlockTransferDstScalarPerVector_GemmK1,
            false, // don't move back src coordinate after threadwise copy
            Sequence<3, 0, 1, 2, 7, 5, 4, 6>,
            7,
            GemmCThreadTransferDstScalarPerVector,
            decltype(wei_gemmk0_gemmm_gemmk1_grid_step_hacks),
            decltype(in_gemmk0_gemmn_gemmk1_grid_step_hacks),
            decltype(out_m0_m1_m2_n_grid_step_hacks),
            decltype(wei_gemmk0_gemmm_gemmk1_grid_move_slice_window_step_hacks),
            decltype(in_gemmk0_gemmn_gemmk1_grid_move_slice_window_step_hacks),
            false>(static_cast<TInWei*>(wei_k_c_y_x_device_buf.GetDeviceBuffer()),
                   static_cast<TInWei*>(in_n_c_hi_wi_device_buf.GetDeviceBuffer()),
                   static_cast<TOut*>(out_n_k_ho_wo_device_buf.GetDeviceBuffer()),
                   wei_gemmk0_gemmm_gemmk1_grid_desc,
                   in_gemmk0_gemmn_gemmk1_grid_desc,
                   out_gemmm_gemmn_grid_desc,
                   wei_gemmk0_gemmm_gemmk1_grid_step_hacks,
                   in_gemmk0_gemmn_gemmk1_grid_step_hacks,
                   out_m0_m1_m2_n_grid_step_hacks,
                   wei_gemmk0_gemmm_gemmk1_grid_move_slice_window_step_hacks,
                   in_gemmk0_gemmn_gemmk1_grid_move_slice_window_step_hacks,
                   nrepeat);

        float perf = static_cast<float>(calculate_convolution_flops(
                         in_n_c_hi_wi_desc, wei_k_c_y_x_desc, out_n_k_ho_wo_desc)) /
                     (std::size_t(1000) * 1000 * 1000) / ave_time;

        std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s" << std::endl;
    }

    // copy result back to host
    out_n_k_ho_wo_device_buf.FromDevice(out_n_k_ho_wo.mData.data());
}
