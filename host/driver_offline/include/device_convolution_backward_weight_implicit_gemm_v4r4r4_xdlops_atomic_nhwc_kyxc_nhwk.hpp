#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "transform_backward_weight_convolution_into_gemm_v4r4r4_atomic_nhwc_kyxc_nhwk.hpp"
#include "driver_gemm_xdlops_v2r4.hpp"

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
          typename InRightPads,
          typename GridSizeType>
void device_convolution_backward_weight_implicit_gemm_v4r4r4_xdlops_atomic_nhwc_kyxc_nhwk(
    const InLengths& in_n_hi_wi_c_lengths,
    const WeiLengths& wei_k_y_x_c_lengths,
    const OutLengths& out_n_ho_wo_k_lengths,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads,
    const Tensor<TIn>& in_n_hi_wi_c,
    Tensor<TWei>& wei_k_y_x_c,
    const Tensor<TOut>& out_n_ho_wo_k,
    GridSizeType desired_grid_size,
    ck::index_t nrepeat)
{
    using namespace ck;

    std::cout << __func__ << std::endl;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    DeviceMem in_n_hi_wi_c_device_buf(sizeof(TIn) * in_n_hi_wi_c.mDesc.GetElementSpace());
    DeviceMem wei_k_y_x_c_device_buf(sizeof(TWei) * wei_k_y_x_c.mDesc.GetElementSpace());
    DeviceMem out_n_ho_wo_k_device_buf(sizeof(TOut) * out_n_ho_wo_k.mDesc.GetElementSpace());

    in_n_hi_wi_c_device_buf.ToDevice(in_n_hi_wi_c.mData.data());
    wei_k_y_x_c_device_buf.ToDevice(wei_k_y_x_c.mData.data());
    out_n_ho_wo_k_device_buf.ToDevice(out_n_ho_wo_k.mData.data());

    const auto in_n_hi_wi_c_desc  = make_naive_tensor_descriptor_packed(in_n_hi_wi_c_lengths);
    const auto wei_k_y_x_c_desc   = make_naive_tensor_descriptor_packed(wei_k_y_x_c_lengths);
    const auto out_n_ho_wo_k_desc = make_naive_tensor_descriptor_packed(out_n_ho_wo_k_lengths);

#if 0
    // [M, N, K0, K1] = [128, 256, 4, 4] for fp32
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 256;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerXDL = 32;
    constexpr index_t GemmNPerXDL = 32;
    constexpr index_t GemmK1      = 4;

    constexpr index_t MRepeat = 2;
    constexpr index_t NRepeat = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1   = Sequence<1, 1, 4, 2>;
    using GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1 = Sequence<1, 4, 32, 2>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmM  = 4;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmK1 = 2;

    using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1   = Sequence<1, 1, 8, 2>;
    using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 = Sequence<1, 4, 32, 2>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN  = 8;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmK1 = 2;

    constexpr index_t GemmCThreadTransferDstScalarPerVector = 4;
#elif 1
    // [M, N, K0, K1] = [128, 128, 4, 4] for fp32
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerXDL = 32;
    constexpr index_t GemmNPerXDL = 32;
    constexpr index_t GemmK1      = 4;

    constexpr index_t MRepeat = 2;
    constexpr index_t NRepeat = 2;

    using GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1   = Sequence<1, 1, 4, 2>;
    using GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1 = Sequence<1, 4, 32, 2>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmM  = 4;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmK1 = 2;

    using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1   = Sequence<1, 1, 4, 2>;
    using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 = Sequence<1, 4, 32, 2>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN  = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmK1 = 2;

    constexpr index_t GemmCThreadTransferDstScalarPerVector = 1;
#endif

    const auto N = in_n_hi_wi_c_desc.GetLength(I0);
    const auto C = in_n_hi_wi_c_desc.GetLength(I3);
    const auto K = out_n_ho_wo_k_desc.GetLength(I3);

    const auto Ho = out_n_ho_wo_k_desc.GetLength(I1);
    const auto Wo = out_n_ho_wo_k_desc.GetLength(I2);

    const auto Y = wei_k_y_x_c_desc.GetLength(I1);
    const auto X = wei_k_y_x_c_desc.GetLength(I2);

    const auto GemmM      = Y * X * C;
    const auto GemmN      = K;
    const auto GemmKTotal = N * Ho * Wo;

    const auto GridMN        = GemmM * GemmN / (GemmMPerBlock * GemmNPerBlock);
    const index_t GemmKBatch = std::max(desired_grid_size / GridMN, 1);
    const index_t GemmK0 =
        math::integer_divide_ceil(GemmKTotal, GemmK1 * GemmKPerBlock * GemmKBatch) * GemmKPerBlock;
    const index_t GemmKPad = GemmKBatch * GemmK0 * GemmK1;

    std::cout << "GemmKTotal: " << GemmKTotal << " GrideSizeMN: " << GridMN
              << " GemmKBatch: " << GemmKBatch << " GemmK0: " << GemmK0 << " gemmKPad: " << GemmKPad
              << std::endl;

    const auto descs =
        transform_backward_weight_convolution_into_gemm_v4r4r4_atomic_nhwc_kyxc_nhwk_pad(
            in_n_hi_wi_c_desc,
            wei_k_y_x_c_desc,
            out_n_ho_wo_k_desc,
            conv_strides,
            conv_dilations,
            in_left_pads,
            in_right_pads,
            Number<GemmK1>{},
            GemmKBatch,
            GemmKPad);

    const auto in_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc  = descs[I0];
    const auto out_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = descs[I1];
    const auto wei_gemmm_gemmn_grid_desc                    = descs[I2];

    // HACK: hacks that control index calculation when iterating over A, B, C matrix
    constexpr auto in_gemmkbatch_gemmk0_gemmm_gemmk1_grid_step_hacks = make_tuple(
        make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0>{},   // 0+: GemmKBatch
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0>{},   // 1+: GemmK0
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0>{},   // 2+: GemmM
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0>{}),  // 3+: GemmK1
        make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0>{},   // 0-: GemmKBatch
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0>{},   // 1-: GemmK0
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0>{},   // 2-: GemmM
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0>{})); // 3-: GemmK1

    constexpr auto out_gemmkbatch_gemmk0_gemmn_gemmk1_grid_step_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},   // 0+: GemmK0
                              Sequence<0, 0, 0, 0, 0>{},   // 0+: GemmK0
                              Sequence<0, 0, 0, 0, 0>{},   // 1+: GemmN
                              Sequence<0, 0, 0, 0, 0>{}),  // 2+: GemmK1
                   make_tuple(Sequence<0, 0, 0, 0, 0>{},   // 0+: GemmK0
                              Sequence<0, 0, 0, 0, 0>{},   // 0-: GemmK0
                              Sequence<0, 0, 0, 0, 0>{},   // 1-: GemmN
                              Sequence<0, 0, 0, 0, 0>{})); // 2-: GemmK1

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

    constexpr auto in_gemmkbatch_gemmk0_gemmm_gemmk1_grid_move_slice_window_step_hacks =
        Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0>{};

    constexpr auto out_gemmkbatch_gemmk0_gemmn_gemmk1_grid_move_slice_window_step_hacks =
        Sequence<0, 0, 0, 0, 0>{};

    const auto driver_gemm_xdlops = driver_gemm_xdlops_v2r4<
        BlockSize,
        TIn,
        TAcc,
        TWei,
        InMemoryDataOperationEnum_t::AtomicAdd,
        decltype(in_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc),
        decltype(out_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc),
        decltype(wei_gemmm_gemmn_grid_desc),
        GemmMPerBlock,
        GemmNPerBlock,
        GemmKPerBlock,
        GemmMPerXDL,
        GemmNPerXDL,
        GemmK1,
        MRepeat,
        NRepeat,
        GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1,
        GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1,
        Sequence<0, 1, 2, 3>,
        Sequence<0, 1, 2, 3>,
        2,
        GemmABlockTransferSrcScalarPerVector_GemmM,
        GemmABlockTransferDstScalarPerVector_GemmK1,
        false, // don't move back src coordinate after threadwise copy
        GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1,
        GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1,
        Sequence<0, 1, 2, 3>,
        Sequence<0, 1, 2, 3>,
        2,
        GemmBBlockTransferSrcScalarPerVector_GemmN,
        GemmBBlockTransferDstScalarPerVector_GemmK1,
        false, // don't move back src coordinate after threadwise copy
        Sequence<2, 3, 0, 1, 7, 5, 4, 6>,
        6,
        GemmCThreadTransferDstScalarPerVector,
        decltype(in_gemmkbatch_gemmk0_gemmm_gemmk1_grid_step_hacks),
        decltype(out_gemmkbatch_gemmk0_gemmn_gemmk1_grid_step_hacks),
        decltype(wei_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks),
        decltype(in_gemmkbatch_gemmk0_gemmm_gemmk1_grid_move_slice_window_step_hacks),
        decltype(out_gemmkbatch_gemmk0_gemmn_gemmk1_grid_move_slice_window_step_hacks),
        false, // CAccessOrderMRepeatNRepeat
        true,
        true>;

    for(index_t i = 0; i < 5; ++i)
    {
        float ave_time =
            driver_gemm_xdlops(static_cast<TIn*>(in_n_hi_wi_c_device_buf.GetDeviceBuffer()),
                               static_cast<TOut*>(out_n_ho_wo_k_device_buf.GetDeviceBuffer()),
                               static_cast<TWei*>(wei_k_y_x_c_device_buf.GetDeviceBuffer()),
                               in_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                               out_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                               wei_gemmm_gemmn_grid_desc,
                               debug::debug_driver_gemm_xdlops_v2r3::M01,
                               debug::debug_driver_gemm_xdlops_v2r3::N01,
                               in_gemmkbatch_gemmk0_gemmm_gemmk1_grid_step_hacks,
                               out_gemmkbatch_gemmk0_gemmn_gemmk1_grid_step_hacks,
                               wei_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks,
                               in_gemmkbatch_gemmk0_gemmm_gemmk1_grid_move_slice_window_step_hacks,
                               out_gemmkbatch_gemmk0_gemmn_gemmk1_grid_move_slice_window_step_hacks,
                               nrepeat);

        {

            float perf = static_cast<float>((std::size_t(2) * N * K * Ho * Wo * C * Y * X)) /
                         (std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                      << std::endl;
        }
    }

    wei_k_y_x_c_device_buf.ToDevice(wei_k_y_x_c.mData.data());
    driver_gemm_xdlops(static_cast<TIn*>(in_n_hi_wi_c_device_buf.GetDeviceBuffer()),
                       static_cast<TOut*>(out_n_ho_wo_k_device_buf.GetDeviceBuffer()),
                       static_cast<TWei*>(wei_k_y_x_c_device_buf.GetDeviceBuffer()),
                       in_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                       out_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                       wei_gemmm_gemmn_grid_desc,
                       debug::debug_driver_gemm_xdlops_v2r3::M01,
                       debug::debug_driver_gemm_xdlops_v2r3::N01,
                       in_gemmkbatch_gemmk0_gemmm_gemmk1_grid_step_hacks,
                       out_gemmkbatch_gemmk0_gemmn_gemmk1_grid_step_hacks,
                       wei_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks,
                       in_gemmkbatch_gemmk0_gemmm_gemmk1_grid_move_slice_window_step_hacks,
                       out_gemmkbatch_gemmk0_gemmn_gemmk1_grid_move_slice_window_step_hacks,
                       0);
    // copy result back to host
    wei_k_y_x_c_device_buf.FromDevice(wei_k_y_x_c.mData.data());
}
