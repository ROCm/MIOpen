#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "transform_forward_convolution_into_gemm_v4r4r4_nhwc_kyxc_nhwk.hpp"
#include "driver_dynamic_gemm_dlops_v1r3.hpp"

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
void device_dynamic_convolution_forward_implicit_gemm_v4r4r2_dlops_nhwc_kyxc_nhwk(
    const InLengths& in_n_hi_wi_c_lengths,
    const WeiLengths& wei_k_y_x_c_lengths,
    const OutLengths& out_n_ho_wo_k_lengths,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads,
    const Tensor<TInWei>& in_n_hi_wi_c,
    const Tensor<TInWei>& wei_k_y_x_c,
    Tensor<TOut>& out_n_ho_wo_k,
    ck::index_t nrepeat)
{
    using namespace ck;

    std::cout << __func__ << std::endl;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};
    constexpr auto I5 = Number<5>{};
    constexpr auto I6 = Number<6>{};
    constexpr auto I7 = Number<7>{};
    constexpr auto I8 = Number<8>{};

    DeviceMem in_n_hi_wi_c_device_buf(sizeof(TInWei) * in_n_hi_wi_c.mDesc.GetElementSpace());
    DeviceMem wei_k_y_x_c_device_buf(sizeof(TInWei) * wei_k_y_x_c.mDesc.GetElementSpace());
    DeviceMem out_n_ho_wo_k_device_buf(sizeof(TOut) * out_n_ho_wo_k.mDesc.GetElementSpace());

    in_n_hi_wi_c_device_buf.ToDevice(in_n_hi_wi_c.mData.data());
    wei_k_y_x_c_device_buf.ToDevice(wei_k_y_x_c.mData.data());
    out_n_ho_wo_k_device_buf.ToDevice(out_n_ho_wo_k.mData.data());

    const auto in_n_hi_wi_c_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(in_n_hi_wi_c_lengths);
    const auto wei_k_y_x_c_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(wei_k_y_x_c_lengths);
    const auto out_n_ho_wo_k_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(out_n_ho_wo_k_lengths);

#if 1
    // [M, N, K0, K1] = [128, 128, 8, 1] for fp32
    // cdata = 64, BlockSize = 256
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlockM1 = 128;
    constexpr index_t GemmNPerBlockN1 = 128;
    constexpr index_t GemmKPerBlock   = 8;
    constexpr index_t GemmK1          = 1;

    constexpr index_t GemmM1PerThreadM111 = 4;
    constexpr index_t GemmN1PerThreadN111 = 4;
    constexpr index_t GemmKPerThread      = 1;

    using GemmM11N11ThreadClusterM110Xs = Sequence<8, 2>;
    using GemmM11N11ThreadClusterN110Xs = Sequence<8, 2>;

    using GemmABlockTransferThreadSliceLengths_K0_M0_M1_K1   = Sequence<4, 1, 1, 1>;
    using GemmABlockTransferThreadClusterLengths_K0_M0_M1_K1 = Sequence<2, 1, 128, 1>;

    using GemmABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1 = Sequence<4, 1, 1, 1>;
    using GemmABlockTransferDstVectorTensorLengths_K0_M0_M1_K1 = Sequence<1, 1, 1, 1>;

    using GemmBBlockTransferThreadSliceLengths_K0_N0_N1_K1   = Sequence<4, 1, 1, 1>;
    using GemmBBlockTransferThreadClusterLengths_K0_N0_N1_K1 = Sequence<2, 1, 128, 1>;

    using GemmBBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1 = Sequence<4, 1, 1, 1>;
    using GemmBBlockTransferDstVectorTensorLengths_K0_N0_N1_K1 = Sequence<1, 1, 1, 1>;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_N11 = 4;
#elif 1
    // [M, N, K0, K1] = [128, 128, 8, 2] for fp16
    // cdata = 64, BlockSize = 256
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlockM1 = 128;
    constexpr index_t GemmNPerBlockN1 = 128;
    constexpr index_t GemmKPerBlock   = 8;
    constexpr index_t GemmK1          = 2;

    constexpr index_t GemmM1PerThreadM111 = 4;
    constexpr index_t GemmN1PerThreadN111 = 4;
    constexpr index_t GemmKPerThread      = 1;

    using GemmM11N11ThreadClusterM110Xs = Sequence<8, 2>;
    using GemmM11N11ThreadClusterN110Xs = Sequence<8, 2>;

    using GemmABlockTransferThreadSliceLengths_K0_M0_M1_K1   = Sequence<4, 1, 1, 2>;
    using GemmABlockTransferThreadClusterLengths_K0_M0_M1_K1 = Sequence<2, 1, 128, 1>;

    using GemmABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1 = Sequence<4, 1, 1, 2>;
    using GemmABlockTransferDstVectorTensorLengths_K0_M0_M1_K1 = Sequence<1, 1, 1, 2>;

    using GemmBBlockTransferThreadSliceLengths_K0_N0_N1_K1   = Sequence<4, 1, 1, 2>;
    using GemmBBlockTransferThreadClusterLengths_K0_N0_N1_K1 = Sequence<2, 1, 128, 1>;

    using GemmBBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1 = Sequence<4, 1, 1, 2>;
    using GemmBBlockTransferDstVectorTensorLengths_K0_N0_N1_K1 = Sequence<1, 1, 1, 2>;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_N11 = 4;
#elif 1
    // [M, N, K0, K1] = [128, 128, 8, 4] for i8
    // cdata = 64, BlockSize = 256
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlockM1 = 128;
    constexpr index_t GemmNPerBlockN1 = 128;
    constexpr index_t GemmKPerBlock   = 8;
    constexpr index_t GemmK1          = 4;

    constexpr index_t GemmM1PerThreadM111 = 4;
    constexpr index_t GemmN1PerThreadN111 = 4;
    constexpr index_t GemmKPerThread      = 1;

    using GemmM11N11ThreadClusterM110Xs = Sequence<8, 2>;
    using GemmM11N11ThreadClusterN110Xs = Sequence<8, 2>;

    using GemmABlockTransferThreadSliceLengths_K0_M0_M1_K1   = Sequence<4, 1, 1, 4>;
    using GemmABlockTransferThreadClusterLengths_K0_M0_M1_K1 = Sequence<2, 1, 128, 1>;

    using GemmABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1 = Sequence<4, 1, 1, 4>;
    using GemmABlockTransferDstVectorTensorLengths_K0_M0_M1_K1 = Sequence<1, 1, 1, 4>;

    using GemmBBlockTransferThreadSliceLengths_K0_N0_N1_K1   = Sequence<4, 1, 1, 4>;
    using GemmBBlockTransferThreadClusterLengths_K0_N0_N1_K1 = Sequence<2, 1, 128, 1>;

    using GemmBBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1 = Sequence<4, 1, 1, 4>;
    using GemmBBlockTransferDstVectorTensorLengths_K0_N0_N1_K1 = Sequence<1, 1, 1, 4>;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_N11 = 4;
#endif

    const auto descs =
        transform_forward_convolution_into_gemm_v4r4r4_nhwc_kyxc_nhwk_pad(in_n_hi_wi_c_desc,
                                                                          wei_k_y_x_c_desc,
                                                                          out_n_ho_wo_k_desc,
                                                                          conv_strides,
                                                                          conv_dilations,
                                                                          in_left_pads,
                                                                          in_right_pads,
                                                                          Number<GemmK1>{});

    const auto in_gemmk0_gemmm_gemmk1_grid_desc  = descs[I0];
    const auto wei_gemmk0_gemmn_gemmk1_grid_desc = descs[I1];
    const auto out_gemmm_gemmn_grid_desc         = descs[I2];

    // HACK: hacks that control index calculation when iterating over A, B, C matrix
    constexpr auto in_gemmk0_gemmm0_gemmm1_gemmk1_grid_iterator_hacks = make_tuple(
        make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0>{},   // 0+: GemmK0
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0>{},   // 1+: GemmM0
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0>{},   // 2+: GemmM1
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0>{}),  // 3+: GemmK1
        make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0>{},   // 0-: GemmK0
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0>{},   // 1-: GemmM0
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0>{},   // 3-: GemmM1
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0>{})); // 3-: GemmK1

    constexpr auto wei_gemmk0_gemmn0_gemmn1_gemmk1_grid_iterator_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0>{},   // 0+: GemmK0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0>{},   // 1+: GemmN0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0>{},   // 2+: GemmN1
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0>{}),  // 3+: GemmK1
                   make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0>{},   // 0-: GemmK0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0>{},   // 1-: GemmN0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0>{},   // 2-: GemmN1
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0>{})); // 3-: GemmK1

    constexpr auto out_gemmm0_gemmm10_gemmm11_gemmn0_gemmn10_gemmn11_grid_iterator_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},   // 0+: GemmM0
                              Sequence<0, 0, 0, 0, 0>{},   // 1+: GemmM10
                              Sequence<0, 0, 0, 0, 0>{},   // 2+: GemmM11
                              Sequence<0, 0, 0, 0, 0>{},   // 3+: GemmN0
                              Sequence<0, 0, 0, 0, 0>{},   // 4+: GemmN10
                              Sequence<0, 0, 0, 0, 0>{}),  // 5+: GemmN11
                   make_tuple(Sequence<0, 0, 0, 0, 0>{},   // 0-: GemmM0
                              Sequence<0, 0, 0, 0, 0>{},   // 1-: GemmM10
                              Sequence<0, 0, 0, 0, 0>{},   // 2-: GemmM11
                              Sequence<0, 0, 0, 0, 0>{},   // 3-: GemmN0
                              Sequence<0, 0, 0, 0, 0>{},   // 4-: GemmN10
                              Sequence<0, 0, 0, 0, 0>{})); // 5-: GemmN11

    constexpr auto in_gemmk0_gemmm0_gemmm1_gemmk1_grid_move_slice_window_iterator_hacks =
        Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0>{};

    constexpr auto wei_gemmk0_gemmn0_gemmn1_gemmk1_grid_move_slice_window_iterator_hacks =
        Sequence<0, 0, 0, 0, 0, 0, 0, 0>{};

    for(index_t i = 0; i < 5; ++i)
    {
        float ave_time = driver_dynamic_gemm_dlops_v1r3<
            BlockSize,
            TInWei,
            TAcc,
            TOut,
            InMemoryDataOperation::Set,
            decltype(in_gemmk0_gemmm_gemmk1_grid_desc),
            decltype(wei_gemmk0_gemmn_gemmk1_grid_desc),
            decltype(out_gemmm_gemmn_grid_desc),
            GemmMPerBlockM1,
            GemmNPerBlockN1,
            GemmKPerBlock,
            GemmM1PerThreadM111,
            GemmN1PerThreadN111,
            GemmKPerThread,
            GemmM11N11ThreadClusterM110Xs,
            GemmM11N11ThreadClusterN110Xs,
            GemmABlockTransferThreadSliceLengths_K0_M0_M1_K1,
            GemmABlockTransferThreadClusterLengths_K0_M0_M1_K1,
            Sequence<1, 2, 0, 3>, // ABlockTransferThreadClusterArrangeOrder
            Sequence<1, 2, 0, 3>, // ABlockTransferSrcAccessOrder
            GemmABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
            Sequence<1, 2, 0, 3>, // ABlockTransferSrcVectorTensorContiguousDimOrder
            GemmABlockTransferDstVectorTensorLengths_K0_M0_M1_K1,
            GemmBBlockTransferThreadSliceLengths_K0_N0_N1_K1,
            GemmBBlockTransferThreadClusterLengths_K0_N0_N1_K1,
            Sequence<1, 2, 0, 3>, // BBlockTransferThreadClusterArrangeOrder
            Sequence<1, 2, 0, 3>, // BBlockTransferSrcAccessOrder
            GemmBBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1,
            Sequence<1, 2, 0, 3>, // BBlockTransferSrcVectorTensorContiguousDimOrder
            GemmBBlockTransferDstVectorTensorLengths_K0_N0_N1_K1,
            Sequence<0, 1, 2, 3, 4, 5>, // CThreadTransferSrcDstAccessOrder
            5,                          // CThreadTransferSrcDstVectorDim
            GemmCThreadTransferDstScalarPerVector_N11,
            decltype(in_gemmk0_gemmm0_gemmm1_gemmk1_grid_iterator_hacks),
            decltype(wei_gemmk0_gemmn0_gemmn1_gemmk1_grid_iterator_hacks),
            decltype(out_gemmm0_gemmm10_gemmm11_gemmn0_gemmn10_gemmn11_grid_iterator_hacks),
            decltype(in_gemmk0_gemmm0_gemmm1_gemmk1_grid_move_slice_window_iterator_hacks),
            decltype(wei_gemmk0_gemmn0_gemmn1_gemmk1_grid_move_slice_window_iterator_hacks)>(
            static_cast<TInWei*>(in_n_hi_wi_c_device_buf.GetDeviceBuffer()),
            static_cast<TInWei*>(wei_k_y_x_c_device_buf.GetDeviceBuffer()),
            static_cast<TOut*>(out_n_ho_wo_k_device_buf.GetDeviceBuffer()),
            in_gemmk0_gemmm_gemmk1_grid_desc,
            wei_gemmk0_gemmn_gemmk1_grid_desc,
            out_gemmm_gemmn_grid_desc,
            in_gemmk0_gemmm0_gemmm1_gemmk1_grid_iterator_hacks,
            wei_gemmk0_gemmn0_gemmn1_gemmk1_grid_iterator_hacks,
            out_gemmm0_gemmm10_gemmm11_gemmn0_gemmn10_gemmn11_grid_iterator_hacks,
            in_gemmk0_gemmm0_gemmm1_gemmk1_grid_move_slice_window_iterator_hacks,
            wei_gemmk0_gemmn0_gemmn1_gemmk1_grid_move_slice_window_iterator_hacks,
            nrepeat);

        {
            const auto N = out_n_ho_wo_k_lengths[I0];
            const auto K = out_n_ho_wo_k_lengths[I3];
            const auto C = wei_k_y_x_c_lengths[I3];

            const auto Hi = in_n_hi_wi_c_lengths[I1];
            const auto Wi = in_n_hi_wi_c_lengths[I2];

            const auto Ho = out_n_ho_wo_k_lengths[I1];
            const auto Wo = out_n_ho_wo_k_lengths[I2];

            const auto Y = wei_k_y_x_c_lengths[I1];
            const auto X = wei_k_y_x_c_lengths[I2];

            float perf = (float)(std::size_t(2) * N * K * Ho * Wo * C * Y * X) /
                         (std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                      << std::endl;
        }
    }

    // copy result back to host
    out_n_ho_wo_k_device_buf.FromDevice(out_n_ho_wo_k.mData.data());
}
