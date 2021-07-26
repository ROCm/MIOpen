#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "transform_forward_convolution_into_gemm_v4r4_nchw_kcyx_nkhw.hpp"
#include "driver_dynamic_gemm_dlops_v1r2.hpp"

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
void device_dynamic_convolution_forward_implicit_gemm_v4r4_dlops_nchw_kcyx_nkhw(
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
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};
    constexpr auto I5 = Number<5>{};
    constexpr auto I6 = Number<6>{};
    constexpr auto I7 = Number<7>{};
    constexpr auto I8 = Number<8>{};

    DeviceMem in_n_c_hi_wi_device_buf(sizeof(TInWei) * in_n_c_hi_wi.mDesc.GetElementSpace());
    DeviceMem wei_k_c_y_x_device_buf(sizeof(TInWei) * wei_k_c_y_x.mDesc.GetElementSpace());
    DeviceMem out_n_k_ho_wo_device_buf(sizeof(TOut) * out_n_k_ho_wo.mDesc.GetElementSpace());

    in_n_c_hi_wi_device_buf.ToDevice(in_n_c_hi_wi.mData.data());
    wei_k_c_y_x_device_buf.ToDevice(wei_k_c_y_x.mData.data());
    out_n_k_ho_wo_device_buf.ToDevice(out_n_k_ho_wo.mData.data());

    const auto in_n_c_hi_wi_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(in_n_c_hi_wi_lengths);
    const auto wei_k_c_y_x_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(wei_k_c_y_x_lengths);
    const auto out_n_k_ho_wo_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(out_n_k_ho_wo_lengths);

#if 1
    // cdata = 64, BlockSize = 256, 128x128x8
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlockM1 = 128;
    constexpr index_t GemmNPerBlockN1 = 128;
    constexpr index_t GemmKPerBlock   = 8;

    constexpr index_t GemmM1PerThreadM111 = 4;
    constexpr index_t GemmN1PerThreadN111 = 4;
    constexpr index_t GemmKPerThread      = 1;

    constexpr index_t GemmM11N11ThreadClusterM1100 = 8;
    constexpr index_t GemmM11N11ThreadClusterN1100 = 8;
    constexpr index_t GemmM11N11ThreadClusterM1101 = 2;
    constexpr index_t GemmM11N11ThreadClusterN1101 = 2;

    using GemmABlockTransferThreadSliceLengths_K_M0_M1   = Sequence<4, 1, 1>;
    using GemmABlockTransferThreadClusterLengths_K_M0_M1 = Sequence<2, 1, 128>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_K  = 4;
    constexpr index_t GemmABlockTransferDstScalarPerVector_M1 = 1;

    using GemmBBlockTransferThreadSliceLengths_K_N0_N1   = Sequence<4, 1, 1>;
    using GemmBBlockTransferThreadClusterLengths_K_N0_N1 = Sequence<2, 1, 128>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_N1 = 1;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_N1 = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_N11 = 1;
#endif

    const auto descs =
        transform_forward_convolution_into_gemm_v4r4_nchw_kcyx_nkhw_pad(wei_k_c_y_x_desc,
                                                                        in_n_c_hi_wi_desc,
                                                                        out_n_k_ho_wo_desc,
                                                                        conv_strides,
                                                                        conv_dilations,
                                                                        in_left_pads,
                                                                        in_right_pads);

    // HACK: hacks that control index calculation when iterating over A, B, C matrix
    constexpr auto wei_gemmk_gemmm0_gemmn1_grid_iterator_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{}));

    constexpr auto in_gemmk_gemmn0_gemmn1_grid_iterator_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0>{}));

    constexpr auto out_gemmm0_gemmm10_gemmm11_gemmn0_gemmn10_gemmn11_grid_iterator_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 1, 0, 0>{},
                              Sequence<0, 0, 1, 0, 0>{},
                              Sequence<0, 0, 1, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 2, 0, 0>{},
                              Sequence<0, 0, 2, 0, 0>{},
                              Sequence<0, 0, 2, 0, 0>{}));

    constexpr auto wei_gemmk_gemmm0_gemmm1_grid_move_slice_window_iterator_hacks =
        Sequence<0, 0, 0, 0, 0>{};

    constexpr auto in_gemmk_gemmn0_gemmn1_grid_move_slice_window_iterator_hacks =
        Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0>{};

    const auto wei_gemmk_gemmm_grid_desc = descs[I0];
    const auto in_gemmk_gemmn_grid_desc  = descs[I1];
    const auto out_gemmm_gemmn_grid_desc = descs[I2];

    for(index_t i = 0; i < 5; ++i)
    {
        float ave_time = driver_dynamic_gemm_dlops_v1r2<
            BlockSize,
            TInWei,
            TAcc,
            TOut,
            InMemoryDataOperationEnum_t::Set,
            decltype(wei_gemmk_gemmm_grid_desc),
            decltype(in_gemmk_gemmn_grid_desc),
            decltype(out_gemmm_gemmn_grid_desc),
            GemmMPerBlockM1,
            GemmNPerBlockN1,
            GemmKPerBlock,
            GemmM1PerThreadM111,
            GemmN1PerThreadN111,
            GemmKPerThread,
            GemmM11N11ThreadClusterM1100,
            GemmM11N11ThreadClusterN1100,
            GemmM11N11ThreadClusterM1101,
            GemmM11N11ThreadClusterN1101,
            GemmABlockTransferThreadSliceLengths_K_M0_M1,
            GemmABlockTransferThreadClusterLengths_K_M0_M1,
            Sequence<2, 1, 0>, // ABlockTransferThreadClusterArrangeOrder
            Sequence<2, 1, 0>, // ABlockTransferSrcAccessOrder
            0,                 // ABlockTransferSrcVectorDim
            GemmABlockTransferSrcScalarPerVector_K,
            GemmABlockTransferDstScalarPerVector_M1,
            false, // don't move back src coordinate after threadwise copy
            GemmBBlockTransferThreadSliceLengths_K_N0_N1,
            GemmBBlockTransferThreadClusterLengths_K_N0_N1,
            Sequence<0, 1, 2>, // BBlockTransferThreadClusterArrangeOrder
            Sequence<0, 1, 2>, // BBlockTransferSrcAccessOrder
            2,                 // BBlockTransferSrcVectorDim
            GemmBBlockTransferSrcScalarPerVector_N1,
            GemmBBlockTransferDstScalarPerVector_N1,
            false,                      // don't move back src coordinate after threadwise copy
            Sequence<3, 4, 5, 0, 1, 2>, // CThreadTransferSrcDstAccessOrder
            5,                          // CThreadTransferSrcDstVectorDim
            GemmCThreadTransferDstScalarPerVector_N11,
            decltype(wei_gemmk_gemmm0_gemmn1_grid_iterator_hacks),
            decltype(in_gemmk_gemmn0_gemmn1_grid_iterator_hacks),
            decltype(out_gemmm0_gemmm10_gemmm11_gemmn0_gemmn10_gemmn11_grid_iterator_hacks),
            decltype(wei_gemmk_gemmm0_gemmm1_grid_move_slice_window_iterator_hacks),
            decltype(in_gemmk_gemmn0_gemmn1_grid_move_slice_window_iterator_hacks)>(
            static_cast<TInWei*>(wei_k_c_y_x_device_buf.GetDeviceBuffer()),
            static_cast<TInWei*>(in_n_c_hi_wi_device_buf.GetDeviceBuffer()),
            static_cast<TOut*>(out_n_k_ho_wo_device_buf.GetDeviceBuffer()),
            wei_gemmk_gemmm_grid_desc,
            in_gemmk_gemmn_grid_desc,
            out_gemmm_gemmn_grid_desc,
            wei_gemmk_gemmm0_gemmn1_grid_iterator_hacks,
            in_gemmk_gemmn0_gemmn1_grid_iterator_hacks,
            out_gemmm0_gemmm10_gemmm11_gemmn0_gemmn10_gemmn11_grid_iterator_hacks,
            wei_gemmk_gemmm0_gemmm1_grid_move_slice_window_iterator_hacks,
            in_gemmk_gemmn0_gemmn1_grid_move_slice_window_iterator_hacks,
            nrepeat);

        float perf = (float)calculate_convolution_flops(
                         in_n_c_hi_wi_desc, wei_k_c_y_x_desc, out_n_k_ho_wo_desc) /
                     (std::size_t(1000) * 1000 * 1000) / ave_time;

        std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s" << std::endl;
    }

    // copy result back to host
    out_n_k_ho_wo_device_buf.FromDevice(out_n_k_ho_wo.mData.data());
}
