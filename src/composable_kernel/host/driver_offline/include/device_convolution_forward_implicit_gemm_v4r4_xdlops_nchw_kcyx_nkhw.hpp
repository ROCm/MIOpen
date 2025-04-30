#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "driver_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw.hpp"

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
void device_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw(
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

    const auto in_n_c_hi_wi_desc  = make_naive_tensor_descriptor_packed(in_n_c_hi_wi_lengths);
    const auto wei_k_c_y_x_desc   = make_naive_tensor_descriptor_packed(wei_k_c_y_x_lengths);
    const auto out_n_k_ho_wo_desc = make_naive_tensor_descriptor_packed(out_n_k_ho_wo_lengths);

#if 0
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerWave = 64;
    constexpr index_t GemmNPerWave = 64;
    constexpr index_t GemmKPack    = 8;

    constexpr index_t MRepeat = 1;
    constexpr index_t NRepeat = 1;

    using GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1   = Sequence<1, 2, 8>;
    using GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 8;
    constexpr index_t GemmABlockTransferDstScalarPerVector_KPack = 8;

    using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1   = Sequence<1, 4, 4>;
    using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 = Sequence<4, 32, 2>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_KPack = 4;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmN1 = 1;
#elif 0
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 256;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerWave = 64;
    constexpr index_t GemmNPerWave = 64;
    constexpr index_t GemmKPack    = 8;

    constexpr index_t MRepeat = 2;
    constexpr index_t NRepeat = 1;

    using GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1   = Sequence<1, 4, 8>;
    using GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 8;
    constexpr index_t GemmABlockTransferDstScalarPerVector_KPack = 8;

    using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1   = Sequence<1, 4, 4>;
    using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 = Sequence<4, 32, 2>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_KPack = 4;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmN1 = 1;
#elif 0
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 256;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerWave = 64;
    constexpr index_t GemmNPerWave = 64;
    constexpr index_t GemmKPack    = 8;

    constexpr index_t MRepeat = 2;
    constexpr index_t NRepeat = 1;

    using GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1   = Sequence<1, 4, 8>;
    using GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 8;
    constexpr index_t GemmABlockTransferDstScalarPerVector_KPack = 8;

    using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1   = Sequence<1, 4, 4>;
    using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 = Sequence<4, 32, 2>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN = 1;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_KPack = 4;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmN1 = 1;
#elif 1
    // [M, N, K0, K1] = [256, 128, 4, 4]
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 256;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerWave = 64;
    constexpr index_t GemmNPerWave = 64;
    constexpr index_t GemmKPack    = 4;

    constexpr index_t MRepeat = 2;
    constexpr index_t NRepeat = 1;

    using GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1   = Sequence<1, 4, 4>;
    using GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmABlockTransferDstScalarPerVector_KPack = 4;

    using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1   = Sequence<1, 2, 4>;
    using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN = 1;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_KPack = 4;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmN1 = 1;
#elif 1
    // [M, N, K0, K1] = [128, 128, 4, 4]
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerWave = 64;
    constexpr index_t GemmNPerWave = 64;
    constexpr index_t GemmKPack    = 4;

    constexpr index_t MRepeat = 1;
    constexpr index_t NRepeat = 1;

    using GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1   = Sequence<1, 2, 4>;
    using GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmABlockTransferDstScalarPerVector_KPack = 4;

    using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1   = Sequence<1, 2, 4>;
    using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN = 1;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_KPack = 4;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmN1 = 1;
#endif

    const auto descs =
#if 1
        transform_forward_convolution_into_gemm_v4r4_xdlops_nchw_kcyx_nkhw_pad
#else
        transform_forward_convolution_into_gemm_v4r4_xdlops_nchw_kcyx_nkhw_1x1
#endif
        <TInWei, GemmMPerBlock, GemmNPerBlock, GemmMPerWave, GemmNPerWave, GemmKPack>(
            wei_k_c_y_x_desc,
            in_n_c_hi_wi_desc,
            out_n_k_ho_wo_desc,
            conv_strides,
            conv_dilations,
            in_left_pads,
            in_right_pads);

    for(index_t i = 0; i < 5; ++i)
    {
#if 0
        float ave_time = launch_kernel_gemm_xdlops_v1
#else
        float ave_time = launch_kernel_gemm_xdlops_v2
#endif
        <BlockSize,
         TInWei,
         TAcc,
         TOut,
         InMemoryDataOperationEnum_t::Set,
         decltype(descs[I0]),
         decltype(descs[I1]),
         decltype(descs[I2]),
         decltype(descs[I3]),
         GemmMPerBlock,
         GemmNPerBlock,
         GemmKPerBlock,
         GemmMPerWave,
         GemmNPerWave,
         GemmKPack,
         MRepeat,
         NRepeat,
         GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1,
         GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1,
         Sequence<1, 0, 2>,
         Sequence<1, 0, 2>,
         2,
         GemmABlockTransferSrcScalarPerVector_GemmK,
         GemmABlockTransferDstScalarPerVector_KPack,
         false, // don't move back src coordinate after threadwise copy
         GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1,
         GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1,
         Sequence<0, 2, 1>,
         Sequence<1, 0, 2>,
         1,
         GemmBBlockTransferSrcScalarPerVector_GemmN,
         GemmBBlockTransferDstScalarPerVector_KPack,
         false, // don't move back src coordinate after threadwise copy, which will be fused
                // with MoveSrcSliceWindow() to save addr computation
         Sequence<2, 3, 0, 1>,
         3,
         GemmCThreadTransferDstScalarPerVector_GemmN1,
         decltype(descs[I4]),
         decltype(descs[I5]),
         decltype(descs[I6]),
         decltype(descs[I7]),
         decltype(descs[I8])>(static_cast<TInWei*>(wei_k_c_y_x_device_buf.GetDeviceBuffer()),
                              static_cast<TInWei*>(in_n_c_hi_wi_device_buf.GetDeviceBuffer()),
                              static_cast<TOut*>(out_n_k_ho_wo_device_buf.GetDeviceBuffer()),
                              descs[I0],
                              descs[I1],
                              descs[I2],
                              descs[I3],
                              descs[I4],
                              descs[I5],
                              descs[I6],
                              descs[I7],
                              descs[I8],
                              nrepeat);

        float perf = (float)calculate_convolution_flops(
                         in_n_c_hi_wi_desc, wei_k_c_y_x_desc, out_n_k_ho_wo_desc) /
                     (std::size_t(1000) * 1000 * 1000) / ave_time;

        std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s" << std::endl;
    }

    // copy result back to host
    out_n_k_ho_wo_device_buf.FromDevice(out_n_k_ho_wo.mData.data());
}
