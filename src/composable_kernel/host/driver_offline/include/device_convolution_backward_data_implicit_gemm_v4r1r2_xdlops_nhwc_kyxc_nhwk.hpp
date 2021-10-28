#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "transform_backward_data_convolution_into_gemm_v4r1r2_nhwc_kyxc_nhwk.hpp"
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
void device_convolution_backward_data_implicit_gemm_v4r1r2_xdlops_nhwc_kyxc_nhwk(
    const InLengths& in_n_hi_wi_c_lengths,
    const WeiLengths& wei_k_y_x_c_lengths,
    const OutLengths& out_n_ho_wo_k_lengths,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads,
    Tensor<TInWei>& in_n_hi_wi_c,
    const Tensor<TInWei>& wei_k_y_x_c,
    const Tensor<TOut>& out_n_ho_wo_k,
    ck::index_t nrepeat)
{
    using namespace ck;

    std::cout << __func__ << std::endl;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    DeviceMem in_n_hi_wi_c_device_buf(sizeof(TInWei) * in_n_hi_wi_c.mDesc.GetElementSpace());
    DeviceMem wei_k_y_x_c_device_buf(sizeof(TInWei) * wei_k_y_x_c.mDesc.GetElementSpace());
    DeviceMem out_n_ho_wo_k_device_buf(sizeof(TOut) * out_n_ho_wo_k.mDesc.GetElementSpace());

    in_n_hi_wi_c_device_buf.ToDevice(in_n_hi_wi_c.mData.data());
    wei_k_y_x_c_device_buf.ToDevice(wei_k_y_x_c.mData.data());
    out_n_ho_wo_k_device_buf.ToDevice(out_n_ho_wo_k.mData.data());

    const auto in_n_hi_wi_c_desc  = make_naive_tensor_descriptor_packed(in_n_hi_wi_c_lengths);
    const auto wei_k_y_x_c_desc   = make_naive_tensor_descriptor_packed(wei_k_y_x_c_lengths);
    const auto out_n_ho_wo_k_desc = make_naive_tensor_descriptor_packed(out_n_ho_wo_k_lengths);

#if 0
    // [M, N, K0, K1] = [256, 128, 4, 4], C = 128,  for fp32
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 256;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerWave = 32;
    constexpr index_t GemmNPerWave = 32;
    constexpr index_t GemmK1       = 4;

    constexpr index_t MRepeat = 4;
    constexpr index_t NRepeat = 2;

    using GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1   = Sequence<1, 4, 4>;
    using GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK1 = 4;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmK1 = 4;

    using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1   = Sequence<1, 2, 4>;
    using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN  = 2;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmK1 = 4;

    constexpr index_t GemmCThreadTransferDstScalarPerVector = 1;
#elif 0
    // [M, N, K0, K1] = [128, 128, 4, 4], C = 64, for fp32
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerWave = 32;
    constexpr index_t GemmNPerWave = 32;
    constexpr index_t GemmK1       = 4;

    constexpr index_t MRepeat = 2;
    constexpr index_t NRepeat = 2;

    using GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1   = Sequence<1, 2, 4>;
    using GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK1 = 4;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmK1 = 4;

    using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1   = Sequence<1, 2, 4>;
    using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN  = 2;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmK1 = 4;

    constexpr index_t GemmCThreadTransferDstScalarPerVector = 1;
#elif 0
    // [M, N, K0, K1] = [256, 128, 4, 8], C = 128, for fp16
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

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN  = 2;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmK1 = 8;

    constexpr index_t GemmCThreadTransferDstScalarPerVector = 1;
#elif 1
    // [M, N, K0, K1] = [128, 256, 4, 8], C = 128, for fp16
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 256;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerWave = 32;
    constexpr index_t GemmNPerWave = 32;
    constexpr index_t GemmK1       = 8;

    constexpr index_t MRepeat = 2;
    constexpr index_t NRepeat = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1   = Sequence<1, 2, 8>;
    using GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK1 = 8;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmK1 = 8;

    using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1   = Sequence<1, 4, 8>;
    using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN  = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmK1 = 8;

    constexpr index_t GemmCThreadTransferDstScalarPerVector = 1;
#elif 1
    // [M, N, K0, K1] = [128, 128, 4, 8], C = 64, for fp16
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

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK1 = 8;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmK1 = 8;

    using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1   = Sequence<1, 2, 8>;
    using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN  = 2;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmK1 = 8;

    constexpr index_t GemmCThreadTransferDstScalarPerVector = 1;
#elif 0
    // [M, N, K0, K1] = [128, 64, 4, 8], C = 64, for fp16
    constexpr index_t BlockSize = 128;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 64;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerWave = 32;
    constexpr index_t GemmNPerWave = 32;
    constexpr index_t GemmK1       = 8;

    constexpr index_t MRepeat = 2;
    constexpr index_t NRepeat = 2;

    using GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1   = Sequence<1, 4, 8>;
    using GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1 = Sequence<4, 32, 1>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK1 = 8;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmK1 = 8;

    using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1   = Sequence<1, 2, 8>;
    using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 = Sequence<4, 32, 1>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN  = 2;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmK1 = 8;

    constexpr index_t GemmCThreadTransferDstScalarPerVector = 1;
#elif 0
    // [M, N, K0, K1] = [128, 64, 4, 8], C = 32, for fp16
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 64;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerWave = 32;
    constexpr index_t GemmNPerWave = 32;
    constexpr index_t GemmK1       = 8;

    constexpr index_t MRepeat = 2;
    constexpr index_t NRepeat = 1;

    using GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1   = Sequence<1, 2, 8>;
    using GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK1 = 8;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmK1 = 8;

    using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1   = Sequence<1, 1, 8>;
    using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN  = 1;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmK1 = 8;

    constexpr index_t GemmCThreadTransferDstScalarPerVector = 1;
#endif

    // HACK: hacks that control index calculation when iterating over A, B, C matrix
    constexpr auto out_gemmk0_gemmm_gemmk1_grid_step_hacks = make_tuple(
        make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0>{},   // 0+: gemmk0
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0>{},   // 1+: gemmm
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}),  // 2+: gemmk1
        make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0>{},   // 0-: gemmk0
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0>{},   // 1-: gemmm
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{})); // 2-:
                                                                                       // gemmk1

    constexpr auto wei_gemmk0_gemmn_gemmk1_grid_step_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0>{},   // 0+: gemmk0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 1+: gemmn
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}),  // 2+: gemmk1
                   make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0>{},   // 0-: Gemmk0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 1-: Gemmn
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{})); // 2-: Gemmk1

    // clang-format off
    constexpr auto in_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks = make_tuple(
        make_tuple(
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 0+: M0
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 1+: N0
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 2+: M1
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 3+: N1
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 4+: M2
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 5+: M3
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 6+: M4
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}),  // 7+: N2
        make_tuple(
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 0-: M0
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 1-: N0
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 2-: M1
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 3-: N1
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 4-: M2
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 5-: M3
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 6-: M4
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{})); // 7-: N2
    // clang-format on

    constexpr auto out_gemmk0_gemmm_gemmk1_grid_move_slice_window_step_hacks =
        Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0>{};

    constexpr auto wei_gemmk0_gemmn_gemmk1_grid_move_slice_window_step_hacks =
        Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0>{};

    for(index_t i = 0; i < 5; ++i)
    {
        const auto ConvStrideH = conv_strides[I0];
        const auto ConvStrideW = conv_strides[I1];

        const auto ConvDilationH = conv_dilations[I0];
        const auto ConvDilationW = conv_dilations[I1];

        const auto GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
        const auto GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

        const auto YTilda = ConvStrideH / GcdStrideDilationH;
        const auto XTilda = ConvStrideW / GcdStrideDilationW;

        float ave_time = 0;

        for(index_t i_ytilda = 0; i_ytilda < YTilda; ++i_ytilda)
        {
            for(index_t i_xtilda = 0; i_xtilda < XTilda; ++i_xtilda)
            {
                const auto descs =
                    transform_backward_data_convolution_into_gemm_v4r1r2_nhwc_kyxc_nhwk(
                        out_n_ho_wo_k_desc,
                        wei_k_y_x_c_desc,
                        in_n_hi_wi_c_desc,
                        conv_strides,
                        conv_dilations,
                        in_left_pads,
                        in_right_pads,
                        i_ytilda,
                        i_xtilda,
                        Number<GemmK1>{});

                const auto out_gemmk0_gemmm_gemmk1_grid_desc = descs[I0];
                const auto wei_gemmk0_gemmn_gemmk1_grid_desc = descs[I1];
                const auto in_gemmm_gemmn_grid_desc          = descs[I2];

                const auto GemmK0 = out_gemmk0_gemmm_gemmk1_grid_desc.GetLength(I0);

                if(GemmK0 != 0)
                {
                    ave_time += driver_gemm_xdlops_v2r3<
                        BlockSize,
                        TInWei,
                        TAcc,
                        TOut,
                        InMemoryDataOperationEnum_t::Set,
                        decltype(out_gemmk0_gemmm_gemmk1_grid_desc),
                        decltype(wei_gemmk0_gemmn_gemmk1_grid_desc),
                        decltype(in_gemmm_gemmn_grid_desc),
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
                        Sequence<2, 0, 1>,
                        Sequence<0, 2, 1>,
                        1,
                        GemmBBlockTransferSrcScalarPerVector_GemmN,
                        GemmBBlockTransferDstScalarPerVector_GemmK1,
                        false, // don't move back src coordinate after threadwise copy
#if 0
                        Sequence<0, 2, 4, 5, 6, 1, 3, 7>,
#else
                        Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
#endif
                        7,
                        GemmCThreadTransferDstScalarPerVector,
                        decltype(out_gemmk0_gemmm_gemmk1_grid_step_hacks),
                        decltype(wei_gemmk0_gemmn_gemmk1_grid_step_hacks),
                        decltype(in_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks),
                        decltype(out_gemmk0_gemmm_gemmk1_grid_move_slice_window_step_hacks),
                        decltype(wei_gemmk0_gemmn_gemmk1_grid_move_slice_window_step_hacks),
                        true,  // CAccessOrderMRepeatNRepeat
                        false, // ABlockLdsExtraM
                        false  // BBlockLdsExtraN
                        >(static_cast<TOut*>(out_n_ho_wo_k_device_buf.GetDeviceBuffer()),
                          static_cast<TInWei*>(wei_k_y_x_c_device_buf.GetDeviceBuffer()),
                          static_cast<TInWei*>(in_n_hi_wi_c_device_buf.GetDeviceBuffer()),
                          out_gemmk0_gemmm_gemmk1_grid_desc,
                          wei_gemmk0_gemmn_gemmk1_grid_desc,
                          in_gemmm_gemmn_grid_desc,
                          debug::debug_driver_gemm_xdlops_v2r3::M01,
                          debug::debug_driver_gemm_xdlops_v2r3::N01,
                          out_gemmk0_gemmm_gemmk1_grid_step_hacks,
                          wei_gemmk0_gemmn_gemmk1_grid_step_hacks,
                          in_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks,
                          out_gemmk0_gemmm_gemmk1_grid_move_slice_window_step_hacks,
                          wei_gemmk0_gemmn_gemmk1_grid_move_slice_window_step_hacks,
                          nrepeat);
                }
            }
        }

        {
            const auto N = out_n_ho_wo_k_lengths[I0];
            const auto K = out_n_ho_wo_k_lengths[I3];
            const auto C = wei_k_y_x_c_lengths[I3];

            const auto Ho = out_n_ho_wo_k_lengths[I1];
            const auto Wo = out_n_ho_wo_k_lengths[I2];

            const auto Y = wei_k_y_x_c_lengths[I1];
            const auto X = wei_k_y_x_c_lengths[I2];

            float perf = static_cast<float>((std::size_t(2) * N * K * Ho * Wo * C * Y * X)) /
                         (std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                      << std::endl;
        }
    }

    // copy result back to host
    in_n_hi_wi_c_device_buf.FromDevice(in_n_hi_wi_c.mData.data());
}
