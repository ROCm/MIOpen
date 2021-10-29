#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm_xdlops_v2r3.hpp"
#include "transform_forward_convolution_into_gemm_v4r4r4_nhwc_kyxc_nhwk.hpp"

using namespace ck;

constexpr DataTypeEnum_t ABDataTypeEnum  = static_cast<DataTypeEnum_t>(CK_PARAM_ABDataTypeEnum);
constexpr DataTypeEnum_t AccDataTypeEnum = static_cast<DataTypeEnum_t>(CK_PARAM_AccDataTypeEnum);
constexpr DataTypeEnum_t CDataTypeEnum   = static_cast<DataTypeEnum_t>(CK_PARAM_CDataTypeEnum);

using FloatAB  = typename get_datatype_from_enum<ABDataTypeEnum>::type;
using FloatAcc = typename get_datatype_from_enum<AccDataTypeEnum>::type;
using FloatC   = typename get_datatype_from_enum<CDataTypeEnum>::type;

constexpr index_t BlockSize = CK_PARAM_BlockSize;

constexpr auto GemmMPerBlock     = Number<CK_PARAM_MPerBlock>{};
constexpr auto GemmNPerBlock     = Number<CK_PARAM_NPerBlock>{};
constexpr index_t GemmK0PerBlock = CK_PARAM_K0PerBlock;

constexpr index_t MPerXDL = CK_PARAM_MPerXDL;
constexpr index_t NPerXDL = CK_PARAM_NPerXDL;
constexpr index_t GemmK1  = CK_PARAM_K1;

constexpr index_t MRepeat = CK_PARAM_MRepeat;
constexpr index_t NRepeat = CK_PARAM_NRepeat;

using GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1 =
    Sequence<CK_PARAM_ABlockTransferThreadSliceLengths_K0_M_K1>;
using GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1 =
    Sequence<CK_PARAM_ABlockTransferThreadClusterLengths_K0_M_K1>;
using GemmABlockTransferThreadClusterArrangeOrder =
    Sequence<CK_PARAM_ABlockTransferThreadClusterArrangeOrder>;
using GemmABlockTransferSrcAccessOrder           = Sequence<CK_PARAM_ABlockTransferSrcAccessOrder>;
constexpr index_t GemmABlockTransferSrcVectorDim = CK_PARAM_ABlockTransferSrcVectorDim;
constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK1 =
    CK_PARAM_ABlockTransferSrcScalarPerVector;
constexpr index_t GemmABlockTransferDstScalarPerVector_GemmK1 =
    CK_PARAM_ABlockTransferDstScalarPerVector_K1;
constexpr index_t GemmAThreadTransferSrcResetCoordinateAfterRun =
    CK_PARAM_AThreadTransferSrcResetCoordinateAfterRun;

using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1 =
    Sequence<CK_PARAM_BBlockTransferThreadSliceLengths_K0_N_K1>;
using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 =
    Sequence<CK_PARAM_BBlockTransferThreadClusterLengths_K0_N_K1>;
using GemmBBlockTransferThreadClusterArrangeOrder =
    Sequence<CK_PARAM_BBlockTransferThreadClusterArrangeOrder>;
using GemmBBlockTransferSrcAccessOrder           = Sequence<CK_PARAM_BBlockTransferSrcAccessOrder>;
constexpr index_t GemmBBlockTransferSrcVectorDim = CK_PARAM_BBlockTransferSrcVectorDim;
constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK1 =
    CK_PARAM_BBlockTransferSrcScalarPerVector;
constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmK1 =
    CK_PARAM_BBlockTransferDstScalarPerVector_K1;
constexpr index_t GemmBThreadTransferSrcResetCoordinateAfterRun =
    CK_PARAM_BThreadTransferSrcResetCoordinateAfterRun;

using GemmCThreadTransferSrcDstAccessOrder = Sequence<CK_PARAM_CThreadTransferSrcDstAccessOrder>;
constexpr index_t GemmCThreadTransferSrcDstVectorDim = CK_PARAM_CThreadTransferSrcDstVectorDim;
constexpr index_t GemmCThreadTransferDstScalarPerVector =
    CK_PARAM_CThreadTransferDstScalarPerVector;

constexpr bool HasMainKBlockLoop = static_cast<bool>(CK_PARAM_HasMainKBlockLoop);

extern "C" __global__ void
convolution_forward_implicit_gemm_v4r4r4_xdlops_nhwc_kyxc_nhwk_prepare(int N_,
                                                                       int C_,
                                                                       int Hi_,
                                                                       int Wi_,
                                                                       int K_,
                                                                       int Y_,
                                                                       int X_,
                                                                       int ConvStrideH_,
                                                                       int ConvStrideW_,
                                                                       int ConvDilationH_,
                                                                       int ConvDilationW_,
                                                                       int InLeftPadH_,
                                                                       int InLeftPadW_,
                                                                       int InRightPadH_,
                                                                       int InRightPadW_,
                                                                       void* p_desc_tuple)
{
    index_t N             = static_cast<index_t>(N_);
    index_t C             = static_cast<index_t>(C_);
    index_t Hi            = static_cast<index_t>(Hi_);
    index_t Wi            = static_cast<index_t>(Wi_);
    index_t K             = static_cast<index_t>(K_);
    index_t Y             = static_cast<index_t>(Y_);
    index_t X             = static_cast<index_t>(X_);
    index_t ConvStrideH   = static_cast<index_t>(ConvStrideH_);
    index_t ConvStrideW   = static_cast<index_t>(ConvStrideW_);
    index_t ConvDilationH = static_cast<index_t>(ConvDilationH_);
    index_t ConvDilationW = static_cast<index_t>(ConvDilationW_);
    index_t InLeftPadH    = static_cast<index_t>(InLeftPadH_);
    index_t InLeftPadW    = static_cast<index_t>(InLeftPadW_);
    index_t InRightPadH   = static_cast<index_t>(InRightPadH_);
    index_t InRightPadW   = static_cast<index_t>(InRightPadW_);

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};

    const index_t Ho =
        (Hi + InLeftPadH + InRightPadH - ConvDilationH * (Y - 1) - 1) / ConvStrideH + 1;
    const index_t Wo =
        (Wi + InLeftPadW + InRightPadW - ConvDilationW * (X - 1) - 1) / ConvStrideW + 1;

    const auto in_n_hi_wi_c_desc  = make_naive_tensor_descriptor_packed(make_tuple(N, Hi, Wi, C));
    const auto wei_k_y_x_c_desc   = make_naive_tensor_descriptor_packed(make_tuple(K, Y, X, C));
    const auto out_n_ho_wo_k_desc = make_naive_tensor_descriptor_packed(make_tuple(N, Ho, Wo, K));

    const auto descs = transform_forward_convolution_into_gemm_v4r4r4_nhwc_kyxc_nhwk_pad(
        in_n_hi_wi_c_desc,
        wei_k_y_x_c_desc,
        out_n_ho_wo_k_desc,
        make_tuple(ConvStrideH, ConvStrideW),
        make_tuple(ConvDilationH, ConvDilationW),
        make_tuple(InLeftPadH, InLeftPadW),
        make_tuple(InRightPadH, InRightPadW),
        Number<GemmK1>{});

    const auto in_gemmk0_gemmm_gemmk1_grid_desc  = descs[I0];
    const auto wei_gemmk0_gemmn_gemmk1_grid_desc = descs[I1];
    const auto out_gemmm_gemmn_grid_desc         = descs[I2];

    // HACK: hacks that control index calculation when iterating over A, B, C matrix
    constexpr auto in_gemmk0_gemmm_gemmk1_grid_step_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{},   // 0+: GemmK0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0>{},   // 1+: GemmM
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{}),  // 2+: GemmK1
                   make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{},   // 0-: GemmK0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0>{},   // 1-: GemmM
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{})); // 2-: GemmK1

    constexpr auto wei_gemmk0_gemmn_gemmk1_grid_step_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},   // 0+: GemmK0
                              Sequence<0, 0, 0, 0, 0>{},   // 1+: GemmN
                              Sequence<0, 0, 0, 0, 0>{}),  // 2+: GemmK1
                   make_tuple(Sequence<0, 0, 0, 0, 0>{},   // 0-: GemmK0
                              Sequence<0, 0, 0, 0, 0>{},   // 1-: GemmN
                              Sequence<0, 0, 0, 0, 0>{})); // 2-: GemmK1

    constexpr auto out_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks =
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

    constexpr auto in_gemmk0_gemmm_gemmk1_grid_move_slice_window_step_hacks =
        Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0>{};

    constexpr auto wei_gemmk0_gemmn_gemmk1_grid_move_slice_window_step_hacks =
        Sequence<0, 0, 0, 0, 0>{};

    using GridwiseContraction = GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3<
        BlockSize,
        FloatAB,
        FloatAcc,
        FloatC,
        InMemoryDataOperationEnum_t::Set,
        decltype(in_gemmk0_gemmm_gemmk1_grid_desc),
        decltype(wei_gemmk0_gemmn_gemmk1_grid_desc),
        decltype(out_gemmm_gemmn_grid_desc),
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
        GemmABlockTransferThreadClusterArrangeOrder,
        GemmABlockTransferSrcAccessOrder,
        GemmABlockTransferSrcVectorDim,
        GemmABlockTransferSrcScalarPerVector_GemmK1,
        GemmABlockTransferDstScalarPerVector_GemmK1,
        GemmAThreadTransferSrcResetCoordinateAfterRun,
        GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1,
        GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1,
        GemmBBlockTransferThreadClusterArrangeOrder,
        GemmBBlockTransferSrcAccessOrder,
        GemmBBlockTransferSrcVectorDim,
        GemmBBlockTransferSrcScalarPerVector_GemmK1,
        GemmBBlockTransferDstScalarPerVector_GemmK1,
        GemmBThreadTransferSrcResetCoordinateAfterRun,
        GemmCThreadTransferSrcDstAccessOrder,
        GemmCThreadTransferSrcDstVectorDim,
        GemmCThreadTransferDstScalarPerVector,
        decltype(in_gemmk0_gemmm_gemmk1_grid_step_hacks),
        decltype(wei_gemmk0_gemmn_gemmk1_grid_step_hacks),
        decltype(out_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks),
        decltype(in_gemmk0_gemmm_gemmk1_grid_move_slice_window_step_hacks),
        decltype(wei_gemmk0_gemmn_gemmk1_grid_move_slice_window_step_hacks),
        false>;

    if(get_block_1d_id() == 0 && get_thread_local_1d_id() == 0)
    {
        auto desc_tuple = make_tuple(
            in_gemmk0_gemmm_gemmk1_grid_desc,
            wei_gemmk0_gemmn_gemmk1_grid_desc,
            GridwiseContraction::MakeCM0N0M1N1M2M3M4N2GridDescriptor(out_gemmm_gemmn_grid_desc),
            GridwiseContraction::MakeCBlockClusterAdaptor(out_gemmm_gemmn_grid_desc));

        *static_cast<decltype(desc_tuple)*>(p_desc_tuple) = desc_tuple;
    }
};

extern "C" __global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        convolution_forward_implicit_gemm_v4r4r4_xdlops_nhwc_kyxc_nhwk(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            const void CONSTANT* p_desc_tuple)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_n_hi_wi_c_desc =
        make_naive_tensor_descriptor_packed(make_tuple(256, 28, 28, 256));
    constexpr auto wei_k_y_x_c_desc =
        make_naive_tensor_descriptor_packed(make_tuple(256, 3, 3, 256));
    constexpr auto out_n_ho_wo_k_desc =
        make_naive_tensor_descriptor_packed(make_tuple(256, 28, 28, 256));

    constexpr auto descs =
        transform_forward_convolution_into_gemm_v4r4r4_nhwc_kyxc_nhwk_pad(in_n_hi_wi_c_desc,
                                                                          wei_k_y_x_c_desc,
                                                                          out_n_ho_wo_k_desc,
                                                                          make_tuple(1, 1),
                                                                          make_tuple(1, 1),
                                                                          make_tuple(1, 1),
                                                                          make_tuple(1, 1),
                                                                          Number<GemmK1>{});

    const auto in_gemmk0_gemmm_gemmk1_grid_desc  = descs[I0];
    const auto wei_gemmk0_gemmn_gemmk1_grid_desc = descs[I1];
    const auto out_gemmm_gemmn_grid_desc         = descs[I2];

    using AK0MK1GridDesc = decltype(a_k0_m_k1_grid_desc_tmp);
    using BK0NK1GridDesc = decltype(b_k0_n_k1_grid_desc_tmp);
    using CMNGridDesc    = decltype(c_m_n_grid_desc);

    // HACK: hacks that control index calculation when iterating over A, B, C matrix
    constexpr auto in_gemmk0_gemmm_gemmk1_grid_step_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{},   // 0+: GemmK0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0>{},   // 1+: GemmM
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{}),  // 2+: GemmK1
                   make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{},   // 0-: GemmK0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0>{},   // 1-: GemmM
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{})); // 2-: GemmK1

    constexpr auto wei_gemmk0_gemmn_gemmk1_grid_step_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},   // 0+: GemmK0
                              Sequence<0, 0, 0, 0, 0>{},   // 1+: GemmN
                              Sequence<0, 0, 0, 0, 0>{}),  // 2+: GemmK1
                   make_tuple(Sequence<0, 0, 0, 0, 0>{},   // 0-: GemmK0
                              Sequence<0, 0, 0, 0, 0>{},   // 1-: GemmN
                              Sequence<0, 0, 0, 0, 0>{})); // 2-: GemmK1

    constexpr auto out_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks =
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

    constexpr auto in_gemmk0_gemmm_gemmk1_grid_move_slice_window_step_hacks =
        Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0>{};

    constexpr auto wei_gemmk0_gemmn_gemmk1_grid_move_slice_window_step_hacks =
        Sequence<0, 0, 0, 0, 0>{};

    using GridwiseContraction = GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3<
        BlockSize,
        FloatAB,
        FloatAcc,
        FloatC,
        InMemoryDataOperationEnum_t::Set,
        decltype(in_gemmk0_gemmm_gemmk1_grid_desc),
        decltype(wei_gemmk0_gemmn_gemmk1_grid_desc),
        decltype(out_gemmm_gemmn_grid_desc),
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
        GemmABlockTransferThreadClusterArrangeOrder,
        GemmABlockTransferSrcAccessOrder,
        GemmABlockTransferSrcVectorDim,
        GemmABlockTransferSrcScalarPerVector_GemmK1,
        GemmABlockTransferDstScalarPerVector_GemmK1,
        GemmAThreadTransferSrcResetCoordinateAfterRun,
        GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1,
        GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1,
        GemmBBlockTransferThreadClusterArrangeOrder,
        GemmBBlockTransferSrcAccessOrder,
        GemmBBlockTransferSrcVectorDim,
        GemmBBlockTransferSrcScalarPerVector_GemmK1,
        GemmBBlockTransferDstScalarPerVector_GemmK1,
        GemmBThreadTransferSrcResetCoordinateAfterRun,
        GemmCThreadTransferSrcDstAccessOrder,
        GemmCThreadTransferSrcDstVectorDim,
        GemmCThreadTransferDstScalarPerVector,
        decltype(in_gemmk0_gemmm_gemmk1_grid_step_hacks),
        decltype(wei_gemmk0_gemmn_gemmk1_grid_step_hacks),
        decltype(out_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks),
        decltype(in_gemmk0_gemmm_gemmk1_grid_move_slice_window_step_hacks),
        decltype(wei_gemmk0_gemmn_gemmk1_grid_move_slice_window_step_hacks),
        false>;

    constexpr auto c_m0_m1_m2_n_grid_desc_tmp =
        GridwiseGemm::MakeCM0M1M2NGridDescriptor(c_m_n_grid_desc);
    constexpr auto c_blockid_to_m0_n0_block_cluster_adaptor_tmp =
        GridwiseGemm::MakeCBlockClusterAdaptor(c_m_n_grid_desc);

    using DescTuple = decltype(make_tuple(in_gemmk0_gemmm_gemmk1_grid_desc,
                                          wei_gemmk0_gemmn_gemmk1_grid_desc,
                                          c_m0_m1_m2_n_grid_desc_tmp,
                                          c_blockid_to_m0_n0_block_cluster_adaptor_tmp));

    const auto desc_tuple = *reinterpret_cast<const DescTuple*>(
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
        // TODO: how to cast?
        (const void*)p_desc_tuple
#pragma clang diagnostic pop
    );

    const auto a_k0_m_k1_grid_desc                      = desc_tuple[I0];
    const auto b_k0_n_k1_grid_desc                      = desc_tuple[I1];
    const auto c_m0_m1_m2_n_grid_desc                   = desc_tuple[I2];
    const auto c_blockid_to_m0_n0_block_cluster_adaptor = desc_tuple[I3];

    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    GridwiseGemm::Run(p_a_grid,
                      p_b_grid,
                      p_c_grid,
                      p_shared_block,
                      a_k0_m_k1_grid_desc,
                      b_k0_n_k1_grid_desc,
                      c_m0_m1_m2_n_grid_desc,
                      c_blockid_to_m0_n0_block_cluster_adaptor);
};
