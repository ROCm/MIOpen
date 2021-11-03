#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_contraction_dlops_v1r2.hpp"
#include "transform_forward_convolution_into_gemm_v6r1_nchw_kcyx_nkhw.hpp"

using namespace ck;

constexpr DataTypeEnum_t ABDataTypeEnum  = static_cast<DataTypeEnum_t>(CK_PARAM_ABDataTypeEnum);
constexpr DataTypeEnum_t AccDataTypeEnum = static_cast<DataTypeEnum_t>(CK_PARAM_AccDataTypeEnum);
constexpr DataTypeEnum_t CDataTypeEnum   = static_cast<DataTypeEnum_t>(CK_PARAM_CDataTypeEnum);

using FloatAB  = typename get_datatype_from_enum<ABDataTypeEnum>::type;
using FloatAcc = typename get_datatype_from_enum<AccDataTypeEnum>::type;
using FloatC   = typename get_datatype_from_enum<CDataTypeEnum>::type;

constexpr index_t BlockSize = CK_PARAM_BlockSize;

constexpr auto GN0 = Number<CK_PARAM_GN0>{};
constexpr auto GK1 = Number<CK_PARAM_GK1>{};

constexpr index_t GM1PerBlockGM11 = CK_PARAM_GM1PerBlockGM11;
constexpr index_t GN1PerBlockGN11 = CK_PARAM_GN1PerBlockGN11;
constexpr index_t GK0PerBlock     = CK_PARAM_GK0PerBlock;

constexpr index_t BM1PerThreadBM11 = CK_PARAM_BM1PerThreadBM11;
constexpr index_t BN1PerThreadBN11 = CK_PARAM_BN1PerThreadBN11;
constexpr index_t BK0PerThread     = CK_PARAM_BK0PerThread;

using BM10BN10ThreadClusterBM10Xs = Sequence<CK_PARAM_BM10BN10ThreadClusterBM10Xs>;
using BM10BN10ThreadClusterBN10Xs = Sequence<CK_PARAM_BM10BN10ThreadClusterBN10Xs>;

using ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1 =
    Sequence<CK_PARAM_ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1>;
using ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1 =
    Sequence<CK_PARAM_ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1>;
using ABlockTransferThreadClusterArrangeOrder = Sequence<1, 2, 3, 0, 4>;
using ABlockTransferSrcAccessOrder            = Sequence<3, 2, 1, 0, 4>;
using ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1 =
    Sequence<CK_PARAM_ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1>;
using ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1 =
    Sequence<CK_PARAM_ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1>;
using ABlockTransferSrcVectorTensorContiguousDimOrder = Sequence<0, 1, 2, 3, 4>;

using BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1 =
    Sequence<CK_PARAM_BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1>;
using BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1 =
    Sequence<CK_PARAM_BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1>;
using BBlockTransferThreadClusterArrangeOrder = Sequence<0, 4, 1, 2, 3>;
using BBlockTransferSrcAccessOrder            = Sequence<4, 3, 2, 0, 1>;
using BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1 =
    Sequence<CK_PARAM_BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1>;
using BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1 =
    Sequence<CK_PARAM_BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1>;
using BBlockTransferSrcVectorTensorContiguousDimOrder = Sequence<0, 1, 2, 3, 4>;

using CThreadTransferSrcDstAccessOrder              = Sequence<3, 4, 5, 0, 1, 2>;
constexpr index_t CThreadTransferSrcDstVectorDim    = 5;
constexpr index_t CThreadTransferDstScalarPerVector = CK_PARAM_CThreadTransferDstScalarPerVector;

constexpr bool HasMainKBlockLoop       = static_cast<bool>(CK_PARAM_HasMainKBlockLoop);
constexpr bool HasDoubleTailKBlockLoop = static_cast<bool>(CK_PARAM_HasDoubleTailKBlockLoop);

extern "C" __global__ void
convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw_prepare(int N_,
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

    const auto in_n_c_hi_wi_desc  = make_naive_tensor_descriptor_packed(make_tuple(N, C, Hi, Wi));
    const auto wei_k_c_y_x_desc   = make_naive_tensor_descriptor_packed(make_tuple(K, C, Y, X));
    const auto out_n_k_ho_wo_desc = make_naive_tensor_descriptor_packed(make_tuple(N, K, Ho, Wo));

    const auto descs = transform_forward_convolution_into_contraction_v6r1_nchw_kcyx_nkhw_pad(
        wei_k_c_y_x_desc,
        in_n_c_hi_wi_desc,
        out_n_k_ho_wo_desc,
        make_tuple(ConvStrideH, ConvStrideW),
        make_tuple(ConvDilationH, ConvDilationW),
        make_tuple(InLeftPadH, InLeftPadW),
        make_tuple(InRightPadH, InRightPadW),
        GN0,
        GK1);

    const auto a_grid_desc_gk0_gm0_gm1_gk1 = descs[I0];
    const auto b_grid_desc_gk0_gn0_gn1_gk1 = descs[I1];
    const auto c_grid_desc_gm0_gm1_gn0_gn1 = descs[I2];

    using AGridDesc_GK0_GM0_GM1_GK1 = decltype(a_grid_desc_gk0_gm0_gm1_gk1);
    using BGridDesc_GK0_GN0_GN1_GK1 = decltype(b_grid_desc_gk0_gn0_gn1_gk1);
    using CGridDesc_GM0_GM1_GN0_GN1 = decltype(c_grid_desc_gm0_gm1_gn0_gn1);

    using AGridStepHacks =
        decltype(make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0>{},    // 0+: GK0
                                       Sequence<0, 0, 0, 0, 0, 0, 0>{},    // 1+: GM0
                                       Sequence<0, 0, 0, 0, 0, 0, 0>{},    // 2+: GM10
                                       Sequence<0, 0, 0, 0, 0, 0, 0>{},    // 3+: GM11
                                       Sequence<0, 0, 0, 0, 0, 0, 0>{}),   // 4+: GK1
                            make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0>{},    // 0-: GK0
                                       Sequence<0, 0, 0, 0, 0, 0, 0>{},    // 1-: GM0
                                       Sequence<0, 0, 0, 0, 0, 0, 0>{},    // 2-: GM10
                                       Sequence<0, 0, 0, 0, 0, 0, 0>{},    // 3-: GM11
                                       Sequence<0, 0, 0, 0, 0, 0, 0>{}))); // 4-: GK1

    using BGridStepHacks = decltype(make_tuple(
        make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0>{},    // 0+: GK0
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0>{},    // 1+: GN0
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0>{},    // 2+: GN10
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0>{},    // 3+: GN11
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}),   // 4+: GK1
        make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0>{},    // 0-: GK0
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0>{},    // 1-: GN0
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0>{},    // 2-: GN10
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0>{},    // 3-: GN11
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}))); // 4-: GK1

    using CGridStepHacks = decltype(make_tuple(
        make_tuple(
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},  // 0+: GM10
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0>{},  // 1+: BM0
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0>{},  // 2+: BM1
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},  // 3+: GN10
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0>{},  // 4+: BN0
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0>{}), // 5+: GN1
        make_tuple(
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},    // 0-: GM10
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0>{},    // 1-: BM0
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0>{},    // 2-: BM1
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},    // 3-: GN10
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0>{},    // 4-: BN0
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0>{}))); // 5-: GN1

    using AGridMoveSliceWindowStepHacks = Sequence<0, 0, 0, 0, 0, 0, 0>;

    using BGridMoveSliceWindowStepHacks =
        Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0>;

    using GridwiseContraction =
        GridwiseContractionDlops_A_GK0_GM0_GM1_GK1_B_GK0_GN0_GN1_GK1_C_GM0_GM1_GN0_GN1<
            BlockSize,
            FloatAB,
            FloatAcc,
            FloatC,
            InMemoryDataOperationEnum_t::Set,
            AGridDesc_GK0_GM0_GM1_GK1,
            BGridDesc_GK0_GN0_GN1_GK1,
            CGridDesc_GM0_GM1_GN0_GN1,
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
            ABlockTransferThreadClusterArrangeOrder,
            ABlockTransferSrcAccessOrder,
            ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1,
            ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1,
            ABlockTransferSrcVectorTensorContiguousDimOrder,
            BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1,
            BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1,
            BBlockTransferThreadClusterArrangeOrder,
            BBlockTransferSrcAccessOrder,
            BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1,
            BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1,
            BBlockTransferSrcVectorTensorContiguousDimOrder,
            CThreadTransferSrcDstAccessOrder,
            CThreadTransferSrcDstVectorDim,
            CThreadTransferDstScalarPerVector,
            AGridStepHacks,
            BGridStepHacks,
            CGridStepHacks,
            AGridMoveSliceWindowStepHacks,
            BGridMoveSliceWindowStepHacks>;

    if(get_block_1d_id() == 0 && get_thread_local_1d_id() == 0)
    {
        auto desc_tuple =
            make_tuple(GridwiseContraction::MakeAGridDescriptor_GK0_GM0_GM10_GM11_GK1(
                           a_grid_desc_gk0_gm0_gm1_gk1),
                       GridwiseContraction::MakeBGridDescriptor_GK0_GN0_GN10_GN11_GK1(
                           b_grid_desc_gk0_gn0_gn1_gk1),
                       GridwiseContraction::MakeCGridDescriptor_GM10_BM0_BM1_GN10_BN0_BN1(
                           c_grid_desc_gm0_gm1_gn0_gn1),
                       GridwiseContraction::MakeCGridBlockCluster_BlockId_To_GM10_GN10(
                           c_grid_desc_gm0_gm1_gn0_gn1));

        *static_cast<decltype(desc_tuple)*>(p_desc_tuple) = desc_tuple;
    }
};

extern "C" __global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            const void CONSTANT* p_desc_tuple)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_n_c_hi_wi_desc =
        make_naive_tensor_descriptor_packed(make_tuple(256, 256, 28, 28));
    constexpr auto wei_k_c_y_x_desc =
        make_naive_tensor_descriptor_packed(make_tuple(256, 256, 3, 3));
    constexpr auto out_n_k_ho_wo_desc =
        make_naive_tensor_descriptor_packed(make_tuple(256, 256, 28, 28));

    constexpr auto descs =
        transform_forward_convolution_into_contraction_v6r1_nchw_kcyx_nkhw_pad(wei_k_c_y_x_desc,
                                                                               in_n_c_hi_wi_desc,
                                                                               out_n_k_ho_wo_desc,
                                                                               make_tuple(1, 1),
                                                                               make_tuple(1, 1),
                                                                               make_tuple(1, 1),
                                                                               make_tuple(1, 1),
                                                                               GN0,
                                                                               GK1);

    constexpr auto a_grid_desc_gk0_gm0_gm1_gk1 = descs[I0];
    constexpr auto b_grid_desc_gk0_gn0_gn1_gk1 = descs[I1];
    constexpr auto c_grid_desc_gm0_gm1_gn0_gn1 = descs[I2];

    using AGridDesc_GK0_GM0_GM1_GK1 = decltype(a_grid_desc_gk0_gm0_gm1_gk1);
    using BGridDesc_GK0_GN0_GN1_GK1 = decltype(b_grid_desc_gk0_gn0_gn1_gk1);
    using CGridDesc_GM0_GM1_GN0_GN1 = decltype(c_grid_desc_gm0_gm1_gn0_gn1);

    using AGridStepHacks =
        decltype(make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0>{},    // 0+: GK0
                                       Sequence<0, 0, 0, 0, 0, 0, 0>{},    // 1+: GM0
                                       Sequence<0, 0, 0, 0, 0, 0, 0>{},    // 2+: GM10
                                       Sequence<0, 0, 0, 0, 0, 0, 0>{},    // 3+: GM11
                                       Sequence<0, 0, 0, 0, 0, 0, 0>{}),   // 4+: GK1
                            make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0>{},    // 0-: GK0
                                       Sequence<0, 0, 0, 0, 0, 0, 0>{},    // 1-: GM0
                                       Sequence<0, 0, 0, 0, 0, 0, 0>{},    // 2-: GM10
                                       Sequence<0, 0, 0, 0, 0, 0, 0>{},    // 3-: GM11
                                       Sequence<0, 0, 0, 0, 0, 0, 0>{}))); // 4-: GK1

    using BGridStepHacks = decltype(make_tuple(
        make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0>{},    // 0+: GK0
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0>{},    // 1+: GN0
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0>{},    // 2+: GN10
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0>{},    // 3+: GN11
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}),   // 4+: GK1
        make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0>{},    // 0-: GK0
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0>{},    // 1-: GN0
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0>{},    // 2-: GN10
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0>{},    // 3-: GN11
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}))); // 4-: GK1

    using CGridStepHacks = decltype(make_tuple(
        make_tuple(
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},  // 0+: GM10
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0>{},  // 1+: BM0
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0>{},  // 2+: BM1
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},  // 3+: GN10
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0>{},  // 4+: BN0
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0>{}), // 5+: GN1
        make_tuple(
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},    // 0-: GM10
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0>{},    // 1-: BM0
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0>{},    // 2-: BM1
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},    // 3-: GN10
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0>{},    // 4-: BN0
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0>{}))); // 5-: GN1

    using AGridMoveSliceWindowStepHacks = Sequence<0, 0, 0, 0, 0, 0, 0>;

    using BGridMoveSliceWindowStepHacks =
        Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0>;

    using GridwiseContraction =
        GridwiseContractionDlops_A_GK0_GM0_GM1_GK1_B_GK0_GN0_GN1_GK1_C_GM0_GM1_GN0_GN1<
            BlockSize,
            FloatAB,
            FloatAcc,
            FloatC,
            InMemoryDataOperationEnum_t::Set,
            AGridDesc_GK0_GM0_GM1_GK1,
            BGridDesc_GK0_GN0_GN1_GK1,
            CGridDesc_GM0_GM1_GN0_GN1,
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
            ABlockTransferThreadClusterArrangeOrder,
            ABlockTransferSrcAccessOrder,
            ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1,
            ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1,
            ABlockTransferSrcVectorTensorContiguousDimOrder,
            BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1,
            BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1,
            BBlockTransferThreadClusterArrangeOrder,
            BBlockTransferSrcAccessOrder,
            BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1,
            BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1,
            BBlockTransferSrcVectorTensorContiguousDimOrder,
            CThreadTransferSrcDstAccessOrder,
            CThreadTransferSrcDstVectorDim,
            CThreadTransferDstScalarPerVector,
            AGridStepHacks,
            BGridStepHacks,
            CGridStepHacks,
            AGridMoveSliceWindowStepHacks,
            BGridMoveSliceWindowStepHacks>;

    using AGridDesc_GK0_GM0_GM10_GM11_GK1 =
        decltype(GridwiseContraction::MakeAGridDescriptor_GK0_GM0_GM10_GM11_GK1(
            a_grid_desc_gk0_gm0_gm1_gk1));
    using BGridDesc_GK0_GN0_GN10_GN11_GK1 =
        decltype(GridwiseContraction::MakeBGridDescriptor_GK0_GN0_GN10_GN11_GK1(
            b_grid_desc_gk0_gn0_gn1_gk1));
    using CGridDesc_GM10_BM0_BM1_GN10_BN0_BN1 =
        decltype(GridwiseContraction::MakeCGridDescriptor_GM10_BM0_BM1_GN10_BN0_BN1(
            c_grid_desc_gm0_gm1_gn0_gn1));
    using CGridBlockCluster_BlockId_To_GM10_GN10 =
        decltype(GridwiseContraction::MakeCGridBlockCluster_BlockId_To_GM10_GN10(
            c_grid_desc_gm0_gm1_gn0_gn1));

    using DescTuple = decltype(make_tuple(AGridDesc_GK0_GM0_GM10_GM11_GK1{},
                                          BGridDesc_GK0_GN0_GN10_GN11_GK1{},
                                          CGridDesc_GM10_BM0_BM1_GN10_BN0_BN1{},
                                          CGridBlockCluster_BlockId_To_GM10_GN10{}));

    const auto desc_tuple =
        *reinterpret_cast<const DescTuple*>(cast_pointer_to_generic_address_space(p_desc_tuple));

    const auto a_grid_desc_gk0_gm0_gm10_gm11_gk1         = desc_tuple[I0];
    const auto b_grid_desc_gk0_gn0_gn10_gn11_gk1         = desc_tuple[I1];
    const auto c_grid_desc_gm10_bm0_bm1_gn10_bn0_bn1     = desc_tuple[I2];
    const auto c_grid_block_cluster_blockid_to_gm10_gn10 = desc_tuple[I3];

    constexpr index_t shared_block_size =
        GridwiseContraction::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    GridwiseContraction::Run(p_a_grid,
                             p_b_grid,
                             p_c_grid,
                             p_shared_block,
                             a_grid_desc_gk0_gm0_gm10_gm11_gk1,
                             b_grid_desc_gk0_gn0_gn10_gn11_gk1,
                             c_grid_desc_gm10_bm0_bm1_gn10_bn0_bn1,
                             c_grid_block_cluster_blockid_to_gm10_gn10,
                             integral_constant<bool, HasMainKBlockLoop>{},
                             integral_constant<bool, HasDoubleTailKBlockLoop>{});
};
