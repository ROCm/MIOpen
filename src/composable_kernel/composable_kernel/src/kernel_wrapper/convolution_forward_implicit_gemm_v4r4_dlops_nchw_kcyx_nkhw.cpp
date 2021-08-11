#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm_dlops_v1r2.hpp"
#include "transform_forward_convolution_into_gemm_v4r4_nchw_kcyx_nkhw.hpp"

using namespace ck;

constexpr DataTypeEnum_t ABDataTypeEnum  = static_cast<DataTypeEnum_t>(CK_PARAM_ABDataTypeEnum);
constexpr DataTypeEnum_t AccDataTypeEnum = static_cast<DataTypeEnum_t>(CK_PARAM_AccDataTypeEnum);
constexpr DataTypeEnum_t CDataTypeEnum   = static_cast<DataTypeEnum_t>(CK_PARAM_CDataTypeEnum);

using FloatAB  = typename get_datatype_from_enum<ABDataTypeEnum>::type;
using FloatAcc = typename get_datatype_from_enum<AccDataTypeEnum>::type;
using FloatC   = typename get_datatype_from_enum<CDataTypeEnum>::type;

constexpr index_t BlockSize = CK_PARAM_BlockSize;

constexpr index_t MPerBlock            = CK_PARAM_MPerBlock;
constexpr index_t NPerBlock            = CK_PARAM_NPerBlock;
constexpr index_t KPerBlock            = CK_PARAM_KPerBlock;
constexpr index_t M1PerThread          = CK_PARAM_M1PerThread;
constexpr index_t N1PerThread          = CK_PARAM_N1PerThread;
constexpr index_t KPerThread           = CK_PARAM_KPerThread;
constexpr index_t M1N1ThreadClusterM10 = CK_PARAM_M1N1ThreadClusterM10;
constexpr index_t M1N1ThreadClusterN10 = CK_PARAM_M1N1ThreadClusterN10;
constexpr index_t M1N1ThreadClusterM11 = CK_PARAM_M1N1ThreadClusterM11;
constexpr index_t M1N1ThreadClusterN11 = CK_PARAM_M1N1ThreadClusterN11;

using ABlockTransferThreadSliceLengths_K_M0_M1 =
    Sequence<CK_PARAM_ABlockTransferThreadSliceLengths_K_M0_M1>;
using ABlockTransferThreadClusterLengths_K_M0_M1 =
    Sequence<CK_PARAM_ABlockTransferThreadClusterLengths_K_M0_M1>;
using ABlockTransferThreadClusterArrangeOrder =
    Sequence<CK_PARAM_ABlockTransferThreadClusterArrangeOrder>;
using ABlockTransferSrcAccessOrder = Sequence<CK_PARAM_ABlockTransferSrcAccessOrder>;

constexpr index_t ABlockTransferSrcVectorDim       = CK_PARAM_ABlockTransferSrcVectorDim;
constexpr index_t ABlockTransferSrcScalarPerVector = CK_PARAM_ABlockTransferSrcScalarPerVector;
constexpr index_t ABlockTransferDstScalarPerVector_M1 =
    CK_PARAM_ABlockTransferDstScalarPerVector_M1;
constexpr bool AThreadTransferSrcResetCoordinateAfterRun =
    static_cast<bool>(CK_PARAM_AThreadTransferSrcResetCoordinateAfterRun);

using BBlockTransferThreadSliceLengths_K_N0_N1 =
    Sequence<CK_PARAM_BBlockTransferThreadSliceLengths_K_N0_N1>;
using BBlockTransferThreadClusterLengths_K_N0_N1 =
    Sequence<CK_PARAM_BBlockTransferThreadClusterLengths_K_N0_N1>;
using BBlockTransferThreadClusterArrangeOrder =
    Sequence<CK_PARAM_BBlockTransferThreadClusterArrangeOrder>;
using BBlockTransferSrcAccessOrder = Sequence<CK_PARAM_BBlockTransferSrcAccessOrder>;

constexpr index_t BBlockTransferSrcVectorDim       = CK_PARAM_BBlockTransferSrcVectorDim;
constexpr index_t BBlockTransferSrcScalarPerVector = CK_PARAM_BBlockTransferSrcScalarPerVector;
constexpr index_t BBlockTransferDstScalarPerVector_N1 =
    CK_PARAM_BBlockTransferDstScalarPerVector_N1;
constexpr bool BThreadTransferSrcResetCoordinateAfterRun =
    static_cast<bool>(CK_PARAM_BThreadTransferSrcResetCoordinateAfterRun);

using CThreadTransferSrcDstAccessOrder = Sequence<CK_PARAM_CThreadTransferSrcDstAccessOrder>;
constexpr index_t CThreadTransferSrcDstVectorDim    = CK_PARAM_CThreadTransferSrcDstVectorDim;
constexpr index_t CThreadTransferDstScalarPerVector = CK_PARAM_CThreadTransferDstScalarPerVector;

constexpr bool HasMainKBlockLoop       = static_cast<bool>(CK_PARAM_HAS_MAIN_KBLOCK_LOOP);
constexpr bool HasDoubleTailKBlockLoop = static_cast<bool>(CK_PARAM_HAS_DOUBLE_TAIL_KBLOCK_LOOP);

extern "C" __global__ void convolution_forward_implicit_gemm_v4r4_dlops_nchw_kcyx_nkhw_prepare(
    int n,
    int c,
    int hi,
    int wi,
    int k,
    int y,
    int x,
    int convStrideH,
    int convStrideW,
    int convDilationY,
    int convDilationX,
    int leftPadH,
    int leftPadW,
    int rightPadH,
    int rightPadW,
    void* p_a_k_m0_m1_grid_desc,
    void* p_b_k_n0_n1_grid_desc,
    void* p_c_m0_m10_m11_n0_n10_n11_grid_desc,
    void* p_c_blockid_to_m0_n0_block_cluster_adaptor)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};

    const index_t ho = (hi + leftPadH + rightPadH - convDilationY * (y - 1) - 1) / convStrideH + 1;
    const index_t wo = (wi + leftPadW + rightPadW - convDilationX * (x - 1) - 1) / convStrideW + 1;

    const auto in_n_c_hi_wi_desc  = make_naive_tensor_descriptor_packed(make_tuple(n, c, hi, wi));
    const auto wei_k_c_y_x_desc   = make_naive_tensor_descriptor_packed(make_tuple(k, c, y, x));
    const auto out_n_k_ho_wo_desc = make_naive_tensor_descriptor_packed(make_tuple(n, k, ho, wo));

    const auto descs = transform_forward_convolution_into_gemm_v4r4_nchw_kcyx_nkhw_pad(
        wei_k_c_y_x_desc,
        in_n_c_hi_wi_desc,
        out_n_k_ho_wo_desc,
        make_tuple(convStrideH, convStrideW),
        make_tuple(convDilationY, convDilationX),
        make_tuple(leftPadH, leftPadW),
        make_tuple(rightPadH, rightPadW));

    const auto a_k_m_grid_desc = descs[I0];
    const auto b_k_n_grid_desc = descs[I1];
    const auto c_m_n_grid_desc = descs[I2];

    using AKMGridDesc = decltype(a_k_m_grid_desc);
    using BKNGridDesc = decltype(b_k_n_grid_desc);
    using CMNGridDesc = decltype(c_m_n_grid_desc);

    using AGridStepHacks = decltype(make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                                                          Sequence<0, 0, 0, 0, 0>{},
                                                          Sequence<0, 0, 0, 0, 0>{},
                                                          Sequence<0, 0, 0, 0, 0>{}),
                                               make_tuple(Sequence<0, 0, 0, 0, 0>{},
                                                          Sequence<0, 0, 0, 0, 0>{},
                                                          Sequence<0, 0, 0, 0, 0>{},
                                                          Sequence<0, 0, 0, 0, 0>{})));

    using BGridStepHacks =
        decltype(make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0>{}),
                            make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0>{})));

    using CGridStepHacks = decltype(make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
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
                                                          Sequence<0, 0, 2, 0, 0>{})));

    using AGridMoveSliceWindowStepHacks = Sequence<0, 0, 0, 0, 0>;
    using BGridMoveSliceWindowStepHacks = Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0>;

    using GridwiseGemm =
        GridwiseGemmDlops_km_kn_mn_v1r2<BlockSize,
                                        FloatAB,
                                        FloatAcc,
                                        FloatC,
                                        InMemoryDataOperationEnum_t::Set, /* ToDo tunable */
                                        AKMGridDesc,
                                        BKNGridDesc,
                                        CMNGridDesc,
                                        MPerBlock,
                                        NPerBlock,
                                        KPerBlock,
                                        M1PerThread,
                                        N1PerThread,
                                        KPerThread,
                                        M1N1ThreadClusterM10,
                                        M1N1ThreadClusterN10,
                                        M1N1ThreadClusterM11,
                                        M1N1ThreadClusterN11,
                                        ABlockTransferThreadSliceLengths_K_M0_M1,
                                        ABlockTransferThreadClusterLengths_K_M0_M1,
                                        ABlockTransferThreadClusterArrangeOrder,
                                        ABlockTransferSrcAccessOrder,
                                        ABlockTransferSrcVectorDim,
                                        ABlockTransferSrcScalarPerVector,
                                        ABlockTransferDstScalarPerVector_M1,
                                        AThreadTransferSrcResetCoordinateAfterRun,
                                        BBlockTransferThreadSliceLengths_K_N0_N1,
                                        BBlockTransferThreadClusterLengths_K_N0_N1,
                                        BBlockTransferThreadClusterArrangeOrder,
                                        BBlockTransferSrcAccessOrder,
                                        BBlockTransferSrcVectorDim,
                                        BBlockTransferSrcScalarPerVector,
                                        BBlockTransferDstScalarPerVector_N1,
                                        BThreadTransferSrcResetCoordinateAfterRun,
                                        CThreadTransferSrcDstAccessOrder,
                                        CThreadTransferSrcDstVectorDim,
                                        CThreadTransferDstScalarPerVector,
                                        AGridStepHacks,
                                        BGridStepHacks,
                                        CGridStepHacks,
                                        AGridMoveSliceWindowStepHacks,
                                        BGridMoveSliceWindowStepHacks>;

    auto a_k_m0_m1_grid_desc = GridwiseGemm::MakeAKM0M1GridDescriptor(a_k_m_grid_desc);
    auto b_k_n0_n1_grid_desc = GridwiseGemm::MakeBKN0N1GridDescriptor(b_k_n_grid_desc);
    auto c_m0_m10_m11_n0_n10_n11_grid_desc =
        GridwiseGemm::MakeCM0M10M11N0N10N11GridDescriptor(c_m_n_grid_desc);
    auto c_blockid_to_m0_n0_block_cluster_adaptor =
        GridwiseGemm::MakeCBlockIdToM0N0BlockClusterAdaptor(c_m_n_grid_desc);

    if(hipThreadIdx_x == 0)
    {
        *static_cast<decltype(a_k_m0_m1_grid_desc)*>(p_a_k_m0_m1_grid_desc) = a_k_m0_m1_grid_desc;
        *static_cast<decltype(b_k_n0_n1_grid_desc)*>(p_b_k_n0_n1_grid_desc) = b_k_n0_n1_grid_desc;
        *static_cast<decltype(c_m0_m10_m11_n0_n10_n11_grid_desc)*>(
            p_c_m0_m10_m11_n0_n10_n11_grid_desc) = c_m0_m10_m11_n0_n10_n11_grid_desc;
        *static_cast<decltype(c_blockid_to_m0_n0_block_cluster_adaptor)*>(
            p_c_blockid_to_m0_n0_block_cluster_adaptor) = c_blockid_to_m0_n0_block_cluster_adaptor;
    };
};

extern "C" __global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        convolution_forward_implicit_gemm_v4r4_dlops_nchw_kcyx_nkhw(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            const void CONSTANT* p_a_k_m0_m1_grid_desc,
            const void CONSTANT* p_b_k_n0_n1_grid_desc,
            const void CONSTANT* p_c_m0_m10_m11_n0_n10_n11_grid_desc,
            const void CONSTANT* p_c_blockid_to_m0_n0_block_cluster_adaptor)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};

    constexpr auto in_n_c_hi_wi_desc =
        make_naive_tensor_descriptor_packed(make_tuple(256, 256, 28, 28));
    constexpr auto wei_k_c_y_x_desc =
        make_naive_tensor_descriptor_packed(make_tuple(256, 256, 3, 3));
    constexpr auto out_n_k_ho_wo_desc =
        make_naive_tensor_descriptor_packed(make_tuple(256, 256, 28, 28));

    constexpr auto descs =
        transform_forward_convolution_into_gemm_v4r4_nchw_kcyx_nkhw_pad(wei_k_c_y_x_desc,
                                                                        in_n_c_hi_wi_desc,
                                                                        out_n_k_ho_wo_desc,
                                                                        make_tuple(1, 1),
                                                                        make_tuple(1, 1),
                                                                        make_tuple(1, 1),
                                                                        make_tuple(1, 1));

    constexpr auto a_k_m_grid_desc = descs[I0];
    constexpr auto b_k_n_grid_desc = descs[I1];
    constexpr auto c_m_n_grid_desc = descs[I2];

    using AKMGridDesc = decltype(a_k_m_grid_desc);
    using BKNGridDesc = decltype(b_k_n_grid_desc);
    using CMNGridDesc = decltype(c_m_n_grid_desc);

    using AGridStepHacks = decltype(make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                                                          Sequence<0, 0, 0, 0, 0>{},
                                                          Sequence<0, 0, 0, 0, 0>{},
                                                          Sequence<0, 0, 0, 0, 0>{}),
                                               make_tuple(Sequence<0, 0, 0, 0, 0>{},
                                                          Sequence<0, 0, 0, 0, 0>{},
                                                          Sequence<0, 0, 0, 0, 0>{},
                                                          Sequence<0, 0, 0, 0, 0>{})));

    using BGridStepHacks =
        decltype(make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0>{}),
                            make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0>{})));

    using CGridStepHacks = decltype(make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
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
                                                          Sequence<0, 0, 2, 0, 0>{})));

    using AGridMoveSliceWindowStepHacks = Sequence<0, 0, 0, 0, 0>;
    using BGridMoveSliceWindowStepHacks = Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0>;

    using GridwiseGemm =
        GridwiseGemmDlops_km_kn_mn_v1r2<BlockSize,
                                        FloatAB,
                                        FloatAcc,
                                        FloatC,
                                        InMemoryDataOperationEnum_t::Set, /* ToDo tunable */
                                        AKMGridDesc,
                                        BKNGridDesc,
                                        CMNGridDesc,
                                        MPerBlock,
                                        NPerBlock,
                                        KPerBlock,
                                        M1PerThread,
                                        N1PerThread,
                                        KPerThread,
                                        M1N1ThreadClusterM10,
                                        M1N1ThreadClusterN10,
                                        M1N1ThreadClusterM11,
                                        M1N1ThreadClusterN11,
                                        ABlockTransferThreadSliceLengths_K_M0_M1,
                                        ABlockTransferThreadClusterLengths_K_M0_M1,
                                        ABlockTransferThreadClusterArrangeOrder,
                                        ABlockTransferSrcAccessOrder,
                                        ABlockTransferSrcVectorDim,
                                        ABlockTransferSrcScalarPerVector,
                                        ABlockTransferDstScalarPerVector_M1,
                                        AThreadTransferSrcResetCoordinateAfterRun,
                                        BBlockTransferThreadSliceLengths_K_N0_N1,
                                        BBlockTransferThreadClusterLengths_K_N0_N1,
                                        BBlockTransferThreadClusterArrangeOrder,
                                        BBlockTransferSrcAccessOrder,
                                        BBlockTransferSrcVectorDim,
                                        BBlockTransferSrcScalarPerVector,
                                        BBlockTransferDstScalarPerVector_N1,
                                        BThreadTransferSrcResetCoordinateAfterRun,
                                        CThreadTransferSrcDstAccessOrder,
                                        CThreadTransferSrcDstVectorDim,
                                        CThreadTransferDstScalarPerVector,
                                        AGridStepHacks,
                                        BGridStepHacks,
                                        CGridStepHacks,
                                        AGridMoveSliceWindowStepHacks,
                                        BGridMoveSliceWindowStepHacks>;

    constexpr auto a_k_m0_m1_grid_desc_tmp =
        GridwiseGemm::MakeAKM0M1GridDescriptor(a_k_m_grid_desc);
    constexpr auto b_k_n0_n1_grid_desc_tmp =
        GridwiseGemm::MakeBKN0N1GridDescriptor(b_k_n_grid_desc);
    constexpr auto c_m0_m10_m11_n0_n10_n11_grid_desc_tmp =
        GridwiseGemm::MakeCM0M10M11N0N10N11GridDescriptor(c_m_n_grid_desc);
    constexpr auto c_blockid_to_m0_n0_block_cluster_adaptor_tmp =
        GridwiseGemm::MakeCBlockIdToM0N0BlockClusterAdaptor(c_m_n_grid_desc);

    using AKM0M1GridDesc            = decltype(a_k_m0_m1_grid_desc_tmp);
    using BKN0N1GridDesc            = decltype(b_k_n0_n1_grid_desc_tmp);
    using CM0M10M11N0N10N11GridDesc = decltype(c_m0_m10_m11_n0_n10_n11_grid_desc_tmp);
    using CBlockIdToM0N0BlockClusterAdaptor =
        decltype(c_blockid_to_m0_n0_block_cluster_adaptor_tmp);

    const auto a_k_m0_m1_grid_desc =
        *reinterpret_cast<const AKM0M1GridDesc*>((const void*)p_a_k_m0_m1_grid_desc);
    const auto b_k_n0_n1_grid_desc =
        *reinterpret_cast<const BKN0N1GridDesc*>((const void*)p_b_k_n0_n1_grid_desc);
    const auto c_m0_m10_m11_n0_n10_n11_grid_desc =
        *reinterpret_cast<const CM0M10M11N0N10N11GridDesc*>(
            (const void*)p_c_m0_m10_m11_n0_n10_n11_grid_desc);
    const auto c_blockid_to_m0_n0_block_cluster_adaptor =
        *reinterpret_cast<const CBlockIdToM0N0BlockClusterAdaptor*>(
            (const void*)p_c_blockid_to_m0_n0_block_cluster_adaptor);

    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    GridwiseGemm::Run(p_a_grid,
                      p_b_grid,
                      p_c_grid,
                      p_shared_block,
                      a_k_m0_m1_grid_desc,
                      b_k_n0_n1_grid_desc,
                      c_m0_m10_m11_n0_n10_n11_grid_desc,
                      c_blockid_to_m0_n0_block_cluster_adaptor,
                      integral_constant<bool, HasMainKBlockLoop>{},
                      integral_constant<bool, HasDoubleTailKBlockLoop>{});
};
