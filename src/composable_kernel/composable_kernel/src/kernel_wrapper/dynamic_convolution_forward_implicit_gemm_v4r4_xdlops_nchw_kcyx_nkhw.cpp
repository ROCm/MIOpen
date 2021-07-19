#include "common_header.hpp"
#include "type_helper.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "gridwise_dynamic_gemm_xdlops_v2r3.hpp"
#include "transform_forward_convolution_into_gemm_v4r4r2_nchw_kcyx_nkhw.hpp"

using namespace ck;

using FloatAB  = typename get_type_from_type_id<static_cast<char>(CK_PARAM_IN_WEI_DATATYPE)>::type;
using FloatC   = typename get_type_from_type_id<static_cast<char>(CK_PARAM_OUT_DATATYPE)>::type;
using FloatAcc = typename get_type_from_type_id<static_cast<char>(CK_PARAM_CONV_COMPTYPE)>::type;

constexpr index_t BlockSize = CK_PARAM_BlockSize;

constexpr index_t MPerBlock = CK_PARAM_MPerBlock;
constexpr index_t NPerBlock = CK_PARAM_NPerBlock;
constexpr index_t KPerBlock = CK_PARAM_KPerBlock;

constexpr index_t MPerWave = CK_PARAM_MPerWave;
constexpr index_t NPerWave = CK_PARAM_NPerWave;
constexpr index_t MRepeat  = CK_PARAM_MRepeat;
constexpr index_t NRepeat  = CK_PARAM_NRepeat;
constexpr index_t K1       = CK_PARAM_K1;

using ABlockTransferThreadSliceLengths_K0_M_K1 =
    Sequence<CK_PARAM_ABlockTransferThreadSliceLengths_K0_M_K1>;
using ABlockTransferThreadClusterLengths_K0_M_K1 =
    Sequence<CK_PARAM_ABlockTransferThreadClusterLengths_K0_M_K1>;
using ABlockTransferThreadClusterArrangeOrder =
    Sequence<CK_PARAM_ABlockTransferThreadClusterArrangeOrder>;
using ABlockTransferSrcAccessOrder = Sequence<CK_PARAM_ABlockTransferSrcAccessOrder>;

constexpr index_t ABlockTransferSrcVectorDim       = CK_PARAM_ABlockTransferSrcVectorDim;
constexpr index_t ABlockTransferSrcScalarPerVector = CK_PARAM_ABlockTransferSrcScalarPerVector;
constexpr index_t ABlockTransferDstScalarPerVector_K1 =
    CK_PARAM_ABlockTransferDstScalarPerVector_K1;
constexpr bool AThreadTransferSrcResetCoordinateAfterRun =
    static_cast<bool>(CK_PARAM_AThreadTransferSrcResetCoordinateAfterRun);

using BBlockTransferThreadSliceLengths_K0_N_K1 =
    Sequence<CK_PARAM_BBlockTransferThreadSliceLengths_K0_N_K1>;
using BBlockTransferThreadClusterLengths_K0_N_K1 =
    Sequence<CK_PARAM_BBlockTransferThreadClusterLengths_K0_N_K1>;
using BBlockTransferThreadClusterArrangeOrder =
    Sequence<CK_PARAM_BBlockTransferThreadClusterArrangeOrder>;
using BBlockTransferSrcAccessOrder = Sequence<CK_PARAM_BBlockTransferSrcAccessOrder>;

constexpr index_t BBlockTransferSrcVectorDim       = CK_PARAM_BBlockTransferSrcVectorDim;
constexpr index_t BBlockTransferSrcScalarPerVector = CK_PARAM_BBlockTransferSrcScalarPerVector;
constexpr index_t BBlockTransferDstScalarPerVector_K1 =
    CK_PARAM_BBlockTransferDstScalarPerVector_K1;
constexpr bool BThreadTransferSrcResetCoordinateAfterRun =
    static_cast<bool>(CK_PARAM_BThreadTransferSrcResetCoordinateAfterRun);

using CThreadTransferSrcDstAccessOrder = Sequence<CK_PARAM_CThreadTransferSrcDstAccessOrder>;
constexpr index_t CThreadTransferSrcDstVectorDim    = CK_PARAM_CThreadTransferSrcDstVectorDim;
constexpr index_t CThreadTransferDstScalarPerVector = CK_PARAM_CThreadTransferDstScalarPerVector;

extern "C" __global__ void
dynamic_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw_prepare(
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
    void* p_a_k0_m_k1_grid_desc,
    void* p_b_k0_n_k1_grid_desc,
    void* p_c_m0_m1_m2_n_grid_desc,
    void* p_c_blockid_to_m0_n0_block_cluster_adaptor)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};

    const index_t ho = (hi + leftPadH + rightPadH - convDilationY * (y - 1) - 1) / convStrideH + 1;
    const index_t wo = (wi + leftPadW + rightPadW - convDilationX * (x - 1) - 1) / convStrideW + 1;

    const auto in_n_c_hi_wi_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(n, c, hi, wi));
    const auto wei_k_c_y_x_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(k, c, y, x));
    const auto out_n_k_ho_wo_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(n, k, ho, wo));

    const auto descs = transform_forward_convolution_into_gemm_v4r4r2_nchw_kcyx_nkhw_pad(
        wei_k_c_y_x_desc,
        in_n_c_hi_wi_desc,
        out_n_k_ho_wo_desc,
        make_tuple(convStrideH, convStrideW),
        make_tuple(convDilationY, convDilationX),
        make_tuple(leftPadH, leftPadW),
        make_tuple(rightPadH, rightPadW),
        Number<K1>{});

    const auto a_k0_m_k1_grid_desc = descs[I0];
    const auto b_k0_n_k1_grid_desc = descs[I1];
    const auto c_m_n_grid_desc     = descs[I2];

    using AK0MK1GridDesc = decltype(a_k0_m_k1_grid_desc);
    using BK0NK1GridDesc = decltype(b_k0_n_k1_grid_desc);
    using CMNGridDesc    = decltype(c_m_n_grid_desc);

    using AGridIteratorHacks = decltype(make_tuple(
        make_tuple(Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}),
        make_tuple(
            Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{})));

    using BGridIteratorHacks =
        decltype(make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{}),
                            make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{})));

    using CGridIteratorHacks = decltype(make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
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
                                                              Sequence<0, 0, 2, 0, 0>{})));

    using AGridMoveSliceWindowIteratorHacks = Sequence<0, 0, 0, 0, 0>;
    using BGridMoveSliceWindowIteratorHacks = Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0>;

    using GridwiseGemm =
        GridwiseDynamicGemm_k0mk1_k0nk1_mn_xdlops_v2r3<BlockSize,
                                                       FloatAB,
                                                       FloatAcc,
                                                       FloatC,
                                                       InMemoryDataOperation::Set,
                                                       AK0MK1GridDesc,
                                                       BK0NK1GridDesc,
                                                       CMNGridDesc,
                                                       MPerBlock,
                                                       NPerBlock,
                                                       KPerBlock,
                                                       MPerWave,
                                                       NPerWave,
                                                       K1,
                                                       MRepeat,
                                                       NRepeat,
                                                       ABlockTransferThreadSliceLengths_K0_M_K1,
                                                       ABlockTransferThreadClusterLengths_K0_M_K1,
                                                       ABlockTransferThreadClusterArrangeOrder,
                                                       ABlockTransferSrcAccessOrder,
                                                       ABlockTransferSrcVectorDim,
                                                       ABlockTransferSrcScalarPerVector,
                                                       ABlockTransferDstScalarPerVector_K1,
                                                       AThreadTransferSrcResetCoordinateAfterRun,
                                                       BBlockTransferThreadSliceLengths_K0_N_K1,
                                                       BBlockTransferThreadClusterLengths_K0_N_K1,
                                                       BBlockTransferThreadClusterArrangeOrder,
                                                       BBlockTransferSrcAccessOrder,
                                                       BBlockTransferSrcVectorDim,
                                                       BBlockTransferSrcScalarPerVector,
                                                       BBlockTransferDstScalarPerVector_K1,
                                                       BThreadTransferSrcResetCoordinateAfterRun,
                                                       CThreadTransferSrcDstAccessOrder,
                                                       CThreadTransferSrcDstVectorDim,
                                                       CThreadTransferDstScalarPerVector,
                                                       AGridIteratorHacks,
                                                       BGridIteratorHacks,
                                                       CGridIteratorHacks,
                                                       AGridMoveSliceWindowIteratorHacks,
                                                       BGridMoveSliceWindowIteratorHacks,
                                                       false>;

    auto c_m0_m1_m2_n_grid_desc = GridwiseGemm::MakeCM0M1M2NGridDescriptor(c_m_n_grid_desc);

    auto c_blockid_to_m0_n0_block_cluster_adaptor =
        GridwiseGemm::MakeCBlockClusterAdaptor(c_m_n_grid_desc);

    if(hipThreadIdx_x == 0)
    {
        *static_cast<remove_cv_t<decltype(a_k0_m_k1_grid_desc)>*>(p_a_k0_m_k1_grid_desc) =
            a_k0_m_k1_grid_desc;
        *static_cast<remove_cv_t<decltype(b_k0_n_k1_grid_desc)>*>(p_b_k0_n_k1_grid_desc) =
            b_k0_n_k1_grid_desc;
        *static_cast<decltype(c_m0_m1_m2_n_grid_desc)*>(p_c_m0_m1_m2_n_grid_desc) =
            c_m0_m1_m2_n_grid_desc;
        *static_cast<decltype(c_blockid_to_m0_n0_block_cluster_adaptor)*>(
            p_c_blockid_to_m0_n0_block_cluster_adaptor) = c_blockid_to_m0_n0_block_cluster_adaptor;
    }
};

extern "C" __global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        dynamic_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            const void __CONSTANT__* p_a_k0_m_k1_grid_desc,
            const void __CONSTANT__* p_b_k0_n_k1_grid_desc,
            const void __CONSTANT__* p_c_m0_m1_m2_n_grid_desc,
            const void __CONSTANT__* p_c_blockid_to_m0_n0_block_cluster_adaptor)
{

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};

    constexpr auto in_n_c_hi_wi_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(256, 256, 28, 28));
    constexpr auto wei_k_c_y_x_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(256, 256, 3, 3));
    constexpr auto out_n_k_ho_wo_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(256, 256, 28, 28));

    constexpr auto descs =
        transform_forward_convolution_into_gemm_v4r4r2_nchw_kcyx_nkhw_pad(wei_k_c_y_x_desc,
                                                                          in_n_c_hi_wi_desc,
                                                                          out_n_k_ho_wo_desc,
                                                                          make_tuple(1, 1),
                                                                          make_tuple(1, 1),
                                                                          make_tuple(1, 1),
                                                                          make_tuple(1, 1),
                                                                          Number<K1>{});

    constexpr auto a_k0_m_k1_grid_desc_tmp = descs[I0];
    constexpr auto b_k0_n_k1_grid_desc_tmp = descs[I1];
    constexpr auto c_m_n_grid_desc         = descs[I2];

    using AGridIteratorHacks = decltype(make_tuple(
        make_tuple(Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}),
        make_tuple(
            Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{})));

    using BGridIteratorHacks =
        decltype(make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{}),
                            make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{})));

    using CGridIteratorHacks = decltype(make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
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
                                                              Sequence<0, 0, 2, 0, 0>{})));

    using AGridMoveSliceWindowIteratorHacks = Sequence<0, 0, 0, 0, 0>;
    using BGridMoveSliceWindowIteratorHacks = Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0>;

    using AK0MK1GridDesc = decltype(a_k0_m_k1_grid_desc_tmp);
    using BK0NK1GridDesc = decltype(b_k0_n_k1_grid_desc_tmp);
    using CMNGridDesc    = decltype(c_m_n_grid_desc);

    using GridwiseGemm =
        GridwiseDynamicGemm_k0mk1_k0nk1_mn_xdlops_v2r3<BlockSize,
                                                       FloatAB,
                                                       FloatAcc,
                                                       FloatC,
                                                       InMemoryDataOperation::Set,
                                                       AK0MK1GridDesc,
                                                       BK0NK1GridDesc,
                                                       CMNGridDesc,
                                                       MPerBlock,
                                                       NPerBlock,
                                                       KPerBlock,
                                                       MPerWave,
                                                       NPerWave,
                                                       K1,
                                                       MRepeat,
                                                       NRepeat,
                                                       ABlockTransferThreadSliceLengths_K0_M_K1,
                                                       ABlockTransferThreadClusterLengths_K0_M_K1,
                                                       ABlockTransferThreadClusterArrangeOrder,
                                                       ABlockTransferSrcAccessOrder,
                                                       ABlockTransferSrcVectorDim,
                                                       ABlockTransferSrcScalarPerVector,
                                                       ABlockTransferDstScalarPerVector_K1,
                                                       AThreadTransferSrcResetCoordinateAfterRun,
                                                       BBlockTransferThreadSliceLengths_K0_N_K1,
                                                       BBlockTransferThreadClusterLengths_K0_N_K1,
                                                       BBlockTransferThreadClusterArrangeOrder,
                                                       BBlockTransferSrcAccessOrder,
                                                       BBlockTransferSrcVectorDim,
                                                       BBlockTransferSrcScalarPerVector,
                                                       BBlockTransferDstScalarPerVector_K1,
                                                       BThreadTransferSrcResetCoordinateAfterRun,
                                                       CThreadTransferSrcDstAccessOrder,
                                                       CThreadTransferSrcDstVectorDim,
                                                       CThreadTransferDstScalarPerVector,
                                                       AGridIteratorHacks,
                                                       BGridIteratorHacks,
                                                       CGridIteratorHacks,
                                                       AGridMoveSliceWindowIteratorHacks,
                                                       BGridMoveSliceWindowIteratorHacks,
                                                       false>;

    constexpr auto c_m0_m1_m2_n_grid_desc_tmp =
        GridwiseGemm::MakeCM0M1M2NGridDescriptor(c_m_n_grid_desc);
    constexpr auto c_blockid_to_m0_n0_block_cluster_adaptor_tmp =
        GridwiseGemm::MakeCBlockClusterAdaptor(c_m_n_grid_desc);

    using CM0M1M2NGridDesc = decltype(c_m0_m1_m2_n_grid_desc_tmp);
    using CBlockIdToM0N0BlockClusterAdaptor =
        decltype(c_blockid_to_m0_n0_block_cluster_adaptor_tmp);

    const auto a_k0_m_k1_grid_desc =
        *reinterpret_cast<const AK0MK1GridDesc*>((const void*)p_a_k0_m_k1_grid_desc);
    const auto b_k0_n_k1_grid_desc =
        *reinterpret_cast<const BK0NK1GridDesc*>((const void*)p_b_k0_n_k1_grid_desc);
    const auto c_m0_m1_m2_n_grid_desc =
        *reinterpret_cast<const CM0M1M2NGridDesc*>((const void*)p_c_m0_m1_m2_n_grid_desc);
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
                      a_k0_m_k1_grid_desc,
                      b_k0_n_k1_grid_desc,
                      c_m0_m1_m2_n_grid_desc,
                      c_blockid_to_m0_n0_block_cluster_adaptor);
};
