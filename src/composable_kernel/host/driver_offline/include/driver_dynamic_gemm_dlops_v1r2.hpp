#ifndef DRIVER_DYNAMIC_GEMM_DLOPS_V1R2
#define DRIVER_DYNAMIC_GEMM_DLOPS_V1R2

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "gridwise_dynamic_gemm_dlops_v1r2.hpp"

template <ck::index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          ck::InMemoryDataOperationEnum_t CGlobalMemoryDataOperation,
          typename AKMGridDesc,
          typename BKNGridDesc,
          typename CMNGridDesc,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t M1PerThread,
          ck::index_t N1PerThread,
          ck::index_t KPerThread,
          ck::index_t M1N1ThreadClusterM10,
          ck::index_t M1N1ThreadClusterN10,
          ck::index_t M1N1ThreadClusterM11,
          ck::index_t M1N1ThreadClusterN11,
          typename ABlockTransferThreadSliceLengths_K_M0_M1,
          typename ABlockTransferThreadClusterLengths_K_M0_M1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_M1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          typename BBlockTransferThreadSliceLengths_K_N0_N1,
          typename BBlockTransferThreadClusterLengths_K_N0_N1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_N1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          typename CThreadTransferSrcDstAccessOrder,
          ck::index_t CThreadTransferSrcDstVectorDim,
          ck::index_t CThreadTransferDstScalarPerVector,
          typename AGridIteratorHacks,
          typename BGridIteratorHacks,
          typename CGridIteratorHacks,
          typename AGridMoveSliceWindowIteratorHacks,
          typename BGridMoveSliceWindowIteratorHacks>
__host__ float driver_dynamic_gemm_dlops_v1r2(const FloatAB* p_a_grid,
                                              const FloatAB* p_b_grid,
                                              FloatC* p_c_grid,
                                              const AKMGridDesc& a_k_m_grid_desc,
                                              const BKNGridDesc& b_k_n_grid_desc,
                                              const CMNGridDesc& c_m_n_grid_desc,
                                              AGridIteratorHacks,
                                              BGridIteratorHacks,
                                              CGridIteratorHacks,
                                              AGridMoveSliceWindowIteratorHacks,
                                              BGridMoveSliceWindowIteratorHacks,
                                              ck::index_t nrepeat)

{
    using namespace ck;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};
    constexpr auto I5 = Number<5>{};

    // GEMM
    using GridwiseGemm =
        GridwiseDynamicGemmDlops_km_kn_mn_v1r2<BlockSize,
                                               FloatAB,
                                               FloatAcc,
                                               FloatC,
                                               CGlobalMemoryDataOperation,
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
                                               AGridIteratorHacks,
                                               BGridIteratorHacks,
                                               CGridIteratorHacks,
                                               AGridMoveSliceWindowIteratorHacks,
                                               BGridMoveSliceWindowIteratorHacks>;

    const auto M = a_k_m_grid_desc.GetLength(I1);
    const auto N = b_k_n_grid_desc.GetLength(I1);
    const auto K = a_k_m_grid_desc.GetLength(I0);

    if(!GridwiseGemm::CheckValidity(a_k_m_grid_desc, b_k_n_grid_desc, c_m_n_grid_desc))
    {
        throw std::runtime_error(
            "wrong! GridwiseDynamicGemmDlops_km_kn_mn_v1r2 has invalid setting");
    }

    const auto a_k_m0_m1_grid_desc = GridwiseGemm::MakeAKM0M1GridDescriptor(a_k_m_grid_desc);
    const auto b_k_n0_n1_grid_desc = GridwiseGemm::MakeBKN0N1GridDescriptor(b_k_n_grid_desc);

    using AKM0M1GridDesc = decltype(a_k_m0_m1_grid_desc);
    using BKN0N1GridDesc = decltype(b_k_n0_n1_grid_desc);

    // c_m0_m10_m11_n0_n10_n11_grid_desc
    const auto c_m0_m10_m11_n0_n10_n11_grid_desc =
        GridwiseGemm::MakeCM0M10M11N0N10N11GridDescriptor(c_m_n_grid_desc);

    using CM0M10M11N0N10N11GridDesc = decltype(c_m0_m10_m11_n0_n10_n11_grid_desc);

    // c_blockid_to_m0_n0_block_cluster_adaptor
    const auto c_blockid_to_m0_n0_block_cluster_adaptor =
        GridwiseGemm::MakeCBlockIdToM0N0BlockClusterAdaptor(c_m_n_grid_desc);

    using CBlockIdToM0N0BlockClusterAdaptor = decltype(c_blockid_to_m0_n0_block_cluster_adaptor);

    const index_t grid_size = GridwiseGemm::CalculateGridSize(M, N);

    const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K);

    const bool has_double_tail_k_block_loop = GridwiseGemm::CalculateHasDoubleTailKBlockLoop(K);

    {
        std::cout << "a_k_m0_m1_grid_desc{" << a_k_m0_m1_grid_desc.GetLength(I0) << ", "
                  << a_k_m0_m1_grid_desc.GetLength(I1) << ", " << a_k_m0_m1_grid_desc.GetLength(I2)
                  << "}" << std::endl;

        std::cout << "b_k_n0_n1_grid_desc{" << b_k_n0_n1_grid_desc.GetLength(I0) << ", "
                  << b_k_n0_n1_grid_desc.GetLength(I1) << ", " << b_k_n0_n1_grid_desc.GetLength(I2)
                  << "}" << std::endl;

        std::cout << "c_m0_m10_m11_n0_n10_n11_grid_desc{ "
                  << c_m0_m10_m11_n0_n10_n11_grid_desc.GetLength(I0) << ", "
                  << c_m0_m10_m11_n0_n10_n11_grid_desc.GetLength(I1) << ", "
                  << c_m0_m10_m11_n0_n10_n11_grid_desc.GetLength(I2) << ", "
                  << c_m0_m10_m11_n0_n10_n11_grid_desc.GetLength(I3) << ", "
                  << c_m0_m10_m11_n0_n10_n11_grid_desc.GetLength(I4) << ", "
                  << c_m0_m10_m11_n0_n10_n11_grid_desc.GetLength(I5) << "}" << std::endl;
    }

#if CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VALUE
    float ave_time = 0;

    if(has_main_k_block_loop && has_double_tail_k_block_loop)
    {
        const auto kernel =
            kernel_dynamic_gemm_dlops_v1r2<GridwiseGemm,
                                           FloatAB,
                                           FloatC,
                                           remove_reference_t<AKM0M1GridDesc>,
                                           remove_reference_t<BKN0N1GridDesc>,
                                           remove_reference_t<CM0M10M11N0N10N11GridDesc>,
                                           remove_reference_t<CBlockIdToM0N0BlockClusterAdaptor>,
                                           true,
                                           true>;

        ave_time = launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(grid_size),
                                          dim3(BlockSize),
                                          0,
                                          0,
                                          p_a_grid,
                                          p_b_grid,
                                          p_c_grid,
                                          a_k_m0_m1_grid_desc,
                                          b_k_n0_n1_grid_desc,
                                          c_m0_m10_m11_n0_n10_n11_grid_desc,
                                          c_blockid_to_m0_n0_block_cluster_adaptor);
    }
    else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
    {
        const auto kernel =
            kernel_dynamic_gemm_dlops_v1r2<GridwiseGemm,
                                           FloatAB,
                                           FloatC,
                                           remove_reference_t<AKM0M1GridDesc>,
                                           remove_reference_t<BKN0N1GridDesc>,
                                           remove_reference_t<CM0M10M11N0N10N11GridDesc>,
                                           remove_reference_t<CBlockIdToM0N0BlockClusterAdaptor>,
                                           true,
                                           false>;

        ave_time = launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(grid_size),
                                          dim3(BlockSize),
                                          0,
                                          0,
                                          p_a_grid,
                                          p_b_grid,
                                          p_c_grid,
                                          a_k_m0_m1_grid_desc,
                                          b_k_n0_n1_grid_desc,
                                          c_m0_m10_m11_n0_n10_n11_grid_desc,
                                          c_blockid_to_m0_n0_block_cluster_adaptor);
    }
    else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
    {
        const auto kernel =
            kernel_dynamic_gemm_dlops_v1r2<GridwiseGemm,
                                           FloatAB,
                                           FloatC,
                                           remove_reference_t<AKM0M1GridDesc>,
                                           remove_reference_t<BKN0N1GridDesc>,
                                           remove_reference_t<CM0M10M11N0N10N11GridDesc>,
                                           remove_reference_t<CBlockIdToM0N0BlockClusterAdaptor>,
                                           false,
                                           true>;

        ave_time = launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(grid_size),
                                          dim3(BlockSize),
                                          0,
                                          0,
                                          p_a_grid,
                                          p_b_grid,
                                          p_c_grid,
                                          a_k_m0_m1_grid_desc,
                                          b_k_n0_n1_grid_desc,
                                          c_m0_m10_m11_n0_n10_n11_grid_desc,
                                          c_blockid_to_m0_n0_block_cluster_adaptor);
    }
    else
    {
        const auto kernel =
            kernel_dynamic_gemm_dlops_v1r2<GridwiseGemm,
                                           FloatAB,
                                           FloatC,
                                           remove_reference_t<AKM0M1GridDesc>,
                                           remove_reference_t<BKN0N1GridDesc>,
                                           remove_reference_t<CM0M10M11N0N10N11GridDesc>,
                                           remove_reference_t<CBlockIdToM0N0BlockClusterAdaptor>,
                                           false,
                                           false>;

        ave_time = launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(grid_size),
                                          dim3(BlockSize),
                                          0,
                                          0,
                                          p_a_grid,
                                          p_b_grid,
                                          p_c_grid,
                                          a_k_m0_m1_grid_desc,
                                          b_k_n0_n1_grid_desc,
                                          c_m0_m10_m11_n0_n10_n11_grid_desc,
                                          c_blockid_to_m0_n0_block_cluster_adaptor);
    }

    return ave_time;
#elif CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VOID_POINTER
    DeviceMem a_k_m0_m1_grid_desc_dev_buf(sizeof(AKM0M1GridDesc));
    DeviceMem b_k_n0_n1_grid_desc_dev_buf(sizeof(BKN0N1GridDesc));
    DeviceMem c_m0_m10_m11_n0_n10_n11_grid_desc_dev_buf(sizeof(CM0M10M11N0N10N11GridDesc));
    DeviceMem c_blockid_to_m0_n0_block_cluster_adaptor_dev_buf(
        sizeof(CBlockIdToM0N0BlockClusterAdaptor));

    a_k_m0_m1_grid_desc_dev_buf.ToDevice(&a_k_m0_m1_grid_desc);
    b_k_n0_n1_grid_desc_dev_buf.ToDevice(&b_k_n0_n1_grid_desc);
    c_m0_m10_m11_n0_n10_n11_grid_desc_dev_buf.ToDevice(&c_m0_m10_m11_n0_n10_n11_grid_desc);
    c_blockid_to_m0_n0_block_cluster_adaptor_dev_buf.ToDevice(
        &c_blockid_to_m0_n0_block_cluster_adaptor);

    float ave_time = 0;

    if(has_main_k_block_loop && has_double_tail_k_block_loop)
    {
        const auto kernel =
            kernel_dynamic_gemm_dlops_v1r2<GridwiseGemm,
                                           FloatAB,
                                           FloatC,
                                           remove_reference_t<AKM0M1GridDesc>,
                                           remove_reference_t<BKN0N1GridDesc>,
                                           remove_reference_t<CM0M10M11N0N10N11GridDesc>,
                                           remove_reference_t<CBlockIdToM0N0BlockClusterAdaptor>,
                                           true,
                                           true>;

        ave_time = launch_and_time_kernel(
            kernel,
            nrepeat,
            dim3(grid_size),
            dim3(BlockSize),
            0,
            0,
            p_a_grid,
            p_b_grid,
            p_c_grid,
            (void CONSTANT*)a_k_m0_m1_grid_desc_dev_buf.GetDeviceBuffer(),
            (void CONSTANT*)b_k_n0_n1_grid_desc_dev_buf.GetDeviceBuffer(),
            (void CONSTANT*)c_m0_m10_m11_n0_n10_n11_grid_desc_dev_buf.GetDeviceBuffer(),
            (void CONSTANT*)c_blockid_to_m0_n0_block_cluster_adaptor_dev_buf.GetDeviceBuffer());
    }
    else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
    {
        const auto kernel =
            kernel_dynamic_gemm_dlops_v1r2<GridwiseGemm,
                                           FloatAB,
                                           FloatC,
                                           remove_reference_t<AKM0M1GridDesc>,
                                           remove_reference_t<BKN0N1GridDesc>,
                                           remove_reference_t<CM0M10M11N0N10N11GridDesc>,
                                           remove_reference_t<CBlockIdToM0N0BlockClusterAdaptor>,
                                           true,
                                           false>;

        ave_time = launch_and_time_kernel(
            kernel,
            nrepeat,
            dim3(grid_size),
            dim3(BlockSize),
            0,
            0,
            p_a_grid,
            p_b_grid,
            p_c_grid,
            (void CONSTANT*)a_k_m0_m1_grid_desc_dev_buf.GetDeviceBuffer(),
            (void CONSTANT*)b_k_n0_n1_grid_desc_dev_buf.GetDeviceBuffer(),
            (void CONSTANT*)c_m0_m10_m11_n0_n10_n11_grid_desc_dev_buf.GetDeviceBuffer(),
            (void CONSTANT*)c_blockid_to_m0_n0_block_cluster_adaptor_dev_buf.GetDeviceBuffer());
    }
    else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
    {
        const auto kernel =
            kernel_dynamic_gemm_dlops_v1r2<GridwiseGemm,
                                           FloatAB,
                                           FloatC,
                                           remove_reference_t<AKM0M1GridDesc>,
                                           remove_reference_t<BKN0N1GridDesc>,
                                           remove_reference_t<CM0M10M11N0N10N11GridDesc>,
                                           remove_reference_t<CBlockIdToM0N0BlockClusterAdaptor>,
                                           false,
                                           true>;

        ave_time = launch_and_time_kernel(
            kernel,
            nrepeat,
            dim3(grid_size),
            dim3(BlockSize),
            0,
            0,
            p_a_grid,
            p_b_grid,
            p_c_grid,
            (void CONSTANT*)a_k_m0_m1_grid_desc_dev_buf.GetDeviceBuffer(),
            (void CONSTANT*)b_k_n0_n1_grid_desc_dev_buf.GetDeviceBuffer(),
            (void CONSTANT*)c_m0_m10_m11_n0_n10_n11_grid_desc_dev_buf.GetDeviceBuffer(),
            (void CONSTANT*)c_blockid_to_m0_n0_block_cluster_adaptor_dev_buf.GetDeviceBuffer());
    }
    else
    {
        const auto kernel =
            kernel_dynamic_gemm_dlops_v1r2<GridwiseGemm,
                                           FloatAB,
                                           FloatC,
                                           remove_reference_t<AKM0M1GridDesc>,
                                           remove_reference_t<BKN0N1GridDesc>,
                                           remove_reference_t<CM0M10M11N0N10N11GridDesc>,
                                           remove_reference_t<CBlockIdToM0N0BlockClusterAdaptor>,
                                           false,
                                           false>;

        ave_time = launch_and_time_kernel(
            kernel,
            nrepeat,
            dim3(grid_size),
            dim3(BlockSize),
            0,
            0,
            p_a_grid,
            p_b_grid,
            p_c_grid,
            (void CONSTANT*)a_k_m0_m1_grid_desc_dev_buf.GetDeviceBuffer(),
            (void CONSTANT*)b_k_n0_n1_grid_desc_dev_buf.GetDeviceBuffer(),
            (void CONSTANT*)c_m0_m10_m11_n0_n10_n11_grid_desc_dev_buf.GetDeviceBuffer(),
            (void CONSTANT*)c_blockid_to_m0_n0_block_cluster_adaptor_dev_buf.GetDeviceBuffer());
    }

    return ave_time;
#endif
}
#endif
