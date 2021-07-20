#ifndef CK_DRIVER_DYNAMIC_GEMM_DLOPS_V1R3
#define CK_DRIVER_DYNAMIC_GEMM_DLOPS_V1R3

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "gridwise_dynamic_gemm_dlops_v1r3.hpp"

namespace ck {

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          InMemoryDataOperation CGlobalMemoryDataOperation,
          typename AK0MK1GridDesc,
          typename BK0NK1GridDesc,
          typename CMNGridDesc,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t M1PerThread,
          index_t N1PerThread,
          index_t KPerThread,
          index_t M1N1ThreadClusterM10,
          index_t M1N1ThreadClusterN10,
          index_t M1N1ThreadClusterM11,
          index_t M1N1ThreadClusterN11,
          typename ABlockTransferThreadSliceLengths_K0_M0_M1_K1,
          typename ABlockTransferThreadClusterLengths_K0_M0_M1_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          typename ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
          typename ABlockTransferSrcVectorTensorContiguousDimOrder,
          typename ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1,
          typename BBlockTransferThreadSliceLengths_K0_N0_N1_K1,
          typename BBlockTransferThreadClusterLengths_K0_N0_N1_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          typename BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1,
          typename BBlockTransferSrcVectorTensorContiguousDimOrder,
          typename BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector,
          typename AGridIteratorHacks,
          typename BGridIteratorHacks,
          typename CGridIteratorHacks,
          typename AGridMoveSliceWindowIteratorHacks,
          typename BGridMoveSliceWindowIteratorHacks>
__host__ float driver_dynamic_gemm_dlops_v1r3(const FloatAB* p_a_grid,
                                              const FloatAB* p_b_grid,
                                              FloatC* p_c_grid,
                                              const AK0MK1GridDesc& a_k0_m_k1_grid_desc,
                                              const BK0NK1GridDesc& b_k0_n_k1_grid_desc,
                                              const CMNGridDesc& c_m_n_grid_desc,
                                              AGridIteratorHacks,
                                              BGridIteratorHacks,
                                              CGridIteratorHacks,
                                              AGridMoveSliceWindowIteratorHacks,
                                              BGridMoveSliceWindowIteratorHacks,
                                              index_t nrepeat)

{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};
    constexpr auto I5 = Number<5>{};

    // GEMM
    using GridwiseGemm =
        GridwiseDynamicGemmDlops_km_kn_mn_v1r3<BlockSize,
                                               FloatAB,
                                               FloatAcc,
                                               FloatC,
                                               CGlobalMemoryDataOperation,
                                               AK0MK1GridDesc,
                                               BK0NK1GridDesc,
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
                                               ABlockTransferThreadSliceLengths_K0_M0_M1_K1,
                                               ABlockTransferThreadClusterLengths_K0_M0_M1_K1,
                                               ABlockTransferThreadClusterArrangeOrder,
                                               ABlockTransferSrcAccessOrder,
                                               ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
                                               ABlockTransferSrcVectorTensorContiguousDimOrder,
                                               ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1,
                                               BBlockTransferThreadSliceLengths_K0_N0_N1_K1,
                                               BBlockTransferThreadClusterLengths_K0_N0_N1_K1,
                                               BBlockTransferThreadClusterArrangeOrder,
                                               BBlockTransferSrcAccessOrder,
                                               BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1,
                                               BBlockTransferSrcVectorTensorContiguousDimOrder,
                                               BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1,
                                               CThreadTransferSrcDstAccessOrder,
                                               CThreadTransferSrcDstVectorDim,
                                               CThreadTransferDstScalarPerVector,
                                               AGridIteratorHacks,
                                               BGridIteratorHacks,
                                               CGridIteratorHacks,
                                               AGridMoveSliceWindowIteratorHacks,
                                               BGridMoveSliceWindowIteratorHacks>;

    const auto M  = a_k0_m_k1_grid_desc.GetLength(I1);
    const auto N  = b_k0_n_k1_grid_desc.GetLength(I1);
    const auto K0 = a_k0_m_k1_grid_desc.GetLength(I0);

    if(!GridwiseGemm::CheckValidity(a_k0_m_k1_grid_desc, b_k0_n_k1_grid_desc, c_m_n_grid_desc))
    {
        throw std::runtime_error(
            "wrong! GridwiseDynamicGemmDlops_km_kn_mn_v1r3 has invalid setting");
    }

    const auto a_k0_m0_m1_k1_grid_desc =
        GridwiseGemm::MakeAK0M0M1K1GridDescriptor(a_k0_m_k1_grid_desc);
    const auto b_k0_n0_n1_k1_grid_desc =
        GridwiseGemm::MakeBK0N0N1K1GridDescriptor(b_k0_n_k1_grid_desc);

    using AK0M0M1K1GridDesc = decltype(a_k0_m0_m1_k1_grid_desc);
    using BK0N0N1K1GridDesc = decltype(b_k0_n0_n1_k1_grid_desc);

    // c_m0_m10_m11_n0_n10_n11_grid_desc
    const auto c_m0_m10_m11_n0_n10_n11_grid_desc =
        GridwiseGemm::MakeCM0M10M11N0N10N11GridDescriptor(c_m_n_grid_desc);

    using CM0M10M11N0N10N11GridDesc = decltype(c_m0_m10_m11_n0_n10_n11_grid_desc);

    // c_blockid_to_m0_n0_block_cluster_adaptor
    const auto c_blockid_to_m0_n0_block_cluster_adaptor =
        GridwiseGemm::MakeCBlockIdToM0N0BlockClusterAdaptor(c_m_n_grid_desc);

    using CBlockIdToM0N0BlockClusterAdaptor = decltype(c_blockid_to_m0_n0_block_cluster_adaptor);

    const index_t grid_size = GridwiseGemm::CalculateGridSize(M, N);

    const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K0);

    const bool has_double_tail_k_block_loop = GridwiseGemm::CalculateHasDoubleTailKBlockLoop(K0);

    {
        std::cout << "a_k0_m0_m1_k1_grid_desc{" << a_k0_m0_m1_k1_grid_desc.GetLength(I0) << ", "
                  << a_k0_m0_m1_k1_grid_desc.GetLength(I1) << ", "
                  << a_k0_m0_m1_k1_grid_desc.GetLength(I2) << ", "
                  << a_k0_m0_m1_k1_grid_desc.GetLength(I3) << "}" << std::endl;

        std::cout << "b_k0_n0_n1_k1_grid_desc{" << b_k0_n0_n1_k1_grid_desc.GetLength(I0) << ", "
                  << b_k0_n0_n1_k1_grid_desc.GetLength(I1) << ", "
                  << b_k0_n0_n1_k1_grid_desc.GetLength(I2) << ", "
                  << b_k0_n0_n1_k1_grid_desc.GetLength(I3) << "}" << std::endl;

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
            kernel_dynamic_gemm_dlops_v1r3<GridwiseGemm,
                                           FloatAB,
                                           FloatC,
                                           remove_reference_t<AK0M0M1K1GridDesc>,
                                           remove_reference_t<BK0N0N1K1GridDesc>,
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
                                          a_k0_m0_m1_k1_grid_desc,
                                          b_k0_n0_n1_k1_grid_desc,
                                          c_m0_m10_m11_n0_n10_n11_grid_desc,
                                          c_blockid_to_m0_n0_block_cluster_adaptor);
    }
    else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
    {
        const auto kernel =
            kernel_dynamic_gemm_dlops_v1r3<GridwiseGemm,
                                           FloatAB,
                                           FloatC,
                                           remove_reference_t<AK0M0M1K1GridDesc>,
                                           remove_reference_t<BK0N0N1K1GridDesc>,
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
                                          a_k0_m0_m1_k1_grid_desc,
                                          b_k0_n0_n1_k1_grid_desc,
                                          c_m0_m10_m11_n0_n10_n11_grid_desc,
                                          c_blockid_to_m0_n0_block_cluster_adaptor);
    }
    else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
    {
        const auto kernel =
            kernel_dynamic_gemm_dlops_v1r3<GridwiseGemm,
                                           FloatAB,
                                           FloatC,
                                           remove_reference_t<AK0M0M1K1GridDesc>,
                                           remove_reference_t<BK0N0N1K1GridDesc>,
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
                                          a_k0_m0_m1_k1_grid_desc,
                                          b_k0_n0_n1_k1_grid_desc,
                                          c_m0_m10_m11_n0_n10_n11_grid_desc,
                                          c_blockid_to_m0_n0_block_cluster_adaptor);
    }
    else
    {
        const auto kernel =
            kernel_dynamic_gemm_dlops_v1r3<GridwiseGemm,
                                           FloatAB,
                                           FloatC,
                                           remove_reference_t<AK0M0M1K1GridDesc>,
                                           remove_reference_t<BK0N0N1K1GridDesc>,
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
                                          a_k0_m0_m1_k1_grid_desc,
                                          b_k0_n0_n1_k1_grid_desc,
                                          c_m0_m10_m11_n0_n10_n11_grid_desc,
                                          c_blockid_to_m0_n0_block_cluster_adaptor);
    }

    return ave_time;
#elif CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VOID_POINTER
    DeviceMem a_k0_m0_m1_k1_grid_desc_dev_buf(sizeof(AK0M0M1K1GridDesc));
    DeviceMem b_k0_n0_n1_k1_grid_desc_dev_buf(sizeof(BK0N0N1K1GridDesc));
    DeviceMem c_m0_m10_m11_n0_n10_n11_grid_desc_dev_buf(sizeof(CM0M10M11N0N10N11GridDesc));
    DeviceMem c_blockid_to_m0_n0_block_cluster_adaptor_dev_buf(
        sizeof(CBlockIdToM0N0BlockClusterAdaptor));

    a_k0_m0_m1_k1_grid_desc_dev_buf.ToDevice(&a_k0_m0_m1_k1_grid_desc);
    b_k0_n0_n1_k1_grid_desc_dev_buf.ToDevice(&b_k0_n0_n1_k1_grid_desc);
    c_m0_m10_m11_n0_n10_n11_grid_desc_dev_buf.ToDevice(&c_m0_m10_m11_n0_n10_n11_grid_desc);
    c_blockid_to_m0_n0_block_cluster_adaptor_dev_buf.ToDevice(
        &c_blockid_to_m0_n0_block_cluster_adaptor);

    float ave_time = 0;

    if(has_main_k_block_loop && has_double_tail_k_block_loop)
    {
        const auto kernel =
            kernel_dynamic_gemm_dlops_v1r3<GridwiseGemm,
                                           FloatAB,
                                           FloatC,
                                           remove_reference_t<AK0M0M1K1GridDesc>,
                                           remove_reference_t<BK0N0N1K1GridDesc>,
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
            (void __CONSTANT__*)a_k0_m0_m1_k1_grid_desc_dev_buf.GetDeviceBuffer(),
            (void __CONSTANT__*)b_k0_n0_n1_k1_grid_desc_dev_buf.GetDeviceBuffer(),
            (void __CONSTANT__*)c_m0_m10_m11_n0_n10_n11_grid_desc_dev_buf.GetDeviceBuffer(),
            (void __CONSTANT__*)c_blockid_to_m0_n0_block_cluster_adaptor_dev_buf.GetDeviceBuffer());
    }
    else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
    {
        const auto kernel =
            kernel_dynamic_gemm_dlops_v1r3<GridwiseGemm,
                                           FloatAB,
                                           FloatC,
                                           remove_reference_t<AK0M0M1K1GridDesc>,
                                           remove_reference_t<BK0N0N1K1GridDesc>,
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
            (void __CONSTANT__*)a_k0_m0_m1_k1_grid_desc_dev_buf.GetDeviceBuffer(),
            (void __CONSTANT__*)b_k0_n0_n1_k1_grid_desc_dev_buf.GetDeviceBuffer(),
            (void __CONSTANT__*)c_m0_m10_m11_n0_n10_n11_grid_desc_dev_buf.GetDeviceBuffer(),
            (void __CONSTANT__*)c_blockid_to_m0_n0_block_cluster_adaptor_dev_buf.GetDeviceBuffer());
    }
    else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
    {
        const auto kernel =
            kernel_dynamic_gemm_dlops_v1r3<GridwiseGemm,
                                           FloatAB,
                                           FloatC,
                                           remove_reference_t<AK0M0M1K1GridDesc>,
                                           remove_reference_t<BK0N0N1K1GridDesc>,
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
            (void __CONSTANT__*)a_k0_m0_m1_k1_grid_desc_dev_buf.GetDeviceBuffer(),
            (void __CONSTANT__*)b_k0_n0_n1_k1_grid_desc_dev_buf.GetDeviceBuffer(),
            (void __CONSTANT__*)c_m0_m10_m11_n0_n10_n11_grid_desc_dev_buf.GetDeviceBuffer(),
            (void __CONSTANT__*)c_blockid_to_m0_n0_block_cluster_adaptor_dev_buf.GetDeviceBuffer());
    }
    else
    {
        const auto kernel =
            kernel_dynamic_gemm_dlops_v1r3<GridwiseGemm,
                                           FloatAB,
                                           FloatC,
                                           remove_reference_t<AK0M0M1K1GridDesc>,
                                           remove_reference_t<BK0N0N1K1GridDesc>,
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
            (void __CONSTANT__*)a_k0_m0_m1_k1_grid_desc_dev_buf.GetDeviceBuffer(),
            (void __CONSTANT__*)b_k0_n0_n1_k1_grid_desc_dev_buf.GetDeviceBuffer(),
            (void __CONSTANT__*)c_m0_m10_m11_n0_n10_n11_grid_desc_dev_buf.GetDeviceBuffer(),
            (void __CONSTANT__*)c_blockid_to_m0_n0_block_cluster_adaptor_dev_buf.GetDeviceBuffer());
    }

    return ave_time;
#endif
}

} // namespace ck
#endif
