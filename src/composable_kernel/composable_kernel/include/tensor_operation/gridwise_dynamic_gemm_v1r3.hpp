#ifndef CK_GRIDWISE_DYNAMIC_GEMM_V1R3_HPP
#define CK_GRIDWISE_DYNAMIC_GEMM_V1R3_HPP

#include "common_header.hpp"
#include "dynamic_multi_index_transform_helper.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "blockwise_gemm_v2r3.hpp"
#include "blockwise_dynamic_tensor_slice_transfer_v2.hpp"
#include "threadwise_dynamic_tensor_slice_transfer_v2.hpp"
#include "threadwise_dynamic_tensor_slice_set.hpp"

namespace ck {

#if CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VALUE
template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename AK0M0M1K1GridDesc,
          typename BK0N0N1K1GridDesc,
          typename CM0M10M11N0N10N11GridDesc,
          typename CBlockIdToM0N0BlockClusterAdaptor,
          bool HasMainKBlockLoop,
          bool HasDoubleTailKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_dynamic_gemm_v1r3(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            const AK0M0M1K1GridDesc a_k0_m0_m1_k1_grid_desc,
            const BK0N0N1K1GridDesc b_k0_n0_n1_k1_grid_desc,
            const CM0M10M11N0N10N11GridDesc c_m0_m10_m11_n0_n10_n11_grid_desc,
            const CBlockIdToM0N0BlockClusterAdaptor c_blockid_to_m0_n0_block_cluster_adaptor)
{
    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    GridwiseGemm::Run(p_a_grid,
                      p_b_grid,
                      p_c_grid,
                      p_shared_block,
                      a_k0_m0_m1_k1_grid_desc,
                      b_k0_n0_n1_k1_grid_desc,
                      c_m0_m10_m11_n0_n10_n11_grid_desc,
                      c_blockid_to_m0_n0_block_cluster_adaptor,
                      integral_constant<bool, HasMainKBlockLoop>{},
                      integral_constant<bool, HasDoubleTailKBlockLoop>{});
}
#elif CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VOID_POINTER
// pass tensor descriptor by __CONSTANT__ void pointer
// __CONSTANT__ is needed to inform compiler void pointers in the kernel signature are pointing to
// non-modifiable parameter address space, so compiler can enable corresponding optimization
template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename AK0M0M1K1GridDesc,
          typename BK0N0N1K1GridDesc,
          typename CM0M10M11N0N10N11GridDesc,
          typename CBlockIdToM0N0BlockClusterAdaptor,
          bool HasMainKBlockLoop,
          bool HasDoubleTailKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_dynamic_gemm_v1r3(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            const void __CONSTANT__* p_a_k0_m0_m1_k1_grid_desc,
            const void __CONSTANT__* p_b_k0_n0_n1_k1_grid_desc,
            const void __CONSTANT__* p_c_m0_m10_m11_n0_n10_n11_grid_desc,
            const void __CONSTANT__* p_c_blockid_to_m0_n0_block_cluster_adaptor)
{
    // first cast void __CONSTANT__ void* to void*
    // second cast void* to Desc*
    // the copy constructor of tensor descriptor doesn't take address_space(4)
    const auto a_k0_m0_m1_k1_grid_desc =
        *reinterpret_cast<const AK0M0M1K1GridDesc*>((const void*)p_a_k0_m0_m1_k1_grid_desc);
    const auto b_k0_n0_n1_k1_grid_desc =
        *reinterpret_cast<const BK0N0N1K1GridDesc*>((const void*)p_b_k0_n0_n1_k1_grid_desc);
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
                      a_k0_m0_m1_k1_grid_desc,
                      b_k0_n0_n1_k1_grid_desc,
                      c_m0_m10_m11_n0_n10_n11_grid_desc,
                      c_blockid_to_m0_n0_block_cluster_adaptor,
                      integral_constant<bool, HasMainKBlockLoop>{},
                      integral_constant<bool, HasDoubleTailKBlockLoop>{});
}
#endif

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          InMemoryDataOperation CGlobalMemoryDataOperation,
          typename AK0MK1GridDesc,
          typename BK0NK1GridDesc,
          typename CMNGridDesc,
          index_t MPerBlockM1,
          index_t NPerBlockN1,
          index_t KPerBlock,
          index_t M1PerThreadM111,
          index_t N1PerThreadN111,
          index_t KPerThread,
          index_t M11N11ThreadClusterM1100,
          index_t M11N11ThreadClusterN1100,
          index_t M11N11ThreadClusterM1101,
          index_t M11N11ThreadClusterN1101,
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
struct GridwiseDynamicGemm_km_kn_mn_v1r3
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    // K1 should be Number<...>
    static constexpr auto K1 = AK0MK1GridDesc{}.GetLength(I2);

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        // TODO: change this. I think it needs multi-dimensional alignment
        constexpr auto max_lds_align = K1;

        // TODO: check alignment
        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_k_m_block_desc = make_dynamic_naive_tensor_descriptor_aligned_v2(
            make_tuple(Number<KPerBlock>{}, Number<MPerBlockM1>{}, K1), max_lds_align);

        // TODO: check alignment
        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_k_n_block_desc = make_dynamic_naive_tensor_descriptor_aligned_v2(
            make_tuple(Number<KPerBlock>{}, Number<NPerBlockN1>{}, K1), max_lds_align);

        // TODO: check alignment
        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_aligned_space_size =
            math::integer_least_multiple(a_k_m_block_desc.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_aligned_space_size =
            math::integer_least_multiple(b_k_n_block_desc.GetElementSpaceSize(), max_lds_align);

        return 2 * (a_block_aligned_space_size + b_block_aligned_space_size) * sizeof(FloatAB);
    }

    __host__ __device__ static constexpr bool
    CheckValidity(const AK0MK1GridDesc& a_k0_m_k1_grid_desc,
                  const BK0NK1GridDesc& b_k0_n_k1_grid_desc,
                  const CMNGridDesc& c_m_n_grid_desc)
    {
        const auto M  = a_k0_m_k1_grid_desc.GetLength(I1);
        const auto N  = b_k0_n_k1_grid_desc.GetLength(I1);
        const auto K0 = a_k0_m_k1_grid_desc.GetLength(I0);
        const auto K1 = a_k0_m_k1_grid_desc.GetLength(I2);

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)

        return (M == c_m_n_grid_desc.GetLength(I0) && N == c_m_n_grid_desc.GetLength(I1) &&
                K0 == b_k0_n_k1_grid_desc.GetLength(I0) &&
                K1 == b_k0_n_k1_grid_desc.GetLength(I2)) &&
               (M % MPerBlockM1 == 0 && N % NPerBlockN1 == 0 && K0 % KPerBlock == 0);
    }

    __host__ __device__ static constexpr index_t CalculateGridSize(index_t M, index_t N)
    {
        const index_t grid_size = (M / MPerBlockM1) * (N / NPerBlockN1);

        return grid_size;
    }

    __host__ __device__ static constexpr bool CalculateHasMainKBlockLoop(index_t K0)
    {
        const bool has_main_k_block_loop = (K0 + KPerBlock) / (2 * KPerBlock) > 1;

        return has_main_k_block_loop;
    }

    __host__ __device__ static constexpr bool CalculateHasDoubleTailKBlockLoop(index_t K0)
    {
        const bool has_double_tail_k_block_loop = (K0 / KPerBlock) % 2 == 0;

        return has_double_tail_k_block_loop;
    }

    __host__ __device__ static constexpr auto
    MakeAK0M0M1K1GridDescriptor(const AK0MK1GridDesc& a_k0_m_k1_grid_desc)
    {
        const auto K0 = a_k0_m_k1_grid_desc.GetLength(I0);
        const auto M  = a_k0_m_k1_grid_desc.GetLength(I1);

        const auto M1 = Number<MPerBlockM1>{};
        const auto M0 = M / M1;

        const auto a_k0_m0_m1_k1_grid_desc = transform_dynamic_tensor_descriptor(
            a_k0_m_k1_grid_desc,
            make_tuple(make_pass_through_transform(K0),
                       make_unmerge_transform(make_tuple(M0, M1)),
                       make_pass_through_transform(K1)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

        return a_k0_m0_m1_k1_grid_desc;
    }

    __host__ __device__ static constexpr auto
    MakeBK0N0N1K1GridDescriptor(const BK0NK1GridDesc& b_k0_n_k1_grid_desc)
    {
        const auto K0 = b_k0_n_k1_grid_desc.GetLength(I0);
        const auto N  = b_k0_n_k1_grid_desc.GetLength(I1);

        const auto N1 = Number<NPerBlockN1>{};
        const auto N0 = N / N1;

        const auto b_k0_n0_n1_k1_grid_desc = transform_dynamic_tensor_descriptor(
            b_k0_n_k1_grid_desc,
            make_tuple(make_pass_through_transform(K0),
                       make_unmerge_transform(make_tuple(N0, N1)),
                       make_pass_through_transform(K1)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

        return b_k0_n0_n1_k1_grid_desc;
    }

    __host__ __device__ static constexpr auto
    MakeCM0M10M11N0N10N11GridDescriptor(const CMNGridDesc& c_m_n_grid_desc)
    {
        const auto M = c_m_n_grid_desc.GetLength(I0);
        const auto N = c_m_n_grid_desc.GetLength(I1);

        constexpr auto M1 = Number<MPerBlockM1>{};
        constexpr auto N1 = Number<NPerBlockN1>{};

        const auto M0 = M / M1;
        const auto N0 = N / N1;

        constexpr auto M11 =
            Number<M11N11ThreadClusterM1100 * M11N11ThreadClusterM1101 * M1PerThreadM111>{};
        constexpr auto N11 =
            Number<M11N11ThreadClusterN1100 * M11N11ThreadClusterN1101 * N1PerThreadN111>{};

        constexpr auto M10 = M1 / M11;
        constexpr auto N10 = N1 / N11;

        const auto c_m0_m10_m11_n0_n10_n11_grid_desc = transform_dynamic_tensor_descriptor(
            c_m_n_grid_desc,
            make_tuple(make_unmerge_transform(make_tuple(M0, M10, M11)),
                       make_unmerge_transform(make_tuple(N0, N10, N11))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1, 2>{}, Sequence<3, 4, 5>{}));

        return c_m0_m10_m11_n0_n10_n11_grid_desc;
    }

    __host__ __device__ static constexpr auto
    MakeCBlockIdToM0N0BlockClusterAdaptor(const CMNGridDesc& c_m_n_grid_desc)
    {
        const auto M = c_m_n_grid_desc.GetLength(I0);
        const auto N = c_m_n_grid_desc.GetLength(I1);

        constexpr auto M1 = Number<MPerBlockM1>{};
        constexpr auto N1 = Number<NPerBlockN1>{};

        const auto M0 = M / M1;
        const auto N0 = N / N1;

        const auto c_blockid_to_m0_n0_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(make_tuple(make_merge_transform(make_tuple(M0, N0))),
                                             make_tuple(Sequence<0, 1>{}),
                                             make_tuple(Sequence<0>{}));

        return c_blockid_to_m0_n0_block_cluster_adaptor;
    }

    using AK0M0M1K1GridDesc         = decltype(MakeAK0M0M1K1GridDescriptor(AK0MK1GridDesc{}));
    using BK0N0N1K1GridDesc         = decltype(MakeBK0N0N1K1GridDescriptor(BK0NK1GridDesc{}));
    using CM0M10M11N0N10N11GridDesc = decltype(MakeCM0M10M11N0N10N11GridDescriptor(CMNGridDesc{}));
    using CBlockIdToM0N0BlockClusterAdaptor =
        decltype(MakeCBlockIdToM0N0BlockClusterAdaptor(CMNGridDesc{}));

    template <bool HasMainKBlockLoop, bool HasDoubleTailKBlockLoop>
    __device__ static void
    Run(const FloatAB* __restrict__ p_a_grid,
        const FloatAB* __restrict__ p_b_grid,
        FloatC* __restrict__ p_c_grid,
        FloatAB* __restrict__ p_shared_block,
        const AK0M0M1K1GridDesc& a_k0_m0_m1_k1_grid_desc,
        const BK0N0N1K1GridDesc& b_k0_n0_n1_k1_grid_desc,
        const CM0M10M11N0N10N11GridDesc& c_m0_m10_m11_n0_n10_n11_grid_desc,
        const CBlockIdToM0N0BlockClusterAdaptor& c_blockid_to_m0_n0_block_cluster_adaptor,
        integral_constant<bool, HasMainKBlockLoop>,
        integral_constant<bool, HasDoubleTailKBlockLoop>)
    {
        const auto a_global_buf = make_dynamic_buffer<AddressSpace::Global>(
            p_a_grid, a_k0_m0_m1_k1_grid_desc.GetElementSpaceSize());
        const auto b_global_buf = make_dynamic_buffer<AddressSpace::Global>(
            p_b_grid, b_k0_n0_n1_k1_grid_desc.GetElementSpaceSize());
        auto c_grid_buf = make_dynamic_buffer<AddressSpace::Global>(
            p_c_grid, c_m0_m10_m11_n0_n10_n11_grid_desc.GetElementSpaceSize());

        // divide block work by [M, N]
        const auto c_m0_n0_block_cluster_idx =
            c_blockid_to_m0_n0_block_cluster_adaptor.CalculateBottomIndex(
                make_multi_index(get_block_1d_id()));

        // HACK: this force index data into SGPR
        const index_t im0 = __builtin_amdgcn_readfirstlane(c_m0_n0_block_cluster_idx[I0]);
        const index_t in0 = __builtin_amdgcn_readfirstlane(c_m0_n0_block_cluster_idx[I1]);

        // TODO: change this. I think it needs multi-dimensional alignment
        constexpr auto max_lds_align = K1;

        // TODO: check alignment
        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_k0_m0_m1_k1_block_desc = make_dynamic_naive_tensor_descriptor_aligned_v2(
            make_tuple(Number<KPerBlock>{}, I1, Number<MPerBlockM1>{}, K1), max_lds_align);

        // TODO: check alignment
        // B matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto b_k0_n0_n1_k1_block_desc = make_dynamic_naive_tensor_descriptor_aligned_v2(
            make_tuple(Number<KPerBlock>{}, I1, Number<NPerBlockN1>{}, K1), max_lds_align);

        // TODO: check alignment
        // A matrix in LDS memory, for blockwise GEMM
        constexpr auto a_k0_m_k1_block_desc = make_dynamic_naive_tensor_descriptor_aligned_v2(
            make_tuple(Number<KPerBlock>{}, Number<MPerBlockM1>{}, K1), max_lds_align);

        // TODO: check alignment
        // B matrix in LDS memory, for blockwise GEMM
        constexpr auto b_k0_n_k1_block_desc = make_dynamic_naive_tensor_descriptor_aligned_v2(
            make_tuple(Number<KPerBlock>{}, Number<NPerBlockN1>{}, K1), max_lds_align);

        static_assert(a_k0_m0_m1_k1_block_desc.GetElementSpaceSize() ==
                          a_k0_m_k1_block_desc.GetElementSpaceSize() &&
                      b_k0_n0_n1_k1_block_desc.GetElementSpaceSize() ==
                          b_k0_n_k1_block_desc.GetElementSpaceSize() &&
                      "wrong!");

        // A matrix blockwise copy
        auto a_blockwise_copy = BlockwiseDynamicTensorSliceTransfer_v4r1<
            BlockSize,
            InMemoryDataOperation::Set,
            Sequence<KPerBlock, 1, MPerBlockM1, K1.value>,
            ABlockTransferThreadSliceLengths_K0_M0_M1_K1,
            ABlockTransferThreadClusterLengths_K0_M0_M1_K1,
            ABlockTransferThreadClusterArrangeOrder,
            FloatAB,
            FloatAB,
            decltype(a_k0_m0_m1_k1_grid_desc),
            decltype(a_k0_m0_m1_k1_block_desc),
            ABlockTransferSrcAccessOrder,
            Sequence<0, 1, 2, 3>,
            ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1, // SrcVectorTensorLengths
            ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1, // DstVectorTensorLengths
            ABlockTransferSrcVectorTensorContiguousDimOrder,  // SrcVectorTensorContiguousDimOrder
            Sequence<0, 1, 2, 3>,                             // DstVectorTensorContiguousDimOrder
            false,
            true>(a_k0_m0_m1_k1_grid_desc,
                  make_multi_index(0, im0, 0, 0),
                  a_k0_m0_m1_k1_block_desc,
                  make_multi_index(0, 0, 0, 0));

        // B matrix blockwise copy
        auto b_blockwise_copy = BlockwiseDynamicTensorSliceTransfer_v4r1<
            BlockSize,
            InMemoryDataOperation::Set,
            Sequence<KPerBlock, 1, NPerBlockN1, K1.value>,
            BBlockTransferThreadSliceLengths_K0_N0_N1_K1,
            BBlockTransferThreadClusterLengths_K0_N0_N1_K1,
            BBlockTransferThreadClusterArrangeOrder,
            FloatAB,
            FloatAB,
            decltype(b_k0_n0_n1_k1_grid_desc),
            decltype(b_k0_n0_n1_k1_block_desc),
            BBlockTransferSrcAccessOrder,
            Sequence<0, 1, 2, 3>,
            BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1, // SrcVectorTensorLengths
            BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1, // DstVectorTensorLengths
            BBlockTransferSrcVectorTensorContiguousDimOrder,  // SrcVectorTensorContiguousDimOrder
            Sequence<0, 1, 2, 3>,                             // DstVectorTensorContiguousDimOrder
            false,
            true>(b_k0_n0_n1_k1_grid_desc,
                  make_multi_index(0, in0, 0, 0),
                  b_k0_n0_n1_k1_block_desc,
                  make_multi_index(0, 0, 0, 0));

        // GEMM definition
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[KPerBlock, MPerBlockM1] is in LDS
        //     b_mtx[KPerBlocl, NPerBlockN1] is in LDS
        //     c_mtx[MPerBlockM1, NPerBlockN1] is distributed among threads, and saved in
        //       register
        const auto blockwise_gemm =
            BlockwiseGemm_A_BK0_BM_BK1_B_BK0_BN_BK1_C_BM0_BM1_BN0_BN1_pipeline_BM0_2_BN0_2<
                BlockSize,
                FloatAB,
                FloatAB,
                FloatAcc,
                decltype(a_k0_m_k1_block_desc),
                decltype(b_k0_n_k1_block_desc),
                M1PerThreadM111,
                N1PerThreadN111,
                KPerThread,
                M11N11ThreadClusterM1100,
                M11N11ThreadClusterN1100,
                M11N11ThreadClusterM1101,
                M11N11ThreadClusterN1101,
                M1PerThreadM111,
                N1PerThreadN111>{};

        constexpr auto c_m10_m11_n10_n11_thread_tensor_lengths =
            decltype(blockwise_gemm)::GetCM0M1N0N1ThreadTensorLengths();

        constexpr auto c_m10_m11_n10_n11_thread_desc =
            make_dynamic_naive_tensor_descriptor_packed_v2(
                sequence_to_tuple_of_number(c_m10_m11_n10_n11_thread_tensor_lengths));

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_aligned_space_size = math::integer_least_multiple(
            a_k0_m0_m1_k1_block_desc.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_aligned_space_size = math::integer_least_multiple(
            b_k0_n0_n1_k1_block_desc.GetElementSpaceSize(), max_lds_align);

        FloatAB* p_a_block_double = p_shared_block;
        FloatAB* p_b_block_double = p_shared_block + 2 * a_block_aligned_space_size;

        // register allocation for output
        auto c_thread_buf = make_static_buffer<AddressSpace::Vgpr, FloatAcc>(
            c_m10_m11_n10_n11_thread_desc.GetElementSpaceSize());

        ThreadwiseDynamicTensorSliceSet_v1<FloatAcc,
                                           decltype(c_m10_m11_n10_n11_thread_desc),
                                           decltype(c_m10_m11_n10_n11_thread_tensor_lengths)>{}
            .Run(c_m10_m11_n10_n11_thread_desc,
                 make_tuple(I0, I0, I0, I0),
                 c_thread_buf,
                 FloatAcc{0});

        constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock, 0, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(KPerBlock, 0, 0, 0);

        auto a_block_even_buf = make_dynamic_buffer<AddressSpace::Lds>(
            p_a_block_double, a_k0_m0_m1_k1_block_desc.GetElementSpaceSize());
        auto b_block_even_buf = make_dynamic_buffer<AddressSpace::Lds>(
            p_b_block_double, b_k0_n0_n1_k1_block_desc.GetElementSpaceSize());

        auto a_block_odd_buf =
            make_dynamic_buffer<AddressSpace::Lds>(p_a_block_double + a_block_aligned_space_size,
                                                   a_k0_m0_m1_k1_block_desc.GetElementSpaceSize());
        auto b_block_odd_buf =
            make_dynamic_buffer<AddressSpace::Lds>(p_b_block_double + b_block_aligned_space_size,
                                                   b_k0_n0_n1_k1_block_desc.GetElementSpaceSize());

        // LDS double buffer: preload data into LDS
        {
            a_blockwise_copy.RunRead(a_k0_m0_m1_k1_grid_desc, a_global_buf, AGridIteratorHacks{});
            b_blockwise_copy.RunRead(b_k0_n0_n1_k1_grid_desc, b_global_buf, BGridIteratorHacks{});

            a_blockwise_copy.RunWrite(a_k0_m0_m1_k1_block_desc, a_block_even_buf);
            b_blockwise_copy.RunWrite(b_k0_n0_n1_k1_block_desc, b_block_even_buf);
        }

        if constexpr(HasMainKBlockLoop)
        {
            const auto K0 = a_k0_m0_m1_k1_grid_desc.GetLength(I0);

            index_t k_block_data_begin = 0;

            // LDS double buffer: main body
            // use Do-While loop instead of For loop to simplify control flow
            do
            {
                // even iteration
                a_blockwise_copy.MoveSrcSliceWindow(a_k0_m0_m1_k1_grid_desc,
                                                    a_block_slice_copy_step,
                                                    AGridMoveSliceWindowIteratorHacks{});
                b_blockwise_copy.MoveSrcSliceWindow(b_k0_n0_n1_k1_grid_desc,
                                                    b_block_slice_copy_step,
                                                    BGridMoveSliceWindowIteratorHacks{});

                __syncthreads();

                // LDS doubel buffer: load next data from device mem
                a_blockwise_copy.RunRead(
                    a_k0_m0_m1_k1_grid_desc, a_global_buf, AGridIteratorHacks{});
                b_blockwise_copy.RunRead(
                    b_k0_n0_n1_k1_grid_desc, b_global_buf, BGridIteratorHacks{});

                // LDS double buffer: GEMM on current data
                blockwise_gemm.Run(c_m10_m11_n10_n11_thread_desc,
                                   a_block_even_buf,
                                   b_block_even_buf,
                                   c_thread_buf);

                // LDS double buffer: store next data to LDS
                a_blockwise_copy.RunWrite(a_k0_m0_m1_k1_block_desc, a_block_odd_buf);
                b_blockwise_copy.RunWrite(b_k0_n0_n1_k1_block_desc, b_block_odd_buf);

                // odd iteration
                a_blockwise_copy.MoveSrcSliceWindow(a_k0_m0_m1_k1_grid_desc,
                                                    a_block_slice_copy_step,
                                                    AGridMoveSliceWindowIteratorHacks{});
                b_blockwise_copy.MoveSrcSliceWindow(b_k0_n0_n1_k1_grid_desc,
                                                    b_block_slice_copy_step,
                                                    BGridMoveSliceWindowIteratorHacks{});

                __syncthreads();

                // LDS doubel buffer: load next data from device mem
                a_blockwise_copy.RunRead(
                    a_k0_m0_m1_k1_grid_desc, a_global_buf, AGridIteratorHacks{});
                b_blockwise_copy.RunRead(
                    b_k0_n0_n1_k1_grid_desc, b_global_buf, BGridIteratorHacks{});

                // LDS double buffer: GEMM on current data
                blockwise_gemm.Run(
                    c_m10_m11_n10_n11_thread_desc, a_block_odd_buf, b_block_odd_buf, c_thread_buf);

                // LDS double buffer: store next data to LDS
                a_blockwise_copy.RunWrite(a_k0_m0_m1_k1_block_desc, a_block_even_buf);
                b_blockwise_copy.RunWrite(b_k0_n0_n1_k1_block_desc, b_block_even_buf);

                k_block_data_begin += 2 * KPerBlock;
            } while(k_block_data_begin < K0 - 2 * KPerBlock);
        }

        // LDS double buffer: tail
        if constexpr(HasDoubleTailKBlockLoop) // if has 2 iteration left
        {
            a_blockwise_copy.MoveSrcSliceWindow(a_k0_m0_m1_k1_grid_desc,
                                                a_block_slice_copy_step,
                                                AGridMoveSliceWindowIteratorHacks{});
            b_blockwise_copy.MoveSrcSliceWindow(b_k0_n0_n1_k1_grid_desc,
                                                b_block_slice_copy_step,
                                                BGridMoveSliceWindowIteratorHacks{});

            __syncthreads();

            // LDS double buffer: load last data from device mem
            a_blockwise_copy.RunRead(a_k0_m0_m1_k1_grid_desc, a_global_buf, AGridIteratorHacks{});
            b_blockwise_copy.RunRead(b_k0_n0_n1_k1_grid_desc, b_global_buf, BGridIteratorHacks{});

            // LDS double buffer: GEMM on 2nd-last data
            blockwise_gemm.Run(
                c_m10_m11_n10_n11_thread_desc, a_block_even_buf, b_block_even_buf, c_thread_buf);

            // LDS double buffer: store last data to LDS
            a_blockwise_copy.RunWrite(a_k0_m0_m1_k1_block_desc, a_block_odd_buf);
            b_blockwise_copy.RunWrite(b_k0_n0_n1_k1_block_desc, b_block_odd_buf);

            __syncthreads();

            // LDS double buffer: GEMM on last data
            blockwise_gemm.Run(
                c_m10_m11_n10_n11_thread_desc, a_block_odd_buf, b_block_odd_buf, c_thread_buf);
        }
        else // if has 1 iteration left
        {
            __syncthreads();

            // LDS double buffer: GEMM on last data
            blockwise_gemm.Run(
                c_m10_m11_n10_n11_thread_desc, a_block_even_buf, b_block_even_buf, c_thread_buf);
        }

        // output: register to global memory
        {
            constexpr index_t M11 =
                M1PerThreadM111 * M11N11ThreadClusterM1100 * M11N11ThreadClusterM1101;
            constexpr index_t N11 =
                N1PerThreadN111 * M11N11ThreadClusterN1100 * M11N11ThreadClusterN1101;

            constexpr index_t M10 = MPerBlockM1 / M11;
            constexpr index_t N10 = NPerBlockN1 / N11;

            constexpr index_t M111 = M1PerThreadM111;
            constexpr index_t N111 = N1PerThreadN111;

            constexpr auto c_m0_m10_m11_n0_n10_n11_thread_desc =
                make_dynamic_naive_tensor_descriptor_packed_v2(
                    make_tuple(I1,
                               Number<c_m10_m11_n10_n11_thread_tensor_lengths[I0]>{},
                               Number<c_m10_m11_n10_n11_thread_tensor_lengths[I1]>{},
                               I1,
                               Number<c_m10_m11_n10_n11_thread_tensor_lengths[I2]>{},
                               Number<c_m10_m11_n10_n11_thread_tensor_lengths[I3]>{}));

            const auto c_m10_m11_n10_n11_thread_origin_idx_on_block =
                blockwise_gemm.CalculateCM0M1N0N1ThreadOriginOnBlock(get_thread_local_1d_id());

            ThreadwiseDynamicTensorSliceTransfer_v1r3<
                FloatAcc,
                FloatC,
                decltype(c_m0_m10_m11_n0_n10_n11_thread_desc),
                decltype(c_m0_m10_m11_n0_n10_n11_grid_desc),
                Sequence<1,
                         c_m10_m11_n10_n11_thread_tensor_lengths[I0],
                         c_m10_m11_n10_n11_thread_tensor_lengths[I1],
                         1,
                         c_m10_m11_n10_n11_thread_tensor_lengths[I2],
                         c_m10_m11_n10_n11_thread_tensor_lengths[I3]>,
                CThreadTransferSrcDstAccessOrder,
                CThreadTransferSrcDstVectorDim,
                CThreadTransferDstScalarPerVector,
                CGlobalMemoryDataOperation,
                1,
                true>{c_m0_m10_m11_n0_n10_n11_grid_desc,
                      make_multi_index(im0,
                                       c_m10_m11_n10_n11_thread_origin_idx_on_block[I0],
                                       c_m10_m11_n10_n11_thread_origin_idx_on_block[I1],
                                       in0,
                                       c_m10_m11_n10_n11_thread_origin_idx_on_block[I2],
                                       c_m10_m11_n10_n11_thread_origin_idx_on_block[I3])}
                .Run(c_m0_m10_m11_n0_n10_n11_thread_desc,
                     make_tuple(I0, I0, I0, I0, I0, I0),
                     c_thread_buf,
                     c_m0_m10_m11_n0_n10_n11_grid_desc,
                     c_grid_buf,
                     CGridIteratorHacks{});
        }
    }
};

} // namespace ck
#endif
