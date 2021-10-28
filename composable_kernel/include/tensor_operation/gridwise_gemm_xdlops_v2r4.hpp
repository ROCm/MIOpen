#ifndef CK_GRIDWISE_GEMM_XDLOPS_V2R4_HPP
#define CK_GRIDWISE_GEMM_XDLOPS_V2R4_HPP

#include "common_header.hpp"
#include "multi_index_transform_helper.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "blockwise_gemm_xdlops.hpp"
#include "blockwise_tensor_slice_transfer.hpp"
#include "threadwise_tensor_slice_transfer.hpp"
#include "threadwise_tensor_slice_set.hpp"

namespace ck {

#if CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VALUE
template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename ABK0MK1GridDesc,
          typename BBK0NK1GridDesc,
          typename CM0N0M1N1M2M3M4N2GridDesc,
          typename CBlockClusterAdaptor,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_xdlops_v2r4(const FloatAB* __restrict__ p_a_grid,
                                const FloatAB* __restrict__ p_b_grid,
                                FloatC* __restrict__ p_c_grid,
                                const ABK0MK1GridDesc a_b_k0_m_k1_grid_desc,
                                const BBK0NK1GridDesc b_b_k0_n_k1_grid_desc,
                                const CM0N0M1N1M2M3M4N2GridDesc c_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc,
                                const CBlockClusterAdaptor c_block_cluster_adaptor)
{
    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid,
                                                  p_b_grid,
                                                  p_c_grid,
                                                  p_shared_block,
                                                  a_b_k0_m_k1_grid_desc,
                                                  b_b_k0_n_k1_grid_desc,
                                                  c_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc,
                                                  c_block_cluster_adaptor);
}
#elif CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VOID_POINTER
template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename ABK0MK1GridDesc,
          typename BBK0NK1GridDesc,
          typename CM0N0M1N1M2M3M4N2GridDesc,
          typename CBlockClusterAdaptor,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_xdlops_v2r4(const FloatAB* __restrict__ p_a_grid,
                                const FloatAB* __restrict__ p_b_grid,
                                FloatC* __restrict__ p_c_grid,
                                const void CONSTANT* p_a_b_k0_m_k1_grid_desc,
                                const void CONSTANT* p_b_b_k0_n_k1_grid_desc,
                                const void CONSTANT* p_c_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc,
                                const void CONSTANT* p_c_block_cluster_adaptor)
{
    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    const auto a_b_k0_m_k1_grid_desc = *reinterpret_cast<const ABK0MK1GridDesc*>(
        cast_pointer_to_generic_address_space(p_a_b_k0_m_k1_grid_desc));
    const auto b_b_k0_n_k1_grid_desc = *reinterpret_cast<const BBK0NK1GridDesc*>(
        cast_pointer_to_generic_address_space(p_b_b_k0_n_k1_grid_desc));
    const auto c_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc =
        *reinterpret_cast<const CM0N0M1N1M2M3M4N2GridDesc*>(
            cast_pointer_to_generic_address_space(p_c_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc));
    const auto c_block_cluster_adaptor = *reinterpret_cast<const CBlockClusterAdaptor*>(
        cast_pointer_to_generic_address_space(p_c_block_cluster_adaptor));

    __shared__ FloatAB p_shared_block[shared_block_size];

    GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid,
                                                  p_b_grid,
                                                  p_c_grid,
                                                  p_shared_block,
                                                  a_b_k0_m_k1_grid_desc,
                                                  b_b_k0_n_k1_grid_desc,
                                                  c_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc,
                                                  c_block_cluster_adaptor);
}
#endif

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          InMemoryDataOperationEnum_t CGlobalMemoryDataOperation,
          typename ABK0MK1GridDesc,
          typename BBK0NK1GridDesc,
          typename CMNGridDesc,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t K0PerBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t K1Value,
          index_t MRepeat,
          index_t NRepeat,
          typename ABlockTransferThreadSliceLengths_K0_M_K1,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_K1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          typename BBlockTransferThreadSliceLengths_K0_N_K1,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_K1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector,
          typename AGridStepHacks,
          typename BGridStepHacks,
          typename CGridStepHacks,
          typename AGridMoveSliceWindowStepHacks,
          typename BGridMoveSliceWindowStepHacks,
          bool CAccessOrderMRepeatNRepeat,
          bool ABlockLdsExtraM,
          bool BBlockLdsExtraN>
struct GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};

    // K1 should be Number<...>
    static constexpr auto K1 = Number<K1Value>{};

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        constexpr auto max_lds_align = K1;

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_k0_m_k1_block_desc = [&]() {
            if constexpr(ABlockLdsExtraM)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<K0PerBlock>{}, Number<MPerBlock>{}, K1),
                    make_tuple(Number<MPerBlock + 1>{} * K1, K1, I1));
            }
            else
            {
                return make_naive_tensor_descriptor_aligned(
                    make_tuple(Number<K0PerBlock>{}, Number<MPerBlock>{}, K1), max_lds_align);
            }
        }();

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_k0_n_k1_block_desc = [&]() {
            if constexpr(BBlockLdsExtraN)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<K0PerBlock>{}, Number<NPerBlock>{}, K1),
                    make_tuple(Number<NPerBlock + 1>{} * K1, K1, I1));
            }
            else
            {
                return make_naive_tensor_descriptor_aligned(
                    make_tuple(Number<K0PerBlock>{}, Number<NPerBlock>{}, K1), max_lds_align);
            }
        }();

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size =
            math::integer_least_multiple(a_k0_m_k1_block_desc.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_space_size =
            math::integer_least_multiple(b_k0_n_k1_block_desc.GetElementSpaceSize(), max_lds_align);

        return (a_block_space_size + b_block_space_size) * sizeof(FloatAB);
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    __host__ __device__ static constexpr bool
    CheckValidity(const ABK0MK1GridDesc& a_b_k0_m_k1_grid_desc,
                  const BBK0NK1GridDesc& b_b_k0_n_k1_grid_desc,
                  const CMNGridDesc& c_m_n_grid_desc,
                  index_t M01,
                  index_t N01)
    {
        static_assert(is_known_at_compile_time<remove_cv_t<decltype(K1)>>::value,
                      "wrong! K1 need to be known at compile-time");

        static_assert((MPerBlock % (MPerXDL * MRepeat) == 0) &&
                          (NPerBlock % (NRepeat * NPerXDL)) == 0,
                      "Invalid tuning param!");

        const auto M      = a_b_k0_m_k1_grid_desc.GetLength(I2);
        const auto N      = b_b_k0_n_k1_grid_desc.GetLength(I2);
        const auto K0     = a_b_k0_m_k1_grid_desc.GetLength(I1);
        const auto KBatch = a_b_k0_m_k1_grid_desc.GetLength(I0);

        if(!(M == c_m_n_grid_desc.GetLength(I0) && N == c_m_n_grid_desc.GetLength(I1) &&
             K0 == b_b_k0_n_k1_grid_desc.GetLength(I1) &&
             K1 == a_b_k0_m_k1_grid_desc.GetLength(I3) &&
             K1 == b_b_k0_n_k1_grid_desc.GetLength(I3) &&
             KBatch == b_b_k0_n_k1_grid_desc.GetLength(I0)))
            return false;

        if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && K0 % K0PerBlock == 0))
            return false;

        // check M01, N01
        constexpr auto M1 = Number<MPerBlock>{};
        constexpr auto N1 = Number<NPerBlock>{};

        const auto M0 = M / M1;
        const auto N0 = N / N1;

        if(!(M0 % M01 == 0 && N0 % N01 == 0))
            return false;

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        return true;
    }

    __host__ __device__ static constexpr index_t
    CalculateGridSize(const CMNGridDesc& c_m_n_grid_desc, index_t KBatch)
    {
        const auto M = c_m_n_grid_desc.GetLength(I0);
        const auto N = c_m_n_grid_desc.GetLength(I1);

        const index_t grid_size = (M / MPerBlock) * (N / NPerBlock) * KBatch;

        return grid_size;
    }

    __host__ __device__ static constexpr bool CalculateHasMainK0BlockLoop(index_t K0)
    {
        const bool has_main_k0_block_loop = K0 > K0PerBlock;

        return has_main_k0_block_loop;
    }

    __host__ __device__ static constexpr auto
    MakeCM0N0M1N1M2M3M4N2GridDescriptor(const CMNGridDesc& c_m_n_grid_desc)
    {
        constexpr auto max_lds_align = K1;

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_k0_m_k1_block_desc = [&]() {
            if constexpr(ABlockLdsExtraM)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<K0PerBlock>{}, Number<MPerBlock>{}, K1),
                    make_tuple(Number<MPerBlock + 1>{} * K1, K1, I1));
            }
            else
            {
                return make_naive_tensor_descriptor_aligned(
                    make_tuple(Number<K0PerBlock>{}, Number<MPerBlock>{}, K1), max_lds_align);
            }
        }();

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_k0_n_k1_block_desc = [&]() {
            if constexpr(BBlockLdsExtraN)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<K0PerBlock>{}, Number<NPerBlock>{}, K1),
                    make_tuple(Number<NPerBlock + 1>{} * K1, K1, I1));
            }
            else
            {
                return make_naive_tensor_descriptor_aligned(
                    make_tuple(Number<K0PerBlock>{}, Number<NPerBlock>{}, K1), max_lds_align);
            }
        }();

        using BlockwiseGemm =
            BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1<BlockSize,
                                                                FloatAB,
                                                                FloatAcc,
                                                                decltype(a_k0_m_k1_block_desc),
                                                                decltype(b_k0_n_k1_block_desc),
                                                                MPerXDL,
                                                                NPerXDL,
                                                                MRepeat,
                                                                NRepeat,
                                                                K1>;

        return BlockwiseGemm::MakeCM0N0M1N1M2M3M4N2GridDescriptor(c_m_n_grid_desc);
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto MakeCBlockClusterAdaptor(
        const CMNGridDesc& c_m_n_grid_desc, index_t M01, index_t N01, index_t KBatch)
    {
        const auto M = c_m_n_grid_desc.GetLength(I0);
        const auto N = c_m_n_grid_desc.GetLength(I1);

        constexpr auto M1 = Number<MPerBlock>{};
        constexpr auto N1 = Number<NPerBlock>{};

        const auto M0 = M / M1;
        const auto N0 = N / N1;

        const auto M00 = M0 / M01;
        const auto N00 = N0 / N01;

        const auto kbatch_m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_pass_through_transform(KBatch),
                           make_unmerge_transform(make_tuple(M00, M01)),
                           make_unmerge_transform(make_tuple(N00, N01))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2, 4>{}));

        const auto c_blockid_to_kbatch_m00_m01_n00_n01_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(KBatch, M00, N00, M01, N01))),
                make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                make_tuple(Sequence<0>{}));

        const auto c_blockid_to_kbatch_m0_n0_block_cluster_adaptor =
            chain_tensor_adaptors(kbatch_m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor,
                                  c_blockid_to_kbatch_m00_m01_n00_n01_block_cluster_adaptor);

        return c_blockid_to_kbatch_m0_n0_block_cluster_adaptor;
    }

    using CM0N0M1N1M2M3M4N2GridDesc = decltype(MakeCM0N0M1N1M2M3M4N2GridDescriptor(CMNGridDesc{}));
    using CBlockClusterAdaptor      = decltype(MakeCBlockClusterAdaptor(CMNGridDesc{}, 1, 1, 1));

    template <bool HasMainKBlockLoop>
    __device__ static void Run(const FloatAB* __restrict__ p_a_grid,
                               const FloatAB* __restrict__ p_b_grid,
                               FloatC* __restrict__ p_c_grid,
                               FloatAB* __restrict__ p_shared_block,
                               const ABK0MK1GridDesc& a_b_k0_m_k1_grid_desc,
                               const BBK0NK1GridDesc& b_b_k0_n_k1_grid_desc,
                               const CM0N0M1N1M2M3M4N2GridDesc& c_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc,
                               const CBlockClusterAdaptor& c_block_cluster_adaptor)
    {
        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_a_grid, a_b_k0_m_k1_grid_desc.GetElementSpaceSize());
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_b_grid, b_b_k0_n_k1_grid_desc.GetElementSpaceSize());
        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_c_grid, c_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc.GetElementSpaceSize());

        const auto K0 = a_b_k0_m_k1_grid_desc.GetLength(I1);

        // divide block work by [M, N]
        const auto block_work_idx =
            c_block_cluster_adaptor.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        const index_t k_batch_id = block_work_idx[I0];
        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * MPerBlock);

        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I2] * NPerBlock);

        // lds max alignment
        constexpr auto max_lds_align = K1;

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_k0_m_k1_block_desc = [&]() {
            if constexpr(ABlockLdsExtraM)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<K0PerBlock>{}, Number<MPerBlock>{}, K1),
                    make_tuple(Number<MPerBlock + 1>{} * K1, K1, I1));
            }
            else
            {
                return make_naive_tensor_descriptor_aligned(
                    make_tuple(Number<K0PerBlock>{}, Number<MPerBlock>{}, K1), max_lds_align);
            }
        }();

        constexpr auto a_b_k0_m_k1_block_desc = [&]() {
            if constexpr(ABlockLdsExtraM)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<1>{}, Number<K0PerBlock>{}, Number<MPerBlock>{}, K1),
                    make_tuple(Number<K0PerBlock>{} * Number<MPerBlock + 1>{} * K1,
                               Number<MPerBlock + 1>{} * K1,
                               K1,
                               I1));
            }
            else
            {
                return make_naive_tensor_descriptor_aligned(
                    make_tuple(Number<1>{}, Number<K0PerBlock>{}, Number<MPerBlock>{}, K1),
                    max_lds_align);
            }
        }();
        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_k0_n_k1_block_desc = [&]() {
            if constexpr(BBlockLdsExtraN)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<K0PerBlock>{}, Number<NPerBlock>{}, K1),
                    make_tuple(Number<NPerBlock + 1>{} * K1, K1, I1));
            }
            else
            {
                return make_naive_tensor_descriptor_aligned(
                    make_tuple(Number<K0PerBlock>{}, Number<NPerBlock>{}, K1), max_lds_align);
            }
        }();

        constexpr auto b_b_k0_n_k1_block_desc = [&]() {
            if constexpr(BBlockLdsExtraN)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<1>{}, Number<K0PerBlock>{}, Number<NPerBlock>{}, K1),
                    make_tuple(Number<K0PerBlock>{} * Number<NPerBlock + 1>{} * K1,
                               Number<NPerBlock + 1>{} * K1,
                               K1,
                               I1));
            }
            else
            {
                return make_naive_tensor_descriptor_aligned(
                    make_tuple(Number<1>{}, Number<K0PerBlock>{}, Number<NPerBlock>{}, K1),
                    max_lds_align);
            }
        }();
        // A matrix blockwise copy
        auto a_blockwise_copy =
            BlockwiseTensorSliceTransfer_v4<BlockSize,
                                            InMemoryDataOperationEnum_t::Set,
                                            Sequence<1, K0PerBlock, MPerBlock, K1>,
                                            ABlockTransferThreadSliceLengths_K0_M_K1,
                                            ABlockTransferThreadClusterLengths_K0_M_K1,
                                            ABlockTransferThreadClusterArrangeOrder,
                                            FloatAB,
                                            FloatAB,
                                            decltype(a_b_k0_m_k1_grid_desc),
                                            decltype(a_b_k0_m_k1_block_desc),
                                            ABlockTransferSrcAccessOrder,
                                            Sequence<0, 2, 1, 3>,
                                            ABlockTransferSrcVectorDim,
                                            3,
                                            ABlockTransferSrcScalarPerVector,
                                            ABlockTransferDstScalarPerVector_K1,
                                            1,
                                            1,
                                            AThreadTransferSrcResetCoordinateAfterRun,
                                            true>(
                a_b_k0_m_k1_grid_desc,
                make_multi_index(k_batch_id, 0, m_block_data_idx_on_grid, 0),
                a_b_k0_m_k1_block_desc,
                make_multi_index(0, 0, 0, 0));

        // B matrix blockwise copy
        auto b_blockwise_copy =
            BlockwiseTensorSliceTransfer_v4<BlockSize,
                                            InMemoryDataOperationEnum_t::Set,
                                            Sequence<1, K0PerBlock, NPerBlock, K1>,
                                            BBlockTransferThreadSliceLengths_K0_N_K1,
                                            BBlockTransferThreadClusterLengths_K0_N_K1,
                                            BBlockTransferThreadClusterArrangeOrder,
                                            FloatAB,
                                            FloatAB,
                                            decltype(b_b_k0_n_k1_grid_desc),
                                            decltype(b_b_k0_n_k1_block_desc),
                                            BBlockTransferSrcAccessOrder,
                                            Sequence<0, 2, 1, 3>,
                                            BBlockTransferSrcVectorDim,
                                            3,
                                            BBlockTransferSrcScalarPerVector,
                                            BBlockTransferDstScalarPerVector_K1,
                                            1,
                                            1,
                                            BThreadTransferSrcResetCoordinateAfterRun,
                                            true>(
                b_b_k0_n_k1_grid_desc,
                make_multi_index(k_batch_id, 0, n_block_data_idx_on_grid, 0),
                b_b_k0_n_k1_block_desc,
                make_multi_index(0, 0, 0, 0));

        // GEMM definition
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[K0PerBlock, MPerBlock] is in LDS
        //     b_mtx[K0PerBlock, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //       register
        // sanity check

        auto blockwise_gemm =
            BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1<BlockSize,
                                                                FloatAB,
                                                                FloatAcc,
                                                                decltype(a_k0_m_k1_block_desc),
                                                                decltype(b_k0_n_k1_block_desc),
                                                                MPerXDL,
                                                                NPerXDL,
                                                                MRepeat,
                                                                NRepeat,
                                                                K1>{};

        auto c_thread_buf = blockwise_gemm.GetCThreadBuffer();

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size =
            math::integer_least_multiple(a_k0_m_k1_block_desc.GetElementSpaceSize(), max_lds_align);

        FloatAB* p_a_block = p_shared_block;
        FloatAB* p_b_block = p_shared_block + a_block_space_size;

        constexpr auto a_block_slice_copy_step = make_multi_index(0, K0PerBlock, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(0, K0PerBlock, 0, 0);

        // hack to control index calculation when iterating over A and B matrix for threadwise copy
        constexpr auto a_k0_m_k1_grid_step_hacks = AGridStepHacks{};
        constexpr auto b_k0_n_k1_grid_step_hacks = BGridStepHacks{};

        // hack to control index calculation when move slice window for A and B matrix for
        // threadwise copy
        constexpr auto a_k0_m_k1_grid_move_slice_window_step_hack = AGridMoveSliceWindowStepHacks{};
        constexpr auto b_k0_n_k1_grid_move_slice_window_step_hack = BGridMoveSliceWindowStepHacks{};

        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum_t::Lds>(
            p_a_block, a_k0_m_k1_block_desc.GetElementSpaceSize());
        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum_t::Lds>(
            p_b_block, b_k0_n_k1_block_desc.GetElementSpaceSize());

        // preload data into LDS
        {
            a_blockwise_copy.RunRead(a_b_k0_m_k1_grid_desc, a_grid_buf, a_k0_m_k1_grid_step_hacks);
            b_blockwise_copy.RunRead(b_b_k0_n_k1_grid_desc, b_grid_buf, b_k0_n_k1_grid_step_hacks);

            a_blockwise_copy.RunWrite(a_b_k0_m_k1_block_desc, a_block_buf);
            b_blockwise_copy.RunWrite(b_b_k0_n_k1_block_desc, b_block_buf);
        }

        // main body
        index_t k_block_data_begin = 0;
        if constexpr(HasMainKBlockLoop)
        {
            do
            {
                a_blockwise_copy.MoveSrcSliceWindow(a_b_k0_m_k1_grid_desc,
                                                    a_block_slice_copy_step,
                                                    a_k0_m_k1_grid_move_slice_window_step_hack);
                b_blockwise_copy.MoveSrcSliceWindow(b_b_k0_n_k1_grid_desc,
                                                    b_block_slice_copy_step,
                                                    b_k0_n_k1_grid_move_slice_window_step_hack);

                a_blockwise_copy.RunRead(
                    a_b_k0_m_k1_grid_desc, a_grid_buf, a_k0_m_k1_grid_step_hacks);

                block_sync_lds();

                b_blockwise_copy.RunRead(
                    b_b_k0_n_k1_grid_desc, b_grid_buf, b_k0_n_k1_grid_step_hacks);

                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();

                a_blockwise_copy.RunWrite(a_b_k0_m_k1_block_desc, a_block_buf);
                b_blockwise_copy.RunWrite(b_b_k0_n_k1_block_desc, b_block_buf);

                k_block_data_begin += K0PerBlock;
            } while(k_block_data_begin < (K0 - K0PerBlock));
        }

        // tail
        {
            block_sync_lds();

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }

        // output: register to global memory
        {
            constexpr auto c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc =
                blockwise_gemm.GetCM0N0M1N1M2M3M4N2BlockDescriptor();

            constexpr auto M0 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I0);
            constexpr auto N0 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I1);
            constexpr auto M1 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I2);
            constexpr auto N1 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I3);
            constexpr auto M2 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I4);
            constexpr auto M3 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I5);
            constexpr auto M4 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I6);
            constexpr auto N2 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I7);

            constexpr auto c_m0_n0_m1_n1_m2_m3_m4_n2_thread_desc =
                make_naive_tensor_descriptor_packed(make_tuple(
                    Number<M0>{}, Number<N0>{}, I1, I1, Number<M2>{}, I1, Number<M4>{}, I1));

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

            const index_t m_thread_data_on_grid =
                m_block_data_idx_on_grid + c_thread_mtx_on_block[I0];

            const index_t n_thread_data_on_grid =
                n_block_data_idx_on_grid + c_thread_mtx_on_block[I1];

            constexpr auto c_m0_n0_m1_n1_m2_m3_m4_n2_grid_tensor_step_hacks = CGridStepHacks{};

            const auto m_thread_data_on_grid_to_m0_m1_m2_m3_m4_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(M0, M1, M2, M3, M4))),
                    make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                    make_tuple(Sequence<0>{}));

            const auto m_thread_data_on_grid_idx =
                m_thread_data_on_grid_to_m0_m1_m2_m3_m4_adaptor.CalculateBottomIndex(
                    make_multi_index(m_thread_data_on_grid));

            const auto n_thread_data_on_grid_to_n0_n1_n2_adaptor = make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(N0, N1, N2))),
                make_tuple(Sequence<0, 1, 2>{}),
                make_tuple(Sequence<0>{}));

            const auto n_thread_data_on_grid_idx =
                n_thread_data_on_grid_to_n0_n1_n2_adaptor.CalculateBottomIndex(
                    make_multi_index(n_thread_data_on_grid));

            auto c_thread_copy =
                ThreadwiseTensorSliceTransfer_v1r3<FloatAcc,
                                                   FloatC,
                                                   decltype(c_m0_n0_m1_n1_m2_m3_m4_n2_thread_desc),
                                                   decltype(c_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc),
                                                   Sequence<M0, N0, I1, I1, M2, I1, M4, I1>,
                                                   CThreadTransferSrcDstAccessOrder,
                                                   CThreadTransferSrcDstVectorDim,
                                                   CThreadTransferDstScalarPerVector,
                                                   CGlobalMemoryDataOperation,
                                                   1,
                                                   true>{

                    c_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc,
                    make_multi_index(m_thread_data_on_grid_idx[I0],
                                     n_thread_data_on_grid_idx[I0],
                                     m_thread_data_on_grid_idx[I1],
                                     n_thread_data_on_grid_idx[I1],
                                     m_thread_data_on_grid_idx[I2],
                                     m_thread_data_on_grid_idx[I3],
                                     m_thread_data_on_grid_idx[I4],
                                     n_thread_data_on_grid_idx[I2])};

            c_thread_copy.Run(c_m0_n0_m1_n1_m2_m3_m4_n2_thread_desc,
                              make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                              c_thread_buf,
                              c_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc,
                              c_grid_buf,
                              c_m0_n0_m1_n1_m2_m3_m4_n2_grid_tensor_step_hacks);
        }
    }
}; // namespace ck

} // namespace ck
#endif
