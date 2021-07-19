#ifndef CK_GRIDWISE_DYNAMIC_GEMM_XDLOPS_V2R3_HPP
#define CK_GRIDWISE_DYNAMIC_GEMM_XDLOPS_V2R3_HPP

#include "common_header.hpp"
#include "dynamic_multi_index_transform_helper.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "blockwise_gemm_xdlops.hpp"
#include "blockwise_dynamic_tensor_slice_transfer.hpp"
#include "threadwise_dynamic_tensor_slice_transfer.hpp"
#include "threadwise_dynamic_tensor_slice_set.hpp"

namespace ck {

#if CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VALUE
template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename AK0MK1GridDesc,
          typename BK0NK1GridDesc,
          typename CM0M1M2NGridDesc,
          typename CBlockClusterAdaptor>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_dynamic_gemm_xdlops_v2r3(const FloatAB* __restrict__ p_a_grid,
                                        const FloatAB* __restrict__ p_b_grid,
                                        FloatC* __restrict__ p_c_grid,
                                        const AK0MK1GridDesc a_k0_m_k1_grid_desc,
                                        const BK0NK1GridDesc b_k0_n_k1_grid_desc,
                                        const CM0M1M2NGridDesc c_m0_m1_m2_n_grid_desc,
                                        const CBlockClusterAdaptor c_block_cluster_adaptor)
{
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
                      c_block_cluster_adaptor);
}
#elif CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VOID_POINTER
template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename AK0MK1GridDesc,
          typename BK0NK1GridDesc,
          typename CM0M1M2NGridDesc,
          typename CBlockClusterAdaptor>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_dynamic_gemm_xdlops_v2r3(const FloatAB* __restrict__ p_a_grid,
                                        const FloatAB* __restrict__ p_b_grid,
                                        FloatC* __restrict__ p_c_grid,
                                        const void __CONSTANT__* p_a_k0_m_k1_grid_desc,
                                        const void __CONSTANT__* p_b_k0_n_k1_grid_desc,
                                        const void __CONSTANT__* p_c_m0_m1_m2_n_grid_desc,
                                        const void __CONSTANT__* p_c_block_cluster_adaptor)
{
    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    const auto a_k0_m_k1_grid_desc =
        *reinterpret_cast<const AK0MK1GridDesc*>((const void*)p_a_k0_m_k1_grid_desc);
    const auto b_k0_n_k1_grid_desc =
        *reinterpret_cast<const BK0NK1GridDesc*>((const void*)p_b_k0_n_k1_grid_desc);
    const auto c_m0_m1_m2_n_grid_desc =
        *reinterpret_cast<const CM0M1M2NGridDesc*>((const void*)p_c_m0_m1_m2_n_grid_desc);
    const auto c_block_cluster_adaptor =
        *reinterpret_cast<const CBlockClusterAdaptor*>((const void*)p_c_block_cluster_adaptor);

    __shared__ FloatAB p_shared_block[shared_block_size];

    GridwiseGemm::Run(p_a_grid,
                      p_b_grid,
                      p_c_grid,
                      p_shared_block,
                      a_k0_m_k1_grid_desc,
                      b_k0_n_k1_grid_desc,
                      c_m0_m1_m2_n_grid_desc,
                      c_block_cluster_adaptor);
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
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerWave,
          index_t NPerWave,
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
          typename AGridIteratorHacks,
          typename BGridIteratorHacks,
          typename CGridIteratorHacks,
          typename AGridMoveSliceWindowIteratorHacks,
          typename BGridMoveSliceWindowIteratorHacks,
          bool CAccessOrderMRepeatNRepeat>
struct GridwiseDynamicGemm_k0mk1_k0nk1_mn_xdlops_v2r3
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    // K1 should be Number<...>
    static constexpr auto K1 = Number<K1Value>{};

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        constexpr auto max_lds_align = K1;

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_k0_m_k1_block_desc = make_dynamic_naive_tensor_descriptor_aligned_v2(
            make_tuple(Number<KPerBlock>{}, Number<MPerBlock>{}, K1), max_lds_align);

        // B matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto b_k0_n_k1_block_desc = make_dynamic_naive_tensor_descriptor_aligned_v2(
            make_tuple(Number<KPerBlock>{}, Number<NPerBlock>{}, K1), max_lds_align);

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size =
            math::integer_least_multiple(a_k0_m_k1_block_desc.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_space_size =
            math::integer_least_multiple(b_k0_n_k1_block_desc.GetElementSpaceSize(), max_lds_align);

        return (a_block_space_size + b_block_space_size) * sizeof(FloatAB);
    }

    __host__ __device__ static constexpr bool
    CheckValidity(const AK0MK1GridDesc& a_k0_m_k1_grid_desc,
                  const BK0NK1GridDesc& b_k0_n_k1_grid_desc,
                  const CMNGridDesc& c_m_n_grid_desc)
    {
        // TODO: turn on this
        static_assert(is_known_at_compile_time<remove_cv_t<decltype(K1)>>::value,
                      "wrong! K1 need to be known at compile-time");

        const auto M  = a_k0_m_k1_grid_desc.GetLength(I1);
        const auto N  = b_k0_n_k1_grid_desc.GetLength(I1);
        const auto K0 = a_k0_m_k1_grid_desc.GetLength(I0);

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)

        return (M == c_m_n_grid_desc.GetLength(I0) && N == c_m_n_grid_desc.GetLength(I1) &&
                K0 == b_k0_n_k1_grid_desc.GetLength(I0) &&
                K1 == a_k0_m_k1_grid_desc.GetLength(I2) &&
                K1 == b_k0_n_k1_grid_desc.GetLength(I2)) &&
               (M % MPerBlock == 0 && N % NPerBlock == 0 && K0 % KPerBlock == 0) &&
               (MPerBlock % MPerWave == 0 && NPerBlock % NPerWave == 0);
    }

    __host__ __device__ static constexpr index_t
    CalculateGridSize(const CMNGridDesc& c_m_n_grid_desc)
    {
        const auto M = c_m_n_grid_desc.GetLength(I0);
        const auto N = c_m_n_grid_desc.GetLength(I1);

        const index_t grid_size = (M / MPerBlock) * (N / NPerBlock);

        return grid_size;
    }

    __host__ __device__ static constexpr auto
    MakeCM0M1M2NGridDescriptor(const CMNGridDesc& c_m_n_grid_desc)
    {
        const auto M = c_m_n_grid_desc.GetLength(I0);
        const auto N = c_m_n_grid_desc.GetLength(I1);

        constexpr auto xdlops_gemm = XdlopsGemm<FloatAB, MPerWave, NPerWave, K1>{};

        constexpr auto CLayout = xdlops_gemm.GetCLayout();

        constexpr auto M0 = Number<CLayout.M1()>{};
        constexpr auto M1 = Number<CLayout.N1()>{};
        constexpr auto M2 = Number<CLayout.M0()>{};

        constexpr index_t MWaves = MPerBlock / (MPerWave * MRepeat);
        constexpr index_t NWaves = NPerBlock / (NPerWave * NRepeat);

        constexpr auto N0 = Number<CLayout.N1()>{};
        constexpr auto N1 = Number<CLayout.N0()>{};

        const auto c_m0_m1_m2_n_grid_desc = transform_dynamic_tensor_descriptor(
            c_m_n_grid_desc,
            make_tuple(make_unmerge_transform(make_tuple(MRepeat, MWaves, M0, M1, M2)),
                       make_unmerge_transform(make_tuple(NRepeat, NWaves, N1))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2, 4, 5, 6>{}, Sequence<1, 3, 7>{}));

        return c_m0_m1_m2_n_grid_desc;
    }

    __host__ __device__ static constexpr auto
    MakeCBlockClusterAdaptor(const CMNGridDesc& c_m_n_grid_desc)
    {
        const auto M = c_m_n_grid_desc.GetLength(I0);
        const auto N = c_m_n_grid_desc.GetLength(I1);

        constexpr auto M1 = Number<MPerBlock>{};
        constexpr auto N1 = Number<NPerBlock>{};

        const auto M0 = M / M1;
        const auto N0 = N / N1;

#if 1
        const auto c_blockid_to_m0_n0_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(make_tuple(make_merge_transform(make_tuple(M0, N0))),
                                             make_tuple(Sequence<0, 1>{}),
                                             make_tuple(Sequence<0>{}));
#elif 1
        const auto c_blockid_to_m0_n0_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(make_tuple(make_merge_transform(make_tuple(N0, M0))),
                                             make_tuple(Sequence<1, 0>{}),
                                             make_tuple(Sequence<0>{}));
#endif

        return c_blockid_to_m0_n0_block_cluster_adaptor;
    }

    using CM0M1M2NGridDesc     = decltype(MakeCM0M1M2NGridDescriptor(CMNGridDesc{}));
    using CBlockClusterAdaptor = decltype(MakeCBlockClusterAdaptor(CMNGridDesc{}));

    __device__ static void Run(const FloatAB* __restrict__ p_a_grid,
                               const FloatAB* __restrict__ p_b_grid,
                               FloatC* __restrict__ p_c_grid,
                               FloatAB* __restrict__ p_shared_block,
                               const AK0MK1GridDesc& a_k0_m_k1_grid_desc,
                               const BK0NK1GridDesc& b_k0_n_k1_grid_desc,
                               const CM0M1M2NGridDesc& c_m0_m1_m2_n_grid_desc,
                               const CBlockClusterAdaptor& c_block_cluster_adaptor)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        const auto a_grid_buf = make_dynamic_buffer<AddressSpace::Global>(
            p_a_grid, a_k0_m_k1_grid_desc.GetElementSpaceSize());
        const auto b_grid_buf = make_dynamic_buffer<AddressSpace::Global>(
            p_b_grid, b_k0_n_k1_grid_desc.GetElementSpaceSize());
        auto c_grid_buf = make_dynamic_buffer<AddressSpace::Global>(
            p_c_grid, c_m0_m1_m2_n_grid_desc.GetElementSpaceSize());

        const auto K0 = a_k0_m_k1_grid_desc.GetLength(I0);
        const auto M  = a_k0_m_k1_grid_desc.GetLength(I1);
        const auto N  = b_k0_n_k1_grid_desc.GetLength(I1);

        // divide block work by [M, N]
        const auto block_work_idx =
            c_block_cluster_adaptor.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * MPerBlock);

        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * NPerBlock);

        // lds max alignment
        constexpr auto max_lds_align = K1;

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_k0_m_k1_block_desc = make_dynamic_naive_tensor_descriptor_aligned_v2(
            make_tuple(Number<KPerBlock>{}, Number<MPerBlock>{}, K1), max_lds_align);

        // B matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto b_k0_n_k1_block_desc = make_dynamic_naive_tensor_descriptor_aligned_v2(
            make_tuple(Number<KPerBlock>{}, Number<NPerBlock>{}, K1), max_lds_align);

        // A matrix blockwise copy
        auto a_blockwise_copy =
            BlockwiseDynamicTensorSliceTransfer_v4<BlockSize,
                                                   InMemoryDataOperation::Set,
                                                   Sequence<KPerBlock, MPerBlock, K1>,
                                                   ABlockTransferThreadSliceLengths_K0_M_K1,
                                                   ABlockTransferThreadClusterLengths_K0_M_K1,
                                                   ABlockTransferThreadClusterArrangeOrder,
                                                   FloatAB,
                                                   FloatAB,
                                                   decltype(a_k0_m_k1_grid_desc),
                                                   decltype(a_k0_m_k1_block_desc),
                                                   ABlockTransferSrcAccessOrder,
                                                   Sequence<1, 0, 2>,
                                                   ABlockTransferSrcVectorDim,
                                                   2,
                                                   ABlockTransferSrcScalarPerVector,
                                                   ABlockTransferDstScalarPerVector_K1,
                                                   1,
                                                   1,
                                                   AThreadTransferSrcResetCoordinateAfterRun,
                                                   true>(
                a_k0_m_k1_grid_desc,
                make_multi_index(0, m_block_data_idx_on_grid, 0),
                a_k0_m_k1_block_desc,
                make_multi_index(0, 0, 0));

        // B matrix blockwise copy
        auto b_blockwise_copy =
            BlockwiseDynamicTensorSliceTransfer_v4<BlockSize,
                                                   InMemoryDataOperation::Set,
                                                   Sequence<KPerBlock, NPerBlock, K1>,
                                                   BBlockTransferThreadSliceLengths_K0_N_K1,
                                                   BBlockTransferThreadClusterLengths_K0_N_K1,
                                                   BBlockTransferThreadClusterArrangeOrder,
                                                   FloatAB,
                                                   FloatAB,
                                                   decltype(b_k0_n_k1_grid_desc),
                                                   decltype(b_k0_n_k1_block_desc),
                                                   BBlockTransferSrcAccessOrder,
                                                   Sequence<1, 0, 2>,
                                                   BBlockTransferSrcVectorDim,
                                                   2,
                                                   BBlockTransferSrcScalarPerVector,
                                                   BBlockTransferDstScalarPerVector_K1,
                                                   1,
                                                   1,
                                                   BThreadTransferSrcResetCoordinateAfterRun,
                                                   true>(
                b_k0_n_k1_grid_desc,
                make_multi_index(0, n_block_data_idx_on_grid, 0),
                b_k0_n_k1_block_desc,
                make_multi_index(0, 0, 0));

        // GEMM definition
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[KPerBlock, MPerBlock] is in LDS
        //     b_mtx[KPerBlock, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //       register
        // sanity check

        static_assert(MPerBlock % (MPerWave * MRepeat) == 0 &&
                          NPerBlock % (NPerWave * NRepeat) == 0,
                      "wrong!");

        constexpr auto a_k0_m0_m1_k1_block_desc = transform_dynamic_tensor_descriptor(
            a_k0_m_k1_block_desc,
            make_tuple(make_pass_through_transform(Number<KPerBlock>{}),
                       make_unmerge_transform(
                           make_tuple(Number<MRepeat>{}, Number<MPerBlock / MRepeat>{})),
                       make_pass_through_transform(K1)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

        constexpr auto b_k0_n0_n1_k1_block_desc = transform_dynamic_tensor_descriptor(
            b_k0_n_k1_block_desc,
            make_tuple(make_pass_through_transform(Number<KPerBlock>{}),
                       make_unmerge_transform(
                           make_tuple(Number<NRepeat>{}, Number<NPerBlock / NRepeat>{})),
                       make_pass_through_transform(K1)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

        const auto blockwise_gemm =
            BlockwiseGemmXdlops_km_kn_m0m1m2n_v1<BlockSize,
                                                 FloatAB,
                                                 decltype(a_k0_m0_m1_k1_block_desc),
                                                 decltype(b_k0_n0_n1_k1_block_desc),
                                                 MPerWave,
                                                 NPerWave,
                                                 K1>{};

        constexpr auto CLayout = blockwise_gemm.GetCLayout();

        constexpr index_t BlkSize   = CLayout.GetBlkSize();
        constexpr index_t NumBlks   = CLayout.GetNumBlks();
        constexpr index_t NumXdlops = CLayout.GetNumXdlops();

        static_assert(NumBlks == 1 && NumXdlops == 1, "K Reduction Mfma only");

        constexpr auto c_mr_nr_blk_desc = make_dynamic_naive_tensor_descriptor_packed_v2(
            make_tuple(Number<MRepeat>{}, Number<NRepeat>{}));

        StaticBuffer<AddressSpace::Vgpr,
                     vector_type<FloatAcc, BlkSize>,
                     c_mr_nr_blk_desc.GetElementSpaceSize()>
            c_thread_buf;

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size =
            math::integer_least_multiple(a_k0_m_k1_block_desc.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_space_size =
            math::integer_least_multiple(b_k0_n_k1_block_desc.GetElementSpaceSize(), max_lds_align);

        FloatAB* p_a_block = p_shared_block;
        FloatAB* p_b_block = p_shared_block + a_block_space_size;

        constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(KPerBlock, 0, 0);

        // hack to control index calculation when iterating over A and B matrix for threadwise copy
        constexpr auto a_k0_m_k1_grid_iterator_hacks = AGridIteratorHacks{};
        constexpr auto b_k0_n_k1_grid_iterator_hacks = BGridIteratorHacks{};

        // hack to control index calculation when move slice window for A and B matrix for
        // threadwise copy
        constexpr auto a_k0_m_k1_grid_move_slice_window_iterator_hack =
            AGridMoveSliceWindowIteratorHacks{};
        constexpr auto b_k0_n_k1_grid_move_slice_window_iterator_hack =
            BGridMoveSliceWindowIteratorHacks{};

        auto a_block_buf = make_dynamic_buffer<AddressSpace::Lds>(
            p_a_block, a_k0_m_k1_block_desc.GetElementSpaceSize());
        auto b_block_buf = make_dynamic_buffer<AddressSpace::Lds>(
            p_b_block, b_k0_n_k1_block_desc.GetElementSpaceSize());

        // preload data into LDS
        {
            a_blockwise_copy.RunRead(
                a_k0_m_k1_grid_desc, a_grid_buf, a_k0_m_k1_grid_iterator_hacks);
            b_blockwise_copy.RunRead(
                b_k0_n_k1_grid_desc, b_grid_buf, b_k0_n_k1_grid_iterator_hacks);

            a_blockwise_copy.RunWrite(a_k0_m_k1_block_desc, a_block_buf);
            b_blockwise_copy.RunWrite(b_k0_n_k1_block_desc, b_block_buf);
        }

        // main body
        index_t k_block_data_begin = 0;

        do
        {
            a_blockwise_copy.MoveSrcSliceWindow(a_k0_m_k1_grid_desc,
                                                a_block_slice_copy_step,
                                                a_k0_m_k1_grid_move_slice_window_iterator_hack);
            b_blockwise_copy.MoveSrcSliceWindow(b_k0_n_k1_grid_desc,
                                                b_block_slice_copy_step,
                                                b_k0_n_k1_grid_move_slice_window_iterator_hack);

            a_blockwise_copy.RunRead(
                a_k0_m_k1_grid_desc, a_grid_buf, a_k0_m_k1_grid_iterator_hacks);

            block_sync_lds();

            b_blockwise_copy.RunRead(
                b_k0_n_k1_grid_desc, b_grid_buf, b_k0_n_k1_grid_iterator_hacks);

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

            block_sync_lds();

            a_blockwise_copy.RunWrite(a_k0_m_k1_block_desc, a_block_buf);
            b_blockwise_copy.RunWrite(b_k0_n_k1_block_desc, b_block_buf);

            k_block_data_begin += KPerBlock;
        } while(k_block_data_begin < (K0 - KPerBlock));

        // tail
        {
            block_sync_lds();

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }

#if 0
        // output: register to global memory
        {
            constexpr index_t M0 = CLayout.M1();
            constexpr index_t M1 = CLayout.N1();
            constexpr index_t M2 = CLayout.M0();

            constexpr index_t N0 = CLayout.N1();
            constexpr index_t N1 = CLayout.N0();

            constexpr auto c_m0_m1_m2_n_thread_desc =
                make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(Number<MRepeat>{},
                                                                          Number<NRepeat>{},
                                                                          Number<1>{},
                                                                          Number<1>{},
                                                                          Number<M0>{},
                                                                          Number<1>{},
                                                                          Number<M2>{},
                                                                          Number<1>{}));

            StaticBuffer<AddressSpace::Vgpr, FloatC, c_m0_m1_m2_n_thread_desc.GetElementSpaceSize()>
                c_blk_buf_;

            static_for<0, MRepeat, 1>{}([&](auto mr_i) {
                static_for<0, NRepeat, 1>{}([&](auto nr_i) {
                    constexpr auto blk_off =
                        c_mr_nr_blk_desc.CalculateOffset(make_tuple(mr_i, nr_i));

                    static_for<0, BlkSize, 1>{}([&](auto j) {
                        c_blk_buf_(Number<blk_off * BlkSize + j>{}) =
                            c_thread_buf[Number<blk_off>{}]
                                .template AsType<FloatAcc>()[Number<j>{}];
                    });
                });
            });

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

            const index_t m_thread_data_on_grid =
                m_block_data_idx_on_grid + c_thread_mtx_on_block[I0];

            const index_t n_thread_data_on_grid =
                n_block_data_idx_on_grid + c_thread_mtx_on_block[I1];

            constexpr auto c_m0_m1_m2_n_grid_tensor_iterator_hacks = CGridIteratorHacks{};

            constexpr index_t MWaves = MPerBlock / (MPerWave * MRepeat);
            constexpr index_t NWaves = NPerBlock / (NPerWave * NRepeat);

            ThreadwiseDynamicTensorSliceTransfer_v1r3<
                FloatC,
                FloatC,
                decltype(c_m0_m1_m2_n_thread_desc),
                decltype(c_m0_m1_m2_n_grid_desc),
                Sequence<MRepeat, NRepeat, 1, 1, M0, 1, M2, 1>,
                CThreadTransferSrcDstAccessOrder,
                CThreadTransferSrcDstVectorDim,
                CThreadTransferDstScalarPerVector,
                CGlobalMemoryDataOperation,
                1,
                true>{
                c_m0_m1_m2_n_grid_desc,
                make_multi_index(m_thread_data_on_grid / (M2 * M1 * M0 * MWaves),
                                 n_thread_data_on_grid / (N1 * NWaves),
                                 m_thread_data_on_grid % (M2 * M1 * M0 * MWaves) / (M2 * M1 * M0),
                                 n_thread_data_on_grid % (N1 * NWaves) / N1,
                                 m_thread_data_on_grid % (M2 * M1 * M0) / (M2 * M1),
                                 m_thread_data_on_grid % (M2 * M1) / M2,
                                 m_thread_data_on_grid % M2,
                                 n_thread_data_on_grid % N1)}
                .Run(c_m0_m1_m2_n_thread_desc,
                     make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                     c_blk_buf_,
                     c_m0_m1_m2_n_grid_desc,
                     c_grid_buf,
                     c_m0_m1_m2_n_grid_tensor_iterator_hacks);
        }
#else
        {
            constexpr index_t M0 = CLayout.M1();
            constexpr index_t M1 = CLayout.N1();
            constexpr index_t M2 = CLayout.M0();

            constexpr auto c_m0_m1_m2_n_thread_desc =
                make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(
                    I1, I1, I1, I1, Number<M0>{}, Number<1>{}, Number<M2>{}, Number<1>{}));

            StaticBuffer<AddressSpace::Vgpr, FloatC, BlkSize> c_blk_buf_;

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

            const index_t m_thread_data_on_grid =
                m_block_data_idx_on_grid + c_thread_mtx_on_block[I0];

            const index_t n_thread_data_on_grid =
                n_block_data_idx_on_grid + c_thread_mtx_on_block[I1];

            constexpr auto c_m0_m1_m2_n_grid_tensor_iterator_hacks = CGridIteratorHacks{};

            auto c_thread_copy =
                ThreadwiseDynamicTensorSliceTransfer_v1r3<FloatC,
                                                          FloatC,
                                                          decltype(c_m0_m1_m2_n_thread_desc),
                                                          decltype(c_m0_m1_m2_n_grid_desc),
                                                          Sequence<1, 1, 1, 1, M0, 1, M2, 1>,
                                                          CThreadTransferSrcDstAccessOrder,
                                                          CThreadTransferSrcDstVectorDim,
                                                          CThreadTransferDstScalarPerVector,
                                                          CGlobalMemoryDataOperation,
                                                          1,
                                                          true>{
                    c_m0_m1_m2_n_grid_desc,
                    make_multi_index(0,
                                     0,
                                     0,
                                     0,
                                     m_thread_data_on_grid / (M2 * M1),
                                     m_thread_data_on_grid % (M2 * M1) / M2,
                                     m_thread_data_on_grid % M2,
                                     n_thread_data_on_grid)};

            auto init_copy = [&](auto c_thread_idx_) {
                constexpr auto blk_off = c_mr_nr_blk_desc.CalculateOffset(c_thread_idx_);
                c_thread_copy.Run(c_m0_m1_m2_n_thread_desc,
                                  make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                                  c_thread_buf[Number<blk_off>{}].template AsType<FloatAcc>(),
                                  c_m0_m1_m2_n_grid_desc,
                                  c_grid_buf,
                                  c_m0_m1_m2_n_grid_tensor_iterator_hacks);

                return c_thread_idx_;
            };

            auto mrepeat_plus_copy = [&](auto c_thread_idx_) {
                constexpr auto mrepeat_step_plus = make_multi_index(1, 0, 0, 0, 0, 0, 0, 0);
                c_thread_copy.MoveDstSliceWindow(c_m0_m1_m2_n_grid_desc, mrepeat_step_plus);

                constexpr auto blk_off = c_mr_nr_blk_desc.CalculateOffset(c_thread_idx_);
                c_thread_copy.Run(c_m0_m1_m2_n_thread_desc,
                                  make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                                  c_thread_buf[Number<blk_off>{}].template AsType<FloatAcc>(),
                                  c_m0_m1_m2_n_grid_desc,
                                  c_grid_buf,
                                  c_m0_m1_m2_n_grid_tensor_iterator_hacks);
            };

            auto nrepeat_plus_copy = [&](auto c_thread_idx_) {
                constexpr auto nrepeat_step_plus = make_multi_index(0, 1, 0, 0, 0, 0, 0, 0);
                c_thread_copy.MoveDstSliceWindow(c_m0_m1_m2_n_grid_desc, nrepeat_step_plus);

                constexpr auto blk_off = c_mr_nr_blk_desc.CalculateOffset(c_thread_idx_);
                c_thread_copy.Run(c_m0_m1_m2_n_thread_desc,
                                  make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                                  c_thread_buf[Number<blk_off>{}].template AsType<FloatAcc>(),
                                  c_m0_m1_m2_n_grid_desc,
                                  c_grid_buf,
                                  c_m0_m1_m2_n_grid_tensor_iterator_hacks);
            };

            auto mrepeat_minus_copy = [&](auto c_thread_idx_) {
                constexpr auto mrepeat_step_plus = make_multi_index(-1, 0, 0, 0, 0, 0, 0, 0);
                c_thread_copy.MoveDstSliceWindow(c_m0_m1_m2_n_grid_desc, mrepeat_step_plus);

                constexpr auto blk_off = c_mr_nr_blk_desc.CalculateOffset(c_thread_idx_);
                c_thread_copy.Run(c_m0_m1_m2_n_thread_desc,
                                  make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                                  c_thread_buf[Number<blk_off>{}].template AsType<FloatAcc>(),
                                  c_m0_m1_m2_n_grid_desc,
                                  c_grid_buf,
                                  c_m0_m1_m2_n_grid_tensor_iterator_hacks);
            };

            auto nrepeat_minus_copy = [&](auto c_thread_idx_) {
                constexpr auto nrepeat_step_minus = make_multi_index(0, -1, 0, 0, 0, 0, 0, 0);
                c_thread_copy.MoveDstSliceWindow(c_m0_m1_m2_n_grid_desc, nrepeat_step_minus);

                constexpr auto blk_off = c_mr_nr_blk_desc.CalculateOffset(c_thread_idx_);
                c_thread_copy.Run(c_m0_m1_m2_n_thread_desc,
                                  make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                                  c_thread_buf[Number<blk_off>{}].template AsType<FloatAcc>(),
                                  c_m0_m1_m2_n_grid_desc,
                                  c_grid_buf,
                                  c_m0_m1_m2_n_grid_tensor_iterator_hacks);
            };

            static_assert((MRepeat == 4 && NRepeat == 4) or (MRepeat == 4 && NRepeat == 2) or
                              (MRepeat == 2 && NRepeat == 4) or (MRepeat == 2 && NRepeat == 2) or
                              (MRepeat == 2 && NRepeat == 1) or (MRepeat == 1 && NRepeat == 2) or
                              (MRepeat == 1 && NRepeat == 1),
                          "wrong");

            if constexpr(MRepeat == 4 && NRepeat == 4)
            {
                init_copy(make_tuple(I0, I0));

                if constexpr(CAccessOrderMRepeatNRepeat)
                {
                    nrepeat_plus_copy(make_tuple(I0, I1));
                    nrepeat_plus_copy(make_tuple(I0, I2));
                    nrepeat_plus_copy(make_tuple(I0, I3));
                    mrepeat_plus_copy(make_tuple(I1, I3));
                    nrepeat_minus_copy(make_tuple(I1, I2));
                    nrepeat_minus_copy(make_tuple(I1, I1));
                    nrepeat_minus_copy(make_tuple(I1, I0));
                    mrepeat_plus_copy(make_tuple(I2, I0));
                    nrepeat_plus_copy(make_tuple(I2, I1));
                    nrepeat_plus_copy(make_tuple(I2, I2));
                    nrepeat_plus_copy(make_tuple(I2, I3));
                    mrepeat_plus_copy(make_tuple(I3, I3));
                    nrepeat_minus_copy(make_tuple(I3, I2));
                    nrepeat_minus_copy(make_tuple(I3, I1));
                    nrepeat_minus_copy(make_tuple(I3, I0));
                }
                else
                {
                    mrepeat_plus_copy(make_tuple(I1, I0));
                    mrepeat_plus_copy(make_tuple(I2, I0));
                    mrepeat_plus_copy(make_tuple(I3, I0));
                    nrepeat_plus_copy(make_tuple(I3, I1));
                    mrepeat_minus_copy(make_tuple(I2, I1));
                    mrepeat_minus_copy(make_tuple(I1, I1));
                    mrepeat_minus_copy(make_tuple(I0, I1));
                    nrepeat_plus_copy(make_tuple(I0, I2));
                    mrepeat_plus_copy(make_tuple(I1, I2));
                    mrepeat_plus_copy(make_tuple(I2, I2));
                    mrepeat_plus_copy(make_tuple(I3, I2));
                    nrepeat_plus_copy(make_tuple(I3, I3));
                    mrepeat_minus_copy(make_tuple(I2, I3));
                    mrepeat_minus_copy(make_tuple(I1, I3));
                    mrepeat_minus_copy(make_tuple(I0, I3));
                }
            }
            else if constexpr(MRepeat == 4 && NRepeat == 2)
            {
                init_copy(make_tuple(I0, I0));

                if constexpr(CAccessOrderMRepeatNRepeat)
                {
                    nrepeat_plus_copy(make_tuple(I0, I1));
                    mrepeat_plus_copy(make_tuple(I1, I1));
                    nrepeat_minus_copy(make_tuple(I1, I0));
                    mrepeat_plus_copy(make_tuple(I2, I0));
                    nrepeat_plus_copy(make_tuple(I2, I1));
                    mrepeat_plus_copy(make_tuple(I3, I1));
                    nrepeat_minus_copy(make_tuple(I3, I0));
                }
                else
                {
                    mrepeat_plus_copy(make_tuple(I1, I0));
                    mrepeat_plus_copy(make_tuple(I2, I0));
                    mrepeat_plus_copy(make_tuple(I3, I0));
                    nrepeat_plus_copy(make_tuple(I3, I1));
                    mrepeat_minus_copy(make_tuple(I2, I1));
                    mrepeat_minus_copy(make_tuple(I1, I1));
                    mrepeat_minus_copy(make_tuple(I0, I1));
                }
            }
            else if constexpr(MRepeat == 2 && NRepeat == 4)
            {
                init_copy(make_tuple(I0, I0));

                if constexpr(CAccessOrderMRepeatNRepeat)
                {
                    nrepeat_plus_copy(make_tuple(I0, I1));
                    nrepeat_plus_copy(make_tuple(I0, I2));
                    nrepeat_plus_copy(make_tuple(I0, I3));
                    mrepeat_plus_copy(make_tuple(I1, I3));
                    nrepeat_minus_copy(make_tuple(I1, I2));
                    nrepeat_minus_copy(make_tuple(I1, I1));
                    nrepeat_minus_copy(make_tuple(I1, I0));
                }
                else
                {
                    mrepeat_plus_copy(make_tuple(I1, I0));
                    nrepeat_plus_copy(make_tuple(I1, I1));
                    mrepeat_minus_copy(make_tuple(I0, I1));
                    nrepeat_plus_copy(make_tuple(I0, I2));
                    mrepeat_plus_copy(make_tuple(I1, I2));
                    nrepeat_plus_copy(make_tuple(I1, I3));
                    mrepeat_minus_copy(make_tuple(I0, I3));
                }
            }
            else if constexpr(MRepeat == 2 && NRepeat == 2)
            {
                init_copy(make_tuple(I0, I0));

                if constexpr(CAccessOrderMRepeatNRepeat)
                {
                    nrepeat_plus_copy(make_tuple(I0, I1));
                    mrepeat_plus_copy(make_tuple(I1, I1));
                    nrepeat_minus_copy(make_tuple(I1, I0));
                }
                else
                {
                    mrepeat_plus_copy(make_tuple(I1, I0));
                    nrepeat_plus_copy(make_tuple(I1, I1));
                    mrepeat_minus_copy(make_tuple(I0, I1));
                }
            }
            else if constexpr(MRepeat == 2 && NRepeat == 1)
            {
                init_copy(make_tuple(I0, I0));
                mrepeat_plus_copy(make_tuple(I1, I0));
            }
            else if constexpr(MRepeat == 1 && NRepeat == 2)
            {
                init_copy(make_tuple(I0, I0));
                nrepeat_plus_copy(make_tuple(I0, I1));
            }
            else if constexpr(MRepeat == 1 && NRepeat == 1)
            {
                init_copy(make_tuple(I0, I0));
            }
        }
#endif
    }
}; // namespace ck

} // namespace ck
#endif
