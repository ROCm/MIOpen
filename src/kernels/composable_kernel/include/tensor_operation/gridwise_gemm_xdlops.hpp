#ifndef CK_GRIDWISE_GEMM_XDLOPS_HPP
#define CK_GRIDWISE_GEMM_XDLOPS_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "blockwise_generic_tensor_slice_copy.hpp"
#include "threadwise_generic_tensor_slice_copy.hpp"
#include "blockwise_gemm_xdlops.hpp"

namespace ck {

template <index_t GridSize,
          index_t BlockSize,
          class Float,
          class AccFloat,
          class AGlobalDesc,
          class BGlobalDesc,
          class CGlobalDesc,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerWave,
          index_t NPerWave,
          index_t GemmDataPerReadM,
          index_t GemmDataPerReadN,
          class ABlockCopyThreadSliceLengths_K_M,
          class ABlockCopyThreadClusterLengths_K_M,
          class ABlockCopyThreadClusterArrangeOrder,
          class ABlockCopySrcAccessOrder,
          class ABlockCopyDstAccessOrder,
          index_t ABlockCopySrcVectorReadDim,
          index_t ABlockCopySrcDataPerRead,
          index_t ABlockCopyDstDataPerWrite_M,
          class BBlockCopyThreadSliceLengths_K_N,
          class BBlockCopyThreadClusterLengths_K_N,
          class BBlockCopyThreadClusterArrangeOrder,
          class BBlockCopySrcAccessOrder,
          class BBlockCopyDstAccessOrder,
          index_t BBlockCopySrcVectorReadDim,
          index_t BBlockCopySrcDataPerRead,
          index_t BBlockCopyDstDataPerWrite_N,
          InMemoryDataOperation CGlobalMemoryDataOperation>
struct GridwiseGemmTransposedANormalBNormalCXdlops_v1
{
    __device__ void Run(const Float* const __restrict__ p_a_global,
                        const Float* const __restrict__ p_b_global,
                        Float* const __restrict__ p_c_global) const
    {

        constexpr auto True = integral_constant<bool, true>{};

        constexpr auto a_k_m_global_desc = AGlobalDesc{};
        constexpr auto b_k_n_global_desc = BGlobalDesc{};
        constexpr auto c_m_n_global_desc = CGlobalDesc{};

        constexpr auto K = b_k_n_global_desc.GetLengths()[0];
        constexpr auto N = b_k_n_global_desc.GetLengths()[1];
        constexpr auto M = a_k_m_global_desc.GetLengths()[1];

        // divide block work by [M, N]
        static_assert(M % MPerBlock == 0 && N % NPerBlock == 0 && K % KPerBlock == 0,
                      "wrong! cannot divide work evenly among block");

        constexpr index_t MBlockWork = M / MPerBlock;
        constexpr index_t NBlockWork = N / NPerBlock;

        static_assert(MPerBlock % MPerWave == 0 && NPerBlock % NPerWave == 0,
                      "wrong! M/NPerBlock % M/NPerWave != 0");

        constexpr index_t MWaves = MPerBlock / MPerWave;
        constexpr index_t NWaves = NPerBlock / NPerWave;

        constexpr auto block_work_desc =
            make_cluster_descriptor(Sequence<MBlockWork, NBlockWork>{});

        const auto block_work_id = block_work_desc.CalculateClusterIndex(get_block_1d_id());

        const index_t m_block_data_on_global = block_work_id[0] * MPerBlock;
        const index_t n_block_data_on_global = block_work_id[1] * NPerBlock;

        //   LDS mem
        constexpr index_t max_align = math::lcm(BBlockCopyDstDataPerWrite_N,
                                                ABlockCopyDstDataPerWrite_M,
                                                GemmDataPerReadM,
                                                GemmDataPerReadN);

        //   LDS
        //     be careful of LDS alignment
        constexpr auto a_k_m_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<KPerBlock, MPerBlock>{}, Number<max_align>{});

        auto a_blockwise_copy =
            BlockwiseGenericTensorSliceCopy_v4<BlockSize,
                                               decltype(a_k_m_global_desc),
                                               decltype(a_k_m_block_desc),
                                               decltype(a_k_m_block_desc.GetLengths()),
                                               ABlockCopyThreadSliceLengths_K_M,
                                               ABlockCopyThreadClusterLengths_K_M,
                                               ABlockCopyThreadClusterArrangeOrder,
                                               ABlockCopySrcAccessOrder,
                                               ABlockCopyDstAccessOrder,
                                               ABlockCopySrcVectorReadDim,
                                               1,
                                               ABlockCopySrcDataPerRead,
                                               ABlockCopyDstDataPerWrite_M,
                                               AddressSpace::Global,
                                               AddressSpace::Vgpr,
                                               AddressSpace::Lds,
                                               InMemoryDataOperation::Set>(
                {0, m_block_data_on_global}, {0, 0});

        constexpr auto b_k_n_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<KPerBlock, NPerBlock>{}, Number<max_align>{});

        auto b_blockwise_copy =
            BlockwiseGenericTensorSliceCopy_v4<BlockSize,
                                               decltype(b_k_n_global_desc),
                                               decltype(b_k_n_block_desc),
                                               decltype(b_k_n_block_desc.GetLengths()),
                                               BBlockCopyThreadSliceLengths_K_N,
                                               BBlockCopyThreadClusterLengths_K_N,
                                               BBlockCopyThreadClusterArrangeOrder,
                                               BBlockCopySrcAccessOrder,
                                               BBlockCopyDstAccessOrder,
                                               BBlockCopySrcVectorReadDim,
                                               1,
                                               BBlockCopySrcDataPerRead,
                                               BBlockCopyDstDataPerWrite_N,
                                               AddressSpace::Global,
                                               AddressSpace::Vgpr,
                                               AddressSpace::Lds,
                                               InMemoryDataOperation::Set>(
                {0, n_block_data_on_global}, {0, 0});

        // GEMM definition
        // c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[KPerBlock, MPerBlock] is in LDS
        //     b_mtx[EPerBlocl, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //     register
        constexpr auto a_k_m_block_mtx_desc = make_ConstantMatrixDescriptor(a_k_m_block_desc);
        constexpr auto b_k_n_block_mtx_desc = make_ConstantMatrixDescriptor(b_k_n_block_desc);

        const auto blockwise_gemm = BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_xdlops<
            BlockSize,
            decltype(a_k_m_block_mtx_desc),
            decltype(b_k_n_block_mtx_desc),
            Float,
            MPerWave,
            NPerWave,
            MWaves,
            NWaves,
            GemmDataPerReadM,
            GemmDataPerReadN>{};

        constexpr auto c_k_thread_mtx_desc = blockwise_gemm.GetThreadMatrixCDescriptor();

        constexpr index_t a_block_space =
            math::integer_least_multiple(a_k_m_block_desc.GetElementSpace(), max_align);

        constexpr index_t b_block_space =
            math::integer_least_multiple(b_k_n_block_desc.GetElementSpace(), max_align);

        __shared__ Float p_a_block_double[2 * a_block_space];
        __shared__ Float p_b_block_double[2 * b_block_space];

        // register allocation for output
        AccFloat p_c_thread[c_k_thread_mtx_desc.GetElementSpace()];

        // zero out threadwise output
        threadwise_matrix_set_zero(c_k_thread_mtx_desc, p_c_thread);
        blockwise_gemm.XdlopsMatrixCSetZero();

        // LDS double buffer: preload data into LDS
        {
            a_blockwise_copy.Run(p_a_global, p_a_block_double);
            b_blockwise_copy.Run(p_b_global, p_b_block_double);
        }

        using b_blockwise_copy_src_step = Sequence<KPerBlock, 0>;
        using a_blockwise_copy_src_step = Sequence<KPerBlock, 0>;

        // LDS double buffer: main body
        for(index_t k_block_data_begin = 0; k_block_data_begin + 2 * KPerBlock < K;
            k_block_data_begin += 2 * KPerBlock)
        {
#pragma unroll
            for(index_t iloop = 0; iloop < 2; ++iloop)
            {
                const bool even_loop = (iloop % 2 == 0);

                Float* p_a_block_now =
                    even_loop ? p_a_block_double : p_a_block_double + a_block_space;
                Float* p_b_block_now =
                    even_loop ? p_b_block_double : p_b_block_double + b_block_space;

                Float* p_a_block_next =
                    even_loop ? p_a_block_double + a_block_space : p_a_block_double;
                Float* p_b_block_next =
                    even_loop ? p_b_block_double + b_block_space : p_b_block_double;

                Float p_a_thread_buffer[a_blockwise_copy.GetThreadBufferSize()];
                Float p_b_thread_buffer[b_blockwise_copy.GetThreadBufferSize()];

                a_blockwise_copy.MoveSrcSliceWindow(a_blockwise_copy_src_step{}, True);
                b_blockwise_copy.MoveSrcSliceWindow(b_blockwise_copy_src_step{}, True);

                __syncthreads();

                // LDS doubel buffer: load next data from device mem
                a_blockwise_copy.RunLoadThreadBuffer(p_a_global, p_a_thread_buffer);
                b_blockwise_copy.RunLoadThreadBuffer(p_b_global, p_b_thread_buffer);

                // LDS double buffer: GEMM on current data
                blockwise_gemm.Run(p_a_block_now, p_b_block_now, p_c_thread);

                // LDS double buffer: store next data to LDS
                a_blockwise_copy.RunStoreThreadBuffer(p_a_thread_buffer, p_a_block_next);
                b_blockwise_copy.RunStoreThreadBuffer(p_b_thread_buffer, p_b_block_next);
            }
        }

        // LDS double buffer: tail
        {
            constexpr bool has_two_iteration_left = (K % (2 * KPerBlock) == 0);

            if(has_two_iteration_left) // if has 2 iteration left
            {
                Float p_a_thread_buffer[a_blockwise_copy.GetThreadBufferSize()];
                Float p_b_thread_buffer[b_blockwise_copy.GetThreadBufferSize()];

                a_blockwise_copy.MoveSrcSliceWindow(a_blockwise_copy_src_step{}, True);
                b_blockwise_copy.MoveSrcSliceWindow(b_blockwise_copy_src_step{}, True);

                __syncthreads();

                // LDS double buffer: load last data from device mem
                a_blockwise_copy.RunLoadThreadBuffer(p_a_global, p_a_thread_buffer);
                b_blockwise_copy.RunLoadThreadBuffer(p_b_global, p_b_thread_buffer);

                // LDS double buffer: GEMM on 2nd-last data
                blockwise_gemm.Run(p_a_block_double, p_b_block_double, p_c_thread);

                // LDS double buffer: store last data to LDS
                a_blockwise_copy.RunStoreThreadBuffer(p_a_thread_buffer,
                                                      p_a_block_double + a_block_space);
                b_blockwise_copy.RunStoreThreadBuffer(p_b_thread_buffer,
                                                      p_b_block_double + b_block_space);

                __syncthreads();

                // LDS double buffer: GEMM on current data
                blockwise_gemm.Run(
                    p_a_block_double + a_block_space, p_b_block_double + b_block_space, p_c_thread);
            }
            else // if has 1 iteration left
            {
                __syncthreads();

                // LDS double buffer: GEMM on last data
                blockwise_gemm.Run(p_a_block_double, p_b_block_double, p_c_thread);
            }
        }
        // load data from xldop_acc_regs
        blockwise_gemm.XdlopsMatrixCRead(p_c_thread);

        // copy output: register to global memory
        {
            ///\todo inconsistent layout of xdlops and tensor
            // xdlops layout
            // M1 = num_groups;
            // M0 = group_size;
            // N1 = num_blks_per_wave;
            // N0 = num_threads_per_blks;
            constexpr auto CLayout = blockwise_gemm.GetOutputLayout();
            constexpr index_t M0   = CLayout.M1();
            constexpr index_t M1   = CLayout.N1();
            constexpr index_t M2   = CLayout.M0();

            constexpr auto c_m0_m1_m2_n_global_desc = transform_tensor_descriptor(
                c_m_n_global_desc,
                make_tuple(UnMerge<Sequence<M0, M1, M2>>{}, PassThrough<N>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}));

            //     src descriptor
            constexpr auto c_m0_m1_m2_n_thread_desc =
                make_native_tensor_descriptor_packed(Sequence<M0, 1, M2, 1>{});

            using CThreadCopySliceLengths = Sequence<M0, 1, M2, 1>;

            constexpr index_t BlkSize = CLayout.GetBlkSize();
            constexpr index_t NumBlks = CLayout.GetNumBlks();

            for(index_t i = 0; i < NumBlks; ++i)
            {
                // calculate origin of thread output tensor on global memory
                //     blockwise GEMM c matrix starting index
                const auto c_thread_mtx_on_block = blockwise_gemm.GetBeginOfThreadMatrixC(i);

                const index_t m_thread_data_on_global =
                    m_block_data_on_global + c_thread_mtx_on_block.row;

                const index_t n_thread_data_on_global =
                    n_block_data_on_global + c_thread_mtx_on_block.col;

                ThreadwiseGenericTensorSliceCopy_v4r2<decltype(c_m0_m1_m2_n_thread_desc),
                                                      decltype(c_m0_m1_m2_n_global_desc),
                                                      CThreadCopySliceLengths,
                                                      arithmetic_sequence_gen<0, 4, 1>::type,
                                                      3,
                                                      1,
                                                      1,
                                                      AddressSpace::Vgpr,
                                                      AddressSpace::Global,
                                                      CGlobalMemoryDataOperation>(
                    {0, 0, 0, 0},
                    {m_thread_data_on_global / (M2 * M1),
                     m_thread_data_on_global % (M2 * M1) / M2,
                     m_thread_data_on_global % M2,
                     n_thread_data_on_global})
                    .Run(p_c_thread + i * BlkSize, p_c_global);
            }
        }
    }
};

template <index_t GridSize,
          index_t BlockSize,
          class Float,
          class AccFloat,
          class AGlobalDesc,
          class BGlobalDesc,
          class CGlobalDesc,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerWave,
          index_t NPerWave,
          index_t GemmDataPerReadM,
          index_t GemmDataPerReadN,
          class ABlockCopyThreadSliceLengths_G_K_M,
          class ABlockCopyThreadClusterLengths_G_K_M,
          class ABlockCopyThreadClusterArrangeOrder,
          class ABlockCopySrcAccessOrder,
          class ABlockCopyDstAccessOrder,
          index_t ABlockCopySrcVectorReadDim,
          index_t ABlockCopySrcDataPerRead,
          index_t ABlockCopyDstDataPerWrite_M,
          class BBlockCopyThreadSliceLengths_G_K_N,
          class BBlockCopyThreadClusterLengths_G_K_N,
          class BBlockCopyThreadClusterArrangeOrder,
          class BBlockCopySrcAccessOrder,
          class BBlockCopyDstAccessOrder,
          index_t BBlockCopySrcVectorReadDim,
          index_t BBlockCopySrcDataPerRead,
          index_t BBlockCopyDstDataPerWrite_N,
          InMemoryDataOperation CGlobalMemoryDataOperation>
struct GridwiseBatchedGemmTransposedANormalBNormalCXdlops_v1
{
    __device__ void Run(const Float* const __restrict__ p_a_global,
                        const Float* const __restrict__ p_b_global,
                        Float* const __restrict__ p_c_global) const
    {

        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};

        constexpr auto True = integral_constant<bool, true>{};

        constexpr auto a_g_k_m_global_desc = AGlobalDesc{};
        constexpr auto b_g_k_n_global_desc = BGlobalDesc{};
        constexpr auto c_g_m_n_global_desc = CGlobalDesc{};

        constexpr auto G = b_g_k_n_global_desc.GetLengths()[0];

        constexpr auto K = b_g_k_n_global_desc.GetLengths()[1];
        constexpr auto N = b_g_k_n_global_desc.GetLengths()[2];
        constexpr auto M = a_g_k_m_global_desc.GetLengths()[2];

        // divide block work by [M, N]
        static_assert(M % MPerBlock == 0 && N % NPerBlock == 0 && K % KPerBlock == 0,
                      "wrong! cannot divide work evenly among block");

        constexpr index_t MBlockWork = M / MPerBlock;
        constexpr index_t NBlockWork = N / NPerBlock;

        static_assert(MPerBlock % MPerWave == 0 && NPerBlock % NPerWave == 0,
                      "wrong! M/NPerBlock % M/NPerWave != 0");

        constexpr index_t MWaves = MPerBlock / MPerWave;
        constexpr index_t NWaves = NPerBlock / NPerWave;

        constexpr auto block_work_desc =
            make_cluster_descriptor(Sequence<G, MBlockWork, NBlockWork>{});

        const auto block_work_id = block_work_desc.CalculateClusterIndex(get_block_1d_id());

        const index_t group_id               = block_work_id[0];
        const index_t m_block_data_on_global = block_work_id[1] * MPerBlock;
        const index_t n_block_data_on_global = block_work_id[2] * NPerBlock;

        //   LDS mem
        constexpr index_t max_align = math::lcm(BBlockCopyDstDataPerWrite_N,
                                                ABlockCopyDstDataPerWrite_M,
                                                GemmDataPerReadM,
                                                GemmDataPerReadN);

        //   LDS
        //     be careful of LDS alignment
        constexpr auto a_g_k_m_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<1, KPerBlock, MPerBlock>{}, Number<max_align>{});

        auto a_blockwise_copy =
            BlockwiseGenericTensorSliceCopy_v4<BlockSize,
                                               decltype(a_g_k_m_global_desc),
                                               decltype(a_g_k_m_block_desc),
                                               decltype(a_g_k_m_block_desc.GetLengths()),
                                               ABlockCopyThreadSliceLengths_G_K_M,
                                               ABlockCopyThreadClusterLengths_G_K_M,
                                               ABlockCopyThreadClusterArrangeOrder,
                                               ABlockCopySrcAccessOrder,
                                               ABlockCopyDstAccessOrder,
                                               ABlockCopySrcVectorReadDim,
                                               2,
                                               ABlockCopySrcDataPerRead,
                                               ABlockCopyDstDataPerWrite_M,
                                               AddressSpace::Global,
                                               AddressSpace::Vgpr,
                                               AddressSpace::Lds,
                                               InMemoryDataOperation::Set>(
                {group_id, 0, m_block_data_on_global}, {0, 0, 0});

        constexpr auto b_g_k_n_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<1, KPerBlock, NPerBlock>{}, Number<max_align>{});

        auto b_blockwise_copy =
            BlockwiseGenericTensorSliceCopy_v4<BlockSize,
                                               decltype(b_g_k_n_global_desc),
                                               decltype(b_g_k_n_block_desc),
                                               decltype(b_g_k_n_block_desc.GetLengths()),
                                               BBlockCopyThreadSliceLengths_G_K_N,
                                               BBlockCopyThreadClusterLengths_G_K_N,
                                               BBlockCopyThreadClusterArrangeOrder,
                                               BBlockCopySrcAccessOrder,
                                               BBlockCopyDstAccessOrder,
                                               BBlockCopySrcVectorReadDim,
                                               2,
                                               BBlockCopySrcDataPerRead,
                                               BBlockCopyDstDataPerWrite_N,
                                               AddressSpace::Global,
                                               AddressSpace::Vgpr,
                                               AddressSpace::Lds,
                                               InMemoryDataOperation::Set>(
                {group_id, 0, n_block_data_on_global}, {0, 0, 0});

        // GEMM definition
        // c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[KPerBlock, MPerBlock] is in LDS
        //     b_mtx[EPerBlocl, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //     register
        constexpr auto a_k_m_block_mtx_desc = make_ConstantMatrixDescriptor_packed(
            a_g_k_m_block_desc.GetLength(I1), a_g_k_m_block_desc.GetLength(I2));
        constexpr auto b_k_n_block_mtx_desc = make_ConstantMatrixDescriptor_packed(
            b_g_k_n_block_desc.GetLength(I1), b_g_k_n_block_desc.GetLength(I2));

        const auto blockwise_gemm = BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_xdlops<
            BlockSize,
            decltype(a_k_m_block_mtx_desc),
            decltype(b_k_n_block_mtx_desc),
            Float,
            MPerWave,
            NPerWave,
            MWaves,
            NWaves,
            GemmDataPerReadM,
            GemmDataPerReadN>{};

        constexpr auto c_k_thread_mtx_desc = blockwise_gemm.GetThreadMatrixCDescriptor();

        constexpr index_t a_block_space =
            math::integer_least_multiple(a_g_k_m_block_desc.GetElementSpace(), max_align);

        constexpr index_t b_block_space =
            math::integer_least_multiple(b_g_k_n_block_desc.GetElementSpace(), max_align);

        __shared__ Float p_a_block_double[2 * a_block_space];
        __shared__ Float p_b_block_double[2 * b_block_space];

        // register allocation for output
        AccFloat p_c_thread[c_k_thread_mtx_desc.GetElementSpace()];

        // zero out threadwise output
        threadwise_matrix_set_zero(c_k_thread_mtx_desc, p_c_thread);
        blockwise_gemm.XdlopsMatrixCSetZero();

        // LDS double buffer: preload data into LDS
        {
            a_blockwise_copy.Run(p_a_global, p_a_block_double);
            b_blockwise_copy.Run(p_b_global, p_b_block_double);
        }

        using b_blockwise_copy_src_step = Sequence<0, KPerBlock, 0>;
        using a_blockwise_copy_src_step = Sequence<0, KPerBlock, 0>;

        // LDS double buffer: main body
        for(index_t k_block_data_begin = 0; k_block_data_begin + 2 * KPerBlock < K;
            k_block_data_begin += 2 * KPerBlock)
        {
#pragma unroll
            for(index_t iloop = 0; iloop < 2; ++iloop)
            {
                const bool even_loop = (iloop % 2 == 0);

                Float* p_a_block_now =
                    even_loop ? p_a_block_double : p_a_block_double + a_block_space;
                Float* p_b_block_now =
                    even_loop ? p_b_block_double : p_b_block_double + b_block_space;

                Float* p_a_block_next =
                    even_loop ? p_a_block_double + a_block_space : p_a_block_double;
                Float* p_b_block_next =
                    even_loop ? p_b_block_double + b_block_space : p_b_block_double;

                Float p_a_thread_buffer[a_blockwise_copy.GetThreadBufferSize()];
                Float p_b_thread_buffer[b_blockwise_copy.GetThreadBufferSize()];

                a_blockwise_copy.MoveSrcSliceWindow(a_blockwise_copy_src_step{}, True);
                b_blockwise_copy.MoveSrcSliceWindow(b_blockwise_copy_src_step{}, True);

                __syncthreads();

                // LDS doubel buffer: load next data from device mem
                a_blockwise_copy.RunLoadThreadBuffer(p_a_global, p_a_thread_buffer);
                b_blockwise_copy.RunLoadThreadBuffer(p_b_global, p_b_thread_buffer);

                // LDS double buffer: GEMM on current data
                blockwise_gemm.Run(p_a_block_now, p_b_block_now, p_c_thread);

                // LDS double buffer: store next data to LDS
                a_blockwise_copy.RunStoreThreadBuffer(p_a_thread_buffer, p_a_block_next);
                b_blockwise_copy.RunStoreThreadBuffer(p_b_thread_buffer, p_b_block_next);
            }
        }

        // LDS double buffer: tail
        {
            constexpr bool has_two_iteration_left = (K % (2 * KPerBlock) == 0);

            if(has_two_iteration_left) // if has 2 iteration left
            {
                Float p_a_thread_buffer[a_blockwise_copy.GetThreadBufferSize()];
                Float p_b_thread_buffer[b_blockwise_copy.GetThreadBufferSize()];

                a_blockwise_copy.MoveSrcSliceWindow(a_blockwise_copy_src_step{}, True);
                b_blockwise_copy.MoveSrcSliceWindow(b_blockwise_copy_src_step{}, True);

                __syncthreads();

                // LDS double buffer: load last data from device mem
                a_blockwise_copy.RunLoadThreadBuffer(p_a_global, p_a_thread_buffer);
                b_blockwise_copy.RunLoadThreadBuffer(p_b_global, p_b_thread_buffer);

                // LDS double buffer: GEMM on 2nd-last data
                blockwise_gemm.Run(p_a_block_double, p_b_block_double, p_c_thread);

                // LDS double buffer: store last data to LDS
                a_blockwise_copy.RunStoreThreadBuffer(p_a_thread_buffer,
                                                      p_a_block_double + a_block_space);
                b_blockwise_copy.RunStoreThreadBuffer(p_b_thread_buffer,
                                                      p_b_block_double + b_block_space);

                __syncthreads();

                // LDS double buffer: GEMM on current data
                blockwise_gemm.Run(
                    p_a_block_double + a_block_space, p_b_block_double + b_block_space, p_c_thread);
            }
            else // if has 1 iteration left
            {
                __syncthreads();

                // LDS double buffer: GEMM on last data
                blockwise_gemm.Run(p_a_block_double, p_b_block_double, p_c_thread);
            }
        }
        // load data from xldop_acc_regs
        blockwise_gemm.XdlopsMatrixCRead(p_c_thread);

        // copy output: register to global memory
        {
            ///\todo inconsistent layout of xdlops and tensor
            // xdlops layout
            // M1 = num_groups;
            // M0 = group_size;
            // N1 = num_blks_per_wave;
            // N0 = num_threads_per_blks;
            constexpr auto CLayout = blockwise_gemm.GetOutputLayout();
            constexpr index_t M0   = CLayout.M1();
            constexpr index_t M1   = CLayout.N1();
            constexpr index_t M2   = CLayout.M0();

            constexpr auto c_g_m0_m1_m2_n_global_desc = transform_tensor_descriptor(
                c_g_m_n_global_desc,
                make_tuple(PassThrough<G>{}, UnMerge<Sequence<M0, M1, M2>>{}, PassThrough<N>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2, 3>{}, Sequence<4>{}));

            //     src descriptor
            constexpr auto c_g_m0_m1_m2_n_thread_desc =
                make_native_tensor_descriptor_packed(Sequence<1, M0, 1, M2, 1>{});

            using CThreadCopySliceLengths = Sequence<1, M0, 1, M2, 1>;

            constexpr index_t BlkSize = CLayout.GetBlkSize();
            constexpr index_t NumBlks = CLayout.GetNumBlks();

            for(index_t i = 0; i < NumBlks; ++i)
            {
                // calculate origin of thread output tensor on global memory
                //     blockwise GEMM c matrix starting index
                const auto c_thread_mtx_on_block = blockwise_gemm.GetBeginOfThreadMatrixC(i);

                const index_t m_thread_data_on_global =
                    m_block_data_on_global + c_thread_mtx_on_block.row;

                const index_t n_thread_data_on_global =
                    n_block_data_on_global + c_thread_mtx_on_block.col;

                ThreadwiseGenericTensorSliceCopy_v4r2<decltype(c_g_m0_m1_m2_n_thread_desc),
                                                      decltype(c_g_m0_m1_m2_n_global_desc),
                                                      CThreadCopySliceLengths,
                                                      arithmetic_sequence_gen<0, 5, 1>::type,
                                                      4,
                                                      1,
                                                      1,
                                                      AddressSpace::Vgpr,
                                                      AddressSpace::Global,
                                                      CGlobalMemoryDataOperation>(
                    {0, 0, 0, 0, 0},
                    {group_id,
                     m_thread_data_on_global / (M2 * M1),
                     m_thread_data_on_global % (M2 * M1) / M2,
                     m_thread_data_on_global % M2,
                     n_thread_data_on_global})
                    .Run(p_c_thread + i * BlkSize, p_c_global);
            }
        }
    }
};

} // namespace ck
#endif
