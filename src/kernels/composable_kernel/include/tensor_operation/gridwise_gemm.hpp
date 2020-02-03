#ifndef CK_GRIDWISE_GEMM_HPP
#define CK_GRIDWISE_GEMM_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "blockwise_generic_tensor_slice_copy.hpp"
#include "threadwise_generic_tensor_slice_copy.hpp"
#include "blockwise_gemm.hpp"

namespace ck {

template <index_t GridSize,
          index_t BlockSize,
          typename Float,
          typename AccFloat,
          typename AGlobalDesc,
          typename BGlobalDesc,
          typename CGlobalDesc,
          InMemoryDataOperation CGlobalMemoryDataOperation,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerThreadSubC,
          index_t NPerThreadSubC,
          index_t MLevel0Cluster,
          index_t NLevel0Cluster,
          index_t MLevel1Cluster,
          index_t NLevel1Cluster,
          index_t KPerThreadLoop,
          index_t ThreadGemmDataPerReadM,
          index_t ThreadGemmDataPerReadN,
          typename ABlockCopyThreadSliceLengths_K_M,
          typename ABlockCopyThreadClusterLengths_K_M,
          typename ABlockCopyThreadClusterArrangeOrder,
          typename ABlockCopySrcAccessOrder,
          index_t ABlockCopySrcVectorReadDim,
          index_t ABlockCopySrcDataPerRead,
          index_t ABlockCopyDstDataPerWrite_M,
          typename BBlockCopyThreadSliceLengths_K_N,
          typename BBlockCopyThreadClusterLengths_K_N,
          typename BBlockCopyThreadClusterArrangeOrder,
          typename BBlockCopySrcAccessOrder,
          index_t BBlockCopySrcVectorReadDim,
          index_t BBlockCopySrcDataPerRead,
          index_t BBlockCopyDstDataPerWrite_N,
          typename CThreadCopySrcDstAccessOrder,
          index_t CThreadCopySrcDstVectorReadWriteDim,
          index_t CThreadCopyDstDataPerWrite>
struct GridwiseGemmTransposedANormalBNormalC_v1
{
    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        constexpr index_t max_lds_align = math::lcm(ABlockCopyDstDataPerWrite_M,
                                                    BBlockCopyDstDataPerWrite_N,
                                                    ThreadGemmDataPerReadM,
                                                    ThreadGemmDataPerReadN);

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_k_m_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<KPerBlock, MPerBlock>{}, Number<max_lds_align>{});

        // B matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto b_k_n_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<KPerBlock, NPerBlock>{}, Number<max_lds_align>{});

        // LDS allocation for A and B: be careful of alignment
        constexpr index_t a_block_space =
            math::integer_least_multiple(a_k_m_block_desc.GetElementSpace(), max_lds_align);

        constexpr index_t b_block_space =
            math::integer_least_multiple(b_k_n_block_desc.GetElementSpace(), max_lds_align);

        return 2 * (a_block_space + b_block_space) * sizeof(Float);
    }

    __device__ void Run(const Float* __restrict__ p_a_global,
                        const Float* __restrict__ p_b_global,
                        Float* __restrict__ p_c_global,
                        Float* __restrict__ p_shared_block) const
    {
        constexpr auto True = integral_constant<bool, true>{};

        constexpr auto a_k_m_global_desc = AGlobalDesc{};
        constexpr auto b_k_n_global_desc = BGlobalDesc{};
        constexpr auto c_m_n_global_desc = CGlobalDesc{};

        constexpr auto K = a_k_m_global_desc.GetLengths()[0];
        constexpr auto M = a_k_m_global_desc.GetLengths()[1];
        constexpr auto N = b_k_n_global_desc.GetLengths()[1];

        // don't do anything if K == 0
        if(K == 0)
        {
            return;
        }

        // lds max alignment
        constexpr index_t max_lds_align = math::lcm(ABlockCopyDstDataPerWrite_M,
                                                    BBlockCopyDstDataPerWrite_N,
                                                    ThreadGemmDataPerReadM,
                                                    ThreadGemmDataPerReadN);

        // divide block work by [M, N]
        static_assert(M % MPerBlock == 0 && N % NPerBlock == 0 && K % KPerBlock == 0,
                      "wrong! cannot divide work evenly among block");

        constexpr index_t MBlockWork = M / MPerBlock;
        constexpr index_t NBlockWork = N / NPerBlock;

        constexpr auto block_work_desc =
            make_cluster_descriptor(Sequence<MBlockWork, NBlockWork>{});

        const auto block_work_id = block_work_desc.CalculateClusterIndex(get_block_1d_id());

        const index_t m_block_data_on_global = block_work_id[0] * MPerBlock;
        const index_t n_block_data_on_global = block_work_id[1] * NPerBlock;

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_k_m_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<KPerBlock, MPerBlock>{}, Number<max_lds_align>{});

        // A matrix blockwise copy
        auto a_blockwise_copy =
            BlockwiseGenericTensorSliceCopy_v4<BlockSize,
                                               decltype(a_k_m_global_desc),
                                               decltype(a_k_m_block_desc),
                                               decltype(a_k_m_block_desc.GetLengths()),
                                               ABlockCopyThreadSliceLengths_K_M,
                                               ABlockCopyThreadClusterLengths_K_M,
                                               ABlockCopyThreadClusterArrangeOrder,
                                               ABlockCopySrcAccessOrder,
                                               Sequence<0, 1>,
                                               ABlockCopySrcVectorReadDim,
                                               1,
                                               ABlockCopySrcDataPerRead,
                                               ABlockCopyDstDataPerWrite_M,
                                               AddressSpace::Global,
                                               AddressSpace::Vgpr,
                                               AddressSpace::Lds,
                                               InMemoryDataOperation::Set>(
                {0, m_block_data_on_global}, {0, 0});

        // B matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto b_k_n_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<KPerBlock, NPerBlock>{}, Number<max_lds_align>{});

        // B matrix blockwise copy
        auto b_blockwise_copy =
            BlockwiseGenericTensorSliceCopy_v4<BlockSize,
                                               decltype(b_k_n_global_desc),
                                               decltype(b_k_n_block_desc),
                                               decltype(b_k_n_block_desc.GetLengths()),
                                               BBlockCopyThreadSliceLengths_K_N,
                                               BBlockCopyThreadClusterLengths_K_N,
                                               BBlockCopyThreadClusterArrangeOrder,
                                               BBlockCopySrcAccessOrder,
                                               Sequence<0, 1>,
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
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[KPerBlock, MPerBlock] is in LDS
        //     b_mtx[KPerBlocl, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //       register
        constexpr auto a_k_m_block_mtx_desc = make_ConstantMatrixDescriptor(a_k_m_block_desc);
        constexpr auto b_k_n_block_mtx_desc = make_ConstantMatrixDescriptor(b_k_n_block_desc);

        // sanity check
        static_assert(MPerBlock % (MPerThreadSubC * MLevel0Cluster * MLevel1Cluster) == 0 &&
                          NPerBlock % (NPerThreadSubC * NLevel0Cluster * NLevel1Cluster) == 0,
                      "wrong!");

        constexpr index_t GemmMRepeat =
            MPerBlock / (MPerThreadSubC * MLevel0Cluster * MLevel1Cluster);

        constexpr index_t GemmNRepeat =
            NPerBlock / (NPerThreadSubC * NLevel0Cluster * NLevel1Cluster);

        // c_thread_mtx definition: this is a mess
        // TODO:: more elegent way of defining c_thread_mtx
        constexpr auto c_m0m1_n0n1_thread_mtx_desc = make_ConstantMatrixDescriptor_packed(
            Number<GemmMRepeat * MPerThreadSubC>{}, Number<GemmNRepeat * NPerThreadSubC>{});

        const auto blockwise_gemm = BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_v2<
            BlockSize,
            decltype(a_k_m_block_mtx_desc),
            decltype(b_k_n_block_mtx_desc),
            decltype(c_m0m1_n0n1_thread_mtx_desc),
            MPerThreadSubC,
            NPerThreadSubC,
            MLevel0Cluster,
            NLevel0Cluster,
            MLevel1Cluster,
            NLevel1Cluster,
            KPerThreadLoop,
            ThreadGemmDataPerReadM,
            ThreadGemmDataPerReadN>{};

        // LDS allocation for A and B: be careful of alignment
        constexpr index_t a_block_space =
            math::integer_least_multiple(a_k_m_block_desc.GetElementSpace(), max_lds_align);

        constexpr index_t b_block_space =
            math::integer_least_multiple(b_k_n_block_desc.GetElementSpace(), max_lds_align);

        Float* p_a_block_double = p_shared_block;
        Float* p_b_block_double = p_shared_block + 2 * a_block_space;

        // register allocation for output
        AccFloat p_c_thread[c_m0m1_n0n1_thread_mtx_desc.GetElementSpace()];

        // zero out threadwise output
        threadwise_matrix_set_zero(c_m0m1_n0n1_thread_mtx_desc, p_c_thread);

        // LDS double buffer: preload data into LDS
        {
            a_blockwise_copy.Run(p_a_global, p_a_block_double);
            b_blockwise_copy.Run(p_b_global, p_b_block_double);
        }

        constexpr auto a_block_slice_copy_steps = Sequence<KPerBlock, 0>{};
        constexpr auto b_block_slice_copy_steps = Sequence<KPerBlock, 0>{};

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

                a_blockwise_copy.MoveSrcSliceWindow(a_block_slice_copy_steps, True);
                b_blockwise_copy.MoveSrcSliceWindow(b_block_slice_copy_steps, True);

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

                a_blockwise_copy.MoveSrcSliceWindow(a_block_slice_copy_steps, True);
                b_blockwise_copy.MoveSrcSliceWindow(b_block_slice_copy_steps, True);

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

                // LDS double buffer: GEMM on last data
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

        // input: register to global memory
        {
            constexpr index_t M1 = MPerThreadSubC * MLevel0Cluster * MLevel1Cluster;
            constexpr index_t M0 = M / M1;

            constexpr index_t N1 = NPerThreadSubC * NLevel0Cluster * NLevel1Cluster;
            constexpr index_t N0 = N / N1;

            // define input tensor descriptor for threadwise copy
            //     thread input tensor, src of threadwise copy
            constexpr auto c_m0_m1_n0_n1_thread_desc = make_native_tensor_descriptor_packed(
                Sequence<GemmMRepeat, MPerThreadSubC, GemmNRepeat, NPerThreadSubC>{});

            constexpr auto c_m0_m1_n0_n1_global_desc = transform_tensor_descriptor(
                c_m_n_global_desc,
                make_tuple(UnMerge<Sequence<M0, M1>>{}, UnMerge<Sequence<N0, N1>>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

            // calculate origin of thread input tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                blockwise_gemm.GetBeginOfThreadMatrixC(get_thread_local_1d_id());

            const index_t m_thread_data_on_global =
                m_block_data_on_global + c_thread_mtx_on_block.row;

            const index_t n_thread_data_on_global =
                n_block_data_on_global + c_thread_mtx_on_block.col;

            ThreadwiseGenericTensorSliceCopy_v4r2<decltype(c_m0_m1_n0_n1_thread_desc),
                                                  decltype(c_m0_m1_n0_n1_global_desc),
                                                  decltype(c_m0_m1_n0_n1_thread_desc.GetLengths()),
                                                  CThreadCopySrcDstAccessOrder,
                                                  CThreadCopySrcDstVectorReadWriteDim,
                                                  1,
                                                  CThreadCopyDstDataPerWrite,
                                                  AddressSpace::Vgpr,
                                                  AddressSpace::Global,
                                                  CGlobalMemoryDataOperation>(
                {0, 0, 0, 0},
                {m_thread_data_on_global / M1,
                 m_thread_data_on_global % M1,
                 n_thread_data_on_global / N1,
                 n_thread_data_on_global % N1})
                .Run(p_c_thread, p_c_global);
        }
    }

    __device__ void Run(const Float* __restrict__ p_a_global,
                        const Float* __restrict__ p_b_global,
                        Float* __restrict__ p_c_global) const
    {
        constexpr index_t shared_block_size = GetSharedMemoryNumberOfByte() / sizeof(Float);

        __shared__ Float p_shared_block[shared_block_size];

        Run(p_a_global, p_b_global, p_c_global, p_shared_block);
    }
};

} // namespace ck
#endif
