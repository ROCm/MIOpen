#ifndef CK_GRIDWISE_GEMM_FP16_BFP16_HPP
#define CK_GRIDWISE_GEMM_FP16_BFP16_HPP

#include "static_kernel_common_header.hpp"
#include "static_kernel_tensor_descriptor.hpp"
#include "static_kernel_tensor_descriptor_helper.hpp"
#include "static_kernel_ConstantMatrixDescriptor.hpp"
#include "static_kernel_blockwise_generic_tensor_slice_copy.hpp"
#include "static_kernel_threadwise_generic_tensor_slice_copy.hpp"
#include "static_kernel_blockwise_gemm.hpp"

namespace ck {

template <index_t GridSize,
          index_t BlockSize,
          class ABFloat,
          class AccFloat,
          class CFloat,
          typename AGlobalDesc,
          typename BGlobalDesc,
          typename CGlobalDesc,
          InMemoryDataOperation CGlobalMemoryDataOperation,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t KPACK,
          index_t MPerThread,
          index_t NPerThread,
          index_t KPerThread,
          index_t MLevel0Cluster,
          index_t NLevel0Cluster,
          index_t MLevel1Cluster,
          index_t NLevel1Cluster,
          index_t ThreadGemmAThreadCopySrcDataPerRead_M,
          index_t ThreadGemmBThreadCopySrcDataPerRead_N,
          typename ABlockCopyThreadSliceLengths_K_M_KPACK,
          typename ABlockCopyThreadClusterLengths_K_M_KPACK,
          typename ABlockCopyThreadClusterArrangeOrder,
          typename ABlockCopySrcAccessOrder,
          index_t ABlockCopySrcVectorReadDim,
          index_t ABlockCopySrcDataPerRead,
          index_t ABlockCopyDstDataPerWrite_KPACK,
          typename BBlockCopyThreadSliceLengths_K_N_KPACK,
          typename BBlockCopyThreadClusterLengths_K_N_KPACK,
          typename BBlockCopyThreadClusterArrangeOrder,
          typename BBlockCopySrcAccessOrder,
          index_t BBlockCopySrcVectorReadDim,
          index_t BBlockCopySrcDataPerRead,
          index_t BBlockCopyDstDataPerWrite_KPACK,
          typename CThreadCopySrcDstAccessOrder,
          index_t CThreadCopySrcDstVectorReadWriteDim,
          index_t CThreadCopyDstDataPerWrite,
          index_t ABlockCopySrcDataStride = 1,
          index_t BBlockCopySrcDataStride = 1>
struct GridwiseGemmTransposedANormalBNormalCFp16Bfp16_v1
{
    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        constexpr index_t max_lds_align = math::lcm(ABlockCopyDstDataPerWrite_KPACK,
                                                    BBlockCopyDstDataPerWrite_KPACK,
                                                    ThreadGemmAThreadCopySrcDataPerRead_M,
                                                    ThreadGemmBThreadCopySrcDataPerRead_N);

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_k_m_kpack_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<KPerBlock, MPerBlock, KPACK>{}, Number<max_lds_align>{});

        // B matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto b_k_n_kpack_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<KPerBlock, NPerBlock, KPACK>{}, Number<max_lds_align>{});

        // LDS allocation for A and B: be careful of alignment
        constexpr index_t a_block_space =
            math::integer_least_multiple(a_k_m_kpack_block_desc.GetElementSpace(), max_lds_align);

        constexpr index_t b_block_space =
            math::integer_least_multiple(b_k_n_kpack_block_desc.GetElementSpace(), max_lds_align);

        return 2 * (a_block_space + b_block_space) * sizeof(ABFloat);
    }

    __device__ void Run(const ABFloat* __restrict__ p_a_global,
                        const ABFloat* __restrict__ p_b_global,
                        CFloat* __restrict__ p_c_global,
                        ABFloat* __restrict__ p_shared_block) const
    {
        constexpr auto True = integral_constant<bool, true>{};

        constexpr auto a_k_m_kpack_global_desc = AGlobalDesc{};
        constexpr auto b_k_n_kpack_global_desc = BGlobalDesc{};
        constexpr auto c_m_n_global_desc       = CGlobalDesc{};

        constexpr auto K = a_k_m_kpack_global_desc.GetLengths()[0];
        constexpr auto M = a_k_m_kpack_global_desc.GetLengths()[1];
        constexpr auto N = b_k_n_kpack_global_desc.GetLengths()[1];

        // don't do anything if K == 0
        if(K == 0)
        {
            return;
        }

        // lds max alignment
        constexpr index_t max_lds_align = math::lcm(ABlockCopyDstDataPerWrite_KPACK,
                                                    BBlockCopyDstDataPerWrite_KPACK,
                                                    KPACK * ThreadGemmAThreadCopySrcDataPerRead_M,
                                                    KPACK * ThreadGemmBThreadCopySrcDataPerRead_N);

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
        constexpr auto a_k_m_kpack_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<KPerBlock, MPerBlock, KPACK>{}, Number<max_lds_align>{});

        // A matrix blockwise copy
        auto a_blockwise_copy =
            BlockwiseGenericTensorSliceCopy_v4<BlockSize,
                                               decltype(a_k_m_kpack_global_desc),
                                               decltype(a_k_m_kpack_block_desc),
                                               decltype(a_k_m_kpack_block_desc.GetLengths()),
                                               ABlockCopyThreadSliceLengths_K_M_KPACK,
                                               ABlockCopyThreadClusterLengths_K_M_KPACK,
                                               ABlockCopyThreadClusterArrangeOrder,
                                               ABlockCopySrcAccessOrder,
                                               Sequence<0, 1, 2>,
                                               ABlockCopySrcVectorReadDim,
                                               2,
                                               ABlockCopySrcDataPerRead,
                                               ABlockCopyDstDataPerWrite_KPACK,
                                               AddressSpace::Generic,
                                               AddressSpace::Vgpr,
                                               AddressSpace::Lds,
                                               InMemoryDataOperation::Set,
                                               ABlockCopySrcDataStride>(
                {0, m_block_data_on_global, 0}, {0, 0, 0});

        // B matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto b_k_n_kpack_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<KPerBlock, NPerBlock, KPACK>{}, Number<max_lds_align>{});

        // B matrix blockwise copy
        auto b_blockwise_copy =
            BlockwiseGenericTensorSliceCopy_v4<BlockSize,
                                               decltype(b_k_n_kpack_global_desc),
                                               decltype(b_k_n_kpack_block_desc),
                                               decltype(b_k_n_kpack_block_desc.GetLengths()),
                                               BBlockCopyThreadSliceLengths_K_N_KPACK,
                                               BBlockCopyThreadClusterLengths_K_N_KPACK,
                                               BBlockCopyThreadClusterArrangeOrder,
                                               BBlockCopySrcAccessOrder,
                                               Sequence<0, 1, 2>,
                                               BBlockCopySrcVectorReadDim,
                                               2,
                                               BBlockCopySrcDataPerRead,
                                               BBlockCopyDstDataPerWrite_KPACK,
                                               AddressSpace::Generic,
                                               AddressSpace::Vgpr,
                                               AddressSpace::Lds,
                                               InMemoryDataOperation::Set,
                                               BBlockCopySrcDataStride>(
                {0, n_block_data_on_global, 0}, {0, 0, 0});

        // GEMM definition
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[KPerBlock, MPerBlock] is in LDS
        //     b_mtx[KPerBlocl, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //       register
        constexpr auto a_k_m_block_mtx_desc =
            make_ConstantMatrixDescriptor_packed(Number<KPerBlock>{}, Number<MPerBlock>{});
        constexpr auto b_k_n_block_mtx_desc =
            make_ConstantMatrixDescriptor_packed(Number<KPerBlock>{}, Number<NPerBlock>{});

        // sanity check
        static_assert(MPerBlock % (MPerThread * MLevel0Cluster * MLevel1Cluster) == 0 &&
                          NPerBlock % (NPerThread * NLevel0Cluster * NLevel1Cluster) == 0,
                      "wrong!");

        constexpr index_t GemmMRepeat = MPerBlock / (MPerThread * MLevel0Cluster * MLevel1Cluster);
        constexpr index_t GemmNRepeat = NPerBlock / (NPerThread * NLevel0Cluster * NLevel1Cluster);

        // c_thread_mtx definition: this is a mess
        // TODO:: more elegent way of defining c_thread_mtx
        constexpr auto c_m0m1_n0n1_thread_mtx_desc = make_ConstantMatrixDescriptor_packed(
            Number<GemmMRepeat * MPerThread>{}, Number<GemmNRepeat * NPerThread>{});

        const auto blockwise_gemm = BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_v2<
            BlockSize,
            decltype(a_k_m_block_mtx_desc),
            decltype(b_k_n_block_mtx_desc),
            decltype(c_m0m1_n0n1_thread_mtx_desc),
            MPerThread,
            NPerThread,
            MLevel0Cluster,
            NLevel0Cluster,
            MLevel1Cluster,
            NLevel1Cluster,
            KPerThread,
            ThreadGemmAThreadCopySrcDataPerRead_M,
            ThreadGemmBThreadCopySrcDataPerRead_N>{};

        // LDS allocation for A and B: be careful of alignment
        constexpr index_t a_block_space =
            math::integer_least_multiple(a_k_m_kpack_block_desc.GetElementSpace(), max_lds_align);

        constexpr index_t b_block_space =
            math::integer_least_multiple(b_k_n_kpack_block_desc.GetElementSpace(), max_lds_align);

        ABFloat* p_a_block_double = p_shared_block;
        ABFloat* p_b_block_double = p_shared_block + 2 * a_block_space;

        // register allocation for output
        AccFloat p_c_thread[c_m0m1_n0n1_thread_mtx_desc.GetElementSpace()];

        // zero out threadwise output
        threadwise_matrix_set_zero(c_m0m1_n0n1_thread_mtx_desc, p_c_thread);

        // LDS double buffer: preload data into LDS
        {
            a_blockwise_copy.Run(p_a_global, p_a_block_double);
            b_blockwise_copy.Run(p_b_global, p_b_block_double);
        }

        constexpr auto a_block_slice_copy_steps = Sequence<KPerBlock, 0, 0>{};
        constexpr auto b_block_slice_copy_steps = Sequence<KPerBlock, 0, 0>{};

        // LDS double buffer: main body
        for(index_t k_block_data_begin = 0; k_block_data_begin + 2 * KPerBlock < K;
            k_block_data_begin += 2 * KPerBlock)
        {
#pragma unroll 2
            for(index_t iloop = 0; iloop < 2; ++iloop)
            {
                const bool even_loop = (iloop % 2 == 0);

                ABFloat* p_a_block_now =
                    even_loop ? p_a_block_double : p_a_block_double + a_block_space;
                ABFloat* p_b_block_now =
                    even_loop ? p_b_block_double : p_b_block_double + b_block_space;

                ABFloat* p_a_block_next =
                    even_loop ? p_a_block_double + a_block_space : p_a_block_double;
                ABFloat* p_b_block_next =
                    even_loop ? p_b_block_double + b_block_space : p_b_block_double;

                ABFloat p_a_thread_buffer[a_blockwise_copy.GetThreadBufferSize()];
                ABFloat p_b_thread_buffer[b_blockwise_copy.GetThreadBufferSize()];

                a_blockwise_copy.MoveSrcSliceWindow(a_block_slice_copy_steps, True);
                b_blockwise_copy.MoveSrcSliceWindow(b_block_slice_copy_steps, True);

                __syncthreads();

                // LDS doubel buffer: load next data from device mem
                a_blockwise_copy.RunLoadThreadBuffer(p_a_global, p_a_thread_buffer);
                b_blockwise_copy.RunLoadThreadBuffer(p_b_global, p_b_thread_buffer);

                // LDS double buffer: GEMM on current data
                // Vectorize the pointer to match with how fp16/bfloat16 datatypes are
                // processed in gemm operation. fp16 type packs 4 fp16 values while
                // bfloat16 packs 2 bfloat16 values. Since gemm's matrix A and B
                // 2D indexes are computed with vectorized value in mind (e.g. float, half2, half4),
                // we recast datatype from a single fp16 to 4 packed fp16/2 packed bfloat16
                // respectively.
                const typename vector_type<ABFloat, KPACK>::MemoryType* p_a_block_vec =
                    reinterpret_cast<const typename vector_type<ABFloat, KPACK>::MemoryType*>(
                        p_a_block_now);
                const typename vector_type<ABFloat, KPACK>::MemoryType* p_b_block_vec =
                    reinterpret_cast<const typename vector_type<ABFloat, KPACK>::MemoryType*>(
                        p_b_block_now);
                blockwise_gemm.Run(p_a_block_vec, p_b_block_vec, p_c_thread);

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
                ABFloat p_a_thread_buffer[a_blockwise_copy.GetThreadBufferSize()];
                ABFloat p_b_thread_buffer[b_blockwise_copy.GetThreadBufferSize()];

                a_blockwise_copy.MoveSrcSliceWindow(a_block_slice_copy_steps, True);
                b_blockwise_copy.MoveSrcSliceWindow(b_block_slice_copy_steps, True);

                __syncthreads();

                // LDS double buffer: load last data from device mem
                a_blockwise_copy.RunLoadThreadBuffer(p_a_global, p_a_thread_buffer);
                b_blockwise_copy.RunLoadThreadBuffer(p_b_global, p_b_thread_buffer);

                // LDS double buffer: GEMM on 2nd-last data
                const typename vector_type<ABFloat, KPACK>::MemoryType* p_a_block_vec =
                    reinterpret_cast<const typename vector_type<ABFloat, KPACK>::MemoryType*>(
                        p_a_block_double);
                const typename vector_type<ABFloat, KPACK>::MemoryType* p_b_block_vec =
                    reinterpret_cast<const typename vector_type<ABFloat, KPACK>::MemoryType*>(
                        p_b_block_double);
                blockwise_gemm.Run(p_a_block_vec, p_b_block_vec, p_c_thread);

                // LDS double buffer: store last data to LDS
                a_blockwise_copy.RunStoreThreadBuffer(p_a_thread_buffer,
                                                      p_a_block_double + a_block_space);
                b_blockwise_copy.RunStoreThreadBuffer(p_b_thread_buffer,
                                                      p_b_block_double + b_block_space);

                __syncthreads();

                // LDS double buffer: GEMM on last data
                p_a_block_vec =
                    reinterpret_cast<const typename vector_type<ABFloat, KPACK>::MemoryType*>(
                        p_a_block_double + a_block_space);
                p_b_block_vec =
                    reinterpret_cast<const typename vector_type<ABFloat, KPACK>::MemoryType*>(
                        p_b_block_double + b_block_space);
                blockwise_gemm.Run(p_a_block_vec, p_b_block_vec, p_c_thread);
            }
            else // if has 1 iteration left
            {
                __syncthreads();

                // LDS double buffer: GEMM on last data
                const typename vector_type<ABFloat, KPACK>::MemoryType* p_a_block_vec =
                    reinterpret_cast<const typename vector_type<ABFloat, KPACK>::MemoryType*>(
                        p_a_block_double);
                const typename vector_type<ABFloat, KPACK>::MemoryType* p_b_block_vec =
                    reinterpret_cast<const typename vector_type<ABFloat, KPACK>::MemoryType*>(
                        p_b_block_double);
                blockwise_gemm.Run(p_a_block_vec, p_b_block_vec, p_c_thread);
            }
        }

        // input: register to global memory
        {
            constexpr index_t M1 = MPerThread * MLevel0Cluster * MLevel1Cluster;
            constexpr index_t M0 = M / M1;

            constexpr index_t N1 = NPerThread * NLevel0Cluster * NLevel1Cluster;
            constexpr index_t N0 = N / N1;

            // define input tensor descriptor for threadwise copy
            //     thread input tensor, src of threadwise copy
            constexpr auto c_m0_m1_n0_n1_thread_desc = make_native_tensor_descriptor_packed(
                Sequence<GemmMRepeat, MPerThread, GemmNRepeat, NPerThread>{});

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

            ThreadwiseGenericTensorSliceCopy_v4r2<
                decltype(c_m0_m1_n0_n1_thread_desc),
                decltype(c_m0_m1_n0_n1_global_desc),
                decltype(c_m0_m1_n0_n1_thread_desc.GetLengths()),
                CThreadCopySrcDstAccessOrder,
                CThreadCopySrcDstVectorReadWriteDim,
                1,
                CThreadCopyDstDataPerWrite,
                AddressSpace::Vgpr,
                is_same<AccFloat, CFloat>::value ? AddressSpace::Global : AddressSpace::Generic,
                CGlobalMemoryDataOperation>({0, 0, 0, 0},
                                            {m_thread_data_on_global / M1,
                                             m_thread_data_on_global % M1,
                                             n_thread_data_on_global / N1,
                                             n_thread_data_on_global % N1})
                .Run(p_c_thread, p_c_global);
        }
    }

    __device__ void Run(const ABFloat* __restrict__ p_a_global,
                        const ABFloat* __restrict__ p_b_global,
                        CFloat* __restrict__ p_c_global) const
    {
        constexpr index_t shared_block_size = GetSharedMemoryNumberOfByte() / sizeof(ABFloat);

        __shared__ ABFloat p_shared_block[shared_block_size];

        Run(p_a_global, p_b_global, p_c_global, p_shared_block);
    }
};

} // namespace ck
#endif // CK_GRIDWISE_GEMM_FP16_BFP16_HPP
