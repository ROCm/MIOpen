#ifndef CK_GRIDWISE_GEMM_XDLOPS_FP16_BFP16_HPP
#define CK_GRIDWISE_GEMM_XDLOPS_FP16_BFP16_HPP

#include "static_kernel_common_header.hpp"
#include "static_kernel_tensor_descriptor.hpp"
#include "static_kernel_tensor_descriptor_helper.hpp"
#include "static_kernel_ConstantMatrixDescriptor.hpp"
#include "static_kernel_blockwise_generic_tensor_slice_copy.hpp"
#include "static_kernel_threadwise_generic_tensor_slice_copy.hpp"
#include "static_kernel_blockwise_gemm_xdlops.hpp"

namespace ck {

enum WorkgroupScheduleOrder
{
    MBlock1NBlock0,
    NBlock1MBlock0
};

template <index_t Gi,
          index_t MBlockWork,
          index_t NBlockWork,
          WorkgroupScheduleOrder WorkgroupSchdOrder>
struct make_batch_block_work_sequence;

template <index_t Gi, index_t MBlockWork, index_t NBlockWork>
struct make_batch_block_work_sequence<Gi, MBlockWork, NBlockWork, MBlock1NBlock0>
{
    __device__ constexpr auto get() { return Sequence<Gi, MBlockWork, NBlockWork>{}; }
};

template <index_t Gi, index_t MBlockWork, index_t NBlockWork>
struct make_batch_block_work_sequence<Gi, MBlockWork, NBlockWork, NBlock1MBlock0>
{
    __device__ constexpr auto get() { return Sequence<Gi, NBlockWork, MBlockWork>{}; }
};

template <index_t MBlockWork, index_t NBlockWork, WorkgroupScheduleOrder WorkgroupSchdOrder>
struct make_block_work_sequence;

template <index_t MBlockWork, index_t NBlockWork>
struct make_block_work_sequence<MBlockWork, NBlockWork, MBlock1NBlock0>
{
    __device__ constexpr auto get() { return Sequence<MBlockWork, NBlockWork>{}; }
};

template <index_t MBlockWork, index_t NBlockWork>
struct make_block_work_sequence<MBlockWork, NBlockWork, NBlock1MBlock0>
{
    __device__ constexpr auto get() { return Sequence<NBlockWork, MBlockWork>{}; }
};

template <index_t GridSize,
          index_t BlockSize,
          class ABFloat,
          class AccFloat,
          class CFloat,
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
          class ABlockCopyThreadSliceLengths_K_M_KPACK,
          class ABlockCopyThreadClusterLengths_K_M_KPACK,
          class ABlockCopyThreadClusterArrangeOrder,
          class ABlockCopySrcAccessOrder,
          class ABlockCopyDstAccessOrder,
          index_t ABlockCopySrcVectorReadDim,
          index_t ABlockCopySrcDataPerRead,
          index_t ABlockCopyDstDataPerWrite_KPACK,
          class BBlockCopyThreadSliceLengths_K_N_KPACK,
          class BBlockCopyThreadClusterLengths_K_N_KPACK,
          class BBlockCopyThreadClusterArrangeOrder,
          class BBlockCopySrcAccessOrder,
          class BBlockCopyDstAccessOrder,
          index_t BBlockCopySrcVectorReadDim,
          index_t BBlockCopySrcDataPerRead,
          index_t BBlockCopyDstDataPerWrite_KPACK,
          InMemoryDataOperation OutputMemOp,
          WorkgroupScheduleOrder WorkgroupSchdOrder,
          index_t ABlockCopySrcDataStride = 1,
          index_t BBlockCopySrcDataStride = 1>
struct GridwiseGemmTransposedANormalBNormalCXdlopsFp16Bfp16_v1
{
    __device__ void Run(const ABFloat* const __restrict__ p_a_global,
                        const ABFloat* const __restrict__ p_b_global,
                        CFloat* const __restrict__ p_c_global) const
    {
        constexpr auto b_k_n_kpack_global_desc = BGlobalDesc{};
        constexpr auto a_k_m_kpack_global_desc = AGlobalDesc{};
        constexpr auto c_m_n_global_desc       = CGlobalDesc{};

        constexpr auto True = integral_constant<bool, true>{};

        constexpr auto K     = b_k_n_kpack_global_desc.GetLengths()[0];
        constexpr auto N     = b_k_n_kpack_global_desc.GetLengths()[1];
        constexpr auto M     = a_k_m_kpack_global_desc.GetLengths()[1];
        constexpr auto KPACK = b_k_n_kpack_global_desc.GetLengths()[2];

        // divide block work by [M, N]
        static_assert(M % MPerBlock == 0 && N % NPerBlock == 0 && K % KPerBlock == 0,
                      "wrong! cannot divide work evenly among block");

        constexpr index_t MBlockWork = M / MPerBlock;
        constexpr index_t NBlockWork = N / NPerBlock;

        constexpr index_t MWaves = MPerBlock / MPerWave;
        constexpr index_t NWaves = NPerBlock / NPerWave;

        constexpr auto block_work_sequence =
            make_block_work_sequence<MBlockWork, NBlockWork, WorkgroupSchdOrder>{}.get();
        constexpr auto block_work_desc = make_cluster_descriptor(block_work_sequence);

        const auto block_work_id = block_work_desc.CalculateClusterIndex(get_block_1d_id());

        const index_t k_block_data_on_global = (WorkgroupSchdOrder == MBlock1NBlock0)
                                                   ? (block_work_id[0] * MPerBlock)
                                                   : (block_work_id[1] * MPerBlock);
        const index_t b_block_data_on_global = (WorkgroupSchdOrder == MBlock1NBlock0)
                                                   ? (block_work_id[1] * NPerBlock)
                                                   : (block_work_id[0] * NPerBlock);

        //   LDS mem
        constexpr index_t max_align = math::lcm(BBlockCopyDstDataPerWrite_KPACK,
                                                ABlockCopyDstDataPerWrite_KPACK,
                                                KPACK * GemmDataPerReadM,
                                                KPACK * GemmDataPerReadN);

        //   LDS
        //     be careful of LDS alignment
        constexpr auto a_k_m_kpack_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<KPerBlock, MPerBlock, KPACK>{}, Number<max_align>{});

        auto a_blockwise_copy = BlockwiseGenericTensorSliceCopy_v4<
            BlockSize,
            decltype(a_k_m_kpack_global_desc),
            decltype(a_k_m_kpack_block_desc),
            decltype(a_k_m_kpack_block_desc.GetLengths()),
            ABlockCopyThreadSliceLengths_K_M_KPACK,
            ABlockCopyThreadClusterLengths_K_M_KPACK,
            ABlockCopyThreadClusterArrangeOrder,
            ABlockCopySrcAccessOrder,
            ABlockCopyDstAccessOrder,
            ABlockCopySrcVectorReadDim, // Src dim to be read in vector form (M dimension)
            2,                          // Dst dim to be written in vector form (KPACK dimension)
            ABlockCopySrcDataPerRead,
            ABlockCopyDstDataPerWrite_KPACK,
            AddressSpace::Global,
            AddressSpace::Vgpr,
            AddressSpace::Lds,
            InMemoryDataOperation::Set,
            ABlockCopySrcDataStride>({0, k_block_data_on_global, 0}, {0, 0, 0});

        constexpr auto b_k_n_kpack_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<KPerBlock, NPerBlock, KPACK>{}, Number<max_align>{});

        // input blockwise copy
        auto b_blockwise_copy = BlockwiseGenericTensorSliceCopy_v4<
            BlockSize,
            decltype(b_k_n_kpack_global_desc),
            decltype(b_k_n_kpack_block_desc),
            decltype(b_k_n_kpack_block_desc.GetLengths()),
            BBlockCopyThreadSliceLengths_K_N_KPACK,
            BBlockCopyThreadClusterLengths_K_N_KPACK,
            BBlockCopyThreadClusterArrangeOrder,
            BBlockCopySrcAccessOrder,
            BBlockCopyDstAccessOrder,
            BBlockCopySrcVectorReadDim, // Src dim to be read in vector form (N dimension)
            2,                          // Dst dim to be written in vector form (KPACK dimension)
            BBlockCopySrcDataPerRead,
            BBlockCopyDstDataPerWrite_KPACK,
            AddressSpace::Global,
            AddressSpace::Vgpr,
            AddressSpace::Lds,
            InMemoryDataOperation::Set,
            BBlockCopySrcDataStride>({0, b_block_data_on_global, 0}, {0, 0, 0});

        // GEMM definition
        // c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[KPerBlock, MPerBlock] is in LDS
        //     b_mtx[KPerBlocl, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //     register
        constexpr auto a_k_m_block_mtx_desc =
            make_ConstantMatrixDescriptor_packed(Number<KPerBlock>{}, Number<MPerBlock>{});
        constexpr auto b_k_n_block_mtx_desc =
            make_ConstantMatrixDescriptor_packed(Number<KPerBlock>{}, Number<NPerBlock>{});

        const auto blockwise_gemm = BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_xdlops<
            BlockSize,
            decltype(a_k_m_block_mtx_desc),
            decltype(b_k_n_block_mtx_desc),
            ABFloat,
            MPerWave,
            NPerWave,
            MWaves,
            NWaves,
            GemmDataPerReadM,
            GemmDataPerReadN>{};

        constexpr index_t a_block_space =
            math::integer_least_multiple(a_k_m_kpack_block_desc.GetElementSpace(), max_align);

        constexpr index_t b_block_space =
            math::integer_least_multiple(b_k_n_kpack_block_desc.GetElementSpace(), max_align);

        __shared__ ABFloat p_a_block_double[2 * a_block_space];
        __shared__ ABFloat p_b_block_double[2 * b_block_space];

        // get zero-initialized output register of vector type
        auto c_thread_vec = blockwise_gemm.CreateOutputVecZero();

        // LDS double buffer: preload data into LDS
        {
            a_blockwise_copy.Run(p_a_global, p_a_block_double);
            b_blockwise_copy.Run(p_b_global, p_b_block_double);
        }

        using blockwise_a_copy_src_step = Sequence<KPerBlock, 0, 0>;
        using blockwise_b_copy_src_step = Sequence<KPerBlock, 0, 0>;

        // LDS double buffer: main body
        for(index_t k_block_data_begin = 0; k_block_data_begin + 2 * KPerBlock < K;
            k_block_data_begin += 2 * KPerBlock)
        {
#pragma unroll
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

                a_blockwise_copy.MoveSrcSliceWindow(blockwise_a_copy_src_step{}, True);
                b_blockwise_copy.MoveSrcSliceWindow(blockwise_b_copy_src_step{}, True);

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
                c_thread_vec = blockwise_gemm.Run(p_a_block_vec, p_b_block_vec, c_thread_vec);

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

                a_blockwise_copy.MoveSrcSliceWindow(blockwise_a_copy_src_step{}, True);
                b_blockwise_copy.MoveSrcSliceWindow(blockwise_b_copy_src_step{}, True);

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
                c_thread_vec = blockwise_gemm.Run(p_a_block_vec, p_b_block_vec, c_thread_vec);

                // LDS double buffer: store last data to LDS
                a_blockwise_copy.RunStoreThreadBuffer(p_a_thread_buffer,
                                                      p_a_block_double + a_block_space);
                b_blockwise_copy.RunStoreThreadBuffer(p_b_thread_buffer,
                                                      p_b_block_double + b_block_space);

                __syncthreads();

                // LDS double buffer: GEMM on current data
                p_a_block_vec =
                    reinterpret_cast<const typename vector_type<ABFloat, KPACK>::MemoryType*>(
                        p_a_block_double + a_block_space);
                p_b_block_vec =
                    reinterpret_cast<const typename vector_type<ABFloat, KPACK>::MemoryType*>(
                        p_b_block_double + b_block_space);
                c_thread_vec = blockwise_gemm.Run(p_a_block_vec, p_b_block_vec, c_thread_vec);
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
                c_thread_vec = blockwise_gemm.Run(p_a_block_vec, p_b_block_vec, c_thread_vec);
            }
        }

        // copy output: register to global memory
        {
            constexpr auto OutputLayout = blockwise_gemm.GetOutputLayout();
            constexpr index_t K0        = OutputLayout.M1();
            constexpr index_t K1        = OutputLayout.N1();
            constexpr index_t K2        = OutputLayout.M0();

            constexpr auto out_k0_k1_k2_b_global_desc = transform_tensor_descriptor(
                c_m_n_global_desc,
                make_tuple(UnMerge<Sequence<K0, K1, K2>>{}, PassThrough<N>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}));

            //     src descriptor
            constexpr auto out_k0_k1_k2_b_thread_desc =
                make_native_tensor_descriptor_packed(Sequence<K0, 1, K2, 1>{});

            using OutThreadCopySliceLengths = Sequence<K0, 1, K2, 1>;

            constexpr index_t BlkSize = OutputLayout.GetBlkSize();
            constexpr index_t NumBlks = OutputLayout.GetNumBlks();

// force unrolling the output loop to get ride of scratches
#pragma unroll
            for(index_t i = 0; i < NumBlks; ++i)
            {
                // calculate origin of thread output tensor on global memory
                //     blockwise GEMM c matrix starting index
                const auto c_thread_mtx_on_block = blockwise_gemm.GetBeginOfThreadMatrixC(i);

                const index_t k_thread_data_on_global =
                    k_block_data_on_global + c_thread_mtx_on_block.row;

                const index_t b_thread_data_on_global =
                    b_block_data_on_global + c_thread_mtx_on_block.col;

                ThreadwiseGenericTensorSliceCopy_v4r2<
                    decltype(out_k0_k1_k2_b_thread_desc),
                    decltype(out_k0_k1_k2_b_global_desc),
                    OutThreadCopySliceLengths,
                    arithmetic_sequence_gen<0, 4, 1>::type,
                    3,
                    1,
                    1,
                    AddressSpace::Vgpr,
                    is_same<AccFloat, CFloat>::value ? AddressSpace::Global : AddressSpace::Generic,
                    OutputMemOp>({0, 0, 0, 0},
                                 {k_thread_data_on_global / (K2 * K1),
                                  k_thread_data_on_global % (K2 * K1) / K2,
                                  k_thread_data_on_global % K2,
                                  b_thread_data_on_global})
                    .Run(c_thread_vec.n + i * BlkSize, p_c_global);
            }
        }
    }
};

template <index_t GridSize,
          index_t BlockSize,
          class ABFloat,
          class AccFloat,
          class CFloat,
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
          class ABlockCopyThreadSliceLengths_G_K_M_KPACK,
          class ABlockCopyThreadClusterLengths_G_K_M_KPACK,
          class ABlockCopyThreadClusterArrangeOrder,
          class ABlockCopySrcAccessOrder,
          class ABlockCopyDstAccessOrder,
          index_t ABlockCopySrcVectorReadDim,
          index_t ABlockCopySrcDataPerRead,
          index_t ABlockCopyDstDataPerWrite_KPACK,
          class BBlockCopyThreadSliceLengths_G_K_N_KPACK,
          class BBlockCopyThreadClusterLengths_G_K_N_KPACK,
          class BBlockCopyThreadClusterArrangeOrder,
          class BBlockCopySrcAccessOrder,
          class BBlockCopyDstAccessOrder,
          index_t BBlockCopySrcVectorReadDim,
          index_t BBlockCopySrcDataPerRead,
          index_t BBlockCopyDstDataPerWrite_KPACK,
          InMemoryDataOperation OutputMemOp,
          WorkgroupScheduleOrder WorkgroupSchdOrder,
          index_t ABlockCopySrcDataStride = 1,
          index_t BBlockCopySrcDataStride = 1>
struct GridwiseBatchedGemmTransposedANormalBNormalCXdlopsFp16Bfp16_v1
{
    __device__ void Run(const ABFloat* const __restrict__ p_a_global,
                        const ABFloat* const __restrict__ p_b_global,
                        CFloat* const __restrict__ p_c_global) const
    {

        constexpr auto a_g_k_m_kpack_global_desc = AGlobalDesc{};
        constexpr auto b_g_k_n_kpack_global_desc = BGlobalDesc{};
        constexpr auto c_g_m_n_global_desc       = CGlobalDesc{};

        constexpr auto True = integral_constant<bool, true>{};

        constexpr auto Gi = b_g_k_n_kpack_global_desc.GetLengths()[0];
        constexpr auto Go = c_g_m_n_global_desc.GetLengths()[0];

        constexpr auto K     = b_g_k_n_kpack_global_desc.GetLengths()[1];
        constexpr auto N     = b_g_k_n_kpack_global_desc.GetLengths()[2];
        constexpr auto M     = a_g_k_m_kpack_global_desc.GetLengths()[2];
        constexpr auto KPACK = b_g_k_n_kpack_global_desc.GetLengths()[3];

        // divide block work by [M, N]
        static_assert(M % MPerBlock == 0 && N % NPerBlock == 0 && K % KPerBlock == 0,
                      "wrong! cannot divide work evenly among block");

        constexpr index_t MBlockWork = M / MPerBlock;
        constexpr index_t NBlockWork = N / NPerBlock;

        constexpr index_t MWaves = MPerBlock / MPerWave;
        constexpr index_t NWaves = NPerBlock / NPerWave;

        constexpr auto block_work_sequence =
            make_batch_block_work_sequence<Gi, MBlockWork, NBlockWork, WorkgroupSchdOrder>{}.get();
        constexpr auto block_work_desc = make_cluster_descriptor(block_work_sequence);

        const auto block_work_id = block_work_desc.CalculateClusterIndex(get_block_1d_id());

        const index_t group_id               = block_work_id[0];
        const index_t m_block_data_on_global = (WorkgroupSchdOrder == MBlock1NBlock0)
                                                   ? (block_work_id[1] * MPerBlock)
                                                   : (block_work_id[2] * MPerBlock);
        const index_t n_block_data_on_global = (WorkgroupSchdOrder == MBlock1NBlock0)
                                                   ? (block_work_id[2] * NPerBlock)
                                                   : (block_work_id[1] * NPerBlock);

        //   LDS mem
        constexpr index_t max_align = math::lcm(BBlockCopyDstDataPerWrite_KPACK,
                                                ABlockCopyDstDataPerWrite_KPACK,
                                                KPACK * GemmDataPerReadM,
                                                KPACK * GemmDataPerReadN);

        //   LDS
        //     be careful of LDS alignment
        constexpr auto a_g_k_m_kpack_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<1, KPerBlock, MPerBlock, KPACK>{}, Number<max_align>{});

        auto a_blockwise_copy = BlockwiseGenericTensorSliceCopy_v4<
            BlockSize,
            decltype(a_g_k_m_kpack_global_desc),
            decltype(a_g_k_m_kpack_block_desc),
            decltype(a_g_k_m_kpack_block_desc.GetLengths()),
            ABlockCopyThreadSliceLengths_G_K_M_KPACK,
            ABlockCopyThreadClusterLengths_G_K_M_KPACK,
            ABlockCopyThreadClusterArrangeOrder,
            ABlockCopySrcAccessOrder,
            ABlockCopyDstAccessOrder,
            ABlockCopySrcVectorReadDim, // Src dim to be read in vector form (K dimension)
            3,                          // Dst dim to be written in vector form (KPACK dimension)
            ABlockCopySrcDataPerRead,
            ABlockCopyDstDataPerWrite_KPACK,
            AddressSpace::Global,
            AddressSpace::Vgpr,
            AddressSpace::Lds,
            InMemoryDataOperation::Set,
            ABlockCopySrcDataStride>({group_id, 0, m_block_data_on_global, 0}, {0, 0, 0, 0});

        constexpr auto b_g_k_n_kpack_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<1, KPerBlock, NPerBlock, KPACK>{}, Number<max_align>{});

        // input blockwise copy
        auto b_blockwise_copy = BlockwiseGenericTensorSliceCopy_v4<
            BlockSize,
            decltype(b_g_k_n_kpack_global_desc),
            decltype(b_g_k_n_kpack_block_desc),
            decltype(b_g_k_n_kpack_block_desc.GetLengths()),
            BBlockCopyThreadSliceLengths_G_K_N_KPACK,
            BBlockCopyThreadClusterLengths_G_K_N_KPACK,
            BBlockCopyThreadClusterArrangeOrder,
            BBlockCopySrcAccessOrder,
            BBlockCopyDstAccessOrder,
            BBlockCopySrcVectorReadDim, // Src dim to be read in vector form (K dimension)
            3,                          // Dst dim to be written in vector form (KPACK dimension)
            BBlockCopySrcDataPerRead,   // N dimension
            BBlockCopyDstDataPerWrite_KPACK,
            AddressSpace::Global,
            AddressSpace::Vgpr,
            AddressSpace::Lds,
            InMemoryDataOperation::Set,
            BBlockCopySrcDataStride>({group_id, 0, n_block_data_on_global, 0}, {0, 0, 0, 0});

        // GEMM definition
        // c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[KPerBlock, MPerBlock] is in LDS
        //     b_mtx[KPerBlocl, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //     register
        constexpr auto a_k_m_block_mtx_desc =
            make_ConstantMatrixDescriptor_packed(Number<KPerBlock>{}, Number<MPerBlock>{});
        constexpr auto b_k_n_block_mtx_desc =
            make_ConstantMatrixDescriptor_packed(Number<KPerBlock>{}, Number<NPerBlock>{});

        const auto blockwise_gemm = BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_xdlops<
            BlockSize,
            decltype(a_k_m_block_mtx_desc),
            decltype(b_k_n_block_mtx_desc),
            ABFloat,
            MPerWave,
            NPerWave,
            MWaves,
            NWaves,
            GemmDataPerReadM,
            GemmDataPerReadN>{};

        constexpr index_t a_block_space =
            math::integer_least_multiple(a_g_k_m_kpack_block_desc.GetElementSpace(), max_align);

        constexpr index_t b_block_space =
            math::integer_least_multiple(b_g_k_n_kpack_block_desc.GetElementSpace(), max_align);

        __shared__ ABFloat p_a_block_double[2 * a_block_space];
        __shared__ ABFloat p_b_block_double[2 * b_block_space];

        // get zero-initialized output register of vector type
        auto c_thread_vec = blockwise_gemm.CreateOutputVecZero();

        // LDS double buffer: preload data into LDS
        {
            a_blockwise_copy.Run(p_a_global, p_a_block_double);
            b_blockwise_copy.Run(p_b_global, p_b_block_double);
        }

        using blockwise_a_copy_src_step = Sequence<0, KPerBlock, 0, 0>;
        using blockwise_b_copy_src_step = Sequence<0, KPerBlock, 0, 0>;

        // LDS double buffer: main body
        for(index_t k_block_data_begin = 0; k_block_data_begin + 2 * KPerBlock < K;
            k_block_data_begin += 2 * KPerBlock)
        {
#pragma unroll
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

                a_blockwise_copy.MoveSrcSliceWindow(blockwise_a_copy_src_step{}, True);
                b_blockwise_copy.MoveSrcSliceWindow(blockwise_b_copy_src_step{}, True);

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
                c_thread_vec = blockwise_gemm.Run(p_a_block_vec, p_b_block_vec, c_thread_vec);

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

                a_blockwise_copy.MoveSrcSliceWindow(blockwise_a_copy_src_step{}, True);
                b_blockwise_copy.MoveSrcSliceWindow(blockwise_b_copy_src_step{}, True);

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
                c_thread_vec = blockwise_gemm.Run(p_a_block_vec, p_b_block_vec, c_thread_vec);

                // LDS double buffer: store last data to LDS
                a_blockwise_copy.RunStoreThreadBuffer(p_a_thread_buffer,
                                                      p_a_block_double + a_block_space);
                b_blockwise_copy.RunStoreThreadBuffer(p_b_thread_buffer,
                                                      p_b_block_double + b_block_space);

                __syncthreads();

                // LDS double buffer: GEMM on current data
                p_a_block_vec =
                    reinterpret_cast<const typename vector_type<ABFloat, KPACK>::MemoryType*>(
                        p_a_block_double + a_block_space);
                p_b_block_vec =
                    reinterpret_cast<const typename vector_type<ABFloat, KPACK>::MemoryType*>(
                        p_b_block_double + b_block_space);
                c_thread_vec = blockwise_gemm.Run(p_a_block_vec, p_b_block_vec, c_thread_vec);
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
                c_thread_vec = blockwise_gemm.Run(p_a_block_vec, p_b_block_vec, c_thread_vec);
            }
        }

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
                make_tuple(PassThrough<Go>{}, UnMerge<Sequence<M0, M1, M2>>{}, PassThrough<N>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2, 3>{}, Sequence<4>{}));

            //     src descriptor
            constexpr auto c_g_m0_m1_m2_n_thread_desc =
                make_native_tensor_descriptor_packed(Sequence<1, M0, 1, M2, 1>{});

            using CThreadCopySliceLengths = Sequence<1, M0, 1, M2, 1>;

            constexpr index_t BlkSize = CLayout.GetBlkSize();
            constexpr index_t NumBlks = CLayout.GetNumBlks();

// force unrolling the output loop to get ride of scratches
#pragma unroll
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
                                                      OutputMemOp>(
                    {0, 0, 0, 0, 0},
                    {group_id,
                     m_thread_data_on_global / (M2 * M1),
                     m_thread_data_on_global % (M2 * M1) / M2,
                     m_thread_data_on_global % M2,
                     n_thread_data_on_global})
                    .Run(c_thread_vec.n + i * BlkSize, p_c_global);
            }
        }
    }
};

template <index_t GridSize,
          index_t BlockSize,
          class ABFloat,
          class AccFloat,
          class CFloat,
          class AGlobalDesc,
          class BGlobalDesc,
          class CGlobalDesc,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerWave,
          index_t NPerWave,
          class ABlockCopyThreadSliceLengths_G_K_M_KPACK,
          class ABlockCopyThreadClusterLengths_G_K_M_KPACK,
          class ABlockCopyThreadClusterArrangeOrder,
          class ABlockCopySrcAccessOrder,
          class ABlockCopyDstAccessOrder,
          index_t ABlockCopySrcVectorReadDim,
          index_t ABlockCopySrcDataPerRead,
          index_t ABlockCopyDstDataPerWrite_KPACK,
          class BBlockCopyThreadSliceLengths_G_K_N_KPACK,
          class BBlockCopyThreadClusterLengths_G_K_N_KPACK,
          class BBlockCopyThreadClusterArrangeOrder,
          class BBlockCopySrcAccessOrder,
          class BBlockCopyDstAccessOrder,
          index_t BBlockCopySrcVectorReadDim,
          index_t BBlockCopySrcDataPerRead,
          index_t BBlockCopyDstDataPerWrite_KPACK,
          InMemoryDataOperation CGlobalMemoryOp,
          WorkgroupScheduleOrder WorkgroupSchdOrder>
struct GridwiseBatchGemmXdlops_gkmkpack_gknkpack_gmn_v2
{
    __device__ void Run(const ABFloat* const __restrict__ p_a_global,
                        const ABFloat* const __restrict__ p_b_global,
                        CFloat* const __restrict__ p_c_global) const
    {
        constexpr auto True = integral_constant<bool, true>{};

        constexpr auto a_g_k_m_kpack_global_desc = AGlobalDesc{};
        constexpr auto b_g_k_n_kpack_global_desc = BGlobalDesc{};
        constexpr auto c_g_m_n_global_desc       = CGlobalDesc{};

        constexpr auto G     = c_g_m_n_global_desc.GetLengths()[0];
        constexpr auto M     = c_g_m_n_global_desc.GetLengths()[1];
        constexpr auto N     = c_g_m_n_global_desc.GetLengths()[2];
        constexpr auto K     = b_g_k_n_kpack_global_desc.GetLengths()[1];
        constexpr auto KPack = b_g_k_n_kpack_global_desc.GetLengths()[3];

        // divide block work by [M, N]
        static_assert(M % MPerBlock == 0 && N % NPerBlock == 0 && K % KPerBlock == 0,
                      "wrong! cannot divide work evenly among block");

        constexpr index_t MBlockWork = M / MPerBlock;
        constexpr index_t NBlockWork = N / NPerBlock;

        constexpr index_t MWavePerBlock = MPerBlock / MPerWave;
        constexpr index_t NWavePerBlock = NPerBlock / NPerWave;

        constexpr auto block_work_sequence =
            make_batch_block_work_sequence<G, MBlockWork, NBlockWork, WorkgroupSchdOrder>{}.get();
        constexpr auto block_work_desc = make_cluster_descriptor(block_work_sequence);

        const auto block_work_id = block_work_desc.CalculateClusterIndex(get_block_1d_id());

        const index_t g_block_data_on_global = block_work_id[0];
        const index_t m_block_data_on_global = (WorkgroupSchdOrder == MBlock1NBlock0)
                                                   ? (block_work_id[1] * MPerBlock)
                                                   : (block_work_id[2] * MPerBlock);
        const index_t n_block_data_on_global = (WorkgroupSchdOrder == MBlock1NBlock0)
                                                   ? (block_work_id[2] * NPerBlock)
                                                   : (block_work_id[1] * NPerBlock);

        constexpr index_t max_align = KPack;

        //   LDS be careful of LDS alignment
        constexpr auto a_g_k_m_kpack_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<1, KPerBlock, MPerBlock, KPack>{}, Number<max_align>{});

        auto a_blockwise_copy = BlockwiseGenericTensorSliceCopy_v4<
            BlockSize,
            decltype(a_g_k_m_kpack_global_desc),
            decltype(a_g_k_m_kpack_block_desc),
            decltype(a_g_k_m_kpack_block_desc.GetLengths()),
            ABlockCopyThreadSliceLengths_G_K_M_KPACK,
            ABlockCopyThreadClusterLengths_G_K_M_KPACK,
            ABlockCopyThreadClusterArrangeOrder,
            ABlockCopySrcAccessOrder,
            ABlockCopyDstAccessOrder,
            ABlockCopySrcVectorReadDim, // Src dim to be read in vector form
            3,                          // Dst dim to be written in vector form (KPack dimension)
            ABlockCopySrcDataPerRead,
            ABlockCopyDstDataPerWrite_KPACK,
            AddressSpace::Global,
            AddressSpace::Vgpr,
            AddressSpace::Lds,
            InMemoryDataOperation::Set>({g_block_data_on_global, 0, m_block_data_on_global, 0},
                                        {0, 0, 0, 0});

        constexpr auto b_g_k_n_kpack_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<1, KPerBlock, NPerBlock, KPack>{}, Number<max_align>{});

        // input blockwise copy
        auto b_blockwise_copy = BlockwiseGenericTensorSliceCopy_v4<
            BlockSize,
            decltype(b_g_k_n_kpack_global_desc),
            decltype(b_g_k_n_kpack_block_desc),
            decltype(b_g_k_n_kpack_block_desc.GetLengths()),
            BBlockCopyThreadSliceLengths_G_K_N_KPACK,
            BBlockCopyThreadClusterLengths_G_K_N_KPACK,
            BBlockCopyThreadClusterArrangeOrder,
            BBlockCopySrcAccessOrder,
            BBlockCopyDstAccessOrder,
            BBlockCopySrcVectorReadDim, // Src dim to be read in vector form
            3,                          // Dst dim to be written in vector form (KPack dimension)
            BBlockCopySrcDataPerRead,
            BBlockCopyDstDataPerWrite_KPACK,
            AddressSpace::Global,
            AddressSpace::Vgpr,
            AddressSpace::Lds,
            InMemoryDataOperation::Set>({g_block_data_on_global, 0, n_block_data_on_global, 0},
                                        {0, 0, 0, 0});

        // GEMM definition
        // c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[KPerBlock, MPerBlock] is in LDS
        //     b_mtx[KPerBlocl, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //     register
        constexpr auto a_k_m_block_mtx_desc =
            make_ConstantMatrixDescriptor_packed(Number<KPerBlock>{}, Number<MPerBlock>{});
        constexpr auto b_k_n_block_mtx_desc =
            make_ConstantMatrixDescriptor_packed(Number<KPerBlock>{}, Number<NPerBlock>{});

        const auto blockwise_gemm = BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_xdlops<
            BlockSize,
            decltype(a_k_m_block_mtx_desc),
            decltype(b_k_n_block_mtx_desc),
            ABFloat,
            MPerWave,
            NPerWave,
            MWavePerBlock,
            NWavePerBlock,
            1,
            1>{};

        constexpr index_t a_block_space =
            math::integer_least_multiple(a_g_k_m_kpack_block_desc.GetElementSpace(), max_align);

        constexpr index_t b_block_space =
            math::integer_least_multiple(b_g_k_n_kpack_block_desc.GetElementSpace(), max_align);

        __shared__ ABFloat p_a_block[a_block_space];
        __shared__ ABFloat p_b_block[b_block_space];

        // get zero-initialized output register of vector type
        auto c_thread_vec = blockwise_gemm.CreateOutputVecZero();

        // preload data into LDS
        {
            a_blockwise_copy.Run(p_a_global, p_a_block);
            b_blockwise_copy.Run(p_b_global, p_b_block);
        }

        constexpr auto blockwise_a_copy_src_step = Sequence<0, KPerBlock, 0, 0>{};
        constexpr auto blockwise_b_copy_src_step = Sequence<0, KPerBlock, 0, 0>{};

        // main body
        for(index_t k_block_data_begin = 0; k_block_data_begin < K - KPerBlock;
            k_block_data_begin += KPerBlock)
        {
            ABFloat p_a_thread_buffer[a_blockwise_copy.GetThreadBufferSize()];
            ABFloat p_b_thread_buffer[b_blockwise_copy.GetThreadBufferSize()];

            // load next data from device mem
            a_blockwise_copy.MoveSrcSliceWindow(blockwise_a_copy_src_step, True);
            b_blockwise_copy.MoveSrcSliceWindow(blockwise_b_copy_src_step, True);

            a_blockwise_copy.RunLoadThreadBuffer(p_a_global, p_a_thread_buffer);
            b_blockwise_copy.RunLoadThreadBuffer(p_b_global, p_b_thread_buffer);

            block_sync_lds();

            // GEMM on current data
            const typename vector_type<ABFloat, KPack>::MemoryType* p_a_block_vec =
                reinterpret_cast<const typename vector_type<ABFloat, KPack>::MemoryType*>(
                    p_a_block);
            const typename vector_type<ABFloat, KPack>::MemoryType* p_b_block_vec =
                reinterpret_cast<const typename vector_type<ABFloat, KPack>::MemoryType*>(
                    p_b_block);

            c_thread_vec = blockwise_gemm.Run(p_a_block_vec, p_b_block_vec, c_thread_vec);

            block_sync_lds();

            // store next data to LDS
            a_blockwise_copy.RunStoreThreadBuffer(p_a_thread_buffer, p_a_block);
            b_blockwise_copy.RunStoreThreadBuffer(p_b_thread_buffer, p_b_block);
        }

        // tail
        {
            block_sync_lds();

            // GEMM on last data
            const typename vector_type<ABFloat, KPack>::MemoryType* p_a_block_vec =
                reinterpret_cast<const typename vector_type<ABFloat, KPack>::MemoryType*>(
                    p_a_block);
            const typename vector_type<ABFloat, KPack>::MemoryType* p_b_block_vec =
                reinterpret_cast<const typename vector_type<ABFloat, KPack>::MemoryType*>(
                    p_b_block);

            c_thread_vec = blockwise_gemm.Run(p_a_block_vec, p_b_block_vec, c_thread_vec);
        }

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
                make_tuple(
                    PassThrough<G>{}, UnMerge<Sequence<M / (M1 * M2), M1, M2>>{}, PassThrough<N>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2, 3>{}, Sequence<4>{}));

            //     src descriptor
            constexpr auto c_g_m0_m1_m2_n_thread_desc =
                make_native_tensor_descriptor_packed(Sequence<1, M0, 1, M2, 1>{});

            using CThreadCopySliceLengths = Sequence<1, M0, 1, M2, 1>;

            constexpr index_t BlkSize = blockwise_gemm.GetBlkSize();
            constexpr index_t NumBlks = blockwise_gemm.GetNumBlks();

// force unrolling the output loop to get ride of scratches
#pragma unroll
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
                                                      CGlobalMemoryOp>(
                    {0, 0, 0, 0, 0},
                    {g_block_data_on_global,
                     m_thread_data_on_global / (M2 * M1),
                     m_thread_data_on_global % (M2 * M1) / M2,
                     m_thread_data_on_global % M2,
                     n_thread_data_on_global})
                    .Run(c_thread_vec.n + i * BlkSize, p_c_global);
            }
        }
    }
};

template <index_t GridSize,
          index_t BlockSize,
          class ABFloat,
          class AccFloat,
          class CFloat,
          class AGlobalDesc,
          class BGlobalDesc,
          class CGlobalDesc,
          index_t MPerBlock,
          index_t BPerBlock,
          index_t KPerBlock,
          index_t MPerWave,
          index_t BPerWave,
          class ABlockCopyThreadSliceLengths_G_K_M_KPACK,
          class ABlockCopyThreadClusterLengths_G_K_M_KPACK,
          class ABlockCopyThreadClusterArrangeOrder,
          class ABlockCopySrcAccessOrder,
          class ABlockCopyDstAccessOrder,
          index_t ABlockCopySrcVectorReadDim,
          index_t ABlockCopySrcDataPerRead,
          index_t ABlockCopyDstDataPerWrite_KPACK,
          class BBlockCopyThreadSliceLengths_G_K_N1_B_KPack,
          class BBlockCopyThreadClusterLengths_G_K_N1_B_KPack,
          class BBlockCopyThreadClusterArrangeOrder,
          class BBlockCopySrcAccessOrder,
          class BBlockCopyDstAccessOrder,
          index_t BBlockCopySrcVectorReadDim,
          index_t BBlockCopySrcDataPerRead,
          index_t BBlockCopyDstDataPerWrite_KPACK,
          InMemoryDataOperation CGlobalMemoryOp,
          WorkgroupScheduleOrder WorkgroupSchdOrder>
struct GridwiseBatchGemmXdlops_gkmkpack_gkn1bkpack_gmn_v2
{
    __device__ void Run(const ABFloat* const __restrict__ p_a_global,
                        const ABFloat* const __restrict__ p_b_global,
                        CFloat* const __restrict__ p_c_global) const
    {
        constexpr auto True = integral_constant<bool, true>{};

        constexpr auto a_g_k_m_kpack_global_desc    = AGlobalDesc{};
        constexpr auto b_g_k_n1_b_kpack_global_desc = BGlobalDesc{};
        constexpr auto c_g_m_n_global_desc          = CGlobalDesc{};

        constexpr auto G = c_g_m_n_global_desc.GetLengths()[0];
        constexpr auto M = c_g_m_n_global_desc.GetLengths()[1];
        constexpr auto N = c_g_m_n_global_desc.GetLengths()[2];

        constexpr auto K     = b_g_k_n1_b_kpack_global_desc.GetLengths()[1];
        constexpr auto in_N1 = b_g_k_n1_b_kpack_global_desc.GetLengths()[2];
        constexpr auto B     = b_g_k_n1_b_kpack_global_desc.GetLengths()[3];
        constexpr auto KPack = b_g_k_n1_b_kpack_global_desc.GetLengths()[4];

        // divide block work by [M, N]
        static_assert(M % MPerBlock == 0 && B % BPerBlock == 0 && K % KPerBlock == 0,
                      "wrong! cannot divide work evenly among block");

        constexpr index_t MBlockWork = M / MPerBlock;
        constexpr index_t BBlockWork = B / BPerBlock;

        constexpr index_t MWavePerBlock = MPerBlock / MPerWave;
        constexpr index_t BWavePerBlock = in_N1;

        static_assert((G * MBlockWork * BBlockWork) == GridSize, "Invalid GridSize");

        constexpr auto block_work_sequence =
            make_batch_block_work_sequence<G, MBlockWork, BBlockWork, WorkgroupSchdOrder>{}.get();
        constexpr auto block_work_desc = make_cluster_descriptor(block_work_sequence);

        const auto block_work_id = block_work_desc.CalculateClusterIndex(get_block_1d_id());

        const index_t g_block_data_on_global = block_work_id[0];
        const index_t m_block_data_on_global = (WorkgroupSchdOrder == MBlock1NBlock0)
                                                   ? (block_work_id[1] * MPerBlock)
                                                   : (block_work_id[2] * MPerBlock);
        const index_t b_block_data_on_global = (WorkgroupSchdOrder == MBlock1NBlock0)
                                                   ? (block_work_id[2] * BPerBlock)
                                                   : (block_work_id[1] * BPerBlock);

        constexpr index_t max_align = KPack;

        //   LDS be careful of LDS alignment
        constexpr auto a_g_k_m_kpack_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<1, KPerBlock, MPerBlock, KPack>{}, Number<max_align>{});

        auto a_blockwise_copy = BlockwiseGenericTensorSliceCopy_v4<
            BlockSize,
            decltype(a_g_k_m_kpack_global_desc),
            decltype(a_g_k_m_kpack_block_desc),
            decltype(a_g_k_m_kpack_block_desc.GetLengths()),
            ABlockCopyThreadSliceLengths_G_K_M_KPACK,
            ABlockCopyThreadClusterLengths_G_K_M_KPACK,
            ABlockCopyThreadClusterArrangeOrder,
            ABlockCopySrcAccessOrder,
            ABlockCopyDstAccessOrder,
            ABlockCopySrcVectorReadDim, // Src dim to be read in vector form
            3,                          // Dst dim to be written in vector form (KPack dimension)
            ABlockCopySrcDataPerRead,
            ABlockCopyDstDataPerWrite_KPACK,
            AddressSpace::Global,
            AddressSpace::Vgpr,
            AddressSpace::Lds,
            InMemoryDataOperation::Set>({g_block_data_on_global, 0, m_block_data_on_global, 0},
                                        {0, 0, 0, 0});

        constexpr auto b_g_k_n1_b_kpack_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<1, KPerBlock, in_N1, BPerBlock, KPack>{}, Number<max_align>{});

        // input blockwise copy
        auto b_blockwise_copy = BlockwiseGenericTensorSliceCopy_v4<
            BlockSize,
            decltype(b_g_k_n1_b_kpack_global_desc),
            decltype(b_g_k_n1_b_kpack_block_desc),
            decltype(b_g_k_n1_b_kpack_block_desc.GetLengths()),
            BBlockCopyThreadSliceLengths_G_K_N1_B_KPack,
            BBlockCopyThreadClusterLengths_G_K_N1_B_KPack,
            BBlockCopyThreadClusterArrangeOrder,
            BBlockCopySrcAccessOrder,
            BBlockCopyDstAccessOrder,
            BBlockCopySrcVectorReadDim, // Src dim to be read in vector form
            4,                          // Dst dim to be written in vector form (KPack dimension)
            BBlockCopySrcDataPerRead,
            BBlockCopyDstDataPerWrite_KPACK,
            AddressSpace::Global,
            AddressSpace::Vgpr,
            AddressSpace::Lds,
            InMemoryDataOperation::Set>({g_block_data_on_global, 0, 0, b_block_data_on_global, 0},
                                        {0, 0, 0, 0, 0});

        // GEMM definition
        // c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[KPerBlock, MPerBlock] is in LDS
        //     b_mtx[KPerBlocl, BPerBlock * in_N1] is in LDS
        //     c_mtx[MPerBlock, BPerBlock * in_N1] is distributed among threads, and saved in
        //     register
        constexpr auto a_k_m_block_mtx_desc =
            make_ConstantMatrixDescriptor_packed(Number<KPerBlock>{}, Number<MPerBlock>{});
        constexpr auto b_k_n_block_mtx_desc =
            make_ConstantMatrixDescriptor_packed(Number<KPerBlock>{}, Number<BPerBlock * in_N1>{});

        const auto blockwise_gemm = BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_xdlops<
            BlockSize,
            decltype(a_k_m_block_mtx_desc),
            decltype(b_k_n_block_mtx_desc),
            ABFloat,
            MPerWave,
            BPerWave,
            MWavePerBlock,
            BWavePerBlock,
            1,
            1>{};

        constexpr index_t a_block_space =
            math::integer_least_multiple(a_g_k_m_kpack_block_desc.GetElementSpace(), max_align);

        constexpr index_t b_block_space =
            math::integer_least_multiple(b_g_k_n1_b_kpack_block_desc.GetElementSpace(), max_align);

        __shared__ ABFloat p_a_block[a_block_space];
        __shared__ ABFloat p_b_block[b_block_space];

        // get zero-initialized output register of vector type
        auto c_thread_vec = blockwise_gemm.CreateOutputVecZero();

        // preload data into LDS
        {
            a_blockwise_copy.Run(p_a_global, p_a_block);
            b_blockwise_copy.Run(p_b_global, p_b_block);
        }

        constexpr auto blockwise_a_copy_src_step = Sequence<0, KPerBlock, 0, 0>{};
        constexpr auto blockwise_b_copy_src_step = Sequence<0, KPerBlock, 0, 0, 0>{};

        // main body
        for(index_t k_block_data_begin = 0; k_block_data_begin < K - KPerBlock;
            k_block_data_begin += KPerBlock)
        {
            ABFloat p_a_thread_buffer[a_blockwise_copy.GetThreadBufferSize()];
            ABFloat p_b_thread_buffer[b_blockwise_copy.GetThreadBufferSize()];

            // load next data from device mem
            a_blockwise_copy.MoveSrcSliceWindow(blockwise_a_copy_src_step, True);
            b_blockwise_copy.MoveSrcSliceWindow(blockwise_b_copy_src_step, True);

            a_blockwise_copy.RunLoadThreadBuffer(p_a_global, p_a_thread_buffer);
            b_blockwise_copy.RunLoadThreadBuffer(p_b_global, p_b_thread_buffer);

            block_sync_lds();

            // GEMM on current data
            const typename vector_type<ABFloat, KPack>::MemoryType* p_a_block_vec =
                reinterpret_cast<const typename vector_type<ABFloat, KPack>::MemoryType*>(
                    p_a_block);
            const typename vector_type<ABFloat, KPack>::MemoryType* p_b_block_vec =
                reinterpret_cast<const typename vector_type<ABFloat, KPack>::MemoryType*>(
                    p_b_block);

            c_thread_vec = blockwise_gemm.Run(p_a_block_vec, p_b_block_vec, c_thread_vec);

            block_sync_lds();

            // store next data to LDS
            a_blockwise_copy.RunStoreThreadBuffer(p_a_thread_buffer, p_a_block);
            b_blockwise_copy.RunStoreThreadBuffer(p_b_thread_buffer, p_b_block);
        }

        // tail
        {
            block_sync_lds();

            // GEMM on last data
            const typename vector_type<ABFloat, KPack>::MemoryType* p_a_block_vec =
                reinterpret_cast<const typename vector_type<ABFloat, KPack>::MemoryType*>(
                    p_a_block);
            const typename vector_type<ABFloat, KPack>::MemoryType* p_b_block_vec =
                reinterpret_cast<const typename vector_type<ABFloat, KPack>::MemoryType*>(
                    p_b_block);

            c_thread_vec = blockwise_gemm.Run(p_a_block_vec, p_b_block_vec, c_thread_vec);
        }

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
                make_tuple(
                    PassThrough<G>{}, UnMerge<Sequence<M / (M1 * M2), M1, M2>>{}, PassThrough<N>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2, 3>{}, Sequence<4>{}));

            //     src descriptor
            constexpr auto c_g_m0_m1_m2_n_thread_desc =
                make_native_tensor_descriptor_packed(Sequence<1, M0, 1, M2, 1>{});

            using CThreadCopySliceLengths = Sequence<1, M0, 1, M2, 1>;

            constexpr index_t BlkSize = blockwise_gemm.GetBlkSize();
            constexpr index_t NumBlks = blockwise_gemm.GetNumBlks();

// force unrolling the output loop to get ride of scratches
#pragma unroll
            for(index_t i = 0; i < NumBlks; ++i)
            {
                // calculate origin of thread output tensor on global memory
                //     blockwise GEMM c matrix starting index
                const auto c_thread_mtx_on_block =
                    blockwise_gemm.template GetBeginOfThreadMatrixC<MPerWave, B>(i);

                const index_t m_thread_data_on_global =
                    m_block_data_on_global + c_thread_mtx_on_block.row;

                const index_t n_thread_data_on_global =
                    b_block_data_on_global + c_thread_mtx_on_block.col;

                ThreadwiseGenericTensorSliceCopy_v4r2<decltype(c_g_m0_m1_m2_n_thread_desc),
                                                      decltype(c_g_m0_m1_m2_n_global_desc),
                                                      CThreadCopySliceLengths,
                                                      arithmetic_sequence_gen<0, 5, 1>::type,
                                                      4,
                                                      1,
                                                      1,
                                                      AddressSpace::Vgpr,
                                                      AddressSpace::Global,
                                                      CGlobalMemoryOp>(
                    {0, 0, 0, 0, 0},
                    {g_block_data_on_global,
                     m_thread_data_on_global / (M2 * M1),
                     m_thread_data_on_global % (M2 * M1) / M2,
                     m_thread_data_on_global % M2,
                     n_thread_data_on_global})
                    .Run(c_thread_vec.n + i * BlkSize, p_c_global);
            }
        }
    }
};

} // namespace ck
#endif
