#ifndef CK_BLOCKWISE_GEMM_HPP
#define CK_BLOCKWISE_GEMM_HPP

#include "common_header.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "threadwise_gemm.hpp"

namespace ck {

// blockwise GEMM: C += transpose(A) * B
// A and B are visable to the whole block, C is distributed among each thread
// If following number are power of 2, index calculation shall be greatly reduced:
//    MPerThreadSubC, NPerThreadSubC, MLevel0ThreadCluster, NLevel0ThreadCluster,
//    MLevel1ThreadCluster, NLevel1ThreadCluster
template <index_t BlockSize,
          typename BlockMatrixA,
          typename BlockMatrixB,
          typename ThreadMatrixC,
          index_t MPerThreadSubC,
          index_t NPerThreadSubC,
          index_t MLevel0ThreadCluster,
          index_t NLevel0ThreadCluster,
          index_t MLevel1ThreadCluster,
          index_t NLevel1ThreadCluster,
          index_t KPerThreadLoop,
          index_t ThreadGemmADataPerRead_M,
          index_t ThreadGemmBDataPerRead_N>
struct BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_v2
{
    struct MatrixIndex
    {
        index_t row;
        index_t col;
    };

    index_t mMyThreadOffsetA;
    index_t mMyThreadOffsetB;

    __device__ BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_v2()
    {
        constexpr index_t ThreadPerLevel1Cluster = MLevel0ThreadCluster * NLevel0ThreadCluster *
                                                   MLevel1ThreadCluster * NLevel1ThreadCluster;

        static_assert(BlockSize == ThreadPerLevel1Cluster, "wrong! wrong blocksize\n");

        static_assert(BlockMatrixA::NRow() == BlockMatrixB::NRow(),
                      "wrong! K dimension not consistent\n");

        constexpr index_t M = BlockMatrixA::NCol(); // A is transposed
        constexpr index_t N = BlockMatrixB::NCol();

        static_assert(M % (MPerThreadSubC * MLevel0ThreadCluster * MLevel1ThreadCluster) == 0 &&
                          N % (NPerThreadSubC * NLevel0ThreadCluster * NLevel1ThreadCluster) == 0,
                      "wrong! Cannot evenly divide work among\n");

        static_assert(
            is_same<decltype(ThreadMatrixC::GetLengths()), decltype(GetThreadMatrixCLengths())>{},
            "wrong! ThreadMatrixC lengths is wrong");

        auto c_thread_mtx_index = GetBeginOfThreadMatrixC(get_thread_local_1d_id());

        mMyThreadOffsetA = BlockMatrixA::GetOffsetFromMultiIndex(0, c_thread_mtx_index.row);
        mMyThreadOffsetB = BlockMatrixB::GetOffsetFromMultiIndex(0, c_thread_mtx_index.col);
    }

    __device__ static constexpr auto GetThreadMatrixCLengths()
    {
        constexpr index_t M = BlockMatrixA::NCol(); // A is transposed
        constexpr index_t N = BlockMatrixB::NCol();

        constexpr index_t MRepeat =
            M / (MPerThreadSubC * MLevel0ThreadCluster * MLevel1ThreadCluster);
        constexpr index_t NRepeat =
            N / (NPerThreadSubC * NLevel0ThreadCluster * NLevel1ThreadCluster);

        return Sequence<MRepeat * MPerThreadSubC, NRepeat * NPerThreadSubC>{};
    }

    __device__ static MatrixIndex GetBeginOfThreadMatrixC(index_t thread_id)
    {
        constexpr index_t ThreadPerLevel0Cluster = MLevel0ThreadCluster * NLevel0ThreadCluster;

        index_t level1_id   = thread_id / ThreadPerLevel0Cluster;
        index_t level1_m_id = level1_id / NLevel1ThreadCluster;
        index_t level1_n_id = level1_id % NLevel1ThreadCluster;

        index_t level0_id   = thread_id % ThreadPerLevel0Cluster;
        index_t level0_m_id = level0_id / NLevel0ThreadCluster;
        index_t level0_n_id = level0_id % NLevel0ThreadCluster;

        constexpr index_t MPerLevel0Cluster = MPerThreadSubC * MLevel0ThreadCluster;
        constexpr index_t NPerLevel0Cluster = NPerThreadSubC * NLevel0ThreadCluster;

        return MatrixIndex{level1_m_id * MPerLevel0Cluster + level0_m_id * MPerThreadSubC,
                           level1_n_id * NPerLevel0Cluster + level0_n_id * NPerThreadSubC};
    }

    __device__ static MatrixIndex GetDistanceFromBeginOfThreadMatrixC(index_t m_in_c,
                                                                      index_t n_in_c)
    {
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        constexpr index_t MPerLevel1Cluster =
            MPerThreadSubC * MLevel0ThreadCluster * MLevel1ThreadCluster;
        constexpr index_t NPerLevel1Cluster =
            NPerThreadSubC * NLevel0ThreadCluster * NLevel1ThreadCluster;

        index_t m_repeat = m_in_c / MPerThreadSubC;
        index_t n_repeat = n_in_c / NPerThreadSubC;

        index_t m_in_sub_c = m_in_c % MPerThreadSubC;
        index_t n_in_sub_c = n_in_c % NPerThreadSubC;

        return MatrixIndex{m_repeat * MPerLevel1Cluster + m_in_sub_c,
                           n_repeat * NPerLevel1Cluster + n_in_sub_c};
    }

    template <typename FloatA, typename FloatB, typename FloatC>
    __device__ void
    Run_naive(const FloatA* p_a_block, const FloatB* p_b_block, FloatC* p_c_thread) const
    {
        constexpr auto a_block_mtx  = BlockMatrixA{};
        constexpr auto b_block_mtx  = BlockMatrixB{};
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        constexpr index_t K = a_block_mtx.NRow();

        constexpr index_t MPerThread = c_thread_mtx.NRow();
        constexpr index_t NPerThread = c_thread_mtx.NCol();

        constexpr index_t MPerLevel1Cluster =
            MPerThreadSubC * MLevel0ThreadCluster * MLevel1ThreadCluster;
        constexpr index_t NPerLevel1Cluster =
            NPerThreadSubC * NLevel0ThreadCluster * NLevel1ThreadCluster;

        constexpr index_t MRepeat = MPerThread / MPerThreadSubC;
        constexpr index_t NRepeat = NPerThread / NPerThreadSubC;

        // thread A, B for GEMM
        constexpr auto a_thread_mtx =
            make_ConstantMatrixDescriptor_packed(Number<KPerThreadLoop>{}, Number<MPerThread>{});

        constexpr auto b_thread_mtx =
            make_ConstantMatrixDescriptor_packed(Number<KPerThreadLoop>{}, Number<NPerThread>{});

        FloatA p_a_thread[a_thread_mtx.GetElementSpace()];
        FloatB p_b_thread[b_thread_mtx.GetElementSpace()];

        constexpr auto a_thread_copy = ThreadwiseMatrixSliceCopy<BlockMatrixA,
                                                                 decltype(a_thread_mtx),
                                                                 KPerThreadLoop,
                                                                 MPerThreadSubC,
                                                                 ThreadGemmADataPerRead_M>{};

        constexpr auto b_thread_copy = ThreadwiseMatrixSliceCopy<BlockMatrixB,
                                                                 decltype(b_thread_mtx),
                                                                 KPerThreadLoop,
                                                                 NPerThreadSubC,
                                                                 ThreadGemmBDataPerRead_N>{};

        constexpr auto threadwise_gemm =
            ThreadwiseGemmTransANormalBNormalC<decltype(a_thread_mtx),
                                               decltype(b_thread_mtx),
                                               decltype(c_thread_mtx)>{};
#pragma unroll
        // loop over k
        for(index_t k_begin = 0; k_begin < K; k_begin += KPerThreadLoop)
        {
#pragma unroll
            // read A
            for(index_t m_repeat = 0; m_repeat < MRepeat; ++m_repeat)
            {
                a_thread_copy.Run(
                    p_a_block + a_block_mtx.CalculateOffset(k_begin, m_repeat * MPerLevel1Cluster) +
                        mMyThreadOffsetA,
                    p_a_thread + a_thread_mtx.CalculateOffset(0, m_repeat * MPerThreadSubC));
            }

#pragma unroll
            // read B
            for(index_t n_repeat = 0; n_repeat < NRepeat; ++n_repeat)
            {
                b_thread_copy.Run(
                    p_b_block + b_block_mtx.CalculateOffset(k_begin, n_repeat * NPerLevel1Cluster) +
                        mMyThreadOffsetB,
                    p_b_thread + b_thread_mtx.CalculateOffset(0, n_repeat * NPerThreadSubC));
            }

            // C += A * B
            threadwise_gemm.Run(p_a_thread, p_b_thread, p_c_thread);
        }
    }

    template <typename FloatA, typename FloatB, typename FloatC>
    __device__ void
    Run_pipelined_2x2(const FloatA* p_a_block, const FloatB* p_b_block, FloatC* p_c_thread) const
    {
        constexpr auto a_block_mtx  = BlockMatrixA{};
        constexpr auto b_block_mtx  = BlockMatrixB{};
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        constexpr index_t K = a_block_mtx.NRow();

        constexpr index_t MPerThread = c_thread_mtx.NRow();
        constexpr index_t NPerThread = c_thread_mtx.NCol();

        constexpr index_t MPerLevel1Cluster =
            MPerThreadSubC * MLevel0ThreadCluster * MLevel1ThreadCluster;
        constexpr index_t NPerLevel1Cluster =
            NPerThreadSubC * NLevel0ThreadCluster * NLevel1ThreadCluster;

        constexpr index_t MRepeat = MPerThread / MPerThreadSubC;
        constexpr index_t NRepeat = NPerThread / NPerThreadSubC;

        static_assert(MRepeat == 2 && NRepeat == 2,
                      "wrong! inline asm cannot deal with this GEMM config yet");

        // thread A, B
        constexpr auto a_thread_mtx =
            make_ConstantMatrixDescriptor_packed(Number<KPerThreadLoop>{}, Number<MPerThread>{});
        constexpr auto b_thread_mtx =
            make_ConstantMatrixDescriptor_packed(Number<KPerThreadLoop>{}, Number<NPerThread>{});

        // thread A-sub, B-sub
        constexpr auto a_thread_sub_mtx = a_thread_mtx.MakeSubMatrixDescriptor(
            Number<KPerThreadLoop>{}, Number<MPerThreadSubC>{});
        constexpr auto b_thread_sub_mtx = b_thread_mtx.MakeSubMatrixDescriptor(
            Number<KPerThreadLoop>{}, Number<NPerThreadSubC>{});

        // thread C-sub
        constexpr auto c_thread_sub_mtx = ThreadMatrixC::MakeSubMatrixDescriptor(
            Number<MPerThreadSubC>{}, Number<NPerThreadSubC>{});

        FloatA p_a_thread[a_thread_mtx.GetElementSpace()];
        FloatB p_b_thread[b_thread_mtx.GetElementSpace()];

        constexpr auto a_thread_copy = ThreadwiseMatrixSliceCopy<BlockMatrixA,
                                                                 decltype(a_thread_mtx),
                                                                 KPerThreadLoop,
                                                                 MPerThreadSubC,
                                                                 ThreadGemmADataPerRead_M>{};

        constexpr auto b_thread_copy = ThreadwiseMatrixSliceCopy<BlockMatrixB,
                                                                 decltype(b_thread_mtx),
                                                                 KPerThreadLoop,
                                                                 NPerThreadSubC,
                                                                 ThreadGemmBDataPerRead_N>{};

        constexpr auto threadwise_gemm =
            ThreadwiseGemmTransANormalBNormalC<decltype(a_thread_sub_mtx),
                                               decltype(b_thread_sub_mtx),
                                               decltype(c_thread_sub_mtx)>{};

        const FloatA* p_a_block_off = p_a_block + mMyThreadOffsetA;
        const FloatB* p_b_block_off = p_b_block + mMyThreadOffsetB;

        // read A_sub_0
        a_thread_copy.Run(p_a_block_off, p_a_thread);

        // read B_sub_0
        b_thread_copy.Run(p_b_block_off, p_b_thread);

        // read B_sub_1
        b_thread_copy.Run(p_b_block_off + b_block_mtx.CalculateOffset(0, NPerLevel1Cluster),
                          p_b_thread + b_thread_mtx.CalculateOffset(0, NPerThreadSubC));

        // read A_sub_1
        a_thread_copy.Run(p_a_block_off + a_block_mtx.CalculateOffset(0, MPerLevel1Cluster),
                          p_a_thread + a_thread_mtx.CalculateOffset(0, MPerThreadSubC));

        // C_sub_00 += transpose(A_sub_0) * B_sub_0
        threadwise_gemm.Run(p_a_thread, p_b_thread, p_c_thread);

        // C_sub_01 += transpose(A_sub_0) * B_sub_1
        threadwise_gemm.Run(p_a_thread,
                            p_b_thread + b_thread_mtx.CalculateOffset(0, NPerThreadSubC),
                            p_c_thread + ThreadMatrixC::CalculateOffset(0, NPerThreadSubC));

#pragma unroll
        // loop over rest of k
        for(index_t k = KPerThreadLoop; k < K; k += KPerThreadLoop)
        {
            // read A_sub_0
            a_thread_copy.Run(p_a_block_off + a_block_mtx.CalculateOffset(k, 0), p_a_thread);

            // C_sub_10 += transpose(A_sub_1) * B_sub_0
            threadwise_gemm.Run(p_a_thread + a_thread_mtx.CalculateOffset(0, MPerThreadSubC),
                                p_b_thread,
                                p_c_thread + ThreadMatrixC::CalculateOffset(MPerThreadSubC, 0));

            // read B_sub_0
            b_thread_copy.Run(p_b_block_off + b_block_mtx.CalculateOffset(k, 0), p_b_thread);

            // C_sub_11 += transpose(A_sub_1) * B_sub_1
            threadwise_gemm.Run(p_a_thread + a_thread_mtx.CalculateOffset(0, MPerThreadSubC),
                                p_b_thread + b_thread_mtx.CalculateOffset(0, NPerThreadSubC),
                                p_c_thread +
                                    ThreadMatrixC::CalculateOffset(MPerThreadSubC, NPerThreadSubC));

            // read B_sub_1
            b_thread_copy.Run(p_b_block_off + b_block_mtx.CalculateOffset(k, NPerLevel1Cluster),
                              p_b_thread + b_thread_mtx.CalculateOffset(0, NPerThreadSubC));

            // read A_sub_1
            a_thread_copy.Run(p_a_block_off + a_block_mtx.CalculateOffset(k, MPerLevel1Cluster),
                              p_a_thread + a_thread_mtx.CalculateOffset(0, MPerThreadSubC));

            // C_sub_00 += transpose(A_sub_0) * B_sub_0
            threadwise_gemm.Run(p_a_thread, p_b_thread, p_c_thread);

            // C_sub_01 += transpose(A_sub_0) * B_sub_1
            threadwise_gemm.Run(p_a_thread,
                                p_b_thread + b_thread_mtx.CalculateOffset(0, NPerThreadSubC),
                                p_c_thread + ThreadMatrixC::CalculateOffset(0, NPerThreadSubC));
        }

        // C_sub_10 += transpose(A_sub_1) * B_sub_0
        threadwise_gemm.Run(p_a_thread + a_thread_mtx.CalculateOffset(0, MPerThreadSubC),
                            p_b_thread,
                            p_c_thread + ThreadMatrixC::CalculateOffset(MPerThreadSubC, 0));

        // C_sub_11 += transpose(A_sub_1) * B_sub_1
        threadwise_gemm.Run(p_a_thread + a_thread_mtx.CalculateOffset(0, MPerThreadSubC),
                            p_b_thread + b_thread_mtx.CalculateOffset(0, NPerThreadSubC),
                            p_c_thread +
                                ThreadMatrixC::CalculateOffset(MPerThreadSubC, NPerThreadSubC));
    }

    template <typename FloatA, typename FloatB, typename FloatC>
    __device__ void Run(const FloatA* p_a_block, const FloatB* p_b_block, FloatC* p_c_thread) const
    {
#if CK_EXPERIMENTAL_BLOCKWISE_GEMM_USE_PIPELINE
        constexpr index_t MPerThread = ThreadMatrixC::NRow();
        constexpr index_t NPerThread = ThreadMatrixC::NCol();

        constexpr index_t MRepeat = MPerThread / MPerThreadSubC;
        constexpr index_t NRepeat = NPerThread / NPerThreadSubC;

        static_if<MRepeat == 2 && NRepeat == 2>{}([&](auto) {
            Run_pipelined_2x2(p_a_block, p_b_block, p_c_thread);
        }).Else([&](auto) { Run_naive(p_a_block, p_b_block, p_c_thread); });
#else
        Run_naive(p_a_block, p_b_block, p_c_thread);
#endif
    }
};

} // namespace ck
#endif
