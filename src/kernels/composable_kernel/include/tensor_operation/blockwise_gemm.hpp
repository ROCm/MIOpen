#ifndef CK_BLOCKWISE_GEMM_HPP
#define CK_BLOCKWISE_GEMM_HPP

#include "common_header.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "threadwise_gemm.hpp"

#ifndef CK_BLOCKWISE_GEMM_USE_AMD_INLINE_ASM
#define CK_BLOCKWISE_GEMM_USE_AMD_INLINE_ASM 1
#endif

namespace ck {

// if following number are power of 2, index calculation shall be greatly reduced:
//    MPerThreadSubC, NPerThreadSubC, MLevel0Cluster, NLevel0Cluster, MLevel1Cluster, NLevel1Cluster
template <index_t BlockSize,
          class BlockMatrixA,
          class BlockMatrixB,
          class ThreadMatrixC,
          index_t MPerThreadSubC,
          index_t NPerThreadSubC,
          index_t MLevel0Cluster,
          index_t NLevel0Cluster,
          index_t MLevel1Cluster,
          index_t NLevel1Cluster,
          index_t KPerThreadLoop,
          index_t DataPerReadA,
          index_t DataPerReadB>
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
        constexpr index_t ThreadPerLevel1Cluster =
            MLevel0Cluster * NLevel0Cluster * MLevel1Cluster * NLevel1Cluster;

        static_assert(BlockSize == ThreadPerLevel1Cluster, "wrong! wrong blocksize\n");

        static_assert(BlockMatrixA::NRow() == BlockMatrixB::NRow(),
                      "wrong! K dimension not consistent\n");

        constexpr index_t M = BlockMatrixA::NCol(); // A is transposed
        constexpr index_t N = BlockMatrixB::NCol();

        static_assert(M % (MPerThreadSubC * MLevel0Cluster * MLevel1Cluster) == 0 &&
                          N % (NPerThreadSubC * NLevel0Cluster * NLevel1Cluster) == 0,
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

        constexpr index_t MRepeat = M / (MPerThreadSubC * MLevel0Cluster * MLevel1Cluster);
        constexpr index_t NRepeat = N / (NPerThreadSubC * NLevel0Cluster * NLevel1Cluster);

        return Sequence<MRepeat * MPerThreadSubC, NRepeat * NPerThreadSubC>{};
    }

    __device__ static MatrixIndex GetBeginOfThreadMatrixC(index_t thread_id)
    {
        constexpr index_t ThreadPerLevel0Cluster = MLevel0Cluster * NLevel0Cluster;

        index_t level1_id   = thread_id / ThreadPerLevel0Cluster;
        index_t level1_m_id = level1_id / NLevel1Cluster;
        index_t level1_n_id = level1_id % NLevel1Cluster;

        index_t level0_id   = thread_id % ThreadPerLevel0Cluster;
        index_t level0_m_id = level0_id / NLevel0Cluster;
        index_t level0_n_id = level0_id % NLevel0Cluster;

        constexpr index_t MPerLevel0Cluster = MPerThreadSubC * MLevel0Cluster;
        constexpr index_t NPerLevel0Cluster = NPerThreadSubC * NLevel0Cluster;

        return MatrixIndex{level1_m_id * MPerLevel0Cluster + level0_m_id * MPerThreadSubC,
                           level1_n_id * NPerLevel0Cluster + level0_n_id * NPerThreadSubC};
    }

    __device__ static MatrixIndex GetDistanceFromBeginOfThreadMatrixC(index_t m_in_c,
                                                                      index_t n_in_c)
    {
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        constexpr index_t MPerLevel1Cluster = MPerThreadSubC * MLevel0Cluster * MLevel1Cluster;
        constexpr index_t NPerLevel1Cluster = NPerThreadSubC * NLevel0Cluster * NLevel1Cluster;

        index_t m_repeat = m_in_c / MPerThreadSubC;
        index_t n_repeat = n_in_c / NPerThreadSubC;

        index_t m_in_sub_c = m_in_c % MPerThreadSubC;
        index_t n_in_sub_c = n_in_c % NPerThreadSubC;

        return MatrixIndex{m_repeat * MPerLevel1Cluster + m_in_sub_c,
                           n_repeat * NPerLevel1Cluster + n_in_sub_c};
    }

#if CK_USE_AMD_INLINE_ASM
    template <class FloatA, class FloatB, class FloatC>
    __device__ void Run_amd_asm(const FloatA* __restrict__ p_a_block,
                                const FloatB* __restrict__ p_b_block,
                                FloatC* __restrict__ p_c_thread) const
    {
        constexpr auto a_block_mtx  = BlockMatrixA{};
        constexpr auto b_block_mtx  = BlockMatrixB{};
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        constexpr index_t M = a_block_mtx.NCol();
        constexpr index_t N = b_block_mtx.NCol();
        constexpr index_t K = a_block_mtx.NRow();

        constexpr index_t MPerThread = c_thread_mtx.NRow();
        constexpr index_t NPerThread = c_thread_mtx.NCol();

        // thread A, B for GEMM
        constexpr auto a_thread_mtx =
            make_ConstantMatrixDescriptor(Number<KPerThreadLoop>{}, Number<MPerThread>{});

        constexpr auto b_thread_mtx =
            make_ConstantMatrixDescriptor(Number<KPerThreadLoop>{}, Number<NPerThread>{});

        FloatA p_a_thread[a_thread_mtx.GetElementSpace()];
        FloatB p_b_thread[b_thread_mtx.GetElementSpace()];

        constexpr index_t MPerLevel1Cluster = MPerThreadSubC * MLevel0Cluster * MLevel1Cluster;
        constexpr index_t NPerLevel1Cluster = NPerThreadSubC * NLevel0Cluster * NLevel1Cluster;

        // assertion for inline asm
        static_assert(is_same<FloatA, float>{} && is_same<FloatB, float>{} &&
                          is_same<FloatC, float>{},
                      "Run_amd_asm only deal with float");

        static_assert(MPerThreadSubC == 4 && NPerThreadSubC == 4 && KPerThreadLoop == 1 &&
                          MPerThread == 8 && NPerThread == 8,
                      "Run_amd_asm cannot deal with this GEMM shape yet");

        static_assert(DataPerReadA == 4 && DataPerReadB == 4, "Run_amd_asm only do float4 read");

        using Float4 = vector_type<float, 4>::MemoryType;

        Float4* reg_a = reinterpret_cast<Float4*>(p_a_thread);
        Float4* reg_b = reinterpret_cast<Float4*>(p_b_thread);
        Float4* reg_c = reinterpret_cast<Float4*>(p_c_thread);

        reg_a[0] = *reinterpret_cast<const Float4*>(&p_a_block[mMyThreadOffsetA]);
        reg_b[0] = *reinterpret_cast<const Float4*>(&p_b_block[mMyThreadOffsetB]);
        reg_b[1] =
            *reinterpret_cast<const Float4*>(&p_b_block[mMyThreadOffsetB + NPerLevel1Cluster]);
        reg_a[1] =
            *reinterpret_cast<const Float4*>(&p_a_block[mMyThreadOffsetA + MPerLevel1Cluster]);
        outerProduct4x4(reg_a[0], reg_b[0], reg_c[0], reg_c[2], reg_c[4], reg_c[6]);
        outerProduct4x4(reg_a[0], reg_b[1], reg_c[1], reg_c[3], reg_c[5], reg_c[7]);
#pragma unroll
        for(index_t k = 1; k < K; ++k)
        {
            reg_a[0] = *reinterpret_cast<const Float4*>(&p_a_block[mMyThreadOffsetA + k * M]);
            outerProduct4x4(reg_a[1], reg_b[0], reg_c[8], reg_c[10], reg_c[12], reg_c[14]);
            reg_b[0] = *reinterpret_cast<const Float4*>(&p_b_block[mMyThreadOffsetB + k * N]);
            outerProduct4x4(reg_a[1], reg_b[1], reg_c[9], reg_c[11], reg_c[13], reg_c[15]);
            reg_b[1] = *reinterpret_cast<const Float4*>(
                &p_b_block[mMyThreadOffsetB + k * N + NPerLevel1Cluster]);
            reg_a[1] = *reinterpret_cast<const Float4*>(
                &p_a_block[mMyThreadOffsetA + k * M + MPerLevel1Cluster]);
            outerProduct4x4(reg_a[0], reg_b[0], reg_c[0], reg_c[2], reg_c[4], reg_c[6]);
            outerProduct4x4(reg_a[0], reg_b[1], reg_c[1], reg_c[3], reg_c[5], reg_c[7]);
        }
        outerProduct4x4(reg_a[1], reg_b[0], reg_c[8], reg_c[10], reg_c[12], reg_c[14]);
        outerProduct4x4(reg_a[1], reg_b[1], reg_c[9], reg_c[11], reg_c[13], reg_c[15]);
    }
#endif

    template <class FloatA, class FloatB, class FloatC>
    __device__ void Run_source(const FloatA* const __restrict__ p_a_block,
                               const FloatB* const __restrict__ p_b_block,
                               FloatC* const __restrict__ p_c_thread) const
    {
        constexpr auto True  = integral_constant<bool, true>{};
        constexpr auto False = integral_constant<bool, false>{};

        constexpr auto a_block_mtx  = BlockMatrixA{};
        constexpr auto b_block_mtx  = BlockMatrixB{};
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        constexpr index_t K = a_block_mtx.NRow();

        constexpr index_t MPerThread = c_thread_mtx.NRow();
        constexpr index_t NPerThread = c_thread_mtx.NCol();

        // thread A, B for GEMM
        constexpr auto a_thread_mtx =
            make_ConstantMatrixDescriptor(Number<KPerThreadLoop>{}, Number<MPerThread>{});

        constexpr auto b_thread_mtx =
            make_ConstantMatrixDescriptor(Number<KPerThreadLoop>{}, Number<NPerThread>{});

        // thread A-sub, B-sub for copy
        constexpr auto a_thread_sub_mtx = make_ConstantMatrixDescriptor(
            Number<KPerThreadLoop>{}, Number<MPerThreadSubC>{}, Number<MPerThread>{});

        constexpr auto b_thread_sub_mtx = make_ConstantMatrixDescriptor(
            Number<KPerThreadLoop>{}, Number<NPerThreadSubC>{}, Number<NPerThread>{});

        FloatA p_a_thread[a_thread_mtx.GetElementSpace()];
        FloatB p_b_thread[b_thread_mtx.GetElementSpace()];

        constexpr index_t MPerLevel1Cluster = MPerThreadSubC * MLevel0Cluster * MLevel1Cluster;
        constexpr index_t NPerLevel1Cluster = NPerThreadSubC * NLevel0Cluster * NLevel1Cluster;

        constexpr index_t MRepeat = MPerThread / MPerThreadSubC;
        constexpr index_t NRepeat = NPerThread / NPerThreadSubC;

#pragma unroll
        // loop over k
        for(index_t k_begin = 0; k_begin < K; k_begin += KPerThreadLoop)
        {
#pragma unroll
            // copy A-sub to form A
            for(index_t m_repeat = 0; m_repeat < MRepeat; ++m_repeat)
            {
                threadwise_matrix_copy(
                    a_block_mtx,
                    p_a_block +
                        a_block_mtx.GetOffsetFromMultiIndex(k_begin, m_repeat * MPerLevel1Cluster) +
                        mMyThreadOffsetA,
                    a_thread_mtx,
                    p_a_thread + a_thread_mtx.GetOffsetFromMultiIndex(0, m_repeat * MPerThreadSubC),
                    a_thread_sub_mtx.GetLengths(),
                    Number<DataPerReadA>{});
            }

#pragma unroll
            // copy B-sub to form B
            for(index_t n_repeat = 0; n_repeat < NRepeat; ++n_repeat)
            {
                threadwise_matrix_copy(
                    b_block_mtx,
                    p_b_block +
                        b_block_mtx.GetOffsetFromMultiIndex(k_begin, n_repeat * NPerLevel1Cluster) +
                        mMyThreadOffsetB,
                    b_thread_mtx,
                    p_b_thread + b_thread_mtx.GetOffsetFromMultiIndex(0, n_repeat * NPerThreadSubC),
                    b_thread_sub_mtx.GetLengths(),
                    Number<DataPerReadB>{});
            }

            // C = A * B
            threadwise_gemm(a_thread_mtx,
                            True,
                            p_a_thread,
                            b_thread_mtx,
                            False,
                            p_b_thread,
                            c_thread_mtx,
                            False,
                            p_c_thread);
        }
    }

    template <class FloatA, class FloatB, class FloatC>
    __device__ void RunRegisterDoubleBuffer_source(FloatA* const p_a_block,
                                                   FloatB* const p_b_block,
                                                   FloatC* p_c_thread) const
    {
        constexpr auto True  = integral_constant<bool, true>{};
        constexpr auto False = integral_constant<bool, false>{};

        constexpr auto a_block_mtx  = BlockMatrixA{};
        constexpr auto b_block_mtx  = BlockMatrixB{};
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        constexpr index_t K = a_block_mtx.NRow();

        constexpr index_t MPerThread = c_thread_mtx.NRow();
        constexpr index_t NPerThread = c_thread_mtx.NCol();

        // thread A, B for GEMM
        constexpr auto a_thread_mtx =
            make_ConstantMatrixDescriptor(Number<KPerThreadLoop>{}, Number<MPerThread>{});

        constexpr auto b_thread_mtx =
            make_ConstantMatrixDescriptor(Number<KPerThreadLoop>{}, Number<NPerThread>{});

        // thread A-sub, B-sub for copy
        constexpr auto a_thread_sub_mtx = make_ConstantMatrixDescriptor(
            Number<KPerThreadLoop>{}, Number<MPerThreadSubC>{}, Number<MPerThread>{});

        constexpr auto b_thread_sub_mtx = make_ConstantMatrixDescriptor(
            Number<KPerThreadLoop>{}, Number<NPerThreadSubC>{}, Number<NPerThread>{});

        // register
        FloatA p_a_thread_0[a_thread_mtx.GetElementSpace()];
        FloatB p_b_thread_0[b_thread_mtx.GetElementSpace()];

        FloatA p_a_thread_1[a_thread_mtx.GetElementSpace()];
        FloatB p_b_thread_1[b_thread_mtx.GetElementSpace()];

        constexpr index_t MPerLevel1Cluster = MPerThreadSubC * MLevel0Cluster * MLevel1Cluster;
        constexpr index_t NPerLevel1Cluster = NPerThreadSubC * NLevel0Cluster * NLevel1Cluster;

        constexpr index_t MRepeat = MPerThread / MPerThreadSubC;
        constexpr index_t NRepeat = NPerThread / NPerThreadSubC;

// preload A, B
#pragma unroll
        for(index_t m_repeat = 0; m_repeat < MRepeat; ++m_repeat)
        { // copy A-sub to form A
            threadwise_matrix_copy(a_block_mtx,
                                   p_a_block + mMyThreadOffsetA + m_repeat * MPerLevel1Cluster,
                                   a_thread_sub_mtx,
                                   p_a_thread_0 + m_repeat * MPerThreadSubC,
                                   a_thread_sub_mtx.GetLengths(),
                                   Number<DataPerReadA>{});
        }

#pragma unroll
        for(index_t n_repeat = 0; n_repeat < NRepeat; ++n_repeat)
        { // copy B-sub to form B
            threadwise_matrix_copy(b_block_mtx,
                                   p_b_block + mMyThreadOffsetB + n_repeat * NPerLevel1Cluster,
                                   b_thread_sub_mtx,
                                   p_b_thread_0 + n_repeat * NPerThreadSubC,
                                   b_thread_sub_mtx.GetLengths(),
                                   Number<DataPerReadB>{});
        }

        bool even_loop = true;

#pragma unroll
        for(index_t k_begin = 0; k_begin + KPerThreadLoop < K;
            k_begin += KPerThreadLoop, even_loop = !even_loop)
        { // loop over k
            FloatA* p_a_thread_now = even_loop ? p_a_thread_0 : p_a_thread_1;
            FloatB* p_b_thread_now = even_loop ? p_b_thread_0 : p_b_thread_1;

            FloatA* p_a_thread_next = even_loop ? p_a_thread_1 : p_a_thread_0;
            FloatB* p_b_thread_next = even_loop ? p_b_thread_1 : p_b_thread_0;

// preload next A, B
#pragma unroll
            for(index_t m_repeat = 0; m_repeat < MRepeat; ++m_repeat)
            { // copy A-sub to form A
                threadwise_matrix_copy(a_block_mtx,
                                       p_a_block + mMyThreadOffsetA +
                                           (k_begin + 1) * a_block_mtx.RowStride() +
                                           m_repeat * MPerLevel1Cluster,
                                       a_thread_sub_mtx,
                                       p_a_thread_next + m_repeat * MPerThreadSubC,
                                       a_thread_sub_mtx.GetLengths(),
                                       Number<DataPerReadA>{});
            }

#pragma unroll
            for(index_t n_repeat = 0; n_repeat < NRepeat; ++n_repeat)
            { // copy B-sub to form B
                threadwise_matrix_copy(b_block_mtx,
                                       p_b_block + mMyThreadOffsetB +
                                           (k_begin + 1) * b_block_mtx.RowStride() +
                                           n_repeat * NPerLevel1Cluster,
                                       b_thread_sub_mtx,
                                       p_b_thread_next + n_repeat * NPerThreadSubC,
                                       b_thread_sub_mtx.GetLengths(),
                                       Number<DataPerReadB>{});
            }

            // C = A * B
            threadwise_gemm(a_thread_mtx,
                            True,
                            p_a_thread_now,
                            b_thread_mtx,
                            False,
                            p_b_thread_now,
                            c_thread_mtx,
                            False,
                            p_c_thread);
        }

        // last loop
        {
            FloatA* p_a_thread_now = even_loop ? p_a_thread_0 : p_a_thread_1;
            FloatB* p_b_thread_now = even_loop ? p_b_thread_0 : p_b_thread_1;

            // C = A * B
            threadwise_gemm(a_thread_mtx,
                            True,
                            p_a_thread_now,
                            b_thread_mtx,
                            False,
                            p_b_thread_now,
                            c_thread_mtx,
                            False,
                            p_c_thread);
        }
    }
    template <class FloatA, class FloatB, class FloatC>
    __device__ void Run(const FloatA* __restrict__ p_a_block,
                        const FloatB* __restrict__ p_b_block,
                        FloatC* __restrict__ p_c_thread) const

    {
#if CK_USE_AMD_INLINE_ASM && CK_BLOCKWISE_GEMM_USE_AMD_INLINE_ASM
        Run_amd_asm(p_a_block, p_b_block, p_c_thread);
#else
        Run_source(p_a_block, p_b_block, p_c_thread);
#endif
    }
};

} // namespace ck
#endif
