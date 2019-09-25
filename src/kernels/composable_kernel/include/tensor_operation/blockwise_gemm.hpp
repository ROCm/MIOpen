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
          index_t EPack,
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
    __device__ void outerProduct(const typename vector_type<float, 4>::MemoryType& a,
                                 const typename vector_type<float, 4>::MemoryType& b,
                                 typename vector_type<float, 4>::MemoryType* c) const
    {
        constexpr index_t NRepeat = 2;

        outerProduct1x4(a.x, b, c[0 * NRepeat]);
        outerProduct1x4(a.y, b, c[1 * NRepeat]);
        outerProduct1x4(a.z, b, c[2 * NRepeat]);
        outerProduct1x4(a.w, b, c[3 * NRepeat]);
    }

    __device__ void outerProduct(const typename vector_type<float, 2>::MemoryType& a,
                                 const typename vector_type<float, 4>::MemoryType& b,
                                 typename vector_type<float, 4>::MemoryType* c) const
    {
        constexpr index_t NRepeat = 2;

        outerProduct1x4(a.x, b, c[0 * NRepeat]);
        outerProduct1x4(a.y, b, c[1 * NRepeat]);
    }

    __device__ void outerProduct(const typename vector_type<float, 4>::MemoryType& a,
                                 const typename vector_type<float, 2>::MemoryType& b,
                                 typename vector_type<float, 2>::MemoryType* c) const
    {
        constexpr index_t NRepeat = 2;

        outerProduct1x2(a.x, b, c[0 * NRepeat]);
        outerProduct1x2(a.y, b, c[1 * NRepeat]);
        outerProduct1x2(a.z, b, c[2 * NRepeat]);
        outerProduct1x2(a.w, b, c[3 * NRepeat]);
    }

    __device__ void outerProduct(const typename vector_type<float, 2>::MemoryType& a,
                                 const typename vector_type<float, 2>::MemoryType& b,
                                 typename vector_type<float, 2>::MemoryType* c) const
    {
        constexpr index_t NRepeat = 2;

        outerProduct1x2(a.x, b, c[0 * NRepeat]);
        outerProduct1x2(a.y, b, c[1 * NRepeat]);
    }

    template <index_t PACKSIZE>
    __device__ void
    outerProduct(const typename vector_type<typename vector_type<half, PACKSIZE>::MemoryType,
                                            4>::MemoryType& a,
                 const typename vector_type<typename vector_type<half, PACKSIZE>::MemoryType,
                                            4>::MemoryType& b,
                 typename vector_type<float, 4>::MemoryType* c) const
    {
        constexpr index_t NRepeat = 2;

        const typename vector_type<half, PACKSIZE>::MemoryType* reg_a =
            reinterpret_cast<const typename vector_type<half, PACKSIZE>::MemoryType*>(&a);
        outerProduct1x4Half<PACKSIZE>(reg_a[0], b, c[0 * NRepeat]);
        outerProduct1x4Half<PACKSIZE>(reg_a[1], b, c[1 * NRepeat]);
        outerProduct1x4Half<PACKSIZE>(reg_a[2], b, c[2 * NRepeat]);
        outerProduct1x4Half<PACKSIZE>(reg_a[3], b, c[3 * NRepeat]);
    }

    template <index_t PACKSIZE>
    __device__ void
    outerProduct(const typename vector_type<typename vector_type<half, PACKSIZE>::MemoryType,
                                            2>::MemoryType& a,
                 const typename vector_type<typename vector_type<half, PACKSIZE>::MemoryType,
                                            2>::MemoryType& b,
                 typename vector_type<float, 2>::MemoryType* c) const
    {
        constexpr index_t NRepeat = 2;

        const typename vector_type<half, PACKSIZE>::MemoryType* reg_a =
            reinterpret_cast<const typename vector_type<half, PACKSIZE>::MemoryType*>(&a);
        outerProduct1x2Half<PACKSIZE>(reg_a[0], b, c[0 * NRepeat]);
        outerProduct1x2Half<PACKSIZE>(reg_a[1], b, c[1 * NRepeat]);
    }

    template <index_t PACKSIZE>
    __device__ void
    outerProduct1x4Half(const typename vector_type<half, PACKSIZE>::MemoryType& a,
                        const typename vector_type<typename vector_type<half, PACKSIZE>::MemoryType,
                                                   4>::MemoryType& b,
                        vector_type<float, 4>::MemoryType& c) const
    {
        static_if<PACKSIZE == 4>{}([&](auto) {
            outerProduct1x4dot2TwoTimes(reinterpret_cast<const half2*>(&a),
                                        reinterpret_cast<const half2*>(&b),
                                        reinterpret_cast<float*>(&c));
        }).Else([&](auto) {
            static_if<PACKSIZE == 2>{}([&](auto) {
                outerProduct1x4dot2(reinterpret_cast<const half2*>(&a),
                                    reinterpret_cast<const half2*>(&b),
                                    reinterpret_cast<float*>(&c));
            }).Else([&](auto fwd) {
                // not implemented
                static_assert(fwd(false), "wrong! packsize = 1 for fp16 is insensible.");
            });
        });
    }

    template <index_t PACKSIZE>
    __device__ void
    outerProduct1x2Half(const typename vector_type<half, PACKSIZE>::MemoryType& a,
                        const typename vector_type<typename vector_type<half, PACKSIZE>::MemoryType,
                                                   2>::MemoryType& b,
                        vector_type<float, 2>::MemoryType& c) const
    {
        static_if<PACKSIZE == 4>{}([&](auto) {
            outerProduct1x2dot2TwoTimes(reinterpret_cast<const half2*>(&a),
                                        reinterpret_cast<const half2*>(&b),
                                        reinterpret_cast<float*>(&c));
        }).Else([&](auto) {
            static_if<PACKSIZE == 2>{}([&](auto) {
                outerProduct1x2dot2(reinterpret_cast<const half2*>(&a),
                                    reinterpret_cast<const half2*>(&b),
                                    reinterpret_cast<float*>(&c));
            }).Else([&](auto fwd) {
                // not implemented
                static_assert(fwd(false), "wrong! packsize = 1 for fp16 is insensible.");
            });
        });
    }

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
            make_ConstantMatrixDescriptor_packed(Number<KPerThreadLoop>{}, Number<MPerThread>{});

        constexpr auto b_thread_mtx =
            make_ConstantMatrixDescriptor_packed(Number<KPerThreadLoop>{}, Number<NPerThread>{});

        constexpr index_t MPerLevel1Cluster = MPerThreadSubC * MLevel0Cluster * MLevel1Cluster;
        constexpr index_t NPerLevel1Cluster = NPerThreadSubC * NLevel0Cluster * NLevel1Cluster;

        static_assert((MPerThreadSubC == 4 || MPerThreadSubC == 2) &&
                          (NPerThreadSubC == 4 || NPerThreadSubC == 2) && KPerThreadLoop == 1,
                      "M/NPerThreadSubC wrong!");

        static_assert(MPerThread % 4 == 0 && NPerThread % 4 == 0, "M/NPerThread % 4 != 0");

        constexpr index_t MRepeat = M / (MPerThreadSubC * MLevel0Cluster * MLevel1Cluster);
        constexpr index_t NRepeat = N / (NPerThreadSubC * NLevel0Cluster * NLevel1Cluster);

        static_assert(MRepeat == 2 && NRepeat == 2, "M/NRepeat != 2");

#if MIOPEN_USE_FP32 == 1
        using typeA = typename vector_type<float, MPerThreadSubC>::MemoryType;
        using typeB = typename vector_type<float, NPerThreadSubC>::MemoryType;

        FloatA p_a_thread[a_thread_mtx.GetElementSpace()];
        FloatB p_b_thread[b_thread_mtx.GetElementSpace()];

        typeA* reg_a = reinterpret_cast<typeA*>(p_a_thread);
        typeB* reg_b = reinterpret_cast<typeB*>(p_b_thread);
        typeB* reg_c = reinterpret_cast<typeB*>(p_c_thread);

        reg_a[0] = *reinterpret_cast<const typeA*>(&p_a_block[mMyThreadOffsetA]);
        reg_b[0] = *reinterpret_cast<const typeB*>(&p_b_block[mMyThreadOffsetB]);
        reg_b[1] =
            *reinterpret_cast<const typeB*>(&p_b_block[mMyThreadOffsetB + NPerLevel1Cluster]);
        reg_a[1] =
            *reinterpret_cast<const typeA*>(&p_a_block[mMyThreadOffsetA + MPerLevel1Cluster]);

        outerProduct(reg_a[0], reg_b[0], &reg_c[0]);
        outerProduct(reg_a[0], reg_b[1], &reg_c[1]);

#pragma unroll
        for(index_t k = 1; k < K; ++k)
        {
            reg_a[0] = *reinterpret_cast<const typeA*>(&p_a_block[mMyThreadOffsetA + k * M]);
            outerProduct(reg_a[1], reg_b[0], &reg_c[NRepeat * MPerThreadSubC]);
            reg_b[0] = *reinterpret_cast<const typeB*>(&p_b_block[mMyThreadOffsetB + k * N]);
            outerProduct(reg_a[1], reg_b[1], &reg_c[NRepeat * MPerThreadSubC + 1]);
            reg_b[1] = *reinterpret_cast<const typeB*>(
                &p_b_block[mMyThreadOffsetB + k * N + NPerLevel1Cluster]);
            reg_a[1] = *reinterpret_cast<const typeA*>(
                &p_a_block[mMyThreadOffsetA + k * M + MPerLevel1Cluster]);
            outerProduct(reg_a[0], reg_b[0], &reg_c[0]);
            outerProduct(reg_a[0], reg_b[1], &reg_c[1]);
        }
        outerProduct(reg_a[1], reg_b[0], &reg_c[NRepeat * MPerThreadSubC]);
        outerProduct(reg_a[1], reg_b[1], &reg_c[NRepeat * MPerThreadSubC + 1]);

#elif MIOPEN_USE_FP16 == 1 || MIOPEN_USE_BFP16 == 1

        FloatA p_a_thread[a_thread_mtx.GetElementSpace() * EPack];
        FloatB p_b_thread[b_thread_mtx.GetElementSpace() * EPack];

        using packedHalfType = typename vector_type<half, EPack>::MemoryType;
        using typeA          = typename vector_type<packedHalfType, MPerThreadSubC>::MemoryType;
        using typeB          = typename vector_type<packedHalfType, NPerThreadSubC>::MemoryType;
        using typeC          = typename vector_type<float, NPerThreadSubC>::MemoryType;

        typeA* reg_a = reinterpret_cast<typeA*>(p_a_thread);
        typeB* reg_b = reinterpret_cast<typeB*>(p_b_thread);
        typeC* reg_c = reinterpret_cast<typeC*>(p_c_thread);

        reg_a[0] = *reinterpret_cast<const typeA*>(&p_a_block[mMyThreadOffsetA * EPack]);
        reg_b[0] = *reinterpret_cast<const typeB*>(&p_b_block[mMyThreadOffsetB * EPack]);
        reg_b[1] = *reinterpret_cast<const typeB*>(
            &p_b_block[(mMyThreadOffsetB + NPerLevel1Cluster) * EPack]);
        reg_a[1] = *reinterpret_cast<const typeA*>(
            &p_a_block[(mMyThreadOffsetA + MPerLevel1Cluster) * EPack]);

        outerProduct<EPack>(reg_a[0], reg_b[0], &reg_c[0]);
        outerProduct<EPack>(reg_a[0], reg_b[1], &reg_c[1]);
#pragma unroll
        for(index_t k = 1; k < K; ++k)
        {
            reg_a[0] =
                *reinterpret_cast<const typeA*>(&p_a_block[(mMyThreadOffsetA + k * M) * EPack]);
            outerProduct<EPack>(reg_a[1], reg_b[0], &reg_c[NRepeat * MPerThreadSubC]);
            reg_b[0] =
                *reinterpret_cast<const typeB*>(&p_b_block[(mMyThreadOffsetB + k * N) * EPack]);
            outerProduct<EPack>(reg_a[1], reg_b[1], &reg_c[NRepeat * MPerThreadSubC + 1]);
            reg_b[1] = *reinterpret_cast<const typeB*>(
                &p_b_block[(mMyThreadOffsetB + k * N + NPerLevel1Cluster) * EPack]);
            reg_a[1] = *reinterpret_cast<const typeA*>(
                &p_a_block[(mMyThreadOffsetA + k * M + MPerLevel1Cluster) * EPack]);
            outerProduct<EPack>(reg_a[0], reg_b[0], &reg_c[0]);
            outerProduct<EPack>(reg_a[0], reg_b[1], &reg_c[1]);
        }
        outerProduct<EPack>(reg_a[1], reg_b[0], &reg_c[NRepeat * MPerThreadSubC]);
        outerProduct<EPack>(reg_a[1], reg_b[1], &reg_c[NRepeat * MPerThreadSubC + 1]);
#else
        static_assert(false, "wrong! Only float, fp16 and bfp16 datatypes are supported.");
#endif
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
        constexpr auto a_thread_mtx = make_ConstantMatrixDescriptor(
            Number<KPerThreadLoop>{}, Number<MPerThread>{}, Number<MPerThread>{});

        constexpr auto b_thread_mtx = make_ConstantMatrixDescriptor(
            Number<KPerThreadLoop>{}, Number<NPerThread>{}, Number<NPerThread>{});

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
    __device__ void Run(const FloatA* __restrict__ p_a_block,
                        const FloatB* __restrict__ p_b_block,
                        FloatC* __restrict__ p_c_thread) const

    {

// The assembly path doesn't support bfloat16 using asm instructions
#if CK_USE_AMD_INLINE_ASM && CK_BLOCKWISE_GEMM_USE_AMD_INLINE_ASM
#if MIOPEN_USE_BFP16 == 1
        Run_source(p_a_block, p_b_block, p_c_thread);
#else
        Run_amd_asm(p_a_block, p_b_block, p_c_thread);
#endif //
#else  // CK_USE_AMD_INLINE_ASM && CK_BLOCKWISE_GEMM_USE_AMD_INLINE_ASM
#if MIOPEN_USE_FP16 == 1
        // Vectorize the pointer to match with how half/bfloat16 datatypes are
        // processed in gemm operation. Half type packs 4 half values while
        // bfloat16 packs 2 bfloat16 values. Since gemm's matrix A and B
        // 2D indexes are computed with a single value in mind (e.g. float),
        // to retain the same 2D indexes for half/bfloat16, we recast datatype
        // from a single half to 4 packed half/2 packed bfloat16 respectively.
        const typename vector_type<half, EPack>::MemoryType* p_a_block_vec =
            reinterpret_cast<const typename vector_type<half, EPack>::MemoryType*>(p_a_block);
        const typename vector_type<half, EPack>::MemoryType* p_b_block_vec =
            reinterpret_cast<const typename vector_type<half, EPack>::MemoryType*>(p_b_block);
        Run_source(p_a_block_vec, p_b_block_vec, p_c_thread);

#else
        Run_source(p_a_block, p_b_block, p_c_thread);
#endif // MIOPEN_USE_FP16
#endif // CK_USE_AMD_INLINE_ASM && CK_BLOCKWISE_GEMM_USE_AMD_INLINE_ASM
    }
};

} // namespace ck
#endif
