#ifndef CK_BLOCKWISE_GEMM_XDLOPS_HPP
#define CK_BLOCKWISE_GEMM_XDLOPS_HPP

#include "static_kernel_common_header.hpp"
#include "static_kernel_ConstantMatrixDescriptor.hpp"
#include "static_kernel_xdlops_gemm.hpp"
#include "static_kernel_xdlops_gemm_inline_asm.hpp"
#include "static_kernel_threadwise_gemm.hpp"

namespace ck {

template <index_t BlockSize,
          class BlockMatrixA,
          class BlockMatrixB,
          class Float,
          index_t GemmMPerWave,
          index_t GemmNPerWave,
          index_t GemmMWaves,
          index_t GemmNWaves,
          index_t GemmDataPerReadA, // \todo unused parameter, remove
          index_t GemmDataPerReadB  // \todo unused parameter, remove
          >
struct BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_xdlops
{
    struct MatrixIndex
    {
        index_t row;
        index_t col;
    };

#if CK_WORKAROUND_SWDEV_241664
    static constexpr index_t MRepeats = (GemmMPerWave > 64) ? (GemmMPerWave / 64) : 1;
    static constexpr index_t NRepeats = (GemmNPerWave > 64) ? (GemmNPerWave / 64) : 1;

    static constexpr index_t MPerXdlops = (GemmMPerWave > 64) ? 64 : GemmMPerWave;
    static constexpr index_t NPerXdlops = (GemmNPerWave > 64) ? 64 : GemmNPerWave;

    static constexpr auto XdlopsGemm =
        XdlopsGemm_t<Float, MPerXdlops, NPerXdlops, GemmDataPerReadA, GemmDataPerReadB>{};
#else

#if CK_USE_AMD_XDLOPS_INLINE_ASM
    /// \to-do add inline support for vector type c
    static_assert(false, "Does not support inline asm for vector type c")
#else
    static constexpr auto XdlopsGemm =
        XdlopsGemm_t<Float, GemmMPerWave, GemmNPerWave, GemmDataPerReadA, GemmDataPerReadB>{};
#endif

#endif

    index_t mMyWaveOffsetA;
    index_t mMyWaveOffsetB;

    static constexpr index_t WaveSize = 64;

    __device__ constexpr auto GetOutputLayout() const { return XdlopsGemm.GetOutputLayout(); }

#if CK_WORKAROUND_SWDEV_241664
    template <index_t MRepeats_ = MRepeats, index_t NRepeats_ = NRepeats>
    __device__ constexpr auto CreateOutputVecZero() const;

    template <>
    __device__ constexpr auto CreateOutputVecZero<2, 1>() const
    {
        return c_vec32_2_2_t::CreateVecZero();
    }

    template <>
    __device__ constexpr auto CreateOutputVecZero<1, 2>() const
    {
        return c_vec32_2_2_t::CreateVecZero();
    }

    template <>
    __device__ constexpr auto CreateOutputVecZero<1, 1>() const
    {
        return XdlopsGemm.GetOutputLayout().CreateOutputVecZero();
    }
#else
    __device__ constexpr auto CreateOutputVecZero() const
    {
        return XdlopsGemm.GetOutputLayout().CreateOutputVecZero();
    }
#endif

    __device__ constexpr auto GetNumBlks() const
    {
#if CK_WORKAROUND_SWDEV_241664
        return XdlopsGemm.GetOutputLayout().GetNumBlks() * MRepeats * NRepeats;
#else
        return XdlopsGemm.GetOutputLayout().GetNumBlks();
#endif
    }

    __device__ constexpr auto GetBlkSize() const
    {
        return XdlopsGemm.GetOutputLayout().GetBlkSize();
    }

    __device__ BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_xdlops()
    {
        static_assert(BlockMatrixA::NRow() == BlockMatrixB::NRow(),
                      "wrong! K dimension not consistent\n");

        constexpr index_t M = BlockMatrixA::NCol(); // A is transposed
        constexpr index_t N = BlockMatrixB::NCol();

        static_assert(GemmMPerWave * GemmMWaves == M, "GemmMWaves * GemmMPerWave != M");
        static_assert(GemmNPerWave * GemmNWaves == N, "GemmNWaves * GemmNPerWave != N");

        static_assert(BlockSize == GemmMWaves * GemmNWaves * WaveSize,
                      "BlockSize != GemmMWaves * GemmNWaves * WaveSize\n");

        const index_t waveId   = get_thread_local_1d_id() / WaveSize;
        const index_t waveId_m = waveId / GemmNWaves;
        const index_t waveId_n = waveId % GemmNWaves;

        mMyWaveOffsetA = waveId_m * GemmMPerWave;
        mMyWaveOffsetB = waveId_n * GemmNPerWave;
    }

#if CK_WORKAROUND_SWDEV_241664
    template <index_t MRepeats_, index_t NRepeats_>
    struct WithMNRepeats;

    template <>
    struct WithMNRepeats<2, 1>
    {
        template <index_t M, index_t N, index_t K, class FloatA, class FloatB, class FloatC>
        __device__ static FloatC Run(const FloatA* __restrict__ p_a_block,
                                     const FloatB* __restrict__ p_b_block,
                                     FloatC p_c_thread)
        {
            p_c_thread.s.x.l =
                XdlopsGemm.template Run<M, N, K>(p_a_block, p_b_block, p_c_thread.s.x.l);
            p_c_thread.s.y.l = XdlopsGemm.template Run<M, N, K>(
                p_a_block + MPerXdlops, p_b_block, p_c_thread.s.y.l);

            return p_c_thread;
        }
    };

    template <>
    struct WithMNRepeats<1, 2>
    {
        template <index_t M, index_t N, index_t K, class FloatA, class FloatB, class FloatC>
        __device__ static FloatC Run(const FloatA* __restrict__ p_a_block,
                                     const FloatB* __restrict__ p_b_block,
                                     FloatC p_c_thread)
        {
            p_c_thread.s.x.l =
                XdlopsGemm.template Run<M, N, K>(p_a_block, p_b_block, p_c_thread.s.x.l);
            p_c_thread.s.y.l = XdlopsGemm.template Run<M, N, K>(
                p_a_block, p_b_block + NPerXdlops, p_c_thread.s.y.l);

            return p_c_thread;
        }
    };

    template <>
    struct WithMNRepeats<1, 1>
    {
        template <index_t M, index_t N, index_t K, class FloatA, class FloatB, class FloatC>
        __device__ static FloatC Run(const FloatA* __restrict__ p_a_block,
                                     const FloatB* __restrict__ p_b_block,
                                     FloatC p_c_thread)
        {
            return XdlopsGemm.template Run<M, N, K>(p_a_block, p_b_block, p_c_thread);
        }
    };
#endif

    template <class FloatA, class FloatB, class FloatC>
    __device__ FloatC Run(const FloatA* __restrict__ p_a_block,
                          const FloatB* __restrict__ p_b_block,
                          FloatC p_c_thread) const

    {
        constexpr index_t M = BlockMatrixA::NCol(); // A is transposed
        constexpr index_t N = BlockMatrixB::NCol();
        constexpr index_t K = BlockMatrixA::NRow();

#if CK_WORKAROUND_SWDEV_241664
        return WithMNRepeats<MRepeats, NRepeats>::template Run<M, N, K>(
            &p_a_block[mMyWaveOffsetA], &p_b_block[mMyWaveOffsetB], p_c_thread);
#else
        return XdlopsGemm.template Run<M, N, K>(
            &p_a_block[mMyWaveOffsetA], &p_b_block[mMyWaveOffsetB], p_c_thread);
#endif
    }

    template <index_t AStride = GemmMPerWave, index_t BStride = GemmNPerWave>
    __device__ static MatrixIndex GetBeginOfThreadMatrixC(index_t i)
    {

        const index_t waveId = get_thread_local_1d_id() / WaveSize;

#if CK_WORKAROUND_SWDEV_241664
        const index_t xdlops_i = i / XdlopsGemm.GetOutputLayout().GetNumBlks();
        const index_t j        = i % XdlopsGemm.GetOutputLayout().GetNumBlks();

        const index_t m = xdlops_i / NRepeats;
        const index_t n = xdlops_i % NRepeats;

        const auto thread_mtx_on_blk = XdlopsGemm.GetBeginOfThreadBlk(j);

        const index_t col =
            (waveId % GemmNWaves) * BStride + n * NPerXdlops + thread_mtx_on_blk.col;
        const index_t row =
            (waveId / GemmNWaves) * AStride + m * MPerXdlops + thread_mtx_on_blk.row;
#else
        const auto thread_mtx_on_blk = XdlopsGemm.GetBeginOfThreadBlk(i);

        const index_t col = (waveId % GemmNWaves) * BStride + thread_mtx_on_blk.col;
        const index_t row = (waveId / GemmNWaves) * AStride + thread_mtx_on_blk.row;
#endif

        return MatrixIndex{row, col};
    }

    __device__ constexpr auto GetThreadMatrixCDescriptor() const
    {
        const index_t total_reg_size = GemmMPerWave * GemmNPerWave / WaveSize;
        return make_ConstantMatrixDescriptor_packed(Number<total_reg_size>{}, Number<1>{});
    }

    __device__ void XdlopsMatrixCSetZero() const { XdlopsGemm.SetZeroXdlopsRegs(); }

    template <class FloatC>
    __device__ void XdlopsMatrixCRead(FloatC* __restrict__ p_c_thread) const
    {
        XdlopsGemm.ReadXdlopsRegs(p_c_thread);
    }
};

} // namespace ck
#endif
