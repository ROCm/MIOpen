#ifndef CK_BLOCKWISE_GEMM_XDLOPS_HPP
#define CK_BLOCKWISE_GEMM_XDLOPS_HPP

#include "common_header.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "xdlops_gemm.hpp"
#include "xdlops_gemm_inline_asm.hpp"
#include "threadwise_gemm.hpp"

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
    static constexpr auto XdlopsGemm =
        XdlopsGemmAsm_t<Float, GemmMPerWave, GemmNPerWave, GemmDataPerReadA, GemmDataPerReadB>{};
#else
    static constexpr auto XdlopsGemm =
        XdlopsGemm_t<Float, GemmMPerWave, GemmNPerWave, GemmDataPerReadA, GemmDataPerReadB>{};
#endif

#endif

    index_t mMyWaveOffsetA;
    index_t mMyWaveOffsetB;

    static constexpr index_t WaveSize = 64;

    __device__ constexpr auto GetOutputLayout() const { return XdlopsGemm.GetOutputLayout(); }

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

#if CK_WORKAROUND_SWDEV_241664
        static_assert((MRepeats == 1 && NRepeats == 1) || CK_USE_AMD_XDLOPS_INLINE_ASM == 0,
                      "do not support xdlops repeat with inline asm");
#endif

        const index_t waveId   = get_thread_local_1d_id() / WaveSize;
        const index_t waveId_m = waveId / GemmNWaves;
        const index_t waveId_n = waveId % GemmNWaves;

        mMyWaveOffsetA = waveId_m * GemmMPerWave;
        mMyWaveOffsetB = waveId_n * GemmNPerWave;
    }

    template <class FloatA, class FloatB, class FloatC>
    __device__ void Run(const FloatA* __restrict__ p_a_block,
                        const FloatB* __restrict__ p_b_block,
                        FloatC* __restrict__ p_c_thread) const

    {
        constexpr index_t M = BlockMatrixA::NCol(); // A is transposed
        constexpr index_t N = BlockMatrixB::NCol();
        constexpr index_t K = BlockMatrixA::NRow();

#if CK_WORKAROUND_SWDEV_241664
        constexpr auto reg_size_xdlops = MPerXdlops * NPerXdlops / WaveSize;

        for(index_t m = 0; m < MRepeats; m++)
        {
            for(index_t n = 0; n < NRepeats; n++)
            {
                XdlopsGemm.template Run<M, N, K>(&p_a_block[mMyWaveOffsetA + MPerXdlops * m],
                                                 &p_b_block[mMyWaveOffsetB + NPerXdlops * n],
                                                 p_c_thread + (NRepeats * m + n) * reg_size_xdlops);
            }
        }
#else
        XdlopsGemm.template Run<M, N, K>(
            &p_a_block[mMyWaveOffsetA], &p_b_block[mMyWaveOffsetB], p_c_thread);
#endif
    }

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
            (waveId % GemmNWaves) * GemmNPerWave + n * NPerXdlops + thread_mtx_on_blk.col;

        const index_t row =
            (waveId / GemmNWaves) * GemmMPerWave + m * MPerXdlops + thread_mtx_on_blk.row;
#else
        const auto thread_mtx_on_blk = XdlopsGemm.GetBeginOfThreadBlk(i);

        const index_t col = (waveId % GemmNWaves) * GemmNPerWave + thread_mtx_on_blk.col;
        const index_t row = (waveId / GemmNWaves) * GemmMPerWave + thread_mtx_on_blk.row;
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
