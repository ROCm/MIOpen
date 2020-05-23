#ifndef CK_BLOCKWISE_GEMM_XDLOPS_HPP
#define CK_BLOCKWISE_GEMM_XDLOPS_HPP

#include "common_header.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "xdlops_gemm.hpp"
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
          index_t GemmDataPerReadA,
          index_t GemmDataPerReadB>
struct BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_xdlops
{
    struct MatrixIndex
    {
        index_t row;
        index_t col;
    };

    static constexpr index_t MRepeats = (GemmMPerWave > 64) ? (GemmMPerWave / 64) : 1;
    static constexpr index_t NRepeats = (GemmNPerWave > 64) ? (GemmNPerWave / 64) : 1;

    static constexpr index_t MPerXdlops = (GemmMPerWave > 64) ? 64 : GemmMPerWave;
    static constexpr index_t NPerXdlops = (GemmNPerWave > 64) ? 64 : GemmNPerWave;

    static constexpr auto XdlopsGemm =
        XdlopsGemm_t<Float, MPerXdlops, NPerXdlops, GemmDataPerReadA, GemmDataPerReadB>{};

    index_t mMyWaveOffsetA;
    index_t mMyWaveOffsetB;

    static constexpr index_t WaveSize = 64;

    __device__ constexpr auto GetOutputLayout() const { return XdlopsGemm.GetOutputLayout(); }

    __device__ constexpr auto GetMRepeats() const { return MRepeats; }

    __device__ constexpr auto GetNRepeats() const { return NRepeats; }

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

        static_assert((GemmMPerWave >= 64 && GemmNPerWave >= 64) ||
                          (GemmMPerWave <= 64 && GemmNPerWave <= 64),
                      "invalid GemmMPerWave/GemmNPerWave");

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

        constexpr auto thread_mtx_size = MPerXdlops * NPerXdlops / WaveSize;

        for(index_t m = 0; m < MRepeats; m++)
        {
            for(index_t n = 0; n < NRepeats; n++)
            {
                XdlopsGemm.template Run<M, N, K>(&p_a_block[mMyWaveOffsetA + MPerXdlops * m],
                                                 &p_b_block[mMyWaveOffsetB + NPerXdlops * n],
                                                 p_c_thread + (NRepeats * m + n) * thread_mtx_size);
            }
        }
    }

    __device__ static MatrixIndex GetBeginOfThreadMatrixC(index_t i, index_t m = 0, index_t n = 0)
    {

        const index_t waveId = get_thread_local_1d_id() / WaveSize;

        (void)m;
        (void)n;

        const auto thread_mtx_on_blk = XdlopsGemm.GetBeginOfThreadBlk(i);

        const index_t col =
            (waveId % GemmNWaves) * GemmNPerWave + n * NPerXdlops + thread_mtx_on_blk.col;

        const index_t row =
            (waveId / GemmNWaves) * GemmMPerWave + m * MPerXdlops + thread_mtx_on_blk.row;

        return MatrixIndex{row, col};
    }

    __device__ constexpr auto GetThreadMatrixCDescriptor() const
    {
        const index_t reg_size = GemmMPerWave * GemmNPerWave / WaveSize;
        return make_ConstantMatrixDescriptor_packed(Number<reg_size>{}, Number<1>{});
    }

    __device__ void XdlopsMatrixCSetZero() const
    {
        constexpr auto thread_mtx_size = MPerXdlops * NPerXdlops / WaveSize;
        XdlopsGemm.SetZeroXdlopsRegs(Number<thread_mtx_size>{});
    }

    template <class FloatC>
    __device__ void XdlopsMatrixCRead(FloatC* __restrict__ p_c_thread) const
    {
        constexpr auto thread_mtx_size = MPerXdlops * NPerXdlops / WaveSize;
        XdlopsGemm.ReadXdlopsRegs(Number<thread_mtx_size>{}, p_c_thread);
    }
};

} // namespace ck
#endif
