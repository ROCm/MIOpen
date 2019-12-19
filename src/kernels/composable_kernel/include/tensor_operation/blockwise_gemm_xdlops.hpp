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

    static constexpr auto XdlopsGemm =
        XdlopsGemm_t<Float, GemmMPerWave, GemmNPerWave, GemmDataPerReadA, GemmDataPerReadB>{};

    index_t mMyWaveOffsetA;
    index_t mMyWaveOffsetB;

    static constexpr index_t WaveSize = 64;

    __device__ constexpr auto GetOutputLayout() const { return XdlopsGemm.GetOutputLayout(); }

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

    template <class FloatA, class FloatB, class FloatC>
    __device__ void Run(const FloatA* __restrict__ p_a_block,
                        const FloatB* __restrict__ p_b_block,
                        FloatC* __restrict__ p_c_thread) const

    {
        constexpr index_t M = BlockMatrixA::NCol(); // A is transposed
        constexpr index_t N = BlockMatrixB::NCol();
        constexpr index_t K = BlockMatrixA::NRow();

        XdlopsGemm.template Run<M, N, K>(
            &p_a_block[mMyWaveOffsetA], &p_b_block[mMyWaveOffsetB], p_c_thread);
    }

    __device__ static MatrixIndex GetBeginOfThreadMatrixC(index_t i)
    {

        const index_t waveId = get_thread_local_1d_id() / WaveSize;

        const auto thread_mtx_on_blk = XdlopsGemm.GetBeginOfThreadBlk(i);

        const index_t col = waveId % GemmNWaves * GemmNPerWave + thread_mtx_on_blk.col;

        const index_t row = waveId / GemmNWaves * GemmMPerWave + thread_mtx_on_blk.row;

        return MatrixIndex{row, col};
    }

    __device__ constexpr auto GetThreadMatrixCDescriptor() const
    {
        const index_t reg_size = GemmMPerWave * GemmNPerWave / WaveSize;
        return make_ConstantMatrixDescriptor_packed(Number<reg_size>{}, Number<1>{});
    }

    __device__ void XdlopsMatrixCSetZero() const
    {
        constexpr auto thread_mtx_size = GemmMPerWave * GemmNPerWave / WaveSize;
        XdlopsGemm.SetZeroXdlopsRegs(Number<thread_mtx_size>{});
    }

    template <class FloatC>
    __device__ void XdlopsMatrixCRead(FloatC* __restrict__ p_c_thread) const
    {
        constexpr auto thread_mtx_size = GemmMPerWave * GemmNPerWave / WaveSize;
        XdlopsGemm.ReadXdlopsRegs(Number<thread_mtx_size>{}, p_c_thread);
    }
};

} // namespace ck
#endif
