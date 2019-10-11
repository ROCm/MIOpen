#ifndef CK_BLOCKWISE_GEMM_XDLOPS_HPP
#define CK_BLOCKWISE_GEMM_XDLOPS_HPP

#include "common_header.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "threadwise_gemm.hpp"

namespace ck {

template <class input_type>
struct mfma_info
{
};

template <>
struct mfma_info<float>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 4;
    static constexpr index_t num_blks_wave   = 2;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_blks_wave;
    static constexpr index_t num_threads_blk = 32;
    static constexpr index_t m               = 32;
    static constexpr index_t n               = 32;
    static constexpr index_t k               = 1;
    static constexpr index_t wave_size       = 64;
};

template <>
struct mfma_info<half>
{
    static const index_t group_size      = 4;
    static const index_t num_groups_blk  = 4;
    static const index_t num_blks_wave   = 2;
    static const index_t num_regs_blk    = group_size * num_groups_blk;
    static const index_t num_regs_xdlops = num_regs_blk * num_blks_wave;
    static const index_t num_threads_blk = 32;
    static const index_t m               = 32;
    static const index_t n               = 32;
    static const index_t k               = 4;
    static const index_t wave_size       = 64;
};

template <>
struct mfma_info<ushort>
{
    static const index_t group_size      = 4;
    static const index_t num_groups_blk  = 4;
    static const index_t num_blks_wave   = 2;
    static const index_t num_regs_blk    = group_size * num_groups_blk;
    static const index_t num_regs_xdlops = num_regs_blk * num_blks_wave;
    static const index_t num_threads_blk = 32;
    static const index_t m               = 32;
    static const index_t n               = 32;
    static const index_t k               = 2;
    static const index_t wave_size       = 64;
};

// emulate xdlops
template <index_t M,
          index_t N,
          index_t K,
          index_t MPerWave,
          index_t GemmDataPerReadA,
          index_t GemmDataPerReadB,
          class mfma_info,
          class FloatA,
          class FloatB,
          class FloatC>
__device__ void WaveWiseGemmMx64(const FloatA* const __restrict__ p_a_wave,
                                 const FloatB* const __restrict__ p_b_wave,
                                 FloatC* const __restrict__ p_c_thread)
{
    static_assert(GemmDataPerReadA == 1 && GemmDataPerReadB == 1, "GemmDataPerReadA/B != 1");

    const index_t laneId = get_thread_local_1d_id() % mfma_info::wave_size;
    const index_t blk_id = laneId / mfma_info::num_threads_blk;
    const index_t lane_b = laneId % mfma_info::num_threads_blk;

    for(index_t k = 0; k < K; ++k)
    {
        for(index_t b = 0; b < MPerWave / mfma_info::m; ++b)
        {
            index_t a_off = k * M + b * mfma_info::m;
            index_t b_off = k * N;
            // pseudo mfma
            for(index_t n = 0; n < mfma_info::num_blks_wave; ++n)
            {
                index_t output_m = mfma_info::num_regs_blk;
                for(index_t m = 0; m < output_m; ++m)
                {
                    index_t aindex = m % mfma_info::group_size + blk_id * mfma_info::group_size +
                                     m / mfma_info::group_size *
                                         (mfma_info::group_size * mfma_info::num_blks_wave) +
                                     a_off; // A is transposed
                    index_t bindex = b_off + lane_b + n * mfma_info::num_threads_blk;
                    p_c_thread[m + n * output_m + b * output_m * mfma_info::num_blks_wave] +=
                        inner_product_with_conversion<FloatC>{}(p_a_wave[aindex], p_b_wave[bindex]);
                }
            }
        }
    }
}

template <index_t M,
          index_t N,
          index_t K,
          index_t MPerWave,
          index_t GemmDataPerReadA,
          index_t GemmDataPerReadB,
          class mfma_info>
__device__ void WaveWiseGemmMx64_xdlops(const float* const __restrict__ p_a_wave,
                                        const float* const __restrict__ p_b_wave,
                                        float* const __restrict__ p_c_thread)
{
    static_assert(MPerWave == 32 || MPerWave == 64, "only support MPerWave = 32/64");

    static_assert(GemmDataPerReadA == 1 && GemmDataPerReadB == 1, "GemmDataPerReadA/B != 1");

    const index_t laneId = get_thread_local_1d_id() % mfma_info::wave_size;

    for(index_t k = 0; k < K; k += mfma_info::k)
    {
        float reg_a      = p_a_wave[k * M + laneId];
        float reg_b      = p_b_wave[k * N + laneId];
        float32_t* reg_c = reinterpret_cast<float32_t*>(p_c_thread);
        gcnasm_mfma_f32_32x32x1f32<MPerWave>(reg_a, reg_b, reg_c);
    }
}

template <index_t M,
          index_t N,
          index_t K,
          index_t MPerWave,
          index_t GemmDataPerReadA,
          index_t GemmDataPerReadB,
          class mfma_info>
__device__ void WaveWiseGemmMx64_xdlops(
    const typename vector_type<half, 4>::MemoryType* const __restrict__ p_a_wave,
    const typename vector_type<half, 4>::MemoryType* const __restrict__ p_b_wave,
    float* const __restrict__ p_c_thread)
{
    static_assert(MPerWave == 32 || MPerWave == 64, "only support MPerWave = 32/64");

    const index_t laneId = threadIdx.x % mfma_info::wave_size;

    for(index_t k = 0; k < K; k += mfma_info::k / 4)
    {
        typename vector_type<half, 4>::MemoryType reg_a = p_a_wave[k * M + laneId];
        typename vector_type<half, 4>::MemoryType reg_b = p_b_wave[k * N + laneId];
        float32_t* reg_c = reinterpret_cast<float32_t*>(p_c_thread);
        gcnasm_mfma_f32_32x32x4f16<MPerWave>(reg_a, reg_b, reg_c);
    }
}

template <index_t M,
          index_t N,
          index_t K,
          index_t MPerWave,
          index_t GemmDataPerReadA,
          index_t GemmDataPerReadB,
          class mfma_info>
__device__ void WaveWiseGemmMx64_xdlops(
    const typename vector_type<ushort, 2>::MemoryType* const __restrict__ p_a_wave,
    const typename vector_type<ushort, 2>::MemoryType* const __restrict__ p_b_wave,
    float* const __restrict__ p_c_thread)
{
    static_assert(MPerWave == 32 || MPerWave == 64, "only support MPerWave = 32/64");

    const index_t laneId = threadIdx.x % mfma_info::wave_size;

    for(index_t k = 0; k < K; k += mfma_info::k / 2)
    {
        typename vector_type<ushort, 2>::MemoryType reg_a = p_a_wave[k * M + laneId];
        typename vector_type<ushort, 2>::MemoryType reg_b = p_b_wave[k * N + laneId];
        float32_t* reg_c = reinterpret_cast<float32_t*>(p_c_thread);
        gcnasm_mfma_f32_32x32x2bf16<MPerWave>(reg_a, reg_b, reg_c);
    }
}

template <index_t BlockSize,
          class BlockMatrixA,
          class BlockMatrixB,
          class mfma_info,
          bool EnableXdlops,
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

    struct OutputLayout_t
    {
        static constexpr index_t M3 = GemmMPerWave / mfma_info::m;
        static constexpr index_t M2 = mfma_info::num_groups_blk;
        static constexpr index_t M1 = mfma_info::num_blks_wave;
        static constexpr index_t M0 = mfma_info::group_size;
    };

    index_t mMyWaveOffsetA;
    index_t mMyWaveOffsetB;

    OutputLayout_t OutputLayout;

    __device__ BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_xdlops()
    {
        static_assert(BlockMatrixA::NRow() == BlockMatrixB::NRow(),
                      "wrong! K dimension not consistent\n");

        constexpr index_t M = BlockMatrixA::NCol(); // A is transposed
        constexpr index_t N = BlockMatrixB::NCol();

        static_assert(GemmNPerWave == 64, "Only support GemmNPerWave == 64 for xdlops");

        static_assert(GemmMPerWave == 32 || GemmMPerWave == 64,
                      "Only support GemmMPerWave == 32 or 64 for xdlops");

        static_assert(GemmMPerWave * GemmMWaves == M, "GemmMWaves * GemmMPerWave != M");
        static_assert(GemmNPerWave * GemmNWaves == N, "GemmNWaves * GemmNPerWave != N");

        static_assert(BlockSize == GemmMWaves * GemmNWaves * 64,
                      "BlockSize != GemmMWaves * GemmNWaves * 64\n");

        const index_t waveId   = get_thread_local_1d_id() / mfma_info::wave_size;
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

        static_if<EnableXdlops>{}([&](auto) {
            WaveWiseGemmMx64_xdlops<M,
                                    N,
                                    K,
                                    GemmMPerWave,
                                    GemmDataPerReadA,
                                    GemmDataPerReadB,
                                    mfma_info>(
                &p_a_block[mMyWaveOffsetA], &p_b_block[mMyWaveOffsetB], p_c_thread);
        }).Else([&](auto) {
            WaveWiseGemmMx64<M, N, K, GemmMPerWave, GemmDataPerReadA, GemmDataPerReadB, mfma_info>(
                &p_a_block[mMyWaveOffsetA], &p_b_block[mMyWaveOffsetB], p_c_thread);
        });
    }

    __device__ static MatrixIndex GetBeginOfThreadMatrixC(index_t i)
    {

        const index_t laneId = get_thread_local_1d_id() % mfma_info::wave_size;
        const index_t waveId = get_thread_local_1d_id() / mfma_info::wave_size;

        const index_t col_i = i % mfma_info::num_blks_wave;
        const index_t col   = waveId % GemmNWaves * mfma_info::wave_size +
                            laneId % mfma_info::num_threads_blk +
                            col_i * mfma_info::num_threads_blk;

        const index_t row_i = i / mfma_info::num_blks_wave;
        const index_t row   = waveId / GemmNWaves * GemmMPerWave +
                            laneId / mfma_info::num_threads_blk * mfma_info::group_size +
                            row_i * mfma_info::num_threads_blk;

        return MatrixIndex{row, col};
    }

    __device__ constexpr auto GetThreadMatrixCDescriptor() const
    {
        constexpr index_t num_xdlops = GemmMPerWave / mfma_info::m;
        return make_ConstantMatrixDescriptor_packed(
            Number<mfma_info::num_regs_xdlops * num_xdlops>{}, Number<1>{});
    }
};

} // namespace ck
#endif
