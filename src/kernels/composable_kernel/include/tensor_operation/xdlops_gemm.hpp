#ifndef CK_XDLOPS_GEMM_HPP
#define CK_XDLOPS_GEMM_HPP

#include "common_header.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "math.hpp"

namespace ck {

enum struct mfma_instr
{
    mfma_f32_32x32x1xf32 = 0,
    mfma_f32_32x32x2xf32,
    mfma_f32_32x32x4f16,
    mfma_f32_32x32x8f16,
    mfma_f32_32x32x2bf16,
    mfma_f32_32x32x4bf16,
};

struct mfma_info_base
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 4;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 32;
    static constexpr index_t wave_size       = 64;
};

template <mfma_instr instr>
struct mfma_info;

template <>
struct mfma_info<mfma_instr::mfma_f32_32x32x1xf32> : public mfma_info_base
{
    static constexpr index_t num_blks_wave   = 2;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_blks_wave;
    static constexpr index_t m               = 32;
    static constexpr index_t n               = 32;
    static constexpr index_t k               = 1;

    template <index_t MPerWave, index_t NPerWave>
    __device__ void
    run(Number<MPerWave>, Number<NPerWave>, float& reg_a, float& reg_b, float* reg_c) const
    {
        float32_t* reg_c_ = reinterpret_cast<float32_t*>(reg_c);
        gcnasm_mfma_f32_32x32x1f32<MPerWave, NPerWave>(reg_a, reg_b, reg_c_);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_32x32x2xf32> : public mfma_info_base
{
    static constexpr index_t num_blks_wave   = 1;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_blks_wave;
    static constexpr index_t m               = 32;
    static constexpr index_t n               = 32;
    static constexpr index_t k               = 2;

    template <index_t MPerWave, index_t NPerWave>
    __device__ void
    run(Number<MPerWave>, Number<NPerWave>, float& reg_a, float& reg_b, float* reg_c) const
    {
        static_assert(MPerWave == 32 && NPerWave == 32, "mfma_f32_32x32x2xf32 only support 32x32");
        float16_t* reg_c_ = reinterpret_cast<float16_t*>(reg_c);
        gcnasm_mfma_f32_32x32x2f32(reg_a, reg_b, reg_c_);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_32x32x4f16> : public mfma_info_base
{
    static constexpr index_t num_blks_wave   = 2;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_blks_wave;
    static constexpr index_t m               = 32;
    static constexpr index_t n               = 32;
    static constexpr index_t k               = 4;

    template <index_t MPerWave, index_t NPerWave>
    __device__ void run(Number<MPerWave>,
                        Number<NPerWave>,
                        typename vector_type<half, 4>::MemoryType& reg_a,
                        typename vector_type<half, 4>::MemoryType& reg_b,
                        float* reg_c) const
    {
        float32_t* reg_c_ = reinterpret_cast<float32_t*>(reg_c);
        gcnasm_mfma_f32_32x32x4f16<MPerWave, NPerWave>(reg_a, reg_b, reg_c_);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_32x32x8f16> : public mfma_info_base
{
    static constexpr index_t num_blks_wave   = 1;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_blks_wave;
    static constexpr index_t m               = 32;
    static constexpr index_t n               = 32;
    static constexpr index_t k               = 8;

    template <index_t MPerWave, index_t NPerWave>
    __device__ void run(Number<MPerWave>,
                        Number<NPerWave>,
                        typename vector_type<half, 4>::MemoryType& reg_a,
                        typename vector_type<half, 4>::MemoryType& reg_b,
                        float* reg_c) const
    {
        float16_t* reg_c_ = reinterpret_cast<float16_t*>(reg_c);
        gcnasm_mfma_f32_32x32x8f16(reg_a, reg_b, reg_c_);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_32x32x2bf16> : public mfma_info_base
{
    static constexpr index_t num_blks_wave   = 2;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_blks_wave;
    static constexpr index_t m               = 32;
    static constexpr index_t n               = 32;
    static constexpr index_t k               = 2;

    template <index_t MPerWave, index_t NPerWave>
    __device__ void run(Number<MPerWave>,
                        Number<NPerWave>,
                        typename vector_type<ushort, 2>::MemoryType& reg_a,
                        typename vector_type<ushort, 2>::MemoryType& reg_b,
                        float* reg_c) const
    {
        float32_t* reg_c_ = reinterpret_cast<float32_t*>(reg_c);
        gcnasm_mfma_f32_32x32x2bf16<MPerWave, NPerWave>(reg_a, reg_b, reg_c_);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_32x32x4bf16> : public mfma_info_base
{
    static constexpr index_t num_blks_wave   = 1;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_blks_wave;
    static constexpr index_t m               = 32;
    static constexpr index_t n               = 32;
    static constexpr index_t k               = 4;

    template <index_t MPerWave, index_t NPerWave>
    __device__ void run(Number<MPerWave>,
                        Number<NPerWave>,
                        typename vector_type<ushort, 2>::MemoryType& reg_a,
                        typename vector_type<ushort, 2>::MemoryType& reg_b,
                        float* reg_c) const
    {
        static_assert(MPerWave == 32 && NPerWave == 32, "mfma_f32_32x32x4xbf16 only support 32x32");
        float16_t* reg_c_ = reinterpret_cast<float16_t*>(reg_c);
        gcnasm_mfma_f32_32x32x4bf16(reg_a, reg_b, reg_c_);
    }
};

template <class data_type,
          index_t MPerWave,
          index_t NPerWave,
          index_t GemmDataPerReadA,
          index_t GemmDataPerReadB>
struct XdlopsGemm_t
{
    struct MatrixIndex
    {
        index_t row;
        index_t col;
    };

    template <index_t M1_, index_t M0_, index_t N1_, index_t N0_>
    struct OutputLayout
    {
        __device__ static constexpr index_t M1() { return M1_; }
        __device__ static constexpr index_t M0() { return M0_; }
        __device__ static constexpr index_t N1() { return N1_; }
        __device__ static constexpr index_t N0() { return N0_; }
        __device__ static constexpr index_t GetSizeM() { return M0_ * M1_; }
        __device__ static constexpr index_t GetSizeN() { return N0_ * N1_; }
    };

    __device__ constexpr XdlopsGemm_t()
    {
        static_assert(NPerWave == 32 || NPerWave == 64,
                      "Only support GemmNPerWave == 32 or 64 for xdlops");

        static_assert(MPerWave == 32 || MPerWave == 64,
                      "Only support GemmMPerWave == 32 or 64 for xdlops");

        static_assert(GemmDataPerReadA == 1 && GemmDataPerReadB == 1, "GemmDataPerReadA/B != 1");
    }

    __device__ static constexpr bool IsABroadcast() { return NPerWave != 32; }

    __device__ static constexpr bool IsOneBlk() { return NPerWave == 32 && MPerWave == 32; }

    template <class data_type_ = data_type>
    __device__ static constexpr auto GetMFMAInfo();

    template <>
    __device__ static constexpr auto GetMFMAInfo<float>()
    {
        return
            typename std::conditional<!IsOneBlk(),
                                      decltype(mfma_info<mfma_instr::mfma_f32_32x32x1xf32>{}),
                                      decltype(
                                          mfma_info<mfma_instr::mfma_f32_32x32x2xf32>{})>::type{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<half>()
    {
        return typename std::conditional<!IsOneBlk(),
                                         decltype(mfma_info<mfma_instr::mfma_f32_32x32x4f16>{}),
                                         decltype(
                                             mfma_info<mfma_instr::mfma_f32_32x32x8f16>{})>::type{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<ushort>()
    {
        return
            typename std::conditional<!IsOneBlk(),
                                      decltype(mfma_info<mfma_instr::mfma_f32_32x32x2bf16>{}),
                                      decltype(
                                          mfma_info<mfma_instr::mfma_f32_32x32x4bf16>{})>::type{};
    }

#if CK_USE_AMD_XDLOPS_EMULATE
    // emulate xdlops
    template <index_t M, index_t N, index_t K, class FloatA, class FloatB, class FloatC>
    __device__ void XdlopsEmulate(const FloatA* const __restrict__ p_a_wave,
                                  const FloatB* const __restrict__ p_b_wave,
                                  FloatC* const __restrict__ p_c_thread) const
    {
        constexpr auto mfma_type = GetMFMAInfo();

        const index_t laneId   = get_thread_local_1d_id() % mfma_type.wave_size;
        const index_t blk_id   = laneId / mfma_type.num_threads_blk;
        const index_t lane_b   = laneId % mfma_type.num_threads_blk;
        const index_t num_blks = mfma_type.wave_size / mfma_type.num_threads_blk;

        // one or two mfma with ABroadcast
        static_if<IsABroadcast()>{}([&](auto) {
            for(index_t k = 0; k < K; ++k)
            {
                for(index_t b = 0; b < MPerWave / mfma_type.m; ++b)
                {
                    for(index_t n = 0; n < num_blks; ++n)
                    {
                        index_t a_off = k * M + b * mfma_type.m;
                        index_t b_off = k * N + n * mfma_type.num_threads_blk;
                        index_t c_off = n * mfma_type.num_regs_blk + b * mfma_type.num_regs_xdlops;
                        for(index_t m = 0; m < mfma_type.num_regs_blk; ++m)
                        {
                            index_t aindex =
                                m % mfma_type.group_size + blk_id * mfma_type.group_size +
                                m / mfma_type.group_size * (mfma_type.group_size * num_blks);
                            index_t bindex = lane_b;
                            p_c_thread[m + c_off] += inner_product_with_conversion<FloatC>{}(
                                p_a_wave[aindex + a_off], p_b_wave[bindex + b_off]);
                        }
                    }
                }
            }
        });

        // a single mfma with BBroadcast
        static_if<!(IsABroadcast() || IsOneBlk())>{}([&](auto) {
            for(index_t k = 0; k < K; ++k)
            {
                for(index_t n = 0; n < num_blks; ++n)
                {
                    index_t a_off = k * M + n * mfma_type.m;
                    index_t b_off = k * N;
                    index_t c_off = n * mfma_type.num_regs_blk;
                    for(index_t m = 0; m < mfma_type.num_regs_blk; ++m)
                    {
                        index_t aindex =
                            m % mfma_type.group_size + blk_id * mfma_type.group_size +
                            m / mfma_type.group_size * (mfma_type.group_size * num_blks);
                        index_t bindex = lane_b;
                        p_c_thread[m + c_off] += inner_product_with_conversion<FloatC>{}(
                            p_a_wave[aindex + a_off], p_b_wave[bindex + b_off]);
                    }
                }
            }
        });

        // a single mfma wth Broadcast on both A and B, but double k
        static_if<IsOneBlk()>{}([&](auto) {
            for(index_t k = 0; k < K; k += 2)
            {
                for(index_t n = 0; n < num_blks; ++n)
                {
                    index_t a_off = (k + n) * M;
                    index_t b_off = (k + n) * N;
                    index_t c_off = 0;
                    for(index_t m = 0; m < mfma_type.num_regs_blk; ++m)
                    {
                        index_t aindex =
                            m % mfma_type.group_size + blk_id * mfma_type.group_size +
                            m / mfma_type.group_size * (mfma_type.group_size * num_blks);
                        index_t bindex = lane_b;
                        p_c_thread[m + c_off] += inner_product_with_conversion<FloatC>{}(
                            p_a_wave[aindex + a_off], p_b_wave[bindex + b_off]);
                    }
                }
            }
        });
    }
#endif

    template <index_t M, index_t N, index_t K, class FloatA, class FloatB, class FloatC>
    __device__ void Run(const FloatA* const __restrict__ p_a_wave,
                        const FloatB* const __restrict__ p_b_wave,
                        FloatC* const __restrict__ p_c_thread) const
    {

        static_assert(GemmDataPerReadA == 1 && GemmDataPerReadB == 1, "GemmDataPerReadA/B != 1");

#if CK_USE_AMD_XDLOPS_EMULATE
        XdlopsEmulate<M, N, K>(p_a_wave, p_b_wave, p_c_thread);
#else
        constexpr auto mfma_type = GetMFMAInfo();

        static_if<!IsOneBlk()>{}([&](auto) {

            for(index_t k = 0; k < K; k += 1)
            {
                const index_t laneId = get_thread_local_1d_id() % mfma_type.wave_size;
                FloatA reg_a         = p_a_wave[k * M + laneId];
                FloatB reg_b         = p_b_wave[k * N + laneId];
                mfma_type.run(Number<MPerWave>{}, Number<NPerWave>{}, reg_a, reg_b, p_c_thread);
            }

        }).Else([&](auto) {

            for(index_t k = 0; k < K; k += 2)
            {
                const index_t laneId = get_thread_local_1d_id() % mfma_type.wave_size;
                const index_t blk_i  = laneId / mfma_type.num_threads_blk;
                const index_t blk_n  = laneId % mfma_type.num_threads_blk;
                FloatA reg_a         = p_a_wave[(k + blk_i) * M + blk_n];
                FloatB reg_b         = p_b_wave[(k + blk_i) * N + blk_n];
                mfma_type.run(Number<MPerWave>{}, Number<NPerWave>{}, reg_a, reg_b, p_c_thread);
            }

        });
#endif
    }

    __device__ static MatrixIndex GetBeginOfThreadBlk(index_t i)
    {
        constexpr auto mfma_type = GetMFMAInfo();
        const index_t laneId     = get_thread_local_1d_id() % mfma_type.wave_size;

        index_t col_blk = i % mfma_type.num_blks_wave;
        index_t row_blk = i / mfma_type.num_blks_wave;

        static_if<!IsABroadcast()>{}([&](auto) {
            col_blk = i / mfma_type.num_blks_wave;
            row_blk = i % mfma_type.num_blks_wave;
        });

        const index_t col = col_blk * mfma_type.num_threads_blk + // blk
                            laneId % mfma_type.num_threads_blk;   // thread

        const index_t row = row_blk * mfma_type.num_threads_blk +                      // blk
                            laneId / mfma_type.num_threads_blk * mfma_type.group_size; // thread

        return MatrixIndex{row, col};
    }

    __device__ static constexpr auto GetOutputLayout()
    {
        constexpr auto mfma_type = GetMFMAInfo();

        constexpr auto M1 = mfma_type.num_groups_blk;
        constexpr auto M0 = mfma_type.group_size;
        constexpr auto N1 = mfma_type.wave_size / mfma_type.num_threads_blk;
        constexpr auto N0 = mfma_type.num_threads_blk;

        return OutputLayout<M1, M0, N1, N0>{};
    }

    template <index_t Size>
    __device__ void SetZeroXdlopsRegs(Number<Size>) const
    {
#if !CK_USE_AMD_XDLOPS_EMULATE
        gcnasm_accvgpr_zero<Size>();
#endif
    }

    template <index_t Size, class FloatC>
    __device__ void ReadXdlopsRegs(Number<Size>, FloatC* const __restrict__ p_c_thread) const
    {
#if !CK_USE_AMD_XDLOPS_EMULATE
        gcnasm_accvgpr_read<Size>(p_c_thread);
#else
        (void)p_c_thread;
#endif
    }
};

} // namespace ck
#endif
