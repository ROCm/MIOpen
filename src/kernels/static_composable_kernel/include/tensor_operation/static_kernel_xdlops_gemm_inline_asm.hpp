#ifndef CK_XDLOPS_GEMM_INLINE_ASM_HPP
#define CK_XDLOPS_GEMM_INLINE_ASM_HPP

#include "static_kernel_common_header.hpp"
#include "static_kernel_ConstantMatrixDescriptor.hpp"
#include "static_kernel_math.hpp"

namespace ck {

template <mfma_instr instr>
struct mfma_info_asm;

template <>
struct mfma_info_asm<mfma_instr::mfma_f32_32x32x1xf32>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 4;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 32;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = wave_size / num_threads_blk;
    static constexpr index_t num_output_blks = 2;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_output_blks;
    static constexpr index_t m               = 32;
    static constexpr index_t n               = 32;
    static constexpr index_t k               = 1;
    static constexpr index_t cycles          = 64;
    static constexpr index_t k_base          = 1;

    template <index_t MPerXdlops, index_t NPerXdlops, index_t AStride = 1, index_t BStride = 1>
    __device__ void run(const float* a, const float* b) const
    {
        const auto p_a = a;
        const auto p_b = b;

        gcnasm_mfma_f32_32x32x1f32<MPerXdlops, NPerXdlops, AStride, BStride>{}.run(p_a, p_b);
    }
};

template <>
struct mfma_info_asm<mfma_instr::mfma_f32_32x32x2xf32>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 4;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 32;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = wave_size / num_threads_blk;
    static constexpr index_t num_output_blks = 1;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_output_blks;
    static constexpr index_t m               = 32;
    static constexpr index_t n               = 32;
    static constexpr index_t k               = 2;
    static constexpr index_t cycles          = 64;
    static constexpr index_t k_base          = 1;

    template <index_t MPerXdlops, index_t NPerXdlops, index_t AStride = 1, index_t BStride = 1>
    __device__ void run(const float* a, const float* b) const
    {
        const auto p_a = a;
        const auto p_b = b;

        gcnasm_mfma_f32_32x32x2f32(p_a, p_b);
    }
};

template <>
struct mfma_info_asm<mfma_instr::mfma_f32_16x16x4xf32>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 1;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 16;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = wave_size / num_threads_blk;
    static constexpr index_t num_output_blks = 1;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_output_blks;
    static constexpr index_t m               = 16;
    static constexpr index_t n               = 16;
    static constexpr index_t k               = 4;
    static constexpr index_t cycles          = 32;
    static constexpr index_t k_base          = 1;

    template <index_t MPerXdlops, index_t NPerXdlops, index_t AStride = 1, index_t BStride = 1>
    __device__ void run(const float* a, const float* b) const
    {
        const auto p_a = a;
        const auto p_b = b;

        gcnasm_mfma_f32_16x16x4f32(p_a, p_b);
    }
};

template <>
struct mfma_info_asm<mfma_instr::mfma_f32_16x16x1xf32>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 1;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 16;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = wave_size / num_threads_blk;
    static constexpr index_t num_output_blks = 4;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_output_blks;
    static constexpr index_t m               = 16;
    static constexpr index_t n               = 16;
    static constexpr index_t k               = 1;
    static constexpr index_t cycles          = 32;
    static constexpr index_t k_base          = 1;

    template <index_t MPerXdlops, index_t NPerXdlops, index_t AStride = 1, index_t BStride = 1>
    __device__ void run(const float* a, const float* b) const
    {
        const auto p_a = a;
        const auto p_b = b;

        gcnasm_mfma_f32_16x16x1f32<MPerXdlops, NPerXdlops>(p_a, p_b);
    }
};

// treat 4x4x1 as a single-blk 4x64 mfma
template <>
struct mfma_info_asm<mfma_instr::mfma_f32_4x4x1xf32>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 1;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 64;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = 1;
    static constexpr index_t num_output_blks = 1;
    static constexpr index_t num_regs_xdlops = 4;
    static constexpr index_t m               = 4;
    static constexpr index_t n               = 64;
    static constexpr index_t k               = 1;
    static constexpr index_t cycles          = 8;
    static constexpr index_t k_base          = 1;

    template <index_t MPerXdlops, index_t NPerXdlops, index_t AStride = 1, index_t BStride = 1>
    __device__ void run(const float* a, const float* b) const
    {
        const auto p_a = a;
        const auto p_b = b;

        gcnasm_mfma_f32_4x4x1f32<MPerXdlops, NPerXdlops>(p_a, p_b);
    }
};

template <>
struct mfma_info_asm<mfma_instr::mfma_f32_32x32x4f16>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 4;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 32;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = wave_size / num_threads_blk;
    static constexpr index_t num_output_blks = 2;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_output_blks;
    static constexpr index_t m               = 32;
    static constexpr index_t n               = 32;
    static constexpr index_t k               = 4;
    static constexpr index_t cycles          = 64;
    static constexpr index_t k_base          = 4;

    template <index_t MPerXdlops, index_t NPerXdlops, index_t AStride = 1, index_t BStride = 1>
    __device__ void run(const half_t* a, const half_t* b) const
    {
        const auto p_a = reinterpret_cast<const half4_t*>(a);
        const auto p_b = reinterpret_cast<const half4_t*>(b);

        gcnasm_mfma_f32_32x32x4f16<MPerXdlops, NPerXdlops, AStride, BStride>{}.run(p_a, p_b);
    }
};

template <>
struct mfma_info_asm<mfma_instr::mfma_f32_32x32x8f16>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 4;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 32;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = wave_size / num_threads_blk;
    static constexpr index_t num_output_blks = 1;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_output_blks;
    static constexpr index_t m               = 32;
    static constexpr index_t n               = 32;
    static constexpr index_t k               = 8;
    static constexpr index_t cycles          = 64;
    static constexpr index_t k_base          = 4;

    template <index_t MPerXdlops, index_t NPerXdlops, index_t AStride = 1, index_t BStride = 1>
    __device__ void run(const half_t* a, const half_t* b) const
    {
        const auto p_a = reinterpret_cast<const half4_t*>(a);
        const auto p_b = reinterpret_cast<const half4_t*>(b);

        gcnasm_mfma_f32_32x32x8f16(p_a, p_b);
    }
};

template <>
struct mfma_info_asm<mfma_instr::mfma_f32_16x16x16f16>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 1;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 16;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = wave_size / num_threads_blk;
    static constexpr index_t num_output_blks = 1;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_output_blks;
    static constexpr index_t m               = 16;
    static constexpr index_t n               = 16;
    static constexpr index_t k               = 16;
    static constexpr index_t cycles          = 32;
    static constexpr index_t k_base          = 4;

    template <index_t MPerXdlops, index_t NPerXdlops, index_t AStride = 1, index_t BStride = 1>
    __device__ void run(const half_t* a, const half_t* b) const
    {
        const auto p_a = reinterpret_cast<const half4_t*>(a);
        const auto p_b = reinterpret_cast<const half4_t*>(b);

        gcnasm_mfma_f32_16x16x16f16(p_a, p_b);
    }
};

template <>
struct mfma_info_asm<mfma_instr::mfma_f32_16x16x4f16>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 1;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 16;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = wave_size / num_threads_blk;
    static constexpr index_t num_output_blks = 4;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_output_blks;
    static constexpr index_t m               = 16;
    static constexpr index_t n               = 16;
    static constexpr index_t k               = 4;
    static constexpr index_t cycles          = 32;
    static constexpr index_t k_base          = 4;

    template <index_t MPerXdlops, index_t NPerXdlops, index_t AStride = 1, index_t BStride = 1>
    __device__ void run(const half_t* a, const half_t* b) const
    {
        const auto p_a = reinterpret_cast<const half4_t*>(a);
        const auto p_b = reinterpret_cast<const half4_t*>(b);

        gcnasm_mfma_f32_16x16x4f16<MPerXdlops, NPerXdlops>(p_a, p_b);
    }
};

template <>
struct mfma_info_asm<mfma_instr::mfma_f32_4x4x4f16>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 1;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 64;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = 1;
    static constexpr index_t num_output_blks = 1;
    static constexpr index_t num_regs_xdlops = 4;
    static constexpr index_t m               = 4;
    static constexpr index_t n               = 64;
    static constexpr index_t k               = 4;
    static constexpr index_t cycles          = 8;
    static constexpr index_t k_base          = 4;

    template <index_t MPerXdlops, index_t NPerXdlops, index_t AStride = 1, index_t BStride = 1>
    __device__ void run(const half_t* a, const half_t* b) const
    {
        const auto p_a = reinterpret_cast<const half4_t*>(a);
        const auto p_b = reinterpret_cast<const half4_t*>(b);

        gcnasm_mfma_f32_4x4x4f16<MPerXdlops, NPerXdlops>(p_a, p_b);
    }
};

template <>
struct mfma_info_asm<mfma_instr::mfma_f32_32x32x2bf16>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 4;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 32;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = wave_size / num_threads_blk;
    static constexpr index_t num_output_blks = 2;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_output_blks;
    static constexpr index_t m               = 32;
    static constexpr index_t n               = 32;
    static constexpr index_t k               = 2;
    static constexpr index_t cycles          = 64;
    static constexpr index_t k_base          = 2;

    template <index_t MPerXdlops, index_t NPerXdlops, index_t AStride = 1, index_t BStride = 1>
    __device__ void run(const ushort* a, const ushort* b) const
    {
        const auto p_a = reinterpret_cast<const ushort2_t*>(a);
        const auto p_b = reinterpret_cast<const ushort2_t*>(b);

        gcnasm_mfma_f32_32x32x2bf16<MPerXdlops, NPerXdlops, AStride, BStride>{}.run(p_a, p_b);
    }
};

template <>
struct mfma_info_asm<mfma_instr::mfma_f32_32x32x4bf16>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 4;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 32;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = wave_size / num_threads_blk;
    static constexpr index_t num_output_blks = 1;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_output_blks;
    static constexpr index_t m               = 32;
    static constexpr index_t n               = 32;
    static constexpr index_t k               = 4;
    static constexpr index_t cycles          = 64;
    static constexpr index_t k_base          = 2;

    template <index_t MPerXdlops, index_t NPerXdlops, index_t AStride = 1, index_t BStride = 1>
    __device__ void run(const ushort* a, const ushort* b) const
    {
        const auto p_a = reinterpret_cast<const ushort2_t*>(a);
        const auto p_b = reinterpret_cast<const ushort2_t*>(b);

        gcnasm_mfma_f32_32x32x4bf16(p_a, p_b);
    }
};

template <>
struct mfma_info_asm<mfma_instr::mfma_f32_16x16x8bf16>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 1;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 16;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = wave_size / num_threads_blk;
    static constexpr index_t num_output_blks = 1;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_output_blks;
    static constexpr index_t m               = 16;
    static constexpr index_t n               = 16;
    static constexpr index_t k               = 8;
    static constexpr index_t cycles          = 32;
    static constexpr index_t k_base          = 2;

    template <index_t MPerXdlops, index_t NPerXdlops, index_t AStride = 1, index_t BStride = 1>
    __device__ void run(const ushort* a, const ushort* b) const
    {
        const auto p_a = reinterpret_cast<const ushort2_t*>(a);
        const auto p_b = reinterpret_cast<const ushort2_t*>(b);

        gcnasm_mfma_f32_16x16x8bf16(p_a, p_b);
    }
};

template <>
struct mfma_info_asm<mfma_instr::mfma_f32_16x16x2bf16>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 1;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 16;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = wave_size / num_threads_blk;
    static constexpr index_t num_output_blks = 4;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_output_blks;
    static constexpr index_t m               = 16;
    static constexpr index_t n               = 16;
    static constexpr index_t k               = 2;
    static constexpr index_t cycles          = 32;
    static constexpr index_t k_base          = 2;

    template <index_t MPerXdlops, index_t NPerXdlops, index_t AStride = 1, index_t BStride = 1>
    __device__ void run(const ushort* a, const ushort* b) const
    {
        const auto p_a = reinterpret_cast<const ushort2_t*>(a);
        const auto p_b = reinterpret_cast<const ushort2_t*>(b);

        gcnasm_mfma_f32_16x16x2bf16<MPerXdlops, NPerXdlops>(p_a, p_b);
    }
};

template <>
struct mfma_info_asm<mfma_instr::mfma_f32_4x4x2bf16>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 1;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 64;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = 1;
    static constexpr index_t num_output_blks = 1;
    static constexpr index_t num_regs_xdlops = 4;
    static constexpr index_t m               = 4;
    static constexpr index_t n               = 64;
    static constexpr index_t k               = 2;
    static constexpr index_t cycles          = 8;
    static constexpr index_t k_base          = 2;

    template <index_t MPerXdlops, index_t NPerXdlops, index_t AStride = 1, index_t BStride = 1>
    __device__ void run(const ushort* a, const ushort* b) const
    {
        const auto p_a = reinterpret_cast<const ushort2_t*>(a);
        const auto p_b = reinterpret_cast<const ushort2_t*>(b);

        gcnasm_mfma_f32_4x4x2bf16<MPerXdlops, NPerXdlops>(p_a, p_b);
    }
};

template <class data_type,
          index_t GemmMPerWave,
          index_t GemmNPerWave,
          index_t GemmDataPerReadA,
          index_t GemmDataPerReadB>
struct XdlopsGemmAsm_t
{
    static constexpr index_t WaveSize = 64;

    static constexpr index_t MRepeats = (GemmMPerWave > 64) ? (GemmMPerWave / 64) : 1;
    static constexpr index_t NRepeats = (GemmNPerWave > 64) ? (GemmNPerWave / 64) : 1;

    static constexpr index_t MPerXdlops = (GemmMPerWave > 64) ? 64 : GemmMPerWave;
    static constexpr index_t NPerXdlops = (GemmNPerWave > 64) ? 64 : GemmNPerWave;

    __device__ constexpr auto GetMRepeats() const { return MRepeats; }

    __device__ constexpr auto GetNRepeats() const { return NRepeats; }

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
        __device__ static constexpr index_t GetBlkSize() { return GetMFMAInfo().num_regs_blk; }

        __device__ static constexpr index_t GetNumBlks()
        {
            return GetNumBlksPerXdlops() * MRepeats * NRepeats;
        }
    };

    __device__ static constexpr index_t GetNumBlksPerXdlops()
    {
        constexpr auto mfma_type = GetMFMAInfo();
        return (MPerXdlops * NPerXdlops) / (mfma_type.m * mfma_type.n);
    }

    __device__ constexpr XdlopsGemmAsm_t()
    {
        static_assert(!(GemmMPerWave == 128 && GemmNPerWave == 128),
                      "does not support 128x128 xdlops gemm");

        static_assert(NPerXdlops == 4 || NPerXdlops == 8 || NPerXdlops == 16 || NPerXdlops == 32 ||
                          NPerXdlops == 64,
                      "Only support GemmNPerXdlops == 4, 8, 16, 32 or 64 for xdlops");

        static_assert(MPerXdlops == 4 || MPerXdlops == 8 || MPerXdlops == 16 || MPerXdlops == 32 ||
                          MPerXdlops == 64,
                      "Only support GemmMPerXdlops == 4, 8, 16, 32 or 64 for xdlops");

        static_assert(GemmDataPerReadA == 1 && GemmDataPerReadB == 1, "GemmDataPerReadA/B != 1");

        constexpr auto mfma_type = GetMFMAInfo();

        static_assert(mfma_type.num_threads_blk == mfma_type.n, "n != num_threads_blk");
        static_assert(mfma_type.num_regs_blk * mfma_type.num_input_blks == mfma_type.m,
                      "m != num_input_blks * num_regs_blk");
        static_assert(mfma_type.num_output_blks == mfma_type.num_input_blks ||
                          mfma_type.num_output_blks == 1,
                      "incorrect num_output_blks");
        static_assert(mfma_type.num_regs_blk * mfma_type.wave_size == mfma_type.m * mfma_type.n,
                      "num_regs_blk incorrect");

        static_assert(mfma_type.k % mfma_type.k_base == 0, "k and k_base is inconsistent!");
    }

    __device__ static constexpr index_t GetRegSizePerXdlops()
    {
        return MPerXdlops * NPerXdlops / WaveSize;
    }

    __device__ static constexpr bool IsABroadcast() { return NPerXdlops >= MPerXdlops; }

    __device__ static constexpr bool IsKReduction()
    {
        constexpr auto mfma_type = GetMFMAInfo();
        return (mfma_type.num_output_blks == 1) && (mfma_type.num_input_blks > 1);
    }

    template <class data_type_    = data_type,
              index_t MPerXdlops_ = MPerXdlops,
              index_t NPerXdlops_ = NPerXdlops>
    __device__ static constexpr auto GetMFMAInfo();

    template <>
    __device__ static constexpr auto GetMFMAInfo<float, 32, 64>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_32x32x1xf32>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<float, 64, 64>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_32x32x1xf32>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<float, 64, 32>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_32x32x1xf32>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<float, 32, 32>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_32x32x2xf32>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<float, 16, 16>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_16x16x4xf32>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<float, 16, 64>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_16x16x1xf32>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<float, 64, 16>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_16x16x1xf32>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<float, 8, 64>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_4x4x1xf32>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<float, 4, 64>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_4x4x1xf32>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<half_t, 64, 64>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_32x32x4f16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<half_t, 64, 32>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_32x32x4f16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<half_t, 32, 64>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_32x32x4f16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<half_t, 32, 32>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_32x32x8f16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<half_t, 16, 16>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_16x16x16f16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<half_t, 16, 64>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_16x16x4f16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<half_t, 64, 16>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_16x16x4f16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<half_t, 4, 64>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_4x4x4f16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<half_t, 8, 64>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_4x4x4f16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<ushort, 64, 64>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_32x32x2bf16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<ushort, 64, 32>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_32x32x2bf16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<ushort, 32, 64>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_32x32x2bf16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<ushort, 32, 32>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_32x32x4bf16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<ushort, 16, 16>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_16x16x8bf16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<ushort, 16, 64>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_16x16x2bf16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<ushort, 64, 16>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_16x16x2bf16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<ushort, 4, 64>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_4x4x2bf16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<ushort, 8, 64>()
    {
        return mfma_info_asm<mfma_instr::mfma_f32_4x4x2bf16>{};
    }

    template <index_t M, index_t N, index_t K, class FloatA, class FloatB, class FloatC>
    __device__ void Run(const FloatA* const __restrict__ p_a_wave,
                        const FloatB* const __restrict__ p_b_wave,
                        FloatC* const __restrict__) const
    {

        static_assert(GemmDataPerReadA == 1 && GemmDataPerReadB == 1, "GemmDataPerReadA/B != 1");

        static_assert(is_same<FloatA, FloatB>::value, "FloatA != FloatB");
        static_assert(is_same<FloatC, float>::value, "FloatC != float");

        static_assert(is_same<data_type, float>::value || is_same<data_type, half_t>::value ||
                          is_same<data_type, ushort>::value,
                      "base data_type must be float, half, ushort!");

        constexpr auto mfma_type = GetMFMAInfo();

        const index_t laneId = get_thread_local_1d_id() % mfma_type.wave_size;

        FloatA a[K * MRepeats];
        FloatB b[K * NRepeats];

        static_assert(sizeof(FloatA) % (sizeof(data_type) * mfma_type.k_base) == 0,
                      "wrong! FloatA is consistent with mfma");

        constexpr index_t nxdlops = sizeof(FloatA) / (sizeof(data_type) * mfma_type.k_base);

        static_assert(!IsKReduction() || K % mfma_type.num_input_blks == 0,
                      "K cannot divided by mfma_type.num_input_blks!");

        static_assert(!IsKReduction() || (MRepeats == 1 && NRepeats == 1),
                      "KReduction does not support xdlops repeats");

        static_if<!IsKReduction()>{}([&](auto) {
            constexpr index_t AStride = K * nxdlops;
            constexpr index_t BStride = K * nxdlops;

            for(index_t m_i = 0; m_i < MRepeats; ++m_i)
                for(index_t k_i = 0; k_i < K; ++k_i)
                    a[k_i + m_i * K] = p_a_wave[k_i * M + laneId + MPerXdlops * m_i];

            for(index_t n_i = 0; n_i < NRepeats; ++n_i)
                for(index_t k_i = 0; k_i < K; ++k_i)
                    b[k_i + n_i * K] = p_b_wave[k_i * N + laneId + NPerXdlops * n_i];

            // get pointer of registers
            auto pa = reinterpret_cast<const data_type*>(&a);
            auto pb = reinterpret_cast<const data_type*>(&b);

#if CK_WORKAROUND_SWDEV_229564
#pragma unroll
#endif
            for(index_t k_i = 0; k_i < K; ++k_i)
            {
                for(index_t i = 0; i < nxdlops; ++i)
                    mfma_type.template run<GemmMPerWave, GemmNPerWave, AStride, BStride>(
                        &pa[(k_i * nxdlops + i) * mfma_type.k_base],
                        &pb[(k_i * nxdlops + i) * mfma_type.k_base]);
            }
        }).Else([&](auto) {
            const index_t blk_id = laneId / mfma_type.num_threads_blk;
            const index_t blk_td = laneId % mfma_type.num_threads_blk;

            // load into registers
            for(index_t k_i = 0; k_i < K; k_i += mfma_type.num_input_blks)
            {
                a[k_i] = p_a_wave[(k_i + blk_id) * M + blk_td];
                b[k_i] = p_b_wave[(k_i + blk_id) * N + blk_td];
            }

            // get pointer of registers
            auto pa = reinterpret_cast<const data_type*>(&a);
            auto pb = reinterpret_cast<const data_type*>(&b);

#if CK_WORKAROUND_SWDEV_229564
#pragma unroll
#endif
            for(index_t k_i = 0; k_i < K; k_i += mfma_type.num_input_blks)
            {
                for(index_t i = 0; i < nxdlops; ++i)
                    mfma_type.template run<MPerXdlops, NPerXdlops>(
                        &pa[(k_i * nxdlops + i) * mfma_type.k_base],
                        &pb[(k_i * nxdlops + i) * mfma_type.k_base]);
            }
        });
    }

    __device__ static MatrixIndex GetBeginOfThreadBlk(index_t i)
    {
        const index_t xdlops_i = i / GetNumBlksPerXdlops();
        const index_t j        = i % GetNumBlksPerXdlops();

        const index_t m_i = xdlops_i / NRepeats;
        const index_t n_i = xdlops_i % NRepeats;

        constexpr auto mfma_type = GetMFMAInfo();

        const index_t laneId = get_thread_local_1d_id() % mfma_type.wave_size;
        const index_t blk_id = laneId / mfma_type.num_threads_blk;
        const index_t blk_td = laneId % mfma_type.num_threads_blk;

        index_t col_blk = j % mfma_type.num_output_blks;
        index_t row_blk = j / mfma_type.num_output_blks;
        index_t col     = col_blk * mfma_type.n + blk_td + n_i * NPerXdlops;
        index_t row     = row_blk * mfma_type.m + blk_id * mfma_type.group_size + m_i * MPerXdlops;

        static_if<!IsABroadcast()>{}([&](auto) {
            col_blk = j / mfma_type.num_output_blks;
            row_blk = j % mfma_type.num_output_blks;
            col     = col_blk * mfma_type.n + blk_td + n_i * NPerXdlops;
            row     = row_blk * mfma_type.m + blk_id * mfma_type.group_size + m_i * MPerXdlops;
        });

        return MatrixIndex{row, col};
    }

    __device__ static constexpr auto GetOutputLayout()
    {
        constexpr auto mfma_type = GetMFMAInfo();

        constexpr auto M1 = mfma_type.num_groups_blk;
        constexpr auto M0 = mfma_type.group_size;
        constexpr auto N1 = mfma_type.num_input_blks;
        constexpr auto N0 = mfma_type.num_threads_blk;

        return OutputLayout<M1, M0, N1, N0>{};
    }

    __device__ void SetZeroXdlopsRegs() const
    {
        constexpr auto reg_size = GetRegSizePerXdlops() * MRepeats * NRepeats;
        gcnasm_accvgpr_zero<reg_size>();
    }

    template <class FloatC>
    __device__ void ReadXdlopsRegs(FloatC* const __restrict__ p_c_thread) const
    {
        constexpr auto mfma_type = GetMFMAInfo();
        constexpr auto reg_size  = GetRegSizePerXdlops() * MRepeats * NRepeats;
        gcnasm_nop<mfma_type.cycles>();
        gcnasm_accvgpr_read<reg_size>(p_c_thread);
    }
};

} // namespace ck
#endif
