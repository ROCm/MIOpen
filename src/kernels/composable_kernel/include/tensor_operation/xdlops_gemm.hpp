#ifndef CK_XDLOPS_GEMM_HPP
#define CK_XDLOPS_GEMM_HPP

#include "common_header.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "math.hpp"

namespace ck {

enum struct mfma_instr
{
    // fp32
    mfma_f32_32x32x1xf32 = 0,
    mfma_f32_16x16x1xf32,
    mfma_f32_4x4x1xf32,
    mfma_f32_32x32x2xf32, // k reduction
    mfma_f32_16x16x4xf32, // k reduction
    // fp16
    mfma_f32_32x32x4f16,
    mfma_f32_16x16x4f16,
    mfma_f32_4x4x4f16,
    mfma_f32_32x32x8f16,  // k reduction
    mfma_f32_16x16x16f16, // k reduction
    // bfp16
    mfma_f32_32x32x2bf16,
    mfma_f32_16x16x2bf16,
    mfma_f32_4x4x2bf16,
    mfma_f32_32x32x4bf16, // k reduction
    mfma_f32_16x16x8bf16, // k reduction
};

template <mfma_instr instr>
struct mfma_info;

template <>
struct mfma_info<mfma_instr::mfma_f32_32x32x1xf32>
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

    template <index_t MPerWave, index_t NPerWave>
    __device__ void
    run(Number<MPerWave>, Number<NPerWave>, const float* a, const float* b, float* reg_c) const
    {
        static_assert((MPerWave == 64 && NPerWave == 64) || (MPerWave == 32 && NPerWave == 64) ||
                          (MPerWave == 64 && NPerWave == 32),
                      "unsupported xdlops gemm");

        const auto reg_a = *a;
        const auto reg_b = *b;

        auto reg_c_ = reinterpret_cast<float32_t*>(reg_c);
        gcnasm_mfma_f32_32x32x1f32<MPerWave, NPerWave>(reg_a, reg_b, reg_c_);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_32x32x2xf32>
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

    template <index_t MPerWave, index_t NPerWave>
    __device__ void
    run(Number<MPerWave>, Number<NPerWave>, const float* a, const float* b, float* reg_c) const
    {
        static_assert((MPerWave == 32 && NPerWave == 32), "unsupported xdlops gemm");

        const auto reg_a = *a;
        const auto reg_b = *b;

        auto reg_c_ = reinterpret_cast<float16_t*>(reg_c);
        gcnasm_mfma_f32_32x32x2f32(reg_a, reg_b, reg_c_);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_16x16x4xf32>
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

    template <index_t MPerWave, index_t NPerWave>
    __device__ void
    run(Number<MPerWave>, Number<NPerWave>, const float* a, const float* b, float* reg_c) const
    {
        static_assert((MPerWave == 16 && NPerWave == 16), "unsupported xdlops gemm");

        const auto reg_a = *a;
        const auto reg_b = *b;

        auto reg_c_ = reinterpret_cast<float4_t*>(reg_c);
        gcnasm_mfma_f32_16x16x4f32(reg_a, reg_b, reg_c_);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_16x16x1xf32>
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

    template <index_t MPerWave, index_t NPerWave>
    __device__ void
    run(Number<MPerWave>, Number<NPerWave>, const float* a, const float* b, float* reg_c) const
    {
        static_assert((MPerWave == 16 && NPerWave == 64) || (MPerWave == 64 && NPerWave == 16),
                      "unsupported xdlops gemm");

        const auto reg_a = *a;
        const auto reg_b = *b;
        auto reg_c_      = reinterpret_cast<float16_t*>(reg_c);

        gcnasm_mfma_f32_16x16x1f32<MPerWave, NPerWave>(reg_a, reg_b, reg_c_);
    }
};

// treat 4x4x1 as a single-blk 4x64 mfma
template <>
struct mfma_info<mfma_instr::mfma_f32_4x4x1xf32>
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

    template <index_t MPerWave, index_t NPerWave>
    __device__ void
    run(Number<MPerWave>, Number<NPerWave>, const float* a, const float* b, float* reg_c) const
    {
        static_assert((MPerWave == 4 || MPerWave == 8) && NPerWave == 64,
                      "unsupported xdlops gemm");

        const auto reg_a = *a;
        const auto reg_b = *b;
        auto reg_c_      = reinterpret_cast<float4_t*>(reg_c);

        gcnasm_mfma_f32_4x4x1f32<MPerWave, NPerWave>(reg_a, reg_b, reg_c_);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_32x32x4f16>
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

    template <index_t MPerWave, index_t NPerWave>
    __device__ void
    run(Number<MPerWave>, Number<NPerWave>, const half_t* a, const half_t* b, float* reg_c) const
    {
        static_assert((MPerWave == 64 && NPerWave == 64) || (MPerWave == 32 && NPerWave == 64) ||
                          (MPerWave == 64 && NPerWave == 32),
                      "unsupported xdlops gemm");

        const auto reg_a = *(reinterpret_cast<const half4_t*>(a));
        const auto reg_b = *(reinterpret_cast<const half4_t*>(b));
        auto reg_c_      = reinterpret_cast<float32_t*>(reg_c);

        gcnasm_mfma_f32_32x32x4f16<MPerWave, NPerWave>(reg_a, reg_b, reg_c_);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_32x32x8f16>
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

    template <index_t MPerWave, index_t NPerWave>
    __device__ void
    run(Number<MPerWave>, Number<NPerWave>, const half_t* a, const half_t* b, float* reg_c) const
    {
        static_assert((MPerWave == 32 && NPerWave == 32), "unsupported xdlops gemm");

        const auto reg_a = *(reinterpret_cast<const half4_t*>(a));
        const auto reg_b = *(reinterpret_cast<const half4_t*>(b));
        auto reg_c_      = reinterpret_cast<float16_t*>(reg_c);

        gcnasm_mfma_f32_32x32x8f16(reg_a, reg_b, reg_c_);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_16x16x16f16>
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

    template <index_t MPerWave, index_t NPerWave>
    __device__ void
    run(Number<MPerWave>, Number<NPerWave>, const half_t* a, const half_t* b, float* reg_c) const
    {
        static_assert((MPerWave == 16 && NPerWave == 16), "unsupported xdlops gemm");

        const auto reg_a = *(reinterpret_cast<const half4_t*>(a));
        const auto reg_b = *(reinterpret_cast<const half4_t*>(b));
        auto reg_c_      = reinterpret_cast<float4_t*>(reg_c);

        gcnasm_mfma_f32_16x16x16f16(reg_a, reg_b, reg_c_);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_16x16x4f16>
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

    template <index_t MPerWave, index_t NPerWave>
    __device__ void
    run(Number<MPerWave>, Number<NPerWave>, const half_t* a, const half_t* b, float* reg_c) const
    {
        static_assert((MPerWave == 16 && NPerWave == 64) || (MPerWave == 64 && NPerWave == 16),
                      "unsupported xdlops gemm");

        const auto reg_a = *(reinterpret_cast<const half4_t*>(a));
        const auto reg_b = *(reinterpret_cast<const half4_t*>(b));
        auto reg_c_      = reinterpret_cast<float16_t*>(reg_c);

        gcnasm_mfma_f32_16x16x4f16<MPerWave, NPerWave>(reg_a, reg_b, reg_c_);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_4x4x4f16>
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

    template <index_t MPerWave, index_t NPerWave>
    __device__ void
    run(Number<MPerWave>, Number<NPerWave>, const half_t* a, const half_t* b, float* reg_c) const
    {
        static_assert((MPerWave == 4 || MPerWave == 8) && NPerWave == 64,
                      "unsupported xdlops gemm");

        const auto reg_a = *(reinterpret_cast<const half4_t*>(a));
        const auto reg_b = *(reinterpret_cast<const half4_t*>(b));
        auto reg_c_      = reinterpret_cast<float4_t*>(reg_c);

        gcnasm_mfma_f32_4x4x4f16<MPerWave, NPerWave>(reg_a, reg_b, reg_c_);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_32x32x2bf16>
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

    template <index_t MPerWave, index_t NPerWave>
    __device__ void
    run(Number<MPerWave>, Number<NPerWave>, const ushort* a, const ushort* b, float* reg_c) const
    {
        static_assert((MPerWave == 64 && NPerWave == 64) || (MPerWave == 32 && NPerWave == 64) ||
                          (MPerWave == 64 && NPerWave == 32),
                      "unsupported xdlops gemm");

        const auto reg_a = *(reinterpret_cast<const ushort2_t*>(a));
        const auto reg_b = *(reinterpret_cast<const ushort2_t*>(b));
        auto reg_c_      = reinterpret_cast<float32_t*>(reg_c);

        gcnasm_mfma_f32_32x32x2bf16<MPerWave, NPerWave>(reg_a, reg_b, reg_c_);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_32x32x4bf16>
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

    template <index_t MPerWave, index_t NPerWave>
    __device__ void
    run(Number<MPerWave>, Number<NPerWave>, const ushort* a, const ushort* b, float* reg_c) const
    {
        static_assert((MPerWave == 32 && NPerWave == 32), "unsupported xdlops gemm");

        const auto reg_a = *(reinterpret_cast<const ushort2_t*>(a));
        const auto reg_b = *(reinterpret_cast<const ushort2_t*>(b));
        auto reg_c_      = reinterpret_cast<float16_t*>(reg_c);

        gcnasm_mfma_f32_32x32x4bf16(reg_a, reg_b, reg_c_);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_16x16x8bf16>
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

    template <index_t MPerWave, index_t NPerWave>
    __device__ void
    run(Number<MPerWave>, Number<NPerWave>, const ushort* a, const ushort* b, float* reg_c) const
    {
        static_assert((MPerWave == 16 && NPerWave == 16), "unsupported xdlops gemm");

        const auto reg_a = *(reinterpret_cast<const ushort2_t*>(a));
        const auto reg_b = *(reinterpret_cast<const ushort2_t*>(b));
        auto reg_c_      = reinterpret_cast<float4_t*>(reg_c);

        gcnasm_mfma_f32_16x16x8bf16(reg_a, reg_b, reg_c_);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_16x16x2bf16>
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

    template <index_t MPerWave, index_t NPerWave>
    __device__ void
    run(Number<MPerWave>, Number<NPerWave>, const ushort* a, const ushort* b, float* reg_c) const
    {
        static_assert((MPerWave == 16 && NPerWave == 64) || (MPerWave == 64 && NPerWave == 16),
                      "unsupported xdlops gemm");

        const auto reg_a = *(reinterpret_cast<const ushort2_t*>(a));
        const auto reg_b = *(reinterpret_cast<const ushort2_t*>(b));
        auto reg_c_      = reinterpret_cast<float16_t*>(reg_c);

        gcnasm_mfma_f32_16x16x2bf16<MPerWave, NPerWave>(reg_a, reg_b, reg_c_);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_4x4x2bf16>
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

    template <index_t MPerWave, index_t NPerWave>
    __device__ void
    run(Number<MPerWave>, Number<NPerWave>, const ushort* a, const ushort* b, float* reg_c) const
    {
        static_assert((MPerWave == 4 || MPerWave == 8) && NPerWave == 64,
                      "unsupported xdlops gemm");

        const auto reg_a = *(reinterpret_cast<const ushort2_t*>(a));
        const auto reg_b = *(reinterpret_cast<const ushort2_t*>(b));
        auto reg_c_      = reinterpret_cast<float4_t*>(reg_c);

        gcnasm_mfma_f32_4x4x2bf16<MPerWave, NPerWave>(reg_a, reg_b, reg_c_);
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
        __device__ static constexpr index_t GetBlkSize() { return GetMFMAInfo().num_regs_blk; }

        __device__ static constexpr index_t GetNumBlks()
        {
            constexpr auto mfma_type = GetMFMAInfo();
            return MPerWave * NPerWave / (mfma_type.m * mfma_type.n);
        }
    };

    __device__ constexpr XdlopsGemm_t()
    {
        static_assert(NPerWave == 4 || NPerWave == 8 || NPerWave == 16 || NPerWave == 32 ||
                          NPerWave == 64,
                      "Only support GemmNPerWave == 4, 8, 16, 32 or 64 for xdlops");

        static_assert(MPerWave == 4 || MPerWave == 8 || MPerWave == 16 || MPerWave == 32 ||
                          MPerWave == 64,
                      "Only support GemmMPerWave == 4, 8, 16, 32 or 64 for xdlops");

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
    }

    __device__ static constexpr bool IsABroadcast() { return NPerWave >= MPerWave; }

    __device__ static constexpr bool IsKReduction()
    {
        constexpr auto mfma_type = GetMFMAInfo();
        return mfma_type.num_output_blks == 1 && mfma_type.num_input_blks != 1;
    }

    template <class data_type_  = data_type,
              index_t MPerWave_ = MPerWave,
              index_t NPerWave_ = NPerWave>
    __device__ static constexpr auto GetMFMAInfo();

    template <>
    __device__ static constexpr auto GetMFMAInfo<float, 32, 64>()
    {
        return mfma_info<mfma_instr::mfma_f32_32x32x1xf32>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<float, 64, 64>()
    {
        return mfma_info<mfma_instr::mfma_f32_32x32x1xf32>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<float, 64, 32>()
    {
        return mfma_info<mfma_instr::mfma_f32_32x32x1xf32>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<float, 32, 32>()
    {
        return mfma_info<mfma_instr::mfma_f32_32x32x2xf32>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<float, 16, 16>()
    {
        return mfma_info<mfma_instr::mfma_f32_16x16x4xf32>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<float, 16, 64>()
    {
        return mfma_info<mfma_instr::mfma_f32_16x16x1xf32>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<float, 64, 16>()
    {
        return mfma_info<mfma_instr::mfma_f32_16x16x1xf32>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<float, 8, 64>()
    {
        return mfma_info<mfma_instr::mfma_f32_4x4x1xf32>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<float, 4, 64>()
    {
        return mfma_info<mfma_instr::mfma_f32_4x4x1xf32>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<half_t, 64, 64>()
    {
        return mfma_info<mfma_instr::mfma_f32_32x32x4f16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<half_t, 64, 32>()
    {
        return mfma_info<mfma_instr::mfma_f32_32x32x4f16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<half_t, 32, 64>()
    {
        return mfma_info<mfma_instr::mfma_f32_32x32x4f16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<half_t, 32, 32>()
    {
        return mfma_info<mfma_instr::mfma_f32_32x32x8f16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<half_t, 16, 16>()
    {
        return mfma_info<mfma_instr::mfma_f32_16x16x16f16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<half_t, 16, 64>()
    {
        return mfma_info<mfma_instr::mfma_f32_16x16x4f16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<half_t, 64, 16>()
    {
        return mfma_info<mfma_instr::mfma_f32_16x16x4f16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<half_t, 4, 64>()
    {
        return mfma_info<mfma_instr::mfma_f32_4x4x4f16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<half_t, 8, 64>()
    {
        return mfma_info<mfma_instr::mfma_f32_4x4x4f16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<ushort, 64, 64>()
    {
        return mfma_info<mfma_instr::mfma_f32_32x32x2bf16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<ushort, 64, 32>()
    {
        return mfma_info<mfma_instr::mfma_f32_32x32x2bf16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<ushort, 32, 64>()
    {
        return mfma_info<mfma_instr::mfma_f32_32x32x2bf16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<ushort, 32, 32>()
    {
        return mfma_info<mfma_instr::mfma_f32_32x32x4bf16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<ushort, 16, 16>()
    {
        return mfma_info<mfma_instr::mfma_f32_16x16x8bf16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<ushort, 16, 64>()
    {
        return mfma_info<mfma_instr::mfma_f32_16x16x2bf16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<ushort, 64, 16>()
    {
        return mfma_info<mfma_instr::mfma_f32_16x16x2bf16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<ushort, 4, 64>()
    {
        return mfma_info<mfma_instr::mfma_f32_4x4x2bf16>{};
    }

    template <>
    __device__ static constexpr auto GetMFMAInfo<ushort, 8, 64>()
    {
        return mfma_info<mfma_instr::mfma_f32_4x4x2bf16>{};
    }

#if CK_USE_AMD_XDLOPS_EMULATE
    // emulate xdlops
    template <index_t M, index_t N, index_t K, class FloatA, class FloatB, class FloatC>
    __device__ void XdlopsEmulate(const FloatA* const __restrict__ p_a_wave,
                                  const FloatB* const __restrict__ p_b_wave,
                                  FloatC* const __restrict__ p_c_thread) const
    {
        constexpr auto mfma_type = GetMFMAInfo();

        const index_t laneId = get_thread_local_1d_id() % mfma_type.wave_size;
        const index_t blk_id = laneId / mfma_type.num_threads_blk;
        const index_t blk_td = laneId % mfma_type.num_threads_blk;

        // K reduction
        static_if<IsKReduction()>{}([&](auto) {
            for(index_t k = 0; k < K; k += mfma_type.num_input_blks)
            {
                for(index_t n = 0; n < mfma_type.num_input_blks; ++n)
                {
                    index_t a_off = (k + n) * M;
                    index_t b_off = (k + n) * N;
                    index_t c_off = 0;

                    for(index_t m = 0; m < mfma_type.num_regs_blk; ++m)
                    {
                        index_t aindex = m % mfma_type.group_size + blk_id * mfma_type.group_size +
                                         m / mfma_type.group_size *
                                             (mfma_type.group_size * mfma_type.num_input_blks);
                        index_t bindex = blk_td;
                        p_c_thread[m + c_off] += inner_product_with_conversion<FloatC>{}(
                            p_a_wave[aindex + a_off], p_b_wave[bindex + b_off]);
                    }
                }
            }

        }).Else([&](auto) {
            static_if<IsABroadcast()>{}([&](auto) {
                // ABroadcast
                for(index_t k = 0; k < K; ++k)
                {
                    for(index_t b = 0; b < MPerWave / mfma_type.m; ++b)
                    {
                        for(index_t n = 0; n < mfma_type.num_input_blks; ++n)
                        {
                            index_t a_off = k * M + b * mfma_type.m;
                            index_t b_off = k * N + n * mfma_type.num_threads_blk;
                            index_t c_off =
                                n * mfma_type.num_regs_blk + b * mfma_type.num_regs_xdlops;

                            for(index_t m = 0; m < mfma_type.num_regs_blk; ++m)
                            {
                                index_t aindex =
                                    m % mfma_type.group_size + blk_id * mfma_type.group_size +
                                    m / mfma_type.group_size *
                                        (mfma_type.group_size * mfma_type.num_input_blks);
                                index_t bindex = blk_td;
                                p_c_thread[m + c_off] += inner_product_with_conversion<FloatC>{}(
                                    p_a_wave[aindex + a_off], p_b_wave[bindex + b_off]);
                            }
                        }
                    }
                }

            }).Else([&](auto) {
                // BBroadcast
                for(index_t k = 0; k < K; ++k)
                {
                    for(index_t b = 0; b < NPerWave / mfma_type.n; ++b)
                    {
                        for(index_t n = 0; n < mfma_type.num_input_blks; ++n)
                        {
                            index_t a_off = k * M + n * mfma_type.m;
                            index_t b_off = k * N + b * mfma_type.n;
                            index_t c_off =
                                n * mfma_type.num_regs_blk + b * mfma_type.num_regs_xdlops;

                            for(index_t m = 0; m < mfma_type.num_regs_blk; ++m)
                            {
                                index_t aindex =
                                    m % mfma_type.group_size + blk_id * mfma_type.group_size +
                                    m / mfma_type.group_size *
                                        (mfma_type.group_size * mfma_type.num_input_blks);
                                index_t bindex = blk_td;
                                p_c_thread[m + c_off] += inner_product_with_conversion<FloatC>{}(
                                    p_a_wave[aindex + a_off], p_b_wave[bindex + b_off]);
                            }
                        }
                    }
                }
            });
        });
    }
#endif

    template <index_t M, index_t N, index_t K, class FloatA, class FloatB, class FloatC>
    __device__ void Run(const FloatA* const __restrict__ p_a_wave,
                        const FloatB* const __restrict__ p_b_wave,
                        FloatC* const __restrict__ p_c_thread) const
    {

        static_assert(GemmDataPerReadA == 1 && GemmDataPerReadB == 1, "GemmDataPerReadA/B != 1");

        static_assert(is_same<FloatA, FloatB>::value, "FloatA != FloatB");
        static_assert(is_same<FloatC, float>::value, "FloatC != float");

#if CK_USE_AMD_XDLOPS_EMULATE
        XdlopsEmulate<M, N, K>(p_a_wave, p_b_wave, p_c_thread);
#else
        constexpr auto mfma_type = GetMFMAInfo();

        static_if<!IsKReduction()>{}([&](auto) {

            const index_t laneId = get_thread_local_1d_id() % mfma_type.wave_size;

            FloatA a[K];
            FloatB b[K];

            // load into registers
            for(index_t k = 0; k < K; ++k)
            {
                a[k] = p_a_wave[k * M + laneId];
                b[k] = p_b_wave[k * N + laneId];
            }

            // get pointer of registers
            auto pa = reinterpret_cast<const data_type*>(&a);
            auto pb = reinterpret_cast<const data_type*>(&b);

            for(index_t k = 0; k < K; ++k)
            {
                constexpr index_t nxdlops = sizeof(FloatA) / (mfma_type.k * sizeof(data_type));

                for(index_t i = 0; i < nxdlops; ++i, pa += mfma_type.k, pb += mfma_type.k)
                    mfma_type.run(Number<MPerWave>{}, Number<NPerWave>{}, pa, pb, p_c_thread);
            }

        }).Else([&](auto) {

            const index_t laneId = get_thread_local_1d_id() % mfma_type.wave_size;

            FloatA a[K];
            FloatB b[K];

            const index_t blk_id = laneId / mfma_type.num_threads_blk;
            const index_t blk_td = laneId % mfma_type.num_threads_blk;

            // load into registers
            for(index_t k = 0; k < K; k += mfma_type.num_input_blks)
            {
                a[k] = p_a_wave[(k + blk_id) * M + blk_td];
                b[k] = p_b_wave[(k + blk_id) * N + blk_td];
            }

            // get pointer of registers
            auto pa = reinterpret_cast<const data_type*>(&a);
            auto pb = reinterpret_cast<const data_type*>(&b);

            constexpr index_t nxdlops =
                (sizeof(FloatA) * mfma_type.num_input_blks) / (mfma_type.k * sizeof(data_type));

            for(index_t k = 0; k < K; k += mfma_type.num_input_blks)
            {
                for(index_t i = 0; i < nxdlops; ++i, pa += mfma_type.k, pb += mfma_type.k)
                    mfma_type.run(Number<MPerWave>{}, Number<NPerWave>{}, pa, pb, p_c_thread);
            }

        });
#endif
    }

    __device__ static MatrixIndex GetBeginOfThreadBlk(index_t i)
    {
        constexpr auto mfma_type = GetMFMAInfo();

        const index_t laneId = get_thread_local_1d_id() % mfma_type.wave_size;
        const index_t blk_id = laneId / mfma_type.num_threads_blk;
        const index_t blk_td = laneId % mfma_type.num_threads_blk;

        index_t col_blk = i % mfma_type.num_output_blks;
        index_t row_blk = i / mfma_type.num_output_blks;
        index_t col     = col_blk * mfma_type.n + blk_td;
        index_t row     = row_blk * mfma_type.m + blk_id * mfma_type.group_size;

        static_if<!IsABroadcast()>{}([&](auto) {
            col_blk = i / mfma_type.num_output_blks;
            row_blk = i % mfma_type.num_output_blks;
            col     = col_blk * mfma_type.n + blk_td;
            row     = row_blk * mfma_type.m + blk_id * mfma_type.group_size;
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
        constexpr auto mfma_type = GetMFMAInfo();
        gcnasm_nop<mfma_type.cycles>();
        gcnasm_accvgpr_read<Size>(p_c_thread);
#else
        (void)p_c_thread;
#endif
    }
};

} // namespace ck
#endif
