#ifndef CK_XDLOPS_GEMM_HPP
#define CK_XDLOPS_GEMM_HPP

#include "common_header.hpp"
#include "math.hpp"
#include "amd_xdlops.hpp"

namespace ck {

enum struct MfmaInstr
{
    mfma_f32_32x32x1xf32 = 0,
    mfma_f32_16x16x1xf32,
    mfma_f32_4x4x1xf32,
    mfma_f32_32x32x2xf32, // k reduction
    mfma_f32_16x16x4xf32, // k reduction
    mfma_f32_32x32x4f16,
    mfma_f32_16x16x4f16,
    mfma_f32_4x4x4f16,
    mfma_f32_32x32x8f16,  // k reduction
    mfma_f32_16x16x16f16, // k reduction
    mfma_f32_32x32x2bf16,
    mfma_f32_16x16x2bf16,
    mfma_f32_4x4x2bf16,
    mfma_f32_32x32x4bf16, // k reduction
    mfma_f32_16x16x8bf16, // k reduction
};

template <MfmaInstr instr>
struct mfma_type;

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x1xf32>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 2;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 1;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x1f32<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x2xf32>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 1;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x2f32<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x4xf32>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 1;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x4f32<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x1xf32>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 4;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 1;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x1f32<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

// treat 4x4x1 as a single-blk 4x64 mfma
template <>
struct mfma_type<MfmaInstr::mfma_f32_4x4x1xf32>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 64;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 1;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 4;
    static constexpr index_t n_per_blk           = 64;
    static constexpr index_t k_per_blk           = 1;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_4x4x1f32<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x4f16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 2;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x4f16<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x8f16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x8f16<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x16f16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x16f16<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x4f16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 4;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x4f16<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_4x4x4f16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 64;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 1;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 4;
    static constexpr index_t n_per_blk           = 64;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_4x4x4f16<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

#if 0
template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x2bf16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 2;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 2;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t AStride,
              index_t BStride,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ FloatC run(const FloatA* a, const FloatB* b, FloatC reg_c) const
    {
        const auto p_a = c_style_pointer_cast<const ushort2_t*>(a);
        const auto p_b = c_style_pointer_cast<const ushort2_t*>(b);

        return intrin_mfma_f32_32x32x2bf16<MPerXdlops, NPerXdlops, AStride, BStride>::run(
            p_a, p_b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x4bf16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 2;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t AStride,
              index_t BStride,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ FloatC run(const FloatA* a, const FloatB* b, FloatC reg_c) const
    {
        const auto p_a = c_style_pointer_cast<const ushort2_t*>(a);
        const auto p_b = c_style_pointer_cast<const ushort2_t*>(b);

        return intrin_mfma_f32_32x32x4bf16(p_a, p_b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x8bf16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 2;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t AStride,
              index_t BStride,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ FloatC run(const FloatA* a, const FloatB* b, FloatC reg_c) const
    {
        const auto p_a = c_style_pointer_cast<const ushort2_t*>(a);
        const auto p_b = c_style_pointer_cast<const ushort2_t*>(b);

        return intrin_mfma_f32_16x16x8bf16(p_a, p_b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x2bf16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 4;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 2;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t AStride,
              index_t BStride,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ FloatC run(const FloatA* a, const FloatB* b, FloatC reg_c) const
    {
        const auto p_a = c_style_pointer_cast<const ushort2_t*>(a);
        const auto p_b = c_style_pointer_cast<const ushort2_t*>(b);

        return intrin_mfma_f32_16x16x2bf16<MPerXdlops, NPerXdlops>(p_a, p_b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_4x4x2bf16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 64;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 1;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 4;
    static constexpr index_t n_per_blk           = 64;
    static constexpr index_t k_per_blk           = 2;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t AStride,
              index_t BStride,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ FloatC run(const FloatA* a, const FloatB* b, FloatC reg_c) const
    {
        const auto p_a = c_style_pointer_cast<const ushort2_t*>(a);
        const auto p_b = c_style_pointer_cast<const ushort2_t*>(b);

        return intrin_mfma_f32_4x4x2bf16<MPerXdlops, NPerXdlops>::run(p_a, p_b, reg_c);
    }
};
#endif

template <typename base_type, index_t MPerXdlops, index_t NPerXdlops>
struct MfmaSelector
{
    template <typename base_type_, index_t MPerXdlops_, index_t NPerXdlops_>
    static constexpr auto GetMfma();

    template <>
    static constexpr auto GetMfma<float, 64, 64>()
    {
        return MfmaInstr::mfma_f32_32x32x1xf32;
    }

    template <>
    static constexpr auto GetMfma<float, 32, 64>()
    {
        return MfmaInstr::mfma_f32_32x32x1xf32;
    }

    template <>
    static constexpr auto GetMfma<float, 16, 64>()
    {
        return MfmaInstr::mfma_f32_16x16x1xf32;
    }

    template <>
    static constexpr auto GetMfma<float, 8, 64>()
    {
        return MfmaInstr::mfma_f32_4x4x1xf32;
    }

    template <>
    static constexpr auto GetMfma<float, 4, 64>()
    {
        return MfmaInstr::mfma_f32_4x4x1xf32;
    }

    template <>
    static constexpr auto GetMfma<float, 32, 32>()
    {
        return MfmaInstr::mfma_f32_32x32x2xf32;
    }

    template <>
    static constexpr auto GetMfma<float, 16, 16>()
    {
        return MfmaInstr::mfma_f32_16x16x4xf32;
    }

    template <>
    static constexpr auto GetMfma<half_t, 64, 64>()
    {
        return MfmaInstr::mfma_f32_32x32x4f16;
    }

    template <>
    static constexpr auto GetMfma<half_t, 32, 64>()
    {
        return MfmaInstr::mfma_f32_32x32x4f16;
    }

    template <>
    static constexpr auto GetMfma<half_t, 32, 32>()
    {
        return MfmaInstr::mfma_f32_32x32x8f16;
    }

    template <>
    static constexpr auto GetMfma<half_t, 16, 16>()
    {
        return MfmaInstr::mfma_f32_16x16x16f16;
    }

    template <>
    static constexpr auto GetMfma<half_t, 16, 64>()
    {
        return MfmaInstr::mfma_f32_16x16x4f16;
    }

    template <>
    static constexpr auto GetMfma<half_t, 8, 64>()
    {
        return MfmaInstr::mfma_f32_4x4x4f16;
    }

    template <>
    static constexpr auto GetMfma<half_t, 4, 64>()
    {
        return MfmaInstr::mfma_f32_4x4x4f16;
    }

#if 0
    template <>
    static constexpr auto GetMfma<ushort, 128, 64>()
    {
        return xdlops_info<MfmaInstr::mfma_f32_32x32x2bf16, 64, 64, 2, 1, c_vec32_4_t>{};
    }

    template <>
    static constexpr auto GetMfma<ushort, 64, 128>()
    {
        return xdlops_info<MfmaInstr::mfma_f32_32x32x2bf16, 64, 64, 1, 2, c_vec32_4_t>{};
    }

    template <>
    static constexpr auto GetMfma<ushort, 64, 64>()
    {
        return xdlops_info<MfmaInstr::mfma_f32_32x32x2bf16, 64, 64, 1, 1, c_vec32_2_t>{};
    }

    template <>
    static constexpr auto GetMfma<ushort, 64, 32>()
    {
        return xdlops_info<MfmaInstr::mfma_f32_32x32x2bf16, 64, 32, 1, 1, c_vec32_1_t>{};
    }

    template <>
    static constexpr auto GetMfma<ushort, 32, 64>()
    {
        return xdlops_info<MfmaInstr::mfma_f32_32x32x2bf16, 32, 64, 1, 1, c_vec32_1_t>{};
    }

    template <>
    static constexpr auto GetMfma<ushort, 64, 16>()
    {
        return xdlops_info<MfmaInstr::mfma_f32_16x16x2bf16, 64, 16, 1, 1, c_vec16_1_t>{};
    }

    template <>
    static constexpr auto GetMfma<ushort, 16, 64>()
    {
        return xdlops_info<MfmaInstr::mfma_f32_16x16x2bf16, 16, 64, 1, 1, c_vec16_1_t>{};
    }

    template <>
    static constexpr auto GetMfma<ushort, 8, 64>()
    {
        return xdlops_info<MfmaInstr::mfma_f32_4x4x2bf16, 8, 64, 1, 1, c_vec4_2_t>{};
    }

    template <>
    static constexpr auto GetMfma<ushort, 4, 64>()
    {
        return xdlops_info<MfmaInstr::mfma_f32_4x4x2bf16, 4, 64, 1, 1, c_vec4_1_t>{};
    }

    template <>
    static constexpr auto GetMfma<ushort, 32, 32>()
    {
        return xdlops_info<MfmaInstr::mfma_f32_32x32x4bf16, 32, 32, 1, 1, c_vec16_1_t>{};
    }

    template <>
    static constexpr auto GetMfma<ushort, 16, 16>()
    {
        return xdlops_info<MfmaInstr::mfma_f32_16x16x8bf16, 16, 16, 1, 1, c_vec4_1_t>{};
    }
#endif

    static constexpr auto selected_mfma = mfma_type<GetMfma<base_type, MPerXdlops, NPerXdlops>()>{};

    __host__ __device__ static constexpr void mfma_check()
    {
        static_assert(selected_mfma.group_size * selected_mfma.num_groups_per_blk ==
                          selected_mfma.num_regs_per_blk,
                      "wrong! num_regs_per_blk");

        static_assert(selected_mfma.num_threads_per_blk == selected_mfma.n_per_blk,
                      "n_per_blk != num_threads_per_blk");

        static_assert(selected_mfma.num_regs_per_blk * selected_mfma.num_input_blks ==
                          selected_mfma.m_per_blk,
                      "m_per_blk != num_input_blks * num_regs_per_blk");

        static_assert(selected_mfma.num_output_blks == selected_mfma.num_input_blks ||
                          selected_mfma.num_output_blks == 1,
                      "incorrect num_output_blks");

        static_assert(selected_mfma.num_regs_per_blk * selected_mfma.wave_size ==
                          selected_mfma.m_per_blk * selected_mfma.n_per_blk,
                      "num_regs_per_blk incorrect");

        static_assert(selected_mfma.is_k_reduction ||
                          (selected_mfma.num_input_blks == selected_mfma.num_output_blks),
                      "is_k_reduction wrong!");
    }

    __host__ __device__ constexpr MfmaSelector() { mfma_check(); }

    static constexpr bool IsABroadcast()
    {
        static_assert(NPerXdlops >= MPerXdlops, "only support ABroadcast");
        return true;
    }

    static constexpr index_t GetKPerXdlops()
    {
        return (selected_mfma.is_k_reduction ? selected_mfma.num_input_blks : 1) *
               selected_mfma.k_per_blk;
    }

    static constexpr index_t GetKPerThread() { return selected_mfma.k_per_blk; }
};

template <typename base_type, index_t MPerXdlops, index_t NPerXdlops, index_t KPack>
struct XdlopsGemm
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    using CIndex = MultiIndex<2>;

    __device__ static constexpr index_t GetNumBlks() { return mfma_instr.num_output_blks; }

    __device__ static constexpr index_t GetNumXdlops()
    {
        return MPerXdlops * NPerXdlops /
               (mfma_instr.m_per_blk * mfma_instr.n_per_blk * mfma_instr.num_output_blks);
    }

    __host__ __device__ constexpr XdlopsGemm()
    {
        static_assert(NPerXdlops == 4 || NPerXdlops == 8 || NPerXdlops == 16 || NPerXdlops == 32 ||
                          NPerXdlops == 64,
                      "Only support GemmNPerXdlops == 4, 8, 16, 32 or 64 for xdlops");

        static_assert(MPerXdlops == 4 || MPerXdlops == 8 || MPerXdlops == 16 || MPerXdlops == 32 ||
                          MPerXdlops == 64,
                      "Only support GemmMPerXdlops == 4, 8, 16, 32 or 64 for xdlops");

        static_assert(KPack % mfma_instr.k_per_blk == 0, "KPack cannot be divided by k_per_blk");
    }

    template <typename CM0N0M1N1M2N2Desc>
    __host__ __device__ static constexpr auto
    MakeCM0N0M1N1M2M3M4N2Descriptor(const CM0N0M1N1M2N2Desc& c_m0_n0_m1_n1_m2_n2_desc)
    {
        const auto M0 = c_m0_n0_m1_n1_m2_n2_desc.GetLength(I0);
        const auto N0 = c_m0_n0_m1_n1_m2_n2_desc.GetLength(I1);
        const auto M1 = c_m0_n0_m1_n1_m2_n2_desc.GetLength(I2);
        const auto N1 = c_m0_n0_m1_n1_m2_n2_desc.GetLength(I3);

        return transform_tensor_descriptor(
            c_m0_n0_m1_n1_m2_n2_desc,
            make_tuple(make_pass_through_transform(M0),
                       make_pass_through_transform(N0),
                       make_pass_through_transform(M1),
                       make_pass_through_transform(N1),
                       make_unmerge_transform(make_tuple(mfma_instr.num_groups_per_blk,
                                                         mfma_instr.num_input_blks,
                                                         mfma_instr.group_size)),
                       make_pass_through_transform(mfma_instr.num_threads_per_blk)),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{}),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4, 5, 6>{},
                       Sequence<7>{}));
    }

    __device__ static constexpr index_t GetRegSizePerXdlops()
    {
        return MPerXdlops * NPerXdlops / mfma_instr.wave_size;
    }

    template <class FloatA, class FloatB, class FloatC>
    __device__ void Run(const FloatA& p_a_wave, const FloatB& p_b_wave, FloatC& p_c_thread) const
    {
        static_assert(is_same<base_type, float>::value || is_same<base_type, half_t>::value ||
                          is_same<base_type, ushort>::value,
                      "base base_type must be float, half, ushort!");

        static_for<0, KPack / mfma_instr.k_per_blk, 1>{}([&](auto k) {
            mfma_instr.template run<MPerXdlops, NPerXdlops>(p_a_wave[k], p_b_wave[k], p_c_thread);
        });
    }

    __device__ static auto GetLaneId() { return get_thread_local_1d_id() % mfma_instr.wave_size; }

    __device__ static auto GetBlkIdx()
    {
        const auto laneId = GetLaneId();

        constexpr auto threadidx_to_blk_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(
                make_tuple(1, mfma_instr.num_input_blks, mfma_instr.num_threads_per_blk))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        const auto blk_idx =
            threadidx_to_blk_idx_adaptor.CalculateBottomIndex(make_multi_index(laneId));

        const auto blk_id = blk_idx[I1];
        const auto blk_td = blk_idx[I2];

        return make_tuple(blk_id, blk_td);
    }

    __host__ __device__ static auto CalculateAThreadOriginDataIndex()
    {
        const auto laneId  = GetLaneId();
        const auto blk_idx = GetBlkIdx();

        const auto blk_id = blk_idx[I0];
        const auto blk_td = blk_idx[I1];

        if constexpr(mfma_instr.is_k_reduction)
        {
            return make_tuple(blk_id, blk_td);
        }
        else
        {
            return make_tuple(0, laneId);
        }
    }

    __host__ __device__ static auto CalculateBThreadOriginDataIndex()
    {
        const auto laneId  = GetLaneId();
        const auto blk_idx = GetBlkIdx();

        const auto blk_id = blk_idx[I0];
        const auto blk_td = blk_idx[I1];

        if constexpr(mfma_instr.is_k_reduction)
        {
            return make_tuple(blk_id, blk_td);
        }
        else
        {
            return make_tuple(0, laneId);
        }
    }

    __device__ static CIndex GetBeginOfThreadBlk(index_t xdlops_i, index_t blk_i)
    {
        const auto blk_idx = GetBlkIdx();

        const auto blk_id = blk_idx[I0];
        const auto blk_td = blk_idx[I1];

        index_t n_offset = blk_i * mfma_instr.n_per_blk + blk_td;
        index_t m_offset = xdlops_i * mfma_instr.m_per_blk + blk_id * mfma_instr.group_size;

        return CIndex{m_offset, n_offset};
    }

    static constexpr auto mfma = MfmaSelector<base_type, MPerXdlops, NPerXdlops>{};

    static constexpr auto mfma_instr = mfma.selected_mfma;

    static constexpr auto KPerXdlops  = mfma.GetKPerXdlops();
    static constexpr auto K1PerXdlops = mfma.GetKPerThread();
    static constexpr auto K0PerXdlops = KPerXdlops / K1PerXdlops;

    __host__ __device__ static constexpr auto GetCM0M1M2NThreadBlkLengths()
    {
        return make_tuple(
            Number<mfma_instr.num_groups_per_blk>{}, I1, Number<mfma_instr.group_size>{}, I1);
    }
};

} // namespace ck
#endif
