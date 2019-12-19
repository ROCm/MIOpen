#ifndef CK_AMD_XDLOPS_HPP
#define CK_AMD_XDLOPS_HPP

#include "float_type.hpp"

namespace ck {

// A, B, C, cbsz, abid, blgp
extern "C" __device__ float32_t llvm_intrin_amdgcn_mfma_f32_32x32x1f32(
    float, float, float32_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x1f32");

extern "C" __device__ float16_t llvm_intrin_amdgcn_mfma_f32_32x32x2f32(
    float, float, float16_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x2f32");

extern "C" __device__ float4_t llvm_intrin_amdgcn_mfma_f32_16x16x4f32(
    float, float, float4_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.16x16x4f32");

extern "C" __device__ float16_t llvm_intrin_amdgcn_mfma_f32_16x16x1f32(
    float, float, float16_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.16x16x1f32");

extern "C" __device__ float4_t llvm_intrin_amdgcn_mfma_f32_4x4x1f32(
    float, float, float4_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.4x4x1f32");

extern "C" __device__ float32_t llvm_intrin_amdgcn_mfma_f32_32x32x4f16(
    half4_t, half4_t, float32_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x4f16");

extern "C" __device__ float16_t llvm_intrin_amdgcn_mfma_f32_32x32x8f16(
    half4_t, half4_t, float16_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x8f16");

extern "C" __device__ float4_t llvm_intrin_amdgcn_mfma_f32_16x16x16f16(
    half4_t, half4_t, float4_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.16x16x16f16");

extern "C" __device__ float16_t llvm_intrin_amdgcn_mfma_f32_16x16x4f16(
    half4_t, half4_t, float16_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.16x16x4f16");

extern "C" __device__ float4_t llvm_intrin_amdgcn_mfma_f32_4x4x4f16(
    half4_t, half4_t, float4_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.4x4x1f16");

extern "C" __device__ float32_t llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(
    ushort2_t, ushort2_t, float32_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x2bf16");

extern "C" __device__ float16_t llvm_intrin_amdgcn_mfma_f32_32x32x4bf16(
    ushort2_t, ushort2_t, float16_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x4bf16");

extern "C" __device__ float4_t llvm_intrin_amdgcn_mfma_f32_16x16x8bf16(
    ushort2_t, ushort2_t, float4_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.16x16x8bf16");

extern "C" __device__ float16_t llvm_intrin_amdgcn_mfma_f32_16x16x2bf16(
    ushort2_t, ushort2_t, float16_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.16x16x2bf16");

extern "C" __device__ float4_t llvm_intrin_amdgcn_mfma_f32_4x4x2bf16(
    ushort2_t, ushort2_t, float4_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.4x4x2bf16");
// clang-format off

#define REPEATx4(f, off) f(off) f(off + 1) f(off + 2) f(off + 3)

#define REPEATx16(f, off) \
    REPEATx4(f, off) REPEATx4(f, off + 4) REPEATx4(f, off + 8) REPEATx4(f, off + 12)

#define REPEATx64(f, off) \
    REPEATx16(f, off) REPEATx16(f, off + 16) REPEATx16(f, off + 32) REPEATx16(f, off + 48)

#define REPEAT_STRIDEx4(f, stride, off) \
    f(off) f(off + 1 * stride) f(off + 2 * stride) f(off + 3 * stride)

#define REPEAT_STRIDEx16(f, stride, off)                                             \
    REPEAT_STRIDEx4(f, stride, off) REPEAT_STRIDEx4(f, stride, off + 1 * stride * 4) \
        REPEAT_STRIDEx4(f, stride, off + 2 * stride * 4)                             \
            REPEAT_STRIDEx4(f, stride, off + 3 * stride * 4)

#define REPEAT_STRIDEx64(f, stride, off)                                                \
    REPEAT_STRIDEx16(f, stride, off) REPEAT_STRIDEx16(f, stride, off + 1 * stride * 16) \
        REPEAT_STRIDEx16(f, stride, off + 2 * stride * 16)                              \
            REPEAT_STRIDEx16(f, stride, off + 3 * stride * 16)

#define NOP(n) asm volatile("\n s_nop " #n " " : :);

#define MFMA_F32_32x32x1F32(acc, reg_a, reg_b, cbsz, abid, blgp)                           \
    asm volatile("v_mfma_f32_32x32x1f32 a[" #acc ":" #acc "+31], %0, %1, a[" #acc ":" #acc \
                 "+31] cbsz: " #cbsz " abid: " #abid " blgp:" #blgp " "                    \
                 :                                                                         \
                 : "v"(reg_a), "v"(reg_b));

#define MFMA_F32_32x32x2F32(acc, reg_a, reg_b, cbsz, abid, blgp)                           \
    asm volatile("v_mfma_f32_32x32x2f32 a[" #acc ":" #acc "+15], %0, %1, a[" #acc ":" #acc \
                 "+15] cbsz: " #cbsz " abid: " #abid " blgp:" #blgp " "                    \
                 :                                                                         \
                 : "v"(reg_a), "v"(reg_b));

#define MFMA_F32_16x16x4F32(acc, reg_a, reg_b, cbsz, abid, blgp)                           \
    asm volatile("v_mfma_f32_16x16x4f32 a[" #acc ":" #acc "+3], %0, %1, a[" #acc ":" #acc \
                 "+3] cbsz: " #cbsz " abid: " #abid " blgp:" #blgp " "                    \
                 :                                                                         \
                 : "v"(reg_a), "v"(reg_b));

#define MFMA_F32_16x16x1F32(acc, reg_a, reg_b, cbsz, abid, blgp)                           \
    asm volatile("v_mfma_f32_16x16x1f32 a[" #acc ":" #acc "+15], %0, %1, a[" #acc ":" #acc \
                 "+15] cbsz: " #cbsz " abid: " #abid " blgp:" #blgp " "                    \
                 :                                                                         \
                 : "v"(reg_a), "v"(reg_b));

#define MFMA_F32_4x4x1F32(acc, reg_a, reg_b, cbsz, abid, blgp)                           \
    asm volatile("v_mfma_f32_4x4x1f32 a[" #acc ":" #acc "+3], %0, %1, a[" #acc ":" #acc \
                 "+3] cbsz: " #cbsz " abid: " #abid " blgp:" #blgp " "                    \
                 :                                                                         \
                 : "v"(reg_a), "v"(reg_b));

#define MFMA_F32_32x32x4F16(acc, reg_a, reg_b, cbsz, abid, blgp)                           \
    asm volatile("v_mfma_f32_32x32x4f16 a[" #acc ":" #acc "+31], %0, %1, a[" #acc ":" #acc \
                 "+31] cbsz: " #cbsz " abid: " #abid " blgp:" #blgp " "                    \
                 :                                                                         \
                 : "v"(reg_a), "v"(reg_b));

#define MFMA_F32_32x32x8F16(acc, reg_a, reg_b, cbsz, abid, blgp)                           \
    asm volatile("v_mfma_f32_32x32x8f16 a[" #acc ":" #acc "+15], %0, %1, a[" #acc ":" #acc \
                 "+15] cbsz: " #cbsz " abid: " #abid " blgp:" #blgp " "                    \
                 :                                                                         \
                 : "v"(reg_a), "v"(reg_b));

#define MFMA_F32_16x16x16F16(acc, reg_a, reg_b, cbsz, abid, blgp)                           \
    asm volatile("v_mfma_f32_16x16x16f16 a[" #acc ":" #acc "+3], %0, %1, a[" #acc ":" #acc \
                 "+3] cbsz: " #cbsz " abid: " #abid " blgp:" #blgp " "                    \
                 :                                                                         \
                 : "v"(reg_a), "v"(reg_b));

#define MFMA_F32_16x16x4F16(acc, reg_a, reg_b, cbsz, abid, blgp)                           \
    asm volatile("v_mfma_f32_16x16x4f16 a[" #acc ":" #acc "+15], %0, %1, a[" #acc ":" #acc \
                 "+15] cbsz: " #cbsz " abid: " #abid " blgp:" #blgp " "                    \
                 :                                                                         \
                 : "v"(reg_a), "v"(reg_b));

#define MFMA_F32_4x4x4F16(acc, reg_a, reg_b, cbsz, abid, blgp)                           \
    asm volatile("v_mfma_f32_4x4x4f16 a[" #acc ":" #acc "+3], %0, %1, a[" #acc ":" #acc \
                 "+3] cbsz: " #cbsz " abid: " #abid " blgp:" #blgp " "                    \
                 :                                                                         \
                 : "v"(reg_a), "v"(reg_b));

#define MFMA_F32_32x32x2BF16(acc, reg_a, reg_b, cbsz, abid, blgp)                           \
    asm volatile("v_mfma_f32_32x32x2bf16 a[" #acc ":" #acc "+31], %0, %1, a[" #acc ":" #acc \
                 "+31] cbsz: " #cbsz " abid: " #abid " blgp:" #blgp " "                     \
                 :                                                                          \
                 : "v"(reg_a), "v"(reg_b));

#define MFMA_F32_32x32x4BF16(acc, reg_a, reg_b, cbsz, abid, blgp)                           \
    asm volatile("v_mfma_f32_32x32x4bf16 a[" #acc ":" #acc "+15], %0, %1, a[" #acc ":" #acc \
                 "+15] cbsz: " #cbsz " abid: " #abid " blgp:" #blgp " "                     \
                 :                                                                          \
                 : "v"(reg_a), "v"(reg_b));

#define MFMA_F32_16x16x8BF16(acc, reg_a, reg_b, cbsz, abid, blgp)                           \
    asm volatile("v_mfma_f32_16x16x8bf16 a[" #acc ":" #acc "+3], %0, %1, a[" #acc ":" #acc \
                 "+3] cbsz: " #cbsz " abid: " #abid " blgp:" #blgp " "                     \
                 :                                                                          \
                 : "v"(reg_a), "v"(reg_b));

#define MFMA_F32_16x16x2BF16(acc, reg_a, reg_b, cbsz, abid, blgp)                           \
    asm volatile("v_mfma_f32_16x16x2bf16 a[" #acc ":" #acc "+15], %0, %1, a[" #acc ":" #acc \
                 "+15] cbsz: " #cbsz " abid: " #abid " blgp:" #blgp " "                     \
                 :                                                                          \
                 : "v"(reg_a), "v"(reg_b));

#define MFMA_F32_4x4x2BF16(acc, reg_a, reg_b, cbsz, abid, blgp)                           \
    asm volatile("v_mfma_f32_4x4x2bf16 a[" #acc ":" #acc "+3], %0, %1, a[" #acc ":" #acc \
                 "+3] cbsz: " #cbsz " abid: " #abid " blgp:" #blgp " "                     \
                 :                                                                          \
                 : "v"(reg_a), "v"(reg_b));

#define ACCVGPR_READ(acc_reg_id) \
    asm volatile("v_accvgpr_read_b32 %0, a[" #acc_reg_id "]" : "=v"(arch_reg[acc_reg_id]) :);

#define ACCVGPR_WRITE(acc_reg_id) \
    asm volatile("v_accvgpr_write_b32 a[" #acc_reg_id "], %0" : : "v"(arch_reg[acc_reg_id]));

#define ACCVGPR_ZERO(acc_reg_id) \
    asm volatile("v_accvgpr_write_b32 a[" #acc_reg_id "], 0" : :);

template <index_t Size>
__device__ void gcnasm_accvgpr_read(float*);

template <>
__device__ void gcnasm_accvgpr_read<4>(float* arch_reg)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    REPEATx4(ACCVGPR_READ, 0)
#else
    (void)arch_reg;
#endif
}

template <>
__device__ void gcnasm_accvgpr_read<8>(float* arch_reg)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    REPEATx4(ACCVGPR_READ, 0)
    REPEATx4(ACCVGPR_READ, 4)
#else
    (void)arch_reg;
#endif
}

template <>
__device__ void gcnasm_accvgpr_read<16>(float* arch_reg)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    REPEATx16(ACCVGPR_READ, 0)
#else
    (void)arch_reg;
#endif
}

template <>
__device__ void gcnasm_accvgpr_read<32>(float* arch_reg)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    REPEATx16(ACCVGPR_READ, 0)
    REPEATx16(ACCVGPR_READ, 16)
#else
    (void)arch_reg;
#endif
}

template <>
__device__ void gcnasm_accvgpr_read<64>(float* arch_reg)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    REPEATx64(ACCVGPR_READ, 0)
#else
    (void)arch_reg;
#endif
}

template <index_t MPerWave>
__device__ void gcnasm_accvgpr_zero();

template <>
__device__ void gcnasm_accvgpr_zero<4>()
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    REPEATx4(ACCVGPR_ZERO, 0)
#endif
}

template <>
__device__ void gcnasm_accvgpr_zero<8>()
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    REPEATx4(ACCVGPR_ZERO, 0)
    REPEATx4(ACCVGPR_ZERO, 4)
#endif
}

template <>
__device__ void gcnasm_accvgpr_zero<16>()
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    REPEATx16(ACCVGPR_ZERO, 0)
#endif
}

template <>
__device__ void gcnasm_accvgpr_zero<32>()
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    REPEATx16(ACCVGPR_ZERO, 0)
    REPEATx16(ACCVGPR_ZERO, 16)
#endif
}

template <>
__device__ void gcnasm_accvgpr_zero<64>()
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    REPEATx64(ACCVGPR_ZERO, 0)
#endif
}

template <index_t Cycles>
__device__ void gcnasm_nop();

template <>
__device__ void gcnasm_nop<8>()
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    NOP(2)
#endif
}

template <>
__device__ void gcnasm_nop<32>()
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    NOP(8)
#endif
}

template <>
__device__ void gcnasm_nop<64>()
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    NOP(16)
#endif
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_32x32x1f32(const float&, const float&, float32_t*);

template <>
__device__ void gcnasm_mfma_f32_32x32x1f32<64, 64>(const float& reg_a, const float& reg_b, float32_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_32x32x1F32(0, reg_a, reg_b, 1, 0, 0)
    MFMA_F32_32x32x1F32(32, reg_a, reg_b, 1, 1, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(reg_a, reg_b, reg_c[0], 1, 0, 0);
    reg_c[1] = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(reg_a, reg_b, reg_c[1], 1, 1, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_32x32x1f32<32, 64>(const float& reg_a, const float& reg_b, float32_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_32x32x1F32(0, reg_a, reg_b, 1, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(reg_a, reg_b, reg_c[0], 1, 0, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_32x32x1f32<64, 32>(const float& reg_a, const float& reg_b, float32_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_32x32x1F32(0, reg_a, reg_b, 0, 0, 1)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(reg_a, reg_b, reg_c[0], 0, 0, 1);
#endif
}

__device__ void gcnasm_mfma_f32_32x32x2f32(const float& reg_a, const float& reg_b, float16_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_32x32x2F32(0, reg_a, reg_b, 0, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x2f32(reg_a, reg_b, reg_c[0], 0, 0, 0);
#endif
}

__device__ void gcnasm_mfma_f32_16x16x4f32(const float& reg_a, const float& reg_b, float4_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_16x16x4F32(0, reg_a, reg_b, 0, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x4f32(reg_a, reg_b, reg_c[0], 0, 0, 0);
#endif
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_16x16x1f32(const float&, const float&, float16_t*);

template <>
__device__ void gcnasm_mfma_f32_16x16x1f32<16, 64>(const float& reg_a, const float& reg_b, float16_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_16x16x1F32(0, reg_a, reg_b, 2, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x1f32(reg_a, reg_b, reg_c[0], 2, 0, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_16x16x1f32<64, 16>(const float& reg_a, const float& reg_b, float16_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_16x16x1F32(0, reg_a, reg_b, 0, 0, 4)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x1f32(reg_a, reg_b, reg_c[0], 0, 0, 4);
#endif
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_4x4x1f32(const float* a, const float* b, float4_t* reg_c);

template <>
__device__ void gcnasm_mfma_f32_4x4x1f32<4, 64>(const float* a, const float* b, float4_t* reg_c)
{
    const float reg_a = *a;
    const float reg_b = *b;

#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_4x4x1F32(0, reg_a, reg_b, 4, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_4x4x1f32(reg_a, reg_b, reg_c[0], 4, 0, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_4x4x1f32<8, 64>(const float* a, const float* b, float4_t* reg_c)
{
    const float reg_a_0 = *a;
    const float reg_b_0 = *b;
    const float reg_a_1 = *(a + 4);
    const float reg_b_1 = reg_b_0;

#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_4x4x1F32(0, reg_a_0, reg_b_0, 4, 0, 0)
    MFMA_F32_4x4x1F32(4, reg_a_1, reg_b_1, 4, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_4x4x1f32(reg_a_0, reg_b_0, reg_c[0], 4, 0, 0);
    reg_c[1] = llvm_intrin_amdgcn_mfma_f32_4x4x1f32(reg_a_1, reg_b_1, reg_c[1], 4, 0, 0);
#endif
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_32x32x4f16(const half4_t&,
                                           const half4_t&,
                                           float32_t*);
template <>
__device__ void gcnasm_mfma_f32_32x32x4f16<64, 64>(const half4_t& reg_a,
                                               const half4_t& reg_b,
                                               float32_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_32x32x4F16(0, reg_a, reg_b, 1, 0, 0)
    MFMA_F32_32x32x4F16(32, reg_a, reg_b, 1, 1, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x4f16(reg_a, reg_b, reg_c[0], 1, 0, 0);
    reg_c[1] = llvm_intrin_amdgcn_mfma_f32_32x32x4f16(reg_a, reg_b, reg_c[1], 1, 1, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_32x32x4f16<32, 64>(const half4_t& reg_a,
                                               const half4_t& reg_b,
                                               float32_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_32x32x4F16(0, reg_a, reg_b, 1, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x4f16(reg_a, reg_b, reg_c[0], 1, 0, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_32x32x4f16<64, 32>(const half4_t& reg_a,
                                               const half4_t& reg_b,
                                               float32_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_32x32x4F16(0, reg_a, reg_b, 0, 0, 1)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x4f16(reg_a, reg_b, reg_c[0], 0, 0, 1);
#endif
}

__device__ void gcnasm_mfma_f32_32x32x8f16(const half4_t& reg_a,
                                               const half4_t& reg_b,
                                               float16_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_32x32x8F16(0, reg_a, reg_b, 0, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x8f16(reg_a, reg_b, reg_c[0], 0, 0, 0);
#endif
}

__device__ void gcnasm_mfma_f32_16x16x16f16(const half4_t& reg_a,
                                               const half4_t& reg_b,
                                               float4_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_16x16x16F16(0, reg_a, reg_b, 0, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x16f16(reg_a, reg_b, reg_c[0], 0, 0, 0);
#endif
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_16x16x4f16(const half4_t& reg_a,
                                               const half4_t& reg_b,
                                               float16_t* reg_c);

template <>
__device__ void gcnasm_mfma_f32_16x16x4f16<16, 64>(const half4_t& reg_a,
                                               const half4_t& reg_b,
                                               float16_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_16x16x4F16(0, reg_a, reg_b, 2, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x4f16(reg_a, reg_b, reg_c[0], 2, 0, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_16x16x4f16<64, 16>(const half4_t& reg_a,
                                               const half4_t& reg_b,
                                               float16_t* reg_c)

{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_16x16x4F16(0, reg_a, reg_b, 0, 0, 4)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x4f16(reg_a, reg_b, reg_c[0], 0, 0, 4);
#endif
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_4x4x4f16(const half4_t *a,
                                               const half4_t* b,
                                               float4_t* reg_c);

template <>
__device__ void gcnasm_mfma_f32_4x4x4f16<4, 64>(const half4_t *a,
                                               const half4_t* b,
                                               float4_t* reg_c)
{
    const half4_t reg_a = *a;
    const half4_t reg_b = *b;

#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_4x4x4F16(0, reg_a, reg_b, 4, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_4x4x4f16(reg_a, reg_b, reg_c[0], 4, 0, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_4x4x4f16<8, 64>(const half4_t *a,
                                               const half4_t* b,
                                               float4_t* reg_c)
{
    const half4_t reg_a_0 = *a;
    const half4_t reg_b_0 = *b;
    const half4_t reg_a_1 = *(a + 4);
    const half4_t reg_b_1 = reg_b_0;

#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_4x4x4F16(0, reg_a_0, reg_b_0, 4, 0, 0)
    MFMA_F32_4x4x4F16(4, reg_a_1, reg_b_1, 4, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_4x4x4f16(reg_a_0, reg_b_0, reg_c[0], 4, 0, 0);
    reg_c[1] = llvm_intrin_amdgcn_mfma_f32_4x4x4f16(reg_a_1, reg_b_1, reg_c[1], 4, 0, 0);
#endif
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_32x32x2bf16(const ushort2_t&,
                                            const ushort2_t&,
                                            float32_t*);

template <>
__device__ void gcnasm_mfma_f32_32x32x2bf16<64, 64>(const ushort2_t& reg_a,
                                                const ushort2_t& reg_b,
                                                float32_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_32x32x2BF16(0, reg_a, reg_b, 1, 0, 0)
    MFMA_F32_32x32x2BF16(32, reg_a, reg_b, 1, 1, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a, reg_b, reg_c[0], 1, 0, 0);
    reg_c[1] = llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a, reg_b, reg_c[1], 1, 1, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_32x32x2bf16<32, 64>(const ushort2_t& reg_a,
                                                const ushort2_t& reg_b,
                                                float32_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_32x32x2BF16(0, reg_a, reg_b, 1, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a, reg_b, reg_c[0], 1, 0, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_32x32x2bf16<64, 32>(const ushort2_t& reg_a,
                                                const ushort2_t& reg_b,
                                                float32_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_32x32x2BF16(0, reg_a, reg_b, 0, 0, 1)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a, reg_b, reg_c[0], 0, 0, 1);
#endif
}

__device__ void gcnasm_mfma_f32_32x32x4bf16(const ushort2_t& reg_a,
                                                const ushort2_t& reg_b,
                                                float16_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_32x32x4BF16(0, reg_a, reg_b, 0, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x4bf16(reg_a, reg_b, reg_c[0], 0, 0, 0);
#endif
}

__device__ void gcnasm_mfma_f32_16x16x8bf16(const ushort2_t& reg_a,
                                                const ushort2_t& reg_b,
                                                float4_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_16x16x8BF16(0, reg_a, reg_b, 0, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x8bf16(reg_a, reg_b, reg_c[0], 0, 0, 0);
#endif
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_16x16x2bf16(const ushort2_t& reg_a,
                                                const ushort2_t& reg_b,
                                                float16_t* reg_c);

template <>
__device__ void gcnasm_mfma_f32_16x16x2bf16<16, 64>(const ushort2_t& reg_a,
                                                const ushort2_t& reg_b,
                                                float16_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_16x16x2BF16(0, reg_a, reg_b, 2, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x2bf16(reg_a, reg_b, reg_c[0], 2, 0, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_16x16x2bf16<64, 16>(const ushort2_t& reg_a,
                                                const ushort2_t& reg_b,
                                                float16_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_16x16x2BF16(0, reg_a, reg_b, 0, 0, 4)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x2bf16(reg_a, reg_b, reg_c[0], 0, 0, 4);
#endif
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_4x4x2bf16(const ushort2_t *a,
                                                const ushort2_t *b,
                                                float4_t* reg_c);

template <>
__device__ void gcnasm_mfma_f32_4x4x2bf16<4, 64>(const ushort2_t *a,
                                                const ushort2_t *b,
                                                float4_t* reg_c)
{
    const ushort2_t reg_a = *a;
    const ushort2_t reg_b = *b;

#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_4x4x2BF16(0, reg_a, reg_b, 4, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_4x4x2bf16(reg_a, reg_b, reg_c[0], 4, 0, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_4x4x2bf16<8, 64>(const ushort2_t *a,
                                                const ushort2_t *b,
                                                float4_t* reg_c)
{
    const ushort2_t reg_a_0 = *a;
    const ushort2_t reg_b_0 = *b;
    const ushort2_t reg_a_1 = *(a + 4);
    const ushort2_t reg_b_1 = reg_b_0;

#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    NOP(1)
    MFMA_F32_4x4x2BF16(0, reg_a_0, reg_b_0, 4, 0, 0)
    MFMA_F32_4x4x2BF16(4, reg_a_1, reg_b_1, 4, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_4x4x2bf16(reg_a_0, reg_b_0, reg_c[0], 4, 0, 0);
    reg_c[1] = llvm_intrin_amdgcn_mfma_f32_4x4x2bf16(reg_a_1, reg_b_1, reg_c[1], 4, 0, 0);
#endif
}
// clang-format on
}
#endif
