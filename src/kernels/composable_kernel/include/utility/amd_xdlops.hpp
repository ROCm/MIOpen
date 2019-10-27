#ifndef CK_AMD_XDLOPS_HPP
#define CK_AMD_XDLOPS_HPP

#include "float_type.hpp"

namespace ck {

// A, B, C, cbsz, abid, blgp
extern "C" __device__ float32_t llvm_intrin_amdgcn_mfma_f32_32x32x1f32(
    float, float, float32_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x1f32");

extern "C" __device__ float16_t llvm_intrin_amdgcn_mfma_f32_32x32x2f32(
    float, float, float16_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x2f32");

extern "C" __device__ float32_t llvm_intrin_amdgcn_mfma_f32_32x32x4f16(
    half4_t, half4_t, float32_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x4f16");

extern "C" __device__ float16_t llvm_intrin_amdgcn_mfma_f32_32x32x8f16(
    half4_t, half4_t, float16_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x8f16");

extern "C" __device__ float32_t llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(
    ushort2_t, ushort2_t, float32_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x2bf16");

extern "C" __device__ float16_t llvm_intrin_amdgcn_mfma_f32_32x32x4bf16(
    ushort2_t, ushort2_t, float16_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x4bf16");
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

#define ACCVGPR_READ(acc_reg_id) \
    asm volatile("v_accvgpr_read_b32 %0, a[" #acc_reg_id "]" : "=v"(arch_reg[acc_reg_id]) :);

#define ACCVGPR_WRITE(acc_reg_id) \
    asm volatile("v_accvgpr_write_b32 a[" #acc_reg_id "], %0" : : "v"(arch_reg[acc_reg_id]));

#define ACCVGPR_ZERO(acc_reg_id) \
    asm volatile("v_accvgpr_write_b32 a[" #acc_reg_id "], 0" : :);

template <index_t Size>
__device__ void gcnasm_accvgpr_read(float*);


template <>
__device__ void gcnasm_accvgpr_read<16>(float* arch_reg)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    NOP(16)
    REPEATx16(ACCVGPR_READ, 0)
#else
    (void)arch_reg;
#endif
}

template <>
__device__ void gcnasm_accvgpr_read<32>(float* arch_reg)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    NOP(16)
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
    NOP(16)
    REPEATx64(ACCVGPR_READ, 0)
#else
    (void)arch_reg;
#endif
}

template <index_t MPerWave>
__device__ void gcnasm_accvgpr_zero();

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

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_32x32x1f32(float&, float&, float32_t*);

template <>
__device__ void gcnasm_mfma_f32_32x32x1f32<64, 64>(float& reg_a, float& reg_b, float32_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    NOP(1)
    (void)reg_c;
    MFMA_F32_32x32x1F32(0, reg_a, reg_b, 1, 0, 0)
    MFMA_F32_32x32x1F32(32, reg_a, reg_b, 1, 1, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(reg_a, reg_b, reg_c[0], 1, 0, 0);
    reg_c[1] = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(reg_a, reg_b, reg_c[1], 1, 1, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_32x32x1f32<32, 64>(float& reg_a, float& reg_b, float32_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    NOP(1)
    (void)reg_c;
    MFMA_F32_32x32x1F32(0, reg_a, reg_b, 1, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(reg_a, reg_b, reg_c[0], 1, 0, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_32x32x1f32<64, 32>(float& reg_a, float& reg_b, float32_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    NOP(1)
    (void)reg_c;
    MFMA_F32_32x32x1F32(0, reg_a, reg_b, 0, 0, 1)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(reg_a, reg_b, reg_c[0], 0, 0, 1);
#endif
}


__device__ void gcnasm_mfma_f32_32x32x2f32(float& reg_a, float& reg_b, float16_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    NOP(1)
    (void)reg_c;
    MFMA_F32_32x32x2F32(0, reg_a, reg_b, 0, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x2f32(reg_a, reg_b, reg_c[0], 0, 0, 0);
#endif
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_32x32x4f16(typename vector_type<half, 4>::MemoryType&,
                                           typename vector_type<half, 4>::MemoryType&,
                                           float32_t*);
template <>
__device__ void gcnasm_mfma_f32_32x32x4f16<64, 64>(typename vector_type<half, 4>::MemoryType& reg_a,
                                               typename vector_type<half, 4>::MemoryType& reg_b,
                                               float32_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    NOP(1)
    (void)reg_c;
    MFMA_F32_32x32x4F16(0, reg_a, reg_b, 1, 0, 0)
    MFMA_F32_32x32x4F16(32, reg_a, reg_b, 1, 1, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x4f16(reg_a, reg_b, reg_c[0], 1, 0, 0);
    reg_c[1] = llvm_intrin_amdgcn_mfma_f32_32x32x4f16(reg_a, reg_b, reg_c[1], 1, 1, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_32x32x4f16<32, 64>(typename vector_type<half, 4>::MemoryType& reg_a,
                                               typename vector_type<half, 4>::MemoryType& reg_b,
                                               float32_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    NOP(1)
    (void)reg_c;
    MFMA_F32_32x32x4F16(0, reg_a, reg_b, 1, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x4f16(reg_a, reg_b, reg_c[0], 1, 0, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_32x32x4f16<64, 32>(typename vector_type<half, 4>::MemoryType& reg_a,
                                               typename vector_type<half, 4>::MemoryType& reg_b,
                                               float32_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    NOP(1)
    (void)reg_c;
    MFMA_F32_32x32x4F16(0, reg_a, reg_b, 0, 0, 1)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x4f16(reg_a, reg_b, reg_c[0], 0, 0, 1);
#endif
}

__device__ void gcnasm_mfma_f32_32x32x8f16(typename vector_type<half, 4>::MemoryType& reg_a,
                                               typename vector_type<half, 4>::MemoryType& reg_b,
                                               float16_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    NOP(1)
    (void)reg_c;
    MFMA_F32_32x32x8F16(0, reg_a, reg_b, 0, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x8f16(reg_a, reg_b, reg_c[0], 0, 0, 0);
#endif
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_32x32x2bf16(typename vector_type<ushort, 2>::MemoryType&,
                                            typename vector_type<ushort, 2>::MemoryType&,
                                            float32_t*);

template <>
__device__ void gcnasm_mfma_f32_32x32x2bf16<64, 64>(typename vector_type<ushort, 2>::MemoryType& reg_a,
                                                typename vector_type<ushort, 2>::MemoryType& reg_b,
                                                float32_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    NOP(1)
    (void)reg_c;
    MFMA_F32_32x32x2BF16(0, reg_a, reg_b, 1, 0, 0)
    MFMA_F32_32x32x2BF16(32, reg_a, reg_b, 1, 1, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a, reg_b, reg_c[0], 1, 0, 0);
    reg_c[1] = llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a, reg_b, reg_c[1], 1, 1, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_32x32x2bf16<32, 64>(typename vector_type<ushort, 2>::MemoryType& reg_a,
                                                typename vector_type<ushort, 2>::MemoryType& reg_b,
                                                float32_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    NOP(1)
    (void)reg_c;
    MFMA_F32_32x32x2BF16(0, reg_a, reg_b, 1, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a, reg_b, reg_c[0], 1, 0, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_32x32x2bf16<64, 32>(typename vector_type<ushort, 2>::MemoryType& reg_a,
                                                typename vector_type<ushort, 2>::MemoryType& reg_b,
                                                float32_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    NOP(1)
    (void)reg_c;
    MFMA_F32_32x32x2BF16(0, reg_a, reg_b, 0, 0, 1)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a, reg_b, reg_c[0], 0, 0, 1);
#endif
}

__device__ void gcnasm_mfma_f32_32x32x4bf16(typename vector_type<ushort, 2>::MemoryType& reg_a,
                                                typename vector_type<ushort, 2>::MemoryType& reg_b,
                                                float16_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    NOP(1)
    (void)reg_c;
    MFMA_F32_32x32x4BF16(0, reg_a, reg_b, 0, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x4bf16(reg_a, reg_b, reg_c[0], 0, 0, 0);
#endif
}
// clang-format on
}
#endif
