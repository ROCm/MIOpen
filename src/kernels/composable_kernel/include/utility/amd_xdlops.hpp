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
    half4_t, half4_t, float4_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.4x4x4f16");

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

#define S_NOP(n) \
    static_assert((n) >=0 && (n) <= 15, "s_nop operand must be within [0..15]"); \
    asm volatile("\n s_nop " #n " " : :);

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
  asm volatile("\
      v_accvgpr_read_b32 %0,  a[ 0] \n \
      v_accvgpr_read_b32 %1,  a[ 1] \n \
      v_accvgpr_read_b32 %2,  a[ 2] \n \
      v_accvgpr_read_b32 %3,  a[ 3] \n \
      v_accvgpr_read_b32 %4,  a[ 4] \n \
      v_accvgpr_read_b32 %5,  a[ 5] \n \
      v_accvgpr_read_b32 %6,  a[ 6] \n \
      v_accvgpr_read_b32 %7,  a[ 7] \n \
      "
      :
      "=v"(arch_reg[ 0]),
      "=v"(arch_reg[ 1]),
      "=v"(arch_reg[ 2]),
      "=v"(arch_reg[ 3]),
      "=v"(arch_reg[ 4]),
      "=v"(arch_reg[ 5]),
      "=v"(arch_reg[ 6]),
      "=v"(arch_reg[ 7])
      :);

#else
    (void)arch_reg;
#endif
}

template <>
__device__ void gcnasm_accvgpr_read<16>(float* arch_reg)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    asm volatile("\
      v_accvgpr_read_b32 %0,  a[ 0] \n \
      v_accvgpr_read_b32 %1,  a[ 1] \n \
      v_accvgpr_read_b32 %2,  a[ 2] \n \
      v_accvgpr_read_b32 %3,  a[ 3] \n \
      v_accvgpr_read_b32 %4,  a[ 4] \n \
      v_accvgpr_read_b32 %5,  a[ 5] \n \
      v_accvgpr_read_b32 %6,  a[ 6] \n \
      v_accvgpr_read_b32 %7,  a[ 7] \n \
      v_accvgpr_read_b32 %8,  a[ 8] \n \
      v_accvgpr_read_b32 %9,  a[ 9] \n \
      v_accvgpr_read_b32 %10, a[10] \n \
      v_accvgpr_read_b32 %11, a[11] \n \
      v_accvgpr_read_b32 %12, a[12] \n \
      v_accvgpr_read_b32 %13, a[13] \n \
      v_accvgpr_read_b32 %14, a[14] \n \
      v_accvgpr_read_b32 %15, a[15] \n \
      "
      :
      "=v"(arch_reg[ 0]),
      "=v"(arch_reg[ 1]),
      "=v"(arch_reg[ 2]),
      "=v"(arch_reg[ 3]),
      "=v"(arch_reg[ 4]),
      "=v"(arch_reg[ 5]),
      "=v"(arch_reg[ 6]),
      "=v"(arch_reg[ 7]),
      "=v"(arch_reg[ 8]),
      "=v"(arch_reg[ 9]),
      "=v"(arch_reg[10]),
      "=v"(arch_reg[11]),
      "=v"(arch_reg[12]),
      "=v"(arch_reg[13]),
      "=v"(arch_reg[14]),
      "=v"(arch_reg[15])
      :);

#else
    (void)arch_reg;
#endif
}

template <>
__device__ void gcnasm_accvgpr_read<32>(float* arch_reg)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    asm volatile("\
      v_accvgpr_read_b32 %0,  a[ 0] \n \
      v_accvgpr_read_b32 %1,  a[ 1] \n \
      v_accvgpr_read_b32 %2,  a[ 2] \n \
      v_accvgpr_read_b32 %3,  a[ 3] \n \
      v_accvgpr_read_b32 %4,  a[ 4] \n \
      v_accvgpr_read_b32 %5,  a[ 5] \n \
      v_accvgpr_read_b32 %6,  a[ 6] \n \
      v_accvgpr_read_b32 %7,  a[ 7] \n \
      v_accvgpr_read_b32 %8,  a[ 8] \n \
      v_accvgpr_read_b32 %9,  a[ 9] \n \
      v_accvgpr_read_b32 %10, a[10] \n \
      v_accvgpr_read_b32 %11, a[11] \n \
      v_accvgpr_read_b32 %12, a[12] \n \
      v_accvgpr_read_b32 %13, a[13] \n \
      v_accvgpr_read_b32 %14, a[14] \n \
      v_accvgpr_read_b32 %15, a[15] \n \
      v_accvgpr_read_b32 %16, a[16] \n \
      v_accvgpr_read_b32 %17, a[17] \n \
      v_accvgpr_read_b32 %18, a[18] \n \
      v_accvgpr_read_b32 %19, a[19] \n \
      v_accvgpr_read_b32 %20, a[20] \n \
      v_accvgpr_read_b32 %21, a[21] \n \
      v_accvgpr_read_b32 %22, a[22] \n \
      v_accvgpr_read_b32 %23, a[23] \n \
      v_accvgpr_read_b32 %24, a[24] \n \
      v_accvgpr_read_b32 %25, a[25] \n \
      v_accvgpr_read_b32 %26, a[26] \n \
      v_accvgpr_read_b32 %27, a[27] \n \
      v_accvgpr_read_b32 %28, a[28] \n \
      v_accvgpr_read_b32 %29, a[29] \n \
      v_accvgpr_read_b32 %30, a[30] \n \
      v_accvgpr_read_b32 %31, a[31] \n \
      "
      :
      "=v"(arch_reg[ 0]),
      "=v"(arch_reg[ 1]),
      "=v"(arch_reg[ 2]),
      "=v"(arch_reg[ 3]),
      "=v"(arch_reg[ 4]),
      "=v"(arch_reg[ 5]),
      "=v"(arch_reg[ 6]),
      "=v"(arch_reg[ 7]),
      "=v"(arch_reg[ 8]),
      "=v"(arch_reg[ 9]),
      "=v"(arch_reg[10]),
      "=v"(arch_reg[11]),
      "=v"(arch_reg[12]),
      "=v"(arch_reg[13]),
      "=v"(arch_reg[14]),
      "=v"(arch_reg[15]),
      "=v"(arch_reg[16]),
      "=v"(arch_reg[17]),
      "=v"(arch_reg[18]),
      "=v"(arch_reg[19]),
      "=v"(arch_reg[20]),
      "=v"(arch_reg[21]),
      "=v"(arch_reg[22]),
      "=v"(arch_reg[23]),
      "=v"(arch_reg[24]),
      "=v"(arch_reg[25]),
      "=v"(arch_reg[26]),
      "=v"(arch_reg[27]),
      "=v"(arch_reg[28]),
      "=v"(arch_reg[29]),
      "=v"(arch_reg[30]),
      "=v"(arch_reg[31])
      :);
#else
    (void)arch_reg;
#endif
}

template <>
__device__ void gcnasm_accvgpr_read<64>(float* arch_reg)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    asm volatile("\
      v_accvgpr_read_b32 %0,  a[ 0] \n \
      v_accvgpr_read_b32 %1,  a[ 1] \n \
      v_accvgpr_read_b32 %2,  a[ 2] \n \
      v_accvgpr_read_b32 %3,  a[ 3] \n \
      v_accvgpr_read_b32 %4,  a[ 4] \n \
      v_accvgpr_read_b32 %5,  a[ 5] \n \
      v_accvgpr_read_b32 %6,  a[ 6] \n \
      v_accvgpr_read_b32 %7,  a[ 7] \n \
      v_accvgpr_read_b32 %8,  a[ 8] \n \
      v_accvgpr_read_b32 %9,  a[ 9] \n \
      v_accvgpr_read_b32 %10, a[10] \n \
      v_accvgpr_read_b32 %11, a[11] \n \
      v_accvgpr_read_b32 %12, a[12] \n \
      v_accvgpr_read_b32 %13, a[13] \n \
      v_accvgpr_read_b32 %14, a[14] \n \
      v_accvgpr_read_b32 %15, a[15] \n \
      v_accvgpr_read_b32 %16, a[16] \n \
      v_accvgpr_read_b32 %17, a[17] \n \
      v_accvgpr_read_b32 %18, a[18] \n \
      v_accvgpr_read_b32 %19, a[19] \n \
      v_accvgpr_read_b32 %20, a[20] \n \
      v_accvgpr_read_b32 %21, a[21] \n \
      v_accvgpr_read_b32 %22, a[22] \n \
      v_accvgpr_read_b32 %23, a[23] \n \
      v_accvgpr_read_b32 %24, a[24] \n \
      v_accvgpr_read_b32 %25, a[25] \n \
      v_accvgpr_read_b32 %26, a[26] \n \
      v_accvgpr_read_b32 %27, a[27] \n \
      v_accvgpr_read_b32 %28, a[28] \n \
      v_accvgpr_read_b32 %29, a[29] \n \
      v_accvgpr_read_b32 %30, a[30] \n \
      v_accvgpr_read_b32 %31, a[31] \n \
      v_accvgpr_read_b32 %32, a[32] \n \
      v_accvgpr_read_b32 %33, a[33] \n \
      v_accvgpr_read_b32 %34, a[34] \n \
      v_accvgpr_read_b32 %35, a[35] \n \
      v_accvgpr_read_b32 %36, a[36] \n \
      v_accvgpr_read_b32 %37, a[37] \n \
      v_accvgpr_read_b32 %38, a[38] \n \
      v_accvgpr_read_b32 %39, a[39] \n \
      v_accvgpr_read_b32 %40, a[40] \n \
      v_accvgpr_read_b32 %41, a[41] \n \
      v_accvgpr_read_b32 %42, a[42] \n \
      v_accvgpr_read_b32 %43, a[43] \n \
      v_accvgpr_read_b32 %44, a[44] \n \
      v_accvgpr_read_b32 %45, a[45] \n \
      v_accvgpr_read_b32 %46, a[46] \n \
      v_accvgpr_read_b32 %47, a[47] \n \
      v_accvgpr_read_b32 %48, a[48] \n \
      v_accvgpr_read_b32 %49, a[49] \n \
      v_accvgpr_read_b32 %50, a[50] \n \
      v_accvgpr_read_b32 %51, a[51] \n \
      v_accvgpr_read_b32 %52, a[52] \n \
      v_accvgpr_read_b32 %53, a[53] \n \
      v_accvgpr_read_b32 %54, a[54] \n \
      v_accvgpr_read_b32 %55, a[55] \n \
      v_accvgpr_read_b32 %56, a[56] \n \
      v_accvgpr_read_b32 %57, a[57] \n \
      v_accvgpr_read_b32 %58, a[58] \n \
      v_accvgpr_read_b32 %59, a[59] \n \
      v_accvgpr_read_b32 %60, a[60] \n \
      v_accvgpr_read_b32 %61, a[61] \n \
      v_accvgpr_read_b32 %62, a[62] \n \
      v_accvgpr_read_b32 %63, a[63] \n \
      "
      :
      "=v"(arch_reg[ 0]),
      "=v"(arch_reg[ 1]),
      "=v"(arch_reg[ 2]),
      "=v"(arch_reg[ 3]),
      "=v"(arch_reg[ 4]),
      "=v"(arch_reg[ 5]),
      "=v"(arch_reg[ 6]),
      "=v"(arch_reg[ 7]),
      "=v"(arch_reg[ 8]),
      "=v"(arch_reg[ 9]),
      "=v"(arch_reg[10]),
      "=v"(arch_reg[11]),
      "=v"(arch_reg[12]),
      "=v"(arch_reg[13]),
      "=v"(arch_reg[14]),
      "=v"(arch_reg[15]),
      "=v"(arch_reg[16]),
      "=v"(arch_reg[17]),
      "=v"(arch_reg[18]),
      "=v"(arch_reg[19]),
      "=v"(arch_reg[20]),
      "=v"(arch_reg[21]),
      "=v"(arch_reg[22]),
      "=v"(arch_reg[23]),
      "=v"(arch_reg[24]),
      "=v"(arch_reg[25]),
      "=v"(arch_reg[26]),
      "=v"(arch_reg[27]),
      "=v"(arch_reg[28]),
      "=v"(arch_reg[29]),
      "=v"(arch_reg[30]),
      "=v"(arch_reg[31]),
      "=v"(arch_reg[32]),
      "=v"(arch_reg[33]),
      "=v"(arch_reg[34]),
      "=v"(arch_reg[35]),
      "=v"(arch_reg[36]),
      "=v"(arch_reg[37]),
      "=v"(arch_reg[38]),
      "=v"(arch_reg[39]),
      "=v"(arch_reg[40]),
      "=v"(arch_reg[41]),
      "=v"(arch_reg[42]),
      "=v"(arch_reg[43]),
      "=v"(arch_reg[44]),
      "=v"(arch_reg[45]),
      "=v"(arch_reg[46]),
      "=v"(arch_reg[47]),
      "=v"(arch_reg[48]),
      "=v"(arch_reg[49]),
      "=v"(arch_reg[50]),
      "=v"(arch_reg[51]),
      "=v"(arch_reg[52]),
      "=v"(arch_reg[53]),
      "=v"(arch_reg[54]),
      "=v"(arch_reg[55]),
      "=v"(arch_reg[56]),
      "=v"(arch_reg[57]),
      "=v"(arch_reg[58]),
      "=v"(arch_reg[59]),
      "=v"(arch_reg[60]),
      "=v"(arch_reg[61]),
      "=v"(arch_reg[62]),
      "=v"(arch_reg[63])
      :);
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
    S_NOP(1)
#endif
}

template <>
__device__ void gcnasm_accvgpr_zero<8>()
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    REPEATx4(ACCVGPR_ZERO, 0)
    REPEATx4(ACCVGPR_ZERO, 4)
    S_NOP(1)
#endif
}

template <>
__device__ void gcnasm_accvgpr_zero<16>()
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    REPEATx16(ACCVGPR_ZERO, 0)
    S_NOP(1)
#endif
}

template <>
__device__ void gcnasm_accvgpr_zero<32>()
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    REPEATx16(ACCVGPR_ZERO, 0)
    REPEATx16(ACCVGPR_ZERO, 16)
    S_NOP(1)
#endif
}

template <>
__device__ void gcnasm_accvgpr_zero<64>()
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    REPEATx64(ACCVGPR_ZERO, 0)
    S_NOP(1)
#endif
}

template <index_t Cycles>
__device__ void gcnasm_nop();

template <>
__device__ void gcnasm_nop<8>()
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    S_NOP(3)
#endif
}

template <>
__device__ void gcnasm_nop<32>()
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    S_NOP(9)
#endif
}

template <>
__device__ void gcnasm_nop<64>()
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    S_NOP(8)
    S_NOP(8)
#endif
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_32x32x1f32(const float&, const float&, float32_t*);

template <>
__device__ void gcnasm_mfma_f32_32x32x1f32<64, 64>(const float& reg_a, const float& reg_b, float32_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
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
    MFMA_F32_32x32x1F32(0, reg_a, reg_b, 0, 0, 1)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(reg_a, reg_b, reg_c[0], 0, 0, 1);
#endif
}

__device__ void gcnasm_mfma_f32_32x32x2f32(const float& reg_a, const float& reg_b, float16_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    MFMA_F32_32x32x2F32(0, reg_a, reg_b, 0, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x2f32(reg_a, reg_b, reg_c[0], 0, 0, 0);
#endif
}

__device__ void gcnasm_mfma_f32_16x16x4f32(const float& reg_a, const float& reg_b, float4_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
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
    MFMA_F32_16x16x1F32(0, reg_a, reg_b, 0, 0, 4)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x1f32(reg_a, reg_b, reg_c[0], 0, 0, 4);
#endif
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_4x4x1f32(const float& reg_a, const float& reg_b, float4_t* reg_c);

template <>
__device__ void gcnasm_mfma_f32_4x4x1f32<4, 64>(const float& reg_a, const float& reg_b, float4_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    MFMA_F32_4x4x1F32(0, reg_a, reg_b, 4, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_4x4x1f32(reg_a, reg_b, reg_c[0], 4, 0, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_4x4x1f32<8, 64>(const float& reg_a, const float& reg_b, float4_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    MFMA_F32_4x4x1F32(0, reg_a, reg_b, 4, 0, 0)
    MFMA_F32_4x4x1F32(4, reg_a, reg_b, 4, 1, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_4x4x1f32(reg_a, reg_b, reg_c[0], 4, 0, 0);
    reg_c[1] = llvm_intrin_amdgcn_mfma_f32_4x4x1f32(reg_a, reg_b, reg_c[1], 4, 1, 0);
#endif
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_32x32x4f16(const half4_t&,
                                           const half4_t&,
                                           float32_t*);
template <>
__device__ void gcnasm_mfma_f32_32x32x4f16<64, 64>(const half4_t& reg_a, const half4_t& reg_b, float32_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    MFMA_F32_32x32x4F16(0, reg_a, reg_b, 1, 0, 0)
    MFMA_F32_32x32x4F16(32, reg_a, reg_b, 1, 1, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x4f16(reg_a, reg_b, reg_c[0], 1, 0, 0);
    reg_c[1] = llvm_intrin_amdgcn_mfma_f32_32x32x4f16(reg_a, reg_b, reg_c[1], 1, 1, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_32x32x4f16<32, 64>(const half4_t& reg_a, const half4_t& reg_b, float32_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    MFMA_F32_32x32x4F16(0, reg_a, reg_b, 1, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x4f16(reg_a, reg_b, reg_c[0], 1, 0, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_32x32x4f16<64, 32>(const half4_t& reg_a, const half4_t& reg_b, float32_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    MFMA_F32_32x32x4F16(0, reg_a, reg_b, 0, 0, 1)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x4f16(reg_a, reg_b, reg_c[0], 0, 0, 1);
#endif
}

__device__ void gcnasm_mfma_f32_32x32x8f16(const half4_t& reg_a, const half4_t& reg_b, float16_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    MFMA_F32_32x32x8F16(0, reg_a, reg_b, 0, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x8f16(reg_a, reg_b, reg_c[0], 0, 0, 0);
#endif
}

__device__ void gcnasm_mfma_f32_16x16x16f16(const half4_t& reg_a, const half4_t& reg_b, float4_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    MFMA_F32_16x16x16F16(0, reg_a, reg_b, 0, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x16f16(reg_a, reg_b, reg_c[0], 0, 0, 0);
#endif
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_16x16x4f16(const half4_t& reg_a, const half4_t& reg_b, float16_t* reg_c);

template <>
__device__ void gcnasm_mfma_f32_16x16x4f16<16, 64>(const half4_t& reg_a, const half4_t& reg_b, float16_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    MFMA_F32_16x16x4F16(0, reg_a, reg_b, 2, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x4f16(reg_a, reg_b, reg_c[0], 2, 0, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_16x16x4f16<64, 16>(const half4_t& reg_a, const half4_t& reg_b, float16_t* reg_c)

{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    MFMA_F32_16x16x4F16(0, reg_a, reg_b, 0, 0, 4)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x4f16(reg_a, reg_b, reg_c[0], 0, 0, 4);
#endif
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_4x4x4f16(const half4_t& reg_a, const half4_t& reg_b, float4_t* reg_c);

template <>
__device__ void gcnasm_mfma_f32_4x4x4f16<4, 64>(const half4_t& reg_a, const half4_t& reg_b, float4_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    MFMA_F32_4x4x4F16(0, reg_a, reg_b, 4, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_4x4x4f16(reg_a, reg_b, reg_c[0], 4, 0, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_4x4x4f16<8, 64>(const half4_t& reg_a, const half4_t& reg_b, float4_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    MFMA_F32_4x4x4F16(0, reg_a, reg_b, 4, 0, 0)
    MFMA_F32_4x4x4F16(4, reg_a, reg_b, 4, 1, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_4x4x4f16(reg_a, reg_b, reg_c[0], 4, 0, 0);
    reg_c[1] = llvm_intrin_amdgcn_mfma_f32_4x4x4f16(reg_a, reg_b, reg_c[1], 4, 1, 0);
#endif
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_32x32x2bf16(const ushort2_t&, const ushort2_t&, float32_t*);

template <>
__device__ void gcnasm_mfma_f32_32x32x2bf16<64, 64>(const ushort2_t& reg_a, const ushort2_t& reg_b, float32_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    MFMA_F32_32x32x2BF16(0, reg_a, reg_b, 1, 0, 0)
    MFMA_F32_32x32x2BF16(32, reg_a, reg_b, 1, 1, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a, reg_b, reg_c[0], 1, 0, 0);
    reg_c[1] = llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a, reg_b, reg_c[1], 1, 1, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_32x32x2bf16<32, 64>(const ushort2_t& reg_a, const ushort2_t& reg_b, float32_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    MFMA_F32_32x32x2BF16(0, reg_a, reg_b, 1, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a, reg_b, reg_c[0], 1, 0, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_32x32x2bf16<64, 32>(const ushort2_t& reg_a, const ushort2_t& reg_b, float32_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    MFMA_F32_32x32x2BF16(0, reg_a, reg_b, 0, 0, 1)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a, reg_b, reg_c[0], 0, 0, 1);
#endif
}

__device__ void gcnasm_mfma_f32_32x32x4bf16(const ushort2_t& reg_a, const ushort2_t& reg_b, float16_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    MFMA_F32_32x32x4BF16(0, reg_a, reg_b, 0, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x4bf16(reg_a, reg_b, reg_c[0], 0, 0, 0);
#endif
}

__device__ void gcnasm_mfma_f32_16x16x8bf16(const ushort2_t& reg_a, const ushort2_t& reg_b, float4_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    MFMA_F32_16x16x8BF16(0, reg_a, reg_b, 0, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x8bf16(reg_a, reg_b, reg_c[0], 0, 0, 0);
#endif
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_16x16x2bf16(const ushort2_t& reg_a, const ushort2_t& reg_b, float16_t* reg_c);

template <>
__device__ void gcnasm_mfma_f32_16x16x2bf16<16, 64>(const ushort2_t& reg_a, const ushort2_t& reg_b, float16_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    MFMA_F32_16x16x2BF16(0, reg_a, reg_b, 2, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x2bf16(reg_a, reg_b, reg_c[0], 2, 0, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_16x16x2bf16<64, 16>(const ushort2_t& reg_a, const ushort2_t& reg_b, float16_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    MFMA_F32_16x16x2BF16(0, reg_a, reg_b, 0, 0, 4)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x2bf16(reg_a, reg_b, reg_c[0], 0, 0, 4);
#endif
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_4x4x2bf16(const ushort2_t& reg_a, const ushort2_t& reg_b, float4_t* reg_c);

template <>
__device__ void gcnasm_mfma_f32_4x4x2bf16<4, 64>(const ushort2_t& reg_a, const ushort2_t& reg_b, float4_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    MFMA_F32_4x4x2BF16(0, reg_a, reg_b, 4, 0, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_4x4x2bf16(reg_a, reg_b, reg_c[0], 4, 0, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_4x4x2bf16<8, 64>(const ushort2_t& reg_a, const ushort2_t& reg_b, float4_t* reg_c)
{
#if CK_USE_AMD_XDLOPS_INLINE_ASM
    (void)reg_c;
    MFMA_F32_4x4x2BF16(0, reg_a, reg_b, 4, 0, 0)
    MFMA_F32_4x4x2BF16(4, reg_a, reg_b, 4, 1, 0)
#else
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_4x4x2bf16(reg_a, reg_b, reg_c[0], 4, 0, 0);
    reg_c[1] = llvm_intrin_amdgcn_mfma_f32_4x4x2bf16(reg_a, reg_b, reg_c[1], 4, 1, 0);
#endif
}
// clang-format on
}
#endif
