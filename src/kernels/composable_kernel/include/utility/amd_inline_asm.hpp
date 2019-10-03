#ifndef CK_AMD_INLINE_ASM_HPP
#define CK_AMD_INLINE_ASM_HPP

#include "vector_type.hpp"

#define WORKAROUND_SWDEV_202749 1

namespace ck {

#if !CK_USE_INLINE_ASM_XDLOPS
// A, B, C, cbsz, abid, blgp
extern "C" __device__ float32_t __llvm_amdgcn_mfma_f32_32x32x1f32(
    float, float, float32_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x1f32");

extern "C" __device__ float32_t __llvm_amdgcn_mfma_f32_32x32x4f16(
    half4_t, half4_t, float32_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x4f16");

extern "C" __device__ float32_t __llvm_amdgcn_mfma_f32_32x32x2bf16(
    ushort2_t, ushort2_t, float32_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x2bf16");
#endif

// cast a pointer of LDS to its address

extern "C" __attribute__((address_space(3))) __device__ void* __to_local(const void* p);

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

#define DS_READ_B32(off)                                                                      \
    if(offset == off)                                                                         \
    {                                                                                         \
        asm volatile("ds_read_b32 %0, %1 offset:" #off " " : "=v"(r) : "v"(__to_local(lds))); \
    }

#define DS_READ_B128(off)                                                                      \
    if(offset == off)                                                                          \
    {                                                                                          \
        asm volatile("ds_read_b128 %0, %1 offset:" #off " " : "=v"(r) : "v"(__to_local(lds))); \
    }

#define DS_WRITE_B128(off)                                                                      \
    if(offset == off)                                                                           \
    {                                                                                           \
        asm volatile("ds_write_b128 %0, %1 offset:" #off " " : : "v"(__to_local(lds)), "v"(r)); \
    }

#define MFMA_F32_32x32x1F32(acc, reg_a, reg_b, cbsz, abid, blgp)                           \
    asm volatile("v_mfma_f32_32x32x1f32 a[" #acc ":" #acc "+31], %0, %1, a[" #acc ":" #acc \
                 "+31] cbsz: " #cbsz " abid: " #abid " blgp:" #blgp " "                    \
                 :                                                                         \
                 : "v"(reg_a), "v"(reg_b));

#define MFMA_F32_32x32x4F16(acc, reg_a, reg_b, cbsz, abid, blgp)                           \
    asm volatile("v_mfma_f32_32x32x4f16 a[" #acc ":" #acc "+31], %0, %1, a[" #acc ":" #acc \
                 "+31] cbsz: " #cbsz " abid: " #abid " blgp:" #blgp " "                    \
                 :                                                                         \
                 : "v"(reg_a), "v"(reg_b));


#define MFMA_F32_32x32x2BF16(acc, reg_a, reg_b, cbsz, abid, blgp)                           \
    asm volatile("v_mfma_f32_32x32x2bf16 a[" #acc ":" #acc "+31], %0, %1, a[" #acc ":" #acc \
                 "+31] cbsz: " #cbsz " abid: " #abid " blgp:" #blgp " "                     \
                 :                                                                          \
                 : "v"(reg_a), "v"(reg_b));

#define ACCVGPR_READ(acc_reg_id) \
    asm volatile("v_accvgpr_read_b32 %0, a[" #acc_reg_id "]" : "=v"(arch_reg[acc_reg_id]) :);

#define ACCVGPR_WRITE(acc_reg_id) \
    asm volatile("v_accvgpr_write_b32 a[" #acc_reg_id "], %0" : : "v"(arch_reg[acc_reg_id]));

#define ACCVGPR_ZERO(acc_reg_id) \
    asm volatile("v_accvgpr_write_b32 a[" #acc_reg_id "], 0" : :);

#define S_WAIT_VMCNT(id)                             \
    if(cnt == id)                                    \
    {                                                \
        asm volatile("s_waitcnt vmcnt(" #id ")" ::); \
    }

#define S_WAIT_LGKMCNT(id)                             \
    if(cnt == id)                                      \
    {                                                  \
        asm volatile("s_waitcnt lgkmcnt(" #id ")" ::); \
    }

__device__ void s_wait_vmcnt(index_t cnt) { REPEATx4(S_WAIT_VMCNT, 0) }

__device__ void s_wait_lgkmcnt(index_t cnt) { REPEATx4(S_WAIT_LGKMCNT, 0) }

__device__ void outerProduct1x4(const float* a, const float* b, float* c)
{
    asm volatile("\n \
            v_mac_f32 %0, %4, %5 \n \
            v_mac_f32 %1, %4, %6 \n \
            v_mac_f32 %2, %4, %7 \n \
            v_mac_f32 %3, %4, %8 \n \
            "
                 : "=v"(c[0]), "=v"(c[1]), "=v"(c[2]), "=v"(c[3])
                 : "v"(a[0]),
                   "v"(b[0]),
                   "v"(b[1]),
                   "v"(b[2]),
                   "v"(b[3]),
                   "0"(c[0]),
                   "1"(c[1]),
                   "2"(c[2]),
                   "3"(c[3]));
}

__device__ void outerProduct1x2(const float* a, const float* b, float* c)
{
// disable inline asm due to the compiler issue: SWDEV-202749
///\to-do: enable the inline asm after the compiler fix
#if WORKAROUND_SWDEV_202749
    c[0] += a[0] * b[0];
    c[1] += a[0] * b[1];
#else
    asm volatile("\n \
            v_mac_f32 %0, %2, %3 \n \
            v_mac_f32 %1, %2, %4 \n \
            "
                 : "=v"(c[0]), "=v"(c[1])
                 : "v"(a[0]), "v"(b[0]), "v"(b[1]), "0"(c[0]), "1"(c[1]));
#endif
}

__device__ void outerProduct1x4(const float& a,
                                const vector_type<float, 4>::MemoryType& b,
                                vector_type<float, 4>::MemoryType& c)
{
    outerProduct1x4(&a, reinterpret_cast<const float*>(&b), reinterpret_cast<float*>(&c));
}

__device__ void outerProduct1x2(const float& a,
                                const vector_type<float, 2>::MemoryType& b,
                                vector_type<float, 2>::MemoryType& c)
{
    outerProduct1x2(&a, reinterpret_cast<const float*>(&b), reinterpret_cast<float*>(&c));
}

__device__ void outerProduct1x4dot2TwoTimes(const half2* a, const half2* b, float* c)
{
    asm volatile("\n \
            v_dot2_f32_f16 %0, %4, %6  %0\n \
            v_dot2_f32_f16 %1, %4, %8  %1\n \
            v_dot2_f32_f16 %2, %4, %10 %2\n \
            v_dot2_f32_f16 %3, %4, %12 %3\n \
            v_dot2_f32_f16 %0, %5, %7  %0\n \
            v_dot2_f32_f16 %1, %5, %9  %1\n \
            v_dot2_f32_f16 %2, %5, %11 %2\n \
            v_dot2_f32_f16 %3, %5, %13 %3\n \
            "
                 : "=v"(c[0]), "=v"(c[1]), "=v"(c[2]), "=v"(c[3]) // Dest registers
                 : "v"(a[0]),
                   "v"(a[1]), // 1st Src registers for 2 half2 registers
                   "v"(b[0]),
                   "v"(b[1]),
                   "v"(b[2]),
                   "v"(b[3]), // 2nd Src registers for 2 half2 registers
                   "v"(b[4]),
                   "v"(b[5]),
                   "v"(b[6]),
                   "v"(b[7]), // 2nd Src registers for 2 half2 registers
                   "0"(c[0]),
                   "1"(c[1]),
                   "2"(c[2]),
                   "3"(c[3])); // 3rd Src Acc registers for 2 half2 registers
}

__device__ void outerProduct1x4dot2(const half2* a, const half2* b, float* c)
{
    asm volatile("\n \
            v_dot2_f32_f16 %0, %4, %5  %0\n \
            v_dot2_f32_f16 %1, %4, %6  %1\n \
            v_dot2_f32_f16 %2, %4, %7  %2\n \
            v_dot2_f32_f16 %3, %4, %8  %3\n \
            "
                 : "=v"(c[0]), "=v"(c[1]), "=v"(c[2]), "=v"(c[3]) // Dest registers
                 : "v"(a[0]), // 1st Src register for 1 half2 registers
                   "v"(b[0]), // 2nd Src register
                   "v"(b[1]),
                   "v"(b[2]),
                   "v"(b[3]),
                   "0"(c[0]), // 3rd Src register
                   "1"(c[1]),
                   "2"(c[2]),
                   "3"(c[3]));
}

__device__ void outerProduct1x2dot2TwoTimes(const half2* a, const half2* b, float* c)
{
    asm volatile("\n \
            v_dot2_f32_f16 %0, %2, %4  %0\n \
            v_dot2_f32_f16 %1, %2, %6  %1\n \
            v_dot2_f32_f16 %0, %3, %5  %0\n \
            v_dot2_f32_f16 %1, %3, %7  %1\n \
            "
                 : "=v"(c[0]), "=v"(c[1]) // Dest registers
                 : "v"(a[0]),
                   "v"(a[1]), // 1st Src registers for 2 half2 registers
                   "v"(b[0]),
                   "v"(b[1]),
                   "v"(b[2]),
                   "v"(b[3]), // 2nd Src registers for 2 half2 registers
                   "0"(c[0]),
                   "1"(c[1])); // 3rd Src Acc registers for 2 half2 registers
}

__device__ void outerProduct1x2dot2(const half2* a, const half2* b, float* c)
{
    asm volatile("\n \
            v_dot2_f32_f16 %0, %2, %3  %0\n \
            v_dot2_f32_f16 %1, %2, %4  %1\n \
            "
                 : "=v"(c[0]), "=v"(c[1]) // Dest registers
                 : "v"(a[0]),             // 1st Src register for 1 half2 registers
                   "v"(b[0]),             // 2nd Src register
                   "v"(b[1]),
                   "0"(c[0]), // 3rd Src register
                   "1"(c[1]));
}

__device__ void ds_read_b32(float& r, const void* lds, index_t offset = 0) { DS_READ_B32(0) }

__device__ void
ds_read_b128(vector_type<float, 4>::MemoryType& r, const void* lds, index_t offset = 0)
{
    REPEAT_STRIDEx64(DS_READ_B128, 64, 0)
}

__device__ void
ds_write_b128(const vector_type<float, 4>::MemoryType& r, const void* lds, index_t offset = 0)
{
    REPEAT_STRIDEx64(DS_WRITE_B128, 64, 0)
}

template <index_t Size>
__device__ void gcnasm_accvgpr_read(float*)
{
}

template <>
__device__ void gcnasm_accvgpr_read<16>(float* arch_reg)
{
#if CK_USE_INLINE_ASM_XDLOPS
    NOP(16)
    REPEATx16(ACCVGPR_READ, 0)
#else
    (void)arch_reg;
#endif
}

template <>
__device__ void gcnasm_accvgpr_read<32>(float* arch_reg)
{
#if CK_USE_INLINE_ASM_XDLOPS
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
#if CK_USE_INLINE_ASM_XDLOPS
    NOP(16)
    REPEATx64(ACCVGPR_READ, 0)
#else
    (void)arch_reg;
#endif
}

template <index_t MPerWave>
__device__ void gcnasm_accvgpr_zero()
{
}

template <>
__device__ void gcnasm_accvgpr_zero<32>()
{
#if CK_USE_INLINE_ASM_XDLOPS
    REPEATx16(ACCVGPR_ZERO, 0)
    REPEATx16(ACCVGPR_ZERO, 16)
#endif
}

template <>
__device__ void gcnasm_accvgpr_zero<64>()
{
#if CK_USE_INLINE_ASM_XDLOPS
    REPEATx64(ACCVGPR_ZERO, 0)
#endif
}

template <index_t MPerWave>
__device__ void gcnasm_mfma_f32_32x32x1f32(float&, float&, float32_t*)
{
}

template <>
__device__ void gcnasm_mfma_f32_32x32x1f32<64>(float& reg_a, float& reg_b, float32_t* reg_c)
{
#if CK_USE_INLINE_ASM_XDLOPS
    NOP(1)
    (void)reg_c;
    MFMA_F32_32x32x1F32(0, reg_a, reg_b, 1, 0, 0)
    MFMA_F32_32x32x1F32(32, reg_a, reg_b, 1, 1, 0)
#else
    reg_c[0] = __llvm_amdgcn_mfma_f32_32x32x1f32(reg_a, reg_b, reg_c[0], 1, 0, 0);
    reg_c[1] = __llvm_amdgcn_mfma_f32_32x32x1f32(reg_a, reg_b, reg_c[1], 1, 1, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_32x32x1f32<32>(float& reg_a, float& reg_b, float32_t* reg_c)
{
#if CK_USE_INLINE_ASM_XDLOPS
    NOP(1)
    (void)reg_c;
    MFMA_F32_32x32x1F32(0, reg_a, reg_b, 1, 0, 0)
#else
    reg_c[0] = __llvm_amdgcn_mfma_f32_32x32x1f32(reg_a, reg_b, reg_c[0], 1, 0, 0);
#endif
}

template <index_t MPerWave>
__device__ void gcnasm_mfma_f32_32x32x4f16(typename vector_type<half, 4>::MemoryType&,
                                           typename vector_type<half, 4>::MemoryType&,
                                           float32_t*)
{
}

template <>
__device__ void gcnasm_mfma_f32_32x32x4f16<64>(typename vector_type<half, 4>::MemoryType& reg_a,
                                               typename vector_type<half, 4>::MemoryType& reg_b,
                                               float32_t* reg_c)
{
#if CK_USE_INLINE_ASM_XDLOPS
    NOP(1)
    (void)reg_c;
    MFMA_F32_32x32x4F16(0, reg_a, reg_b, 1, 0, 0)
    MFMA_F32_32x32x4F16(32, reg_a, reg_b, 1, 1, 0)
#else
    reg_c[0] = __llvm_amdgcn_mfma_f32_32x32x4f16(reg_a, reg_b, reg_c[0], 1, 0, 0);
    reg_c[1] = __llvm_amdgcn_mfma_f32_32x32x4f16(reg_a, reg_b, reg_c[1], 1, 1, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_32x32x4f16<32>(typename vector_type<half, 4>::MemoryType& reg_a,
                                               typename vector_type<half, 4>::MemoryType& reg_b,
                                               float32_t* reg_c)
{
#if CK_USE_INLINE_ASM_XDLOPS
    NOP(1)
    (void)reg_c;
    MFMA_F32_32x32x4F16(0, reg_a, reg_b, 1, 0, 0)
#else
    reg_c[0] = __llvm_amdgcn_mfma_f32_32x32x4f16(reg_a, reg_b, reg_c[0], 1, 0, 0);
#endif
}

template <index_t MPerWave>
__device__ void gcnasm_mfma_f32_32x32x2bf16(typename vector_type<ushort, 2>::MemoryType&,
                                            typename vector_type<ushort, 2>::MemoryType&,
                                            float32_t*)
{
}

template <>
__device__ void gcnasm_mfma_f32_32x32x2bf16<64>(typename vector_type<ushort, 2>::MemoryType& reg_a,
                                                typename vector_type<ushort, 2>::MemoryType& reg_b,
                                                float32_t* reg_c)
{
#if CK_USE_INLINE_ASM_XDLOPS
    NOP(1)
    (void)reg_c;
    MFMA_F32_32x32x2BF16(0, reg_a, reg_b, 1, 0, 0)
    MFMA_F32_32x32x2BF16(32, reg_a, reg_b, 1, 1, 0)
#else
    reg_c[0] = __llvm_amdgcn_mfma_f32_32x32x2bf16(reg_a, reg_b, reg_c[0], 1, 0, 0);
    reg_c[1] = __llvm_amdgcn_mfma_f32_32x32x2bf16(reg_a, reg_b, reg_c[1], 1, 1, 0);
#endif
}

template <>
__device__ void gcnasm_mfma_f32_32x32x2bf16<32>(typename vector_type<ushort, 2>::MemoryType& reg_a,
                                                typename vector_type<ushort, 2>::MemoryType& reg_b,
                                                float32_t* reg_c)
{
#if CK_USE_INLINE_ASM_XDLOPS
    NOP(1)
    (void)reg_c;
    MFMA_F32_32x32x2BF16(0, reg_a, reg_b, 1, 0, 0)
#else
    reg_c[0] = __llvm_amdgcn_mfma_f32_32x32x2bf16(reg_a, reg_b, reg_c[0], 1, 0, 0);
#endif
}

// clang-format on

} // namespace ck
#endif
