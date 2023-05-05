#ifndef CK_AMD_XDLOPS_INLINE_ASM_HPP
#define CK_AMD_XDLOPS_INLINE_ASM_HPP

#include "static_kernel_float_type.hpp"

namespace ck {
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
    REPEATx4(ACCVGPR_READ, 0)
}

template <>
__device__ void gcnasm_accvgpr_read<8>(float* arch_reg)
{
    REPEATx4(ACCVGPR_READ, 0)
    REPEATx4(ACCVGPR_READ, 4)
}

template <>
__device__ void gcnasm_accvgpr_read<16>(float* arch_reg)
{
    REPEATx16(ACCVGPR_READ, 0)
}

template <>
__device__ void gcnasm_accvgpr_read<32>(float* arch_reg)
{
    REPEATx16(ACCVGPR_READ, 0)
    REPEATx16(ACCVGPR_READ, 16)
}

template <>
__device__ void gcnasm_accvgpr_read<64>(float* arch_reg)
{
    REPEATx64(ACCVGPR_READ, 0)
}

template <>
__device__ void gcnasm_accvgpr_read<128>(float* arch_reg)
{
    REPEATx64(ACCVGPR_READ, 0)
    REPEATx64(ACCVGPR_READ, 64)
}

template <index_t MPerWave>
__device__ void gcnasm_accvgpr_zero();

template <>
__device__ void gcnasm_accvgpr_zero<4>()
{
    REPEATx4(ACCVGPR_ZERO, 0)
    S_NOP(1)
}

template <>
__device__ void gcnasm_accvgpr_zero<8>()
{
    REPEATx4(ACCVGPR_ZERO, 0)
    REPEATx4(ACCVGPR_ZERO, 4)
    S_NOP(1)
}

template <>
__device__ void gcnasm_accvgpr_zero<16>()
{
    REPEATx16(ACCVGPR_ZERO, 0)
    S_NOP(1)
}

template <>
__device__ void gcnasm_accvgpr_zero<32>()
{
    REPEATx16(ACCVGPR_ZERO, 0)
    REPEATx16(ACCVGPR_ZERO, 16)
    S_NOP(1)
}

template <>
__device__ void gcnasm_accvgpr_zero<64>()
{
    REPEATx64(ACCVGPR_ZERO, 0)
    S_NOP(1)
}

template <>
__device__ void gcnasm_accvgpr_zero<128>()
{
    REPEATx64(ACCVGPR_ZERO, 0)
    REPEATx64(ACCVGPR_ZERO, 64)
    S_NOP(1)
}

template <index_t Cycles>
__device__ void gcnasm_nop();

template <>
__device__ void gcnasm_nop<8>()
{
    S_NOP(3)
}

template <>
__device__ void gcnasm_nop<32>()
{
    S_NOP(9)
}

template <>
__device__ void gcnasm_nop<64>()
{
    S_NOP(8)
    S_NOP(8)
}

template <index_t MPerWave, index_t NPerWave, index_t AStride, index_t BStride>
struct gcnasm_mfma_f32_32x32x1f32;

template <index_t AStride, index_t BStride>
struct gcnasm_mfma_f32_32x32x1f32<128, 128, AStride, BStride>
{
    __device__ void run(const float* reg_a, const float* reg_b)
    {
        MFMA_F32_32x32x1F32(0,  reg_a[0],       reg_b[0], 1, 0, 0)
        MFMA_F32_32x32x1F32(32, reg_a[0],       reg_b[0], 1, 1, 0)
        MFMA_F32_32x32x1F32(64, reg_a[AStride], reg_b[0], 1, 0, 0)
        MFMA_F32_32x32x1F32(96, reg_a[AStride], reg_b[0], 1, 1, 0)

        MFMA_F32_32x32x1F32(128, reg_a[0],       reg_b[BStride], 1, 0, 0)
        MFMA_F32_32x32x1F32(160, reg_a[0],       reg_b[BStride], 1, 1, 0)
        MFMA_F32_32x32x1F32(192, reg_a[AStride], reg_b[BStride], 1, 0, 0)
        MFMA_F32_32x32x1F32(224, reg_a[AStride], reg_b[BStride], 1, 1, 0)
    }
};

template <index_t AStride, index_t BStride>
struct gcnasm_mfma_f32_32x32x1f32<128, 64, AStride, BStride>
{
    __device__ void run(const float* reg_a, const float* reg_b)
    {
        MFMA_F32_32x32x1F32(0,  reg_a[0], reg_b[0], 1, 0, 0)
        MFMA_F32_32x32x1F32(32, reg_a[0], reg_b[0], 1, 1, 0)
        MFMA_F32_32x32x1F32(64, reg_a[AStride], reg_b[0], 1, 0, 0)
        MFMA_F32_32x32x1F32(96, reg_a[AStride], reg_b[0], 1, 1, 0)
    }
};

template <index_t AStride, index_t BStride>
struct gcnasm_mfma_f32_32x32x1f32<64, 128, AStride, BStride>
{
    __device__ void run(const float* reg_a, const float* reg_b)
    {
        MFMA_F32_32x32x1F32(0,  reg_a[0], reg_b[0], 1, 0, 0)
        MFMA_F32_32x32x1F32(32, reg_a[0], reg_b[0], 1, 1, 0)
        MFMA_F32_32x32x1F32(64, reg_a[0], reg_b[BStride], 1, 0, 0)
        MFMA_F32_32x32x1F32(96, reg_a[0], reg_b[BStride], 1, 1, 0)
    }
};

template <index_t AStride, index_t BStride>
struct gcnasm_mfma_f32_32x32x1f32<64, 64, AStride, BStride>
{
    __device__ void run(const float* reg_a, const float* reg_b)
    {
        MFMA_F32_32x32x1F32(0, reg_a[0], reg_b[0], 1, 0, 0)
        MFMA_F32_32x32x1F32(32, reg_a[0], reg_b[0], 1, 1, 0)
    }
};

template <index_t AStride, index_t BStride>
struct gcnasm_mfma_f32_32x32x1f32<32, 64, AStride, BStride>
{
    __device__ void run(const float* reg_a, const float* reg_b)
    {
        MFMA_F32_32x32x1F32(0, reg_a[0], reg_b[0], 1, 0, 0)
    }
};

template <index_t AStride, index_t BStride>
struct gcnasm_mfma_f32_32x32x1f32<64, 32, AStride, BStride>
{
    __device__ void run(const float* reg_a, const float* reg_b)
    {
        MFMA_F32_32x32x1F32(0, reg_a[0], reg_b[0], 0, 0, 1)
    }
};

__device__ void gcnasm_mfma_f32_32x32x2f32(const float* reg_a, const float* reg_b)
{
    MFMA_F32_32x32x2F32(0, reg_a[0], reg_b[0], 0, 0, 0)
}

__device__ void gcnasm_mfma_f32_16x16x4f32(const float* reg_a, const float* reg_b)
{
    MFMA_F32_16x16x4F32(0, reg_a[0], reg_b[0], 0, 0, 0)
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_16x16x1f32(const float* reg_a, const float* reg_b);

template <>
__device__ void gcnasm_mfma_f32_16x16x1f32<16, 64>(const float* reg_a, const float* reg_b)
{
    MFMA_F32_16x16x1F32(0, reg_a[0], reg_b[0], 2, 0, 0)
}

template <>
__device__ void gcnasm_mfma_f32_16x16x1f32<64, 16>(const float* reg_a, const float* reg_b)
{
    MFMA_F32_16x16x1F32(0, reg_a[0], reg_b[0], 0, 0, 4)
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_4x4x1f32(const float* reg_a, const float* reg_b);

template <>
__device__ void gcnasm_mfma_f32_4x4x1f32<4, 64>(const float* reg_a, const float* reg_b)
{
    MFMA_F32_4x4x1F32(0, reg_a[0], reg_b[0], 4, 0, 0)
}

template <>
__device__ void gcnasm_mfma_f32_4x4x1f32<8, 64>(const float* reg_a, const float* reg_b)
{
    MFMA_F32_4x4x1F32(0, reg_a[0], reg_b[0], 4, 0, 0)
    MFMA_F32_4x4x1F32(4, reg_a[0], reg_b[0], 4, 1, 0)
}

template <index_t MPerWave, index_t NPerWave, index_t AStride, index_t BStride>
struct gcnasm_mfma_f32_32x32x4f16;

template <index_t AStride, index_t BStride>
struct gcnasm_mfma_f32_32x32x4f16<128, 128, AStride, BStride>
{
    __device__ void run(const half4_t* reg_a, const half4_t* reg_b)
    {
        MFMA_F32_32x32x4F16(0,  reg_a[0],       reg_b[0], 1, 0, 0)
        MFMA_F32_32x32x4F16(32, reg_a[0],       reg_b[0], 1, 1, 0)
        MFMA_F32_32x32x4F16(64, reg_a[AStride], reg_b[0], 1, 0, 0)
        MFMA_F32_32x32x4F16(96, reg_a[AStride], reg_b[0], 1, 1, 0)

        MFMA_F32_32x32x4F16(128, reg_a[0],       reg_b[BStride], 1, 0, 0)
        MFMA_F32_32x32x4F16(160, reg_a[0],       reg_b[BStride], 1, 1, 0)
        MFMA_F32_32x32x4F16(192, reg_a[AStride], reg_b[BStride], 1, 0, 0)
        MFMA_F32_32x32x4F16(224, reg_a[AStride], reg_b[BStride], 1, 1, 0)
    }
};

template <index_t AStride, index_t BStride>
struct gcnasm_mfma_f32_32x32x4f16<128, 64, AStride, BStride>
{
    __device__ void run(const half4_t* reg_a, const half4_t* reg_b)
    {
        MFMA_F32_32x32x4F16(0,  reg_a[0],       reg_b[0], 1, 0, 0)
        MFMA_F32_32x32x4F16(32, reg_a[0],       reg_b[0], 1, 1, 0)
        MFMA_F32_32x32x4F16(64, reg_a[AStride], reg_b[0], 1, 0, 0)
        MFMA_F32_32x32x4F16(96, reg_a[AStride], reg_b[0], 1, 1, 0)
    }
};

template <index_t AStride, index_t BStride>
struct gcnasm_mfma_f32_32x32x4f16<64, 128, AStride, BStride>
{
    __device__ void run(const half4_t* reg_a, const half4_t* reg_b)
    {
        MFMA_F32_32x32x4F16(0,  reg_a[0], reg_b[0], 1, 0, 0)
        MFMA_F32_32x32x4F16(32, reg_a[0], reg_b[0], 1, 1, 0)
        MFMA_F32_32x32x4F16(64, reg_a[0], reg_b[BStride], 1, 0, 0)
        MFMA_F32_32x32x4F16(96, reg_a[0], reg_b[BStride], 1, 1, 0)
    }
};

template <index_t AStride, index_t BStride>
struct gcnasm_mfma_f32_32x32x4f16<64, 64, AStride, BStride>
{
    __device__ void run(const half4_t* reg_a, const half4_t* reg_b)
    {
        MFMA_F32_32x32x4F16(0, reg_a[0], reg_b[0], 1, 0, 0)
        MFMA_F32_32x32x4F16(32, reg_a[0], reg_b[0], 1, 1, 0)
    }
};

template <index_t AStride, index_t BStride>
struct gcnasm_mfma_f32_32x32x4f16<32, 64, AStride, BStride>
{
    __device__ void run(const half4_t* reg_a, const half4_t* reg_b)
    {
        MFMA_F32_32x32x4F16(0, reg_a[0], reg_b[0], 1, 0, 0)
    }
};

template <index_t AStride, index_t BStride>
struct gcnasm_mfma_f32_32x32x4f16<64, 32, AStride, BStride>
{
    __device__ void run(const half4_t* reg_a, const half4_t* reg_b)
    {
        MFMA_F32_32x32x4F16(0, reg_a[0], reg_b[0], 0, 0, 1)
    }
};

__device__ void gcnasm_mfma_f32_32x32x8f16(const half4_t* reg_a, const half4_t* reg_b)
{
    MFMA_F32_32x32x8F16(0, reg_a[0], reg_b[0], 0, 0, 0)
}

__device__ void gcnasm_mfma_f32_16x16x16f16(const half4_t* reg_a, const half4_t* reg_b)
{
    MFMA_F32_16x16x16F16(0, reg_a[0], reg_b[0], 0, 0, 0)
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_16x16x4f16(const half4_t* reg_a, const half4_t* reg_b);

template <>
__device__ void gcnasm_mfma_f32_16x16x4f16<16, 64>(const half4_t* reg_a, const half4_t* reg_b)
{
    MFMA_F32_16x16x4F16(0, reg_a[0], reg_b[0], 2, 0, 0)
}

template <>
__device__ void gcnasm_mfma_f32_16x16x4f16<64, 16>(const half4_t* reg_a, const half4_t* reg_b)

{
    MFMA_F32_16x16x4F16(0, reg_a[0], reg_b[0], 0, 0, 4)
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_4x4x4f16(const half4_t* reg_a, const half4_t* reg_b);

template <>
__device__ void gcnasm_mfma_f32_4x4x4f16<4, 64>(const half4_t* reg_a, const half4_t* reg_b)
{
    MFMA_F32_4x4x4F16(0, reg_a[0], reg_b[0], 4, 0, 0)
}

template <>
__device__ void gcnasm_mfma_f32_4x4x4f16<8, 64>(const half4_t* reg_a, const half4_t* reg_b)
{
    MFMA_F32_4x4x4F16(0, reg_a[0], reg_b[0], 4, 0, 0)
    MFMA_F32_4x4x4F16(4, reg_a[0], reg_b[0], 4, 1, 0)
}

template <index_t MPerWave, index_t NPerWave, index_t AStride, index_t BStride>
struct gcnasm_mfma_f32_32x32x2bf16;

template <index_t AStride, index_t BStride>
struct gcnasm_mfma_f32_32x32x2bf16<128, 128, AStride, BStride>
{
    __device__ void run(const ushort2_t* reg_a, const ushort2_t* reg_b)
    {
        MFMA_F32_32x32x2BF16(0,  reg_a[0],       reg_b[0], 1, 0, 0)
        MFMA_F32_32x32x2BF16(32, reg_a[0],       reg_b[0], 1, 1, 0)
        MFMA_F32_32x32x2BF16(64, reg_a[AStride], reg_b[0], 1, 0, 0)
        MFMA_F32_32x32x2BF16(96, reg_a[AStride], reg_b[0], 1, 1, 0)

        MFMA_F32_32x32x2BF16(128, reg_a[0],       reg_b[BStride], 1, 0, 0)
        MFMA_F32_32x32x2BF16(160, reg_a[0],       reg_b[BStride], 1, 1, 0)
        MFMA_F32_32x32x2BF16(192, reg_a[AStride], reg_b[BStride], 1, 0, 0)
        MFMA_F32_32x32x2BF16(224, reg_a[AStride], reg_b[BStride], 1, 1, 0)
    }
};

template <index_t AStride, index_t BStride>
struct gcnasm_mfma_f32_32x32x2bf16<128, 64, AStride, BStride>
{
    __device__ void run(const ushort2_t* reg_a, const ushort2_t* reg_b)
    {
        MFMA_F32_32x32x2BF16(0,  reg_a[0],       reg_b[0], 1, 0, 0)
        MFMA_F32_32x32x2BF16(32, reg_a[0],       reg_b[0], 1, 1, 0)
        MFMA_F32_32x32x2BF16(64, reg_a[AStride], reg_b[0], 1, 0, 0)
        MFMA_F32_32x32x2BF16(96, reg_a[AStride], reg_b[0], 1, 1, 0)
    }
};

template <index_t AStride, index_t BStride>
struct gcnasm_mfma_f32_32x32x2bf16<64, 128, AStride, BStride>
{
    __device__ void run(const ushort2_t* reg_a, const ushort2_t* reg_b)
    {
        MFMA_F32_32x32x2BF16(0,  reg_a[0], reg_b[0], 1, 0, 0)
        MFMA_F32_32x32x2BF16(32, reg_a[0], reg_b[0], 1, 1, 0)
        MFMA_F32_32x32x2BF16(64, reg_a[0], reg_b[BStride], 1, 0, 0)
        MFMA_F32_32x32x2BF16(96, reg_a[0], reg_b[BStride], 1, 1, 0)
    }
};

template <index_t AStride, index_t BStride>
struct gcnasm_mfma_f32_32x32x2bf16<64, 64, AStride, BStride>
{
    __device__ void run(const ushort2_t* reg_a, const ushort2_t* reg_b)
    {
        MFMA_F32_32x32x2BF16(0, reg_a[0], reg_b[0], 1, 0, 0)
        MFMA_F32_32x32x2BF16(32, reg_a[0], reg_b[0], 1, 1, 0)
    }
};

template <index_t AStride, index_t BStride>
struct gcnasm_mfma_f32_32x32x2bf16<32, 64, AStride, BStride>
{
    __device__ void run(const ushort2_t* reg_a, const ushort2_t* reg_b)
    {
        MFMA_F32_32x32x2BF16(0, reg_a[0], reg_b[0], 1, 0, 0)
    }
};

template <index_t AStride, index_t BStride>
struct gcnasm_mfma_f32_32x32x2bf16<64, 32, AStride, BStride>
{
    __device__ void run(const ushort2_t* reg_a, const ushort2_t* reg_b)
    {
        MFMA_F32_32x32x2BF16(0, reg_a[0], reg_b[0], 0, 0, 1)
    }
};

__device__ void gcnasm_mfma_f32_32x32x4bf16(const ushort2_t* reg_a, const ushort2_t* reg_b)
{
    MFMA_F32_32x32x4BF16(0, reg_a[0], reg_b[0], 0, 0, 0)
}

__device__ void gcnasm_mfma_f32_16x16x8bf16(const ushort2_t* reg_a, const ushort2_t* reg_b)
{
    MFMA_F32_16x16x8BF16(0, reg_a[0], reg_b[0], 0, 0, 0)
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_16x16x2bf16(const ushort2_t* reg_a, const ushort2_t* reg_b);

template <>
__device__ void gcnasm_mfma_f32_16x16x2bf16<16, 64>(const ushort2_t* reg_a, const ushort2_t* reg_b)
{
    MFMA_F32_16x16x2BF16(0, reg_a[0], reg_b[0], 2, 0, 0)
}

template <>
__device__ void gcnasm_mfma_f32_16x16x2bf16<64, 16>(const ushort2_t* reg_a, const ushort2_t* reg_b)
{
    MFMA_F32_16x16x2BF16(0, reg_a[0], reg_b[0], 0, 0, 4)
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_4x4x2bf16(const ushort2_t* reg_a, const ushort2_t* reg_b);

template <>
__device__ void gcnasm_mfma_f32_4x4x2bf16<4, 64>(const ushort2_t* reg_a, const ushort2_t* reg_b)
{
    MFMA_F32_4x4x2BF16(0, reg_a[0], reg_b[0], 4, 0, 0)
}

template <>
__device__ void gcnasm_mfma_f32_4x4x2bf16<8, 64>(const ushort2_t* reg_a, const ushort2_t* reg_b)
{
    MFMA_F32_4x4x2BF16(0, reg_a[0], reg_b[0], 4, 0, 0)
    MFMA_F32_4x4x2BF16(4, reg_a[0], reg_b[0], 4, 1, 0)
}
// clang-format on
} // namespace ck
#endif
