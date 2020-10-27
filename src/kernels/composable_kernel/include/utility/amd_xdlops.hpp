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

template <index_t MPerWave, index_t NPerWave>
__device__ void
intrin_mfma_f32_32x32x1f32(const float* reg_a, const float* reg_b, float32_t* reg_c);

template <>
__device__ void
intrin_mfma_f32_32x32x1f32<64, 64>(const float* reg_a, const float* reg_b, float32_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(reg_a[0], reg_b[0], reg_c[0], 1, 0, 0);
    reg_c[1] = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(reg_a[0], reg_b[0], reg_c[1], 1, 1, 0);
}

template <>
__device__ void
intrin_mfma_f32_32x32x1f32<32, 64>(const float* reg_a, const float* reg_b, float32_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(reg_a[0], reg_b[0], reg_c[0], 1, 0, 0);
}

template <>
__device__ void
intrin_mfma_f32_32x32x1f32<64, 32>(const float* reg_a, const float* reg_b, float32_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(reg_a[0], reg_b[0], reg_c[0], 0, 0, 1);
}

__device__ void intrin_mfma_f32_32x32x2f32(const float* reg_a, const float* reg_b, float16_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x2f32(reg_a[0], reg_b[0], reg_c[0], 0, 0, 0);
}

__device__ void intrin_mfma_f32_16x16x4f32(const float* reg_a, const float* reg_b, float4_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x4f32(reg_a[0], reg_b[0], reg_c[0], 0, 0, 0);
}

template <index_t MPerWave, index_t NPerWave>
__device__ void
intrin_mfma_f32_16x16x1f32(const float* reg_a, const float* reg_b, float16_t* reg_c);

template <>
__device__ void
intrin_mfma_f32_16x16x1f32<16, 64>(const float* reg_a, const float* reg_b, float16_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x1f32(reg_a[0], reg_b[0], reg_c[0], 2, 0, 0);
}

template <>
__device__ void
intrin_mfma_f32_16x16x1f32<64, 16>(const float* reg_a, const float* reg_b, float16_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x1f32(reg_a[0], reg_b[0], reg_c[0], 0, 0, 4);
}

template <index_t MPerWave, index_t NPerWave>
__device__ void intrin_mfma_f32_4x4x1f32(const float* reg_a, const float* reg_b, float4_t* reg_c);

template <>
__device__ void
intrin_mfma_f32_4x4x1f32<4, 64>(const float* reg_a, const float* reg_b, float4_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_4x4x1f32(reg_a[0], reg_b[0], reg_c[0], 4, 0, 0);
}

template <>
__device__ void
intrin_mfma_f32_4x4x1f32<8, 64>(const float* reg_a, const float* reg_b, float4_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_4x4x1f32(reg_a[0], reg_b[0], reg_c[0], 4, 0, 0);
    reg_c[1] = llvm_intrin_amdgcn_mfma_f32_4x4x1f32(reg_a[0], reg_b[0], reg_c[1], 4, 1, 0);
}

template <index_t MPerWave, index_t NPerWave>
__device__ void
intrin_mfma_f32_32x32x4f16(const half4_t* reg_a, const half4_t* reg_b, float32_t* reg_c);
template <>
__device__ void
intrin_mfma_f32_32x32x4f16<64, 64>(const half4_t* reg_a, const half4_t* reg_b, float32_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x4f16(reg_a[0], reg_b[0], reg_c[0], 1, 0, 0);
    reg_c[1] = llvm_intrin_amdgcn_mfma_f32_32x32x4f16(reg_a[0], reg_b[0], reg_c[1], 1, 1, 0);
}

template <>
__device__ void
intrin_mfma_f32_32x32x4f16<32, 64>(const half4_t* reg_a, const half4_t* reg_b, float32_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x4f16(reg_a[0], reg_b[0], reg_c[0], 1, 0, 0);
}

template <>
__device__ void
intrin_mfma_f32_32x32x4f16<64, 32>(const half4_t* reg_a, const half4_t* reg_b, float32_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x4f16(reg_a[0], reg_b[0], reg_c[0], 0, 0, 1);
}

__device__ void
intrin_mfma_f32_32x32x8f16(const half4_t* reg_a, const half4_t* reg_b, float16_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x8f16(reg_a[0], reg_b[0], reg_c[0], 0, 0, 0);
}

__device__ void
intrin_mfma_f32_16x16x16f16(const half4_t* reg_a, const half4_t* reg_b, float4_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x16f16(reg_a[0], reg_b[0], reg_c[0], 0, 0, 0);
}

template <index_t MPerWave, index_t NPerWave>
__device__ void
intrin_mfma_f32_16x16x4f16(const half4_t* reg_a, const half4_t* reg_b, float16_t* reg_c);

template <>
__device__ void
intrin_mfma_f32_16x16x4f16<16, 64>(const half4_t* reg_a, const half4_t* reg_b, float16_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x4f16(reg_a[0], reg_b[0], reg_c[0], 2, 0, 0);
}

template <>
__device__ void
intrin_mfma_f32_16x16x4f16<64, 16>(const half4_t* reg_a, const half4_t* reg_b, float16_t* reg_c)

{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x4f16(reg_a[0], reg_b[0], reg_c[0], 0, 0, 4);
}

template <index_t MPerWave, index_t NPerWave>
__device__ void
intrin_mfma_f32_4x4x4f16(const half4_t* reg_a, const half4_t* reg_b, float4_t* reg_c);

template <>
__device__ void
intrin_mfma_f32_4x4x4f16<4, 64>(const half4_t* reg_a, const half4_t* reg_b, float4_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_4x4x4f16(reg_a[0], reg_b[0], reg_c[0], 4, 0, 0);
}

template <>
__device__ void
intrin_mfma_f32_4x4x4f16<8, 64>(const half4_t* reg_a, const half4_t* reg_b, float4_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_4x4x4f16(reg_a[0], reg_b[0], reg_c[0], 4, 0, 0);
    reg_c[1] = llvm_intrin_amdgcn_mfma_f32_4x4x4f16(reg_a[0], reg_b[0], reg_c[1], 4, 1, 0);
}

template <index_t MPerWave, index_t NPerWave>
__device__ void
intrin_mfma_f32_32x32x2bf16(const ushort2_t* reg_a, const ushort2_t* reg_b, float32_t* reg_c);

template <>
__device__ void intrin_mfma_f32_32x32x2bf16<64, 64>(const ushort2_t* reg_a,
                                                    const ushort2_t* reg_b,
                                                    float32_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a[0], reg_b[0], reg_c[0], 1, 0, 0);
    reg_c[1] = llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a[0], reg_b[0], reg_c[1], 1, 1, 0);
}

template <>
__device__ void intrin_mfma_f32_32x32x2bf16<32, 64>(const ushort2_t* reg_a,
                                                    const ushort2_t* reg_b,
                                                    float32_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a[0], reg_b[0], reg_c[0], 1, 0, 0);
}

template <>
__device__ void intrin_mfma_f32_32x32x2bf16<64, 32>(const ushort2_t* reg_a,
                                                    const ushort2_t* reg_b,
                                                    float32_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a[0], reg_b[0], reg_c[0], 0, 0, 1);
}

__device__ void
intrin_mfma_f32_32x32x4bf16(const ushort2_t* reg_a, const ushort2_t* reg_b, float16_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x4bf16(reg_a[0], reg_b[0], reg_c[0], 0, 0, 0);
}

__device__ void
intrin_mfma_f32_16x16x8bf16(const ushort2_t* reg_a, const ushort2_t* reg_b, float4_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x8bf16(reg_a[0], reg_b[0], reg_c[0], 0, 0, 0);
}

template <index_t MPerWave, index_t NPerWave>
__device__ void
intrin_mfma_f32_16x16x2bf16(const ushort2_t* reg_a, const ushort2_t* reg_b, float16_t* reg_c);

template <>
__device__ void intrin_mfma_f32_16x16x2bf16<16, 64>(const ushort2_t* reg_a,
                                                    const ushort2_t* reg_b,
                                                    float16_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x2bf16(reg_a[0], reg_b[0], reg_c[0], 2, 0, 0);
}

template <>
__device__ void intrin_mfma_f32_16x16x2bf16<64, 16>(const ushort2_t* reg_a,
                                                    const ushort2_t* reg_b,
                                                    float16_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x2bf16(reg_a[0], reg_b[0], reg_c[0], 0, 0, 4);
}

template <index_t MPerWave, index_t NPerWave>
__device__ void
intrin_mfma_f32_4x4x2bf16(const ushort2_t* reg_a, const ushort2_t* reg_b, float4_t* reg_c);

template <>
__device__ void
intrin_mfma_f32_4x4x2bf16<4, 64>(const ushort2_t* reg_a, const ushort2_t* reg_b, float4_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_4x4x2bf16(reg_a[0], reg_b[0], reg_c[0], 4, 0, 0);
}

template <>
__device__ void
intrin_mfma_f32_4x4x2bf16<8, 64>(const ushort2_t* reg_a, const ushort2_t* reg_b, float4_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_4x4x2bf16(reg_a[0], reg_b[0], reg_c[0], 4, 0, 0);
    reg_c[1] = llvm_intrin_amdgcn_mfma_f32_4x4x2bf16(reg_a[0], reg_b[0], reg_c[1], 4, 1, 0);
}
}
#endif
