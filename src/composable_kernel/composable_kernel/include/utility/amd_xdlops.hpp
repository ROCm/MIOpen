#ifndef CK_AMD_XDLOPS_HPP
#define CK_AMD_XDLOPS_HPP

#include "data_type.hpp"

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

template <index_t MPerWave, index_t NPerWave, index_t COffset>
struct intrin_mfma_f32_32x32x1f32;

template <index_t COffset>
struct intrin_mfma_f32_32x32x1f32<64, 64, COffset>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c(Number<COffset>{}).template AsType<float32_t>()(Number<0>{}) =
            llvm_intrin_amdgcn_mfma_f32_32x32x1f32(
                reg_a,
                reg_b,
                reg_c[Number<COffset>{}].template AsType<float32_t>()[Number<0>{}],
                1,
                0,
                0);
        reg_c(Number<COffset + 1>{}).template AsType<float32_t>()(Number<0>{}) =
            llvm_intrin_amdgcn_mfma_f32_32x32x1f32(
                reg_a,
                reg_b,
                reg_c[Number<COffset + 1>{}].template AsType<float32_t>()[Number<0>{}],
                1,
                1,
                0);
    }
};

template <index_t COffset>
struct intrin_mfma_f32_32x32x1f32<32, 64, COffset>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c(Number<COffset>{}).template AsType<float32_t>()(Number<0>{}) =
            llvm_intrin_amdgcn_mfma_f32_32x32x1f32(
                reg_a,
                reg_b,
                reg_c[Number<COffset>{}].template AsType<float32_t>()[Number<0>{}],
                1,
                0,
                0);
    }
};

template <index_t MPerWave, index_t NPerWave, index_t COffset>
struct intrin_mfma_f32_32x32x2f32;

template <index_t COffset>
struct intrin_mfma_f32_32x32x2f32<32, 32, COffset>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c(Number<COffset>{}).template AsType<float16_t>()(Number<0>{}) =
            llvm_intrin_amdgcn_mfma_f32_32x32x2f32(
                reg_a,
                reg_b,
                reg_c[Number<COffset>{}].template AsType<float16_t>()[Number<0>{}],
                0,
                0,
                0);
    }
};

template <index_t MPerWave, index_t NPerWave, index_t COffset>
struct intrin_mfma_f32_16x16x4f32;

template <index_t COffset>
struct intrin_mfma_f32_16x16x4f32<16, 16, COffset>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c(Number<COffset>{}).template AsType<float4_t>()(Number<0>{}) =
            llvm_intrin_amdgcn_mfma_f32_16x16x4f32(
                reg_a,
                reg_b,
                reg_c[Number<COffset>{}].template AsType<float4_t>()[Number<0>{}],
                0,
                0,
                0);
    }
};

template <index_t MPerWave, index_t NPerWave, index_t COffset>
struct intrin_mfma_f32_16x16x1f32;

template <index_t COffset>
struct intrin_mfma_f32_16x16x1f32<16, 64, COffset>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {

        reg_c(Number<COffset>{}).template AsType<float16_t>()(Number<0>{}) =
            llvm_intrin_amdgcn_mfma_f32_16x16x1f32(
                reg_a,
                reg_b,
                reg_c[Number<COffset>{}].template AsType<float16_t>()[Number<0>{}],
                2,
                0,
                0);
    }
};

template <index_t MPerWave, index_t NPerWave, index_t COffset>
struct intrin_mfma_f32_4x4x1f32;

template <index_t COffset>
struct intrin_mfma_f32_4x4x1f32<4, 64, COffset>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c(Number<COffset>{}).template AsType<float4_t>()(Number<0>{}) =
            llvm_intrin_amdgcn_mfma_f32_4x4x1f32(
                reg_a,
                reg_b,
                reg_c[Number<COffset>{}].template AsType<float4_t>()[Number<0>{}],
                4,
                0,
                0);
    }
};

template <index_t COffset>
struct intrin_mfma_f32_4x4x1f32<8, 64, COffset>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c(Number<COffset>{}).template AsType<float4_t>()(Number<0>{}) =
            llvm_intrin_amdgcn_mfma_f32_4x4x1f32(
                reg_a,
                reg_b,
                reg_c[Number<COffset>{}].template AsType<float4_t>()[Number<0>{}],
                4,
                0,
                0);
        reg_c(Number<COffset + 1>{}).template AsType<float4_t>()(Number<0>{}) =
            llvm_intrin_amdgcn_mfma_f32_4x4x1f32(
                reg_a,
                reg_b,
                reg_c[Number<COffset + 1>{}].template AsType<float4_t>()[Number<0>{}],
                4,
                1,
                0);
    }
};

template <index_t MPerWave, index_t NPerWave, index_t COffset>
struct intrin_mfma_f32_32x32x4f16;

template <index_t COffset>
struct intrin_mfma_f32_32x32x4f16<64, 64, COffset>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c(Number<COffset>{}).template AsType<float32_t>()(Number<0>{}) =
            llvm_intrin_amdgcn_mfma_f32_32x32x4f16(
                reg_a,
                reg_b,
                reg_c[Number<COffset>{}].template AsType<float32_t>()[Number<0>{}],
                1,
                0,
                0);
        reg_c(Number<COffset + 1>{}).template AsType<float32_t>()(Number<0>{}) =
            llvm_intrin_amdgcn_mfma_f32_32x32x4f16(
                reg_a,
                reg_b,
                reg_c[Number<COffset + 1>{}].template AsType<float32_t>()[Number<0>{}],
                1,
                1,
                0);
    }
};

template <index_t COffset>
struct intrin_mfma_f32_32x32x4f16<32, 64, COffset>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c(Number<COffset>{}).template AsType<float32_t>()(Number<0>{}) =
            llvm_intrin_amdgcn_mfma_f32_32x32x4f16(
                reg_a,
                reg_b,
                reg_c[Number<COffset>{}].template AsType<float32_t>()[Number<0>{}],
                1,
                0,
                0);
    }
};

template <index_t MPerWave, index_t NPerWave, index_t COffset>
struct intrin_mfma_f32_32x32x8f16;

template <index_t COffset>
struct intrin_mfma_f32_32x32x8f16<32, 32, COffset>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c(Number<COffset>{}).template AsType<float16_t>()(Number<0>{}) =
            llvm_intrin_amdgcn_mfma_f32_32x32x8f16(
                reg_a,
                reg_b,
                reg_c[Number<COffset>{}].template AsType<float16_t>()[Number<0>{}],
                0,
                0,
                0);
    }
};

template <index_t MPerWave, index_t NPerWave, index_t COffset>
struct intrin_mfma_f32_16x16x16f16;

template <index_t COffset>
struct intrin_mfma_f32_16x16x16f16<16, 16, COffset>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c(Number<COffset>{}).template AsType<float4_t>()(Number<0>{}) =
            llvm_intrin_amdgcn_mfma_f32_16x16x16f16(
                reg_a,
                reg_b,
                reg_c[Number<COffset>{}].template AsType<float4_t>()[Number<0>{}],
                0,
                0,
                0);
    }
};

template <index_t MPerWave, index_t NPerWave, index_t COffset>
struct intrin_mfma_f32_16x16x4f16;

template <index_t COffset>
struct intrin_mfma_f32_16x16x4f16<16, 64, COffset>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c(Number<COffset>{}).template AsType<float16_t>()(Number<0>{}) =
            llvm_intrin_amdgcn_mfma_f32_16x16x4f16(
                reg_a,
                reg_b,
                reg_c[Number<COffset>{}].template AsType<float16_t>()[Number<0>{}],
                2,
                0,
                0);
    }
};

template <index_t MPerWave, index_t NPerWave, index_t COffset>
struct intrin_mfma_f32_4x4x4f16;

template <index_t COffset>
struct intrin_mfma_f32_4x4x4f16<4, 64, COffset>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c(Number<COffset>{}).template AsType<float4_t>()(Number<0>{}) =
            llvm_intrin_amdgcn_mfma_f32_4x4x4f16(
                reg_a,
                reg_b,
                reg_c[Number<COffset>{}].template AsType<float4_t>()[Number<0>{}],
                4,
                0,
                0);
    }
};

template <index_t COffset>
struct intrin_mfma_f32_4x4x4f16<8, 64, COffset>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c(Number<COffset>{}).template AsType<float4_t>()(Number<0>{}) =
            llvm_intrin_amdgcn_mfma_f32_4x4x4f16(
                reg_a,
                reg_b,
                reg_c[Number<COffset>{}].template AsType<float4_t>()[Number<0>{}],
                4,
                0,
                0);
        reg_c(Number<COffset + 1>{}).template AsType<float4_t>()(Number<0>{}) =
            llvm_intrin_amdgcn_mfma_f32_4x4x4f16(
                reg_a,
                reg_b,
                reg_c[Number<COffset + 1>{}].template AsType<float4_t>()[Number<0>{}],
                4,
                1,
                0);
    }
};

#if 0
template <index_t MPerWave, index_t NPerWave, index_t AStride, index_t BStride>
struct intrin_mfma_f32_32x32x2bf16;

template <index_t AStride, index_t BStride>
struct intrin_mfma_f32_32x32x2bf16<128, 64, AStride, BStride>
{
    __device__ static c_vec32_4_t::VecType
    run(const ushort2_t* reg_a, const ushort2_t* reg_b, c_vec32_4_t::VecType reg_c)
    {
        reg_c.s.x = llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a[0], reg_b[0], reg_c.s.x, 1, 0, 0);
        reg_c.s.y = llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a[0], reg_b[0], reg_c.s.y, 1, 1, 0);

        reg_c.s.z =
            llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a[AStride], reg_b[0], reg_c.s.z, 1, 0, 0);
        reg_c.s.w =
            llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a[AStride], reg_b[0], reg_c.s.w, 1, 1, 0);

        return reg_c;
    }
};

template <index_t AStride, index_t BStride>
struct intrin_mfma_f32_32x32x2bf16<64, 128, AStride, BStride>
{
    __device__ static c_vec32_4_t::VecType
    run(const ushort2_t* reg_a, const ushort2_t* reg_b, c_vec32_4_t::VecType reg_c)
    {
        reg_c.s.x = llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a[0], reg_b[0], reg_c.s.x, 1, 0, 0);
        reg_c.s.y = llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a[0], reg_b[0], reg_c.s.y, 1, 1, 0);

        reg_c.s.z =
            llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a[0], reg_b[BStride], reg_c.s.z, 1, 0, 0);
        reg_c.s.w =
            llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a[0], reg_b[BStride], reg_c.s.w, 1, 1, 0);

        return reg_c;
    }
};

template <index_t AStride, index_t BStride>
struct intrin_mfma_f32_32x32x2bf16<64, 64, AStride, BStride>
{
    __device__ static c_vec32_2_t::VecType
    run(const ushort2_t* reg_a, const ushort2_t* reg_b, c_vec32_2_t::VecType reg_c)
    {
        reg_c.s.x = llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a[0], reg_b[0], reg_c.s.x, 1, 0, 0);
        reg_c.s.y = llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a[0], reg_b[0], reg_c.s.y, 1, 1, 0);

        return reg_c;
    }
};

template <index_t AStride, index_t BStride>
struct intrin_mfma_f32_32x32x2bf16<64, 32, AStride, BStride>
{
    __device__ static c_vec32_1_t::VecType
    run(const ushort2_t* reg_a, const ushort2_t* reg_b, c_vec32_1_t::VecType reg_c)
    {
        reg_c.s.x = llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a[0], reg_b[0], reg_c.s.x, 0, 0, 1);

        return reg_c;
    }
};

template <index_t AStride, index_t BStride>
struct intrin_mfma_f32_32x32x2bf16<32, 64, AStride, BStride>
{
    __device__ static c_vec32_1_t::VecType
    run(const ushort2_t* reg_a, const ushort2_t* reg_b, c_vec32_1_t::VecType reg_c)
    {
        reg_c.s.x = llvm_intrin_amdgcn_mfma_f32_32x32x2bf16(reg_a[0], reg_b[0], reg_c.s.x, 1, 0, 0);
        return reg_c;
    }
};

__device__ c_vec16_1_t::VecType intrin_mfma_f32_32x32x4bf16(const ushort2_t* reg_a,
                                                            const ushort2_t* reg_b,
                                                            c_vec16_1_t::VecType reg_c)
{
    reg_c.s.x = llvm_intrin_amdgcn_mfma_f32_32x32x4bf16(reg_a[0], reg_b[0], reg_c.s.x, 0, 0, 0);
    return reg_c;
}

__device__ c_vec4_1_t::VecType intrin_mfma_f32_16x16x8bf16(const ushort2_t* reg_a,
                                                           const ushort2_t* reg_b,
                                                           c_vec4_1_t::VecType reg_c)
{
    reg_c.s.x = llvm_intrin_amdgcn_mfma_f32_16x16x8bf16(reg_a[0], reg_b[0], reg_c.s.x, 0, 0, 0);
    return reg_c;
}

template <index_t MPerWave, index_t NPerWave>
__device__ c_vec16_1_t::VecType intrin_mfma_f32_16x16x2bf16(const ushort2_t* reg_a,
                                                            const ushort2_t* reg_b,
                                                            c_vec16_1_t::VecType reg_c);

template <>
__device__ c_vec16_1_t::VecType intrin_mfma_f32_16x16x2bf16<16, 64>(const ushort2_t* reg_a,
                                                                    const ushort2_t* reg_b,
                                                                    c_vec16_1_t::VecType reg_c)
{
    reg_c.s.x = llvm_intrin_amdgcn_mfma_f32_16x16x2bf16(reg_a[0], reg_b[0], reg_c.s.x, 2, 0, 0);
    return reg_c;
}

template <>
__device__ c_vec16_1_t::VecType intrin_mfma_f32_16x16x2bf16<64, 16>(const ushort2_t* reg_a,
                                                                    const ushort2_t* reg_b,
                                                                    c_vec16_1_t::VecType reg_c)
{
    reg_c.s.x = llvm_intrin_amdgcn_mfma_f32_16x16x2bf16(reg_a[0], reg_b[0], reg_c.s.x, 0, 0, 4);
    return reg_c;
}

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_4x4x2bf16;

template <>
struct intrin_mfma_f32_4x4x2bf16<4, 64>
{
    __device__ static c_vec4_1_t::VecType
    run(const ushort2_t* reg_a, const ushort2_t* reg_b, c_vec4_1_t::VecType reg_c)
    {
        reg_c.s.x = llvm_intrin_amdgcn_mfma_f32_4x4x2bf16(reg_a[0], reg_b[0], reg_c.s.x, 4, 0, 0);
        return reg_c;
    }
};

template <>
struct intrin_mfma_f32_4x4x2bf16<8, 64>
{
    __device__ static c_vec4_2_t::VecType
    run(const ushort2_t* reg_a, const ushort2_t* reg_b, c_vec4_2_t::VecType reg_c)
    {
        reg_c.s.x = llvm_intrin_amdgcn_mfma_f32_4x4x2bf16(reg_a[0], reg_b[0], reg_c.s.x, 4, 0, 0);
        reg_c.s.y = llvm_intrin_amdgcn_mfma_f32_4x4x2bf16(reg_a[0], reg_b[0], reg_c.s.y, 4, 1, 0);
        return reg_c;
    }
};

#endif

} // namespace ck
#endif
