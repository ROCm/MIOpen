#ifndef CK_AMD_INLINE_ASM_HPP
#define CK_AMD_INLINE_ASM_HPP

#include "vector_type.hpp"

namespace ck {

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

// clang-format on

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
    asm volatile("\n \
            v_mac_f32 %0, %2, %3 \n \
            v_mac_f32 %1, %2, %4 \n \
            "
                 : "=v"(c[0]), "=v"(c[1])
                 : "v"(a[0]), "v"(b[0]), "v"(b[1]), "0"(c[0]), "1"(c[1]));
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
            v_dot2_f32_f16 %0, %4, %6  %0\n \
            v_dot2_f32_f16 %1, %4, %8  %1\n \
            v_dot2_f32_f16 %0, %5, %7  %0\n \
            v_dot2_f32_f16 %1, %5, %9  %1\n \
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

} // namespace ck
#endif
