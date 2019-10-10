#ifndef CK_AMD_INLINE_ASM_HPP
#define CK_AMD_INLINE_ASM_HPP

#include "vector_type.hpp"

namespace ck {

// outer-product: c[i,j] += inner_product(a[i], b[j])
__device__ void __outer_product_1x2(float a, float b0, float b1, float& c0, float& c1)
{
// disable inline asm due to the compiler issue: SWDEV-202749
///\to-do: enable the inline asm after the compiler fix
#if CK_WORKAROUND_SWDEV_202749
    c0 += a * b0;
    c1 += a * b1;
#else
    asm volatile("\n \
            v_mac_f32 %0, %2, %3 \n \
            v_mac_f32 %1, %2, %4 \n \
            "
                 : "=v"(c0), "=v"(c1)
                 : "v"(a), "v"(b0), "v"(b1), "0"(c0), "1"(c1));
#endif
}

// outer-product: c[i,j] += inner_product(a[i], b[j])
__device__ void __outer_product_1x4(
    float a, float b0, float b1, float b2, float b3, float& c0, float& c1, float& c2, float& c3)
{
    asm volatile("\n \
            v_mac_f32 %0, %4, %5 \n \
            v_mac_f32 %1, %4, %6 \n \
            v_mac_f32 %2, %4, %7 \n \
            v_mac_f32 %3, %4, %8 \n \
            "
                 : "=v"(c0), "=v"(c1), "=v"(c2), "=v"(c3)
                 : "v"(a), "v"(b0), "v"(b1), "v"(b2), "v"(b3), "0"(c0), "1"(c1), "2"(c2), "3"(c3));
}

// outer-product: c[i,j] += inner_product(a[i], b[j])
__device__ void __outer_product_1x2(half2_t a, half2_t b0, half2_t b1, float& c0, float& c1)
{
    asm volatile("\n \
            v_dot2_f32_f16 %0, %2, %3  %0\n \
            v_dot2_f32_f16 %1, %2, %4  %1\n \
            "
                 : "=v"(c0), "=v"(c1) // Dest registers
                 : "v"(a),            // 1st Src register for 1 half2 registers
                   "v"(b0),           // 2nd Src register
                   "v"(b1),
                   "0"(c0), // 3rd Src register
                   "1"(c1));
}

// outer-product: c[i,j] += inner_product(a[i], b[j])
__device__ void __outer_product_1x2(half4_t a, half4_t b0, half4_t b1, float& c0, float& c1)
{
    const half2_t* p_a_half2  = reinterpret_cast<const half2_t*>(&a);
    const half2_t* p_b0_half2 = reinterpret_cast<const half2_t*>(&b0);
    const half2_t* p_b1_half2 = reinterpret_cast<const half2_t*>(&b1);

    // do dot2 two times
    asm volatile("\n \
            v_dot2_f32_f16 %0, %2, %4  %0\n \
            v_dot2_f32_f16 %1, %2, %6  %1\n \
            v_dot2_f32_f16 %0, %3, %5  %0\n \
            v_dot2_f32_f16 %1, %3, %7  %1\n \
            "
                 : "=v"(c0), "=v"(c1) // Dest registers
                 : "v"(p_a_half2[0]),
                   "v"(p_a_half2[1]), // 1st Src registers for 2 half2 registers
                   "v"(p_b0_half2[0]),
                   "v"(p_b0_half2[1]),
                   "v"(p_b1_half2[0]),
                   "v"(p_b1_half2[1]), // 2nd Src registers for 2 half2 registers
                   "0"(c0),
                   "1"(c1)); // 3rd Src Acc registers for 2 half2 registers
}

// outer-product: c[i,j] += inner_product(a[i], b[j])
__device__ void __outer_product_1x4(half2_t a,
                                    half2_t b0,
                                    half2_t b1,
                                    half2_t b2,
                                    half2_t b3,
                                    float& c0,
                                    float& c1,
                                    float& c2,
                                    float& c3)
{
    asm volatile("\n \
            v_dot2_f32_f16 %0, %4, %5  %0\n \
            v_dot2_f32_f16 %1, %4, %6  %1\n \
            v_dot2_f32_f16 %2, %4, %7  %2\n \
            v_dot2_f32_f16 %3, %4, %8  %3\n \
            "
                 : "=v"(c0), "=v"(c1), "=v"(c2), "=v"(c3) // Dest registers
                 : "v"(a),                                // 1st Src register for 1 half2 registers
                   "v"(b0),                               // 2nd Src register
                   "v"(b1),
                   "v"(b2),
                   "v"(b3),
                   "0"(c0), // 3rd Src register
                   "1"(c1),
                   "2"(c2),
                   "3"(c3));
}

// outer-product: c[i,j] += inner_product(a[i], b[j])
__device__ void __outer_product_1x4(half4_t a,
                                    half4_t b0,
                                    half4_t b1,
                                    half4_t b2,
                                    half4_t b3,
                                    float& c0,
                                    float& c1,
                                    float& c2,
                                    float& c3)
{
    const half2_t* p_a_half2  = reinterpret_cast<const half2_t*>(&a);
    const half2_t* p_b0_half2 = reinterpret_cast<const half2_t*>(&b0);
    const half2_t* p_b1_half2 = reinterpret_cast<const half2_t*>(&b1);
    const half2_t* p_b2_half2 = reinterpret_cast<const half2_t*>(&b2);
    const half2_t* p_b3_half2 = reinterpret_cast<const half2_t*>(&b3);

    // do dot2 two times
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
                 : "=v"(c0), "=v"(c1), "=v"(c2), "=v"(c3) // Dest registers
                 : "v"(p_a_half2[0]),
                   "v"(p_a_half2[1]), // 1st Src registers for 2 half2 registers
                   "v"(p_b0_half2[0]),
                   "v"(p_b0_half2[1]),
                   "v"(p_b1_half2[0]),
                   "v"(p_b1_half2[1]), // 2nd Src registers for 2 half2 registers
                   "v"(p_b2_half2[0]),
                   "v"(p_b2_half2[1]),
                   "v"(p_b3_half2[0]),
                   "v"(p_b3_half2[1]), // 2nd Src registers for 2 half2 registers
                   "0"(c0),
                   "1"(c1),
                   "2"(c2),
                   "3"(c3)); // 3rd Src Acc registers for 2 half2 registers
}

} // namespace ck
#endif
