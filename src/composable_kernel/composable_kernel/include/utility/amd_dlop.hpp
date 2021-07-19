#ifndef CK_AMD_DLOP_HPP
#define CK_AMD_DLOP_HPP

#include "float_type.hpp"

namespace ck {

template <typename TA, typename TB, typename TC>
__device__ void amd_inner_product_dlop(const TA& a, const TB& b, TC& c);

template <>
__device__ void
amd_inner_product_dlop<float, float, float>(const float& a, const float& b, float& c)
{
#if CK_USE_AMD_DLOP_INLINE_ASM
    asm volatile("\n \
            v_fmac_f32 %0, %1, %2 \n \
            "
                 : "=v"(c)
                 : "v"(a), "v"(b), "0"(c));
#else
    c += a * b;
#endif
}

template <>
__device__ void
amd_inner_product_dlop<float2_t, float2_t, float>(const float2_t& a, const float2_t& b, float& c)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    amd_inner_product_dlop(vector_type<float, 2>{a}.AsType<float>()[I0],
                           vector_type<float, 2>{b}.AsType<float>()[I0],
                           c);

    amd_inner_product_dlop(vector_type<float, 2>{a}.AsType<float>()[I1],
                           vector_type<float, 2>{b}.AsType<float>()[I1],
                           c);
}

template <>
__device__ void
amd_inner_product_dlop<float4_t, float4_t, float>(const float4_t& a, const float4_t& b, float& c)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    amd_inner_product_dlop(vector_type<float, 4>{a}.AsType<float>()[I0],
                           vector_type<float, 4>{b}.AsType<float>()[I0],
                           c);

    amd_inner_product_dlop(vector_type<float, 4>{a}.AsType<float>()[I1],
                           vector_type<float, 4>{b}.AsType<float>()[I1],
                           c);

    amd_inner_product_dlop(vector_type<float, 4>{a}.AsType<float>()[I2],
                           vector_type<float, 4>{b}.AsType<float>()[I2],
                           c);

    amd_inner_product_dlop(vector_type<float, 4>{a}.AsType<float>()[I3],
                           vector_type<float, 4>{b}.AsType<float>()[I3],
                           c);
}

#if CK_USE_AMD_DLOP
template <>
__device__ void
amd_inner_product_dlop<half2_t, half2_t, float>(const half2_t& a, const half2_t& b, float& c)
{
#if CK_USE_AMD_DLOP_INLINE_ASM
    asm volatile("\n \
            v_dot2_f32_f16 %0, %1, %2, %0\n \
            "
                 : "=v"(c)
                 : "v"(a), "v"(b), "0"(c));
#else
    c = __builtin_amdgcn_sdot2(a, b, c, false);
#endif
}

template <>
__device__ void
amd_inner_product_dlop<half4_t, half4_t, float>(const half4_t& a, const half4_t& b, float& c)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    amd_inner_product_dlop(vector_type<half_t, 4>{a}.AsType<half2_t>()[I0],
                           vector_type<half_t, 4>{b}.AsType<half2_t>()[I0],
                           c);

    amd_inner_product_dlop(vector_type<half_t, 4>{a}.AsType<half2_t>()[I1],
                           vector_type<half_t, 4>{b}.AsType<half2_t>()[I1],
                           c);
}

template <>
__device__ void
amd_inner_product_dlop<half8_t, half8_t, float>(const half8_t& a, const half8_t& b, float& c)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    amd_inner_product_dlop(vector_type<half_t, 8>{a}.AsType<half2_t>()[I0],
                           vector_type<half_t, 8>{b}.AsType<half2_t>()[I0],
                           c);

    amd_inner_product_dlop(vector_type<half_t, 8>{a}.AsType<half2_t>()[I1],
                           vector_type<half_t, 8>{b}.AsType<half2_t>()[I1],
                           c);

    amd_inner_product_dlop(vector_type<half_t, 8>{a}.AsType<half2_t>()[I2],
                           vector_type<half_t, 8>{b}.AsType<half2_t>()[I2],
                           c);

    amd_inner_product_dlop(vector_type<half_t, 8>{a}.AsType<half2_t>()[I3],
                           vector_type<half_t, 8>{b}.AsType<half2_t>()[I3],
                           c);
}

template <>
__device__ void amd_inner_product_dlop<int8x4_t, int8x4_t, int32_t>(const int8x4_t& a,
                                                                    const int8x4_t& b,
                                                                    int32_t& c)
{
#if CK_USE_AMD_DLOP_INLINE_ASM
    asm volatile("\n \
            v_dot4_i32_i8 %0, %1, %2, %0\n \
            "
                 : "=v"(c)
                 : "v"(as_type<int32_t>(a)), "v"(as_type<int32_t>(b)), "0"(c));
#else
    c = __builtin_amdgcn_sdot4(as_type<int32_t>(a), as_type<int32_t>(b), c, false);
#endif
}

template <>
__device__ void amd_inner_product_dlop<int8x8_t, int8x8_t, int32_t>(const int8x8_t& a,
                                                                    const int8x8_t& b,
                                                                    int32_t& c)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    amd_inner_product_dlop(vector_type<int8_t, 8>{a}.AsType<int8x4_t>()[I0],
                           vector_type<int8_t, 8>{b}.AsType<int8x4_t>()[I0],
                           c);

    amd_inner_product_dlop(vector_type<int8_t, 8>{a}.AsType<int8x4_t>()[I1],
                           vector_type<int8_t, 8>{b}.AsType<int8x4_t>()[I1],
                           c);
}

template <>
__device__ void amd_inner_product_dlop<int8x16_t, int8x16_t, int32_t>(const int8x16_t& a,
                                                                      const int8x16_t& b,
                                                                      int32_t& c)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    amd_inner_product_dlop(vector_type<int8_t, 16>{a}.AsType<int8x4_t>()[I0],
                           vector_type<int8_t, 16>{b}.AsType<int8x4_t>()[I0],
                           c);

    amd_inner_product_dlop(vector_type<int8_t, 16>{a}.AsType<int8x4_t>()[I1],
                           vector_type<int8_t, 16>{b}.AsType<int8x4_t>()[I1],
                           c);

    amd_inner_product_dlop(vector_type<int8_t, 16>{a}.AsType<int8x4_t>()[I2],
                           vector_type<int8_t, 16>{b}.AsType<int8x4_t>()[I2],
                           c);

    amd_inner_product_dlop(vector_type<int8_t, 16>{a}.AsType<int8x4_t>()[I3],
                           vector_type<int8_t, 16>{b}.AsType<int8x4_t>()[I3],
                           c);
}
#endif // CK_USE_AMD_DLOP

} // namespace ck
#endif
