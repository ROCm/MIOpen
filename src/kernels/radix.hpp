#pragma once
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define RADIX_TYPE typename RadixType<DTYPE>::type

#define DEFINE_RADIX_TYPE(DTYPE, cpp_type) \
    template <>                            \
    struct RadixType<DTYPE>                \
    {                                      \
        using type = cpp_type;             \
    };

template <typename T>
struct RadixType
{
};

DEFINE_RADIX_TYPE(int32_t, uint32_t)
DEFINE_RADIX_TYPE(int64_t, uint64_t)
DEFINE_RADIX_TYPE(bool, bool)
DEFINE_RADIX_TYPE(float, uint32_t)
DEFINE_RADIX_TYPE(double, uint64_t)
DEFINE_RADIX_TYPE(__half, ushort)

template <typename DTYPE, typename Radix = typename RadixType<DTYPE>::type>
__device__ inline Radix encode(DTYPE v)
{
    if constexpr(std::is_same<bool, DTYPE>::value)
    {
        return v;
    }
    else if constexpr(std::is_same<int32_t, DTYPE>::value)
    {
        return 2147483648u + v;
    }
    else if constexpr(std::is_same<int64_t, DTYPE>::value)
    {
        return 9223372036854775808ull + v;
    }
    else if constexpr(std::is_same<__half, DTYPE>::value)
    {
        Radix x    = __half_as_ushort(v);
        Radix mask = (x & 0x8000) ? 0xffff : 0x8000;
        return (v == v) ? (x ^ mask) : 0xffff;
    }
    else if constexpr(std::is_same<float, DTYPE>::value)
    {
        Radix x    = __float_as_uint(v);
        Radix mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;
        return (v == v) ? (x ^ mask) : 0xffffffff;
    }
    else if constexpr(std::is_same<double, DTYPE>::value)
    {
        Radix x    = __double_as_ulonglong(v);
        Radix mask = -((x >> 63)) | 0x8000000000000000;
        return (v == v) ? (x ^ mask) : 0xffffffffffffffff;
    }
}

template <typename DTYPE, typename Radix>
__device__ inline DTYPE decode(Radix v)
{
    if constexpr(std::is_same<bool, DTYPE>::value)
    {
        return v;
    }
    else if constexpr(std::is_same<int32_t, DTYPE>::value)
    {
        return v - 2147483648u;
    }
    else if constexpr(std::is_same<int64_t, DTYPE>::value)
    {
        return v - 9223372036854775808ull;
    }
    else if constexpr(std::is_same<__half, DTYPE>::value)
    {
        Radix mask = (v & 0x8000) ? 0x8000 : 0xffff;
        return __ushort_as_half((ushort)(v ^ mask));
    }
    else if constexpr(std::is_same<float, DTYPE>::value)
    {
        Radix mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;
        return __uint_as_float(v ^ mask);
    }
    else if constexpr(std::is_same<double, DTYPE>::value)
    {
        Radix mask = ((v >> 63) - 1) | 0x8000000000000000;
        return __ulonglong_as_double(v ^ mask);
    }
}

// returns x[pos+bits:pos]
template <typename Radix>
__device__ inline Radix GetBitFieldImpl(Radix x, int pos, int bits)
{
    return (x >> pos) & ((1 << bits) - 1);
}

// x[pos+bits:pos] = a
template <typename Radix>
__device__ inline Radix SetBitFieldImpl(Radix x, Radix a, int pos, int bits)
{
    return x | (a << pos);
}
