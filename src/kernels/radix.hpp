/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef GUARD_RADIX_H
#define GUARD_RADIX_H

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
DEFINE_RADIX_TYPE(__half, ushort)
DEFINE_RADIX_TYPE(ushort, ushort) // bfloat16

template <typename DTYPE, typename Radix = typename RadixType<DTYPE>::type>
__device__ inline Radix encode(DTYPE v)
{
    // convert negative number to positive representation in Radix type.
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
    // bfloat16 is passed as ushort in kernel
    else if constexpr(std::is_same<ushort, DTYPE>::value)
    {
        Radix x    = v;
        Radix mask = (x & 0x8000) ? 0xffff : 0x8000;
        return (v == v) ? (x ^ mask) : 0xffff;
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
}

// returns x[pos+bits:pos]
template <typename Radix>
__device__ inline Radix GetBitFieldImpl(Radix x, int pos, int bits)
{
    return (x >> pos) & ((1 << bits) - 1);
}

// x[pos+bits:pos] = a
template <typename Radix>
__device__ inline Radix SetBitFieldImpl(Radix x, Radix a, int pos)
{
    return x | (a << pos);
}

#endif // GUARD_RADIX_H
