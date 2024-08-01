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
 * The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
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
#pragma once

__device__ static inline __half __ushort_as___half(ushort x)
{
    static_assert(sizeof(ushort) == sizeof(__half), "");

    __half tmp;
    __builtin_memcpy(&tmp, &x, sizeof(tmp));

    return tmp;
}

__device__ static inline ushort ____half_as_ushort(__half x)
{
    static_assert(sizeof(ushort) == sizeof(__half), "");

    ushort tmp;
    __builtin_memcpy(&tmp, &x, sizeof(tmp));

    return tmp;
}

__device__ inline void atomic_add_g(ushort* addr, const float val)
{
    size_t offset    = reinterpret_cast<size_t>(addr) & 0x2;
    bool is_32_align = offset;
    uint32_t* addr_as_uint32_t =
        reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(addr) - offset);
    uint32_t current = *addr_as_uint32_t;

    uint32_t expected;

    do
    {
        expected              = current;
        ushort current_ushort = is_32_align ? current >> 16 : current & 0xffff;

        float next_float   = __uint_as_float(static_cast<uint32_t>(current_ushort) << 16) + val;
        ushort next_ushort = static_cast<ushort>(__float_as_uint(next_float) >> 16);
        uint32_t next      = is_32_align ? (current & 0xffff) | (next_ushort << 16)
                                         : (current & 0xffff0000) | next_ushort;

        current = atomicCAS(addr_as_uint32_t, expected, next);
    } while(current != expected);
}

__device__ inline void atomic_add_g(__half* addr, const __half val)
{
    size_t offset    = reinterpret_cast<size_t>(addr) & 0x2;
    bool is_32_align = offset;
    uint32_t* addr_as_uint32_t =
        reinterpret_cast<uint32_t*>(reinterpret_cast<size_t>(addr) - offset);
    uint32_t current = *addr_as_uint32_t;

    uint32_t expected;

    do
    {
        expected              = current;
        ushort current_ushort = is_32_align ? current >> 16 : current & 0xffff;

        ushort next_ushort = ____half_as_ushort(__ushort_as___half(current_ushort) + val);
        uint32_t next      = is_32_align ? (current & 0xffff) | (next_ushort << 16)
                                         : (current & 0xffff0000) | next_ushort;

        current = atomicCAS(addr_as_uint32_t, expected, next);
    } while(current != expected);
}

__device__ inline void atomic_add_g(float* addr, const float val) { atomicAdd(addr, val); }
