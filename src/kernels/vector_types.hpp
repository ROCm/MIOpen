/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#ifndef VECTOR_TYPES_HPP
#define VECTOR_TYPES_HPP

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif
namespace miopen {

// used by batch norm functions.
template <typename T, int N>
struct mapped_vector_type
{
    static_assert(false, "there is no specialization for this T & N combination.");
};

// Note: we should make sure there is no exception
template <typename T>
struct mapped_vector_type<T, 1>
{
    using type = T;
};

// float
template <>
struct mapped_vector_type<float, 4>
{
    using type = ::float4;
};

template <>
struct mapped_vector_type<float, 2>
{
    using type = ::float2;
};

// half

template <>
struct mapped_vector_type<_Float16, 8>
{
    using type = _Float16 __attribute__((ext_vector_type(8)));
};

template <>
struct mapped_vector_type<_Float16, 4>
{
    using type = _Float16 __attribute__((ext_vector_type(4)));
};

template <>
struct mapped_vector_type<_Float16, 2>
{
    using type = _Float16 __attribute__((ext_vector_type(2)));
};

// ushort
template <>
struct mapped_vector_type<ushort, 8>
{
    using type = ushort __attribute__((ext_vector_type(8)));
};

template <>
struct mapped_vector_type<ushort, 4>
{
    using type = ushort4;
};

template <>
struct mapped_vector_type<ushort, 2>
{
    using type = ushort2;
};

// int
template <>
struct mapped_vector_type<int, 4>
{
    using type = int4;
};

template <>
struct mapped_vector_type<int, 2>
{
    using type = int2;
};

} // namespace miopen

#endif
