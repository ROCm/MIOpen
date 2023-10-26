/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_REDUCE_COMMON_HPP
#define GUARD_MIOPEN_REDUCE_COMMON_HPP

#if !defined(_WIN32) && (HIP_PACKAGE_VERSION_FLAT >= 5006000000ULL)
#include <half/half.hpp>
#else
#include <half.hpp>
#endif
#include <miopen/bfloat16.hpp>

namespace reduce {

template <typename Tdst, typename Tsrc>
static inline Tdst convert_type(Tsrc x)
{
    return static_cast<Tdst>(x);
}

template <>
inline float convert_type<float>(half_float::half x)
{
    return half_float::half_cast<float>(x);
};

template <>
inline half_float::half convert_type<half_float::half>(float x)
{
    return half_float::half_cast<half_float::half>(x);
};

template <>
inline float convert_type<float>(bfloat16 x)
{
    return float(x);
};

template <>
inline bfloat16 convert_type<bfloat16>(float x)
{
    return bfloat16(x);
};

}; // end of namespace reduce

#endif
