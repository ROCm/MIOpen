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

#include <half.hpp>
#include <miopen/bfloat16.hpp>

namespace reduce {

// data type conversion
template <typename T>
struct type_convert
{
    template <typename X>
    T operator()(X x) const
    {
        return static_cast<T>(x);
    }
};

template <>
template <>
inline float type_convert<float>::operator()<half_float::half>(half_float::half x) const
{
    return half_float::half_cast<float>(x);
};

template <>
template <>
inline half_float::half type_convert<half_float::half>::operator()<float>(float x) const
{
    return half_float::half_cast<half_float::half>(x);
};

template <>
template <>
inline float type_convert<float>::operator()<bfloat16>(bfloat16 x) const
{
    return float(x);
};

template <>
template <>
inline bfloat16 type_convert<bfloat16>::operator()<float>(float x) const
{
    return bfloat16(x);
};

}; // end of namespace reduce

#endif
