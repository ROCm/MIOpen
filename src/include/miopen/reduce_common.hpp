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
#ifndef GURAD_REDUCE_COMMON_HPP_
#define GURAD_REDUCE_COMMON_HPP_ 1

namespace reduce {

typedef enum
{
    Reduce_DirectThreadWise = 1,
    Reduce_DirectWarpWise   = 2,
    Reduce_BlockWise        = 3,
    Reduce_MultiBlock       = 4
} ReductionMethod_t;

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
float type_convert<float>::operator()<half_float::half>(half_float::half x) const
{
    return half_float::half_cast<float>(x);
};

template <>
template <>
half_float::half type_convert<half_float::half>::operator()<float>(float x) const
{
    return half_float::half_cast<half_float::half>(x);
};

}; // end of namespace reduce

#endif
