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
#ifndef _CK_REDUCTION_OPERATOR_HPP_
#define _CK_REDUCTION_OPERATOR_HPP_ 1

#include <limits>
#include "reduction_common.hpp"

namespace ck {

namespace reduce {

template <class T>
struct Add
{
    using dataType                = T;
    static constexpr auto zeroVal = static_cast<T>(0.0);

    __device__ constexpr T operator()(T a, T b) const { return a + b; }
    static constexpr bool indexable = false;
};

template <class T>
struct Mul
{
    using dataType                = T;
    static constexpr auto zeroVal = static_cast<T>(1.0);

    __device__ constexpr T operator()(T a, T b) const { return a * b; }
    static constexpr bool indexable = false;
};

template <class T>
struct Max
{
    using dataType                = T;
    static constexpr auto zeroVal = std::numeric_limits<T>::min();

    __host__ __device__ constexpr T operator()(T a, T b) const { return a >= b ? a : b; }
    static constexpr bool indexable = true;
};

/*
template <class T>
struct Amax
{
    using dataType = T;
    static constexpr auto zeroVal = static_cast<T>(0.0);

    __device__ constexpr T operator()(T a, T b) const {
        return std::abs<T>(a) >= std::abs<T>(b) ? std::abs<T>(a) : std::abs<T>(b) ;
    }
};
*/

template <class T>
struct Min
{
    using dataType                = T;
    static constexpr auto zeroVal = std::numeric_limits<T>::max();

    __device__ constexpr T operator()(T a, T b) const { return a <= b ? a : b; }
    static constexpr bool indexable = true;
};

}; // end of namespace reduce

template <typename T, ckReduceTensorOp_t op>
struct reduce_binary_operator;

template <typename T>
struct reduce_binary_operator<T, CK_REDUCE_TENSOR_ADD>
{
    using opType   = reduce::Add<T>;
    using dataType = T;

    static constexpr auto zeroVal   = reduce::Add<T>::zeroVal;
    static constexpr bool indexable = reduce::Add<T>::indexable;
};

template <typename T>
struct reduce_binary_operator<T, CK_REDUCE_TENSOR_MUL>
{
    using opType   = reduce::Mul<T>;
    using dataType = T;

    static constexpr auto zeroVal   = reduce::Mul<T>::zeroVal;
    static constexpr bool indexable = reduce::Mul<T>::indexable;
};

template <typename T>
struct reduce_binary_operator<T, CK_REDUCE_TENSOR_MIN>
{
    using opType   = reduce::Min<T>;
    using dataType = T;

    static constexpr auto zeroVal   = reduce::Min<T>::zeroVal;
    static constexpr bool indexable = reduce::Min<T>::indexable;
};

template <typename T>
struct reduce_binary_operator<T, CK_REDUCE_TENSOR_MAX>
{
    using opType   = reduce::Max<T>;
    using dataType = T;

    static constexpr auto zeroVal   = reduce::Max<T>::zeroVal;
    static constexpr bool indexable = reduce::Max<T>::indexable;
};

} // end of namespace ck

#endif
