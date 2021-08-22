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
#ifndef CK_REDUCTION_OPERATOR_HPP
#define CK_REDUCTION_OPERATOR_HPP

#include "reduction_common.hpp"

namespace ck {

namespace reduce {

// Every binary operator used in reduction is represented by a templated functor class. Each functor
// class must provide at least
// three members:
// 1) GetZeroVal() -- the interface to return the "identity element" for the binary operator,
// "identity element" is the unique
//                    element in the algebraic space that doesn't affect the value of other elements
//                    when operated with any of them.
// 2) indexable -- boolean value indicating whether indices of the operated elements could be
// recorded. Usually, Min/Max operator could
//                 need to record the indices of elements. For operator like Add/Mul, no need to
//                 record the indices.
// 3) operator() -- the first argument of the operator must be both an input & output, and the
// corresponding variable usually stores
//                  the accumulated result of many operator() calls; the second argument is only an
//                  input. For indexable binary
//                  operator, the second version of operator() has third argument (which is an
//                  output) to indicate whether the
//                  accumulated value (the first argument) has changed, in which case the recorded
//                  accumulated index also need be
//                  changed.

template <class T>
struct Add
{
    using dataType = T;

    __device__ static T GetZeroVal() { return type_convert<T>{}(0.0f); };

    __device__ inline constexpr void operator()(T& a, T b) const { a = a + b; }

    static constexpr bool indexable = false;
};

template <class T>
struct Mul
{
    using dataType = T;

    __device__ static T GetZeroVal() { return type_convert<T>{}(1.0f); };

    __device__ inline constexpr void operator()(T& a, T b) const { a = a * b; }

    static constexpr bool indexable = false;
};

template <class T>
struct Max
{
    using dataType = T;

    __device__ static T GetZeroVal() { return std::numeric_limits<T>::min(); };

    __device__ inline constexpr void operator()(T& a, T b) const
    {
        if(a < b)
            a = b;
    }

    __device__ inline constexpr void operator()(T& a, T b, bool& changed) const
    {
        if(a < b)
        {
            a       = b;
            changed = true;
        }
    }

    static constexpr bool indexable = true;
};

template <class T>
struct Min
{
    using dataType = T;

    __device__ static T GetZeroVal() { return std::numeric_limits<T>::max(); };

    __device__ inline constexpr void operator()(T& a, T b) const
    {
        if(a > b)
            a = b;
    }

    __device__ inline constexpr void operator()(T& a, T b, bool& changed) const
    {
        if(a > b)
        {
            a       = b;
            changed = true;
        }
    }

    static constexpr bool indexable = true;
};

template <>
__device__ half_t Max<half_t>::GetZeroVal()
{
    return type_convert<half_t>{}(std::numeric_limits<float>::min());
};

template <>
__device__ half_t Min<half_t>::GetZeroVal()
{
    return type_convert<half_t>{}(std::numeric_limits<float>::max());
};

// Unary operators are usually called element-wisely before the reduction is executed on the
// elements.
// They are needed for easy implementation of reduction types of AVG, NRM1, NRM2
template <class T, bool hasDividing>
struct unary_identic
{
    __device__ unary_identic(const int divider = 1)
    {
        scaler = 1.0f / static_cast<float>(divider);
    };

    __device__ inline constexpr void operator()(T& a) const { a = a * type_convert<T>{}(scaler); };

    float scaler = 1.0f;
};

template <class T>
struct unary_identic<T, false>
{
    __device__ unary_identic(const int divider = 1) { (void)divider; };

    __device__ inline constexpr void operator()(T&) const {};
};

template <class T, bool hasDividing>
struct unary_square
{
    __device__ unary_square(const int divider = 1) { scaler = 1.0f / static_cast<float>(divider); };

    __device__ inline constexpr void operator()(T& a) const
    {
        a = a * a;

        a = a * type_convert<T>{}(scaler);
    };

    float scaler = 1.0f;
};

template <class T>
struct unary_square<T, false>
{
    __device__ unary_square(const int divider = 1) { (void)divider; };

    __device__ inline constexpr void operator()(T& a) const { a = a * a; };
};

template <class T, bool hasDividing>
struct unary_abs
{
    __device__ unary_abs(const int divider = 1) { scaler = 1.0f / static_cast<float>(divider); };

    __device__ inline constexpr void operator()(T& a) const
    {
        a = abs(a);

        a = a * type_convert<T>{}(scaler);
    };

    float scaler = 1.0f;
};

template <class T>
struct unary_abs<T, false>
{
    __device__ unary_abs(const int divider = 1) { (void)divider; };

    __device__ inline constexpr void operator()(T& a) const { a = abs(a); };
};

// We know for sure that 4.0 has __habs(), but 3.0 does not have it.
// Let's assume that __habs() exists since 3.5.
#if HIP_PACKAGE_VERSION_FLAT < 3005000000
inline __device__ __half __habs(__half x)
{
    union
    {
        __half half;
        unsigned short u16;
    } val;
    val.half = x;
    val.u16  = val.u16 & 0x7fff;
    return val.half;
}
#endif

template <bool hasDividing>
struct unary_abs<half_t, hasDividing>
{
    __device__ unary_abs(const int divider = 1) { scaler = 1.0f / static_cast<float>(divider); };

    __device__ inline void operator()(half_t& a) const
    {
        a = static_cast<half_t>(__habs(a));

        a = a * type_convert<half_t>{}(scaler);
    };

    float scaler = 1.0f;
};

template <>
struct unary_abs<half_t, false>
{
    __device__ unary_abs(const int divider = 1) { (void)divider; };

    __device__ inline void operator()(half_t& a) const { a = static_cast<half_t>(__habs(a)); };
};

template <class T>
struct unary_sqrt
{
    __device__ unary_sqrt(const int divider = 1) { (void)divider; };

    __device__ inline constexpr void operator()(T& a) const { a = sqrtf(a); };
};

template <>
struct unary_sqrt<half_t>
{
    __device__ unary_sqrt(const int divider = 1) { (void)divider; };

    __device__ inline void operator()(half_t& a) const { a = static_cast<half_t>(hsqrt(a)); };
};

}; // end of namespace reduce

// The templated struct reduce_binary_operator maps the enum Ids of binary operators to their
// respective functor classes.
// The "GetZeroVal()" interface and boolean member "indexable" are also provided in
// reduce_binary_operactor for
// easier checking by the upper-layer codes in the kernels.

template <typename T, ReduceTensorOp_t op>
struct reduce_binary_operator;

template <typename T>
struct reduce_binary_operator<T, ReduceTensorOp_t::ADD>
{
    using opType   = reduce::Add<T>;
    using dataType = T;

    __device__ static T GetZeroVal() { return reduce::Add<T>::GetZeroVal(); };

    static constexpr bool indexable = reduce::Add<T>::indexable;
};

template <typename T>
struct reduce_binary_operator<T, ReduceTensorOp_t::MUL>
{
    using opType   = reduce::Mul<T>;
    using dataType = T;

    __device__ static T GetZeroVal() { return reduce::Mul<T>::GetZeroVal(); };

    static constexpr bool indexable = reduce::Mul<T>::indexable;
};

template <typename T>
struct reduce_binary_operator<T, ReduceTensorOp_t::MIN>
{
    using opType   = reduce::Min<T>;
    using dataType = T;

    __device__ static T GetZeroVal() { return reduce::Min<T>::GetZeroVal(); };

    static constexpr bool indexable = reduce::Min<T>::indexable;
};

template <typename T>
struct reduce_binary_operator<T, ReduceTensorOp_t::MAX>
{
    using opType   = reduce::Max<T>;
    using dataType = T;

    __device__ static T GetZeroVal() { return reduce::Max<T>::GetZeroVal(); };

    static constexpr bool indexable = reduce::Max<T>::indexable;
};

template <typename T>
struct reduce_binary_operator<T, ReduceTensorOp_t::AMAX>
{
    using opType   = reduce::Max<T>;
    using dataType = T;

    __device__ static T GetZeroVal() { return reduce::Max<T>::GetZeroVal(); };

    static constexpr bool indexable = reduce::Max<T>::indexable;
};

template <typename T>
struct reduce_binary_operator<T, ReduceTensorOp_t::AVG>
{
    using opType   = reduce::Add<T>;
    using dataType = T;

    __device__ static T GetZeroVal() { return reduce::Add<T>::GetZeroVal(); };

    static constexpr bool indexable = reduce::Add<T>::indexable;
};

template <typename T>
struct reduce_binary_operator<T, ReduceTensorOp_t::NORM1>
{
    using opType   = reduce::Add<T>;
    using dataType = T;

    __device__ static T GetZeroVal() { return reduce::Add<T>::GetZeroVal(); };

    static constexpr bool indexable = reduce::Add<T>::indexable;
};

template <typename T>
struct reduce_binary_operator<T, ReduceTensorOp_t::NORM2>
{
    using opType   = reduce::Add<T>;
    using dataType = T;

    __device__ static T GetZeroVal() { return reduce::Add<T>::GetZeroVal(); };

    static constexpr bool indexable = reduce::Add<T>::indexable;
};

// The templated struct reduce_unary_operator maps the enum Ids of Reduce operators to two unary
// functor classes.
// The two unary functors are called before and afer the Reduction is executed respectively
template <typename T, ReduceTensorOp_t op, bool isFirsReduce, bool isLastReduce>
struct reduce_unary_operator
{
    using preUnaryOp = reduce::unary_identic<T, false>;
    using posUnaryOp = reduce::unary_identic<T, false>;
};

template <typename T, bool isFirstReduce>
struct reduce_unary_operator<T, ReduceTensorOp_t::AVG, isFirstReduce, true>
{
    using preUnaryOp = reduce::unary_identic<T, false>;
    using posUnaryOp = reduce::unary_identic<T, true>;
};

template <typename T, bool isLastReduce>
struct reduce_unary_operator<T, ReduceTensorOp_t::NORM1, true, isLastReduce>
{
    using preUnaryOp = reduce::unary_abs<T, false>;
    using posUnaryOp = reduce::unary_identic<T, false>;
};

template <typename T, bool isLastReduce>
struct reduce_unary_operator<T, ReduceTensorOp_t::AMAX, true, isLastReduce>
{
    using preUnaryOp = reduce::unary_abs<T, false>;
    using posUnaryOp = reduce::unary_identic<T, false>;
};

template <typename T>
struct reduce_unary_operator<T, ReduceTensorOp_t::NORM2, true, false>
{
    using preUnaryOp = reduce::unary_square<T, false>;
    using posUnaryOp = reduce::unary_identic<T, false>;
};

template <typename T>
struct reduce_unary_operator<T, ReduceTensorOp_t::NORM2, true, true>
{
    using preUnaryOp = reduce::unary_square<T, false>;
    using posUnaryOp = reduce::unary_sqrt<T>;
};

template <typename T>
struct reduce_unary_operator<T, ReduceTensorOp_t::NORM2, false, true>
{
    using preUnaryOp = reduce::unary_identic<T, false>;
    using posUnaryOp = reduce::unary_sqrt<T>;
};

} // end of namespace ck

#endif
