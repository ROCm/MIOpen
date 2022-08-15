/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#ifndef CK_MATH_HPP
#define CK_MATH_HPP

#include "static_kernel_config.hpp"
#include "static_kernel_integral_constant.hpp"
#include "static_kernel_ck_utils_type.hpp"

namespace ck {
namespace math {

template <class T, T s>
struct scales
{
    __host__ __device__ constexpr T operator()(T a) const { return s * a; }
};

template <class T>
struct plus
{
    __host__ __device__ constexpr T operator()(T a, T b) const { return a + b; }
};

template <class T>
struct minus
{
    __host__ __device__ constexpr T operator()(T a, T b) const { return a - b; }
};

template <class T>
struct multiplies
{
    __host__ __device__ constexpr T operator()(T a, T b) const { return a * b; }
};

template <class T>
struct maxer
{
    __host__ __device__ constexpr T operator()(T a, T b) const { return a >= b ? a : b; }
};

template <class T>
struct integer_divide_ceiler
{
    __host__ __device__ constexpr T operator()(T a, T b) const
    {
        static_assert(is_same<T, index_t>{} || is_same<T, int>{}, "wrong type");

        return (a + b - 1) / b;
    }
};

template <class X, class Y>
__host__ __device__ constexpr auto integer_divide_floor(X x, Y y)
{
    return x / y;
}

template <class X, class Y>
__host__ __device__ constexpr auto integer_divide_ceil(X x, Y y)
{
    return (x + y - 1) / y;
}

template <class X, class Y>
__host__ __device__ constexpr auto integer_least_multiple(X x, Y y)
{
    return y * integer_divide_ceil(x, y);
}

template <class T>
__host__ __device__ constexpr T max(T x)
{
    return x;
}

template <class T, class... Ts>
__host__ __device__ constexpr T max(T x, Ts... xs)
{
    static_assert(sizeof...(xs) > 0, "not enough argument");

    auto y = max(xs...);

    static_assert(is_same<decltype(y), T>{}, "not the same type");

    return x > y ? x : y;
}

template <class T>
__host__ __device__ constexpr T min(T x)
{
    return x;
}

template <class T, class... Ts>
__host__ __device__ constexpr T min(T x, Ts... xs)
{
    static_assert(sizeof...(xs) > 0, "not enough argument");

    auto y = min(xs...);

    static_assert(is_same<decltype(y), T>{}, "not the same type");

    return x < y ? x : y;
}

// greatest common divisor, aka highest common factor
template <typename T>
__host__ __device__ constexpr T gcd(T x, T y)
{
    if(x == y || x == 0)
    {
        return y;
    }
    else if(y == 0)
    {
        return x;
    }
    else if(x > y)
    {
        return gcd(x - y, y);
    }
    else
    {
        return gcd(x, y - x);
    }
}

template <index_t X, index_t Y>
__host__ __device__ constexpr auto gcd(Number<X>, Number<Y>)
{
    constexpr auto result = gcd(X, Y);
    return Number<result>{};
}

template <typename X, typename... Ys>
__host__ __device__ constexpr auto gcd(X x, Ys... ys)
{
    return gcd(x, ys...);
}

// least common multiple
template <typename T>
__host__ __device__ constexpr T lcm(T x, T y)
{
    return (x * y) / gcd(x, y);
}

template <typename X, typename... Ys>
__host__ __device__ constexpr auto lcm(X x, Ys... ys)
{
    return lcm(x, lcm(ys...));
}

template <class T>
struct equal
{
    __host__ __device__ constexpr bool operator()(T x, T y) const { return x == y; }
};

template <class T>
struct less
{
    __host__ __device__ constexpr bool operator()(T x, T y) const { return x < y; }
};

} // namespace math
} // namespace ck

#endif
