#ifndef CK_MATH_HPP
#define CK_MATH_HPP

#include "config.hpp"
#include "integral_constant.hpp"
#include "type.hpp"

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

// this is WRONG
// TODO: implement least common multiple properly, instead of calling max()
template <class T, class... Ts>
__host__ __device__ constexpr T lcm(T x, Ts... xs)
{
    return max(x, xs...);
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
} // namspace ck

#endif
