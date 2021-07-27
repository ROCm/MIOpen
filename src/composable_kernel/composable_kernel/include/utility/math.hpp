#ifndef CK_MATH_HPP
#define CK_MATH_HPP

#include "config.hpp"
#include "integral_constant.hpp"
#include "number.hpp"
#include "type.hpp"

namespace ck {
namespace math {

template <typename T, T s>
struct scales
{
    __host__ __device__ constexpr T operator()(T a) const { return s * a; }
};

template <typename T>
struct plus
{
    __host__ __device__ constexpr T operator()(T a, T b) const { return a + b; }
};

template <typename T>
struct minus
{
    __host__ __device__ constexpr T operator()(T a, T b) const { return a - b; }
};

template <typename T>
struct multiplies
{
    __host__ __device__ constexpr T operator()(T a, T b) const { return a * b; }
};

struct multiplies_v2
{
    template <typename A, typename B>
    __host__ __device__ constexpr auto operator()(const A& a, const B& b) const
    {
        return a * b;
    }
};

template <typename T>
struct maximize
{
    __host__ __device__ constexpr T operator()(T a, T b) const { return a >= b ? a : b; }
};

template <typename T>
struct minimize
{
    __host__ __device__ constexpr T operator()(T a, T b) const { return a <= b ? a : b; }
};

template <typename T>
struct integer_divide_ceiler
{
    __host__ __device__ constexpr T operator()(T a, T b) const
    {
        static_assert(is_same<T, index_t>{} || is_same<T, int>{}, "wrong type");

        return (a + b - Number<1>{}) / b;
    }
};

template <typename X, typename Y>
__host__ __device__ constexpr auto integer_divide_floor(X x, Y y)
{
    return x / y;
}

template <typename X, typename Y>
__host__ __device__ constexpr auto integer_divide_ceil(X x, Y y)
{
    return (x + y - Number<1>{}) / y;
}

template <typename X, typename Y>
__host__ __device__ constexpr auto integer_least_multiple(X x, Y y)
{
    return y * integer_divide_ceil(x, y);
}

template <typename T>
__host__ __device__ constexpr T max(T x)
{
    return x;
}

template <typename T>
__host__ __device__ constexpr T max(T x, T y)
{
    return x > y ? x : y;
}

template <index_t X>
__host__ __device__ constexpr index_t max(Number<X>, index_t y)
{
    return X > y ? X : y;
}

template <index_t Y>
__host__ __device__ constexpr index_t max(index_t x, Number<Y>)
{
    return x > Y ? x : Y;
}

template <typename X, typename... Ys>
__host__ __device__ constexpr auto max(X x, Ys... ys)
{
    static_assert(sizeof...(Ys) > 0, "not enough argument");

    return max(x, max(ys...));
}

template <typename T>
__host__ __device__ constexpr T min(T x)
{
    return x;
}

template <typename T>
__host__ __device__ constexpr T min(T x, T y)
{
    return x < y ? x : y;
}

template <index_t X>
__host__ __device__ constexpr index_t min(Number<X>, index_t y)
{
    return X < y ? X : y;
}

template <index_t Y>
__host__ __device__ constexpr index_t min(index_t x, Number<Y>)
{
    return x < Y ? x : Y;
}

template <typename X, typename... Ys>
__host__ __device__ constexpr auto min(X x, Ys... ys)
{
    static_assert(sizeof...(Ys) > 0, "not enough argument");

    return min(x, min(ys...));
}

// greatest common divisor, aka highest common factor
__host__ __device__ constexpr index_t gcd(index_t x, index_t y)
{
    if(x < 0)
    {
        return gcd(-x, y);
    }
    else if(y < 0)
    {
        return gcd(x, -y);
    }
    else if(x == y || x == 0)
    {
        return y;
    }
    else if(y == 0)
    {
        return x;
    }
    else if(x > y)
    {
        return gcd(x % y, y);
    }
    else
    {
        return gcd(x, y % x);
    }
}

template <index_t X, index_t Y>
__host__ __device__ constexpr auto gcd(Number<X>, Number<Y>)
{
    constexpr auto r = gcd(X, Y);

    return Number<r>{};
}

template <typename X,
          typename... Ys,
          typename std::enable_if<sizeof...(Ys) >= 2, bool>::type = false>
__host__ __device__ constexpr auto gcd(X x, Ys... ys)
{
    return gcd(x, gcd(ys...));
}

// least common multiple
template <typename X, typename Y>
__host__ __device__ constexpr auto lcm(X x, Y y)
{
    return (x * y) / gcd(x, y);
}

template <typename X,
          typename... Ys,
          typename std::enable_if<sizeof...(Ys) >= 2, bool>::type = false>
__host__ __device__ constexpr auto lcm(X x, Ys... ys)
{
    return lcm(x, lcm(ys...));
}

template <typename T>
struct equal
{
    __host__ __device__ constexpr bool operator()(T x, T y) const { return x == y; }
};

template <typename T>
struct less
{
    __host__ __device__ constexpr bool operator()(T x, T y) const { return x < y; }
};

} // namespace math
} // namespace ck

#endif
