#ifndef CK_MATH_HPP
#define CK_MATH_HPP

#include "config.hpp"
#include "integral_constant.hpp"
#include "vector_type.hpp"

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
struct inner_product_with_conversion
{
    static constexpr auto convert = type_convert<T>();

    __device__ T operator()(float a, float b) const { return convert(a) * convert(b); }

    __device__ T operator()(const vector_type<half, 2>::MemoryType& a,
                            const vector_type<half, 2>::MemoryType& b) const
    {
        const half* p_a_half = reinterpret_cast<const half*>(&a);
        const half* p_b_half = reinterpret_cast<const half*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 2; ++v)
        {
            acc += convert(p_a_half[v]) * convert(p_b_half[v]);
        }

        return acc;
    }

    __device__ T operator()(const vector_type<half, 4>::MemoryType& a,
                            const vector_type<half, 4>::MemoryType& b) const
    {
        const half* p_a_half = reinterpret_cast<const half*>(&a);
        const half* p_b_half = reinterpret_cast<const half*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 4; ++v)
        {
            acc += convert(p_a_half[v]) * convert(p_b_half[v]);
        }
        return acc;
    }

    __device__ T operator()(const vector_type<ushort, 2>::MemoryType& a,
                            const vector_type<ushort, 2>::MemoryType& b) const
    {
        const ushort* p_a_bfloat16 = reinterpret_cast<const ushort*>(&a);
        const ushort* p_b_bfloat16 = reinterpret_cast<const ushort*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 2; ++v)
        {
            acc += convert(p_a_bfloat16[v]) * convert(p_b_bfloat16[v]);
        }

        return acc;
    }

    __device__ T operator()(const vector_type<ushort, 4>::MemoryType& a,
                            const vector_type<ushort, 4>::MemoryType& b) const
    {
        const ushort* p_a_bfloat16 = reinterpret_cast<const ushort*>(&a);
        const ushort* p_b_bfloat16 = reinterpret_cast<const ushort*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 4; ++v)
        {
            acc += convert(p_a_bfloat16[v]) * convert(p_b_bfloat16[v]);
        }
        return acc;
    }
};

} // namespace math
} // namspace ck

#endif
