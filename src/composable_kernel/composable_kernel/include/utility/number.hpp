#ifndef CK_NUMBER_HPP
#define CK_NUMBER_HPP

#include "integral_constant.hpp"

namespace ck {

template <index_t N>
using Number = integral_constant<index_t, N>;

template <index_t X, index_t Y>
__host__ __device__ constexpr auto operator+(Number<X>, Number<Y>)
{
    return Number<X + Y>{};
}

template <index_t X, index_t Y>
__host__ __device__ constexpr auto operator-(Number<X>, Number<Y>)
{
    static_assert(Y <= X, "wrong!");
    return Number<X - Y>{};
}

template <index_t X, index_t Y>
__host__ __device__ constexpr auto operator*(Number<X>, Number<Y>)
{
    return Number<X * Y>{};
}

template <index_t X, index_t Y>
__host__ __device__ constexpr auto operator/(Number<X>, Number<Y>)
{
    static_assert(Y > 0, "wrong!");
    return Number<X / Y>{};
}

template <index_t X, index_t Y>
__host__ __device__ constexpr auto operator%(Number<X>, Number<Y>)
{
    static_assert(Y > 0, "wrong!");
    return Number<X % Y>{};
}
} // namespace ck
#endif
