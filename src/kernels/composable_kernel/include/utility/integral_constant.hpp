#ifndef CK_INTEGRAL_CONSTANT_HPP
#define CK_INTEGRAL_CONSTANT_HPP

#include <type_traits>

namespace ck {

template <class T, T v>
using integral_constant = std::integral_constant<T, v>;

template <class T, T X, T Y>
__host__ __device__ constexpr auto operator+(integral_constant<T, X>, integral_constant<T, Y>)
{
    return integral_constant<T, X + Y>{};
}

template <class T, T X, T Y>
__host__ __device__ constexpr auto operator*(integral_constant<T, X>, integral_constant<T, Y>)
{
    return integral_constant<T, X * Y>{};
}

template <index_t N>
using Number = integral_constant<index_t, N>;

} // namespace ck
#endif
