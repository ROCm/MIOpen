#ifndef CK_INTEGRAL_CONSTANT_HPP
#define CK_INTEGRAL_CONSTANT_HPP

namespace ck {

template <class T, T v>
struct integral_constant
{
    static constexpr T value = v;
    typedef T value_type;
    typedef integral_constant type;
    __host__ __device__ constexpr operator value_type() const noexcept { return value; }
    __host__ __device__ constexpr value_type operator()() const noexcept { return value; }
};

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

template <class X, class Y>
struct is_same : public integral_constant<bool, false>
{
};

template <class X>
struct is_same<X, X> : public integral_constant<bool, true>
{
};

} // namespace ck
#endif
