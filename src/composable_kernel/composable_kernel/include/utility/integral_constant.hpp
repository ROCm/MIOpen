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

} // namespace ck
#endif
