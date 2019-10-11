#ifndef CK_FUNCTIONAL4_HPP
#define CK_FUNCTIONAL4_HPP

#include "sequence.hpp"
#include "tuple.hpp"
#include "array.hpp"

namespace ck {

namespace detail {

template <typename Indices>
struct unpack_impl;

template <index_t... Is>
struct unpack_impl<Sequence<Is...>>
{
    template <typename F, typename X>
    __host__ __device__ constexpr auto operator()(F f, const X& x) const
    {
        return f(x.At(Number<Is>{})...);
    }
};

} // namespace detail

template <typename F, typename X>
__host__ __device__ constexpr auto unpack(F f, const X& x)
{
    return detail::unpack_impl<typename arithmetic_sequence_gen<0, X::Size(), 1>::type>{}(f, x);
}

} // namespace ck
#endif
