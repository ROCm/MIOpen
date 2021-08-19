#ifndef CK_FUNCTIONAL2_HPP
#define CK_FUNCTIONAL2_HPP

#include "functional.hpp"
#include "sequence.hpp"

namespace ck {

namespace detail {

template <class>
struct static_for_impl;

template <index_t... Is>
struct static_for_impl<Sequence<Is...>>
{
    template <class F>
    __host__ __device__ constexpr void operator()(F f) const
    {
        swallow{(f(Number<Is>{}), 0)...};
    }
};

} // namespace detail

// F signature: F(Number<Iter>)
template <index_t NBegin, index_t NEnd, index_t Increment>
struct static_for
{
    __host__ __device__ constexpr static_for()
    {
        static_assert(Increment != 0 && (NEnd - NBegin) % Increment == 0,
                      "Wrong! should satisfy (NEnd - NBegin) % Increment == 0");
        static_assert((Increment > 0 && NBegin <= NEnd) || (Increment < 0 && NBegin >= NEnd),
                      "wrongs! should have NBegin <= NEnd");
    }

    template <class F>
    __host__ __device__ constexpr void operator()(F f) const
    {
        detail::static_for_impl<typename arithmetic_sequence_gen<NBegin, NEnd, Increment>::type>{}(
            f);
    }
};

} // namespace ck
#endif
