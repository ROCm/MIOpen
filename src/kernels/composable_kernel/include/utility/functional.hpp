#ifndef CK_FUNCTIONAL_HPP
#define CK_FUNCTIONAL_HPP

#include "integral_constant.hpp"
#include "Sequence.hpp"

namespace ck {

struct forwarder
{
    template <typename T>
    __host__ __device__ constexpr T&& operator()(T&& x) const
    {
        return static_cast<T&&>(x);
    }
};

struct swallow
{
    template <class... Ts>
    __host__ __device__ constexpr swallow(Ts&&...)
    {
    }
};

// Emulate if constexpr
template <bool>
struct static_if;

template <>
struct static_if<true>
{
    using Type = static_if<true>;

    template <class F>
    __host__ __device__ constexpr auto operator()(F f) const
    {
        // This is a trick for compiler:
        //   Pass forwarder to lambda "f" as "auto" argument, and make sure "f" will use it,
        //   this will make "f" a generic lambda, so that "f" won't be compiled until being
        //   instantiated here
        f(forwarder{});
        return Type{};
    }

    template <class F>
    __host__ __device__ static constexpr auto Else(F)
    {
        return Type{};
    }
};

template <>
struct static_if<false>
{
    using Type = static_if<false>;

    template <class F>
    __host__ __device__ constexpr auto operator()(F) const
    {
        return Type{};
    }

    template <class F>
    __host__ __device__ static constexpr auto Else(F f)
    {
        // This is a trick for compiler:
        //   Pass forwarder to lambda "f" as "auto" argument, and make sure "f" will use it,
        //   this will make "f" a generic lambda, so that "f" won't be compiled until being
        //   instantiated here
        f(forwarder{});
        return Type{};
    }
};

} // namespace ck
#endif
