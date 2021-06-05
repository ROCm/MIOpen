#ifndef CK_FUNCTIONAL_HPP
#define CK_FUNCTIONAL_HPP

#include "integral_constant.hpp"
#include "sequence.hpp"
#include "type.hpp"

namespace ck {

// TODO: right? wrong?
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
    template <typename... Ts>
    __host__ __device__ constexpr swallow(Ts&&...)
    {
    }
};

template <typename T>
struct logical_and
{
    constexpr bool operator()(const T& x, const T& y) const { return x && y; }
};

template <typename T>
struct logical_or
{
    constexpr bool operator()(const T& x, const T& y) const { return x || y; }
};

template <typename T>
struct logical_not
{
    constexpr bool operator()(const T& x) const { return !x; }
};

// Emulate if constexpr
template <bool>
struct static_if;

template <>
struct static_if<true>
{
    using Type = static_if<true>;

    template <typename F>
    __host__ __device__ constexpr auto operator()(F f) const
    {
        // This is a trick for compiler:
        //   Pass forwarder to lambda "f" as "auto" argument, and make sure "f" will use it,
        //   this will make "f" a generic lambda, so that "f" won't be compiled until being
        //   instantiated here
        f(forwarder{});
        return Type{};
    }

    template <typename F>
    __host__ __device__ static void Else(F)
    {
    }
};

template <>
struct static_if<false>
{
    using Type = static_if<false>;

    template <typename F>
    __host__ __device__ constexpr auto operator()(F) const
    {
        return Type{};
    }

    template <typename F>
    __host__ __device__ static void Else(F f)
    {
        // This is a trick for compiler:
        //   Pass forwarder to lambda "f" as "auto" argument, and make sure "f" will use it,
        //   this will make "f" a generic lambda, so that "f" won't be compiled until being
        //   instantiated here
        f(forwarder{});
    }
};

template <bool predicate, class X, class Y>
struct conditional;

template <class X, class Y>
struct conditional<true, X, Y>
{
    using type = X;
};

template <class X, class Y>
struct conditional<false, X, Y>
{
    using type = Y;
};

template <bool predicate, class X, class Y>
using conditional_t = typename conditional<predicate, X, Y>::type;

} // namespace ck
#endif
