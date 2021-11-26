/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef CK_FUNCTIONAL_HPP
#define CK_FUNCTIONAL_HPP

#include "integral_constant.hpp"
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
        //   Pass forwarder to lambda "f" as "auto" argument, and make sure "f" will
        //   use it,
        //   this will make "f" a generic lambda, so that "f" won't be compiled
        //   until being
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
        //   Pass forwarder to lambda "f" as "auto" argument, and make sure "f" will
        //   use it,
        //   this will make "f" a generic lambda, so that "f" won't be compiled
        //   until being
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
