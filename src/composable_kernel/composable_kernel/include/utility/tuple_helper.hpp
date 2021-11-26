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
#ifndef CK_TUPLE_HELPER_HPP
#define CK_TUPLE_HELPER_HPP

#include "functional4.hpp"
#include "tuple.hpp"

namespace ck {

template <typename... Ts>
struct is_known_at_compile_time<Tuple<Ts...>>
{
    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return container_reduce(
            Tuple<Ts...>{},
            [](auto x, bool r) {
                return is_known_at_compile_time<remove_cvref_t<decltype(x)>>::value & r;
            },
            true);
    }

    static constexpr bool value = IsKnownAtCompileTime();
};

template <typename F, index_t N>
__host__ __device__ constexpr auto generate_tuple(F&& f, Number<N>)
{
    return unpack([&f](auto&&... xs) { return make_tuple(f(xs)...); },
                  typename arithmetic_sequence_gen<0, N, 1>::type{});
}

namespace detail {

template <typename F, typename X, index_t... Is>
__host__ __device__ constexpr auto transform_tuples_impl(F f, const X& x, Sequence<Is...>)
{
    return make_tuple(f(x.At(Number<Is>{}))...);
}

template <typename F, typename X, typename Y, index_t... Is>
__host__ __device__ constexpr auto
transform_tuples_impl(F f, const X& x, const Y& y, Sequence<Is...>)
{
    return make_tuple(f(x.At(Number<Is>{}), y.At(Number<Is>{}))...);
}

template <typename F, typename X, typename Y, typename Z, index_t... Is>
__host__ __device__ constexpr auto
transform_tuples_impl(F f, const X& x, const Y& y, const Z& z, Sequence<Is...>)
{
    return make_tuple(f(x.At(Number<Is>{}), y.At(Number<Is>{}), z.At(Number<Is>{}))...);
}

} // namespace detail

template <typename F, typename X>
__host__ __device__ constexpr auto transform_tuples(F f, const X& x)
{
    return detail::transform_tuples_impl(
        f, x, typename arithmetic_sequence_gen<0, X::Size(), 1>::type{});
}

template <typename F, typename X, typename Y>
__host__ __device__ constexpr auto transform_tuples(F f, const X& x, const Y& y)
{
    return detail::transform_tuples_impl(
        f, x, y, typename arithmetic_sequence_gen<0, X::Size(), 1>::type{});
}

template <typename F, typename X, typename Y, typename Z>
__host__ __device__ constexpr auto transform_tuples(F f, const X& x, const Y& y, const Z& z)
{
    return detail::transform_tuples_impl(
        f, x, y, z, typename arithmetic_sequence_gen<0, X::Size(), 1>::type{});
}

} // namespace ck
#endif
