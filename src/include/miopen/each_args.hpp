/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_EACH_ARGS_HPP
#define GUARD_MIOPEN_EACH_ARGS_HPP

#include <initializer_list>
#include <type_traits>
#include <utility>

namespace miopen {
namespace detail {

template <class F, std::size_t... Ns, class... Ts>
void each_args_i_impl(F f, std::index_sequence<Ns...>, Ts&&... xs)
{
    (void)std::initializer_list<int>{
        (f(std::integral_constant<std::size_t, Ns>{}, std::forward<Ts>(xs)), 0)...};
}

template <class F, std::size_t... Ns, class T>
auto unpack_impl(F f, std::index_sequence<Ns...>, T&& x)
{
    return f(std::get<Ns>(x)...);
}

} // namespace detail

template <class F, class... Ts>
void each_args_i(F f, Ts&&... xs)
{
    detail::each_args_i_impl(f, std::make_index_sequence<sizeof...(Ts)>(), std::forward<Ts>(xs)...);
}

template <class F, class... Ts>
void each_args(F f, Ts&&... xs)
{
    (void)std::initializer_list<int>{(f(std::forward<Ts>(xs)), 0)...};
}

// Workaround for gcc warnings
template <class F>
void each_args(F)
{
}

template <class F, std::size_t... Ns, class T>
auto unpack(F f, T&& x)
{
    using type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
    return detail::unpack_impl(
        f, std::make_index_sequence<std::tuple_size<type>::value>(), std::forward<T>(x));
}

} // namespace miopen

#endif
