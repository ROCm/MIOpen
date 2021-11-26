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
#ifndef CK_STATICALLY_INDEXED_ARRAY_MULTI_INDEX_HPP
#define CK_STATICALLY_INDEXED_ARRAY_MULTI_INDEX_HPP

#include "common_header.hpp"

namespace ck {

template <index_t N>
using MultiIndex = StaticallyIndexedArray<index_t, N>;

template <typename... Xs>
__host__ __device__ constexpr auto make_multi_index(Xs&&... xs)
{
    return make_statically_indexed_array<index_t>(index_t{xs}...);
}

template <index_t NSize>
__host__ __device__ constexpr auto make_zero_multi_index()
{
    return unpack([](auto... xs) { return make_multi_index(xs...); },
                  typename uniform_sequence_gen<NSize, 0>::type{});
}

template <typename T>
__host__ __device__ constexpr auto to_multi_index(const T& x)
{
    return unpack([](auto... ys) { return make_multi_index(ys...); }, x);
}

// Here should use MultiIndex<NSize>, instead of Tuple<Ys...>, although the former
// is the alias of the latter. This is because compiler cannot infer the NSize if
// using MultiIndex<NSize>
// TODO: how to fix this?
template <typename... Ys, typename X>
__host__ __device__ constexpr auto operator+=(Tuple<Ys...>& y, const X& x)
{
    static_assert(X::Size() == sizeof...(Ys), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Ys);
    static_for<0, NSize, 1>{}([&](auto i) { y(i) += x[i]; });
    return y;
}

template <typename... Ys, typename X>
__host__ __device__ constexpr auto operator-=(Tuple<Ys...>& y, const X& x)
{
    static_assert(X::Size() == sizeof...(Ys), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Ys);
    static_for<0, NSize, 1>{}([&](auto i) { y(i) -= x[i]; });
    return y;
}

template <typename... Xs, typename Y>
__host__ __device__ constexpr auto operator+(const Tuple<Xs...>& x, const Y& y)
{
    static_assert(Y::Size() == sizeof...(Xs), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Xs);

    Tuple<Xs...> r;
    static_for<0, NSize, 1>{}([&](auto i) { r(i) = x[i] + y[i]; });
    return r;
}

template <typename... Xs, typename Y>
__host__ __device__ constexpr auto operator-(const Tuple<Xs...>& x, const Y& y)
{
    static_assert(Y::Size() == sizeof...(Xs), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Xs);

    Tuple<Xs...> r;
    static_for<0, NSize, 1>{}([&](auto i) { r(i) = x[i] - y[i]; });
    return r;
}

template <typename... Xs, typename Y>
__host__ __device__ constexpr auto operator*(const Tuple<Xs...>& x, const Y& y)
{
    static_assert(Y::Size() == sizeof...(Xs), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Xs);

    Tuple<Xs...> r;
    static_for<0, NSize, 1>{}([&](auto i) { r(i) = x[i] * y[i]; });
    return r;
}

// MultiIndex = index_t * MultiIndex
template <typename... Xs>
__host__ __device__ constexpr auto operator*(index_t a, const Tuple<Xs...>& x)
{
    constexpr index_t NSize = sizeof...(Xs);

    Tuple<Xs...> r;
    static_for<0, NSize, 1>{}([&](auto i) { r(i) = a * x[i]; });
    return r;
}

template <typename... Xs>
__host__ __device__ void print_multi_index(const Tuple<Xs...>& x)
{
    printf("{");
    printf("MultiIndex, ");
    printf("size %d,", index_t{sizeof...(Xs)});
    static_for<0, sizeof...(Xs), 1>{}(
        [&](auto i) { printf("%d ", static_cast<index_t>(x.At(i))); });
    printf("}");
}

} // namespace ck
#endif
