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
