#ifndef CK_ARRAY_HPP
#define CK_ARRAY_HPP

#include "Sequence.hpp"
#include "functional2.hpp"

namespace ck {

template <class TData, index_t NSize>
struct Array
{
    using Type      = Array<TData, NSize>;
    using data_type = TData;

    static constexpr index_t nSize = NSize;

    index_t mData[nSize];

    template <class... Xs>
    __host__ __device__ constexpr Array(Xs... xs) : mData{static_cast<TData>(xs)...}
    {
    }

    __host__ __device__ static constexpr index_t GetSize() { return NSize; }

    template <index_t I>
    __host__ __device__ constexpr TData operator[](Number<I>) const
    {
        return mData[I];
    }

    __host__ __device__ constexpr TData operator[](index_t i) const { return mData[i]; }

    template <index_t I>
    __host__ __device__ TData& operator()(Number<I>)
    {
        return mData[I];
    }

    __host__ __device__ TData& operator()(index_t i) { return mData[i]; }

    template <index_t I>
    __host__ __device__ constexpr void Set(Number<I>, TData x)
    {
        static_assert(I < NSize, "wrong!");

        mData[I] = x;
    }

    __host__ __device__ constexpr void Set(index_t I, TData x) { mData[I] = x; }

    struct lambda_PushBack // emulate constexpr lambda
    {
        const Array<TData, NSize>& old_array;
        Array<TData, NSize + 1>& new_array;

        __host__ __device__ constexpr lambda_PushBack(const Array<TData, NSize>& old_array_,
                                                      Array<TData, NSize + 1>& new_array_)
            : old_array(old_array_), new_array(new_array_)
        {
        }

        template <index_t I>
        __host__ __device__ constexpr void operator()(Number<I>) const
        {
            new_array.Set(Number<I>{}, old_array[I]);
        }
    };

    __host__ __device__ constexpr auto PushBack(TData x) const
    {
        Array<TData, NSize + 1> new_array;

        static_for<0, NSize, 1>{}(lambda_PushBack(*this, new_array));

        new_array.Set(Number<NSize>{}, x);

        return new_array;
    }
};

template <index_t... Is>
__host__ __device__ constexpr auto sequence2array(Sequence<Is...>)
{
    return Array<index_t, sizeof...(Is)>{Is...};
}

template <class TData, index_t NSize>
__host__ __device__ constexpr auto make_zero_array()
{
    constexpr auto zero_sequence = typename uniform_sequence_gen<NSize, 0>::type{};
    constexpr auto zero_array    = sequence2array(zero_sequence);
    return zero_array;
}

template <class TData, index_t NSize, index_t... IRs>
__host__ __device__ constexpr auto reorder_array_given_new2old(const Array<TData, NSize>& old_array,
                                                               Sequence<IRs...> /*new2old*/)
{
    static_assert(NSize == sizeof...(IRs), "NSize not consistent");

    static_assert(is_valid_sequence_map<Sequence<IRs...>>{}, "wrong! invalid reorder map");

    return Array<TData, NSize>{old_array[IRs]...};
}

template <class TData, index_t NSize, class MapOld2New>
struct lambda_reorder_array_given_old2new
{
    const Array<TData, NSize>& old_array;
    Array<TData, NSize>& new_array;

    __host__ __device__ constexpr lambda_reorder_array_given_old2new(
        const Array<TData, NSize>& old_array_, Array<TData, NSize>& new_array_)
        : old_array(old_array_), new_array(new_array_)
    {
    }

    template <index_t IOldDim>
    __host__ __device__ constexpr void operator()(Number<IOldDim>) const
    {
        TData old_data = old_array[IOldDim];

        constexpr index_t INewDim = MapOld2New::Get(Number<IOldDim>{});

        new_array.Set(Number<INewDim>{}, old_data);
    }
};

template <class TData, index_t NSize, index_t... IRs>
__host__ __device__ constexpr auto reorder_array_given_old2new(const Array<TData, NSize>& old_array,
                                                               Sequence<IRs...> /*old2new*/)
{
    Array<TData, NSize> new_array;

    static_assert(NSize == sizeof...(IRs), "NSize not consistent");

    static_assert(is_valid_sequence_map<Sequence<IRs...>>::value, "wrong! invalid reorder map");

    static_for<0, NSize, 1>{}(
        lambda_reorder_array_given_old2new<TData, NSize, Sequence<IRs...>>(old_array, new_array));

    return new_array;
}

template <class TData, index_t NSize, class ExtractSeq>
__host__ __device__ constexpr auto extract_array(const Array<TData, NSize>& old_array, ExtractSeq)
{
    Array<TData, ExtractSeq::GetSize()> new_array;

    constexpr index_t new_size = ExtractSeq::GetSize();

    static_assert(new_size <= NSize, "wrong! too many extract");

    static_for<0, new_size, 1>{}([&](auto I) { new_array(I) = old_array[ExtractSeq::Get(I)]; });

    return new_array;
}

template <class F, class X, class Y, class Z> // emulate constepxr lambda for array math
struct lambda_array_math
{
    const F& f;
    const X& x;
    const Y& y;
    Z& z;

    __host__ __device__ constexpr lambda_array_math(const F& f_, const X& x_, const Y& y_, Z& z_)
        : f(f_), x(x_), y(y_), z(z_)
    {
    }

    template <index_t IDim_>
    __host__ __device__ constexpr void operator()(Number<IDim_>) const
    {
        constexpr auto IDim = Number<IDim_>{};

        z.Set(IDim, f(x[IDim], y[IDim]));
    }
};

// Array = Array + Array
template <class TData, index_t NSize>
__host__ __device__ constexpr auto operator+(Array<TData, NSize> a, Array<TData, NSize> b)
{
    Array<TData, NSize> result;

    auto f = math::plus<index_t>{};

    static_for<0, NSize, 1>{}(
        lambda_array_math<decltype(f), decltype(a), decltype(b), decltype(result)>(
            f, a, b, result));

    return result;
}

// Array = Array - Array
template <class TData, index_t NSize>
__host__ __device__ constexpr auto operator-(Array<TData, NSize> a, Array<TData, NSize> b)
{
    Array<TData, NSize> result;

    auto f = math::minus<index_t>{};

    static_for<0, NSize, 1>{}(
        lambda_array_math<decltype(f), decltype(a), decltype(b), decltype(result)>(
            f, a, b, result));

    return result;
}

// Array += Array
template <class TData, index_t NSize>
__host__ __device__ constexpr auto operator+=(Array<TData, NSize>& a, Array<TData, NSize> b)
{
    a = a + b;
    return a;
}

// Array -= Array
template <class TData, index_t NSize>
__host__ __device__ constexpr auto operator-=(Array<TData, NSize>& a, Array<TData, NSize> b)
{
    a = a - b;
    return a;
}
// Array = Array + Sequence
template <class TData, index_t NSize, index_t... Is>
__host__ __device__ constexpr auto operator+(Array<TData, NSize> a, Sequence<Is...> b)
{
    static_assert(sizeof...(Is) == NSize, "wrong! size not the same");

    Array<TData, NSize> result;

    auto f = math::plus<index_t>{};

    static_for<0, NSize, 1>{}(
        lambda_array_math<decltype(f), decltype(a), decltype(b), decltype(result)>(
            f, a, b, result));

    return result;
}

// Array = Array - Sequence
template <class TData, index_t NSize, index_t... Is>
__host__ __device__ constexpr auto operator-(Array<TData, NSize> a, Sequence<Is...> b)
{
    static_assert(sizeof...(Is) == NSize, "wrong! size not the same");

    Array<TData, NSize> result;

    auto f = math::minus<index_t>{};

    static_for<0, NSize, 1>{}(
        lambda_array_math<decltype(f), decltype(a), decltype(b), decltype(result)>(
            f, a, b, result));

    return result;
}

// Array = Array * Sequence
template <class TData, index_t NSize, index_t... Is>
__host__ __device__ constexpr auto operator*(Array<TData, NSize> a, Sequence<Is...> b)
{
    static_assert(sizeof...(Is) == NSize, "wrong! size not the same");

    Array<TData, NSize> result;

    auto f = math::multiplies<index_t>{};

    static_for<0, NSize, 1>{}(
        lambda_array_math<decltype(f), decltype(a), decltype(b), decltype(result)>(
            f, a, b, result));

    return result;
}

// Array = Sequence - Array
template <class TData, index_t NSize, index_t... Is>
__host__ __device__ constexpr auto operator-(Sequence<Is...> a, Array<TData, NSize> b)
{
    static_assert(sizeof...(Is) == NSize, "wrong! size not the same");

    Array<TData, NSize> result;

    auto f = math::minus<index_t>{};

    static_for<0, NSize, 1>{}(
        lambda_array_math<decltype(f), decltype(a), decltype(b), decltype(result)>(
            f, a, b, result));

    return result;
}

template <class TData, index_t NSize, class Reduce>
__host__ __device__ constexpr TData
accumulate_on_array(const Array<TData, NSize>& a, Reduce f, TData init)
{
    TData result = init;

    static_assert(NSize > 0, "wrong");

    static_for<0, NSize, 1>{}([&](auto I) { result = f(result, a[I]); });

    return result;
}

template <class T, index_t NSize>
__host__ __device__ void print_Array(const char* s, Array<T, NSize> a)
{
    constexpr index_t nsize = a.GetSize();

    static_assert(nsize > 0 && nsize <= 10, "wrong!");

    static_if<nsize == 1>{}([&](auto) { printf("%s size %u, {%u}\n", s, nsize, a[0]); });

    static_if<nsize == 2>{}([&](auto) { printf("%s size %u, {%u %u}\n", s, nsize, a[0], a[1]); });

    static_if<nsize == 3>{}(
        [&](auto) { printf("%s size %u, {%u %u %u}\n", s, nsize, a[0], a[1], a[2]); });

    static_if<nsize == 4>{}(
        [&](auto) { printf("%s size %u, {%u %u %u %u}\n", s, nsize, a[0], a[1], a[2], a[3]); });

    static_if<nsize == 5>{}([&](auto) {
        printf("%s size %u, {%u %u %u %u %u}\n", s, nsize, a[0], a[1], a[2], a[3], a[4]);
    });

    static_if<nsize == 6>{}([&](auto) {
        printf("%s size %u, {%u %u %u %u %u %u}\n", s, nsize, a[0], a[1], a[2], a[3], a[4], a[5]);
    });

    static_if<nsize == 7>{}([&](auto) {
        printf("%s size %u, {%u %u %u %u %u %u %u}\n",
               s,
               nsize,
               a[0],
               a[1],
               a[2],
               a[3],
               a[4],
               a[5],
               a[6]);
    });

    static_if<nsize == 8>{}([&](auto) {
        printf("%s size %u, {%u %u %u %u %u %u %u %u}\n",
               s,
               nsize,
               a[0],
               a[1],
               a[2],
               a[3],
               a[4],
               a[5],
               a[6],
               a[7]);
    });

    static_if<nsize == 9>{}([&](auto) {
        printf("%s size %u, {%u %u %u %u %u %u %u %u %u}\n",
               s,
               nsize,
               a[0],
               a[1],
               a[2],
               a[3],
               a[4],
               a[5],
               a[6],
               a[7],
               a[8]);
    });

    static_if<nsize == 10>{}([&](auto) {
        printf("%s size %u, {%u %u %u %u %u %u %u %u %u %u}\n",
               s,
               nsize,
               a[0],
               a[1],
               a[2],
               a[3],
               a[4],
               a[5],
               a[6],
               a[7],
               a[8],
               a[9]);
    });
}

} // namespace ck
#endif
