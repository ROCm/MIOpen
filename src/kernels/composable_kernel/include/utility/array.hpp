#ifndef CK_ARRAY_HPP
#define CK_ARRAY_HPP

#include "sequence.hpp"
#include "functional2.hpp"

namespace ck {

template <typename TData, index_t NSize>
struct Array
{
    using type      = Array<TData, NSize>;
    using data_type = TData;

    index_t mData[NSize];

    __host__ __device__ explicit constexpr Array() {}

    template <typename X, typename... Xs>
    __host__ __device__ constexpr Array(X x, Xs... xs)
        : mData{static_cast<TData>(x), static_cast<TData>(xs)...}
    {
        static_assert(sizeof...(Xs) + 1 == NSize, "wrong! size");
    }

    __host__ __device__ static constexpr index_t Size() { return NSize; }

    // TODO: remove
    __host__ __device__ static constexpr index_t GetSize() { return Size(); }

    template <index_t I>
    __host__ __device__ constexpr const TData& At(Number<I>) const
    {
        static_assert(I < NSize, "wrong!");

        return mData[I];
    }

    template <index_t I>
    __host__ __device__ constexpr TData& At(Number<I>)
    {
        static_assert(I < NSize, "wrong!");

        return mData[I];
    }

    __host__ __device__ constexpr const TData& At(index_t i) const { return mData[i]; }

    __host__ __device__ constexpr TData& At(index_t i) { return mData[i]; }

    template <typename I>
    __host__ __device__ constexpr const TData& operator[](I i) const
    {
        return At(i);
    }

    template <typename I>
    __host__ __device__ constexpr TData& operator()(I i)
    {
        return At(i);
    }

    template <typename T>
    __host__ __device__ constexpr type& operator=(const T& x)
    {
        static_for<0, Size(), 1>{}([&](auto i) { operator()(i) = x[i]; });

        return *this;
    }

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
            new_array(Number<I>{}) = old_array[I];
        }
    };

    __host__ __device__ constexpr auto PushBack(TData x) const
    {
        Array<TData, NSize + 1> new_array;

        static_for<0, NSize, 1>{}(lambda_PushBack(*this, new_array));

        new_array(Number<NSize>{}) = x;

        return new_array;
    }
};

// Arr: Array
// Picks: Sequence<...>
template <typename Arr, typename Picks>
struct ArrayElementPicker
{
    using type      = ArrayElementPicker;
    using data_type = typename Arr::data_type;

    __host__ __device__ constexpr ArrayElementPicker() = delete;

    __host__ __device__ explicit constexpr ArrayElementPicker(Arr& array) : mArray{array}
    {
        constexpr index_t imax = reduce_on_sequence(Picks{}, math::maxer<index_t>{}, Number<0>{});

        static_assert(imax < Arr::Size(), "wrong! exceeding # array element");
    }

    __host__ __device__ static constexpr auto Size() { return Picks::Size(); }

    template <index_t I>
    __host__ __device__ constexpr const data_type& At(Number<I>) const
    {
        static_assert(I < Size(), "wrong!");

        constexpr auto IP = Picks{}[I];
        return mArray[IP];
    }

    template <index_t I>
    __host__ __device__ constexpr data_type& At(Number<I>)
    {
        static_assert(I < Size(), "wrong!");

        constexpr auto IP = Picks{}[I];
        return mArray(IP);
    }

    template <typename I>
    __host__ __device__ constexpr const data_type& operator[](I i) const
    {
        return At(i);
    }

    template <typename I>
    __host__ __device__ constexpr data_type& operator()(I i)
    {
        return At(i);
    }

    template <typename T>
    __host__ __device__ constexpr type& operator=(const T& a)
    {
        static_for<0, Size(), 1>{}([&](auto i) { operator()(i) = a[i]; });

        return *this;
    }

    Arr& mArray;
};

template <typename Arr, typename Picks>
__host__ __device__ constexpr auto pick_array_element(Arr& a, Picks)
{
    return ArrayElementPicker<Arr, Picks>(a);
}

template <typename T>
__host__ __device__ constexpr auto to_array(const T& x)
{
    Array<typename T::data_type, T::Size()> y;

    static_for<0, T::Size(), 1>{}([&](auto i) { y.At(i) = x.At(i); });

    return y;
}

// TODO: remove this
template <index_t... Is>
__host__ __device__ constexpr auto sequence2array(Sequence<Is...>)
{
    return Array<index_t, sizeof...(Is)>{Is...};
}

template <typename TData, index_t NSize>
__host__ __device__ constexpr auto make_zero_array()
{
    constexpr auto zero_sequence = typename uniform_sequence_gen<NSize, 0>::type{};
    constexpr auto zero_array    = sequence2array(zero_sequence);
    return zero_array;
}

template <typename TData, index_t NSize, index_t... IRs>
__host__ __device__ constexpr auto reorder_array_given_new2old(const Array<TData, NSize>& old_array,
                                                               Sequence<IRs...> /*new2old*/)
{
    static_assert(NSize == sizeof...(IRs), "NSize not consistent");

    static_assert(is_valid_sequence_map<Sequence<IRs...>>{}, "wrong! invalid reorder map");

    return Array<TData, NSize>{old_array[IRs]...};
}

template <typename TData, index_t NSize, typename MapOld2New>
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

        constexpr index_t INewDim = MapOld2New::At(Number<IOldDim>{});

        new_array(Number<INewDim>{}) = old_data;
    }
};

template <typename TData, index_t NSize, index_t... IRs>
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

template <typename TData, index_t NSize, typename ExtractSeq>
__host__ __device__ constexpr auto extract_array(const Array<TData, NSize>& old_array, ExtractSeq)
{
    Array<TData, ExtractSeq::GetSize()> new_array;

    constexpr index_t new_size = ExtractSeq::GetSize();

    static_assert(new_size <= NSize, "wrong! too many extract");

    static_for<0, new_size, 1>{}([&](auto I) { new_array(I) = old_array[ExtractSeq::At(I)]; });

    return new_array;
}

// emulate constepxr lambda for array
template <typename F, typename X, typename Y, typename Z>
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
        z(IDim)             = f(x[IDim], y[IDim]);
    }
};

// Array = Array + Array
template <typename TData, index_t NSize>
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
template <typename TData, index_t NSize>
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
template <typename TData, index_t NSize>
__host__ __device__ constexpr auto operator+=(Array<TData, NSize>& a, Array<TData, NSize> b)
{
    a = a + b;
    return a;
}

// Array -= Array
template <typename TData, index_t NSize>
__host__ __device__ constexpr auto operator-=(Array<TData, NSize>& a, Array<TData, NSize> b)
{
    a = a - b;
    return a;
}
// Array = Array + Sequence
template <typename TData, index_t NSize, index_t... Is>
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
template <typename TData, index_t NSize, index_t... Is>
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
template <typename TData, index_t NSize, index_t... Is>
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
template <typename TData, index_t NSize, index_t... Is>
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

// Array = Array * TData
template <typename TData, index_t NSize>
__host__ __device__ constexpr auto operator*(TData v, Array<TData, NSize> a)
{
    Array<TData, NSize> result;

    for(index_t i = 0; i < NSize; ++i)
    {
        result(i) = a[i] * v;
    }

    return result;
}

template <typename TData, index_t NSize, typename Reduce>
__host__ __device__ constexpr TData
accumulate_on_array(const Array<TData, NSize>& a, Reduce f, TData init)
{
    TData result = init;

    static_assert(NSize > 0, "wrong");

    static_for<0, NSize, 1>{}([&](auto I) { result = f(result, a[I]); });

    return result;
}

} // namespace ck
#endif
