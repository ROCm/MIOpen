#ifndef CK_SEQUENCE_HPP
#define CK_SEQUENCE_HPP

#include "integral_constant.hpp"
#include "functional.hpp"

namespace ck {

template <index_t...>
struct Sequence;

template <class Seq, index_t I>
struct sequence_split;

template <class>
struct sequence_reverse;

template <class>
struct sequence_map_inverse;

template <class>
struct is_valid_sequence_map;

template <index_t I, index_t... Is>
__host__ __device__ constexpr auto sequence_pop_front(Sequence<I, Is...>);

template <class Seq>
__host__ __device__ constexpr auto sequence_pop_back(Seq);

template <index_t... Is>
struct Sequence
{
    using Type      = Sequence;
    using data_type = index_t;

    static constexpr index_t mSize = sizeof...(Is);

    __host__ __device__ static constexpr auto GetSize() { return Number<mSize>{}; }

    __host__ __device__ static constexpr index_t GetImpl(index_t I)
    {
        // the last dummy element is to prevent compiler complain about empty array, when mSize = 0
        const index_t mData[mSize + 1] = {Is..., 0};
        return mData[I];
    }

    template <index_t I>
    __host__ __device__ static constexpr auto Get(Number<I>)
    {
        static_assert(I < mSize, "wrong! I too large");

        return Number<GetImpl(Number<I>{})>{};
    }

    __host__ __device__ static constexpr auto Get(index_t I) { return GetImpl(I); }

    template <index_t I>
    __host__ __device__ constexpr auto operator[](Number<I>) const
    {
        return Get(Number<I>{});
    }

    // make sure I is constepxr if you want a constexpr return type
    __host__ __device__ constexpr index_t operator[](index_t I) const { return GetImpl(I); }

    template <index_t... IRs>
    __host__ __device__ static constexpr auto ReorderGivenNew2Old(Sequence<IRs...> /*new2old*/)
    {
        static_assert(sizeof...(Is) == sizeof...(IRs),
                      "wrong! reorder map should have the same size as Sequence to be rerodered");

        static_assert(is_valid_sequence_map<Sequence<IRs...>>::value, "wrong! invalid reorder map");

        return Sequence<Type::Get(Number<IRs>{})...>{};
    }

    // MapOld2New is Sequence<...>
    template <class MapOld2New>
    __host__ __device__ static constexpr auto ReorderGivenOld2New(MapOld2New)
    {
        static_assert(MapOld2New::GetSize() == GetSize(),
                      "wrong! reorder map should have the same size as Sequence to be rerodered");

        static_assert(is_valid_sequence_map<MapOld2New>::value, "wrong! invalid reorder map");

        return ReorderGivenNew2Old(typename sequence_map_inverse<MapOld2New>::type{});
    }

    __host__ __device__ static constexpr auto Reverse()
    {
        return typename sequence_reverse<Type>::type{};
    }

    __host__ __device__ static constexpr auto Front()
    {
        static_assert(mSize > 0, "wrong!");
        return Get(Number<0>{});
    }

    __host__ __device__ static constexpr auto Back()
    {
        static_assert(mSize > 0, "wrong!");
        return Get(Number<mSize - 1>{});
    }

    __host__ __device__ static constexpr auto PopFront() { return sequence_pop_front(Type{}); }

    __host__ __device__ static constexpr auto PopBack() { return sequence_pop_back(Type{}); }

    template <index_t... Xs>
    __host__ __device__ static constexpr auto PushFront(Sequence<Xs...>)
    {
        return Sequence<Xs..., Is...>{};
    }

    template <index_t... Xs>
    __host__ __device__ static constexpr auto PushFront(Number<Xs>...)
    {
        return Sequence<Xs..., Is...>{};
    }

    template <index_t... Xs>
    __host__ __device__ static constexpr auto PushBack(Sequence<Xs...>)
    {
        return Sequence<Is..., Xs...>{};
    }

    template <index_t... Xs>
    __host__ __device__ static constexpr auto PushBack(Number<Xs>...)
    {
        return Sequence<Is..., Xs...>{};
    }

    template <index_t... Ns>
    __host__ __device__ static constexpr auto Extract(Number<Ns>...)
    {
        return Sequence<Type::Get(Number<Ns>{})...>{};
    }

    template <index_t... Ns>
    __host__ __device__ static constexpr auto Extract(Sequence<Ns...>)
    {
        return Sequence<Type::Get(Number<Ns>{})...>{};
    }

    template <index_t I, index_t X>
    __host__ __device__ static constexpr auto Modify(Number<I>, Number<X>)
    {
        static_assert(I < GetSize(), "wrong!");

        using seq_split          = sequence_split<Type, I>;
        constexpr auto seq_left  = typename seq_split::SeqType0{};
        constexpr auto seq_right = typename seq_split::SeqType1{}.PopFront();

        return seq_left.PushBack(Number<X>{}).PushBack(seq_right);
    }

    template <class F>
    __host__ __device__ static constexpr auto Transform(F f)
    {
        return Sequence<f(Is)...>{};
    }
};

// merge sequence
template <class, class>
struct sequence_merge;

template <index_t... Xs, index_t... Ys>
struct sequence_merge<Sequence<Xs...>, Sequence<Ys...>>
{
    using type = Sequence<Xs..., Ys...>;
};

// generate sequence
template <index_t IBegin, index_t NRemain, class F>
struct sequence_gen_impl
{
    static constexpr index_t NRemainLeft  = NRemain / 2;
    static constexpr index_t NRemainRight = NRemain - NRemainLeft;
    static constexpr index_t IMiddle      = IBegin + NRemainLeft;

    using type =
        typename sequence_merge<typename sequence_gen_impl<IBegin, NRemainLeft, F>::type,
                                typename sequence_gen_impl<IMiddle, NRemainRight, F>::type>::type;
};

template <index_t I, class F>
struct sequence_gen_impl<I, 1, F>
{
    static constexpr index_t Is = F{}(Number<I>{});
    using type                  = Sequence<Is>;
};

template <index_t I, class F>
struct sequence_gen_impl<I, 0, F>
{
    using type = Sequence<>;
};

template <index_t NSize, class F>
struct sequence_gen
{
    using type = typename sequence_gen_impl<0, NSize, F>::type;
};

// arithmetic sequence
template <index_t IBegin, index_t IEnd, index_t Increment>
struct arithmetic_sequence_gen
{
    struct F
    {
        __host__ __device__ constexpr index_t operator()(index_t i) const
        {
            return i * Increment + IBegin;
        }
    };

    using type = typename sequence_gen<(IEnd - IBegin) / Increment, F>::type;
};

// uniform sequence
template <index_t NSize, index_t I>
struct uniform_sequence_gen
{
    struct F
    {
        __host__ __device__ constexpr index_t operator()(index_t) const { return I; }
    };

    using type = typename sequence_gen<NSize, F>::type;
};

// reverse inclusive scan (with init) sequence
template <class, class, index_t>
struct sequence_reverse_inclusive_scan;

template <index_t I, index_t... Is, class Reduce, index_t Init>
struct sequence_reverse_inclusive_scan<Sequence<I, Is...>, Reduce, Init>
{
    using old_scan = typename sequence_reverse_inclusive_scan<Sequence<Is...>, Reduce, Init>::type;

    static constexpr index_t new_reduce = Reduce{}(I, old_scan{}.Front());

    using type = typename sequence_merge<Sequence<new_reduce>, old_scan>::type;
};

template <index_t I, class Reduce, index_t Init>
struct sequence_reverse_inclusive_scan<Sequence<I>, Reduce, Init>
{
    using type = Sequence<Reduce{}(I, Init)>;
};

template <class Reduce, index_t Init>
struct sequence_reverse_inclusive_scan<Sequence<>, Reduce, Init>
{
    using type = Sequence<>;
};

// split sequence
template <class Seq, index_t I>
struct sequence_split
{
    static constexpr index_t NSize = Seq{}.GetSize();

    using range0 = typename arithmetic_sequence_gen<0, I, 1>::type;
    using range1 = typename arithmetic_sequence_gen<I, NSize, 1>::type;

    using SeqType0 = decltype(Seq::Extract(range0{}));
    using SeqType1 = decltype(Seq::Extract(range1{}));
};

// reverse sequence
template <class Seq>
struct sequence_reverse
{
    static constexpr index_t NSize = Seq{}.GetSize();

    using seq_split = sequence_split<Seq, NSize / 2>;
    using type      = typename sequence_merge<
        typename sequence_reverse<typename seq_split::SeqType1>::type,
        typename sequence_reverse<typename seq_split::SeqType0>::type>::type;
};

template <index_t I>
struct sequence_reverse<Sequence<I>>
{
    using type = Sequence<I>;
};

template <index_t I0, index_t I1>
struct sequence_reverse<Sequence<I0, I1>>
{
    using type = Sequence<I1, I0>;
};

template <class Seq>
struct is_valid_sequence_map
{
    // not implemented yet, always return true
    static constexpr integral_constant<bool, true> value = integral_constant<bool, true>{};

    // TODO: add proper check for is_valid, something like:
    // static constexpr bool value =
    //     is_same<typename arithmetic_sequence_gen<0, Seq::GetSize(), 1>::type,
    //             typename sequence_sort<Seq>::SortedSeqType>{};
};

template <class X2Y, class WorkingY2X, index_t XBegin, index_t XRemain>
struct sequence_map_inverse_impl
{
    private:
    static constexpr auto new_y2x =
        WorkingY2X::Modify(X2Y::Get(Number<XBegin>{}), Number<XBegin>{});

    public:
    using type =
        typename sequence_map_inverse_impl<X2Y, decltype(new_y2x), XBegin + 1, XRemain - 1>::type;
};

template <class X2Y, class WorkingY2X, index_t XBegin>
struct sequence_map_inverse_impl<X2Y, WorkingY2X, XBegin, 0>
{
    using type = WorkingY2X;
};

template <class X2Y>
struct sequence_map_inverse
{
    using type =
        typename sequence_map_inverse_impl<X2Y,
                                           typename uniform_sequence_gen<X2Y::GetSize(), 0>::type,
                                           0,
                                           X2Y::GetSize()>::type;
};

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto operator+(Sequence<Xs...>, Sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return Sequence<(Xs + Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto operator-(Sequence<Xs...>, Sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return Sequence<(Xs - Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto operator*(Sequence<Xs...>, Sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return Sequence<(Xs * Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto operator/(Sequence<Xs...>, Sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return Sequence<(Xs / Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto operator%(Sequence<Xs...>, Sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return Sequence<(Xs % Ys)...>{};
}

template <index_t... Xs, index_t Y>
__host__ __device__ constexpr auto operator+(Sequence<Xs...>, Number<Y>)
{
    return Sequence<(Xs + Y)...>{};
}

template <index_t... Xs, index_t Y>
__host__ __device__ constexpr auto operator-(Sequence<Xs...>, Number<Y>)
{
    return Sequence<(Xs - Y)...>{};
}

template <index_t... Xs, index_t Y>
__host__ __device__ constexpr auto operator*(Sequence<Xs...>, Number<Y>)
{
    return Sequence<(Xs * Y)...>{};
}

template <index_t... Xs, index_t Y>
__host__ __device__ constexpr auto operator/(Sequence<Xs...>, Number<Y>)
{
    return Sequence<(Xs / Y)...>{};
}

template <index_t... Xs, index_t Y>
__host__ __device__ constexpr auto operator%(Sequence<Xs...>, Number<Y>)
{
    return Sequence<(Xs % Y)...>{};
}

template <index_t Y, index_t... Xs>
__host__ __device__ constexpr auto operator+(Number<Y>, Sequence<Xs...>)
{
    return Sequence<(Y + Xs)...>{};
}

template <index_t Y, index_t... Xs>
__host__ __device__ constexpr auto operator-(Number<Y>, Sequence<Xs...>)
{
    constexpr auto seq_x = Sequence<Xs...>{};

    return Sequence<(Y - Xs)...>{};
}

template <index_t Y, index_t... Xs>
__host__ __device__ constexpr auto operator*(Number<Y>, Sequence<Xs...>)
{
    return Sequence<(Y * Xs)...>{};
}

template <index_t Y, index_t... Xs>
__host__ __device__ constexpr auto operator/(Number<Y>, Sequence<Xs...>)
{
    return Sequence<(Y / Xs)...>{};
}

template <index_t Y, index_t... Xs>
__host__ __device__ constexpr auto operator%(Number<Y>, Sequence<Xs...>)
{
    return Sequence<(Y % Xs)...>{};
}

template <index_t I, index_t... Is>
__host__ __device__ constexpr auto sequence_pop_front(Sequence<I, Is...>)
{
    return Sequence<Is...>{};
}

template <class Seq>
__host__ __device__ constexpr auto sequence_pop_back(Seq)
{
    static_assert(Seq::GetSize() > 0, "wrong! cannot pop an empty Sequence!");
    return sequence_pop_front(Seq::Reverse()).Reverse();
}

template <class F, index_t... Xs>
__host__ __device__ constexpr auto transform_sequences(F f, Sequence<Xs...>)
{
    return Sequence<f(Xs)...>{};
}

template <class F, index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto transform_sequences(F f, Sequence<Xs...>, Sequence<Ys...>)
{
    static_assert(Sequence<Xs...>::mSize == Sequence<Ys...>::mSize, "Dim not the same");

    return Sequence<f(Xs, Ys)...>{};
}

template <class F, index_t... Xs, index_t... Ys, index_t... Zs>
__host__ __device__ constexpr auto
transform_sequences(F f, Sequence<Xs...>, Sequence<Ys...>, Sequence<Zs...>)
{
    static_assert(Sequence<Xs...>::mSize == Sequence<Ys...>::mSize &&
                      Sequence<Xs...>::mSize == Sequence<Zs...>::mSize,
                  "Dim not the same");

    return Sequence<f(Xs, Ys, Zs)...>{};
}

template <class Seq, class Reduce, index_t Init>
__host__ __device__ constexpr auto reverse_inclusive_scan_sequence(Seq, Reduce, Number<Init>)
{
    return typename sequence_reverse_inclusive_scan<Seq, Reduce, Init>::type{};
}

template <class Seq, class Reduce, index_t Init>
__host__ __device__ constexpr auto inclusive_scan_sequence(Seq, Reduce, Number<Init>)
{
    return reverse_inclusive_scan_sequence(Seq{}.Reverse(), Reduce{}, Number<Init>{}).Reverse();
}

template <index_t... Xs>
__host__ __device__ void print_Sequence(const char* s, Sequence<Xs...>)
{
    constexpr index_t nsize = Sequence<Xs...>::GetSize();

    static_assert(nsize <= 10, "wrong!");

    static_if<nsize == 0>{}([&](auto) { printf("%s size %u, {}\n", s, nsize, Xs...); });

    static_if<nsize == 1>{}([&](auto) { printf("%s size %u, {%u}\n", s, nsize, Xs...); });

    static_if<nsize == 2>{}([&](auto) { printf("%s size %u, {%u %u}\n", s, nsize, Xs...); });

    static_if<nsize == 3>{}([&](auto) { printf("%s size %u, {%u %u %u}\n", s, nsize, Xs...); });

    static_if<nsize == 4>{}([&](auto) { printf("%s size %u, {%u %u %u %u}\n", s, nsize, Xs...); });

    static_if<nsize == 5>{}(
        [&](auto) { printf("%s size %u, {%u %u %u %u %u}\n", s, nsize, Xs...); });

    static_if<nsize == 6>{}(
        [&](auto) { printf("%s size %u, {%u %u %u %u %u %u}\n", s, nsize, Xs...); });

    static_if<nsize == 7>{}(
        [&](auto) { printf("%s size %u, {%u %u %u %u %u %u %u}\n", s, nsize, Xs...); });

    static_if<nsize == 8>{}(
        [&](auto) { printf("%s size %u, {%u %u %u %u %u %u %u %u}\n", s, nsize, Xs...); });

    static_if<nsize == 9>{}(
        [&](auto) { printf("%s size %u, {%u %u %u %u %u %u %u %u %u}\n", s, nsize, Xs...); });

    static_if<nsize == 10>{}(
        [&](auto) { printf("%s size %u, {%u %u %u %u %u %u %u %u %u %u}\n", s, nsize, Xs...); });
}

} // namespace ck
#endif
