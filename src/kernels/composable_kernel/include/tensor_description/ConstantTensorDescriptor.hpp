#ifndef CK_CONSTANT_TENSOR_DESCRIPTOR_HPP
#define CK_CONSTANT_TENSOR_DESCRIPTOR_HPP

#include "common_header.hpp"

namespace ck {

template <class Lengths>
__host__ __device__ constexpr auto calculate_tensor_strides_packed(Lengths)
{
    return reverse_inclusive_scan_sequence(
               Lengths{}.PopFront(), math::multiplies<index_t>{}, Number<1>{})
        .PushBack(Number<1>{});
}

template <class Lengths, index_t Align>
__host__ __device__ constexpr auto calculate_tensor_strides_aligned(Lengths, Number<Align>)
{
    constexpr index_t L_back_align =
        Align * math::integer_divide_ceiler<index_t>{}(Lengths{}.Back(), Align);

    return calculate_tensor_strides_packed(
        Lengths{}.Modify(Number<Lengths{}.GetSize() - 1>{}, Number<L_back_align>{}));
}

template <class Lengths, class Strides>
struct ConstantTensorDescriptor
{
    using Type = ConstantTensorDescriptor;

    static constexpr index_t nDim = Lengths::GetSize();

    __host__ __device__ constexpr ConstantTensorDescriptor()
    {
        static_assert(Lengths::GetSize() == Strides::GetSize(), "nDim not consistent");
    }

    __host__ __device__ static constexpr auto GetOriginalTensorDescriptor() { return Type{}; }

    template <index_t IDim>
    __host__ __device__ static constexpr auto GetContainedOriginalDimensions(Number<IDim>)
    {
        return Sequence<IDim>{};
    }

    __host__ __device__ static constexpr index_t GetNumOfDimension() { return nDim; }

    __host__ __device__ static constexpr auto GetLengths() { return Lengths{}; }

    __host__ __device__ static constexpr auto GetStrides() { return Strides{}; }

    template <index_t I>
    __host__ __device__ static constexpr index_t GetLength(Number<I>)
    {
        return Lengths::Get(Number<I>{});
    }

    template <index_t I>
    __host__ __device__ static constexpr index_t GetStride(Number<I>)
    {
        return Strides::Get(Number<I>{});
    }

    struct lambda_AreDimensionsContinuous
    {
        bool& is_continuous;

        __host__ __device__ constexpr lambda_AreDimensionsContinuous(bool& is_continuous_)
            : is_continuous(is_continuous_)
        {
        }

        template <index_t IDim_>
        __host__ __device__ constexpr void operator()(Number<IDim_>) const
        {
            constexpr auto IDim    = Number<IDim_>{};
            constexpr auto IDim_p1 = Number<IDim_ + 1>{};

            is_continuous =
                is_continuous && (GetStride(IDim) >= GetStride(IDim_p1) &&
                                  GetStride(IDim) == GetStride(IDim_p1) * GetLength(IDim_p1));
        }
    };

    __host__ __device__ static constexpr bool AreDimensionsContinuous()
    {
        bool is_continuous = true;

        static_for<0, nDim - 1, 1>{}(lambda_AreDimensionsContinuous(is_continuous));

        return is_continuous;
    }

    __host__ __device__ static constexpr bool IsPackedTensor()
    {
        return AreDimensionsContinuous() && GetStride(Number<nDim - 1>{}) == 1;
    }

    template <class T>
    __host__ __device__ static constexpr bool ContainMultipleOriginalDimensions(T)
    {
        return false;
    }

    __host__ __device__ static constexpr index_t GetElementSize()
    {
        return accumulate_on_sequence(Lengths{}, math::multiplies<index_t>{}, Number<1>{});
    }

    __host__ __device__ static constexpr index_t GetElementSpace()
    {
        constexpr index_t element_space_unaligned = accumulate_on_sequence(
            (GetLengths() - Number<1>{}) * GetStrides(), math::plus<index_t>{}, Number<1>{});

        return element_space_unaligned;
    }

    // emulate constexpr lambda
    template <index_t NSize>
    struct lambda_GetOffsetFromMultiIndex
    {
        Array<index_t, NSize>& multi_id;
        index_t& offset;

        __host__
            __device__ constexpr lambda_GetOffsetFromMultiIndex(Array<index_t, NSize>& multi_id_,
                                                                index_t& offset_)
            : multi_id(multi_id_), offset(offset_)
        {
        }

        template <class X>
        __host__ __device__ constexpr void operator()(X IDim) const
        {
            offset += multi_id[IDim] * Type::GetStride(IDim);
        }
    };

    template <index_t NSize>
    __host__ __device__ static constexpr index_t
    GetOffsetFromMultiIndex(Array<index_t, NSize> multi_id)
    {
        static_assert(NSize == nDim, "wrong! Dimension not consistent");

        index_t offset = 0;

        static_for<0, nDim, 1>{}(lambda_GetOffsetFromMultiIndex<NSize>(multi_id, offset));

        return offset;
    }

    template <class... Is>
    __host__ __device__ static constexpr index_t GetOffsetFromMultiIndex(Is... is)
    {
        return GetOffsetFromMultiIndex(Array<index_t, sizeof...(Is)>{is...});
    }

    template <index_t... Is>
    __host__ __device__ static constexpr index_t GetOffsetFromMultiIndex(Sequence<Is...>)
    {
        static_assert(sizeof...(Is) == nDim, "wrong! Dimension not consistent");

        constexpr auto multi_id = Sequence<Is...>{};

        return accumulate_on_sequence(multi_id * GetStrides(), math::plus<index_t>{}, Number<0>{});
    }

    // emulate constexpr lambda
    template <class PackedStrides>
    struct lambda_GetMultiIndexFrom1dIndex
    {
        index_t& id;
        Array<index_t, nDim>& multi_id;

        __host__
            __device__ constexpr lambda_GetMultiIndexFrom1dIndex(index_t& id_,
                                                                 Array<index_t, nDim>& multi_id_)
            : id(id_), multi_id(multi_id_)
        {
        }

        template <class IDim_>
        __host__ __device__ constexpr void operator()(IDim_) const
        {
            constexpr auto IDim      = IDim_{};
            constexpr index_t stride = PackedStrides::Get(IDim);
            multi_id.Set(IDim, id / stride);
            id -= multi_id[IDim] * stride;
        }
    };

    __host__ __device__ static constexpr Array<index_t, nDim> GetMultiIndexFrom1dIndex(index_t id)
    {
        Array<index_t, nDim> multi_id;

        using PackedStrides = decltype(calculate_tensor_strides_packed(GetLengths()));

        // calculate index in each of the dimensions in the order of their dimension
        static_for<0, nDim - 1, 1>{}(lambda_GetMultiIndexFrom1dIndex<PackedStrides>(id, multi_id));

        multi_id.Set(Number<nDim - 1>{}, id / PackedStrides::Get(Number<nDim - 1>{}));

        return multi_id;
    }

    __host__ __device__ static constexpr auto
    GetOriginalMultiIndexFromMultiIndex(Array<index_t, nDim> multi_id)
    {
        return multi_id;
    }

    // This function doesn't do carry check on the highest dimension for positive stepping (or
    // borrow check on the lowest dimension for negative stepping) , for performance reason. It is
    // the user's responsibility to make sure the result "new_mutli_id" is not out-of-bound on the
    // highest dimension for positive stepping (or on the lowest dimension for negative stepping)
    template <bool PositiveDirection>
    __host__ __device__ static Array<index_t, nDim>
    UpdateMultiIndexGivenStepSizeOf1dIndex(Array<index_t, nDim> old_multi_id,
                                           index_t step_size_of_1d_index,
                                           integral_constant<bool, PositiveDirection>)
    {
        Array<index_t, nDim> new_multi_id;

        const auto step_sizes = GetMultiIndexFrom1dIndex(step_size_of_1d_index);

        static_if<PositiveDirection>{}([&](auto) {
            new_multi_id = old_multi_id + step_sizes;

            bool carry = false;

            // do carry check in reversed order, starting from lowest dimension
            // don't check the highest dimension
            static_for<0, nDim, 1>{}([&](auto IDimReverse) {
                constexpr index_t idim = nDim - 1 - IDimReverse;
                constexpr auto IDim    = Number<idim>{};

                if(carry)
                {
                    ++new_multi_id(idim);
                }

                carry = false;

                if(new_multi_id[idim] >= GetLength(IDim))
                {
                    new_multi_id(idim) -= GetLength(IDim);
                    carry = true;
                }
            });
        }).Else([&](auto) {
            // shift up multi-id to avoid unsigned integer underflow during intermediate
            // calculations. After the shift, should have new_multi_id[...] >= 1
            new_multi_id = old_multi_id + (GetLengths() - step_sizes);

            bool borrow = false;

            // do borrow check in reversed order, starting from lowest dimension
            // don't check the highest dimension
            static_for<0, nDim, 1>{}([&](auto IDimReverse) {
                constexpr index_t idim = nDim - 1 - IDimReverse;
                constexpr auto IDim    = Number<idim>{};

                if(borrow)
                {
                    --new_multi_id(idim);
                }

                borrow = false;

                if(new_multi_id[idim] < GetLength(IDim))
                {
                    new_multi_id(idim) += GetLength(IDim);
                    borrow = true;
                }
            });

            // shift back down multi-id
            // here, should have new_multi_id[...] >= GetLengths()
            new_multi_id = new_multi_id - GetLengths();
        });

        return new_multi_id;
    }

    template <index_t... IDims>
    __host__ __device__ static constexpr auto Extract(Number<IDims>... extract_dims)
    {
        static_assert(sizeof...(IDims) <= GetNumOfDimension(),
                      "wrong! too many number of dimensions to be extracted");

        using extract_lengths = decltype(Lengths::Extract(extract_dims...));
        using extract_strides = decltype(Strides::Extract(extract_dims...));

        return ConstantTensorDescriptor<extract_lengths, extract_strides>{};
    }

    template <index_t... IDims>
    __host__ __device__ static constexpr auto Extract(Sequence<IDims...>)
    {
        return Extract(Number<IDims>{}...);
    }

    template <class... Ts>
    __host__ __device__ static constexpr auto Embed(ConstantTensorDescriptor<Ts...>)
    {
        using leaf_tensor = ConstantTensorDescriptor<Ts...>;

        return ConstantTensorDescriptor<decltype(GetLengths().PushBack(leaf_tensor::GetLengths())),
                                        decltype(
                                            GetStrides().PushBack(leaf_tensor::GetStrides()))>{};
    }

    template <index_t IDim, index_t SliceLen>
    __host__ __device__ static constexpr auto Slice(Number<IDim>, Number<SliceLen>)
    {
        using slice_lengths = decltype(Lengths{}.Modify(Number<IDim>{}, Number<SliceLen>{}));

        return ConstantTensorDescriptor<slice_lengths, Strides>{};
    }

    template <index_t IDim, index_t SliceLength, index_t SliceStride>
    __host__ __device__ static constexpr auto
        StridedSlice(Number<IDim>, Number<SliceLength>, Number<SliceStride>)
    {
        constexpr index_t new_stride = Strides::Get(Number<IDim>{}) * SliceStride;

        using new_lengths = decltype(Lengths::Modify(Number<IDim>{}, Number<SliceLength>{}));
        using new_strides = decltype(Strides::Modify(Number<IDim>{}, Number<new_stride>{}));

        return ConstantTensorDescriptor<new_lengths, new_strides>{};
    }

    template <index_t IDim, index_t... FoldIntervals>
    __host__ __device__ static constexpr auto Fold(Number<IDim>, Number<FoldIntervals>...)
    {
        constexpr auto fold_intervals = Sequence<FoldIntervals...>{};

        constexpr index_t fold_intervals_product =
            accumulate_on_sequence(fold_intervals, math::multiplies<index_t>{}, Number<1>{});

        constexpr auto unfold_length = GetLength(Number<IDim>{});
        constexpr auto unfold_stride = GetStride(Number<IDim>{});

        // length of the dimension to be folded needs to be dividable by fold_interval_product,
        // otherwise, folding is invalid
        static_assert(unfold_length % fold_intervals_product == 0,
                      "wrong! length on the dimension to be folded cannot be evenly divided!");

        // folded lengths
        constexpr auto fold_lengths =
            Sequence<unfold_length / fold_intervals_product>{}.PushBack(fold_intervals);

        // folded strides
        constexpr auto fold_strides =
            Number<unfold_stride>{} *
            reverse_inclusive_scan_sequence(
                fold_intervals.PushBack(Number<1>{}), math::multiplies<index_t>{}, Number<1>{});

        // left and right
        constexpr auto left = typename arithmetic_sequence_gen<0, IDim, 1>::type{};
        constexpr auto right =
            typename arithmetic_sequence_gen<IDim + 1, GetNumOfDimension(), 1>::type{};

        constexpr auto new_lengths =
            GetLengths().Extract(left).PushBack(fold_lengths).PushBack(GetLengths().Extract(right));
        constexpr auto new_strides =
            GetStrides().Extract(left).PushBack(fold_strides).PushBack(GetStrides().Extract(right));

        return ConstantTensorDescriptor<decltype(new_lengths), decltype(new_strides)>{};
    }

    // this function unfold dimension [FirstUnfoldDim, ..., LastUnfoldDim] into 1 dimension
    template <index_t FirstUnfoldDim, index_t LastUnfoldDim>
    __host__ __device__ static constexpr auto Unfold(Number<FirstUnfoldDim>, Number<LastUnfoldDim>)
    {
        static_assert(FirstUnfoldDim >= 0 && LastUnfoldDim < nDim &&
                          FirstUnfoldDim <= LastUnfoldDim,
                      "wrong! should have FirstUnfoldDim <= LastUnfoldDim!");

        // left and right
        constexpr auto left = typename arithmetic_sequence_gen<0, FirstUnfoldDim, 1>::type{};
        constexpr auto middle =
            typename arithmetic_sequence_gen<FirstUnfoldDim, LastUnfoldDim + 1, 1>::type{};
        constexpr auto right =
            typename arithmetic_sequence_gen<LastUnfoldDim + 1, GetNumOfDimension(), 1>::type{};

        // dimensions to be unfolded need to be continuous
        static_assert(Type::Extract(middle).AreDimensionsContinuous(), "wrong! not unfoldable");

        // unfolded length, stride
        constexpr index_t unfold_length = accumulate_on_sequence(
            GetLengths().Extract(middle), math::multiplies<index_t>{}, Number<1>{});

        constexpr index_t unfold_stride = GetStride(Number<LastUnfoldDim>{});

        // new lengths, strides
        constexpr auto new_lengths = GetLengths()
                                         .Extract(left)
                                         .PushBack(Number<unfold_length>{})
                                         .PushBack(GetLengths().Extract(right));

        constexpr auto new_strides = GetStrides()
                                         .Extract(left)
                                         .PushBack(Number<unfold_stride>{})
                                         .PushBack(GetStrides().Extract(right));

        return ConstantTensorDescriptor<decltype(new_lengths), decltype(new_strides)>{};
    }

    template <class MapNew2Old>
    __host__ __device__ static constexpr auto ReorderGivenNew2Old(MapNew2Old)
    {
        return ConstantTensorDescriptor<decltype(Lengths::ReorderGivenNew2Old(MapNew2Old{})),
                                        decltype(Strides::ReorderGivenNew2Old(MapNew2Old{}))>{};
    }

#if 0 // require sequence_sort, which is not implemented yet
    template <class MapOld2New>
    __host__ __device__ static constexpr auto ReorderGivenOld2New(MapOld2New)
    {
        return ConstantTensorDescriptor<decltype(Lengths::ReorderGivenOld2New(MapOld2New{})),
                                        decltype(Strides::ReorderGivenOld2New(MapOld2New{}))>{}
    }
#endif
};

template <class Lengths>
__host__ __device__ constexpr auto make_ConstantTensorDescriptor_packed(Lengths)
{
    using Strides = decltype(calculate_tensor_strides_packed(Lengths{}));
    return ConstantTensorDescriptor<Lengths, Strides>{};
}

template <class Lengths, class Strides>
__host__ __device__ constexpr auto make_ConstantTensorDescriptor(Lengths, Strides)
{
    return ConstantTensorDescriptor<Lengths, Strides>{};
}

template <class Lengths, index_t Align>
__host__ __device__ constexpr auto make_ConstantTensorDescriptor_aligned(Lengths, Number<Align>)
{
    using Strides = decltype(calculate_tensor_strides_aligned(Lengths{}, Number<Align>{}));
    return ConstantTensorDescriptor<Lengths, Strides>{};
}

template <index_t... Lengths, index_t... Strides>
__host__ __device__ void
print_ConstantTensorDescriptor(const char* s,
                               ConstantTensorDescriptor<Sequence<Lengths...>, Sequence<Strides...>>)
{
    constexpr index_t ndim = sizeof...(Lengths);

    static_assert(ndim > 0 && ndim <= 10, "wrong!");

    static_if<ndim == 1>{}([&](auto) {
        printf("%s dim %u, lengths {%u}, strides {%u}\n", s, ndim, Lengths..., Strides...);
    });

    static_if<ndim == 2>{}([&](auto) {
        printf("%s dim %u, lengths {%u %u}, strides {%u %u}\n", s, ndim, Lengths..., Strides...);
    });

    static_if<ndim == 3>{}([&](auto) {
        printf(
            "%s dim %u, lengths {%u %u %u}, strides {%u %u %u}\n", s, ndim, Lengths..., Strides...);
    });

    static_if<ndim == 4>{}([&](auto) {
        printf("%s dim %u, lengths {%u %u %u %u}, strides {%u %u %u %u}\n",
               s,
               ndim,
               Lengths...,
               Strides...);
    });

    static_if<ndim == 5>{}([&](auto) {
        printf("%s dim %u, lengths {%u %u %u %u %u}, strides {%u %u %u %u %u}\n",
               s,
               ndim,
               Lengths...,
               Strides...);
    });

    static_if<ndim == 6>{}([&](auto) {
        printf("%s dim %u, lengths {%u %u %u %u %u %u}, strides {%u %u %u %u %u %u}\n",
               s,
               ndim,
               Lengths...,
               Strides...);
    });

    static_if<ndim == 7>{}([&](auto) {
        printf("%s dim %u, lengths {%u %u %u %u %u %u %u}, strides {%u %u %u %u %u %u %u}\n",
               s,
               ndim,
               Lengths...,
               Strides...);
    });

    static_if<ndim == 8>{}([&](auto) {
        printf("%s dim %u, lengths {%u %u %u %u %u %u %u %u}, strides {%u %u %u %u %u %u %u %u}\n",
               s,
               ndim,
               Lengths...,
               Strides...);
    });

    static_if<ndim == 9>{}([&](auto) {
        printf("%s dim %u, lengths {%u %u %u %u %u %u %u %u %u}, strides {%u %u %u %u %u %u %u %u "
               "%u}\n",
               s,
               ndim,
               Lengths...,
               Strides...);
    });

    static_if<ndim == 10>{}([&](auto) {
        printf("%s dim %u, lengths {%u %u %u %u %u %u %u %u %u %u}, strides {%u %u %u %u %u %u %u "
               "%u %u %u}\n",
               s,
               ndim,
               Lengths...,
               Strides...);
    });
}

} // namespace ck
#endif
