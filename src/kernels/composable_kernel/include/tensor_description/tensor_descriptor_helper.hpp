#ifndef CK_TENSOR_DESCRIPTOR_HELPER_HPP
#define CK_TENSOR_DESCRIPTOR_HELPER_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"

namespace ck {

template <typename Lengths>
__host__ __device__ constexpr auto calculate_tensor_strides_packed(Lengths)
{
    return reverse_inclusive_scan_sequence(
               Lengths{}.PopFront(), math::multiplies<index_t>{}, Number<1>{})
        .PushBack(Number<1>{});
}

template <typename Lengths, index_t Align>
__host__ __device__ constexpr auto calculate_tensor_strides_aligned(Lengths, Number<Align>)
{
    constexpr index_t L_back_align =
        Align * math::integer_divide_ceiler<index_t>{}(Lengths{}.Back(), Align);

    return calculate_tensor_strides_packed(
        Lengths{}.Modify(Number<Lengths{}.GetSize() - 1>{}, Number<L_back_align>{}));
}

template <index_t... Lengths, index_t... Strides>
__host__ __device__ constexpr auto make_native_tensor_descriptor(Sequence<Lengths...>,
                                                                 Sequence<Strides...>)
{
    return NativeTensorDescriptor<NativeDimension<Lengths, Strides>...>{};
}

template <typename Lengths>
__host__ __device__ constexpr auto make_native_tensor_descriptor_packed(Lengths)
{
    constexpr auto strides = calculate_tensor_strides_packed(Lengths{});

    return make_native_tensor_descriptor(Lengths{}, strides);
}

template <typename Lengths, index_t Align>
__host__ __device__ constexpr auto make_native_tensor_descriptor_aligned(Lengths, Number<Align>)
{
    constexpr auto strides = calculate_tensor_strides_aligned(Lengths{}, Number<Align>{});
    return make_native_tensor_descriptor(Lengths{}, strides);
}

template <typename LowTensorDescriptor,
          typename Transforms,
          typename LowDimensionIds,
          typename UpDimensionIds>
__host__ __device__ constexpr auto
    transform_tensor_descriptor(LowTensorDescriptor, Transforms, LowDimensionIds, UpDimensionIds)
{
    return TransformedTensorDescriptor<LowTensorDescriptor,
                                       Transforms,
                                       LowDimensionIds,
                                       UpDimensionIds>{};
}

template <typename LowerTensorDescriptor,
          index_t... LowerLengths,
          index_t... LowerDimensionIds,
          index_t... UpperDimensionIds>
__host__ __device__ constexpr auto
    reorder_transformed_tensor_descriptor_impl(LowerTensorDescriptor,
                                               Sequence<LowerLengths...>,
                                               Sequence<LowerDimensionIds...>,
                                               Sequence<UpperDimensionIds...>)
{
    return TransformedTensorDescriptor<LowerTensorDescriptor,
                                       Tuple<PassThrough<LowerLengths>...>,
                                       Tuple<Sequence<LowerDimensionIds>...>,
                                       Tuple<Sequence<UpperDimensionIds>...>>{};
}

// reorder a NativeTensorDescriptor
template <typename... Ts, typename MapLower2Upper>
__host__ __device__ constexpr auto
    reorder_tensor_descriptor_given_lower2upper(NativeTensorDescriptor<Ts...>, MapLower2Upper)
{
    static_assert(is_valid_sequence_map<MapLower2Upper>{},
                  "wrong! MapLower2Upper is not a valid map");

    constexpr auto old_desc = NativeTensorDescriptor<Ts...>{};

    static_assert(old_desc.GetNumOfDimension() == MapLower2Upper::Size(), "wrong!");

    constexpr auto new_lengths = old_desc.GetLengths().ReorderGivenOld2New(MapLower2Upper{});
    constexpr auto new_strides = old_desc.GetStrides().ReorderGivenOld2New(MapLower2Upper{});

    return make_native_tensor_descriptor(new_lengths, new_strides);
}

// reorder a TransformedTensorDescriptor
template <typename... Ts, typename MapLower2Upper>
__host__ __device__ constexpr auto
    reorder_tensor_descriptor_given_lower2upper(TransformedTensorDescriptor<Ts...>, MapLower2Upper)
{
    static_assert(is_valid_sequence_map<MapLower2Upper>{},
                  "wrong! MapLower2Upper is not a valid map");

    constexpr auto low_desc = TransformedTensorDescriptor<Ts...>{};

    static_assert(low_desc.GetNumOfDimension() == MapLower2Upper::Size(), "wrong!");

    return reorder_transformed_tensor_descriptor_impl(
        low_desc,
        low_desc.GetLengths(),
        typename arithmetic_sequence_gen<0, low_desc.GetNumOfDimension(), 1>::type{},
        MapLower2Upper{});
}

template <typename LowerTensorDescriptor, typename MapUpper2Lower>
__host__ __device__ constexpr auto
    reorder_tensor_descriptor_given_upper2lower(LowerTensorDescriptor, MapUpper2Lower)
{
    return reorder_tensor_descriptor_given_lower2upper(
        LowerTensorDescriptor{}, typename sequence_map_inverse<MapUpper2Lower>::type{});
}

template <typename Lengths, typename Strides>
__host__ __device__ constexpr bool are_dimensions_unfoldable(Lengths, Strides)
{
    static_assert(Lengths::Size() == Strides::Size(), "wrong!");

    bool flag = true;

    for(index_t i = 0; i < Lengths::Size() - 1; ++i)
    {
        flag = flag && Strides::At(i) == Strides::At(i + 1) * Lengths::At(i + 1);
    }

    return flag;
}

// unfold only support NativeTennsorDescriptor, for now
template <index_t FirstUnfoldDim, index_t LastUnfoldDim, typename... Ts>
__host__ __device__ constexpr auto unfold_tensor_descriptor(NativeTensorDescriptor<Ts...> desc,
                                                            Number<FirstUnfoldDim>,
                                                            Number<LastUnfoldDim>)
{
    constexpr index_t nDim = desc.GetNumOfDimension();

    static_assert(FirstUnfoldDim >= 0 && LastUnfoldDim < nDim && FirstUnfoldDim <= LastUnfoldDim,
                  "wrong! should have FirstUnfoldDim <= LastUnfoldDim!");

    // left and right
    constexpr auto left = typename arithmetic_sequence_gen<0, FirstUnfoldDim, 1>::type{};
    constexpr auto middle =
        typename arithmetic_sequence_gen<FirstUnfoldDim, LastUnfoldDim + 1, 1>::type{};
    constexpr auto right = typename arithmetic_sequence_gen<LastUnfoldDim + 1, nDim, 1>::type{};

    // sanity-checknfoldable
    static_assert(are_dimensions_unfoldable(desc.GetLengths(middle), desc.GetStrides(middle)),
                  "wrong! not unfoldable");

    // unfolded length, stride
    constexpr index_t unfold_length =
        reduce_on_sequence(desc.GetLengths(middle), math::multiplies<index_t>{}, Number<1>{});

    constexpr index_t unfold_stride = desc.GetStride(Number<LastUnfoldDim>{});

    // new lengths, strides
    constexpr auto new_lengths =
        desc.GetLengths(left).PushBack(Number<unfold_length>{}).PushBack(desc.GetLengths(right));

    constexpr auto new_strides =
        desc.GetStrides(left).PushBack(Number<unfold_stride>{}).PushBack(desc.GetStrides(right));

    return make_native_tensor_descriptor(new_lengths, new_strides);
}

// a cluster map 1d index to N-d index
template <typename Lengths, typename ArrangeOrder>
struct ClusterDescriptor
{
    static constexpr index_t nDim = Lengths::Size();

    static constexpr auto mDesc = transform_tensor_descriptor(
        make_native_tensor_descriptor_packed(Lengths{}),
        make_tuple(Merge<decltype(Lengths::ReorderGivenNew2Old(ArrangeOrder{}))>{}),
        make_tuple(ArrangeOrder{}),
        make_tuple(Sequence<0>{}));

    __host__ __device__ constexpr ClusterDescriptor()
    {
        static_assert(Lengths::Size() == nDim && ArrangeOrder::Size() == nDim,
                      "wrong! size not the same");

        static_assert(is_valid_sequence_map<ArrangeOrder>{}, "wrong! ArrangeOrder is wrong");
    }

    __host__ __device__ static constexpr index_t GetElementSize() { return mDesc.GetElementSize(); }

    __host__ __device__ static constexpr auto CalculateClusterIndex(index_t idx_1d)
    {
        return mDesc.CalculateLowerIndex(MultiIndex<1>{idx_1d});
    }
};

template <typename Lengths,
          typename ArrangeOrder = typename arithmetic_sequence_gen<0, Lengths::Size(), 1>::type>
__host__ __device__ constexpr auto make_cluster_descriptor(
    Lengths, ArrangeOrder order = typename arithmetic_sequence_gen<0, Lengths::Size(), 1>::type{})
{
    return ClusterDescriptor<Lengths, decltype(order)>{};
}

} // namespace ck
#endif
