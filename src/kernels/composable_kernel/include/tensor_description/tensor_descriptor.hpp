#ifndef CK_TENSOR_DESCRIPTOR_HPP
#define CK_TENSOR_DESCRIPTOR_HPP

#include "common_header.hpp"
#include "dimension.hpp"
#include "multi_index_transform.hpp"

namespace ck {

// tensor descriptor for "native tensor"
// A "native tensor" is a "true" tensor that can be represented by Lengths and Strides
template <typename... NativeDimensions>
struct NativeTensorDescriptor
{
    using type                        = NativeTensorDescriptor;
    static constexpr index_t nDim     = sizeof...(NativeDimensions);
    static constexpr auto mDimensions = make_tuple(NativeDimensions{}...);

    using Index = MultiIndex<nDim>;

    __host__ __device__ static constexpr auto GetNumOfDimension() { return Number<nDim>{}; }

    template <index_t IDim>
    __host__ __device__ static constexpr auto GetLength(Number<IDim>)
    {
        return mDimensions.At(Number<IDim>{}).GetLength();
    }

    template <index_t IDim>
    __host__ __device__ static constexpr auto GetStride(Number<IDim>)
    {
        return mDimensions.At(Number<IDim>{}).GetStride();
    }

    template <index_t... IDims>
    __host__ __device__ static constexpr auto GetLengths(Sequence<IDims...>)
    {
        return Sequence<GetLength(Number<IDims>{})...>{};
    }

    template <index_t... IDims>
    __host__ __device__ static constexpr auto GetStrides(Sequence<IDims...>)
    {
        return Sequence<GetStride(Number<IDims>{})...>{};
    }

    template <index_t IDim, index_t... IDims>
    __host__ __device__ static constexpr auto GetLengths(Number<IDim>, Number<IDims>...)
    {
        return GetLengths(Sequence<IDim, IDims...>{});
    }

    template <index_t IDim, index_t... IDims>
    __host__ __device__ static constexpr auto GetStrides(Number<IDim>, Number<IDims>...)
    {
        return GetStrides(Sequence<IDim, IDims...>{});
    }

    __host__ __device__ static constexpr auto GetLengths()
    {
        return GetLengths(typename arithmetic_sequence_gen<0, nDim, 1>::type{});
    }

    __host__ __device__ static constexpr auto GetStrides()
    {
        return GetStrides(typename arithmetic_sequence_gen<0, nDim, 1>::type{});
    }

    __host__ __device__ static constexpr index_t GetElementSize()
    {
        return reduce_on_sequence(GetLengths(), math::multiplies<index_t>{}, Number<1>{});
    }

    __host__ __device__ static constexpr index_t GetElementSpace()
    {
        return reduce_on_sequence(
            (GetLengths() - Number<1>{}) * GetStrides(), math::plus<index_t>{}, Number<1>{});
    }

    // TODO: this cannot return constepxr because of use of lambda
    __host__ __device__ static constexpr index_t CalculateOffset(const Index& idx)
    {
        index_t offset = 0;

        static_for<0, nDim, 1>{}([&](auto idim) { offset += idx[idim] * GetStride(idim); });

        return offset;
    }

    __host__ __device__ static constexpr index_t CalculateOffsetDiff(const Index& idx_diff)
    {
        index_t offset_diff = 0;

        static_for<0, nDim, 1>{}(
            [&](auto idim) { offset_diff += idx_diff[idim] * GetStride(idim); });

        return offset_diff;
    }

    template <index_t IDim>
    __host__ __device__ static constexpr bool IsLinearDimension(Number<IDim>)
    {
        return true;
    }

    __host__ __device__ static constexpr auto GetLinearDimensionMask()
    {
        return typename uniform_sequence_gen<nDim, 1>::type{};
    }

    __host__ __device__ static constexpr auto GetNonLinearDimensionMask()
    {
        return typename uniform_sequence_gen<nDim, 0>::type{};
    }

    __host__ __device__ static constexpr auto GetNonLinearDimensions() { return Sequence<>{}; }

    __host__ __device__ static constexpr auto GetNonLinearIndependentDimensionGroups()
    {
        return Tuple<>{};
    }

    __host__ __device__ static constexpr bool
    IsUpperIndexMappedToValidOffset(const Index& /* idx */)
    {
        return true;
    }
};

// Tensor descriptor for "transformed tensor"
template <typename LowTensorDescriptor, // NativeTensorDescriptor or TransformedTensorDescriptor
          typename Transforms,          // Tuple<MultIndexTransforms...>
          typename LowDimensionIds,     // Tuple<Sequence<...>>
          typename UpDimensionIds>      // Tuple<Sequence<...>>
struct TransformedTensorDescriptor
{
    using type                          = TransformedTensorDescriptor;
    static constexpr index_t nTransform = Transforms::Size();

    struct lambda_merge_sequences
    {
        template <typename... Seqs>
        __host__ __device__ constexpr auto operator()(Seqs... seqs) const
        {
            return merge_sequences(seqs...);
        }
    };

    __host__ __device__ static constexpr auto GetNumOfLowerDimension()
    {
        // Here, we assume all lower-dimensions are active
        // TODO: sanity-check all lower-dimension are indeed active

        using duplicated_low_active_dims =
            decltype(unpack(lambda_merge_sequences{}, LowDimensionIds{}));

        using low_active_dims = typename sequence_unique_sort<duplicated_low_active_dims,
                                                              math::less<index_t>,
                                                              math::equal<index_t>>::type;

        return low_active_dims::Size();
    }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension()
    {
        using duplicated_up_active_dims =
            decltype(unpack(lambda_merge_sequences{}, UpDimensionIds{}));

        using up_active_dims = typename sequence_unique_sort<duplicated_up_active_dims,
                                                             math::less<index_t>,
                                                             math::equal<index_t>>::type;

        return up_active_dims::Size();
    }

    static constexpr index_t nDimUp  = GetNumOfUpperDimension();
    static constexpr index_t nDimLow = GetNumOfLowerDimension();

    using UpperIndex = MultiIndex<nDimUp>;
    using LowerIndex = MultiIndex<nDimLow>;

    __host__ __device__ constexpr TransformedTensorDescriptor()
    {
        static_assert(nTransform == Transforms::Size() && nTransform == LowDimensionIds::Size() &&
                          nTransform == UpDimensionIds::Size(),
                      "wrong! # of transformations not the same");

        // sanity check:
        //   LowDimensionIds should include all low-dimensions,
        //   UpDimensionIds should include all up-dimensions
        using mingled_up_dimension_ids =
            decltype(unpack(lambda_merge_sequences{}, UpDimensionIds{}));

        using sorted_up_dimension_ids =
            typename sequence_sort<mingled_up_dimension_ids, math::less<index_t>>::type;

        static_assert(sorted_up_dimension_ids::Size() == nDimUp &&
                          is_valid_sequence_map<sorted_up_dimension_ids>{},
                      "wrong! UpDimensionIds is not configured correctly");

        using mingled_low_dimension_ids =
            decltype(unpack(lambda_merge_sequences{}, LowDimensionIds{}));

        using sorted_low_dimension_ids =
            typename sequence_sort<mingled_low_dimension_ids, math::less<index_t>>::type;

        static_assert(sorted_low_dimension_ids::Size() == nDimLow &&
                          is_valid_sequence_map<sorted_low_dimension_ids>{},
                      "wrong! LowDimensionIds is not configured correctly");

        // TODO: sanity check: while a up-dimension could be associated with multille
        //   transformation, a low-dimension should be associated with only one transformation

        // TODO: sanity-check: GetLowerLengths of each transform should be consistent with lengths
        //   of lower-tensor-descriptor
    }

    __host__ __device__ static constexpr auto GetNumOfDimension()
    {
        return GetNumOfUpperDimension();
    }

    __host__ __device__ static constexpr auto GetLowerTensorDescriptor()
    {
        return LowTensorDescriptor{};
    }

    struct lambda_GetUpperLengths
    {
        template <typename Transform>
        __host__ __device__ constexpr auto operator()(const Transform& tran) const
        {
            return tran.GetUpperLengths();
        }
    };

    __host__ __device__ static constexpr auto GetUpperLengths()
    {
        constexpr auto tuple_of_up_lengths =
            transform_tuples(lambda_GetUpperLengths{}, Transforms{});

        constexpr auto mingled_up_lengths = unpack(lambda_merge_sequences{}, tuple_of_up_lengths);

        constexpr auto mingled_up_dimension_ids =
            unpack(lambda_merge_sequences{}, UpDimensionIds{});

        // TODO: sanity-check mingled_up_dimension_ids contain all upper-dimensions
        // TODO: sanity-check mingled_up_lengths have no conflicting upper-length

        // sort by upper-dimension-ids
        using sort_up_dimension_ids = sequence_unique_sort<decltype(mingled_up_dimension_ids),
                                                           math::less<index_t>,
                                                           math::equal<index_t>>;

        // sanity-check sorted-upper-dimension-ids should be Sequence<0, 1, ... nDimUp-1>
        static_assert(is_same<typename sort_up_dimension_ids::type,
                              typename arithmetic_sequence_gen<0, nDimUp, 1>::type>{},
                      "wrong! UpDimensionIds is not configured correctly");

        constexpr auto sorted2unsorted_map = typename sort_up_dimension_ids::sorted2unsorted_map{};

        constexpr auto sorted_up_lengths =
            pick_sequence_elements_by_ids(mingled_up_lengths, sorted2unsorted_map);

        return sorted_up_lengths;
    }

    __host__ __device__ static constexpr auto GetLengths() { return GetUpperLengths(); }

    template <index_t IDim>
    __host__ __device__ static constexpr auto GetLength(Number<IDim>)
    {
        return GetLengths()[IDim];
    }

    template <index_t... IDims>
    __host__ __device__ static constexpr auto GetLengths(Sequence<IDims...>)
    {
        return Sequence<GetLength(Number<IDims>{})...>{};
    }

    template <index_t IDim, index_t... IDims>
    __host__ __device__ static constexpr auto GetLengths(Number<IDim>, Number<IDims>...)
    {
        return GetLengths(Sequence<IDim, IDims...>{});
    }

    __host__ __device__ static constexpr index_t GetElementSize()
    {
        return reduce_on_sequence(GetLengths(), math::multiplies<index_t>{}, Number<1>{});
    }

    __host__ __device__ static constexpr index_t GetElementSpace()
    {
        // TODO: Is this the correct definition for transformed tensor?
        return GetLowerTensorDescriptor().GetElementSpace();
    }

    // TODO: right now return value is not constexpr because use of non-constexpr lambda
    __host__ __device__ static constexpr LowerIndex CalculateLowerIndex(const UpperIndex& idx_up)
    {
        LowerIndex idx_low;

        static_for<0, nTransform, 1>{}([&](auto itran) {
            constexpr auto tran = Transforms{}.At(itran);

            const auto idx_up_part = pick_array_element(idx_up, UpDimensionIds{}.At(itran));
            auto idx_low_part      = pick_array_element(idx_low, LowDimensionIds{}.At(itran));

            // this assume each lower (single) index is only assocaited with one transformation,
            //   which is required for index transformation, and has been checked during constructor
            //   of TransformedTensorDescriptor
            idx_low_part = tran.CalculateLowerIndex(to_array(idx_up_part));
        });

        return idx_low;
    }

    // TODO: right now return value is not constexpr because use of non-constepxr lambda
    __host__ __device__ static constexpr LowerIndex CalculateLowerIndexDiff(
        const UpperIndex& idx_up_diff, const UpperIndex& idx_up_old, const LowerIndex& idx_low_old)
    {
        LowerIndex idx_low_diff;

        static_for<0, nTransform, 1>{}([&](auto itran) {
            constexpr auto tran = Transforms{}.At(itran);

            const auto idx_up_diff_part =
                pick_array_element(idx_up_diff, UpDimensionIds{}.At(itran));

            const auto idx_up_old_part = pick_array_element(idx_up_old, UpDimensionIds{}.At(itran));

            const auto idx_low_old_part =
                pick_array_element(idx_low_old, LowDimensionIds{}.At(itran));

            auto idx_low_diff_part = pick_array_element(idx_low_diff, LowDimensionIds{}.At(itran));

            // this assume each lower (single) index is associated with only one transformation,
            //   which is required for index transformation, and has been checked during constructor
            //   of TransformedTensorDescriptor
            idx_low_diff_part = tran.CalculateLowerIndexDiff(
                to_array(idx_up_diff_part), to_array(idx_up_old_part), to_array(idx_low_old_part));
        });

        return idx_low_diff;
    }

    __host__ __device__ static constexpr index_t CalculateOffset(const UpperIndex& idx_up)
    {
        return GetLowerTensorDescriptor().CalculateOffset(CalculateLowerIndex(idx_up));
    }

    struct lambda_sequence_logical_and
    {
        template <typename... Seqs>
        __host__ __device__ constexpr auto operator()(Seqs...) const
        {
            return typename sequence_reduce<logical_and<index_t>, Seqs...>::type{};
        }
    };

    template <typename T>
    struct lambda_is_true
    {
        __host__ __device__ constexpr auto operator()(const T& x) const
        {
            // TODO: remove static_cast once Sequence can take bool as entries
            return static_cast<bool>(x) == true;
        }
    };

    struct lambda_get_linear_dimension_mask_of_single_tranform
    {
        // check only one transform at a time
        template <typename Transform, typename LowDimensionId, typename UpDimensionId>
        __host__ __device__ constexpr auto
        operator()(Transform, LowDimensionId, UpDimensionId) const
        {
            // judge if transformation is linear
            constexpr bool is_linear_transform = Transform::IsLinearTransform();

            // judge if all lower dimension are linear
            constexpr bool are_all_low_dim_linear = sequence_all_of(
                pick_sequence_elements_by_ids(GetLowerTensorDescriptor().GetLinearDimensionMask(),
                                              LowDimensionId{}),
                lambda_is_true<index_t>{});

            // create linear mask for upper dimensions
            constexpr bool are_up_dim_linear = is_linear_transform && are_all_low_dim_linear;

            constexpr auto mask_of_up_linear_dims = modify_sequence_elements_by_ids(
                typename uniform_sequence_gen<nDimUp, 1>::type{},
                typename uniform_sequence_gen<UpDimensionId::Size(), are_up_dim_linear>::type{},
                UpDimensionId{});

            return mask_of_up_linear_dims;
        }
    };

    // TODO: this is a hack, transform_tuples() doesn't compile, would complain about constexpr
    template <typename F, typename X, typename Y, typename Z, index_t... Is>
    __host__ __device__ static constexpr auto
    dummy_transform_tuples_impl(F f, X x, Y y, Z z, Sequence<Is...>)
    {
        return make_tuple(f(x.At(Number<Is>{}), y.At(Number<Is>{}), z.At(Number<Is>{}))...);
    }

    __host__ __device__ static constexpr auto GetLinearDimensionMask()
    {
#if 0
        // create tuple of linear dimension masks, for all transformations
        // TODO: this doesn't compile, because transform_tuples() complain about constexpr
        constexpr auto tuple_of_linear_dimension_mask =
            transform_tuples(lambda_get_linear_dimension_mask_of_single_tranform{},
                             Transforms{},
                             LowDimensionIds{},
                             UpDimensionIds{});
#else
        // create tuple of linear dimension masks, for all transformations
        // TODO: this is a hack
        constexpr auto tuple_of_linear_dimension_mask = dummy_transform_tuples_impl(
            lambda_get_linear_dimension_mask_of_single_tranform{},
            Transforms{},
            LowDimensionIds{},
            UpDimensionIds{},
            typename arithmetic_sequence_gen<0, Transforms::Size(), 1>::type{});
#endif

        // reduce tuple of masks into one mask
        constexpr auto linear_dimension_mask =
            unpack(lambda_sequence_logical_and{}, tuple_of_linear_dimension_mask);

        return linear_dimension_mask;
    }

    __host__ __device__ static constexpr auto GetNonLinearDimensionMask()
    {
        return GetLinearDimensionMask().Transform(logical_not<index_t>{});
    }

    template <index_t IDim>
    __host__ __device__ static constexpr bool IsLinearDimension(Number<IDim>)
    {
        return GetLinearDimensionMask().At(Number<IDim>{});
    }

    __host__ __device__ static constexpr auto GetLinearDimensions()
    {
        constexpr auto linear_dimension_mask = GetLinearDimensionMask();

        return pick_sequence_elements_by_mask(
            typename arithmetic_sequence_gen<0, nDimUp, 1>::type{}, linear_dimension_mask);
    }

    __host__ __device__ static constexpr auto GetNonLinearDimensions()
    {
        constexpr auto nonlinear_dimension_mask = GetNonLinearDimensionMask();

        return pick_sequence_elements_by_mask(
            typename arithmetic_sequence_gen<0, nDimUp, 1>::type{}, nonlinear_dimension_mask);
    }

#if 0
    __host__ __device__ static constexpr auto GetNonLinearIndependentDimensionGroups()
    {
        // TODO: not implemented
    }
#endif

    __host__ __device__ static constexpr bool
    IsUpperIndexMappedToValidLowerIndex(const UpperIndex& idx_up)
    {
        bool flag = true;

        static_for<0, nTransform, 1>{}([&](auto itran) {
            constexpr auto tran = Transforms{}.At(itran);

            const auto idx_up_part = pick_array_element(idx_up, UpDimensionIds{}.At(itran));

            flag = flag && tran.IsUpperIndexMappedToValidLowerIndex(to_array(idx_up_part));
        });

        return flag;
    }

    // Whenever this function is called, it will call CalculateLowerIndex() recursively.
    // If you have created a tensor coordinate already, instead of calling this function,
    //   you should call TensorCoordinate::IsUpperIndexMappedToValidOffset() which would
    //   be less expensive.
    __host__ __device__ static constexpr bool
    IsUpperIndexMappedToValidOffset(const UpperIndex& idx_up)
    {
        return IsUpperIndexMappedToValidLowerIndex(idx_up) &&
               GetLowerTensorDescriptor().IsUpperIndexMappedToValidOffset(
                   CalculateLowerIndex(idx_up));
    }
};

} // namespace ck
#endif
