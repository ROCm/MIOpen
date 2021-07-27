#ifndef CK_DYNAMIC_MULTI_INDEX_TRANSFORM_HELPER_HPP
#define CK_DYNAMIC_MULTI_INDEX_TRANSFORM_HELPER_HPP

#include "common_header.hpp"
#include "dynamic_multi_index_transform.hpp"

namespace ck {

template <typename LowLength>
__host__ __device__ constexpr auto make_pass_through_transform(const LowLength& low_length)
{
    return DynamicPassThrough<LowLength>{low_length};
}

template <typename LowLength, typename LeftPad, typename RightPad, bool SkipIsValidCheck = false>
__host__ __device__ constexpr auto
make_pad_transform(const LowLength& low_length,
                   const LeftPad& left_pad,
                   const RightPad& right_pad,
                   integral_constant<bool, SkipIsValidCheck> = integral_constant<bool, false>{})
{
    return DynamicPad<LowLength, LeftPad, RightPad, SkipIsValidCheck>{
        low_length, left_pad, right_pad};
}

template <typename LowLength, typename LeftPad, bool SkipIsValidCheck = false>
__host__ __device__ constexpr auto make_left_pad_transform(
    const LowLength& low_length,
    const LeftPad& left_pad,
    integral_constant<bool, SkipIsValidCheck> = integral_constant<bool, false>{})
{
    return DynamicLeftPad<LowLength, LeftPad, SkipIsValidCheck>{low_length, left_pad};
}

template <typename LowLength, typename RightPad, bool SkipIsValidCheck>
__host__ __device__ constexpr auto make_right_pad_transform(
    const LowLength& low_length,
    const RightPad& right_pad,
    integral_constant<bool, SkipIsValidCheck> = integral_constant<bool, false>{})
{
    return DynamicRightPad<LowLength, RightPad, SkipIsValidCheck>{low_length, right_pad};
}

template <typename UpLengths,
          typename Coefficients,
          typename std::enable_if<UpLengths::Size() == Coefficients::Size(), bool>::type = false>
__host__ __device__ constexpr auto make_embed_transform(const UpLengths& up_lengths,
                                                        const Coefficients& coefficients)
{
    return DynamicEmbed<UpLengths, Coefficients>{up_lengths, coefficients};
}

template <typename LowLengths>
__host__ __device__ constexpr auto make_merge_transform(const LowLengths& low_lengths)
{
#if !CK_EXPERIMENTAL_MERGE_USE_MAGIC_DIVISION
    return DynamicMerge_v1_carry_check<LowLengths>{low_lengths};
#else
#if 1
    return DynamicMerge_v2_magic_division<LowLengths>{low_lengths};
#else
    return DynamicMerge_v2r2_magic_division<LowLengths>{low_lengths};
#endif
#endif
}

template <typename LowLengths>
__host__ __device__ constexpr auto
make_merge_transform_v2_magic_division(const LowLengths& low_lengths)
{
    return DynamicMerge_v2_magic_division<LowLengths>{low_lengths};
}

template <typename UpLengths, bool Use24BitIntegerCalculation = false>
__host__ __device__ constexpr auto make_unmerge_transform(
    const UpLengths& up_lengths,
    integral_constant<bool, Use24BitIntegerCalculation> = integral_constant<bool, false>{})
{
    return DynamicUnMerge<UpLengths, Use24BitIntegerCalculation>{up_lengths};
}

template <typename LowerIndex>
__host__ __device__ constexpr auto make_freeze_transform(const LowerIndex& low_idx)
{
    return DynamicFreeze<LowerIndex>{low_idx};
}

template <typename LowLength, typename SliceBegin, typename SliceEnd>
__host__ __device__ constexpr auto make_slice_transform(const LowLength& low_length,
                                                        const SliceBegin& slice_begin,
                                                        const SliceEnd& slice_end)
{
    return DynamicSlice<LowLength, SliceBegin, SliceEnd>{low_length, slice_begin, slice_end};
}

template <typename VectorSize, typename UpLength>
__host__ __device__ constexpr auto make_vectorize_transform(const VectorSize& vector_size,
                                                            const UpLength& up_length)
{
    return DynamicVectorize<VectorSize, UpLength>{vector_size, up_length};
}

} // namespace ck
#endif
