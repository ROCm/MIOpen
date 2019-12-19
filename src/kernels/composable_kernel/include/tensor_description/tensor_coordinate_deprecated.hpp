#ifndef CK_TENSOR_COORDINATE_DEPRECATED_HPP
#define CK_TENSOR_COORDINATE_DEPRECATED_HPP

#include "common_header.hpp"
#include "ConstantTensorDescriptor_deprecated.hpp"
#include "ConstantMergedTensorDescriptor_deprecated.hpp"

namespace ck {

// TensorDesc is ConstantTensorDescriptor_deprecated
template <class TensorDesc>
struct NormalTensorCoordinate_deprecated
{
    using type             = NormalTensorCoordinate_deprecated;
    using tensor_desc_type = TensorDesc;

    static constexpr index_t nDim = tensor_desc_type::GetNumOfDimension();

    __host__
        __device__ constexpr NormalTensorCoordinate_deprecated(Array<index_t, nDim> tensor_index)
        : mOffset{tensor_desc_type::GetOffsetFromMultiIndex(tensor_index)}
    {
    }

    template <class... Xs>
    __host__ __device__ constexpr NormalTensorCoordinate_deprecated(Xs... xs)
        : NormalTensorCoordinate_deprecated(Array<index_t, nDim>{xs...})
    {
    }

    template <index_t... Xs>
    __host__ __device__ constexpr NormalTensorCoordinate_deprecated(Sequence<Xs...>)
        : NormalTensorCoordinate_deprecated(Array<index_t, nDim>{Xs...})
    {
    }

    __host__ __device__ constexpr index_t GetOffset() const { return mOffset; }

    // T is Array or Sequence
    template <class T>
    __host__ __device__ type operator+=(T step_sizes)
    {
        static_assert(is_same<typename T::data_type, index_t>{} && T::GetSize() == nDim, "wrong!");

        mOffset += tensor_desc_type::GetOffsetFromMultiIndex(step_sizes);

        return *this;
    }

    template <class T>
    __host__ __device__ type operator-=(T step_sizes)
    {
        static_assert(is_same<typename T::data_type, index_t>{} && T::GetSize() == nDim, "wrong!");

        mOffset -= tensor_desc_type::GetOffsetFromMultiIndex(step_sizes);

        return *this;
    }

    template <class T>
    __host__ __device__ constexpr type operator+(T step_sizes) const
    {
        type coord = *this;
        coord += step_sizes;
        return coord;
    }

    template <class T>
    __host__ __device__ constexpr type operator-(T step_sizes) const
    {
        type coord = *this;
        coord -= step_sizes;
        return coord;
    }

    // reposition point of origin, and return compensated offset.
    // This is a hack to reduce index calculation during looping over
    // a tensor whose origin is this TensorCoordinate. It does so, by spitting
    // out the run-time offset to the pointer (to the tensor data) held by this
    // TensorCoordiante, so the caller can add the offset into the run-time pointer of
    // the data, so only 1 run-time variable (update pointer) is needed, instead
    // of 2 run-time variables (old pointer and this offset)
    // TODO: after introducing the concept of "run-time tensor view", which contains the
    // run-time pointer to the data, always keep track of the pointer, instead of both
    // offset and the pointer. This also bring additional benefit that we don't need to
    // worry the offset might underflow (because offset is unsigned integer) when updating it.
    __host__ __device__ constexpr index_t RepositionOrigin()
    {
        index_t offset_diff = mOffset;
        mOffset             = 0;
        return offset_diff;
    }

    private:
    index_t mOffset;
};

// TensorDesc is ConstantMergedTensorDescriptor_deprecated
template <class TensorDesc>
struct MergedTensorCoordinate_deprecated
{
    using type             = MergedTensorCoordinate_deprecated;
    using tensor_desc_type = TensorDesc;

    static constexpr index_t nDim = tensor_desc_type::GetNumOfDimension();
    static constexpr index_t nOriginalDim =
        tensor_desc_type::GetOriginalTensorDescriptor().GetNumOfDimension();

    __host__
        __device__ constexpr MergedTensorCoordinate_deprecated(Array<index_t, nDim> tensor_index)
        : mOriginalIndex{tensor_desc_type::GetOriginalMultiIndexFromMultiIndex(tensor_index)}
    {
        // partial offset on each dimension
        static_for<0, nDim, 1>{}([&](auto idim) {
            constexpr auto partial_original_dims =
                tensor_desc_type::GetContainedOriginalDimensions(idim);

            constexpr auto partial_original_desc =
                tensor_desc_type::GetOriginalTensorDescriptor().Extract(partial_original_dims);

            mPartialOffsets(idim) = partial_original_desc.GetOffsetFromMultiIndex(
                extract_array(mOriginalIndex, partial_original_dims));
        });

        // complete offset
        mOffset =
            accumulate_on_array(mPartialOffsets, math::plus<index_t>{}, static_cast<index_t>(0));
    }

    template <class... Xs>
    __host__ __device__ constexpr MergedTensorCoordinate_deprecated(Xs... xs)
        : MergedTensorCoordinate_deprecated(Array<index_t, nDim>{xs...})
    {
    }

    __host__ __device__ constexpr index_t GetOffset() const { return mOffset; }

    template <class IDim, class T, bool PositiveDirection>
    __host__ __device__ void
    MoveOnDimension(IDim idim_, T step_size, integral_constant<bool, PositiveDirection>)
    {
        constexpr auto idim = idim_;

        // if step_size is known at compile time
        static_if<is_static<T>::value>{}(
            [&](auto) { static_if<T{} == 0>{}([&](auto) { return; }); });

        // update original index
        static_if<tensor_desc_type::ContainMultipleOriginalDimensions(idim)>{}([&](auto) {
            constexpr auto partial_original_dims =
                tensor_desc_type::GetContainedOriginalDimensions(idim);

            constexpr index_t ndim_partial_original = partial_original_dims.GetSize();

            constexpr auto partial_original_desc =
                tensor_desc_type::GetOriginalTensorDescriptor().Extract(partial_original_dims);

            const auto partial_original_step_sizes =
                partial_original_desc.GetMultiIndexFrom1dIndex(step_size);

            // update partial original multi-id
            auto partial_original_id = extract_array(mOriginalIndex, partial_original_dims);

            static_if<PositiveDirection>{}([&](auto) {
                partial_original_id += partial_original_step_sizes;

                bool carry = false;

                // do carry check in reversed order, starting from lowest dimension
                // don't check the highest dimension
                static_for<0, ndim_partial_original - 1, 1>{}([&](auto IReverse) {
                    constexpr index_t i = ndim_partial_original - 1 - IReverse;

                    if(carry)
                    {
                        ++partial_original_id(i);
                    }

                    carry = false;

                    if(partial_original_id[i] >= partial_original_desc.GetLength(i))
                    {
                        partial_original_id(i) -= partial_original_desc.GetLength(i);
                        carry = true;
                    }
                });

                // highest dimension
                if(carry)
                {
                    ++partial_original_id(0);
                }
            }).Else([&](auto) {
                // shift up multi-id to avoid unsigned integer underflow during intermediate
                // calculations. After the shift, should have new_multi_id[...] >= 1
                partial_original_id +=
                    partial_original_desc.GetLengths() - partial_original_step_sizes;

                bool borrow = false;

                // do borrow check in reversed order, starting from lowest dimension
                // don't check the highest dimension
                static_for<0, ndim_partial_original - 1, 1>{}([&](auto IReverse) {
                    constexpr index_t i = ndim_partial_original - 1 - IReverse;

                    if(borrow)
                    {
                        --partial_original_id(i);
                    }

                    borrow = false;

                    if(partial_original_id[i] < partial_original_desc.GetLength(i))
                    {
                        partial_original_id(i) += partial_original_desc.GetLength(i);
                        borrow = true;
                    }
                });

                // highest dimension
                if(borrow)
                {
                    --partial_original_id(0);
                }

                // shift back down multi-id
                // here, should have new_multi_id[...] >= GetLengths()
                partial_original_id = partial_original_id - partial_original_desc.GetLengths();
            });

            // update "mOriginalIndex"
            static_for<0, ndim_partial_original, 1>{}([&](auto I) {
                constexpr auto idim_original = partial_original_dims[I];

                mOriginalIndex(idim_original) = partial_original_id[I];
            });

            // calculate new partial offset on this merged dimension
            const index_t old_partial_offset = mPartialOffsets[idim];

            mPartialOffsets(idim) =
                partial_original_desc.GetOffsetFromMultiIndex(partial_original_id);

            // update "mThreadSrcOffset", do "+" before "-" to avoid underflow
            mOffset = (mOffset + mPartialOffsets[idim]) - old_partial_offset;
        }).Else([&](auto fwd) {
            static_if<PositiveDirection>{}([&](auto) {
                mOffset += step_size * fwd(tensor_desc_type{}).GetStride(idim);
            }).Else([&](auto) { mOffset -= step_size * fwd(tensor_desc_type{}).GetStride(idim); });
        });
    }

    // T is Array or Sequence
    template <class T>
    __host__ __device__ type operator+=(T step_sizes)
    {
        static_assert(is_same<typename T::data_type, index_t>{} && T::GetSize() == nDim, "wrong!");

        static_for<0, nDim, 1>{}([&](auto idim) {
            // compiler should remove dead code path, because step_sizes is known at
            // compile time
            if(step_sizes[idim] != 0)
            {
                this->MoveOnDimension(idim, step_sizes[idim], integral_constant<bool, true>{});
            }
        });

        return *this;
    }

    template <class T>
    __host__ __device__ type operator-=(T step_sizes)
    {
        static_assert(is_same<typename T::data_type, index_t>{} && T::GetSize() == nDim, "wrong!");

        static_for<0, nDim, 1>{}([&](auto idim) {
            // compiler should remove dead code path, because step_sizes is known at
            // compile time
            if(step_sizes[idim] != 0)
            {
                this->MoveOnDimension(idim, step_sizes[idim], integral_constant<bool, false>{});
            }
        });

        return *this;
    }

    template <class T>
    __host__ __device__ constexpr type operator+(T step_sizes) const
    {
        type coord = *this;
        coord += step_sizes;
        return coord;
    }

    template <class T>
    __host__ __device__ constexpr type operator-(T step_sizes) const
    {
        type coord = *this;
        coord -= step_sizes;
        return coord;
    }

    __host__ __device__ static constexpr index_t RepositionOrigin() { return 0; }

    private:
    // Allocate register memory for all merged dimensions and normal dimensions.
    // However, only those merged dimensions, whose index will be involved in arithmetic
    // after the construction of this TensorCoordinate (e.g. when user move a slicing
    // window on the merged dimension), will use these register memory.
    // Let's hope compiler will optimize away those register memory allocated for normal
    // dimensions, and those merged dimensions, that would never be involved in index
    // arithmetic after construction of TensorCoordinate.
    // TODO: refactor TensorCoordinate, after introducing the concept of "dimensions"
    // and simplify implementation of ConstantMergedTensorDescriptor_deprecated, so we don't need to
    // count on compiler to optimize away those register memory for us
    Array<index_t, nOriginalDim> mOriginalIndex;
    Array<index_t, nDim> mPartialOffsets;

    // complete offset
    index_t mOffset;
};

template <class TensorDesc>
struct TensorCoordinate_deprecated
{
    private:
    template <class... Ts>
    __host__ __device__ static constexpr auto
        MakeDummyTensorCoordinate(ConstantTensorDescriptor_deprecated<Ts...>)
    {
        return NormalTensorCoordinate_deprecated<ConstantTensorDescriptor_deprecated<Ts...>>();
    }

    template <class... Ts>
    __host__ __device__ static constexpr auto
        MakeDummyTensorCoordinate(ConstantMergedTensorDescriptor_deprecated<Ts...>)
    {
        return MergedTensorCoordinate_deprecated<
            ConstantMergedTensorDescriptor_deprecated<Ts...>>();
    }

    public:
    using type = decltype(MakeDummyTensorCoordinate(TensorDesc{}));
};

} // namespace ck
#endif
