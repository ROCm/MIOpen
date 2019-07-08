#ifndef CK_CONSTANT_MERGED_TENSOR_DESCRIPTOR_HPP
#define CK_CONSTANT_MERGED_TENSOR_DESCRIPTOR_HPP

#include "common_header.hpp"
#include "ConstantTensorDescriptor.hpp"

namespace ck {

// OriginalTensorDesc : ConstantTensorDescriptor<...>
//     it's the tensor whose dimensions are to be merged
// OriginalDimMergeSeqs : Sequence<...>...
//     each is a sequence of original dimensions (of OriginalTensorDesc) to be merged
template <class OriginalTensorDesc, class... OriginalDimMergeSeqs>
struct ConstantMergedTensorDescriptor
{
    using Type = ConstantMergedTensorDescriptor;

    static constexpr auto mOriginalDimMergeSeqs = std::tuple<OriginalDimMergeSeqs...>{};

    static constexpr index_t nDim         = sizeof...(OriginalDimMergeSeqs);
    static constexpr index_t nOriginalDim = OriginalTensorDesc::GetNumOfDimension();

    __host__ __device__ constexpr ConstantMergedTensorDescriptor()
    {
        static_assert(nDim <= nOriginalDim, "wrong!");

        // TODO: check each of OriginalDimMergeSeqs contains at least 1, and at most
        // OriginalTensorDesc::nDim number of dimensions

        // TODO: check OriginalDimMergeSeqs contains all original dimensions

        // TODO: check there is no duplication in OriginalDimMergeSeqs
    }

    __host__ __device__ static constexpr auto GetOriginalTensorDescriptor()
    {
        return OriginalTensorDesc{};
    }

    __host__ __device__ static constexpr index_t GetNumOfDimension() { return nDim; }

    template <index_t IDim>
    __host__ __device__ static constexpr auto GetContainedOriginalDimensions(Number<IDim>)
    {
        return std::get<IDim>(mOriginalDimMergeSeqs);
    }

    template <index_t IDim>
    __host__ __device__ static constexpr bool ContainMultipleOriginalDimensions(Number<IDim>)
    {
        return (std::get<IDim>(mOriginalDimMergeSeqs).GetSize() > 1);
    }

    template <index_t IDim>
    __host__ __device__ static constexpr index_t GetLength(Number<IDim>)
    {
        constexpr auto original_dims_partial = std::get<IDim>(mOriginalDimMergeSeqs);

        return OriginalTensorDesc::Extract(original_dims_partial).GetElementSize();
    }

    template <index_t IDim>
    __host__ __device__ static constexpr index_t GetStride(Number<IDim>)
    {
        static_assert(!ContainMultipleOriginalDimensions(Number<IDim>{}),
                      "wrong! stride of a merged dimension is undefined");

        constexpr auto idim_original = std::get<IDim>(mOriginalDimMergeSeqs).Front();

        return OriginalTensorDesc::GetStride(Number<idim_original>{});
    }

    __host__ __device__ static constexpr auto GetLengths()
    {
        return Sequence<OriginalTensorDesc::Extract(OriginalDimMergeSeqs{}).GetElementSize()...>{};
    }

    __host__ __device__ static constexpr index_t GetElementSize()
    {
        return OriginalTensorDesc::GetElementSize();
    }

    template <class OriginalDimsPartial>
    struct lambda_1_GetOriginalMultiIndexFromMultiIndex
    {
        const Array<index_t, OriginalDimsPartial::GetSize()>& original_multi_id_partial;
        Array<index_t, nOriginalDim>& original_multi_id;

        __host__ __device__ constexpr lambda_1_GetOriginalMultiIndexFromMultiIndex(
            const Array<index_t, OriginalDimsPartial::GetSize()>& original_multi_id_partial_,
            Array<index_t, nOriginalDim>& original_multi_id_)
            : original_multi_id_partial(original_multi_id_partial_),
              original_multi_id(original_multi_id_)
        {
        }

        template <index_t I>
        __host__ __device__ constexpr void operator()(Number<I>) const
        {
            constexpr index_t idim_original = OriginalDimsPartial::Get(Number<I>{});

            index_t itmp = original_multi_id_partial[I];

            original_multi_id.Set(Number<idim_original>{}, itmp);
        }
    };

    struct lambda_0_GetOriginalMultiIndexFromMultiIndex
    {
        const Array<index_t, nDim>& multi_id;
        Array<index_t, nOriginalDim>& original_multi_id;

        __host__ __device__ constexpr lambda_0_GetOriginalMultiIndexFromMultiIndex(
            const Array<index_t, nDim>& multi_id_, Array<index_t, nOriginalDim>& original_multi_id_)
            : multi_id(multi_id_), original_multi_id(original_multi_id_)
        {
        }

        template <index_t IDim>
        __host__ __device__ constexpr void operator()(Number<IDim>) const
        {
            constexpr auto original_dims_partial = std::get<IDim>(Type::mOriginalDimMergeSeqs);

            // get partial original-multi-id corresponding to this merged dimension
            const auto original_multi_id_partial =
                OriginalTensorDesc::Extract(original_dims_partial)
                    .GetMultiIndexFrom1dIndex(multi_id[IDim]);

            static_for<0, original_dims_partial.GetSize(), 1>{}(
                lambda_1_GetOriginalMultiIndexFromMultiIndex<decltype(original_dims_partial)>(
                    original_multi_id_partial, original_multi_id));
        }
    };

    // return type is Array<...>
    __host__ __device__ static constexpr auto
    GetOriginalMultiIndexFromMultiIndex(Array<index_t, nDim> multi_id)
    {
        Array<index_t, nOriginalDim> original_multi_id;

        static_for<0, nDim, 1>{}(
            lambda_0_GetOriginalMultiIndexFromMultiIndex(multi_id, original_multi_id));

        return original_multi_id;
    }

    template <index_t... Is>
    __host__ __device__ static constexpr index_t GetOffsetFromMultiIndex(Sequence<Is...>)
    {
        constexpr auto multi_id = sequence2array(Sequence<Is...>{});

        constexpr auto original_multi_id = GetOriginalMultiIndexFromMultiIndex(multi_id);

        return OriginalTensorDesc::GetOffsetFromMultiIndex(original_multi_id);
    }

    __host__ __device__ static constexpr index_t
    GetOffsetFromMultiIndex(Array<index_t, nDim> multi_id)
    {
        auto original_multi_id = GetOriginalMultiIndexFromMultiIndex(multi_id);

        return OriginalTensorDesc::GetOffsetFromMultiIndex(original_multi_id);
    }

    template <class... Is>
    __host__ __device__ static constexpr index_t GetOffsetFromMultiIndex(Is... is)
    {
        return GetOffsetFromMultiIndex(Array<index_t, nDim>{is...});
    }

    __host__ __device__ static constexpr Array<index_t, nDim> GetMultiIndexFrom1dIndex(index_t id)
    {
        constexpr auto packed_desc = make_ConstantTensorDescriptor_packed(GetLengths());

        return packed_desc.GetMultiIndexFrom1dIndex(id);
    }
};

template <class OriginalTensorDesc, class... OriginalDimMergeSeqs>
__host__ __device__ constexpr auto make_ConstantMergedTensorDescriptor(OriginalTensorDesc,
                                                                       OriginalDimMergeSeqs...)
{
    return ConstantMergedTensorDescriptor<OriginalTensorDesc, OriginalDimMergeSeqs...>{};
}

template <class TDesc>
__host__ __device__ void print_ConstantMergedTensorDescriptor(const char* s, TDesc)
{
    print_ConstantTensorDescriptor(s, TDesc::GetOriginalTensorDescriptor());
}

} // namespace ck
#endif
