/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef CK_CONSTANT_MERGED_TENSOR_DESCRIPTOR_DEPRECATED_HPP
#define CK_CONSTANT_MERGED_TENSOR_DESCRIPTOR_DEPRECATED_HPP

#include "static_kernel_common_header.hpp"
#include "static_kernel_ConstantTensorDescriptor_deprecated.hpp"

namespace ck {

// OriginalTensorDesc : ConstantTensorDescriptor_deprecated<...>
//     it's the tensor whose dimensions are to be merged
// OriginalDimMergeSeqs : Sequence<...>...
//     each is a sequence of original dimensions (of OriginalTensorDesc) to be merged
template <class OriginalTensorDesc, class... OriginalDimMergeSeqs>
struct ConstantMergedTensorDescriptor_deprecated
{
    using Type = ConstantMergedTensorDescriptor_deprecated;

    static constexpr auto mOriginalDimMergeSeqs = std::tuple<OriginalDimMergeSeqs...>{};

    static constexpr index_t nDim         = sizeof...(OriginalDimMergeSeqs);
    static constexpr index_t nOriginalDim = OriginalTensorDesc::GetNumOfDimension();

    __host__ __device__ constexpr ConstantMergedTensorDescriptor_deprecated()
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

    __host__ __device__ static constexpr auto GetNumOfDimension() { return Number<nDim>{}; }

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
    __host__ __device__ static constexpr auto GetLength(Number<IDim>)
    {
        constexpr auto original_dims_partial = std::get<IDim>(mOriginalDimMergeSeqs);

        return OriginalTensorDesc::Extract(original_dims_partial).GetElementSize();
    }

    template <index_t IDim>
    __host__ __device__ static constexpr auto GetStride(Number<IDim>)
    {
        static_assert(!ContainMultipleOriginalDimensions(Number<IDim>{}),
                      "wrong! stride of a merged dimension is undefined");

        constexpr auto idim_original = std::get<IDim>(mOriginalDimMergeSeqs).Back();

        return OriginalTensorDesc::GetStride(Number<idim_original>{});
    }

    // this is a hack to return the stride of the last original dimension of a merged dimension
    // TODO: refactor this once the concept of "dimension" is used
    template <index_t IDim>
    __host__ __device__ static constexpr auto GetLastOriginalDimensionStride(Number<IDim>)
    {
        constexpr auto idim_last_original = std::get<IDim>(mOriginalDimMergeSeqs).Back();

        return OriginalTensorDesc::GetStride(Number<idim_last_original>{});
    }

    __host__ __device__ static constexpr auto GetLengths()
    {
        return Sequence<OriginalTensorDesc::Extract(OriginalDimMergeSeqs{}).GetElementSize()...>{};
    }

    __host__ __device__ static constexpr auto GetElementSize()
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

            original_multi_id(idim_original) = itmp;
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

    __host__ __device__ static constexpr auto Pack()
    {
        constexpr auto lengths = GetLengths();
        constexpr auto strides = calculate_tensor_strides_packed(lengths);
        return ConstantTensorDescriptor_deprecated<decltype(lengths), decltype(strides)>{};
    }
};

template <class OriginalTensorDesc, class... OriginalDimMergeSeqs>
__host__ __device__ constexpr auto make_ConstantMergedTensorDescriptor(OriginalTensorDesc,
                                                                       OriginalDimMergeSeqs...)
{
    return ConstantMergedTensorDescriptor_deprecated<OriginalTensorDesc, OriginalDimMergeSeqs...>{};
}

template <class TDesc>
__host__ __device__ void print_ConstantMergedTensorDescriptor(const char* s, TDesc)
{
    print_ConstantTensorDescriptor(s, TDesc::GetOriginalTensorDescriptor());
}

} // namespace ck
#endif
