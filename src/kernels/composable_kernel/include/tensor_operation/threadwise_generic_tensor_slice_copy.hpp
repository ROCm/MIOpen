#ifndef CK_THREADWISE_GENERIC_TENSOR_SLICE_COPY_HPP
#define CK_THREADWISE_GENERIC_TENSOR_SLICE_COPY_HPP

#include "common_header.hpp"
#include "ConstantTensorDescriptor.hpp"
#include "ConstantMergedTensorDescriptor.hpp"

#ifndef CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V1
#define CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V1 0
#endif

namespace ck {

// user need to make sure alignment requirement is satisfied when setting DataPerAccesss > 1
template <class Float,
          class SrcDesc,
          class DstDesc,
          class SliceLengths,
          class DimAccessOrder,
          index_t DataPerAccess>
__device__ void threadwise_generic_tensor_slice_copy_v1(
    SrcDesc,
    const Float* __restrict__ p_src,
    Array<index_t, SrcDesc::GetNumOfDimension()> src_multi_id_begin,
    DstDesc,
    Float* __restrict__ p_dst,
    Array<index_t, DstDesc::GetNumOfDimension()> dst_multi_id_begin,
    SliceLengths,
    DimAccessOrder,
    Number<DataPerAccess>)
{
    constexpr index_t nDim = SrcDesc::GetNumOfDimension();

    static_assert(nDim == SrcDesc::GetNumOfDimension() && nDim == DstDesc::GetNumOfDimension() &&
                      nDim == SliceLengths::GetSize() && nDim == DimAccessOrder::GetSize(),
                  "wrong! # of dimensions not the same");

    static_assert(is_valid_sequence_map<DimAccessOrder>::value, "wrong! map is not valid");

    // TODO: do more sanity-check here, something like:
    // constexpr auto src_strides_in_access_order =
    //     SrcDesc::ReorderGivenNew2Old(DimAccessOrder{}).GetStride(Number<nDim-1>{});

    // constexpr auto dst_strides_in_access_order =
    //     SrcDesc::ReorderGivenNew2Old(DimAccessOrder{}).GetStride(Number<nDim-1>{});

    // // check src/dst stride on the lowest access dimension
    // static_assert((DataPerAccess == 1 || src_strides_in_access_order.Back() == 1) &&
    //                   (DataPerAccess == 1 || dst_strides_in_access_order.Back() == 1),
    //               "wrong! src/dst stride on the lowest access dimension needs to be 1 for "
    //               "vectorized read/write");

    constexpr auto slice_lengths_in_access_order =
        SliceLengths::ReorderGivenNew2Old(DimAccessOrder{});

    // check slice length on the lowest access dimension
    static_assert(slice_lengths_in_access_order.Back() % DataPerAccess == 0,
                  "wrong! slice length on the lowest access dimension should be evenly divided by "
                  "DataPerAccess");

    constexpr index_t num_access_on_lowest_access_dimension =
        slice_lengths_in_access_order.Back() / DataPerAccess;

    constexpr auto access_lengths = slice_lengths_in_access_order.Modify(
        Number<nDim - 1>{}, Number<num_access_on_lowest_access_dimension>{});

    using vector_t = typename vector_type<Float, DataPerAccess>::MemoryType;

#if CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V1
    static_ford<decltype(access_lengths)>{}([&](auto access_multi_id) {
        constexpr index_t itmp = access_multi_id.Back() * DataPerAccess;

        constexpr auto data_multi_id_in_access_order =
            access_multi_id.Modify(Number<nDim - 1>{}, Number<itmp>{});

        constexpr auto data_multi_id = reorder_array_given_old2new(
            sequence2array(data_multi_id_in_access_order), DimAccessOrder{});

        const index_t src_index =
            SrcDesc::GetOffsetFromMultiIndex(src_multi_id_begin + data_multi_id);

        const index_t dst_index =
            DstDesc::GetOffsetFromMultiIndex(dst_multi_id_begin + data_multi_id);

        *reinterpret_cast<vector_t*>(&p_dst[dst_index]) =
            *reinterpret_cast<const vector_t*>(&p_src[src_index]);
    });
#else
    ford<decltype(access_lengths)>{}([&](auto access_multi_id) {
        auto data_multi_id_in_access_order      = access_multi_id;
        data_multi_id_in_access_order(nDim - 1) = access_multi_id[nDim - 1] * DataPerAccess;

        const auto data_multi_id =
            reorder_array_given_old2new(data_multi_id_in_access_order, DimAccessOrder{});

        const index_t src_index =
            SrcDesc::GetOffsetFromMultiIndex(src_multi_id_begin + data_multi_id);

        const index_t dst_index =
            DstDesc::GetOffsetFromMultiIndex(dst_multi_id_begin + data_multi_id);

        *reinterpret_cast<vector_t*>(&p_dst[dst_index]) =
            *reinterpret_cast<const vector_t*>(&p_src[src_index]);
    });
#endif
}

} // namespace ck
#endif
