#ifndef CK_THREADWISE_DYNAMIC_TENSOR_SET_HPP
#define CK_THREADWISE_DYNAMIC_TENSOR_SET_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"

namespace ck {

// Assume:
//   1. Desc is known at compile-time
//   2. Buffer is StaticBuffer
//   3. OriginIdx is known at compile-time
//   4. use #-iterator
template <typename Data,
          typename Desc,
          typename SliceLengths,
          typename std::enable_if<Desc::IsKnownAtCompileTime(), bool>::type = false>
struct ThreadwiseDynamicTensorSliceSet_v1
{
    static constexpr index_t nDim = SliceLengths::Size();

    using Index = MultiIndex<nDim>;

    template <typename OriginIdx, typename Buffer>
    __device__ void Run(const Desc&, const OriginIdx&, Buffer& buf, const Data& initial_value) const
    {
        static_assert(Desc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc and DstDesc need to known at compile-time");

        static_assert(Buffer::IsStaticBuffer(), "wrong! DstBuffer need to be StaticBuffer");

        static_assert(is_known_at_compile_time<remove_cv_t<remove_reference_t<OriginIdx>>>::value,
                      "wrong! OriginIdx need to be known at compile-time");

        // Desc is known at compile-time
        constexpr auto desc = remove_cv_t<remove_reference_t<Desc>>{};

        // OriginIdx is known at compile-time
        constexpr auto origin_idx = to_multi_index(OriginIdx{});

        static_ford<SliceLengths>{}([&](auto access_idx) {
            constexpr auto coord = make_dynamic_tensor_coordinate(desc, origin_idx + access_idx);

            constexpr bool is_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(desc, coord);

            constexpr index_t offset = coord.GetOffset();

            if constexpr(is_valid)
            {
                buf(Number<offset>{}) = initial_value;
            }
        });
    }
};

} // namespace ck
#endif
