/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#ifndef CK_THREADWISE_TENSOR_SET_HPP
#define CK_THREADWISE_TENSOR_SET_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"

namespace ck {

// Assume:
//   1. Desc is known at compile-time
//   2. Buffer is StaticBuffer
//   3. OriginIdx is known at compile-time
//   4. use #-step
template <typename Data,
          typename Desc,
          typename SliceLengths,
          typename enable_if<Desc::IsKnownAtCompileTime(), bool>::type = false>
struct ThreadwiseTensorSliceSet_v1
{
    static constexpr index_t nDim = SliceLengths::Size();

    using Index = MultiIndex<nDim>;

    template <typename OriginIdx, typename Buffer>
    __device__ void Run(const Desc&, const OriginIdx&, Buffer& buf, const Data& initial_value) const
    {
        static_assert(Desc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc and DstDesc need to known at compile-time");

        static_assert(Buffer::IsStaticBuffer(), "wrong! DstBuffer need to be StaticBuffer");

        static_assert(is_known_at_compile_time<remove_cvref_t<OriginIdx>>::value,
                      "wrong! OriginIdx need to be known at compile-time");

        // Desc is known at compile-time
        constexpr auto desc = remove_cvref_t<Desc>{};

        // OriginIdx is known at compile-time
        constexpr auto origin_idx = to_multi_index(OriginIdx{});

        static_ford<SliceLengths>{}([&](auto access_idx) {
            constexpr auto coord = make_tensor_coordinate(desc, origin_idx + access_idx);

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
