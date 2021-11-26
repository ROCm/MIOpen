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
#ifndef CK_CLUSTER_DESCRIPTOR_HPP
#define CK_CLUSTER_DESCRIPTOR_HPP

#include "common_header.hpp"
#include "tensor_adaptor.hpp"

namespace ck {

template <typename Lengths,
          typename ArrangeOrder = typename arithmetic_sequence_gen<0, Lengths::Size(), 1>::type>
__host__ __device__ constexpr auto make_cluster_descriptor(
    const Lengths& lengths,
    ArrangeOrder order = typename arithmetic_sequence_gen<0, Lengths::Size(), 1>::type{})
{
    constexpr index_t ndim_low = Lengths::Size();

    const auto reordered_lengths = container_reorder_given_new2old(lengths, order);

    const auto low_lengths = generate_tuple(
        [&](auto idim_low) { return reordered_lengths[idim_low]; }, Number<ndim_low>{});

    const auto transform = make_merge_transform(low_lengths);

    constexpr auto low_dim_old_top_ids = ArrangeOrder{};

    constexpr auto up_dim_new_top_ids = Sequence<0>{};

    return make_single_stage_tensor_adaptor(
        make_tuple(transform), make_tuple(low_dim_old_top_ids), make_tuple(up_dim_new_top_ids));
}

} // namespace ck
#endif
