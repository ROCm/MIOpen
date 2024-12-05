/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#pragma once

#include "tensor_holder.hpp"
#include <miopen/tensor_view_utils.hpp>

template <class T, class TS>
void cpu_UnsortedSegmentSum_forward(const tensor<T>& input,
                                    tensor<T>& output,
                                    const tensor<TS>& segment_ids,
                                    const uint64_t num_segments)
{
    uint64_t N              = input.desc.GetElementSize();
    uint64_t inner_dim_size = input.desc.GetElementSize() / segment_ids.desc.GetElementSize();

    ford(N)([&](uint64_t gid) {
        uint64_t input_segment_index  = gid / inner_dim_size;
        uint64_t segment_offset       = gid % inner_dim_size;
        uint64_t output_segment_index = static_cast<uint64_t>(segment_ids[input_segment_index]);

        if(output_segment_index < num_segments)
        {
            uint64_t output_index = output_segment_index * inner_dim_size + segment_offset;
            double val            = static_cast<double>(output[output_index]);
            val += static_cast<double>(input[gid]);
            output[output_index] = static_cast<T>(val);
        }
    });
}

template <class T, class TS>
void cpu_UnsortedSegmentSum_backward(const tensor<T>& output_grad,
                                     tensor<T>& input_grad,
                                     const tensor<TS>& segment_ids,
                                     const uint64_t num_segments)
{
    uint64_t N              = input_grad.desc.GetElementSize();
    uint64_t inner_dim_size = input_grad.desc.GetElementSize() / segment_ids.desc.GetElementSize();

    par_ford(N)([&](uint64_t gid) {
        uint64_t input_segment_index  = gid / inner_dim_size;
        uint64_t segment_offset       = gid % inner_dim_size;
        uint64_t output_segment_index = static_cast<uint64_t>(segment_ids[input_segment_index]);

        if(output_segment_index < num_segments)
        {
            uint64_t output_index = output_segment_index * inner_dim_size + segment_offset;
            input_grad[gid]       = output_grad[output_index];
        }
    });
}
