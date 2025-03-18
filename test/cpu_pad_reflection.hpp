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

#include <miopen/tensor_view_utils.hpp>

#include "tensor_holder.hpp"

template <class T>
void cpu_pad_reflection_fwd(tensor<T> input_tensor,
                            tensor<T>& ref_output_tensor,
                            const std::vector<int64_t> padding)
{
    auto input_size     = input_tensor.desc.GetNumDims();
    auto input_dims     = input_tensor.desc.GetLengths();
    auto output_dims    = ref_output_tensor.desc.GetLengths();
    auto input          = input_tensor.data.data();
    auto output         = ref_output_tensor.data.data();
    auto input_strides  = input_tensor.desc.GetStrides();
    auto output_strides = ref_output_tensor.desc.GetStrides();
    auto output_size =
        std::accumulate(output_dims.begin(), output_dims.end(), 1L, std::multiplies<int64_t>());

    int contiguous = input_tensor.desc.IsContiguous();

    if(input_size == 3 && contiguous == 1)
    {
        long padding_l = padding[0];
        size_t in_W    = input_dims[2];

        for(size_t gid = 0; gid < output_size; ++gid)
        {
            long n, c, w;
            ulong nc = gid / output_dims[2];
            w        = gid % output_dims[2];
            n        = nc / output_dims[1];
            c        = nc % output_dims[1];

            long in_start_x  = std::max(0L, -padding_l);
            long out_start_x = std::max(0L, padding_l);

            if(w < padding_l)
            {
                w = padding_l * 2 - w;
            }
            else if(padding_l <= w && w < in_W + padding_l)
            {
            }
            else
            {
                w = (in_W + padding_l - 1) * 2 - w;
            }
            w           = w - out_start_x + in_start_x;
            output[gid] = input[(input_strides[2] * (w)) + (input_strides[1] * (c)) +
                                (input_strides[0] * (n)) + 0];
        }
    }
    else if(input_size == 3 && contiguous == 0)
    {
        long padding_l = padding[0];
        size_t in_W    = input_dims[2];

        for(size_t gid = 0; gid < output_size; ++gid)
        {
            long n, c, w;
            ulong nc = gid / output_dims[2];
            w        = gid % output_dims[2];
            n        = nc / output_dims[1];
            c        = nc % output_dims[1];

            long in_start_x  = std::max(0L, -padding_l);
            long out_start_x = std::max(0L, padding_l);

            if(w < padding_l)
            {
                w = padding_l * 2 - w;
            }
            else if(padding_l <= w && w < in_W + padding_l)
            {
            }
            else
            {
                w = (in_W + padding_l - 1) * 2 - w;
            }
            w                 = w - out_start_x + in_start_x;
            size_t output_idx = output_strides[0] * (gid / output_dims[2] / output_dims[1]) +
                                output_strides[1] * ((gid / output_dims[2]) % output_dims[1]) +
                                output_strides[2] * (gid % output_dims[2]) + 0;
            output[output_idx] = input[(input_strides[2] * (w)) + (input_strides[1] * (c)) +
                                       (input_strides[0] * (n)) + 0];
        }
    }
}

template <class T>
void cpu_pad_reflection_bwd(tensor<T>& input_grad,
                            tensor<T> output_grad,
                            const std::vector<int64_t> padding)
{
    std::fill(input_grad.data.begin(), input_grad.data.end(), 0);

    auto input_grad_tv  = miopen::get_inner_expanded_tv<3>(input_grad.desc);
    auto output_grad_tv = miopen::get_inner_expanded_tv<3>(output_grad.desc);

    auto output_grad_size = output_grad.desc.GetElementSize();

    int64_t padding_l = padding[0];
    uint64_t in_W     = input_grad_tv.size[2];

    for(int64_t gid = 0; gid < output_grad_size; ++gid)
    {
        auto i_tensor_layout = tensor_layout_t<3>(output_grad_tv, gid);
        auto o_tensor_layout = tensor_layout_t<3>(output_grad_tv, gid);

        int64_t w = i_tensor_layout.layout[2];

        int64_t in_start_x  = std::max(0L, -padding_l);
        int64_t out_start_x = std::max(0L, padding_l);

        w = (w < padding_l)          ? (2 * padding_l - w)
            : (w < in_W + padding_l) ? w
                                     : (2 * (in_W + padding_l - 1) - w);

        i_tensor_layout.layout[2] = w - out_start_x + in_start_x;

        input_grad[input_grad_tv.get_tensor_view_idx(i_tensor_layout)] +=
            output_grad[output_grad_tv.get_tensor_view_idx(o_tensor_layout)];
    }
}
