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

template <class T>
void cpu_maskedfill_forward(const tensor<T>& input,
                            tensor<T>& output,
                            const tensor<int8_t>& mask,
                            float value)
{
    auto input_tv  = miopen::get_inner_expanded_tv<5>(input.desc);
    auto output_tv = miopen::get_inner_expanded_tv<5>(output.desc);
    auto mask_tv   = miopen::get_inner_expanded_tv<5>(mask.desc);
    par_ford(output.desc.GetElementSize())([&](size_t gid) {
        tensor_layout_t<5> output_layout{output_tv, gid};
        tensor_layout_t<5> input_layout{input_tv, gid};
        tensor_layout_t<5> mask_layout{mask_tv, gid};
        output[output_tv.get_tensor_view_idx(output_layout)] =
            mask[mask_tv.get_tensor_view_idx(mask_layout)]
                ? static_cast<T>(value)
                : input[input_tv.get_tensor_view_idx(input_layout)];
    });
}

template <class T>
void cpu_maskedfill_backward(const tensor<T>& output_grad,
                             tensor<T>& input_grad,
                             const tensor<int8_t>& mask)
{
    auto output_grad_tv = miopen::get_inner_expanded_tv<5>(output_grad.desc);
    auto input_grad_tv  = miopen::get_inner_expanded_tv<5>(input_grad.desc);
    auto mask_tv        = miopen::get_inner_expanded_tv<5>(mask.desc);
    par_ford(input_grad.desc.GetElementSize())([&](size_t gid) {
        tensor_layout_t<5> output_grad_layout{output_grad_tv, gid};
        tensor_layout_t<5> input_grad_layout{input_grad_tv, gid};
        tensor_layout_t<5> mask_layout{mask_tv, gid};
        input_grad[input_grad_tv.get_tensor_view_idx(input_grad_layout)] =
            mask[mask_tv.get_tensor_view_idx(mask_layout)]
                ? static_cast<T>(0)
                : output_grad[output_grad_tv.get_tensor_view_idx(output_grad_layout)];
    });
}
