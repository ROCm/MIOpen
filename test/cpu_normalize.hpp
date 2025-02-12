/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include <cmath>
#include <miopen/tensor_view_utils.hpp>
#include "../src/include/miopen/tensor_view_utils.hpp"

template <class T>
void cpu_normalize_backward(const tensor<T> input,
                            const tensor<T> divisor,
                            const tensor<T> output_grad,
                            tensor<T>& input_grad,
                            tensor<float>& reduce,
                            const float p,
                            const float eps,
                            const uint32_t dim)
{
    // Calculate reduce tensor
    auto input_numel              = input.desc.GetElementSize();
    auto inner_size               = input.desc.GetLengths()[dim];
    auto outer_size               = input_numel / inner_size;
    auto input_tv                 = miopen::get_inner_expanded_tv<5>(input.desc);
    auto output_grad_tv           = miopen::get_inner_expanded_tv<5>(output_grad.desc);
    auto transpose_input_tv       = miopen::move_dims_back(input_tv, dim);
    auto transpose_output_grad_tv = miopen::move_dims_back(output_grad_tv, dim);
    for(size_t outer = 0; outer < outer_size; outer++)
    {
        float res = 0;
        for(size_t inner = 0; inner < inner_size; inner++)
        {
            auto gid = outer * inner_size + inner;
            tensor_layout_t<5> idx(transpose_input_tv, gid);
            float i  = input[transpose_input_tv.get_tensor_view_idx(idx)];
            float og = output_grad[transpose_output_grad_tv.get_tensor_view_idx(idx)];
            res += i * og;
        }
        reduce[outer] = res;
    }

    // Calculate input_grad tensor
    auto divisor_tv    = miopen::get_inner_expanded_tv<5>(divisor.desc);
    auto input_grad_tv = miopen::get_inner_expanded_tv<5>(input_grad.desc);
    auto reduce_tv     = miopen::get_inner_expanded_tv<5>(reduce.desc);
    for(size_t gid = 0; gid < input_numel; gid++)
    {
        tensor_layout_t<5> idx(input_grad_tv, gid);
        tensor_layout_t<5> div_idx(idx);
        div_idx.layout[dim] = 0;
        float div           = divisor[divisor_tv.get_tensor_view_idx(div_idx)];
        float dy            = output_grad[output_grad_tv.get_tensor_view_idx(idx)];
        if(eps == div)
        {
            input_grad[input_grad_tv.get_tensor_view_idx(idx)] = dy / eps;
        }
        else
        {
            float x        = input[input_tv.get_tensor_view_idx(idx)];
            float abs_coef = (x < 0 ? -1 : 1);
            float tmp      = -1 / div / std::pow(div, p) * std::pow(abs_coef * x, p - 1) * abs_coef;
            input_grad[input_grad_tv.get_tensor_view_idx(idx)] =
                reduce[reduce_tv.get_tensor_view_idx(div_idx)] * tmp + dy / div;
        }
    }
}

template <class T>
void cpu_norm_forward(
    const tensor<T> input, tensor<T>& divisor, const float p, const float eps, const uint32_t dim)
{
    auto input_numel          = input.desc.GetElementSize();
    auto inner_size           = input.desc.GetLengths()[dim];
    auto outer_size           = input_numel / inner_size;
    auto input_tv             = miopen::get_inner_expanded_tv<5>(input.desc);
    auto divisor_tv           = miopen::get_inner_expanded_tv<5>(divisor.desc);
    auto transpose_input_tv   = miopen::move_dims_back(input_tv, dim);
    auto transpose_divisor_tv = miopen::move_dims_back(divisor_tv, dim);

    for(size_t outer = 0; outer < outer_size; outer++)
    {
        float norm = 0;
        for(size_t inner = 0; inner < inner_size; inner++)
        {
            auto gid = outer * inner_size + inner;
            tensor_layout_t<5> idx(transpose_input_tv, gid);
            float i = input[transpose_input_tv.get_tensor_view_idx(idx)];
            norm += std::pow(abs(i), p);
        }
        tensor_layout_t<5> idx(transpose_divisor_tv, outer);
        divisor[transpose_divisor_tv.get_tensor_view_idx(idx)] =
            std::max(eps, std::pow(norm, 1.0f / p));
    }
}
