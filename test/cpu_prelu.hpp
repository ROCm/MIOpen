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
void cpu_prelu_backward(const tensor<T> input,
                        const tensor<T> weight,
                        const tensor<T> output_grad,
                        tensor<T>& ref_input_grad,
                        tensor<T>& ref_weight_grad,
                        const bool has_dinput  = true,
                        const bool has_dweight = true)
{
    auto N              = input.desc.GetElementSize();
    auto input_tv       = miopen::get_inner_expanded_tv<5>(input.desc);
    auto weight_tv      = miopen::get_inner_expanded_tv<1>(weight.desc);
    auto output_grad_tv = miopen::get_inner_expanded_tv<5>(output_grad.desc);
    auto input_grad_tv  = miopen::get_inner_expanded_tv<5>(ref_input_grad.desc);
    auto weight_grad_tv = miopen::get_inner_expanded_tv<1>(ref_weight_grad.desc);

    auto weight_grad_collector = std::vector<float>(N);

    par_ford(N)([&](int gid) {
        auto tensor_layout = tensor_layout_t<5>(input_tv, gid);
        float input_v      = static_cast<float>(input[input_tv.get_tensor_view_idx(tensor_layout)]);
        float grad_v =
            static_cast<float>(output_grad[output_grad_tv.get_tensor_view_idx(tensor_layout)]);

        if(has_dinput)
        {
            float weight_v;
            weight_v = static_cast<float>(
                weight[weight.desc.GetElementSize() == 1
                           ? 0
                           : weight_tv.get_tensor_view_idx({tensor_layout.layout[1]})]);

            float input_grad_v = input_v > 0 ? grad_v : weight_v * grad_v;
            ref_input_grad[input_grad_tv.get_tensor_view_idx(tensor_layout)] =
                static_cast<T>(input_grad_v);
        }
        if(has_dweight)
        {
            weight_grad_collector[gid] = input_v > 0 ? 0 : input_v * grad_v;
        }
    });

    if(has_dweight)
    {
        if(weight.desc.GetElementSize() == 1)
        {
            double sum = 0;
            for(int i = 0; i < N; ++i)
                sum += static_cast<double>(weight_grad_collector[i]);
            ref_weight_grad[0] = static_cast<T>(sum);
        }
        else
        {
            size_t inner_size = std::accumulate(
                &input_tv.size[2], &input_tv.size[4], 1ul, std::multiplies<size_t>());
            size_t outer_size = inner_size * input_tv.size[1];
            par_ford(weight.desc.GetElementSize())([&](int i) {
                double sum = 0;
                ford(input_tv.size[0])([&](int j) {
                    ford(inner_size)([&](int k) {
                        sum += static_cast<double>(
                            weight_grad_collector[j * outer_size + i * inner_size + k]);
                    });
                });
                ref_weight_grad[weight_grad_tv.get_tensor_view_idx({i})] = static_cast<T>(sum);
            });
        }
    }
}
