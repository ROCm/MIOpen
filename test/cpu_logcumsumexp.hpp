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
void cpu_logcumsumexp_forward(const tensor<T> input,
                              tensor<T>& ref_output,
                              const int dim,
                              const bool exclusive,
                              const bool reverse)
{
    auto input_tv  = miopen::get_inner_expanded_tv<5>(input.desc);
    auto output_tv = miopen::get_inner_expanded_tv<5>(ref_output.desc);

    const int ndims     = input.desc.GetNumDims();
    const auto true_dim = ((dim % ndims) + ndims) % ndims;

    auto size       = input.desc.GetElementSize();
    auto inner_size = input.desc.GetLengths()[true_dim];
    auto outer_size = size / inner_size;

    tensor_view_t<5> ignore_dim_input_tv = input_tv;
    ignore_dim_input_tv.size[true_dim]   = 1;

    par_ford(outer_size)([&](int gid) {
        auto tensor_layout = tensor_layout_t<5>(ignore_dim_input_tv, gid);
        float cumsum       = 0;

        ford(inner_size)([&](int idx) {
            int tmp_idx =
                (reverse ? input_tv.size[true_dim] - (idx - exclusive) - 1 : (idx - exclusive));
            float tmp_val = 0;
            if(0 <= tmp_idx && tmp_idx < inner_size)
            {
                tensor_layout.layout[true_dim] = tmp_idx;
                tmp_val                        = std::exp(
                    static_cast<float>(input[input_tv.get_tensor_view_idx(tensor_layout)]));
            }

            cumsum += tmp_val;

            tensor_layout.layout[true_dim] = (reverse ? input_tv.size[true_dim] - idx - 1 : idx);
            ref_output[output_tv.get_tensor_view_idx(tensor_layout)] =
                static_cast<T>(std::log(cumsum));
        });
    });
}

template <class T>
void cpu_logcumsumexp_backward(const tensor<T> input,
                               const tensor<T> output,
                               const tensor<T> doutput,
                               tensor<T>& ref_dinput,
                               const int dim,
                               const bool exclusive,
                               const bool reverse)
{
    auto input_tv   = miopen::get_inner_expanded_tv<5>(input.desc);
    auto output_tv  = miopen::get_inner_expanded_tv<5>(output.desc);
    auto doutput_tv = miopen::get_inner_expanded_tv<5>(doutput.desc);
    auto dinput_tv  = miopen::get_inner_expanded_tv<5>(ref_dinput.desc);

    auto size = input.desc.GetElementSize();

    auto log_grad_positive        = tensor<float>{input.desc.GetLengths()};
    auto log_grad_negative        = tensor<float>{input.desc.GetLengths()};
    auto pos_reverse_logcumsumexp = tensor<float>{input.desc.GetLengths()};
    auto neg_reverse_logcumsumexp = tensor<float>{input.desc.GetLengths()};
    auto base_tv                  = miopen::get_inner_expanded_tv<5>(log_grad_positive.desc);

    // InitLogGrad
    par_ford(size)([&](int idx) {
        auto tensor_layout = tensor_layout_t<5>(base_tv, idx);

        auto doutput_v = static_cast<float>(doutput[doutput_tv.get_tensor_view_idx(tensor_layout)]);
        auto output_v  = static_cast<float>(output[output_tv.get_tensor_view_idx(tensor_layout)]);

        log_grad_positive[idx] = (doutput_v > 0 ? std::log(doutput_v) - output_v : std::log(0));
        log_grad_negative[idx] = (doutput_v < 0 ? std::log(-doutput_v) - output_v : std::log(0));
    });

    // LogCumSumExp1dForward pos_reverse_logcumsumexp
    cpu_logcumsumexp_forward(
        log_grad_positive, pos_reverse_logcumsumexp, dim, /*exclusive=*/false, reverse);

    // LogCumSumExp1dForward neg_reverse_logcumsumexp
    cpu_logcumsumexp_forward(
        log_grad_negative, neg_reverse_logcumsumexp, dim, /*exclusive=*/false, reverse);

    // LogCumSumExp1dBackwardStep2
    par_ford(size)([&](int idx) {
        auto tensor_layout = tensor_layout_t<5>(base_tv, idx);

        const int ndims     = input.desc.GetNumDims();
        const auto true_dim = ((dim % ndims) + ndims) % ndims;
        if(tensor_layout.layout[true_dim] + exclusive >= ref_dinput.desc.GetLengths()[true_dim])
        {
            ref_dinput[dinput_tv.get_tensor_view_idx(tensor_layout)] = static_cast<T>(0.0f);
            return;
        }
        else
            idx += exclusive;

        auto input_v = static_cast<float>(input[input_tv.get_tensor_view_idx(tensor_layout)]);

        auto output_pos = std::exp(pos_reverse_logcumsumexp[idx] + input_v);
        auto output_neg = std::exp(neg_reverse_logcumsumexp[idx] + input_v);

        ref_dinput[dinput_tv.get_tensor_view_idx(tensor_layout)] =
            static_cast<T>(output_pos - output_neg);
    });
}
