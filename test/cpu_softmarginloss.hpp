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
#ifndef GUARD_CPU_SOFTMARGINLOSS_HPP
#define GUARD_CPU_SOFTMARGINLOSS_HPP

#include "tensor_holder.hpp"
#include <miopen/tensor_view_utils.hpp>

template <class T>
void cpu_softmarginloss_unreduced_forward(tensor<T> input, tensor<T> target, tensor<T>& ref_output)
{
    auto input_numel = input.desc.GetElementSize();
    auto i_tv        = miopen::get_inner_expanded_tv<5>(input.desc);
    auto t_tv        = miopen::get_inner_expanded_tv<5>(target.desc);
    auto o_tv        = miopen::get_inner_expanded_tv<5>(ref_output.desc);

    par_ford(input_numel)([&](size_t gid) {
        tensor_layout_t<5> idx(i_tv, gid);
        T i                                       = input[i_tv.get_tensor_view_idx(idx)];
        T t                                       = target[t_tv.get_tensor_view_idx(idx)];
        ref_output[o_tv.get_tensor_view_idx(idx)] = log(1 + exp(-i * t));
    });
}

template <class T>
void cpu_softmarginloss_unreduced_backward(tensor<T> input,
                                           tensor<T> target,
                                           tensor<T> dO,
                                           tensor<T>& ref_dI)
{
    auto input_numel = input.desc.GetElementSize();
    auto i_tv        = miopen::get_inner_expanded_tv<5>(input.desc);
    auto t_tv        = miopen::get_inner_expanded_tv<5>(target.desc);
    auto dO_tv       = miopen::get_inner_expanded_tv<5>(dO.desc);
    auto dI_tv       = miopen::get_inner_expanded_tv<5>(ref_dI.desc);

    par_ford(input_numel)([&](size_t gid) {
        tensor_layout_t<5> idx(i_tv, gid);
        T i                                    = input[i_tv.get_tensor_view_idx(idx)];
        T t                                    = target[t_tv.get_tensor_view_idx(idx)];
        T _dO                                  = dO[dO_tv.get_tensor_view_idx(idx)];
        ref_dI[dI_tv.get_tensor_view_idx(idx)] = -t / (exp(i * t) + 1) * _dO;
    });
}

template <class T>
void cpu_softmarginloss_reduced_forward(tensor<T> input,
                                        tensor<T> target,
                                        tensor<T>& ref_output,
                                        tensor<T>& ref_workspace,
                                        const float divisor)
{
    auto input_numel = input.desc.GetElementSize();
    auto i_tv        = miopen::get_inner_expanded_tv<5>(input.desc);
    auto t_tv        = miopen::get_inner_expanded_tv<5>(target.desc);

    par_ford(input_numel)([&](size_t gid) {
        tensor_layout_t<5> idx(i_tv, gid);
        if(idx.layout[0] >= i_tv.size[0])
            return;
        double i           = input[i_tv.get_tensor_view_idx(idx)];
        double t           = target[t_tv.get_tensor_view_idx(idx)];
        ref_workspace[gid] = log(1 + exp(-i * t));
    });

    double sum = 0;
    for(int i = 0; i < input_numel; i++)
        sum += ref_workspace[i];
    sum /= divisor;
    ref_output[0] = static_cast<T>(sum);
}

template <class T>
void cpu_softmarginloss_reduced_backward(
    tensor<T> input, tensor<T> target, tensor<T> dO, tensor<T>& ref_dI, const float divisor)
{
    auto input_numel = input.desc.GetElementSize();
    auto i_tv        = miopen::get_inner_expanded_tv<5>(input.desc);
    auto t_tv        = miopen::get_inner_expanded_tv<5>(target.desc);
    auto dO_tv       = miopen::get_inner_expanded_tv<5>(dO.desc);
    auto dI_tv       = miopen::get_inner_expanded_tv<5>(ref_dI.desc);

    par_ford(input_numel)([&](size_t gid) {
        tensor_layout_t<5> idx(i_tv, gid);
        float i                                = input[i_tv.get_tensor_view_idx(idx)];
        float t                                = target[t_tv.get_tensor_view_idx(idx)];
        float _dO                              = dO[dO_tv.get_tensor_view_idx(idx)];
        ref_dI[dI_tv.get_tensor_view_idx(idx)] = -t / (exp(i * t) + 1) * _dO / divisor;
    });
}

#endif
