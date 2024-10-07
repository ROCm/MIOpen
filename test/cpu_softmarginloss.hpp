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

#include "miopen/miopen.h"
#include "tensor_holder.hpp"
#include <miopen/tensor_view_utils.hpp>

template <class T>
void cpu_softmarginloss_forward(const tensor<T>& input,
                                const tensor<T>& target,
                                tensor<T>& ref_output,
                                miopenLossReductionMode_t reduction_mode)
{
    auto input_numel = input.desc.GetElementSize();
    auto i_tv        = miopen::get_inner_expanded_tv<5>(input.desc);
    auto t_tv        = miopen::get_inner_expanded_tv<5>(target.desc);
    auto o_tv        = miopen::get_inner_expanded_tv<5>(ref_output.desc);

    double sum_loss = 0;
    for(size_t gid = 0; gid < input_numel; gid++)
    {
        tensor_layout_t<5> idx(i_tv, gid);
        // Convert to double for better precision
        double i = input[i_tv.get_tensor_view_idx(idx)];
        double t = target[t_tv.get_tensor_view_idx(idx)];
        if(reduction_mode == MIOPEN_LOSS_REDUCTION_NONE)
            ref_output[o_tv.get_tensor_view_idx(idx)] = log1p(exp(-i * t));
        else
            sum_loss += log1p(exp(-i * t));
    };

    if(reduction_mode == MIOPEN_LOSS_REDUCTION_MEAN)
        ref_output[0] = sum_loss / input_numel;
    else if(reduction_mode == MIOPEN_LOSS_REDUCTION_SUM)
        ref_output[0] = sum_loss;
}

template <class T>
void cpu_softmarginloss_backward(const tensor<T>& input,
                                 const tensor<T>& target,
                                 const tensor<T>& dO,
                                 tensor<T>& ref_dI,
                                 miopenLossReductionMode_t reduction_mode)
{
    auto input_numel = input.desc.GetElementSize();
    auto i_tv        = miopen::get_inner_expanded_tv<5>(input.desc);
    auto t_tv        = miopen::get_inner_expanded_tv<5>(target.desc);
    auto dO_tv       = miopen::get_inner_expanded_tv<5>(dO.desc);
    auto dI_tv       = miopen::get_inner_expanded_tv<5>(ref_dI.desc);

    par_ford(input_numel)([&](size_t gid) {
        tensor_layout_t<5> idx(i_tv, gid);
        // Convert to double for better precision
        double i   = input[i_tv.get_tensor_view_idx(idx)];
        double t   = target[t_tv.get_tensor_view_idx(idx)];
        double _dO = dO[dO_tv.get_tensor_view_idx(idx)];
        if(reduction_mode != MIOPEN_LOSS_REDUCTION_MEAN)
            ref_dI[dI_tv.get_tensor_view_idx(idx)] = -t / (exp(i * t) + 1) * _dO;
        else
            ref_dI[dI_tv.get_tensor_view_idx(idx)] = -t / (exp(i * t) + 1) * _dO / input_numel;
    });
}
