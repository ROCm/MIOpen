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
#include "tensor_view.hpp"
#include <miopen/tensor_view_utils.hpp>

template <class T>
void cpu_kldivloss_backward_5d(tensor<T> input,
                               tensor<T> target,
                               tensor<T> output_grad,
                               tensor<T>& input_grad,
                               tensor<T>& target_grad,
                               bool log_target,
                               bool input_grad_out,
                               bool target_grad_out,
                               miopenLossReductionMode_t reduction)
{
    auto I_tv  = get_inner_expanded_tv<5>(input.desc);
    auto T_tv  = get_inner_expanded_tv<5>(target.desc);
    auto dO_tv = get_inner_expanded_tv<5>(output_grad.desc);
    auto dI_tv = get_inner_expanded_tv<5>(input_grad.desc);
    auto dT_tv = get_inner_expanded_tv<5>(target_grad.desc);

    double d = 1.0f;
    if(reduction == MIOPEN_LOSS_REDUCTION_MEAN)
    {
        d = static_cast<double>(input.desc.GetElementSize());
    }

    for(size_t i = 0; i < input.desc.GetElementSize(); ++i)
    {
        tensor_layout_t<5> tensor_layout = tensor_layout_t<5>(dI_tv, i);
        size_t Iidx                      = I_tv.get_tensor_view_idx(tensor_layout);
        size_t Tidx                      = T_tv.get_tensor_view_idx(tensor_layout);
        size_t dOidx                     = 0;
        if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
        {
            dOidx = dO_tv.get_tensor_view_idx(tensor_layout);
        }
        size_t dIidx = dI_tv.get_tensor_view_idx(tensor_layout);
        size_t dTidx = dT_tv.get_tensor_view_idx(tensor_layout);

        double input_value       = static_cast<double>(input[Iidx]);
        double target_value      = static_cast<double>(target[Tidx]);
        double output_grad_value = static_cast<double>(output_grad[dOidx]);
        double forward_output;

        if(log_target)
        {
            double exp_target = exp(target_value);
            forward_output    = exp_target * (target_value - input_value);
            if(input_grad_out)
            {
                input_grad[dIidx] = std::isnan(forward_output)
                                        ? static_cast<T>(0.0f)
                                        : static_cast<T>(-exp_target / d * output_grad_value);
            }
            if(target_grad_out)
            {
                target_grad[dTidx] =
                    static_cast<T>((forward_output + exp_target) / d * output_grad_value);
            }
        }
        else
        {
            forward_output = target_value * (log(target_value) - input_value);
            if(input_grad_out)
            {
                input_grad[dIidx] = std::isnan(forward_output)
                                        ? static_cast<T>(0.0f)
                                        : static_cast<T>(-target_value / d * output_grad_value);
            }
            if(target_grad_out)
            {
                target_grad[dTidx] = (target_value == 0.0f)
                                         ? static_cast<T>(0.0f)
                                         : static_cast<T>((1.0f + log(target_value) - input_value) /
                                                          d * output_grad_value);
            }
        }
    }
}
