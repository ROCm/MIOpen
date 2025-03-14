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

#include "miopen/miopen.h"
#include "tensor_holder.hpp"
#include <cmath>
#include <miopen/tensor_view_utils.hpp>

template <class T>
void cpu_hinge_embedding_loss_forward(const tensor<T> input,
                                      const tensor<uint8_t> target,
                                      tensor<T>& output,
                                      const float margin,
                                      const miopenLossReductionMode_t reduction_mode)
{
    auto input_tv       = miopen::get_inner_expanded_tv<5>(input.desc);
    auto target_tv      = miopen::get_inner_expanded_tv<5>(target.desc);
    auto output_tv      = miopen::get_inner_expanded_tv<5>(output.desc);
    const auto input_sz = input.desc.GetElementSize();
    double sum_loss     = 0;
    for(size_t gid = 0; gid < input_sz; ++gid)
    {
        tensor_layout_t<5> idx(input_tv, gid);
        double loss;
        if(target[target_tv.get_tensor_view_idx(idx)] == 1)
            loss = input[input_tv.get_tensor_view_idx(idx)];
        else
            loss = std::max(0.0f, margin - input[input_tv.get_tensor_view_idx(idx)]);
        if(reduction_mode != MIOPEN_LOSS_REDUCTION_NONE)
            sum_loss += loss;
        else
            output[output_tv.get_tensor_view_idx(idx)] = loss;
    }
    if(reduction_mode == MIOPEN_LOSS_REDUCTION_MEAN)
    {
        output[0] = static_cast<T>(sum_loss / input_sz);
    }
    else if(reduction_mode == MIOPEN_LOSS_REDUCTION_SUM)
    {
        output[0] = static_cast<T>(sum_loss);
    }
}

template <class T>
void cpu_hinge_embedding_loss_backward(const tensor<T> input,
                                       const tensor<uint8_t> target,
                                       const tensor<T> output_grad,
                                       tensor<T>& input_grad,
                                       const float margin,
                                       const miopenLossReductionMode_t reduction_mode)
{
    auto input_tv       = miopen::get_inner_expanded_tv<5>(input.desc);
    auto target_tv      = miopen::get_inner_expanded_tv<5>(target.desc);
    auto output_grad_tv = miopen::get_inner_expanded_tv<5>(output_grad.desc);
    auto input_grad_tv  = miopen::get_inner_expanded_tv<5>(input_grad.desc);
    const auto input_sz = input.desc.GetElementSize();
    for(size_t gid = 0; gid < input_sz; ++gid)
    {
        tensor_layout_t<5> idx(input_tv, gid);
        if(target[target_tv.get_tensor_view_idx(idx)] == 1)
        {
            if(reduction_mode == MIOPEN_LOSS_REDUCTION_NONE)
            {
                input_grad[input_grad_tv.get_tensor_view_idx(idx)] =
                    output_grad[output_grad_tv.get_tensor_view_idx(idx)];
            }
            else if(reduction_mode == MIOPEN_LOSS_REDUCTION_SUM)
            {
                input_grad[input_grad_tv.get_tensor_view_idx(idx)] = output_grad[0];
            }
            else
            {
                input_grad[input_grad_tv.get_tensor_view_idx(idx)] =
                    static_cast<double>(output_grad[0]) / input_sz;
            }
        }
        else
        {
            if(margin - static_cast<double>(input[input_tv.get_tensor_view_idx(idx)]) > 0)
            {
                if(reduction_mode == MIOPEN_LOSS_REDUCTION_NONE)
                {
                    input_grad[input_grad_tv.get_tensor_view_idx(idx)] =
                        -static_cast<double>(output_grad[output_grad_tv.get_tensor_view_idx(idx)]);
                }
                else if(reduction_mode == MIOPEN_LOSS_REDUCTION_SUM)
                {
                    input_grad[input_grad_tv.get_tensor_view_idx(idx)] =
                        -static_cast<double>(output_grad[0]);
                }
                else
                {
                    input_grad[input_grad_tv.get_tensor_view_idx(idx)] =
                        -static_cast<double>(output_grad[0]) / input_sz;
                }
            }
            else
            {
                input_grad[input_grad_tv.get_tensor_view_idx(idx)] = 0;
            }
        }
    }
}
