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

#include <../test/ford.hpp>
#include "tensor_view.hpp"
#include <miopen/tensor_view_utils.hpp>
#include <miopen/tensor.hpp>

template <typename Tgpu, typename Tcheck>
int32_t mloKLDivLossBackwardRunHost5d(const miopenTensorDescriptor_t inputDesc,
                                      const miopenTensorDescriptor_t targetDesc,
                                      const miopenTensorDescriptor_t outputGradDesc,
                                      const miopenTensorDescriptor_t inputGradDesc,
                                      const miopenTensorDescriptor_t targetGradDesc,
                                      const Tgpu* input,
                                      const Tgpu* target,
                                      const Tgpu* output_grad,
                                      Tcheck* input_grad,
                                      Tcheck* target_grad,
                                      bool log_target,
                                      bool input_grad_out,
                                      bool target_grad_out,
                                      miopenLossReductionMode_t reduction)
{
    auto I_tv  = get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto T_tv  = get_inner_expanded_tv<5>(miopen::deref(targetDesc));
    auto dO_tv = get_inner_expanded_tv<5>(miopen::deref(outputGradDesc));
    auto dI_tv = get_inner_expanded_tv<5>(miopen::deref(inputGradDesc));
    auto dT_tv = get_inner_expanded_tv<5>(miopen::deref(targetGradDesc));

    auto numel = miopen::deref(inputDesc).GetElementSize();
    double d   = 1.0f;
    if(reduction == MIOPEN_LOSS_REDUCTION_MEAN)
    {
        d = static_cast<double>(numel);
    }

    for(size_t i = 0; i < numel; ++i)
    {
        tensor_layout_t<5> tensor_layout = tensor_layout_t<5>(dI_tv, i);
        size_t Iidx                      = I_tv.get_tensor_view_idx(tensor_layout);
        size_t Tidx                      = T_tv.get_tensor_view_idx(tensor_layout);
        size_t dOidx                     = 0;
        if(reduction != MIOPEN_LOSS_REDUCTION_NONE)
        {
            dOidx = dO_tv.get_tensor_view_idx({tensor_layout});
        }
        size_t dIidx = dI_tv.get_tensor_view_idx(tensor_layout);
        size_t dTidx = dT_tv.get_tensor_view_idx(tensor_layout);

        double input_value       = static_cast<double>(input[Iidx]);
        double target_value      = static_cast<double>(target[Tidx]);
        double output_grad_value = static_cast<double>(output_grad[dOidx]);
        double forward_output;

        if(log_target)
        {
            double exp_target = exp(static_cast<double>(target_value));
            forward_output    = exp_target * (target_value - input_value);
            if(input_grad_out)
            {
                input_grad[dIidx] =
                    std::isnan(forward_output)
                        ? static_cast<Tcheck>(0.0f)
                        : static_cast<Tcheck>(-1.0f * exp_target / d * output_grad_value);
            }
            if(target_grad_out)
            {
                target_grad[dTidx] =
                    static_cast<Tcheck>((forward_output + exp_target) / d * output_grad_value);
            }
        }
        else
        {
            forward_output = target_value * (log(target_value) - input_value);
            if(input_grad_out)
            {
                input_grad[dIidx] =
                    std::isnan(forward_output)
                        ? static_cast<Tcheck>(0.0f)
                        : static_cast<Tcheck>(-target_value / d * output_grad_value);
            }
            if(target_grad_out)
            {
                target_grad[dTidx] =
                    (target_value == 0.0f)
                        ? static_cast<Tcheck>(0.0f)
                        : static_cast<Tcheck>((1.0f + log(target_value) - input_value) / d *
                                              output_grad_value);
            }
        }
    }
    return 0;
}
