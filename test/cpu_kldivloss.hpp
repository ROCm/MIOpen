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
#ifndef GUARD_CPU_KLDIVLOSS_HPP
#define GUARD_CPU_KLDIVLOSS_HPP

#include "tensor_holder.hpp"
#include <miopen/tensor_view.hpp>

template <class T>
void cpu_kldivloss_unreduced_forward_5d(tensor<T> input,
                                        tensor<T> target,
                                        tensor<T>& output,
                                        bool log_target)
{
    auto I_tv = get_inner_expanded_tv_5d(input.desc);
    auto T_tv = get_inner_expanded_tv_5d(target.desc);
    auto O_tv = get_inner_expanded_tv_5d(output.desc);

    for(size_t i = 0; i < input.desc.GetElementSize(); ++i)
    {
        uint64_t n[5];
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], i, O_tv);
        size_t Iidx = TV5D_IDX(I_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t Tidx = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t Oidx = TV5D_IDX(O_tv, n[0], n[1], n[2], n[3], n[4]);

        T input_value  = input[Iidx];
        T target_value = target[Tidx];
        T forward_output;

        if(log_target)
        {
            forward_output = static_cast<T>(exp(target_value)) * (target_value - input_value);
        }
        else
        {
            forward_output = target_value * (static_cast<T>(log(target_value)) - input_value);
        }
        output[Oidx] = std::isnan(forward_output) ? 0.0f : forward_output;
    }
}

template <class T>
void cpu_kldivloss_unreduced_backward_5d(tensor<T> input,
                                         tensor<T> target,
                                         tensor<T> output_grad,
                                         tensor<T>& input_grad,
                                         tensor<T>& target_grad,
                                         bool log_target,
                                         bool input_grad_out,
                                         bool target_grad_out)
{
    auto I_tv  = get_inner_expanded_tv_5d(input.desc);
    auto T_tv  = get_inner_expanded_tv_5d(target.desc);
    auto dO_tv = get_inner_expanded_tv_5d(output_grad.desc);
    auto dI_tv = get_inner_expanded_tv_5d(input_grad.desc);
    auto dT_tv = get_inner_expanded_tv_5d(target_grad.desc);

    for(size_t i = 0; i < input.desc.GetElementSize(); ++i)
    {
        uint64_t n[5];
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], i, dI_tv);
        size_t Iidx  = TV5D_IDX(I_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t Tidx  = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t dOidx = TV5D_IDX(dO_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t dIidx = TV5D_IDX(dI_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t dTidx = TV5D_IDX(dT_tv, n[0], n[1], n[2], n[3], n[4]);

        T input_value       = input[Iidx];
        T target_value      = target[Tidx];
        T output_grad_value = output_grad[dOidx];
        T forward_output;

        if(log_target)
        {
            T exp_target   = static_cast<T>(exp(target_value));
            forward_output = exp_target * (target_value - input_value);
            if(input_grad_out)
            {
                input_grad[dIidx] = std::isnan(forward_output)
                                        ? static_cast<T>(0.0f)
                                        : static_cast<T>(-1.0f) * exp_target * output_grad_value;
            }
            if(target_grad_out)
            {
                target_grad[dTidx] =
                    static_cast<T>(forward_output + exp_target) * output_grad_value;
            }
        }
        else
        {
            forward_output = target_value * (static_cast<T>(log(target_value)) - input_value);
            if(input_grad_out)
            {
                input_grad[dIidx] =
                    std::isnan(forward_output) ? 0.0f : -target_value * output_grad_value;
            }
            if(target_grad_out)
            {
                target_grad[dTidx] = (target_value == 0)
                                         ? 0.0f
                                         : (1 + (static_cast<T>(log(target_value)) - input_value)) *
                                               output_grad_value;
            }
        }
    }
}

template <class T>
void cpu_kldivloss_reduced_forward_5d(tensor<T> input,
                                      tensor<T> target,
                                      tensor<T>& output,
                                      tensor<T>& workspace,
                                      float divisor,
                                      bool log_target)
{
    auto I_tv  = get_inner_expanded_tv_5d(input.desc);
    auto T_tv  = get_inner_expanded_tv_5d(target.desc);
    auto numel = input.desc.GetElementSize();
    for(size_t i = 0; i < numel; ++i)
    {
        uint64_t n[5];
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], i, I_tv);
        size_t Iidx = TV5D_IDX(I_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t Tidx = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);

        float input_value  = static_cast<float>(input[Iidx]);
        float target_value = static_cast<float>(target[Tidx]);
        float forward_output;

        if(log_target)
        {
            forward_output = exp(target_value) * (target_value - input_value) / divisor;
        }
        else
        {
            forward_output = target_value * (log(target_value) - input_value) / divisor;
        }
        workspace[i] =
            std::isnan(forward_output) ? static_cast<T>(0.0f) : static_cast<T>(forward_output);
    }

    const int local_size = 256;
    int offset_a         = 0;
    int offset_b         = numel;
    size_t _size         = numel;
    do
    {
        for(int i = 0; i < _size; i += local_size)
        {
            float shared[local_size];
            par_ford(local_size)([&](size_t j) {
                shared[j] = i + j < _size ? static_cast<float>(workspace[offset_a + i + j]) : 0.0f;
            });
            for(int offset = local_size / 2; offset > 0; offset >>= 1)
                ford(local_size)([&](size_t j) {
                    if(j < offset)
                        shared[j] += shared[j + offset];
                });
            if(_size <= local_size)
                output[0] = static_cast<T>(shared[0]);
            else
                workspace[offset_b + i / local_size] = static_cast<T>(shared[0]);
        }
        std::swap(offset_a, offset_b);
        _size = (_size + local_size - 1) / local_size;
    } while(_size > 1);
}

template <class T>
void cpu_kldivloss_reduced_backward_5d(tensor<T> input,
                                       tensor<T> target,
                                       tensor<T> output_grad,
                                       tensor<T>& input_grad,
                                       tensor<T>& target_grad,
                                       float divisor,
                                       bool log_target,
                                       bool input_grad_out,
                                       bool target_grad_out)
{
    auto I_tv  = get_inner_expanded_tv_5d(input.desc);
    auto T_tv  = get_inner_expanded_tv_5d(target.desc);
    auto dO_tv = get_inner_expanded_tv_1d(output_grad.desc);
    auto dI_tv = get_inner_expanded_tv_5d(input_grad.desc);
    auto dT_tv = get_inner_expanded_tv_5d(target_grad.desc);

    for(size_t i = 0; i < input.desc.GetElementSize(); ++i)
    {
        uint64_t n[5];
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], i, dI_tv);
        size_t Iidx  = TV5D_IDX(I_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t Tidx  = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t dOidx = TV1D_IDX(dO_tv, 0);
        size_t dIidx = TV5D_IDX(dI_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t dTidx = TV5D_IDX(dT_tv, n[0], n[1], n[2], n[3], n[4]);

        T input_value       = input[Iidx];
        T target_value      = target[Tidx];
        T output_grad_value = output_grad[dOidx];
        T forward_output;
        T d = static_cast<T>(divisor);

        if(log_target)
        {
            T exp_target   = static_cast<T>(exp(static_cast<double>(target_value)));
            forward_output = exp_target * (target_value - input_value);
            if(input_grad_out)
            {
                input_grad[dIidx] =
                    std::isnan(forward_output)
                        ? static_cast<T>(0.0f)
                        : static_cast<T>(-1.0f) * exp_target / d * output_grad_value;
            }
            if(target_grad_out)
            {
                target_grad[dTidx] =
                    static_cast<T>(forward_output + exp_target) / d * output_grad_value;
            }
        }
        else
        {
            forward_output = target_value *
                             (static_cast<T>(log(static_cast<double>(target_value))) - input_value);
            if(input_grad_out)
            {
                input_grad[dIidx] = std::isnan(forward_output)
                                        ? static_cast<T>(0.0f)
                                        : -target_value / d * output_grad_value;
            }
            if(target_grad_out)
            {
                target_grad[dTidx] =
                    (target_value == static_cast<T>(0.0f))
                        ? static_cast<T>(0.0f)
                        : (static_cast<T>(1.0f) +
                           (static_cast<T>(log(static_cast<double>(target_value))) - input_value)) /
                              d * output_grad_value;
            }
        }
    }
}

#endif
