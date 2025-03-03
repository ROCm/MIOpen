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
void cpu_softmaxcrossentropywithlogits_forward(tensor<T> input,
                                               tensor<T> target,
                                               tensor<T>& output,
                                               tensor<T>& backprop)
{
    auto I_tv = get_inner_expanded_tv<2>(input.desc);
    auto T_tv = get_inner_expanded_tv<2>(target.desc);
    auto O_tv = get_inner_expanded_tv<1>(output.desc);
    auto B_tv = get_inner_expanded_tv<2>(backprop.desc);

    size_t num_batches = I_tv.size[0];
    size_t num_class   = I_tv.size[1];

    for(size_t gid = 0; gid < num_batches; ++gid)
    {
        float max_val = -std::numeric_limits<float>::infinity();

        for(size_t i = 0; i < num_class; ++i)
        {
            size_t Iidx = I_tv.get_tensor_view_idx({gid, i});
            float val   = static_cast<float>(input[Iidx]);
            max_val     = std::max(max_val, val);
        }

        float sum = 0.0f;
        for(size_t i = 0; i < num_class; ++i)
        {
            size_t Iidx = I_tv.get_tensor_view_idx({gid, i});
            sum += std::exp(static_cast<float>(input[Iidx]) - max_val);
        }

        float log_sum;
        if(sum != 0)
            log_sum = std::log(sum);
        else
            log_sum = std::log(std::numeric_limits<float>::epsilon());

        float loss = 0.0f;
        for(size_t i = 0; i < num_class; ++i)
        {
            size_t Iidx = I_tv.get_tensor_view_idx({gid, i});
            size_t Tidx = T_tv.get_tensor_view_idx({gid, i});
            float val   = static_cast<float>(input[Iidx]);
            float label = static_cast<float>(target[Tidx]);
            loss += label * (log_sum - val + max_val);
        }

        size_t Oidx  = O_tv.get_tensor_view_idx({gid});
        output[Oidx] = static_cast<T>(loss);

        for(size_t i = 0; i < num_class; ++i)
        {
            size_t Iidx        = I_tv.get_tensor_view_idx({gid, i});
            size_t Tidx        = T_tv.get_tensor_view_idx({gid, i});
            size_t Bidx        = B_tv.get_tensor_view_idx({gid, i});
            float val          = static_cast<float>(input[Iidx]);
            float label        = static_cast<float>(target[Tidx]);
            float backprop_val = std::exp(val - max_val) / sum - label;
            backprop[Bidx]     = static_cast<T>(backprop_val);
        }
    }
}

template <class T>
void cpu_softmaxcrossentropywithlogits_backward(tensor<T> output_grad,
                                                tensor<T> backprop,
                                                tensor<T> input,
                                                tensor<T>& input_grad,
                                                tensor<T>& target_grad,
                                                bool input_grad_out,
                                                bool target_grad_out)
{
    auto dO_tv = get_inner_expanded_tv<1>(output_grad.desc);
    auto B_tv  = get_inner_expanded_tv<2>(backprop.desc);
    auto I_tv  = get_inner_expanded_tv<2>(input.desc);

    size_t num_batches = I_tv.size[0];
    size_t num_class   = I_tv.size[1];

    for(size_t gid = 0; gid < num_batches; ++gid)
    {
        size_t dOidx          = dO_tv.get_tensor_view_idx({gid});
        float output_grad_val = static_cast<float>(output_grad[dOidx]);

        if(input_grad_out)
        {
            auto dI_tv = get_inner_expanded_tv<2>(input_grad.desc);

            for(size_t i = 0; i < num_class; ++i)
            {
                size_t Bidx        = B_tv.get_tensor_view_idx({gid, i});
                size_t dIidx       = dI_tv.get_tensor_view_idx({gid, i});
                float backprop_val = static_cast<float>(backprop[Bidx]);
                input_grad[dIidx]  = static_cast<T>(output_grad_val * backprop_val);
            }
        }

        if(target_grad_out)
        {
            auto dT_tv = get_inner_expanded_tv<2>(target_grad.desc);

            float max_val = -std::numeric_limits<float>::infinity();
            for(size_t i = 0; i < num_class; ++i)
            {
                size_t Iidx = I_tv.get_tensor_view_idx({gid, i});
                float val   = static_cast<float>(input[Iidx]);
                max_val     = std::max(max_val, val);
            }

            float sum = 0.0f;
            for(size_t i = 0; i < num_class; ++i)
            {
                size_t Iidx = I_tv.get_tensor_view_idx({gid, i});
                float val   = static_cast<float>(input[Iidx]);
                sum += std::exp(val - max_val);
            }

            float log_val;
            if(sum != 0)
                log_val = std::log(sum);
            else
                log_val = std::log(std::numeric_limits<float>::epsilon());

            for(size_t i = 0; i < num_class; ++i)
            {
                size_t Iidx     = I_tv.get_tensor_view_idx({gid, i});
                float logit_val = static_cast<float>(input[Iidx]);
                size_t Tidx     = dT_tv.get_tensor_view_idx({gid, i});
                target_grad[Tidx] =
                    static_cast<T>((max_val + log_val - logit_val) * output_grad_val);
            }
        }
    }
}
