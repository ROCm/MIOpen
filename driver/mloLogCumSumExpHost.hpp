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

#include <miopen/miopen.h>
#include <../test/ford.hpp>

#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>

#include <limits>

template <typename Tgpu, typename Tcheck>
int mloLogCumSumExpForwardRunHost(const miopenTensorDescriptor_t inputDesc,
                                  const miopenTensorDescriptor_t outputDesc,
                                  const Tgpu* input,
                                  Tcheck* output_host,
                                  const int dim,
                                  const bool exclusive,
                                  const bool reverse)
{
    const int ndims     = miopen::deref(inputDesc).GetNumDims();
    const auto exec_dim = ((dim % ndims) + ndims) % ndims;

    auto input_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto output_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(outputDesc));

    auto size       = miopen::deref(inputDesc).GetElementSize();
    auto inner_size = miopen::deref(inputDesc).GetLengths()[exec_dim];
    auto outer_size = size / inner_size;

    tensor_view_t<5> ignore_dim_input_tv = input_tv;
    ignore_dim_input_tv.size[exec_dim]   = 1;

    par_ford(outer_size)([&](int gid) {
        auto tensor_layout = tensor_layout_t<5>(ignore_dim_input_tv, gid);
        float cumsum       = 0;

        ford(inner_size)([&](int idx) {
            int tmp_idx =
                (reverse ? input_tv.size[exec_dim] - (idx - exclusive) - 1 : (idx - exclusive));
            float tmp_val = 0;
            if(0 <= tmp_idx && tmp_idx < inner_size)
            {
                tensor_layout.layout[exec_dim] = tmp_idx;
                tmp_val                        = std::exp(
                    static_cast<float>(input[input_tv.get_tensor_view_idx(tensor_layout)]));
            }

            cumsum += tmp_val;

            tensor_layout.layout[exec_dim] = (reverse ? input_tv.size[exec_dim] - idx - 1 : idx);
            output_host[output_tv.get_tensor_view_idx(tensor_layout)] =
                static_cast<Tcheck>(std::log(cumsum));
        });
    });

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int mloLogCumSumExpBackwardRunHost(const miopenTensorDescriptor_t inputDesc,
                                   const miopenTensorDescriptor_t outputDesc,
                                   const miopenTensorDescriptor_t doutputDesc,
                                   const miopenTensorDescriptor_t dinputDesc,
                                   const Tgpu* input,
                                   const Tgpu* output,
                                   const Tgpu* doutput,
                                   Tcheck* dinput_host,
                                   const int dim,
                                   const bool exclusive,
                                   const bool reverse)
{
    const auto input_tv   = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    const auto output_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(outputDesc));
    const auto doutput_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(doutputDesc));
    const auto dinput_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(dinputDesc));

    const auto size     = miopen::deref(inputDesc).GetElementSize();
    const int ndims     = miopen::deref(inputDesc).GetNumDims();
    const auto true_dim = ((dim % ndims) + ndims) % ndims;
    const auto dim_size = miopen::deref(inputDesc).GetLengths()[true_dim];

    auto baseDesc = miopen::TensorDescriptor(miopenFloat, miopen::deref(inputDesc).GetLengths());
    auto base_tv  = miopen::get_inner_expanded_tv<5>(baseDesc);
    auto log_grad_positive        = std::vector<float>(baseDesc.GetElementSize());
    auto log_grad_negative        = std::vector<float>(baseDesc.GetElementSize());
    auto pos_reverse_logcumsumexp = std::vector<float>(baseDesc.GetElementSize());
    auto neg_reverse_logcumsumexp = std::vector<float>(baseDesc.GetElementSize());

    // InitLogGrad
    par_ford(size)([&](int idx) {
        auto tensor_layout = tensor_layout_t<5>(base_tv, idx);

        auto doutput_v = static_cast<float>(doutput[doutput_tv.get_tensor_view_idx(tensor_layout)]);
        auto output_v  = static_cast<float>(output[output_tv.get_tensor_view_idx(tensor_layout)]);

        if(!reverse ? tensor_layout.layout[true_dim] < exclusive
                    : tensor_layout.layout[true_dim] + exclusive >= dim_size)
            log_grad_positive[idx] = log_grad_negative[idx] =
                -std::numeric_limits<float>::infinity();
        else
        {
            log_grad_positive[idx] = (doutput_v > 0 ? std::log(doutput_v) - output_v
                                                    : -std::numeric_limits<float>::infinity());
            log_grad_negative[idx] = (doutput_v < 0 ? std::log(-doutput_v) - output_v
                                                    : -std::numeric_limits<float>::infinity());
        }
    });

    // LogCumSumExpForward pos_reverse_logcumsumexp
    mloLogCumSumExpForwardRunHost(/*inputDesc=*/&baseDesc,
                                  /*outputDesc=*/&baseDesc,
                                  /*input=*/log_grad_positive.data(),
                                  /*output_host=*/pos_reverse_logcumsumexp.data(),
                                  dim,
                                  /*exclusive=*/false,
                                  reverse);

    // LogCumSumExpForward neg_reverse_logcumsumexp
    mloLogCumSumExpForwardRunHost(/*inputDesc=*/&baseDesc,
                                  /*outputDesc=*/&baseDesc,
                                  /*input=*/log_grad_negative.data(),
                                  /*output_host=*/neg_reverse_logcumsumexp.data(),
                                  dim,
                                  /*exclusive=*/false,
                                  reverse);

    // LogCumSumExpBackwardStep2
    par_ford(size)([&](int idx) {
        auto tensor_layout = tensor_layout_t<5>(base_tv, idx);

        if(reverse ? tensor_layout.layout[true_dim] < exclusive
                   : tensor_layout.layout[true_dim] + exclusive >= dim_size)
        {
            dinput_host[dinput_tv.get_tensor_view_idx(tensor_layout)] = static_cast<Tcheck>(0.0f);
            return;
        }
        else
            idx += (!reverse ? exclusive : -exclusive) * base_tv.stride[true_dim];

        auto input_v = static_cast<float>(input[input_tv.get_tensor_view_idx(tensor_layout)]);

        auto output_pos = std::exp(pos_reverse_logcumsumexp[idx] + input_v);
        auto output_neg = std::exp(neg_reverse_logcumsumexp[idx] + input_v);

        dinput_host[dinput_tv.get_tensor_view_idx(tensor_layout)] =
            static_cast<Tcheck>(output_pos - output_neg);
    });

    return miopenStatusSuccess;
}
