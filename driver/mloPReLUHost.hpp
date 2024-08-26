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

#include <../test/ford.hpp>

#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <miopen/prelu/utils.hpp>

template <typename Tgpu, typename Tcheck>
int32_t mloPReLUBackwardRunHost(const miopenTensorDescriptor_t inputDesc,
                                const miopenTensorDescriptor_t weightDesc,
                                const miopenTensorDescriptor_t doutputDesc,
                                const miopenTensorDescriptor_t dinputDesc,
                                const Tgpu* input,
                                const Tgpu* weight,
                                const Tgpu* doutput,
                                Tcheck* dinput_host,
                                Tcheck* dweight_host)
{
    auto input_tv   = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto doutput_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(doutputDesc));
    auto dinput_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(dinputDesc));

    auto input_sz              = miopen::deref(inputDesc).GetElementSize();
    auto weight_sz             = miopen::deref(weightDesc).GetElementSize();
    auto weight_grad_collector = std::vector<float>(input_sz);

    par_ford(input_sz)([&](int gid) {
        auto tensor_layout = tensor_layout_t<5>(input_tv, gid);
        float input_v      = static_cast<float>(input[input_tv.get_tensor_view_idx(tensor_layout)]);
        float grad_v = static_cast<float>(doutput[doutput_tv.get_tensor_view_idx(tensor_layout)]);

        if(dinput_host)
        {
            float weight_v;
            if(weight_sz == 1)
                weight_v = static_cast<float>(weight[0]);
            else
                weight_v = static_cast<float>(weight[tensor_layout.layout[1]]);
            float input_grad_v = input_v > 0 ? grad_v : weight_v * grad_v;
            dinput_host[dinput_tv.get_tensor_view_idx(tensor_layout)] =
                static_cast<Tcheck>(input_grad_v);
        }
        if(dweight_host)
        {
            weight_grad_collector[gid] = input_v > 0 ? 0 : input_v * grad_v;
        }
    });

    if(dweight_host)
    {
        if(weight_sz == 1)
        {
            double sum = 0;
            for(int i = 0; i < input_sz; ++i)
                sum += static_cast<double>(weight_grad_collector[i]);
            dweight_host[0] = static_cast<Tcheck>(sum);
        }
        else
        {
            size_t inner_size = std::accumulate(
                &input_tv.size[2], &input_tv.size[4], 1ul, std::multiplies<size_t>());
            size_t outer_size = inner_size * input_tv.size[1];
            par_ford(weight_sz)([&](int i) {
                double sum = 0;
                ford(input_tv.size[0])([&](int j) {
                    ford(inner_size)([&](int k) {
                        sum += static_cast<double>(
                            weight_grad_collector[j * outer_size + i * inner_size + k]);
                    });
                });
                dweight_host[i] = static_cast<Tcheck>(sum);
            });
        }
    }

    return miopenStatusSuccess;
}
