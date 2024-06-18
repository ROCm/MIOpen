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

#ifndef GUARD_CPU_MULTILABELSOFTMARGINLOSS_HPP
#define GUARD_CPU_MULTILABELSOFTMARGINLOSS_HPP

#include "tensor_holder.hpp"
#include <miopen/tensor_view_utils.hpp>
#include <math.h>

float sigmoid(float x) { return 1 / (1 + exp(-x)); }
float calc_loss(float x, float y)
{
    float sig = sigmoid(x);
    return y * log(sig) + (1 - y) * log(1 - sig);
}

template <class T>
void cpu_multilabelsoftmarginloss_unreduced_forward(tensor<T> input,
                                                    tensor<T> target,
                                                    tensor<T> weight,
                                                    tensor<T>& ref_output)
{
    auto N    = input.desc.GetLengths()[0];
    auto C    = input.desc.GetLengths()[1];
    auto i_tv = miopen::get_inner_expanded_tv<2>(input.desc);
    auto t_tv = miopen::get_inner_expanded_tv<2>(target.desc);
    auto w_tv = miopen::get_inner_expanded_tv<1>(weight.desc);
    auto o_tv = miopen::get_inner_expanded_tv<1>(ref_output.desc);

    par_ford(N)([&](size_t n) {
        float loss = 0;
        // Convert to float for better precision
        for(size_t c = 0; c < C; c++)
        {
            float w = weight[w_tv.get_tensor_view_idx({c})];
            float i = input[i_tv.get_tensor_view_idx({n, c})];
            float t = target[t_tv.get_tensor_view_idx({n, c})];
            loss += -w * calc_loss(i, t);
        }
        ref_output[o_tv.get_tensor_view_idx({n})] = loss / C;
    });
}

template <class T>
void cpu_multilabelsoftmarginloss_reduced_forward(
    tensor<T> input, tensor<T> target, tensor<T> weight, tensor<T>& ref_output, const float divisor)
{
    auto N    = input.desc.GetLengths()[0];
    auto C    = input.desc.GetLengths()[1];
    auto i_tv = miopen::get_inner_expanded_tv<2>(input.desc);
    auto t_tv = miopen::get_inner_expanded_tv<2>(target.desc);
    auto w_tv = miopen::get_inner_expanded_tv<1>(weight.desc);

    double sum = 0;
    for(size_t n = 0; n < N; n++)
    {
        double loss = 0;
        for(size_t c = 0; c < C; c++)
        {
            double w = weight[w_tv.get_tensor_view_idx({c})];
            double i = input[i_tv.get_tensor_view_idx({n, c})];
            double t = target[t_tv.get_tensor_view_idx({n, c})];
            loss += -w * calc_loss(i, t);
        }

        sum += loss / C;
    }
    sum /= divisor;
    ref_output[0] = static_cast<T>(sum);
}

#endif
