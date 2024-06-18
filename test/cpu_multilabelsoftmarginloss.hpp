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
void cpu_multilabelsoftmarginloss_reduced_forward(tensor<T> input,
                                                  tensor<T> target,
                                                  tensor<T> weight,
                                                  tensor<T>& ref_output,
                                                  tensor<T>& ref_workspace,
                                                  const float divisor)
{
    auto N    = input.desc.GetLengths()[0];
    auto C    = input.desc.GetLengths()[1];
    auto i_tv = miopen::get_inner_expanded_tv<2>(input.desc);
    auto t_tv = miopen::get_inner_expanded_tv<2>(target.desc);
    auto w_tv = miopen::get_inner_expanded_tv<1>(weight.desc);

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
        ref_workspace[n] = loss / C / divisor;
    });
    // Reduce loss
    const int local_size = 256;
    int offset_a         = 0;
    int offset_b         = N;
    size_t _size         = N;
    do
    {
        for(int i = 0; i < _size; i += local_size)
        {
            T shared[local_size];
            for(int j = 0; j < local_size; ++j)
                shared[j] = i + j < _size ? ref_workspace[offset_a + i + j] : 0.0f;
            for(int offset = local_size / 2; offset > 0; offset >>= 1)
                for(int j = 0; j < offset; ++j)
                    shared[j] += shared[j + offset];
            if(_size <= local_size)
                ref_output[0] = shared[0];
            else
                ref_workspace[offset_b + i / local_size] = shared[0];
        }
        std::swap(offset_a, offset_b);
        _size = (_size + local_size - 1) / local_size;
    } while(_size > 1);
}

#endif
