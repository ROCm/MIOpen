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
#include <math.h>

double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double calc_loss(double x, double y)
{
    double sig = sigmoid(x);
    return y * log(sig) + (1 - y) * log(1 - sig);
}

template <class T>
void cpu_multilabelsoftmarginloss_forward(tensor<T> input,
                                          tensor<T> target,
                                          tensor<T> weight,
                                          tensor<T>& ref_output,
                                          miopenLossReductionMode_t reduction_mode)
{
    auto N    = input.desc.GetLengths()[0];
    auto C    = input.desc.GetLengths()[1];
    auto i_tv = miopen::get_inner_expanded_tv<2>(input.desc);
    auto t_tv = miopen::get_inner_expanded_tv<2>(target.desc);
    auto w_tv = miopen::get_inner_expanded_tv<1>(weight.desc);
    auto o_tv = miopen::get_inner_expanded_tv<1>(ref_output.desc);

    double sum_loss = 0;
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
        loss /= C;
        if(reduction_mode == MIOPEN_LOSS_REDUCTION_NONE)
            ref_output[o_tv.get_tensor_view_idx({n})] = loss;
        else
            sum_loss += loss;
    }
    if(reduction_mode == MIOPEN_LOSS_REDUCTION_MEAN)
        ref_output[0] = sum_loss / N;
    else if(reduction_mode == MIOPEN_LOSS_REDUCTION_SUM)
        ref_output[0] = sum_loss;
}
