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

#include "miopen/miopen.h"
#include "tensor_holder.hpp"
#include <miopen/tensor_view_utils.hpp>

template <class T>
void cpu_multimarginloss_forward(const tensor<T>& input,
                                 const tensor<uint64_t>& target,
                                 const tensor<T>& weight,
                                 tensor<T>& ref_output,
                                 const long p,
                                 const float margin,
                                 miopenLossReductionMode_t reduction_mode)
{
    auto I_tv = miopen::get_inner_expanded_tv<2>(input.desc);
    auto T_tv = miopen::get_inner_expanded_tv<1>(target.desc);
    auto W_tv = miopen::get_inner_expanded_tv<1>(weight.desc);
    auto O_tv = miopen::get_inner_expanded_tv<1>(ref_output.desc);
    auto N = I_tv.size[0], C = I_tv.size[1];

    double sum = 0;
    for(size_t n = 0; n < N; n++)
    {
        double loss = 0;
        uint64_t y  = target[T_tv.get_tensor_view_idx({n})];
        if(y >= C)
            continue;
        for(size_t c = 0; c < C; c++)
        {
            if(y == c)
                continue;
            double t = margin - static_cast<double>(input[I_tv.get_tensor_view_idx({n, y})]) +
                       static_cast<double>(input[I_tv.get_tensor_view_idx({n, c})]);

            if(t < 0)
                continue;
            if(p == 2)
                t = t * t;
            t = static_cast<double>(weight[W_tv.get_tensor_view_idx({y})]) * t;
            loss += t / C;
        }
        if(reduction_mode == MIOPEN_LOSS_REDUCTION_NONE)
            ref_output[O_tv.get_tensor_view_idx({n})] = loss;
        else
            sum += loss;
    }
    if(reduction_mode == MIOPEN_LOSS_REDUCTION_MEAN)
    {
        ref_output[0] = static_cast<T>(sum / N);
    }
    else if(reduction_mode == MIOPEN_LOSS_REDUCTION_SUM)
    {
        ref_output[0] = static_cast<T>(sum);
    }
}
