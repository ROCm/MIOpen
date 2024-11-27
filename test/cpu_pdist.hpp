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

#include <algorithm>
#include <math.h>

#include <miopen/tensor_view_utils.hpp>
#include <miopen/pdist/utils.hpp>

#include "tensor_holder.hpp"

template <class T>
void cpu_pdist_backward(const tensor<T> input,
                        const tensor<T> output,
                        const tensor<T> output_grad,
                        tensor<T>& ref_input_grad,
                        const double p)
{
    std::fill(ref_input_grad.begin(), ref_input_grad.end(), static_cast<T>(0));

    auto input_tv       = miopen::get_inner_expanded_tv<2>(input.desc);
    auto output_tv      = miopen::get_inner_expanded_tv<1>(output.desc);
    auto output_grad_tv = miopen::get_inner_expanded_tv<1>(output_grad.desc);
    auto input_grad_tv  = miopen::get_inner_expanded_tv<2>(ref_input_grad.desc);

    auto N = input.desc.GetLengths()[0];
    auto M = input.desc.GetLengths()[1];

    for(size_t i = 0; i < N; ++i)
    {
        for(size_t j = i + 1; j < N; ++j)
        {
            size_t k = j + N * i - i * (i + 1) / 2 - i - 1;

            double grad_k =
                static_cast<double>(output_grad[output_grad_tv.get_tensor_view_idx({k})]);
            double output_k = static_cast<double>(output[output_tv.get_tensor_view_idx({k})]);

            for(size_t m = 0; m < M; ++m)
            {

                double input_first =
                    static_cast<double>(input[input_tv.get_tensor_view_idx({i, m})]);
                double input_second =
                    static_cast<double>(input[input_tv.get_tensor_view_idx({j, m})]);
                double diff = input_first - input_second;

                T res = static_cast<T>(miopen::pdist::backward(diff, grad_k, output_k, p));

                ref_input_grad[input_grad_tv.get_tensor_view_idx({i, m})] += res;
                ref_input_grad[input_grad_tv.get_tensor_view_idx({j, m})] -= res;
            }
        }
    }
}
