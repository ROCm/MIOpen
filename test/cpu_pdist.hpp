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
void cpu_pdist_forward_contiguous(const tensor<T> input,
                                  const tensor<T> output,
                                  const tensor<T> output_grad,
                                  tensor<T>& ref_input_grad,
                                  const double p)
{
    std::fill(ref_input_grad.begin(), ref_input_grad.end(), static_cast<T>(0));

    auto N = input.desc.GetLengths()[0];
    auto M = input.desc.GetLengths()[1];

    for(int i = 0; i < N; ++i)
    {
        for(int j = i + 1; j < N; ++j)
        {
            long k = j + N * i - i * (i + 1) / 2 - i - 1;

            double grad_k   = static_cast<double>(output_grad[k]);
            double output_k = static_cast<double>(output[k]);

            for(int m = 0; m < M; ++m)
            {
                double input_first  = static_cast<double>(input[i * M + m]);
                double input_second = static_cast<double>(input[j * M + m]);
                double diff         = input_first - input_second;

                T res = static_cast<T>(miopen::solver::pdist::backward(diff, grad_k, output_k, p));

                ref_input_grad[i * M + m] += res;
                ref_input_grad[j * M + m] -= res;
            }
        }
    }
}
