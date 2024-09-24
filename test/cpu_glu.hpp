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

#include "ford.hpp"
#include "tensor_holder.hpp"

template <typename T>
T sigmoid(T x)
{
    return static_cast<T>(1.0f / (1.0f + exp(-x)));
}

template <class T>
void cpu_glu_forward(const tensor<T>& input, tensor<T>& ref_output, uint32_t dim)
{
    auto output_dims     = ref_output.desc.GetLengths();
    auto input_lengths   = input.desc.GetLengths();
    size_t dim_size      = input_lengths[dim];
    size_t half_dim_size = dim_size / 2;
    size_t inner_size    = 1;

    for(size_t i = dim + 1; i < input_lengths.size(); i++)
    {
        inner_size *= input_lengths[i];
    }

    auto output_numel =
        std::accumulate(output_dims.begin(), output_dims.end(), 1L, std::multiplies<int64_t>());

    par_ford(output_numel)([&](size_t o) {
        size_t outer_idx = o / (dim_size * inner_size);
        o                = o % (dim_size * inner_size);
        size_t dim_idx   = o / inner_size;
        size_t inner_idx = o % inner_size;
        size_t inputFirstHalf_idx =
            outer_idx * (dim_size * inner_size) + dim_idx * inner_size + inner_idx;
        T valA                     = input[inputFirstHalf_idx];
        size_t inputSecondHalf_idx = outer_idx * (dim_size * inner_size) +
                                     (dim_idx + half_dim_size) * inner_size + inner_idx;
        T valB        = input[inputSecondHalf_idx];
        T val         = valA * sigmoid(valB);
        ref_output[o] = val;
    });
}

template <class T>
void cpu_glu_backward(const tensor<T>& input,
                      const tensor<T>& grad_output,
                      tensor<T>& grad_input,
                      uint32_t dim)
{
    auto outputGrad_dims = grad_output.desc.GetLengths();
    auto input_lengths   = input.desc.GetLengths();
    size_t dim_size      = input_lengths[dim];
    size_t half_dim_size = dim_size / 2;
    size_t inner_size    = 1;

    for(size_t i = dim + 1; i < input_lengths.size(); i++)
    {
        inner_size *= input_lengths[i];
    }

    auto outputGrad_numel = std::accumulate(
        outputGrad_dims.begin(), outputGrad_dims.end(), 1L, std::multiplies<int64_t>());

    par_ford(outputGrad_numel)([&](size_t o) {
        size_t outer_idx = o / (dim_size * inner_size);
        o                = o % (dim_size * inner_size);
        size_t dim_idx   = o / inner_size;
        size_t inner_idx = o % inner_size;
        size_t inputFirstHalf_idx =
            outer_idx * (dim_size * inner_size) + dim_idx * inner_size + inner_idx;
        T inputFirstHalf           = input[inputFirstHalf_idx];
        size_t inputSecondHalf_idx = outer_idx * (dim_size * inner_size) +
                                     (dim_idx + half_dim_size) * inner_size + inner_idx;
        T inputSecondHalf                = input[inputSecondHalf_idx];
        T sigmoid_v                      = sigmoid(inputSecondHalf);
        T grad_v                         = grad_output[o];
        grad_input[o]                    = sigmoid_v * grad_v;
        grad_input[o + outputGrad_numel] = (1 - sigmoid_v) * sigmoid_v * grad_v * inputFirstHalf;
    });
}
