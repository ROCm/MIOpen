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
void cpu_glu_contiguous_dim0_forward(const tensor<T>& input, tensor<T>& ref_output)
{
    auto output_dims  = ref_output.desc.GetLengths();
    auto output_numel = ref_output.desc.GetElementSize();

    par_ford(output_numel)([&](size_t o) {
        T valA        = input[o];
        T valB        = input[o + output_numel];
        T val         = valA * sigmoid(valB);
        ref_output[o] = val;
    });
}

template <class T>
void cpu_glu_contiguous_dim0_backward(const tensor<T>& input,
                                      const tensor<T>& grad_output,
                                      tensor<T>& grad_input)
{
    auto outputGrad_dims  = grad_output.desc.GetLengths();
    auto outputGrad_numel = grad_output.desc.GetElementSize();

    par_ford(outputGrad_numel)([&](size_t o) {
        T inputFirstHalf_v               = input[o];
        T sigmoid_v                      = sigmoid(input[o + outputGrad_numel]);
        T grad_v                         = grad_output[o];
        grad_input[o]                    = sigmoid_v * grad_v;
        grad_input[o + outputGrad_numel] = (1 - sigmoid_v) * sigmoid_v * grad_v * inputFirstHalf_v;
    });
}
