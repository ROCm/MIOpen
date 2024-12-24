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

#include "cpu_kthvalue.hpp"

template <class T>
void cpu_median_fwd(tensor<T> input,
                    tensor<T>& ref_output,
                    tensor<size_t>& ref_indices,
                    uint64_t dim)
{
    auto input_lengths = input.desc.GetLengths();
    size_t k           = (input_lengths[dim] + 1) / 2;

    cpu_kthvalue<T>(input, ref_output, ref_indices.data, ref_indices.desc, k, dim);
}

template <class T>
void cpu_median_bwd(const tensor<T> output_grad,
                    const tensor<size_t> indices,
                    tensor<T>& input_grad,
                    const uint64_t dim)
{
    cpu_kth_value_backward<T>(output_grad, indices, input_grad, dim);
}
