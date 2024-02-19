/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#ifndef GUARD_CPU_MIN_HPP
#define GUARD_CPU_MIN_HPP

#include "tensor_holder.hpp"

template <class T>
void cpu_min_forward(tensor<T> input, tensor<T>& ref_output, tensor<int>& ref_indice, int32_t dim)
{
    auto input_dims  = input.desc.GetLengths();
    auto output_dims = ref_output.desc.GetLengths();

    auto reduce_size = input_dims[dim];
    auto output_numel =
        std::accumulate(output_dims.begin(), output_dims.end(), 1L, std::multiplies<int64_t>());

    auto inner_size = 1ULL;
    for(int32_t i = dim + 1; i < input_dims.size(); i++)
    {
        inner_size *= input_dims[i];
    }

    par_ford(output_numel)([&](size_t o) {
        size_t input_idx = (o / inner_size) * inner_size * reduce_size + o % inner_size;

        int32_t min_idx = 0;
        T min           = input[input_idx];

        ford(reduce_size)([&](size_t i) {
            T val = input[input_idx];
            if(min > val)
            {
                min     = val;
                min_idx = i;
            }
            input_idx += inner_size;
        });

        ref_output[o] = min;
        ref_indice[o] = min_idx;
    });
}
#endif
