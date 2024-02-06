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
#ifndef GUARD_CPU_CAT_HPP
#define GUARD_CPU_CAT_HPP

#include "tensor_holder.hpp"

template <class T>
void cpu_cat_forward(std::vector<tensor<T>> inputs, tensor<T>& ref_output, int32_t dim)
{
    auto dims              = ref_output.desc.GetLengths();
    size_t output_dim_size = dims[dim];
    size_t outer_size      = 1;
    size_t inner_size      = 1;
    size_t i               = 0;
    for(; i < dim; i++)
    {
        outer_size *= dims[i];
    }

    for(; i < dims.size(); i++)
    {
        inner_size *= dims[i];
    }

    size_t output_start_offset = 0;

    par_ford(inputs.size())([&](int32_t i) {
        auto input        = inputs[i];
        size_t dim_size   = inputs[i].desc.GetLengths()[dim];
        size_t copy_size  = inner_size / output_dim_size * dim_size;
        size_t input_size = outer_size * copy_size;
        ford(input_size)([&](int32_t o) {
            size_t outer_idx = o / copy_size;
            size_t copy_idx  = o % copy_size;
            ref_output[output_start_offset + (outer_idx * inner_size) + copy_idx] =
                input[copy_size * outer_idx + copy_idx];
        });

        output_start_offset += copy_size;
    });
}
#endif
