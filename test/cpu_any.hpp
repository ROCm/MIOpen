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

#include <ford.hpp>
#include <tensor_holder.hpp>
#include <miopen/tensor_view_utils.hpp>

template <class T>
void cpu_any_forward(tensor<T> input, tensor<uint8_t>& ref_output, size_t dim, bool keepdim)
{
    auto input_tv  = miopen::get_inner_expanded_tv<5>(input.desc);
    auto output_tv = miopen::get_inner_expanded_tv<5>(ref_output.desc);

    auto input_dims  = input.desc.GetLengths();
    auto output_dims = ref_output.desc.GetLengths();

    auto output_numel = ref_output.desc.GetElementSize();
    auto input_numel  = input.desc.GetElementSize();

    auto reduce_size = input_dims[dim];

    auto inner_size = 1ULL;
    for(int32_t i = dim + 1; i < input_dims.size(); i++)
    {
        inner_size *= input_dims[i];
    }

    if(dim != -1)
    {
        par_ford(output_numel)([&](size_t o) {
            size_t input_idx = (o / inner_size) * inner_size * reduce_size + o % inner_size;
            T any            = 0;
            ford(reduce_size)([&](size_t o) {
                T val = input[input_idx];
                any   = any || val;
                input_idx += inner_size;
            });
            ref_output[o] = static_cast<uint8_t>(any);
        });
    }
    else
    {
        T any = 0;
        par_ford(input_numel)([&](size_t i) { any = any || input[i]; });
        ref_output[0] = static_cast<uint8_t>(any);
    }
}
