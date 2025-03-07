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

#include "miopen/tensor_view_utils.hpp"
#include "tensor_holder.hpp"
#include "tensor_view.hpp"
#include <cstddef>

template <class T>
void cpu_indexselect_forward(const tensor<T>& input,
                             const tensor<size_t>& indices,
                             tensor<T>& output,
                             size_t dim)
{
    tensor_view_t<5> input_tv   = get_inner_expanded_tv<5>(input.desc);
    tensor_view_t<5> output_tv  = get_inner_expanded_tv<5>(output.desc);
    tensor_view_t<1> indices_tv = get_inner_expanded_tv<1>(indices.desc);

    auto max_idx      = input_tv.size[dim];
    auto output_numel = output.desc.GetElementSize();

    for(size_t i = 0; i < output_numel; i++)
    {
        tensor_layout_t<5> output_layout{output_tv, i};
        tensor_layout_t<5> input_layout = output_layout;
        tensor_layout_t<1> indices_layout{output_layout.layout[dim]};
        input_layout.layout[dim] = indices[indices_tv.get_tensor_view_idx(indices_layout)];

        if(input_layout.layout[dim] < max_idx)
        {
            output[output_tv.get_tensor_view_idx(output_layout)] =
                input[input_tv.get_tensor_view_idx(input_layout)];
        }
        else
        {
            output[output_tv.get_tensor_view_idx(output_layout)] = T(0);
        }
    }
}

template <class T>
void cpu_indexselect_backward(const tensor<T>& outputGrad,
                              const tensor<size_t>& indices,
                              tensor<T>& inputGradHost,
                              size_t dim)
{
    auto outputGrad_tv = get_inner_expanded_tv<5>(outputGrad.desc);
    auto inputGrad_tv  = get_inner_expanded_tv<5>(inputGradHost.desc);
    auto indices_tv    = get_inner_expanded_tv<1>(indices.desc);

    size_t output_numel = outputGrad.desc.GetElementSize();
    size_t max_idx      = inputGrad_tv.size[dim];

    for(size_t i = 0; i < output_numel; i++)
    {
        tensor_layout_t<5> output_layout{outputGrad_tv, i};
        tensor_layout_t<5> input_layout = output_layout;
        tensor_layout_t<1> indices_layout{output_layout.layout[dim]};
        input_layout.layout[dim] = indices[indices_tv.get_tensor_view_idx(indices_layout)];

        if(input_layout.layout[dim] < max_idx)
        {
            inputGradHost[inputGrad_tv.get_tensor_view_idx(input_layout)] +=
                outputGrad[outputGrad_tv.get_tensor_view_idx(output_layout)];
        }
    }
}
