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

#include "tensor_holder.hpp"
#include "tensor_view.hpp"

#include <miopen/tensor_view_utils.hpp>

#include <vector>

template <typename TIO>
void cpu_kthvalue(tensor<TIO> input,
                  tensor<TIO>& outputHost,
                  std::vector<size_t>& indices,
                  miopen::TensorDescriptor indiceDesc,
                  size_t k,
                  int dim)
{
    size_t inputSize       = input.desc.GetElementSize();
    size_t dimSize         = input.desc.GetLengths()[dim];
    size_t dimStride       = input.desc.GetStrides()[dim];
    auto inputTv           = miopen::get_inner_expanded_tv<5>(input.desc);
    auto inputTvWithoutDim = miopen::get_tv_without_dim<5>(inputTv, dim);
    auto outputTv          = miopen::get_inner_expanded_tv<5>(outputHost.desc);
    auto indicesTv         = miopen::get_inner_expanded_tv<5>(indiceDesc);

    size_t numSlice = inputSize / dimSize;

    std::vector<float> elements;
    std::vector<size_t> ids(dimSize);
    for(int i = 0; i < dimSize; ++i)
    {
        ids[i] = i;
    }

    for(int slideID = 0; slideID < numSlice; ++slideID)
    {
        elements.clear();
        tensor_layout_t<4> layout(inputTvWithoutDim, slideID);
        auto idx = inputTvWithoutDim.get_tensor_view_idx(layout);

        for(int j = 0; j < dimSize; ++j)
        {
            elements.push_back(static_cast<float>(input[idx + j * dimStride]));
        }

        std::sort(ids.begin(), ids.end(), [=](size_t x, size_t y) -> bool {
            return elements[x] < elements[y];
        });
        auto output_layout  = tensor_layout_t<5>(outputTv, slideID);
        auto indices_layout = tensor_layout_t<5>(indicesTv, slideID);
        outputHost[outputTv.get_tensor_view_idx(output_layout)] =
            static_cast<TIO>(elements[ids[k - 1]]);
        indices[indicesTv.get_tensor_view_idx(indices_layout)] = ids[k - 1];
    }
}
