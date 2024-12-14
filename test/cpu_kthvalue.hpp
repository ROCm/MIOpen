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
// #include "tensor_view_utils.hpp"

#include <miopen/tensor_view_utils.hpp>

// #include <vector>

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

template <class T>
void cpu_kth_value_backward(const tensor<T> output_grad,
                            const tensor<size_t> indices,
                            tensor<T>& ref_input_grad,
                            uint64_t dim)
{
    std::fill(ref_input_grad.begin(), ref_input_grad.end(), static_cast<T>(0));

    // size_t num_slice = input_size / dim_size;
    auto input_grad_size = ref_input_grad.desc.GetElementSize();

    auto input_grad_lengths = ref_input_grad.desc.GetLengths();
    auto input_grad_strides = ref_input_grad.desc.GetStrides();

    size_t dim_size   = input_grad_lengths[dim];
    size_t dim_stride = input_grad_strides[dim];

    auto input_grad_tv             = miopen::get_inner_expanded_tv<5>(ref_input_grad.desc);
    auto input_grad_tv_without_dim = miopen::get_tv_without_dim<5>(input_grad_tv, dim);
    auto output_grad_tv            = miopen::get_inner_expanded_tv<5>(output_grad.desc);
    auto indices_tv                = miopen::get_inner_expanded_tv<5>(indices.desc);

    // auto output_size = output_grad.desc.GetElementSize();
    size_t num_slice = input_grad_size / dim_size;

    // for(int slice_id = 0; slice_id < num_slice; ++slice_id)
    par_ford(num_slice)([&](size_t slice_id) {
        tensor_layout_t<5> indices_layout(indices_tv, slice_id);
        size_t k_index = indices[indices_tv.get_tensor_view_idx(indices_layout)];

        tensor_layout_t<5> out_grad_layout(output_grad_tv, slice_id);
        // size_t grad_output_idx = output_grad_tv.get_tensor_view_idx(out_grad_layout);
        T val = output_grad[output_grad_tv.get_tensor_view_idx(out_grad_layout)];

        tensor_layout_t<4> layout(input_grad_tv_without_dim, slice_id);
        auto idx = input_grad_tv_without_dim.get_tensor_view_idx(layout);

        // Propagate gradient to the k-th position
        // std::cout << "slice_id: " << slice_id << "; k_index: " << k_index << "; grad_output_idx:
        // " << grad_output_idx << "; idx: " << idx << "; ref_input_grad_idx: " << idx + k_index *
        // dim_stride << std::endl; std::cout << "output_grad[grad_output_idx]: " <<
        // output_grad.data[grad_output_idx] << std::endl;

        // ref_input_grad[idx + k_index * dim_stride] = output_grad.data[grad_output_idx];
        ref_input_grad[idx + k_index * dim_stride] = val;
    });
    // }

    // for(size_t i = 0; i < output_size; i++)
    // {
    //     tensor_layout_t<5> out_grad_layout(output_tv, i);
    //     T val = output_grad[output_tv.get_tensor_view_idx(out_grad_layout)];

    //     tensor_layout_t<5> indices_layout(indices_tv, i);
    //     size_t idx = indices[indices_tv.get_tensor_view_idx(indices_layout)];

    //     tensor_layout_t<4> in_grad_layout(input_grad_tv_without_dim, i);
    //     auto ig_idx = input_grad_tv_without_dim.get_tensor_view_idx(in_grad_layout);
    //     for(size_t j = 0; j < dim_size; j++)
    //     {
    //         size_t in_grad_idx = ig_idx + i * dim_stride;
    //         // uint64_t in_grad_idx =
    //         //     input_grad_tv_without_dim.get_tensor_view_idx(in_grad_layout) + j *
    //         dim_stride; ref_input_grad[in_grad_idx] = (j == idx) ? val : static_cast<T>(0);
    //     }
    // }
}
