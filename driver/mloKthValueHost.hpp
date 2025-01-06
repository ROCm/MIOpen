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

#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>

#include <../test/ford.hpp>

template <typename Tgpu, typename Tcheck, typename T_index = size_t>
int32_t mloKthvalueFwdRunHost(const miopenTensorDescriptor_t inputDesc,
                              const miopenTensorDescriptor_t outputDesc,
                              const miopenTensorDescriptor_t indicesDesc,
                              const Tgpu* input,
                              Tcheck* output,
                              T_index* indices,
                              const size_t k,
                              const uint64_t dim)
{
    size_t input_size = miopen::deref(inputDesc).GetElementSize();
    size_t dim_size   = miopen::deref(inputDesc).GetLengths()[dim];
    size_t dim_stride = miopen::deref(inputDesc).GetStrides()[dim];

    auto input_tv             = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto input_tv_without_dim = miopen::get_tv_without_dim<5>(input_tv, dim);
    auto output_tv            = miopen::get_inner_expanded_tv<5>(miopen::deref(outputDesc));
    auto indices_tv           = miopen::get_inner_expanded_tv<5>(miopen::deref(indicesDesc));

    size_t num_slice = input_size / dim_size;

    par_ford(num_slice)([&](size_t slice_id) {
        std::vector<float> elements(dim_size);
        std::vector<size_t> ids(dim_size);
        std::iota(ids.begin(), ids.end(), 0);

        tensor_layout_t<4> layout(input_tv_without_dim, slice_id);
        auto idx = input_tv_without_dim.get_tensor_view_idx(layout);

        par_ford(dim_size)(
            [&](size_t j) { elements[j] = static_cast<float>(input[idx + j * dim_stride]); });

        std::sort(ids.begin(), ids.end(), [=](size_t x, size_t y) -> bool {
            return elements[x] < elements[y];
        });

        auto output_layout  = tensor_layout_t<5>(output_tv, slice_id);
        auto indices_layout = tensor_layout_t<5>(indices_tv, slice_id);
        output[output_tv.get_tensor_view_idx(output_layout)] =
            static_cast<Tcheck>(elements[ids[k - 1]]);
        indices[indices_tv.get_tensor_view_idx(indices_layout)] = static_cast<T_index>(ids[k - 1]);
    });

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck, typename T_index = size_t>
int32_t mloKthvalueBwdRunHost(const miopenTensorDescriptor_t outputGradDesc,
                              const miopenTensorDescriptor_t indicesDesc,
                              const miopenTensorDescriptor_t inputGradDesc,
                              const Tgpu* output_grad,
                              const T_index* indices,
                              Tcheck* input_grad,
                              const uint64_t dim)
{
    size_t input_grad_size = miopen::deref(inputGradDesc).GetElementSize();

    size_t dim_size   = miopen::deref(inputGradDesc).GetLengths()[dim];
    size_t dim_stride = miopen::deref(inputGradDesc).GetStrides()[dim];

    auto input_grad_tv             = miopen::get_inner_expanded_tv<5>(miopen::deref(inputGradDesc));
    auto input_grad_tv_without_dim = miopen::get_tv_without_dim<5>(input_grad_tv, dim);
    auto output_grad_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(outputGradDesc));
    auto indices_tv     = miopen::get_inner_expanded_tv<5>(miopen::deref(indicesDesc));

    size_t num_slice = input_grad_size / dim_size;

    std::fill(input_grad, input_grad + input_grad_size, static_cast<Tcheck>(0));

    par_ford(num_slice)([&](size_t slice_id) {
        tensor_layout_t<5> indices_layout(indices_tv, slice_id);
        size_t k_index = indices[indices_tv.get_tensor_view_idx(indices_layout)];

        tensor_layout_t<5> out_grad_layout(output_grad_tv, slice_id);
        float val =
            static_cast<float>(output_grad[output_grad_tv.get_tensor_view_idx(out_grad_layout)]);

        tensor_layout_t<4> layout(input_grad_tv_without_dim, slice_id);
        auto idx = input_grad_tv_without_dim.get_tensor_view_idx(layout);

        input_grad[idx + k_index * dim_stride] = static_cast<Tcheck>(val);
    });

    return miopenStatusSuccess;
}
