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

#pragma once

#include "tensor_holder.hpp"
#include <vector>

template <class T>
void cpu_logsumexp_forward(tensor<T> input, tensor<T>& output, std::vector<int> dims_vector)
{
    auto input_dims     = input.desc.GetLengths();
    auto input_strides  = input.desc.GetStrides();
    auto output_dims    = output.desc.GetLengths();
    auto output_strides = output.desc.GetStrides();

    for(int64_t d = input_dims.size() - 1; d >= 0; --d)
    {
        if(std::find(dims_vector.begin(), dims_vector.end(), d) == dims_vector.end())
            continue;
        for(int64_t dd = input_dims.size() - 1; dd > d; --dd)
        {
            if(std::find(dims_vector.begin(), dims_vector.end(), dd) != dims_vector.end())
                continue;
            std::swap(input_dims[d], input_dims[dd]);
            std::swap(input_strides[d], input_strides[dd]);
            std::swap(output_dims[d], output_dims[dd]);
            std::swap(output_strides[d], output_strides[dd]);
        }
    }

    auto input_numel =
        std::accumulate(input_dims.begin(), input_dims.end(), 1LL, std::multiplies<int64_t>());
    auto output_numel =
        std::accumulate(output_dims.begin(), output_dims.end(), 1LL, std::multiplies<int64_t>());

    auto K = input_numel / output_numel;

    par_ford(output_numel)([&](size_t gid) {
        std::vector<float> vals(K);
        float max = std::numeric_limits<float>::lowest();

        for(int64_t k = 0; k < K; ++k)
        {
            std::vector<int64_t> input_idx(input_dims.size(), 0);
            int64_t tmp_gid = gid * K + k;
            for(int i = input_dims.size() - 1; i >= 0; --i)
            {
                input_idx[i] = tmp_gid % input_dims[i];
                tmp_gid /= input_dims[i];
            }
            float val = input[std::inner_product(
                input_idx.begin(), input_idx.end(), input_strides.begin(), static_cast<size_t>(0))];
            max       = max > val ? max : val;
            vals[k]   = val;
        }

        float logsum = static_cast<float>(0, 0);
        for(int64_t k = 0; k < K; ++k)
        {
            logsum += std::exp(vals[k] - max);
        }

        std::vector<int64_t> output_idx(input_dims.size(), 0);

        int64_t tmp_gid = gid;
        for(int i = output_idx.size() - 1; i >= 0; --i)
        {
            output_idx[i] = tmp_gid % output_dims[i];
            tmp_gid /= output_dims[i];
        }

        output[std::inner_product(
            output_idx.begin(), output_idx.end(), output_strides.begin(), static_cast<size_t>(0))] =
            static_cast<T>(max + std::log(logsum));
    });
}

template <class T>
void cpu_logsumexp_backward(tensor<T> input,
                            tensor<T>& input_grad,
                            tensor<T> output,
                            tensor<T> output_grad,
                            int* dims,
                            int num_dims)
{
    auto input_dims          = input.desc.GetLengths();
    auto input_strides       = input.desc.GetStrides();
    auto input_grad_dims     = input_grad.desc.GetLengths();
    auto input_grad_strides  = input_grad.desc.GetStrides();
    auto output_dims         = output.desc.GetLengths();
    auto output_strides      = output.desc.GetStrides();
    auto output_grad_dims    = output_grad.desc.GetLengths();
    auto output_grad_strides = output_grad.desc.GetStrides();

    auto input_grad_numel = std::accumulate(input_grad.desc.GetLengths().begin(),
                                            input_grad.desc.GetLengths().end(),
                                            1LL,
                                            std::multiplies<int64_t>());

    std::fill(input_grad.data.begin(), input_grad.data.end(), 0);

    par_ford(input_grad_numel)([&](size_t gid) {
        std::vector<int64_t> input_idx(input_dims.size(), 0);
        int64_t tmp_gid = gid;

        for(int i = input_dims.size() - 1; i >= 0; --i)
        {
            input_idx[i] = tmp_gid % input_dims[i];
            tmp_gid /= input_dims[i];
        }

        std::vector<int64_t> reduced_idx(input_dims.size(), 0);
        for(int i = 0; i < input_dims.size(); ++i)
        {
            if(std::find(dims, dims + num_dims, i) == dims + num_dims)
            {
                reduced_idx[i] = input_idx[i];
            }
        }

        int64_t input_index = std::inner_product(
            input_idx.begin(), input_idx.end(), input_strides.begin(), static_cast<size_t>(0));
        int64_t input_grad_index = std::inner_product(
            input_idx.begin(), input_idx.end(), input_grad_strides.begin(), static_cast<size_t>(0));
        int64_t output_index = std::inner_product(
            reduced_idx.begin(), reduced_idx.end(), output_strides.begin(), static_cast<size_t>(0));
        int64_t output_grad_index = std::inner_product(reduced_idx.begin(),
                                                       reduced_idx.end(),
                                                       output_grad_strides.begin(),
                                                       static_cast<size_t>(0));

        float x  = input[input_index];
        float y  = output[output_index];
        float dy = output_grad[output_grad_index];

        input_grad[input_grad_index] = static_cast<T>(dy * std::exp(x - y));
    });
}
