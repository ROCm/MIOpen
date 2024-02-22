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
#ifndef GUARD_CPU_REDUCEEXTREME_HPP
#define GUARD_CPU_REDUCEEXTREME_HPP

#include "tensor_holder.hpp"

template <class T>
void cpu_argmax_forward(tensor<T> input, tensor<int32_t>& ref_indice, int32_t dim)
{
    auto input_dims  = input.desc.GetLengths();
    auto indice_dims = ref_indice.desc.GetLengths();

    auto reduce_size = input_dims[dim];
    auto indice_numel =
        std::accumulate(indice_dims.begin(), indice_dims.end(), 1L, std::multiplies<int64_t>());

    auto inner_size = std::accumulate(
        input_dims.begin() + dim + 1, input_dims.end(), 1ULL, std::multiplies<uint64_t>());

    par_ford(indice_numel)([&](size_t o) {
        size_t input_idx = (o / inner_size) * inner_size * reduce_size + o % inner_size;

        int32_t max_idx = 0;
        T max           = input[input_idx];

        ford(reduce_size)([&](size_t i) {
            T val = input[input_idx];
            if(max < val)
            {
                max     = val;
                max_idx = i;
            }
            input_idx += inner_size;
        });

        ref_indice[o] = max_idx;
    });
}

template <class T>
void cpu_argmin_forward(tensor<T> input, tensor<int32_t>& ref_indice, int32_t dim)
{
    auto input_dims  = input.desc.GetLengths();
    auto indice_dims = ref_indice.desc.GetLengths();

    auto reduce_size = input_dims[dim];
    auto indice_numel =
        std::accumulate(indice_dims.begin(), indice_dims.end(), 1L, std::multiplies<int64_t>());

    auto inner_size = std::accumulate(
        input_dims.begin() + dim + 1, input_dims.end(), 1ULL, std::multiplies<uint64_t>());

    par_ford(indice_numel)([&](size_t o) {
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

        ref_indice[o] = min_idx;
    });
}

template <class T>
void cpu_max_forward(tensor<T> input,
                     tensor<T>& ref_output,
                     tensor<int32_t>& ref_indice,
                     int32_t dim)
{
    auto input_dims  = input.desc.GetLengths();
    auto output_dims = ref_output.desc.GetLengths();

    auto reduce_size = input_dims[dim];
    auto output_numel =
        std::accumulate(output_dims.begin(), output_dims.end(), 1L, std::multiplies<int64_t>());

    auto inner_size = std::accumulate(
        input_dims.begin() + dim + 1, input_dims.end(), 1ULL, std::multiplies<uint64_t>());

    par_ford(output_numel)([&](size_t o) {
        size_t input_idx = (o / inner_size) * inner_size * reduce_size + o % inner_size;

        int32_t max_idx = 0;
        T max           = input[input_idx];

        ford(reduce_size)([&](size_t i) {
            T val = input[input_idx];
            if(max < val)
            {
                max     = val;
                max_idx = i;
            }
            input_idx += inner_size;
        });

        ref_output[o] = max;
        ref_indice[o] = max_idx;
    });
}

template <class T>
void cpu_min_forward(tensor<T> input,
                     tensor<T>& ref_output,
                     tensor<int32_t>& ref_indice,
                     int32_t dim)
{
    auto input_dims  = input.desc.GetLengths();
    auto output_dims = ref_output.desc.GetLengths();

    auto reduce_size = input_dims[dim];
    auto output_numel =
        std::accumulate(output_dims.begin(), output_dims.end(), 1L, std::multiplies<int64_t>());

    auto inner_size = std::accumulate(
        input_dims.begin() + dim + 1, input_dims.end(), 1ULL, std::multiplies<uint64_t>());

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
