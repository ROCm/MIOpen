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
#include <miopen/reducecalculation.hpp>

#include "tensor_holder.hpp"

#include "../src/kernels/MIOpenReduceCalculation.hpp"

template <typename T, ReduceCalculationOp_t op>
void cpu_calculation_forward(const tensor<T> input,
                             tensor<T>& ref_output,
                             uint32_t dim,
                             miopenReduceCalculationNanPropagation_t nanPropagation)
{
    auto input_dims  = input.desc.GetLengths();
    auto output_dims = ref_output.desc.GetLengths();

    auto reduce_size = input_dims[dim];
    // auto output_numel =
    //     std::accumulate(output_dims.begin(), output_dims.end(), 1LL, std::multiplies<int64_t>());
    auto output_numel = ref_output.desc.GetElementSize();

    auto inner_size = std::accumulate(
        input_dims.begin() + dim + 1, input_dims.end(), 1ULL, std::multiplies<uint64_t>());

    par_ford(output_numel)([&](size_t o) {
        size_t input_idx = (o / inner_size) * inner_size * reduce_size + o % inner_size;

        T calculation = reduce_func<T, op>{}.get_initial_value();

        ford(reduce_size)([&](size_t i) {
            T val = input[input_idx];
            if(nanPropagation && std::isnan(val))
            {
                val = op == ReduceCalculationOp_t::Prod ? static_cast<T>(1.0) : static_cast<T>(0.0);
            }
            reduce_func<T, op>{}.calculate(calculation, val);
            input_idx += inner_size;
        });

        ref_output[o] = calculation;
    });
}

template <typename T, ReduceCalculationOp_t op>
void cpu_logical_calculation_forward(const tensor<T> input,
                                     tensor<uint8_t>& ref_output,
                                     uint32_t dim)
{
    auto input_dims  = input.desc.GetLengths();
    auto output_dims = ref_output.desc.GetLengths();

    auto reduce_size = input_dims[dim];
    // auto output_numel =
    //     std::accumulate(output_dims.begin(), output_dims.end(), 1LL, std::multiplies<int64_t>());
    auto output_numel = ref_output.desc.GetElementSize();

    auto inner_size = std::accumulate(
        input_dims.begin() + dim + 1, input_dims.end(), 1ULL, std::multiplies<uint64_t>());

    par_ford(output_numel)([&](size_t o) {
        size_t input_idx = (o / inner_size) * inner_size * reduce_size + o % inner_size;

        T calculation = reduce_func<T, op>{}.get_initial_value();

        ford(reduce_size)([&](size_t i) {
            T val = input[input_idx];
            reduce_func<T, op>{}.calculate(calculation, val);
            input_idx += inner_size;
        });

        ref_output[o] = calculation == 0 ? 0 : 1;
    });
}
