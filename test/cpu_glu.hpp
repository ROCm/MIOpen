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
#ifndef GUARD_CPU_GLU_HPP
#define GUARD_CPU_GLU_HPP

#include "tensor_holder.hpp"

template <typename T>
T sigmoid(T x) { return 1.0f / (1.0f + exp(-x)); }

template <class T>
void cpu_glu_forward(tensor<T> inputFirstHalf,
                    tensor<T> inputSecondHalf,
                     tensor<T>& ref_output,
                     int32_t dim)
{
    auto input_dims  = inputFirstHalf.desc.GetLengths();
    auto output_dims = ref_output.desc.GetLengths();

    auto output_numel =
        std::accumulate(output_dims.begin(), output_dims.end(), 1L, std::multiplies<int64_t>());

    par_ford(output_numel)([&](size_t o) {
        T valA = inputFirstHalf[o];
        T valB = inputSecondHalf[o];
        T val = valA * sigmoid(valB);
        ref_output[o] = val;
    });
}
#endif
