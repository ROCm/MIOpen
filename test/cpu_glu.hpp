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
#ifndef GUARD_CPU_GLU_HPP
#define GUARD_CPU_GLU_HPP

#include "tensor_holder.hpp"

template <typename T>
T sigmoid(T x)
{
    return static_cast<T>(1.0f / (1.0f + exp(-x)));
}

template <class T>
void cpu_glu_forward(tensor<T> input, tensor<T>& ref_output, int dim)
{
    auto input_dims = input.desc.GetLengths();
    auto output_dims = ref_output.desc.GetLengths();

    auto splitDim_size   = input_dims[dim];
    auto splitedDim_size = output_dims[dim];
    auto output_numel =
        std::accumulate(output_dims.begin(), output_dims.end(), 1L, std::multiplies<int64_t>());

    auto inner_size = 1ULL;
    for(int32_t i = dim + 1; i < input_dims.size(); i++)
    {
        inner_size *= input_dims[i];
    }

    par_ford(output_numel)([&](size_t o) {
        size_t innerIdx       = o % inner_size;
        size_t splittedDimIdx = ((o - innerIdx) / inner_size) % splitedDim_size;
        size_t outerIdx =
            (o - innerIdx - splittedDimIdx * inner_size) / (inner_size * splitedDim_size);
        size_t inputIdx1 =
            outerIdx * splitDim_size * inner_size + splittedDimIdx * inner_size + innerIdx;
        size_t inputIdx2 = outerIdx * splitDim_size * inner_size +
                           (splittedDimIdx + splitedDim_size) * inner_size + innerIdx;
        T valA        = input[inputIdx1];
        T valB        = input[inputIdx2];
        T val         = valA * sigmoid(valB);
        ref_output[o] = val;
    });
}
#endif
