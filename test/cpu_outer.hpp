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
#ifndef GUARD_CPU_OUTER_HPP
#define GUARD_CPU_OUTER_HPP

#include "tensor_holder.hpp"

template <class T>
void cpu_outer_forward(tensor<T> input1, tensor<T> input2, tensor<T>& ref_output)
{
    auto input1_dims = input1.desc.GetLengths();
    auto input2_dims = input2.desc.GetLengths();
    auto output_dims = ref_output.desc.GetLengths();

    size_t in_n = input1_dims[0];
    size_t in_m = input2_dims[0];

    size_t cnt = 0;
    bool ok    = false;
    if(ok)
        std::cout << "cpu output"
                  << "\n";
    for(size_t i = 0; i < in_n; i++)
    {
        for(size_t j = 0; j < in_m; j++)
        {
            ref_output[cnt++] = input1[i] * input2[j];
            if(ok)
                std::cout << ref_output[cnt - 1] << " ";
        }
        if(ok)
            std::cout << "\n";
    }
}

template <class T>
void cpu_outer_backward(tensor<T> input1,
                        tensor<T> input2,
                        tensor<T> outputGrad,
                        tensor<T>& input1Grad,
                        tensor<T>& input2Grad)
{
    auto input1_dims = input1.desc.GetLengths();
    auto input2_dims = input2.desc.GetLengths();
    auto output_dims = outputGrad.desc.GetLengths();

    size_t in_n = input1_dims[0];
    size_t in_m = input2_dims[0];

    for(size_t i = 0; i < in_n; i++)
    {
        float sum = 0;
        for(size_t j = 0; j < in_m; j++)
        {
            sum += static_cast<float>(outputGrad[i * in_m + j]) * static_cast<float>(input2[j]);
        }
        input1Grad[i] = sum;
    }
    for(size_t j = 0; j < in_m; j++)
    {
        float sum = 0;
        for(size_t i = 0; i < in_n; i++)
        {
            sum += static_cast<float>(input1[i]) * static_cast<float>(outputGrad[i * in_m + j]);
        }
        input2Grad[j] = sum;
    }
}
#endif
