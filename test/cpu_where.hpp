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
#ifndef GUARD_CPU_WHERE_HPP
#define GUARD_CPU_WHERE_HPP

#include "ford.hpp"
#include "tensor_holder.hpp"

template <class T>
void cpu_where_forward(tensor<T> input, tensor<T> other, tensor<T> cond, tensor<T>& ref_output)
{
    auto inputSize = input.desc.GetElementSize();
    auto otherSize = other.desc.GetElementSize();
    auto condSize = cond.desc.GetElementSize();
    auto outputSize = ref_output.desc.GetElementSize();

    par_ford(outputSize)([&](size_t o) {
        if (cond[o % condSize]) {
            ref_output[o] = input[o % inputSize];
        } else {
            ref_output[o] = other[o % otherSize];
        }
    });
}

template <class T>
void cpu_where_backward(tensor<T> outputGrad, tensor<T> cond, tensor<T>& inputGrad, tensor<T>& otherGrad)
{
    auto outputGradSize = outputGrad.desc.GetElementSize();
    auto condSize       = cond.desc.GetElementSize();
    auto inputGradSize  = inputGrad.desc.GetElementSize();
    auto otherGradSize  = otherGrad.desc.GetElementSize();

    par_ford(inputGradSize)([&](size_t i) { 
        inputGrad[i] = outputGrad[i % outputGradSize] * cond[i % condSize]; 
    });

    par_ford(otherGradSize)([&](size_t i) { 
        otherGrad[i] = outputGrad[i % outputGradSize] * (1 - cond[i % condSize]); 
    });
}

#endif
