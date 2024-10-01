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

#include <cstddef>
#include <cstdint>

#include "ford.hpp"
#include "tensor_holder.hpp"

template <class T>
void cpu_where_backward(const tensor<T>& outputGrad,
                        const tensor<uint8_t>& cond,
                        tensor<T>& inputGrad,
                        tensor<T>& otherGrad)
{
    auto outputGradSize = outputGrad.desc.GetElementSize();
    auto condSize       = cond.desc.GetElementSize();
    auto condData       = cond.data;
    auto inputGradSize  = inputGrad.data.empty() ? 0 : inputGrad.desc.GetElementSize();
    auto otherGradSize  = otherGrad.data.empty() ? 0 : otherGrad.desc.GetElementSize();

    par_ford(inputGradSize)([&](size_t i) {
        auto condVal = condData[i % condSize];
        inputGrad[i] = outputGrad[i % outputGradSize] * condVal;
    });

    par_ford(otherGradSize)([&](size_t i) {
        auto condVal = condData[i % condSize];
        otherGrad[i] = outputGrad[i % outputGradSize] * (1 - condVal);
    });
}
