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
#ifndef GUARD_CPU_NLLLOSS_HPP
#define GUARD_CPU_NLLLOSS_HPP

#include "tensor_holder.hpp"

template <class T>
void cpu_nllloss_forward_4d(tensor<T> input,
                            tensor<int32_t> target,
                            tensor<T> weight,
                            tensor<T>& output,
                            int32_t ignore_index)
{
    auto dims = input.desc.GetLengths();
    size_t N  = dims[0];
    size_t C  = dims[1];
    size_t D1 = dims[2];
    size_t D2 = dims[3];

    for(size_t n = 0; n < N; n++)
    {
        for(size_t d1 = 0; d1 < D1; d1++)
        {
            for(size_t d2 = 0; d2 < D2; d2++)
            {
                size_t target_index = n * D1 * D2 + d1 * D2 + d2;
                int32_t t           = target[target_index];
                size_t input_index  = (n * C + t) * D1 * D2 + d1 * D2 + d2;
                size_t weight_index = t;
                size_t output_index = target_index;

                if(t < 0 || t == ignore_index || t >= C)
                {
                    output[output_index] = static_cast<T>(0);
                }
                else
                {
                    output[output_index] =
                        static_cast<T>(-1.0f) * weight[weight_index] * input[input_index];
                }
            }
        }
    }
}
#endif
