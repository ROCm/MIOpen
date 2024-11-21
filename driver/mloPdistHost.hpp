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

#include <math.h>

#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <miopen/pdist/utils.hpp>
#include <../test/ford.hpp>

template <typename Tgpu, typename Tcheck>
int32_t mloPdistBackwardRunHost(const miopenTensorDescriptor_t inputDesc,
                                const Tgpu* input,
                                const Tgpu* output,
                                const Tgpu* doutput,
                                Tcheck* dinputHost,
                                const double p)
{
    auto input_numel = miopen::deref(inputDesc).GetElementSize();
    size_t N         = miopen::deref(inputDesc).GetLengths()[0];
    size_t M         = miopen::deref(inputDesc).GetLengths()[1];

    // Fill dinputHost with zeros
    std::fill(dinputHost, dinputHost + input_numel, static_cast<Tcheck>(0));

    for(size_t i = 0; i < N - 1; ++i)
    {
        for(size_t j = i + 1; j < N; ++j)
        {
            size_t k        = j + N * i - i * (i + 1) / 2 - i - 1;
            double grad_k   = static_cast<double>(doutput[k]);
            double output_k = static_cast<double>(output[k]);

            for(size_t m = 0; m < M; ++m)
            {
                double input_first  = static_cast<double>(input[i * M + m]);
                double input_second = static_cast<double>(input[j * M + m]);
                double diff         = input_first - input_second;

                Tcheck res =
                    static_cast<Tcheck>(miopen::solver::pdist::backward(diff, grad_k, output_k, p));

                dinputHost[i * M + m] += res;
                dinputHost[j * M + m] -= res;
            }
        }
    }

    return miopenStatusSuccess;
}
