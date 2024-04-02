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
#ifndef GUARD_CPU_LAYERNORM_HPP
#define GUARD_CPU_LAYERNORM_HPP

#include "tensor_holder.hpp"

template <class T>
void cpu_layernorm_forward(tensor<T> input,
                           tensor<T> weight,
                           tensor<T> bias,
                           tensor<T>& ref_output,
                           tensor<T>& ref_mean,
                           tensor<T>& ref_rstd,
                           float eps,
                           int32_t dim,
                           miopenNormMode_t mode)
{
    auto dims         = input.desc.GetLengths();
    size_t outer_size = 1;
    size_t inner_size = 1;
    size_t i          = 0;
    for(; i < dim; i++)
    {
        outer_size *= dims[i];
    }

    for(; i < dims.size(); i++)
    {
        inner_size *= dims[i];
    }

    par_ford(outer_size)([&](int32_t o) {
        T mean_v = 0.0f;
        T var_v  = 0.0f;

        ford(inner_size)([&](int32_t i) {
            T tmp = input[o * inner_size + i];
            mean_v += tmp;
            var_v += tmp * tmp;
        });

        mean_v   = mean_v / inner_size;
        var_v    = var_v / inner_size - mean_v * mean_v;
        T rstd_v = 1.0f / sqrt(var_v + eps);

        ref_mean[o] = mean_v;
        ref_rstd[o] = rstd_v;

        ford(inner_size)([&](int32_t i) {
            T weight_v = mode ? weight[i] : 1;
            T bias_v   = mode ? bias[i] : 0;
            ref_output[o * inner_size + i] =
                (input[o * inner_size + i] - mean_v) * rstd_v * weight_v + bias_v;
        });
    });
}
#endif
