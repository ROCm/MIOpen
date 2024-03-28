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
#ifndef GUARD_CPU_ADDLAYERNORM_HPP
#define GUARD_CPU_ADDLAYERNORM_HPP

#include "tensor_holder.hpp"

template <class T>
void cpu_addlayernorm_forward(tensor<T> input,
                              tensor<T> input2,
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
        float mean_v = 0;
        float var_v  = 0;

        ford(inner_size)([&](int32_t i) {
            float tmp = static_cast<float>(input[o * inner_size + i]) +
                        static_cast<float>(input2[o * inner_size + i]);
            mean_v += tmp;
            var_v += tmp * tmp;
        });

        mean_v       = mean_v / inner_size;
        var_v        = var_v / inner_size - mean_v * mean_v;
        float rstd_v = 1 / sqrt(var_v + eps);

        ref_mean[o] = static_cast<T>(mean_v);
        ref_rstd[o] = static_cast<T>(rstd_v);

        ford(inner_size)([&](int32_t i) {
            float weight_v = mode ? static_cast<float>(weight[i]) : 1;
            float bias_v   = mode ? static_cast<float>(bias[i]) : 0;
            ref_output[o * inner_size + i] =
                static_cast<T>((static_cast<float>(input[o * inner_size + i]) +
                                static_cast<float>(input2[o * inner_size + i]) - mean_v) *
                                   rstd_v * weight_v +
                               bias_v);
        });
    });
}
#endif
