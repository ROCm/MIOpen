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
#ifdef MIOPEN_BETA_API
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
                           int dim,
                           miopenLayerNormMode_t mode)
{
    auto dims         = input.desc.GetLengths();
    size_t grid_size  = 1;
    size_t outer_size = 1;
    size_t inner_size = 1;
    size_t i          = 0;
    for(; i < dim; i++)
    {
        outer_size *= dims[i];
        grid_size *= dims[i];
    }

    for(; i < dims.size(); i++)
    {
        inner_size *= dims[i];
        grid_size *= dims[i];
    }

    par_ford(outer_size)([&](int o) {
        T mean_v = 0.0f;
        T var_v  = 0.0f;

        ford(inner_size)([&](int i) {
            T tmp = input[o * inner_size + i];
            mean_v += tmp;
            var_v += tmp * tmp;
        });

        mean_v   = mean_v / inner_size;
        var_v    = var_v / inner_size - mean_v * mean_v;
        T rstd_v = 1.0f / sqrt(var_v + eps);

        ref_mean[o] = mean_v;
        ref_rstd[o] = rstd_v;

        ford(inner_size)([&](int i) {
            T weight_v = mode ? 1 : weight[i];
            T bias_v   = mode ? 0 : bias[i];
            ref_output[o * inner_size + i] =
                (input[o * inner_size + i] - mean_v) * rstd_v * weight_v + bias_v;
        });
    });
}

template <class T>
void cpu_layernorm_backward(tensor<T> input,
                            tensor<T> doutput,
                            tensor<T> weight,
                            tensor<T> mean,
                            tensor<T> rstd,
                            tensor<T>& ref_dinput,
                            tensor<T>& ref_dweight,
                            tensor<T>& ref_dbias,
                            int dim,
                            miopenLayerNormMode_t mode)
{
    auto dims         = input.desc.GetLengths();
    size_t grid_size  = 1;
    size_t outer_size = 1;
    size_t inner_size = 1;
    size_t i          = 0;
    for(; i < dim; i++)
    {
        outer_size *= dims[i];
        grid_size *= dims[i];
    }

    for(; i < dims.size(); i++)
    {
        inner_size *= dims[i];
        grid_size *= dims[i];
    }

    par_ford(outer_size)([&](int o) {
        T sum1 = 0.0f;
        T sum2 = 0.0f;
        ford(inner_size)([&](int i) {
            T weight_v = mode ? 1 : weight[i];
            T dy       = doutput[o * inner_size + i];
            T x        = input[o * inner_size + i];

            sum1 += dy * x * weight_v;
            sum2 += dy * weight_v;
        });

        T ds = sum1;
        T db = sum2;

        T s = 1.0f / inner_size;

        T mean_v = mean[o];
        T rstd_v = rstd[o];

        T a  = (db * mean_v - ds) * rstd_v * rstd_v * rstd_v * s;
        T c2 = -(a * mean_v + db * rstd_v * s);

        ford(inner_size)([&](int i) {
            T weight_v = mode ? 1 : weight[i];
            T dy       = doutput[o * inner_size + i];

            T val = rstd_v * dy * weight_v + a * input[o * inner_size + i] + c2;
            ref_dinput[o * inner_size + i] = val;
        });
    });

    if((ref_dweight.data.size() != 0) || ref_dbias.data.size() != 0)
    {
        par_ford(inner_size)([&](int i) {
            T sum1 = 0.0f;
            T sum2 = 0.0f;

            ford(outer_size)([&](int o) {
                T dy = doutput[o * inner_size + i];

                sum1 += dy * (input[o * inner_size + i] - mean[o]) * rstd[o];
                sum2 += dy;
            });

            ref_dweight[i] = sum1;
            ref_dbias[i]   = sum2;
        });
    }
}
#endif
#endif
