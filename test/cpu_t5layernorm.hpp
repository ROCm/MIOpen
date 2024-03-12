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
#ifndef GUARD_CPU_T5LAYERNORM_HPP
#define GUARD_CPU_T5LAYERNORM_HPP

#include "tensor_holder.hpp"

template <class T>
void cpu_t5layernorm_forward(tensor<T> x,
                             tensor<T> weight,
                             tensor<T>& ref_y,
                             tensor<T>& ref_rstd,
                             float eps,
                             miopenNormMode_t mode)
{
    auto dims         = x.desc.GetLengths();
    size_t outer_size = 1;
    size_t inner_size = dims[dims.size() - 1];

    for(size_t i = 0ULL; i < dims.size() - 1; ++i)
    {
        outer_size *= dims[i];
    }

    par_ford(outer_size)([&](int32_t o) {
        float pvar = 0;

        ford(inner_size)([&](int32_t i) {
            float tmp = static_cast<float>(x[o * inner_size + i]);
            pvar += tmp * tmp;
        });

        pvar        = pvar / inner_size;
        float prstd = 1 / sqrt(pvar + eps);

        ref_rstd[o] = static_cast<T>(prstd);

        ford(inner_size)([&](int32_t i) {
            float pweight = mode ? static_cast<float>(weight[i]) : 1;
            ref_y[o * inner_size + i] =
                static_cast<T>(static_cast<float>(x[o * inner_size + i]) * prstd * pweight);
        });
    });
}

template <class T>
void cpu_t5layernorm_backward(tensor<T> dy,
                              tensor<T> x,
                              tensor<T> weight,
                              tensor<T> rstd,
                              tensor<T>& ref_dx,
                              miopenNormMode_t mode)
{
    auto dims         = dy.desc.GetLengths();
    size_t outer_size = 1;
    size_t inner_size = dims[dims.size() - 1];

    for(size_t i = 0ULL; i < dims.size() - 1; ++i)
    {
        outer_size *= dims[i];
    }

    par_ford(outer_size)([&](int32_t o) {
        float sum = 0;

        ford(inner_size)([&](int32_t i) {
            float pweight = mode ? static_cast<float>(weight[i]) : 1;
            float pdy     = (dy.GetSize() != 0) ? static_cast<float>(dy[o * inner_size + i]) : 0;
            float px      = static_cast<float>(x[o * inner_size + i]);
            sum += pdy * px * pweight;
        });

        float s     = 1 / static_cast<float>(inner_size);
        float prstd = static_cast<float>(rstd[o]);
        float a     = sum * prstd * prstd * prstd * s;

        ford(inner_size)([&](int32_t i) {
            float pweight = mode ? static_cast<float>(weight[i]) : 1;
            float pdy     = (dy.GetSize() != 0) ? static_cast<float>(dy[o * inner_size + i]) : 0;

            float val = prstd * pdy * pweight - a * static_cast<float>(x[o * inner_size + i]);
            ref_dx[o * inner_size + i] = static_cast<T>(val);
        });
    });
}

template <class T>
void cpu_t5layernorm_backward_weight(
    tensor<T> dy, tensor<T> x, tensor<T> rstd, tensor<T>& ref_dw, miopenNormMode_t mode)
{
    auto dims         = dy.desc.GetLengths();
    size_t outer_size = 1;
    size_t inner_size = dims[dims.size() - 1];

    for(size_t i = 0ULL; i < dims.size() - 1; ++i)
    {
        outer_size *= dims[i];
    }

    par_ford(inner_size)([&](int32_t o) {
        float sum = 0;

        ford(outer_size)([&](int32_t i) {
            float prstd = static_cast<float>(rstd[o]);
            float pdy   = (dy.GetSize() != 0) ? static_cast<float>(dy[i * inner_size + o]) : 0;
            float px    = static_cast<float>(x[i * inner_size + o]);

            sum += pdy * px * prstd;
        });

        ref_dw[o] = sum;
    });
}
#endif
