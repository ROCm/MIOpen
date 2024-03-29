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
#ifndef GUARD_CPU_ADAM_HPP
#define GUARD_CPU_ADAM_HPP

#include "tensor_holder.hpp"

template <typename T1, typename T2>
void cpu_adam(tensor<T1>& param,
              tensor<T2>& grad,
              tensor<T1>& exp_avg,
              tensor<T1>& exp_avg_sq,
              tensor<T1>& max_exp_avg_sq,
              float lr,
              float beta1,
              float beta2,
              float weight_decay,
              float eps,
              bool amsgrad,
              bool maximize,
              bool is_amp,
              int32_t grad_scale,
              bool found_inf,
              int32_t step_count)
{
    if(is_amp && found_inf)
        return;

    par_ford(param.GetSize())([&](int32_t i) {
        for(int step = 1; step <= step_count; step++)
        {
            T1 grad_tmp = grad[i];
            if(maximize)
                grad_tmp *= -1;
            if(is_amp)
                grad_tmp /= grad_scale;

            float bias_correction1 = 1 - pow(beta1, step);
            float bias_correction2 = 1 - pow(beta2, step);

            if(weight_decay != 0)
                grad_tmp += param[i] * weight_decay;

            exp_avg[i]    = exp_avg[i] * beta1 + grad_tmp * (1 - beta1);
            exp_avg_sq[i] = exp_avg_sq[i] * beta2 + grad_tmp * grad_tmp * (1 - beta2);

            float denom = 0;
            if(amsgrad)
            {
                if(exp_avg_sq[i] > max_exp_avg_sq[i])
                    max_exp_avg_sq[i] = exp_avg_sq[i];

                denom = sqrt(max_exp_avg_sq[i]) / sqrt(bias_correction2) + eps;
            }
            else
            {
                denom = sqrt(exp_avg_sq[i]) / sqrt(bias_correction2) + eps;
            }

            param[i] = param[i] - (lr / bias_correction1) * exp_avg[i] / denom;
        }
    });
}

#endif
