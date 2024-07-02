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
#ifndef GUARD_CPU_TRANSFORMERS_ADAM_W_HPP
#define GUARD_CPU_TRANSFORMERS_ADAM_W_HPP

#include "tensor_holder.hpp"

template <typename T1, typename T2>
void cpu_transformers_adam_w(tensor<T1>& params,
                             tensor<T2>& grads,
                             tensor<T1>& exp_avgs,
                             tensor<T1>& exp_avg_sqs,
                             float lr,
                             float beta1,
                             float beta2,
                             float weight_decay,
                             float eps,
                             bool correct_bias,
                             bool is_amp,
                             int32_t grad_scale,
                             bool found_inf,
                             int32_t step_count)
{
    if(is_amp && found_inf)
        return;

    par_ford(params.GetSize())([&](int32_t i) {
        T1 param      = params[i];
        T1 exp_avg    = exp_avgs[i];
        T1 exp_avg_sq = exp_avg_sqs[i];

        for(int step = 1; step <= step_count; step++)
        {
            T1 grad = grads[i];
            if(is_amp)
                grad /= grad_scale;

            exp_avg    = exp_avg * beta1 + grad * (1 - beta1);
            exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1 - beta2);

            float denorm    = sqrt(exp_avg_sq) + eps;
            float step_size = lr;

            if(correct_bias)
            {
                float bias_correction1 = 1 - pow(beta1, step);
                float bias_correction2 = 1 - pow(beta2, step);
                step_size              = step_size * sqrt(bias_correction2) / bias_correction1;
            }

            param = param + exp_avg / denorm * -step_size;

            if(weight_decay > 0.0)
            {
                param = param - param * (lr * weight_decay);
            }
        }

        params[i] = param;
    });
}

#endif
