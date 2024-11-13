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

#include "tensor_holder.hpp"
#include <miopen/tensor_view_utils.hpp>
#include "ford.hpp"

template <class T>
void cpu_SGD_forward(tensor<T> param_in,
                     tensor<T>& param_out,
                     tensor<T> grad,
                     tensor<T> momentum_buffer_in,
                     tensor<T>& momentum_buffer_out,
                     double lr,
                     double momentum,
                     double dampening,
                     double weight_decay,
                     bool nesterov,
                     bool momentum_initialized)
{
    uint64_t param_size         = param_in.desc.GetElementSize();
    auto param_in_tv            = miopen::get_inner_expanded_tv<4>(param_in.desc);
    auto param_out_tv           = miopen::get_inner_expanded_tv<4>(param_out.desc);
    auto grad_tv                = miopen::get_inner_expanded_tv<4>(grad.desc);
    auto momentum_buffer_in_tv  = miopen::get_inner_expanded_tv<4>(momentum_buffer_in.desc);
    auto momentum_buffer_out_tv = miopen::get_inner_expanded_tv<4>(momentum_buffer_out.desc);

    par_ford(param_size)([&](auto gid) {
        uint64_t nch = gid / param_out_tv.size[3], w = gid % param_out_tv.size[3];
        uint64_t nc = nch / param_out_tv.size[2], h = nch % param_out_tv.size[2];
        uint64_t n = nc / param_out_tv.size[1], c = nc % param_out_tv.size[1];

        double param = static_cast<double>(param_in[param_in_tv.get_tensor_view_idx({n, c, h, w})]);
        double d_p   = static_cast<double>(grad[grad_tv.get_tensor_view_idx({n, c, h, w})]);

        if(weight_decay)
        {
            d_p += param * static_cast<double>(weight_decay);
        }

        if(momentum)
        {
            double momentum_v;
            if(momentum_initialized != 0)
            {
                momentum_v = static_cast<double>(
                    momentum_buffer_in[momentum_buffer_in_tv.get_tensor_view_idx({n, c, h, w})]);
                momentum_v = momentum_v * static_cast<double>(momentum) +
                             d_p * static_cast<double>(1 - dampening);
            }
            else
            {
                momentum_v = d_p;
            }
            momentum_buffer_out[momentum_buffer_out_tv.get_tensor_view_idx({n, c, h, w})] =
                static_cast<T>(momentum_v);

            if(nesterov != 0)
            {
                d_p = d_p + momentum_v * static_cast<double>(momentum);
            }
            else
            {
                d_p = momentum_v;
            }
        }

        param_out[param_out_tv.get_tensor_view_idx({n, c, h, w})] =
            static_cast<T>(param - static_cast<double>(lr) * d_p);
    });
}
