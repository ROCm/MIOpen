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
#include "tensor_view.hpp"
#include <miopen/tensor_view_utils.hpp>

template <class T>
void cpu_KerasMomentum(const tensor<T>& var_in,
                       tensor<T>& var_out,
                       const tensor<T>& accum_in,
                       tensor<T>& accum_out,
                       const tensor<T>& lr_in,
                       const tensor<T>& grad_in,
                       const tensor<T>& momentum_in,
                       const bool nesterov)
{
    auto var_in_tv    = miopen::get_inner_expanded_tv<5>(var_in.desc);
    auto var_out_tv   = miopen::get_inner_expanded_tv<5>(var_out.desc);
    auto accum_in_tv  = miopen::get_inner_expanded_tv<5>(accum_in.desc);
    auto accum_out_tv = miopen::get_inner_expanded_tv<5>(accum_out.desc);
    auto grad_in_tv   = miopen::get_inner_expanded_tv<5>(grad_in.desc);

    uint64_t N = var_in.desc.GetElementSize();

    par_ford(N)([&](uint64_t gid) {
        auto tensor_layout = tensor_layout_t<5>(var_in_tv, gid);
        double var = static_cast<double>(var_in[var_in_tv.get_tensor_view_idx(tensor_layout)]);
        double accum =
            static_cast<double>(accum_in[accum_in_tv.get_tensor_view_idx(tensor_layout)]);
        double grad = static_cast<double>(grad_in[grad_in_tv.get_tensor_view_idx(tensor_layout)]);
        double lr   = static_cast<double>(lr_in[0]);
        double momentum = static_cast<double>(momentum_in[0]);

        accum = accum * momentum - grad * lr;

        if(nesterov)
        {
            var += accum * momentum - grad * lr;
        }
        else
        {
            var += accum;
        }

        var_out[var_out_tv.get_tensor_view_idx(tensor_layout)]     = static_cast<T>(var);
        accum_out[accum_out_tv.get_tensor_view_idx(tensor_layout)] = static_cast<T>(accum);
    });
}
