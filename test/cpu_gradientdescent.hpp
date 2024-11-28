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
void cpu_GradientDescent(const tensor<T>& var_in,
                         tensor<T>& var_out,
                         const tensor<T>& alpha_in,
                         const tensor<T>& delta_in)
{
    auto var_in_tv   = miopen::get_inner_expanded_tv<5>(var_in.desc);
    auto var_out_tv  = miopen::get_inner_expanded_tv<5>(var_out.desc);
    auto alpha_in_tv = miopen::get_inner_expanded_tv<1>(alpha_in.desc);
    auto delta_in_tv = miopen::get_inner_expanded_tv<5>(delta_in.desc);

    uint64_t N = var_in.desc.GetElementSize();

    par_ford(N)([&](uint64_t gid) {
        auto tensor_layout = tensor_layout_t<5>(var_in_tv, gid);
        double var   = static_cast<double>(var_in[var_in_tv.get_tensor_view_idx(tensor_layout)]);
        double alpha = static_cast<double>(alpha_in[alpha_in_tv.get_tensor_view_idx({0})]);
        double delta =
            static_cast<double>(delta_in[delta_in_tv.get_tensor_view_idx(tensor_layout)]);

        var -= alpha * delta;

        var_out[var_out_tv.get_tensor_view_idx(tensor_layout)] = static_cast<T>(var);
    });
}
