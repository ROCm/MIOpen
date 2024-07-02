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
#ifndef GUARD_CPU_UNFOLD_HPP
#define GUARD_CPU_UNFOLD_HPP

#pragma once

#include "miopen/tensor.hpp"
#include "tensor_holder.hpp"
#include "tensor_view.hpp"
#include "miopen/tensor_view_utils.hpp"

template <class T>
void cpu_unfold_fwd_4d(tensor<T> input_tensor,
                            tensor<T>& ref_output_tensor,
                            const std::vector<int32_t> kernel_size,
                            const std::vector<int32_t> stride,
                            const std::vector<int32_t> padding,
                            const std::vector<int32_t> dilation)
{
    auto input_tv                = miopen::get_inner_expanded_tv<4>(input_tensor.desc);
    auto output_tv                = miopen::get_inner_expanded_tv<3>(ref_output_tensor.desc);
    auto input_size     = input_tensor.desc.GetSize();
    auto input_dims     = input_tensor.desc.GetLengths();

    auto input          = input_tensor.data.data();
    auto output         = ref_output_tensor.data.data();

    const int LOCAL_SIZE = 256;
    int spatial_dim_size = input_size - 2;

    const int32_t N = static_cast<int32_t>(input_dims[0]);
    const int32_t C = static_cast<int32_t>(input_dims[1]);

    int32_t P = 1, L = 1;
    std::vector<int32_t> ls;
    for (int i = 0; i < spatial_dim_size; ++i) {
        P *= kernel_size[i];
        int32_t l = (static_cast<int32_t>(input_dims[i + 2]) + 2 * padding[i] -
                    dilation[i] * (kernel_size[i] - 1) - 1) /
                        stride[i] +
                    1;
        L *= l;
        ls.push_back(l);
    }

    int32_t kernel_size_h = kernel_size[0];
    int32_t kernel_size_w = kernel_size[1];
    int32_t stride_h = stride[0];
    int32_t stride_w = stride[1];
    int32_t padding_h = padding[0];
    int32_t padding_w = padding[1];
    int32_t dilation_h = dilation[0];
    int32_t dilation_w = dilation[1];
    int32_t LH = ls[0];
    int32_t LW = ls[1];
    int32_t H = static_cast<int32_t>(input_dims[2]);
    int32_t W = static_cast<int32_t>(input_dims[3]);
    int work_size = (((N * C * P * L) + LOCAL_SIZE - 1) / LOCAL_SIZE) * LOCAL_SIZE;
    par_ford(work_size)([&](int gid) {
            int ncp = gid / L, l = gid % L;
            int nc = ncp / P, p = ncp % P;
            int n = nc / C, c = nc % C;
            if (n >= N) return;

            int lh = l / LW, lw = l % LW;  // sliding window position
            int ph = p / kernel_size_w, pw = p % kernel_size_w;  // position inside kernel
            int h = lh * stride_h - padding_h + ph * dilation_h;
            int w = lw * stride_w - padding_w + pw * dilation_w;

            T x = static_cast<T>(0.0f);
            if (0 <= h && h < H && 0 <= w && w < W) {
                long input_idx = input_tv.stride[3] * w + input_tv.stride[2] * h + input_tv.stride[1] * c + input_tv.stride[0] * n;
                x = input[input_idx];
            }

            long output_idx = output_tv.stride[2] * l + output_tv.stride[1] * (c * P + p) + output_tv.stride[0] * n;
            output[output_idx] = x;
    });
}
#endif
