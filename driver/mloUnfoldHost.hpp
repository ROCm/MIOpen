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

#include <../test/ford.hpp>
#include "tensor_view.hpp"
#include <miopen/tensor_view_utils.hpp>
#include <vector>
#include <miopen/tensor.hpp>

template <typename Tgpu, typename Tcheck>
int32_t mloUnFoldFwd4DRunHost(Tgpu* input,
                              const miopenTensorDescriptor_t inputDesc,
                              Tcheck* ref_output,
                              const miopenTensorDescriptor_t ref_outputDesc,
                              const std::vector<uint64_t> kernel_size,
                              const std::vector<uint64_t> stride,
                              const std::vector<uint64_t> padding,
                              const std::vector<uint64_t> dilation)
{
    auto input_tv   = miopen::get_inner_expanded_tv<4>(miopen::deref(inputDesc));
    auto output_tv  = miopen::get_inner_expanded_tv<3>(miopen::deref(ref_outputDesc));
    auto input_dims = miopen::deref(inputDesc).GetLengths();
    auto input_size = miopen::deref(inputDesc).GetNumDims();

    const int LOCAL_SIZE = 256;
    int spatial_dim_size = input_size - 2;
    const uint64_t N     = static_cast<uint64_t>(input_dims[0]);
    const uint64_t C     = static_cast<uint64_t>(input_dims[1]);
    uint64_t P = 1, L = 1;
    std::vector<uint64_t> ls;
    for(int i = 0; i < spatial_dim_size; ++i)
    {
        P *= kernel_size[i];
        uint64_t l = (static_cast<uint64_t>(input_dims[i + 2]) + 2 * padding[i] -
                      dilation[i] * (kernel_size[i] - 1) - 1) /
                         stride[i] +
                     1;
        L *= l;
        ls.push_back(l);
    }
    uint64_t kernel_size_w = kernel_size[1];
    uint64_t stride_h      = stride[0];
    uint64_t stride_w      = stride[1];
    uint64_t padding_h     = padding[0];
    uint64_t padding_w     = padding[1];
    uint64_t dilation_h    = dilation[0];
    uint64_t dilation_w    = dilation[1];
    uint64_t LW            = ls[1];
    uint64_t H             = static_cast<uint64_t>(input_dims[2]);
    uint64_t W             = static_cast<uint64_t>(input_dims[3]);
    uint64_t work_size     = (((N * C * P * L) + LOCAL_SIZE - 1) / LOCAL_SIZE) * LOCAL_SIZE;
    par_ford(work_size)([&](uint64_t gid) {
        uint64_t ncp = gid / L, l = gid % L;
        uint64_t nc = ncp / P, p = ncp % P;
        uint64_t n = nc / C, c = nc % C;
        if(n >= N)
            return;

        uint64_t lh = l / LW, lw = l % LW;                       // sliding window position
        uint64_t ph = p / kernel_size_w, pw = p % kernel_size_w; // position inside kernel

        Tgpu x = static_cast<Tgpu>(0.0f);
        if(lh * stride_h >= padding_h + ph * dilation_h &&
           lw * stride_w >= padding_w + pw * dilation_w)
        {
            uint64_t h = lh * stride_h - padding_h + ph * dilation_h;
            uint64_t w = lw * stride_w - padding_w + pw * dilation_w;
            if(h < H && w < W)
            {
                long input_idx = input_tv.stride[3] * w + input_tv.stride[2] * h +
                                 input_tv.stride[1] * c + input_tv.stride[0] * n;
                x = input[input_idx];
            }
        }

        long output_idx =
            output_tv.stride[2] * l + output_tv.stride[1] * (c * P + p) + output_tv.stride[0] * n;
        ref_output[output_idx] = static_cast<Tcheck>(x);
    });

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int32_t mloUnFoldBwd4DRunHost(Tcheck* ref_dinput,
                              const miopenTensorDescriptor_t dinputDesc,
                              Tgpu* doutput,
                              const miopenTensorDescriptor_t doutputDesc,
                              const std::vector<uint64_t> kernel_size,
                              const std::vector<uint64_t> stride,
                              const std::vector<uint64_t> padding,
                              const std::vector<uint64_t> dilation)
{
    auto input_grad_tv   = miopen::get_inner_expanded_tv<4>(miopen::deref(dinputDesc));
    auto output_grad_tv  = miopen::get_inner_expanded_tv<3>(miopen::deref(doutputDesc));
    auto input_grad_dims = miopen::deref(dinputDesc).GetLengths();
    auto input_size      = miopen::deref(dinputDesc).GetNumDims();

    const int LOCAL_SIZE = 256;
    int spatial_dim_size = input_size - 2;
    const uint64_t N     = static_cast<uint64_t>(input_grad_dims[0]);
    const uint64_t C     = static_cast<uint64_t>(input_grad_dims[1]);
    uint64_t P           = 1;
    std::vector<uint64_t> ls;
    for(int i = 0; i < spatial_dim_size; ++i)
    {
        P *= kernel_size[i];
        uint64_t l = (static_cast<uint64_t>(input_grad_dims[i + 2]) + 2 * padding[i] -
                      dilation[i] * (kernel_size[i] - 1) - 1) /
                         stride[i] +
                     1;
        ls.push_back(l);
    }
    uint64_t kernel_size_h = kernel_size[0];
    uint64_t kernel_size_w = kernel_size[1];
    uint64_t stride_h      = stride[0];
    uint64_t stride_w      = stride[1];
    uint64_t padding_h     = padding[0];
    uint64_t padding_w     = padding[1];
    uint64_t dilation_h    = dilation[0];
    uint64_t dilation_w    = dilation[1];
    uint64_t LH            = ls[0];
    uint64_t LW            = ls[1];
    uint64_t H             = static_cast<uint64_t>(input_grad_dims[2]);
    uint64_t W             = static_cast<uint64_t>(input_grad_dims[3]);
    uint64_t work_size     = (((N * C * H * W) + LOCAL_SIZE - 1) / LOCAL_SIZE) * LOCAL_SIZE;
    par_ford(work_size)([&](uint64_t gid) {
        uint64_t nch = gid / W, w = gid % W;
        uint64_t nc = nch / H, h = nch % H;
        uint64_t n = nc / C, c = nc % C;
        if(n >= N)
            return;

        float sum = 0.0f;

        for(uint64_t ph = 0; ph < kernel_size_h; ++ph)
        {
            for(uint64_t pw = 0; pw < kernel_size_w; ++pw)
            {
                if(h < ph * dilation_h + padding_h)
                    continue;
                if(w < pw * dilation_w + padding_w)
                    continue;
                uint64_t lhsh = h - ph * dilation_h + padding_h;
                uint64_t lwsw = w - pw * dilation_w + padding_w;
                if(lhsh % stride_h != 0)
                    continue;
                if(lwsw % stride_w != 0)
                    continue;
                uint64_t lh = lhsh / stride_h;
                uint64_t lw = lwsw / stride_w;
                if(LH <= lh)
                    continue;
                if(LW <= lw)
                    continue;
                long output_grad_idx =
                    output_grad_tv.stride[2] * (lh * LW + lw) +
                    output_grad_tv.stride[1] * (c * P + (ph * kernel_size_w + pw)) +
                    output_grad_tv.stride[0] * n;
                sum += static_cast<float>(doutput[output_grad_idx]);
            }
        }

        long input_grad_idx = input_grad_tv.stride[3] * w + input_grad_tv.stride[2] * h +
                              input_grad_tv.stride[1] * c + input_grad_tv.stride[0] * n;
        ref_dinput[input_grad_idx] = static_cast<Tcheck>(sum);
    });

    return miopenStatusSuccess;
}
