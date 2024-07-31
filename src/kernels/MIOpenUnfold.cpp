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

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "tensor_view.hpp"

template <typename DTYPE>
__device__ void unfoldForward4D(const DTYPE* __restrict__ input,
                                DTYPE* __restrict__ output,
                                int N,
                                int C,
                                int H,
                                int W,
                                int P,
                                int L,
                                int LH,
                                int LW,
                                int kernel_size_h,
                                int kernel_size_w,
                                int stride_h,
                                int stride_w,
                                int padding_h,
                                int padding_w,
                                int dilation_h,
                                int dilation_w,
                                tensor_view_t<4> input_tv,
                                tensor_view_t<3> output_tv)
{
    /*
     * input = {N, C, H, W}, output = {N, C * P, L}
     * where P = kernel_size_h * kernel_size_w, L = # of blocks (see host code for
     * formula)
     * => gws = {ceil(N * C * P * L, LOCAL_SIZE)}, lws = {LOCAL_SIZE}
     */

    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int ncp = gid / L, l = gid % L;
    int nc = ncp / P, p = ncp % P;
    int n = nc / C, c = nc % C;
    if(n >= N)
        return;

    int lh = l / LW, lw = l % LW;                       // sliding window position
    int ph = p / kernel_size_w, pw = p % kernel_size_w; // position inside kernel
    int h = lh * stride_h - padding_h + ph * dilation_h;
    int w = lw * stride_w - padding_w + pw * dilation_w;

    DTYPE x = 0;
    if(0 <= h && h < H && 0 <= w && w < W)
    {
        tensor_layout_t<4> input_layout(n, c, h, w);
        x = input[input_tv.get_tensor_view_idx(input_layout)];
    }
    tensor_layout_t<3> output_layout(n, c * P + p, l);
    output[output_tv.get_tensor_view_idx(output_layout)] = x;
}

extern "C" __global__ void UnfoldForward4D(const FLOAT* __restrict__ input,
                                           FLOAT* __restrict__ output,
                                           int N,
                                           int C,
                                           int H,
                                           int W,
                                           int P,
                                           int L,
                                           int LH,
                                           int LW,
                                           int kernel_size_h,
                                           int kernel_size_w,
                                           int stride_h,
                                           int stride_w,
                                           int padding_h,
                                           int padding_w,
                                           int dilation_h,
                                           int dilation_w,
                                           tensor_view_t<4> input_tv,
                                           tensor_view_t<3> output_tv)
{
    unfoldForward4D<FLOAT>(input,
                           output,
                           N,
                           C,
                           H,
                           W,
                           P,
                           L,
                           LH,
                           LW,
                           kernel_size_h,
                           kernel_size_w,
                           stride_h,
                           stride_w,
                           padding_h,
                           padding_w,
                           dilation_h,
                           dilation_w,
                           input_tv,
                           output_tv);
}

template <typename DTYPE>
__device__ void unfoldBackward4D(const DTYPE* __restrict__ output_grad,
                                 DTYPE* __restrict__ input_grad,
                                 int N,
                                 int C,
                                 int H,
                                 int W,
                                 int P,
                                 int L,
                                 int LH,
                                 int LW,
                                 int kernel_size_h,
                                 int kernel_size_w,
                                 int stride_h,
                                 int stride_w,
                                 int padding_h,
                                 int padding_w,
                                 int dilation_h,
                                 int dilation_w,
                                 tensor_view_t<3> output_grad_tv,
                                 tensor_view_t<4> input_grad_tv)
{
    /*
     * output_grad = {N, C * P, L}, input_grad = {N, C, H, W}
     * where P = kernel_size_h * kernel_size_w, L = # of blocks (see host code for
     * formula)
     * => gws = {ceil(N * C * H * W, LOCAL_SIZE)}, lws = {LOCAL_SIZE}
     */

    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int nch = gid / W, w = gid % W;
    int nc = nch / H, h = nch % H;
    int n = nc / C, c = nc % C;
    if(n >= N)
        return;

    FLOAT_ACCUM sum = 0.0f;
    for(int ph = 0; ph < kernel_size_h; ++ph)
    {
        for(int pw = 0; pw < kernel_size_w; ++pw)
        {
            int lhsh = h - ph * dilation_h + padding_h;
            int lwsw = w - pw * dilation_w + padding_w;
            if(lhsh % stride_h != 0)
                continue;
            if(lwsw % stride_w != 0)
                continue;
            int lh = lhsh / stride_h;
            int lw = lwsw / stride_w;
            if(lh < 0 || LH <= lh)
                continue;
            if(lw < 0 || LW <= lw)
                continue;
            tensor_layout_t<3> output_grad_layout(
                n, c * P + (ph * kernel_size_w + pw), lh * LW + lw);
            sum += CVT_FLOAT2ACCUM(
                output_grad[output_grad_tv.get_tensor_view_idx(output_grad_layout)]);
        }
    }
    tensor_layout_t<4> input_grad_layout(n, c, h, w);
    input_grad[input_grad_tv.get_tensor_view_idx(input_grad_layout)] = CVT_ACCUM2FLOAT(sum);
}

extern "C" __global__ void UnfoldBackward4D(const FLOAT* __restrict__ output_grad,
                                            FLOAT* __restrict__ input_grad,
                                            int N,
                                            int C,
                                            int H,
                                            int W,
                                            int P,
                                            int L,
                                            int LH,
                                            int LW,
                                            int kernel_size_h,
                                            int kernel_size_w,
                                            int stride_h,
                                            int stride_w,
                                            int padding_h,
                                            int padding_w,
                                            int dilation_h,
                                            int dilation_w,
                                            tensor_view_t<3> output_grad_tv,
                                            tensor_view_t<4> input_grad_tv)
{
    unfoldBackward4D<FLOAT>(output_grad,
                            input_grad,
                            N,
                            C,
                            H,
                            W,
                            P,
                            L,
                            LH,
                            LW,
                            kernel_size_h,
                            kernel_size_w,
                            stride_h,
                            stride_w,
                            padding_h,
                            padding_w,
                            dilation_h,
                            dilation_w,
                            output_grad_tv,
                            input_grad_tv);
}
