/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

template <typename T>
__device__ void lpPoolForward1d(const T* __restrict__ input,
                                T* __restrict__ output,
                                int64_t N,
                                int64_t C,
                                int64_t D,
                                int64_t OD,
                                int64_t kd,
                                int64_t sd,
                                float norm_type,
                                tensor_view_t<3> input_tv,
                                tensor_view_t<3> output_tv)
{
    int64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t nc = gid / OD, od = gid % OD;
    int64_t n = nc / C, c = nc % C;

    if(n >= N)
        return;

    FLOAT_ACCUM sum  = 0;
    FLOAT_ACCUM invp = static_cast<FLOAT_ACCUM>(1.0f) / static_cast<FLOAT_ACCUM>(norm_type);
    for(int64_t r = 0; r < kd; ++r)
    {
        int64_t d = od * sd + r;
        if(d < 0 || d >= D)
            continue;

        sum += static_cast<FLOAT_ACCUM>(
            pow(CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({n, c, d})]),
                static_cast<FLOAT_ACCUM>(norm_type)));
    }

    output[output_tv.get_tensor_view_idx({n, c, od})] = CVT_ACCUM2FLOAT(pow(sum, invp));
}

extern "C" __global__ void LPPoolForward1d(const D_TYPE* __restrict__ input,
                                           D_TYPE* __restrict__ output,
                                           int64_t N,
                                           int64_t C,
                                           int64_t D,
                                           int64_t OD,
                                           int64_t kd,
                                           int64_t sd,
                                           float norm_type,
                                           tensor_view_t<3> input_tv,
                                           tensor_view_t<3> output_tv)
{
    lpPoolForward1d<D_TYPE>(input, output, N, C, D, OD, kd, sd, norm_type, input_tv, output_tv);
}

template <typename T>
__device__ void lpPoolForward2d(const T* __restrict__ input,
                                T* __restrict__ output,
                                int64_t N,
                                int64_t C,
                                int64_t H,
                                int64_t W,
                                int64_t OH,
                                int64_t OW,
                                int64_t kh,
                                int64_t kw,
                                int64_t sh,
                                int64_t sw,
                                float norm_type,
                                tensor_view_t<4> input_tv,
                                tensor_view_t<4> output_tv)
{
    int64_t gid  = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t ncoh = gid / OW, ow = gid % OW;
    int64_t nc = ncoh / OH, oh = ncoh % OH;
    int64_t n = nc / C, c = nc % C;

    if(n >= N)
        return;

    FLOAT_ACCUM sum  = 0;
    FLOAT_ACCUM invp = static_cast<FLOAT_ACCUM>(1.0f) / static_cast<FLOAT_ACCUM>(norm_type);
    for(int64_t r = 0; r < kh; ++r)
    {
        for(int64_t s = 0; s < kw; ++s)
        {
            int64_t h = oh * sh + r;
            if(h < 0 || h >= H)
                continue;
            int64_t w = ow * sw + s;
            if(w < 0 || w >= W)
                continue;

            sum += static_cast<FLOAT_ACCUM>(
                pow(CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({n, c, h, w})]),
                    static_cast<FLOAT_ACCUM>(norm_type)));
        }
    }

    output[output_tv.get_tensor_view_idx({n, c, oh, ow})] = CVT_ACCUM2FLOAT(pow(sum, invp));
}

extern "C" __global__ void LPPoolForward2d(const D_TYPE* __restrict__ input,
                                           D_TYPE* __restrict__ output,
                                           int64_t N,
                                           int64_t C,
                                           int64_t H,
                                           int64_t W,
                                           int64_t OH,
                                           int64_t OW,
                                           int64_t kh,
                                           int64_t kw,
                                           int64_t sh,
                                           int64_t sw,
                                           float norm_type,
                                           tensor_view_t<4> input_tv,
                                           tensor_view_t<4> output_tv)
{
    lpPoolForward2d<D_TYPE>(
        input, output, N, C, H, W, OH, OW, kh, kw, sh, sw, norm_type, input_tv, output_tv);
}

template <typename T>
__device__ void lpPoolBackward1d(const T* __restrict__ input,
                                 const T* __restrict__ output,
                                 const T* __restrict__ output_grad,
                                 T* __restrict__ input_grad,
                                 int64_t N,
                                 int64_t C,
                                 int64_t D,
                                 int64_t OD,
                                 int64_t kd,
                                 int64_t sd,
                                 float norm_type,
                                 tensor_view_t<3> input_tv,
                                 tensor_view_t<3> output_tv,
                                 tensor_view_t<3> output_grad_tv,
                                 tensor_view_t<3> input_grad_tv)
{
    int64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t nc = gid / D, d = gid % D;
    int64_t n = nc / C, c = nc % C;

    if(n >= N)
        return;

    FLOAT_ACCUM sum     = 0.0f;
    FLOAT_ACCUM d_input = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({n, c, d})]);
    for(int64_t r = 0; r < kd; ++r)
    {
        int64_t odsd = d - r;
        if(odsd % sd != 0)
            continue;
        int64_t od = odsd / sd;
        if(od < 0 || od >= OD)
            continue;

        // gradient of p-norm is x_j * |x_j|^{p-2} / |x|_p^{p-1}
        sum += CVT_FLOAT2ACCUM(output_grad[output_grad_tv.get_tensor_view_idx({n, c, od})]) *
               d_input *
               static_cast<FLOAT_ACCUM>(pow(d_input, static_cast<FLOAT_ACCUM>(norm_type) - 2.0f)) /
               static_cast<FLOAT_ACCUM>(
                   pow(CVT_FLOAT2ACCUM(output[output_tv.get_tensor_view_idx({n, c, od})]),
                       static_cast<FLOAT_ACCUM>(norm_type) - 1.0f));
    }

    input_grad[input_grad_tv.get_tensor_view_idx({n, c, d})] = CVT_ACCUM2FLOAT(sum);
}

extern "C" __global__ void LPPoolBackward1d(const D_TYPE* __restrict__ input,
                                            const D_TYPE* __restrict__ output,
                                            const D_TYPE* __restrict__ output_grad,
                                            D_TYPE* __restrict__ input_grad,
                                            int64_t N,
                                            int64_t C,
                                            int64_t D,
                                            int64_t OD,
                                            int64_t kd,
                                            int64_t sd,
                                            float norm_type,
                                            tensor_view_t<3> input_tv,
                                            tensor_view_t<3> output_tv,
                                            tensor_view_t<3> output_grad_tv,
                                            tensor_view_t<3> input_grad_tv)
{
    lpPoolBackward1d<D_TYPE>(input,
                             output,
                             output_grad,
                             input_grad,
                             N,
                             C,
                             D,
                             OD,
                             kd,
                             sd,
                             norm_type,
                             input_tv,
                             output_tv,
                             output_grad_tv,
                             input_grad_tv);
}

template <typename T>
__device__ void lpPoolBackward2d(const T* __restrict__ input,
                                 const T* __restrict__ output,
                                 const T* __restrict__ output_grad,
                                 T* __restrict__ input_grad,
                                 int64_t N,
                                 int64_t C,
                                 int64_t H,
                                 int64_t W,
                                 int64_t OH,
                                 int64_t OW,
                                 int64_t kh,
                                 int64_t kw,
                                 int64_t sh,
                                 int64_t sw,
                                 float norm_type,
                                 tensor_view_t<4> input_tv,
                                 tensor_view_t<4> output_tv,
                                 tensor_view_t<4> output_grad_tv,
                                 tensor_view_t<4> input_grad_tv)
{
    int64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t nch = gid / W, w = gid % W;
    int64_t nc = nch / H, h = nch % H;
    int64_t n = nc / C, c = nc % C;

    if(n >= N)
        return;

    FLOAT_ACCUM sum     = 0;
    FLOAT_ACCUM d_input = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({n, c, h, w})]);
    for(int64_t r = 0; r < kh; ++r)
    {
        for(int64_t s = 0; s < kw; ++s)
        {
            int64_t ohsh = h - r;
            if(ohsh % sh != 0)
                continue;
            int64_t oh = ohsh / sh;
            if(oh < 0 || oh >= OH)
                continue;
            int64_t owsw = w - s;
            if(owsw % sw != 0)
                continue;
            int64_t ow = owsw / sw;
            if(ow < 0 || ow >= OW)
                continue;

            // gradient of p-norm is x_j * |x_j|^{p-2} / |x|_p^{p-1}
            sum +=
                CVT_FLOAT2ACCUM(output_grad[output_grad_tv.get_tensor_view_idx({n, c, oh, ow})]) *
                d_input *
                static_cast<FLOAT_ACCUM>(pow(d_input, static_cast<FLOAT_ACCUM>(norm_type) - 2.0f)) /
                static_cast<FLOAT_ACCUM>(
                    pow(CVT_FLOAT2ACCUM(output[output_tv.get_tensor_view_idx({n, c, oh, ow})]),
                        static_cast<FLOAT_ACCUM>(norm_type) - 1.0f));
        }
    }

    input_grad[input_grad_tv.get_tensor_view_idx({n, c, h, w})] = CVT_ACCUM2FLOAT(sum);
}

extern "C" __global__ void LPPoolBackward2d(const D_TYPE* __restrict__ input,
                                            const D_TYPE* __restrict__ output,
                                            const D_TYPE* __restrict__ output_grad,
                                            D_TYPE* __restrict__ input_grad,
                                            int64_t N,
                                            int64_t C,
                                            int64_t H,
                                            int64_t W,
                                            int64_t OH,
                                            int64_t OW,
                                            int64_t kh,
                                            int64_t kw,
                                            int64_t sh,
                                            int64_t sw,
                                            float norm_type,
                                            tensor_view_t<4> input_tv,
                                            tensor_view_t<4> output_tv,
                                            tensor_view_t<4> output_grad_tv,
                                            tensor_view_t<4> input_grad_tv)
{
    lpPoolBackward2d<D_TYPE>(input,
                             output,
                             output_grad,
                             input_grad,
                             N,
                             C,
                             H,
                             W,
                             OH,
                             OW,
                             kh,
                             kw,
                             sh,
                             sw,
                             norm_type,
                             input_tv,
                             output_tv,
                             output_grad_tv,
                             input_grad_tv);
}
