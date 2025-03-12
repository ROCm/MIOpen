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
__device__ void adaptiveMaxPoolForward1d(const T* __restrict__ input,
                                         T* __restrict__ output,
                                         int64_t* __restrict__ indices,
                                         uint64_t N,
                                         uint64_t C,
                                         uint64_t H,
                                         uint64_t OH,
                                         tensor_view_t<3> input_tv,
                                         tensor_view_t<3> output_tv,
                                         tensor_view_t<3> indices_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t nc = gid / OH, oh = gid % OH;
    uint64_t n = nc / C, c = nc % C;
    if(n >= N)
        return;

    uint64_t h  = oh * H / OH;
    uint64_t kh = ((oh + 1) * H + OH - 1) / OH;

    FLOAT_ACCUM m = -INFINITY;
    if(!indices)
    {
        for(uint64_t ih = h; ih < kh; ++ih)
        {
            m = max(m, CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({n, c, ih})]));
        }
        output[output_tv.get_tensor_view_idx({n, c, oh})] = CVT_ACCUM2FLOAT(m);
    }
    else
    {
        uint64_t mi = 0;
        for(uint64_t ih = h; ih < kh; ++ih)
        {
            FLOAT_ACCUM input_val =
                CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({n, c, ih})]);
            if(m < input_val)
            {
                m  = input_val;
                mi = ih;
            }
        }
        output[output_tv.get_tensor_view_idx({n, c, oh})]   = CVT_ACCUM2FLOAT(m);
        indices[indices_tv.get_tensor_view_idx({n, c, oh})] = mi;
    }
}
extern "C" __global__ void AdaptiveMaxPoolForward1d(const D_TYPE* __restrict__ input,
                                                    D_TYPE* __restrict__ output,
                                                    int64_t* __restrict__ indices,
                                                    uint64_t N,
                                                    uint64_t C,
                                                    uint64_t H,
                                                    uint64_t OH,
                                                    tensor_view_t<3> input_tv,
                                                    tensor_view_t<3> output_tv,
                                                    tensor_view_t<3> indices_tv)
{
    adaptiveMaxPoolForward1d<D_TYPE>(
        input, output, indices, N, C, H, OH, input_tv, output_tv, indices_tv);
}

template <typename T>
__device__ void adaptiveMaxPoolBackward1d(const int64_t* __restrict__ indices,
                                          const T* __restrict__ output_grad,
                                          T* __restrict__ input_grad,
                                          uint64_t N,
                                          uint64_t C,
                                          uint64_t H,
                                          uint64_t OH,
                                          tensor_view_t<3> indices_tv,
                                          tensor_view_t<3> output_grad_tv,
                                          tensor_view_t<3> input_grad_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t nc = gid / H, h = gid % H;
    uint64_t n = nc / C, c = nc % C;
    if(n >= N)
        return;

    uint64_t oh  = (h * OH) / H;
    uint64_t koh = (((h + 1) * OH + H - 1) / H) - oh;

    FLOAT_ACCUM grad = 0;
    for(uint64_t ih = oh; ih < (oh + koh); ++ih)
    {
        uint64_t idx = indices[indices_tv.get_tensor_view_idx({n, c, ih})];
        if(idx == h)
        {
            grad += CVT_FLOAT2ACCUM(output_grad[output_grad_tv.get_tensor_view_idx({n, c, ih})]);
        }
    }
    input_grad[input_grad_tv.get_tensor_view_idx({n, c, h})] = CVT_ACCUM2FLOAT(grad);
}

extern "C" __global__ void AdaptiveMaxPoolBackward1d(const int64_t* __restrict__ indices,
                                                     const D_TYPE* __restrict__ output_grad,
                                                     D_TYPE* __restrict__ input_grad,
                                                     uint64_t N,
                                                     uint64_t C,
                                                     uint64_t H,
                                                     uint64_t OH,
                                                     tensor_view_t<3> indices_tv,
                                                     tensor_view_t<3> output_grad_tv,
                                                     tensor_view_t<3> input_grad_tv)
{
    adaptiveMaxPoolBackward1d<D_TYPE>(
        indices, output_grad, input_grad, N, C, H, OH, indices_tv, output_grad_tv, input_grad_tv);
}

template <typename T>
__device__ void adaptiveMaxPoolForward2d(const T* __restrict__ input,
                                         T* __restrict__ output,
                                         int64_t* __restrict__ indices,
                                         uint64_t N,
                                         uint64_t C,
                                         uint64_t H,
                                         uint64_t W,
                                         uint64_t OH,
                                         uint64_t OW,
                                         tensor_view_t<4> input_tv,
                                         tensor_view_t<4> output_tv,
                                         tensor_view_t<4> indices_tv)
{
    uint64_t gid  = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t ncoh = gid / OW, ow = gid % OW;
    uint64_t nc = ncoh / OH, oh = ncoh % OH;
    uint64_t n = nc / C, c = nc % C;

    if(n >= N)
        return;

    uint64_t h  = (oh * H) / OH;
    uint64_t kh = ((oh + 1) * H + OH - 1) / OH;

    uint64_t w  = (ow * W) / OW;
    uint64_t kw = ((ow + 1) * W + OW - 1) / OW;

    FLOAT_ACCUM m = -INFINITY;
    if(!indices)
    {
        for(uint64_t ih = h; ih < kh; ++ih)
        {
            for(uint64_t iw = w; iw < kw; ++iw)
            {
                m = max(m, CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({n, c, ih, iw})]));
            }
        }
        output[output_tv.get_tensor_view_idx({n, c, oh, ow})] = CVT_ACCUM2FLOAT(m);
    }
    else
    {
        uint64_t mi = 0;
        for(uint64_t ih = h; ih < kh; ++ih)
        {
            for(uint64_t iw = w; iw < kw; ++iw)
            {
                FLOAT_ACCUM input_val =
                    CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({n, c, ih, iw})]);
                if(m < input_val)
                {
                    m  = input_val;
                    mi = ih * W + iw;
                }
            }
        }
        output[output_tv.get_tensor_view_idx({n, c, oh, ow})]   = CVT_ACCUM2FLOAT(m);
        indices[indices_tv.get_tensor_view_idx({n, c, oh, ow})] = mi;
    }
}

extern "C" __global__ void AdaptiveMaxPoolForward2d(const D_TYPE* __restrict__ input,
                                                    D_TYPE* __restrict__ output,
                                                    int64_t* __restrict__ indices,
                                                    uint64_t N,
                                                    uint64_t C,
                                                    uint64_t H,
                                                    uint64_t W,
                                                    uint64_t OH,
                                                    uint64_t OW,
                                                    tensor_view_t<4> input_tv,
                                                    tensor_view_t<4> output_tv,
                                                    tensor_view_t<4> indices_tv)
{
    adaptiveMaxPoolForward2d<D_TYPE>(
        input, output, indices, N, C, H, W, OH, OW, input_tv, output_tv, indices_tv);
}

template <typename T>
__device__ void adaptiveMaxPoolBackward2d(const int64_t* __restrict__ indices,
                                          const T* __restrict__ output_grad,
                                          T* __restrict__ input_grad,
                                          uint64_t N,
                                          uint64_t C,
                                          uint64_t H,
                                          uint64_t W,
                                          uint64_t OH,
                                          uint64_t OW,
                                          tensor_view_t<4> indices_tv,
                                          tensor_view_t<4> output_grad_tv,
                                          tensor_view_t<4> input_grad_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t nch = gid / W, w = gid % W;
    uint64_t nc = nch / H, h = nch % H;
    uint64_t n = nc / C, c = nc % C;

    if(n >= N)
        return;

    uint64_t oh  = (h * OH) / H;
    uint64_t koh = ((h + 1) * OH + H - 1) / H - oh;

    uint64_t ow  = (w * OW) / W;
    uint64_t kow = ((w + 1) * OW + W - 1) / W - ow;

    FLOAT_ACCUM grad = 0;
    for(uint64_t ih = oh; ih < (oh + koh); ++ih)
    {
        for(uint64_t iw = ow; iw < (ow + kow); ++iw)
        {
            if(indices[indices_tv.get_tensor_view_idx({n, c, ih, iw})] == h * W + w)
            {
                grad += CVT_FLOAT2ACCUM(
                    output_grad[output_grad_tv.get_tensor_view_idx({n, c, ih, iw})]);
            }
        }
    }

    input_grad[input_grad_tv.get_tensor_view_idx({n, c, h, w})] = CVT_ACCUM2FLOAT(grad);
}

extern "C" __global__ void AdaptiveMaxPoolBackward2d(const int64_t* __restrict__ indices,
                                                     const D_TYPE* __restrict__ output_grad,
                                                     D_TYPE* __restrict__ input_grad,
                                                     uint64_t N,
                                                     uint64_t C,
                                                     uint64_t H,
                                                     uint64_t W,
                                                     uint64_t OH,
                                                     uint64_t OW,
                                                     tensor_view_t<4> indices_tv,
                                                     tensor_view_t<4> output_grad_tv,
                                                     tensor_view_t<4> input_grad_tv)
{
    adaptiveMaxPoolBackward2d<D_TYPE>(indices,
                                      output_grad,
                                      input_grad,
                                      N,
                                      C,
                                      H,
                                      W,
                                      OH,
                                      OW,
                                      indices_tv,
                                      output_grad_tv,
                                      input_grad_tv);
}

template <typename T>
__device__ void adaptiveMaxPoolForward3d(const T* __restrict__ input,
                                         T* __restrict__ output,
                                         int64_t* __restrict__ indices,
                                         uint64_t N,
                                         uint64_t C,
                                         uint64_t D,
                                         uint64_t H,
                                         uint64_t W,
                                         uint64_t OD,
                                         uint64_t OH,
                                         uint64_t OW,
                                         tensor_view_t<5> input_tv,
                                         tensor_view_t<5> output_tv,
                                         tensor_view_t<5> indices_tv)
{
    uint64_t gid    = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t ncodoh = gid / OW, ow = gid % OW;
    uint64_t ncod = ncodoh / OH, oh = ncodoh % OH;
    uint64_t nc = ncod / OD, od = ncod % OD;
    uint64_t n = nc / C, c = nc % C;

    if(n >= N)
        return;
    uint64_t d  = (od * D) / OD;
    uint64_t kd = ((od + 1) * D + OD - 1) / OD;

    uint64_t h  = (oh * H) / OH;
    uint64_t kh = ((oh + 1) * H + OH - 1) / OH;

    uint64_t w  = (ow * W) / OW;
    uint64_t kw = ((ow + 1) * W + OW - 1) / OW;

    FLOAT_ACCUM m = -INFINITY;
    if(!indices)
    {
        for(uint64_t id = d; id < kd; ++id)
        {
            for(uint64_t ih = h; ih < kh; ++ih)
            {
                for(uint64_t iw = w; iw < kw; ++iw)
                {
                    m = max(
                        m,
                        CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({n, c, id, ih, iw})]));
                }
            }
        }
        output[output_tv.get_tensor_view_idx({n, c, od, oh, ow})] = CVT_ACCUM2FLOAT(m);
    }
    else
    {
        uint64_t mi = 0;
        for(uint64_t id = d; id < kd; ++id)
        {
            for(uint64_t ih = h; ih < kh; ++ih)
            {
                for(uint64_t iw = w; iw < kw; ++iw)
                {
                    FLOAT_ACCUM input_val =
                        CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({n, c, id, ih, iw})]);
                    if(m < input_val)
                    {
                        m  = input_val;
                        mi = (id * H + ih) * W + iw;
                    }
                }
            }
        }
        output[output_tv.get_tensor_view_idx({n, c, od, oh, ow})]   = CVT_ACCUM2FLOAT(m);
        indices[indices_tv.get_tensor_view_idx({n, c, od, oh, ow})] = mi;
    }
}

extern "C" __global__ void AdaptiveMaxPoolForward3d(const D_TYPE* __restrict__ input,
                                                    D_TYPE* __restrict__ output,
                                                    int64_t* __restrict__ indices,
                                                    uint64_t N,
                                                    uint64_t C,
                                                    uint64_t D,
                                                    uint64_t H,
                                                    uint64_t W,
                                                    uint64_t OD,
                                                    uint64_t OH,
                                                    uint64_t OW,
                                                    tensor_view_t<5> input_tv,
                                                    tensor_view_t<5> output_tv,
                                                    tensor_view_t<5> indices_tv)
{
    adaptiveMaxPoolForward3d<D_TYPE>(
        input, output, indices, N, C, D, H, W, OD, OH, OW, input_tv, output_tv, indices_tv);
}

template <typename T>
__device__ void adaptiveMaxPoolBackward3d(const int64_t* __restrict__ indices,
                                          const T* __restrict__ output_grad,
                                          T* __restrict__ input_grad,
                                          uint64_t N,
                                          uint64_t C,
                                          uint64_t D,
                                          uint64_t H,
                                          uint64_t W,
                                          uint64_t OD,
                                          uint64_t OH,
                                          uint64_t OW,
                                          tensor_view_t<5> indices_tv,
                                          tensor_view_t<5> output_grad_tv,
                                          tensor_view_t<5> input_grad_tv)
{
    uint64_t gid  = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t ncdh = gid / W, w = gid % W;
    uint64_t ncd = ncdh / H, h = ncdh % H;
    uint64_t nc = ncd / D, d = ncd % D;
    uint64_t n = nc / C, c = nc % C;

    if(n >= N)
        return;

    uint64_t od  = (d * OD) / D;
    uint64_t kod = ((d + 1) * OD + D - 1) / D - od;

    uint64_t oh  = (h * OH) / H;
    uint64_t koh = ((h + 1) * OH + H - 1) / H - oh;

    uint64_t ow  = (w * OW) / W;
    uint64_t kow = ((w + 1) * OW + W - 1) / W - ow;

    FLOAT_ACCUM grad = 0;
    for(uint64_t id = od; id < (od + kod); ++id)
    {
        for(uint64_t ih = oh; ih < (oh + koh); ++ih)
        {
            for(uint64_t iw = ow; iw < (ow + kow); ++iw)
            {
                if(indices[indices_tv.get_tensor_view_idx({n, c, id, ih, iw})] ==
                   (d * H + h) * W + w)
                {
                    grad += CVT_FLOAT2ACCUM(
                        output_grad[output_grad_tv.get_tensor_view_idx({n, c, id, ih, iw})]);
                }
            }
        }
    }

    input_grad[input_grad_tv.get_tensor_view_idx({n, c, d, h, w})] = CVT_ACCUM2FLOAT(grad);
}

extern "C" __global__ void AdaptiveMaxPoolBackward3d(const int64_t* __restrict__ indices,
                                                     const D_TYPE* __restrict__ output_grad,
                                                     D_TYPE* __restrict__ input_grad,
                                                     uint64_t N,
                                                     uint64_t C,
                                                     uint64_t D,
                                                     uint64_t H,
                                                     uint64_t W,
                                                     uint64_t OD,
                                                     uint64_t OH,
                                                     uint64_t OW,
                                                     tensor_view_t<5> indices_tv,
                                                     tensor_view_t<5> output_grad_tv,
                                                     tensor_view_t<5> input_grad_tv)
{
    adaptiveMaxPoolBackward3d<D_TYPE>(indices,
                                      output_grad,
                                      input_grad,
                                      N,
                                      C,
                                      D,
                                      H,
                                      W,
                                      OD,
                                      OH,
                                      OW,
                                      indices_tv,
                                      output_grad_tv,
                                      input_grad_tv);
}
