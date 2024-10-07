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

template <typename TI, typename TO>
__device__ void adaptiveAvgPoolForward1d(const TI* __restrict__ input,
                                         TO* __restrict__ output,
                                         uint64_t N,
                                         uint64_t C,
                                         uint64_t H,
                                         uint64_t OH,
                                         tensor_view_t<3> input_tv,
                                         tensor_view_t<3> output_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t nc = gid / OH, oh = gid % OH;
    uint64_t n = nc / C, c = nc % C;
    if(n >= N)
        return;

    uint64_t h  = static_cast<uint64_t>(floor(static_cast<FLOAT_ACCUM>(oh * H) / OH));
    uint64_t kh = static_cast<uint64_t>(ceil(static_cast<FLOAT_ACCUM>((oh + 1) * H) / OH)) - h;

    FLOAT_ACCUM sum = 0;
    for(uint64_t ih = h; ih < (h + kh); ++ih)
    {
        sum += CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({n, c, ih})]);
    }
    output[output_tv.get_tensor_view_idx({n, c, oh})] = CVT_ACCUM2FLOAT(sum / kh);
}
extern "C" __global__ void AdaptiveAvgPoolForward1d(const INPUT_TYPE* __restrict__ input,
                                                    OUTPUT_TYPE* __restrict__ output,
                                                    uint64_t N,
                                                    uint64_t C,
                                                    uint64_t H,
                                                    uint64_t OH,
                                                    tensor_view_t<3> input_tv,
                                                    tensor_view_t<3> output_tv)
{
    adaptiveAvgPoolForward1d<INPUT_TYPE, OUTPUT_TYPE>(
        input, output, N, C, H, OH, input_tv, output_tv);
}

template <typename TI, typename TO>
__device__ void adaptiveAvgPoolBackward1d(const TI* __restrict__ output_grad,
                                          TO* __restrict__ input_grad,
                                          uint64_t N,
                                          uint64_t C,
                                          uint64_t H,
                                          uint64_t OH,
                                          tensor_view_t<3> output_grad_tv,
                                          tensor_view_t<3> input_grad_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t nc = gid / H, h = gid % H;
    uint64_t n = nc / C, c = nc % C;
    if(n >= N)
        return;

    uint64_t oh  = static_cast<uint64_t>(floor(static_cast<FLOAT_ACCUM>(h * OH) / H));
    uint64_t koh = static_cast<uint64_t>(ceil(static_cast<FLOAT_ACCUM>((h + 1) * OH) / H)) - oh;

    FLOAT_ACCUM grad = 0;
    for(uint64_t ih = oh; ih < (oh + koh); ++ih)
    {
        uint64_t kh = static_cast<uint64_t>(ceil(static_cast<FLOAT_ACCUM>((ih + 1) * H) / OH)) -
                      static_cast<uint64_t>(floor(static_cast<FLOAT_ACCUM>(ih * H) / OH));
        grad += CVT_FLOAT2ACCUM(output_grad[output_grad_tv.get_tensor_view_idx({n, c, ih})]) / kh;
    }
    input_grad[input_grad_tv.get_tensor_view_idx({n, c, h})] = CVT_ACCUM2FLOAT(grad);
}

extern "C" __global__ void AdaptiveAvgPoolBackward1d(const INPUT_TYPE* __restrict__ output_grad,
                                                     OUTPUT_TYPE* __restrict__ input_grad,
                                                     uint64_t N,
                                                     uint64_t C,
                                                     uint64_t H,
                                                     uint64_t OH,
                                                     tensor_view_t<3> output_grad_tv,
                                                     tensor_view_t<3> input_grad_tv)
{
    adaptiveAvgPoolBackward1d<INPUT_TYPE, OUTPUT_TYPE>(
        output_grad, input_grad, N, C, H, OH, output_grad_tv, input_grad_tv);
}

template <typename TI, typename TO>
__device__ void adaptiveAvgPoolForward2d(const TI* __restrict__ input,
                                         TO* __restrict__ output,
                                         uint64_t N,
                                         uint64_t C,
                                         uint64_t H,
                                         uint64_t W,
                                         uint64_t OH,
                                         uint64_t OW,
                                         tensor_view_t<4> input_tv,
                                         tensor_view_t<4> output_tv)
{
    uint64_t gid  = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t ncoh = gid / OW, ow = gid % OW;
    uint64_t nc = ncoh / OH, oh = ncoh % OH;
    uint64_t n = nc / C, c = nc % C;

    if(n >= N)
        return;

    uint64_t h  = static_cast<uint64_t>(floor(static_cast<FLOAT_ACCUM>(oh * H) / OH));
    uint64_t kh = static_cast<uint64_t>(ceil(static_cast<FLOAT_ACCUM>((oh + 1) * H) / OH)) - h;

    uint64_t w  = static_cast<uint64_t>(floor(static_cast<FLOAT_ACCUM>(ow * W) / OW));
    uint64_t kw = static_cast<uint64_t>(ceil(static_cast<FLOAT_ACCUM>((ow + 1) * W) / OW)) - w;

    FLOAT_ACCUM divider = static_cast<FLOAT_ACCUM>(kh * kw);
    FLOAT_ACCUM sum     = 0;
    for(uint64_t ih = h; ih < (h + kh); ++ih)
    {
        for(uint64_t iw = w; iw < (w + kw); ++iw)
        {
            sum += CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({n, c, ih, iw})]);
        }
    }
    output[output_tv.get_tensor_view_idx({n, c, oh, ow})] = CVT_ACCUM2FLOAT(sum / divider);
}

extern "C" __global__ void AdaptiveAvgPoolForward2d(const INPUT_TYPE* __restrict__ input,
                                                    OUTPUT_TYPE* __restrict__ output,
                                                    uint64_t N,
                                                    uint64_t C,
                                                    uint64_t H,
                                                    uint64_t W,
                                                    uint64_t OH,
                                                    uint64_t OW,
                                                    tensor_view_t<4> input_tv,
                                                    tensor_view_t<4> output_tv)
{
    adaptiveAvgPoolForward2d<INPUT_TYPE, OUTPUT_TYPE>(
        input, output, N, C, H, W, OH, OW, input_tv, output_tv);
}

template <typename TI, typename TO>
__device__ void adaptiveAvgPoolBackward2d(const TI* __restrict__ output_grad,
                                          TO* __restrict__ input_grad,
                                          uint64_t N,
                                          uint64_t C,
                                          uint64_t H,
                                          uint64_t W,
                                          uint64_t OH,
                                          uint64_t OW,
                                          tensor_view_t<4> output_grad_tv,
                                          tensor_view_t<4> input_grad_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t nch = gid / W, w = gid % W;
    uint64_t nc = nch / H, h = nch % H;
    uint64_t n = nc / C, c = nc % C;

    if(n >= N)
        return;

    uint64_t oh  = static_cast<uint64_t>(floor(static_cast<FLOAT_ACCUM>(h * OH) / H));
    uint64_t koh = static_cast<uint64_t>(ceil(static_cast<FLOAT_ACCUM>((h + 1) * OH) / H)) - oh;

    uint64_t ow  = static_cast<uint64_t>(floor(static_cast<FLOAT_ACCUM>(w * OW) / W));
    uint64_t kow = static_cast<uint64_t>(ceil(static_cast<FLOAT_ACCUM>((w + 1) * OW) / W)) - ow;

    FLOAT_ACCUM grad = 0;
    for(uint64_t ih = oh; ih < (oh + koh); ++ih)
    {
        uint64_t kh = static_cast<uint64_t>(ceil(static_cast<FLOAT_ACCUM>((ih + 1) * H) / OH)) -
                      static_cast<uint64_t>(floor(static_cast<FLOAT_ACCUM>(ih * H) / OH));
        for(uint64_t iw = ow; iw < (ow + kow); ++iw)
        {
            uint64_t kw = static_cast<uint64_t>(ceil(static_cast<FLOAT_ACCUM>((iw + 1) * W) / OW)) -
                          static_cast<uint64_t>(floor(static_cast<FLOAT_ACCUM>(iw * W) / OW));
            grad +=
                CVT_FLOAT2ACCUM(output_grad[output_grad_tv.get_tensor_view_idx({n, c, ih, iw})]) /
                (kh * kw);
        }
    }

    input_grad[input_grad_tv.get_tensor_view_idx({n, c, h, w})] = CVT_ACCUM2FLOAT(grad);
}

extern "C" __global__ void AdaptiveAvgPoolBackward2d(const INPUT_TYPE* __restrict__ output_grad,
                                                     OUTPUT_TYPE* __restrict__ input_grad,
                                                     uint64_t N,
                                                     uint64_t C,
                                                     uint64_t H,
                                                     uint64_t W,
                                                     uint64_t OH,
                                                     uint64_t OW,
                                                     tensor_view_t<4> output_grad_tv,
                                                     tensor_view_t<4> input_grad_tv)
{
    adaptiveAvgPoolBackward2d<INPUT_TYPE, OUTPUT_TYPE>(
        output_grad, input_grad, N, C, H, W, OH, OW, output_grad_tv, input_grad_tv);
}

template <typename TI, typename TO>
__device__ void adaptiveAvgPoolForward3d(const TI* __restrict__ input,
                                         TO* __restrict__ output,
                                         uint64_t N,
                                         uint64_t C,
                                         uint64_t D,
                                         uint64_t H,
                                         uint64_t W,
                                         uint64_t OD,
                                         uint64_t OH,
                                         uint64_t OW,
                                         tensor_view_t<5> input_tv,
                                         tensor_view_t<5> output_tv)
{
    uint64_t gid    = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t ncodoh = gid / OW, ow = gid % OW;
    uint64_t ncod = ncodoh / OH, oh = ncodoh % OH;
    uint64_t nc = ncod / OD, od = ncod % OD;
    uint64_t n = nc / C, c = nc % C;

    if(n >= N)
        return;
    uint64_t d  = static_cast<uint64_t>(floor(static_cast<FLOAT_ACCUM>(od * D) / OD));
    uint64_t kd = static_cast<uint64_t>(ceil(static_cast<FLOAT_ACCUM>((od + 1) * D) / OD)) - d;

    uint64_t h  = static_cast<uint64_t>(floor(static_cast<FLOAT_ACCUM>(oh * H) / OH));
    uint64_t kh = static_cast<uint64_t>(ceil(static_cast<FLOAT_ACCUM>((oh + 1) * H) / OH)) - h;

    uint64_t w  = static_cast<uint64_t>(floor(static_cast<FLOAT_ACCUM>(ow * W) / OW));
    uint64_t kw = static_cast<uint64_t>(ceil(static_cast<FLOAT_ACCUM>((ow + 1) * W) / OW)) - w;

    FLOAT_ACCUM sum = 0;
    for(uint64_t id = d; id < (d + kd); ++id)
    {
        for(uint64_t ih = h; ih < (h + kh); ++ih)
        {
            for(uint64_t iw = w; iw < (w + kw); ++iw)
            {
                sum += CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({n, c, id, ih, iw})]);
            }
        }
    }

    output[output_tv.get_tensor_view_idx({n, c, od, oh, ow})] =
        CVT_ACCUM2FLOAT(sum / (kd * kh * kw));
}

extern "C" __global__ void AdaptiveAvgPoolForward3d(const INPUT_TYPE* __restrict__ input,
                                                    OUTPUT_TYPE* __restrict__ output,
                                                    uint64_t N,
                                                    uint64_t C,
                                                    uint64_t D,
                                                    uint64_t H,
                                                    uint64_t W,
                                                    uint64_t OD,
                                                    uint64_t OH,
                                                    uint64_t OW,
                                                    tensor_view_t<5> input_tv,
                                                    tensor_view_t<5> output_tv)
{
    adaptiveAvgPoolForward3d<INPUT_TYPE, OUTPUT_TYPE>(
        input, output, N, C, D, H, W, OD, OH, OW, input_tv, output_tv);
}

template <typename TI, typename TO>
__device__ void adaptiveAvgPoolBackward3d(const TI* __restrict__ output_grad,
                                          TO* __restrict__ input_grad,
                                          uint64_t N,
                                          uint64_t C,
                                          uint64_t D,
                                          uint64_t H,
                                          uint64_t W,
                                          uint64_t OD,
                                          uint64_t OH,
                                          uint64_t OW,
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

    uint64_t od  = static_cast<uint64_t>(floor(static_cast<FLOAT_ACCUM>(d * OD) / D));
    uint64_t kod = static_cast<uint64_t>(ceil(static_cast<FLOAT_ACCUM>((d + 1) * OD) / D)) - od;

    uint64_t oh  = static_cast<uint64_t>(floor(static_cast<FLOAT_ACCUM>(h * OH) / H));
    uint64_t koh = static_cast<uint64_t>(ceil(static_cast<FLOAT_ACCUM>((h + 1) * OH) / H)) - oh;

    uint64_t ow  = static_cast<uint64_t>(floor(static_cast<FLOAT_ACCUM>(w * OW) / W));
    uint64_t kow = static_cast<uint64_t>(ceil(static_cast<FLOAT_ACCUM>((w + 1) * OW) / W)) - ow;

    FLOAT_ACCUM grad = 0;
    for(uint64_t id = od; id < (od + kod); ++id)
    {
        uint64_t kd = static_cast<uint64_t>(ceil(static_cast<FLOAT_ACCUM>((id + 1) * D) / OD)) -
                      static_cast<uint64_t>(floor(static_cast<FLOAT_ACCUM>(id * D) / OD));
        for(uint64_t ih = oh; ih < (oh + koh); ++ih)
        {
            uint64_t kh = static_cast<uint64_t>(ceil(static_cast<FLOAT_ACCUM>((ih + 1) * H) / OH)) -
                          static_cast<uint64_t>(floor(static_cast<FLOAT_ACCUM>(ih * H) / OH));
            for(uint64_t iw = ow; iw < (ow + kow); ++iw)
            {
                uint64_t kw =
                    static_cast<uint64_t>(ceil(static_cast<FLOAT_ACCUM>((iw + 1) * W) / OW)) -
                    static_cast<uint64_t>(floor(static_cast<FLOAT_ACCUM>(iw * W) / OW));
                grad += CVT_FLOAT2ACCUM(
                            output_grad[output_grad_tv.get_tensor_view_idx({n, c, id, ih, iw})]) /
                        (kd * kh * kw);
            }
        }
    }

    input_grad[input_grad_tv.get_tensor_view_idx({n, c, d, h, w})] = CVT_ACCUM2FLOAT(grad);
}

extern "C" __global__ void AdaptiveAvgPoolBackward3d(const INPUT_TYPE* __restrict__ output_grad,
                                                     OUTPUT_TYPE* __restrict__ input_grad,
                                                     uint64_t N,
                                                     uint64_t C,
                                                     uint64_t D,
                                                     uint64_t H,
                                                     uint64_t W,
                                                     uint64_t OD,
                                                     uint64_t OH,
                                                     uint64_t OW,
                                                     tensor_view_t<5> output_grad_tv,
                                                     tensor_view_t<5> input_grad_tv)
{
    adaptiveAvgPoolBackward3d<INPUT_TYPE, OUTPUT_TYPE>(
        output_grad, input_grad, N, C, D, H, W, OD, OH, OW, output_grad_tv, input_grad_tv);
}
