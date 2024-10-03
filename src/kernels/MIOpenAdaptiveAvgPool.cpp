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

#ifndef INPUT_TYPE
#define INPUT_TYPE float
#endif

#ifndef OUTPUT_TYPE
#define OUTPUT_TYPE float
#endif

template <typename TI, typename TO>
__device__ void adaptiveAvgPoolForward1d(const TI* __restrict__ input,
                                         TO* __restrict__ output,
                                         size_t N,
                                         size_t C,
                                         size_t H,
                                         size_t OH,
                                         tensor_view_t<3> input_tv,
                                         tensor_view_t<3> output_tv)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t nc = gid / OH, oh = gid % OH;
    size_t n = nc / C, c = nc % C;
    if(n >= N)
        return;

    size_t h  = static_cast<size_t>(floor(static_cast<FLOAT_ACCUM>(oh * H) / OH));
    size_t kh = static_cast<size_t>(ceil(static_cast<FLOAT_ACCUM>((oh + 1) * H) / OH)) - h;

    FLOAT_ACCUM sum = 0;
    for(size_t ih = h; ih < (h + kh); ++ih)
    {
        sum += CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({n, c, ih})]);
    }
    output[output_tv.get_tensor_view_idx({n, c, oh})] = CVT_ACCUM2FLOAT(sum / kh);
}
extern "C" __global__ void AdaptiveAvgPoolForward1d(const INPUT_TYPE* __restrict__ input,
                                                    OUTPUT_TYPE* __restrict__ output,
                                                    size_t N,
                                                    size_t C,
                                                    size_t H,
                                                    size_t OH,
                                                    tensor_view_t<3> input_tv,
                                                    tensor_view_t<3> output_tv)
{
    adaptiveAvgPoolForward1d<INPUT_TYPE, OUTPUT_TYPE>(
        input, output, N, C, H, OH, input_tv, output_tv);
}

template <typename TI, typename TO>
__device__ void adaptiveAvgPoolBackward1d(const TI* __restrict__ output_grad,
                                          TO* __restrict__ input_grad,
                                          size_t N,
                                          size_t C,
                                          size_t H,
                                          size_t OH,
                                          tensor_view_t<3> output_grad_tv,
                                          tensor_view_t<3> input_grad_tv)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t nc = gid / H, h = gid % H;
    size_t n = nc / C, c = nc % C;
    if(n >= N)
        return;

    size_t oh  = static_cast<size_t>(floor(static_cast<FLOAT_ACCUM>(h * OH) / H));
    size_t koh = static_cast<size_t>(ceil(static_cast<FLOAT_ACCUM>((h + 1) * OH) / H)) - oh;

    FLOAT_ACCUM grad = 0;
    for(size_t ih = oh; ih < (oh + koh); ++ih)
    {
        size_t kh = static_cast<size_t>(ceil(static_cast<FLOAT_ACCUM>((ih + 1) * H) / OH)) -
                    static_cast<size_t>(floor(static_cast<FLOAT_ACCUM>(ih * H) / OH));
        grad += CVT_FLOAT2ACCUM(output_grad[output_grad_tv.get_tensor_view_idx({n, c, ih})]) / kh;
    }
    input_grad[input_grad_tv.get_tensor_view_idx({n, c, h})] = CVT_ACCUM2FLOAT(grad);
}

extern "C" __global__ void AdaptiveAvgPoolBackward1d(const INPUT_TYPE* __restrict__ output_grad,
                                                     OUTPUT_TYPE* __restrict__ input_grad,
                                                     size_t N,
                                                     size_t C,
                                                     size_t H,
                                                     size_t OH,
                                                     tensor_view_t<3> output_grad_tv,
                                                     tensor_view_t<3> input_grad_tv)
{
    adaptiveAvgPoolBackward1d<INPUT_TYPE, OUTPUT_TYPE>(
        output_grad, input_grad, N, C, H, OH, output_grad_tv, input_grad_tv);
}

template <typename TI, typename TO>
__device__ void adaptiveAvgPoolForward2d(const TI* __restrict__ input,
                                         TO* __restrict__ output,
                                         size_t N,
                                         size_t C,
                                         size_t H,
                                         size_t W,
                                         size_t OH,
                                         size_t OW,
                                         tensor_view_t<4> input_tv,
                                         tensor_view_t<4> output_tv)
{
    size_t gid  = threadIdx.x + blockIdx.x * blockDim.x;
    size_t ncoh = gid / OW, ow = gid % OW;
    size_t nc = ncoh / OH, oh = ncoh % OH;
    size_t n = nc / C, c = nc % C;

    if(n >= N)
        return;

    size_t h  = static_cast<size_t>(floor(static_cast<FLOAT_ACCUM>(oh * H) / OH));
    size_t kh = static_cast<size_t>(ceil(static_cast<FLOAT_ACCUM>((oh + 1) * H) / OH)) - h;

    size_t w  = static_cast<size_t>(floor(static_cast<FLOAT_ACCUM>(ow * W) / OW));
    size_t kw = static_cast<size_t>(ceil(static_cast<FLOAT_ACCUM>((ow + 1) * W) / OW)) - w;

    FLOAT_ACCUM divider = static_cast<FLOAT_ACCUM>(kh * kw);
    FLOAT_ACCUM sum     = 0;
    for(size_t ih = h; ih < (h + kh); ++ih)
    {
        for(size_t iw = w; iw < (w + kw); ++iw)
        {
            sum += CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({n, c, ih, iw})]);
        }
    }
    output[output_tv.get_tensor_view_idx({n, c, oh, ow})] = CVT_ACCUM2FLOAT(sum / divider);
}

extern "C" __global__ void AdaptiveAvgPoolForward2d(const INPUT_TYPE* __restrict__ input,
                                                    OUTPUT_TYPE* __restrict__ output,
                                                    size_t N,
                                                    size_t C,
                                                    size_t H,
                                                    size_t W,
                                                    size_t OH,
                                                    size_t OW,
                                                    tensor_view_t<4> input_tv,
                                                    tensor_view_t<4> output_tv)
{
    adaptiveAvgPoolForward2d<INPUT_TYPE, OUTPUT_TYPE>(
        input, output, N, C, H, W, OH, OW, input_tv, output_tv);
}

template <typename TI, typename TO>
__device__ void adaptiveAvgPoolBackward2d(const TI* __restrict__ output_grad,
                                          TO* __restrict__ input_grad,
                                          size_t N,
                                          size_t C,
                                          size_t H,
                                          size_t W,
                                          size_t OH,
                                          size_t OW,
                                          tensor_view_t<4> output_grad_tv,
                                          tensor_view_t<4> input_grad_tv)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t nch = gid / W, w = gid % W;
    size_t nc = nch / H, h = nch % H;
    size_t n = nc / C, c = nc % C;

    if(n >= N)
        return;

    size_t oh  = static_cast<size_t>(floor(static_cast<FLOAT_ACCUM>(h * OH) / H));
    size_t koh = static_cast<size_t>(ceil(static_cast<FLOAT_ACCUM>((h + 1) * OH) / H)) - oh;

    size_t ow  = static_cast<size_t>(floor(static_cast<FLOAT_ACCUM>(w * OW) / W));
    size_t kow = static_cast<size_t>(ceil(static_cast<FLOAT_ACCUM>((w + 1) * OW) / W)) - ow;

    FLOAT_ACCUM grad = 0;
    for(size_t ih = oh; ih < (oh + koh); ++ih)
    {
        size_t kh = static_cast<size_t>(ceil(static_cast<FLOAT_ACCUM>((ih + 1) * H) / OH)) -
                    static_cast<size_t>(floor(static_cast<FLOAT_ACCUM>(ih * H) / OH));
        for(size_t iw = ow; iw < (ow + kow); ++iw)
        {
            size_t kw = static_cast<size_t>(ceil(static_cast<FLOAT_ACCUM>((iw + 1) * W) / OW)) -
                        static_cast<size_t>(floor(static_cast<FLOAT_ACCUM>(iw * W) / OW));
            grad +=
                CVT_FLOAT2ACCUM(output_grad[output_grad_tv.get_tensor_view_idx({n, c, ih, iw})]) /
                (kh * kw);
        }
    }

    input_grad[input_grad_tv.get_tensor_view_idx({n, c, h, w})] = CVT_ACCUM2FLOAT(grad);
}

extern "C" __global__ void AdaptiveAvgPoolBackward2d(const INPUT_TYPE* __restrict__ output_grad,
                                                     OUTPUT_TYPE* __restrict__ input_grad,
                                                     size_t N,
                                                     size_t C,
                                                     size_t H,
                                                     size_t W,
                                                     size_t OH,
                                                     size_t OW,
                                                     tensor_view_t<4> output_grad_tv,
                                                     tensor_view_t<4> input_grad_tv)
{
    adaptiveAvgPoolBackward2d<INPUT_TYPE, OUTPUT_TYPE>(
        output_grad, input_grad, N, C, H, W, OH, OW, output_grad_tv, input_grad_tv);
}

template <typename TI, typename TO>
__device__ void adaptiveAvgPoolForward3d(const TI* __restrict__ input,
                                         TO* __restrict__ output,
                                         size_t N,
                                         size_t C,
                                         size_t D,
                                         size_t H,
                                         size_t W,
                                         size_t OD,
                                         size_t OH,
                                         size_t OW,
                                         tensor_view_t<5> input_tv,
                                         tensor_view_t<5> output_tv)
{
    size_t gid    = threadIdx.x + blockIdx.x * blockDim.x;
    size_t ncodoh = gid / OW, ow = gid % OW;
    size_t ncod = ncodoh / OH, oh = ncodoh % OH;
    size_t nc = ncod / OD, od = ncod % OD;
    size_t n = nc / C, c = nc % C;

    if(n >= N)
        return;
    size_t d  = static_cast<size_t>(floor(static_cast<FLOAT_ACCUM>(od * D) / OD));
    size_t kd = static_cast<size_t>(ceil(static_cast<FLOAT_ACCUM>((od + 1) * D) / OD)) - d;

    size_t h  = static_cast<size_t>(floor(static_cast<FLOAT_ACCUM>(oh * H) / OH));
    size_t kh = static_cast<size_t>(ceil(static_cast<FLOAT_ACCUM>((oh + 1) * H) / OH)) - h;

    size_t w  = static_cast<size_t>(floor(static_cast<FLOAT_ACCUM>(ow * W) / OW));
    size_t kw = static_cast<size_t>(ceil(static_cast<FLOAT_ACCUM>((ow + 1) * W) / OW)) - w;

    FLOAT_ACCUM sum = 0;
    for(size_t id = d; id < (d + kd); ++id)
    {
        for(size_t ih = h; ih < (h + kh); ++ih)
        {
            for(size_t iw = w; iw < (w + kw); ++iw)
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
                                                    size_t N,
                                                    size_t C,
                                                    size_t D,
                                                    size_t H,
                                                    size_t W,
                                                    size_t OD,
                                                    size_t OH,
                                                    size_t OW,
                                                    tensor_view_t<5> input_tv,
                                                    tensor_view_t<5> output_tv)
{
    adaptiveAvgPoolForward3d<INPUT_TYPE, OUTPUT_TYPE>(
        input, output, N, C, D, H, W, OD, OH, OW, input_tv, output_tv);
}

template <typename TI, typename TO>
__device__ void adaptiveAvgPoolBackward3d(const TI* __restrict__ output_grad,
                                          TO* __restrict__ input_grad,
                                          size_t N,
                                          size_t C,
                                          size_t D,
                                          size_t H,
                                          size_t W,
                                          size_t OD,
                                          size_t OH,
                                          size_t OW,
                                          tensor_view_t<5> output_grad_tv,
                                          tensor_view_t<5> input_grad_tv)
{
    size_t gid  = threadIdx.x + blockIdx.x * blockDim.x;
    size_t ncdh = gid / W, w = gid % W;
    size_t ncd = ncdh / H, h = ncdh % H;
    size_t nc = ncd / D, d = ncd % D;
    size_t n = nc / C, c = nc % C;

    if(n >= N)
        return;

    size_t od  = static_cast<size_t>(floor(static_cast<FLOAT_ACCUM>(d * OD) / D));
    size_t kod = static_cast<size_t>(ceil(static_cast<FLOAT_ACCUM>((d + 1) * OD) / D)) - od;

    size_t oh  = static_cast<size_t>(floor(static_cast<FLOAT_ACCUM>(h * OH) / H));
    size_t koh = static_cast<size_t>(ceil(static_cast<FLOAT_ACCUM>((h + 1) * OH) / H)) - oh;

    size_t ow  = static_cast<size_t>(floor(static_cast<FLOAT_ACCUM>(w * OW) / W));
    size_t kow = static_cast<size_t>(ceil(static_cast<FLOAT_ACCUM>((w + 1) * OW) / W)) - ow;

    FLOAT_ACCUM grad = 0;
    for(size_t id = od; id < (od + kod); ++id)
    {
        size_t kd = static_cast<size_t>(ceil(static_cast<FLOAT_ACCUM>((id + 1) * D) / OD)) -
                    static_cast<size_t>(floor(static_cast<FLOAT_ACCUM>(id * D) / OD));
        for(size_t ih = oh; ih < (oh + koh); ++ih)
        {
            size_t kh = static_cast<size_t>(ceil(static_cast<FLOAT_ACCUM>((ih + 1) * H) / OH)) -
                        static_cast<size_t>(floor(static_cast<FLOAT_ACCUM>(ih * H) / OH));
            for(size_t iw = ow; iw < (ow + kow); ++iw)
            {
                size_t kw = static_cast<size_t>(ceil(static_cast<FLOAT_ACCUM>((iw + 1) * W) / OW)) -
                            static_cast<size_t>(floor(static_cast<FLOAT_ACCUM>(iw * W) / OW));
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
                                                     size_t N,
                                                     size_t C,
                                                     size_t D,
                                                     size_t H,
                                                     size_t W,
                                                     size_t OD,
                                                     size_t OH,
                                                     size_t OW,
                                                     tensor_view_t<5> output_grad_tv,
                                                     tensor_view_t<5> input_grad_tv)
{
    adaptiveAvgPoolBackward3d<INPUT_TYPE, OUTPUT_TYPE>(
        output_grad, input_grad, N, C, D, H, W, OD, OH, OW, output_grad_tv, input_grad_tv);
}
