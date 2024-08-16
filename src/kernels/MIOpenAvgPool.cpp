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
#include <cstddef>
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

// template <typename T, uint32_t Nd>
// struct blockNd
// {
//     T val[Nd];
// };

// template <typename TI, typename TO, uint32_t Nd>
// __device__ void avgPoolForwardNdNew(const TI* __restrict__ input,
//                                     TO* __restrict__ output,
//                                     size_t N,
//                                     size_t C,
//                                     const blockNd<size_t, Nd> sizeIn,
//                                     const blockNd<size_t, Nd> sizeOut,
//                                     const blockNd<int32_t, Nd> ksize,
//                                     const blockNd<int32_t, Nd> stride,
//                                     const blockNd<int32_t, Nd> padding,
//                                     bool count_include_pad,
//                                     int32_t divisor_override,
//                                     tensor_view_t<Nd + 2> input_tv,
//                                     tensor_view_t<Nd + 2> output_tv);

template <typename TI, typename TO>
__device__ void avgPoolForward2d(const TI* __restrict__ input,
                                 TO* __restrict__ output,
                                 size_t N,
                                 size_t C,
                                 size_t H,
                                 size_t W,
                                 size_t OH,
                                 size_t OW,
                                 const int32_t* __restrict__ ksize,
                                 const int32_t* __restrict__ stride,
                                 const int32_t* __restrict__ padding,
                                 bool count_include_pad,
                                 int32_t divisor_override,
                                 tensor_view_t<4> input_tv,
                                 tensor_view_t<4> output_tv)
{
    int32_t gid  = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t ncoh = gid / OW, ow = gid % OW;
    int32_t nc = ncoh / OH, oh = ncoh % OH;
    int32_t n = nc / C, c = nc % C;
    int32_t R  = ksize[0];
    int32_t S  = ksize[1];
    int32_t sh = stride[0];
    int32_t sw = stride[1];
    int32_t ph = padding[0];
    int32_t pw = padding[1];

    if(n >= N)
        return;

    FLOAT_ACCUM m = 0;
    for(int32_t r = 0; r < R; ++r)
    {
        for(int32_t s = 0; s < S; ++s)
        {
            // input idx : (n, c, h, w)
            int32_t h = oh * sh - ph + r;
            if(h < 0 || h >= H)
                continue;
            int32_t w = ow * sw - pw + s;
            if(w < 0 || w >= W)
                continue;
            // int32_t input_idx = ((n * C + c) * H + h) * W + w;
            m += CVT_FLOAT2ACCUM(
                input[input_tv.get_tensor_view_idx(tensor_layout_t<4>(n, c, h, w))]);
        }
    }

    int32_t hstart = oh * sh - ph;
    int32_t wstart = ow * sw - pw;
    int32_t hend   = min(hstart + R, H + ph);
    int32_t wend   = min(wstart + S, W + pw);

    const int32_t pool_size = (hend - hstart) * (wend - wstart);

    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend   = min(hend, H);
    wend   = min(wend, W);

    int32_t divide_factor;
    if(divisor_override != 0)
    {
        divide_factor = divisor_override;
    }
    else
    {
        if(count_include_pad)
        {
            divide_factor = pool_size;
        }
        else
        {
            divide_factor = (hend - hstart) * (wend - wstart);
        }
    }
    FLOAT_ACCUM val = m / divide_factor;

    output[output_tv.get_tensor_view_idx(tensor_layout_t<4>(n, c, oh, ow))] = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void AvgPoolForward2d(const INPUT_TYPE* __restrict__ input,
                                            OUTPUT_TYPE* __restrict__ output,
                                            size_t N,
                                            size_t C,
                                            size_t H,
                                            size_t W,
                                            size_t OH,
                                            size_t OW,
                                            int32_t* ksize,
                                            int32_t* stride,
                                            int32_t* padding,
                                            bool count_include_pad,
                                            int32_t divisor_override,
                                            tensor_view_t<4> input_tv,
                                            tensor_view_t<4> output_tv)
{
    avgPoolForward2d<INPUT_TYPE, OUTPUT_TYPE>(input,
                                              output,
                                              N,
                                              C,
                                              H,
                                              W,
                                              OH,
                                              OW,
                                              ksize,
                                              stride,
                                              padding,
                                              count_include_pad,
                                              divisor_override,
                                              input_tv,
                                              output_tv);
}

template <typename TI, typename TO>
__device__ void avgPoolForward3d(const TI* __restrict__ input,
                                 TO* __restrict__ output,
                                 size_t N,
                                 size_t C,
                                 size_t D,
                                 size_t H,
                                 size_t W,
                                 size_t OD,
                                 size_t OH,
                                 size_t OW,
                                 int32_t* ksize,
                                 int32_t* stride,
                                 int32_t* padding,
                                 bool count_include_pad,
                                 int32_t divisor_override,
                                 tensor_view_t<5> input_tv,
                                 tensor_view_t<5> output_tv)
{
    int32_t gid    = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t ncodoh = gid / OW, ow = gid % OW;
    int32_t ncod = ncodoh / OH, oh = ncodoh % OH;
    int32_t nc = ncod / OD, od = ncod % OD;
    int32_t n = nc / C, c = nc % C;
    int32_t KD = ksize[0];
    int32_t R  = ksize[1];
    int32_t S  = ksize[2];
    int32_t sd = stride[0];
    int32_t sh = stride[1];
    int32_t sw = stride[2];
    int32_t pd = padding[0];
    int32_t ph = padding[1];
    int32_t pw = padding[2];

    if(n >= N)
        return;
    FLOAT_ACCUM sum = 0;
    for(int32_t kd = 0; kd < KD; ++kd)
    {
        for(int32_t r = 0; r < R; ++r)
        {
            for(int32_t s = 0; s < S; ++s)
            {
                // input idx : (n, c, d, h, w)
                int32_t d = od * sd - pd + kd;
                if(d < 0 || d >= D)
                    continue;
                int32_t h = oh * sh - ph + r;
                if(h < 0 || h >= H)
                    continue;
                int32_t w = ow * sw - pw + s;
                if(w < 0 || w >= W)
                    continue;
                // int32_t input_idx = ((n * C + c) * H + h) * W + w;
                sum += CVT_FLOAT2ACCUM(
                    input[input_tv.get_tensor_view_idx(tensor_layout_t<5>(n, c, d, h, w))]);
            }
        }
    }
    int32_t dstart = od * sd - pd;
    int32_t hstart = oh * sh - ph;
    int32_t wstart = ow * sw - pw;
    int32_t dend   = min(dstart + KD, D + pd);
    int32_t hend   = min(hstart + R, H + ph);
    int32_t wend   = min(wstart + S, W + pw);

    const int32_t pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
    dstart                  = max(dstart, 0);
    hstart                  = max(hstart, 0);
    wstart                  = max(wstart, 0);
    dend                    = min(dend, D);
    hend                    = min(hend, H);
    wend                    = min(wend, W);

    int32_t divide_factor;
    if(divisor_override != 0)
    {
        divide_factor = divisor_override;
    }
    else
    {
        if(count_include_pad)
        {
            divide_factor = pool_size;
        }
        else
        {
            divide_factor = (dend - dstart) * (hend - hstart) * (wend - wstart);
        }
    }
    FLOAT_ACCUM val = sum / divide_factor;
    output[output_tv.get_tensor_view_idx(tensor_layout_t<5>(n, c, od, oh, ow))] =
        CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void AvgPoolForward3d(const INPUT_TYPE* __restrict__ input,
                                            OUTPUT_TYPE* __restrict__ output,
                                            size_t N,
                                            size_t C,
                                            size_t D,
                                            size_t H,
                                            size_t W,
                                            size_t OD,
                                            size_t OH,
                                            size_t OW,
                                            int32_t* ksize,
                                            int32_t* stride,
                                            int32_t* padding,
                                            bool count_include_pad,
                                            int32_t divisor_override,
                                            tensor_view_t<5> input_tv,
                                            tensor_view_t<5> output_tv)
{
    avgPoolForward3d<INPUT_TYPE, OUTPUT_TYPE>(input,
                                              output,
                                              N,
                                              C,
                                              D,
                                              H,
                                              W,
                                              OD,
                                              OH,
                                              OW,
                                              ksize,
                                              stride,
                                              padding,
                                              count_include_pad,
                                              divisor_override,
                                              input_tv,
                                              output_tv);
}

template <typename TI, typename TO>
__device__ void avgPoolBackward2d(const TI* __restrict__ output_grad,
                                  TO* __restrict__ input_grad,
                                  size_t N,
                                  size_t C,
                                  size_t H,
                                  size_t W,
                                  size_t OH,
                                  size_t OW,
                                  int32_t* ksize,
                                  int32_t* stride,
                                  int32_t* padding,
                                  bool count_include_pad,
                                  int32_t divisor_override,
                                  tensor_view_t<4> output_grad_tv,
                                  tensor_view_t<4> input_grad_tv)
{
    int32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t nch = gid / W, w = gid % W;
    int32_t nc = nch / H, h = nch % H;
    int32_t n = nc / C, c = nc % C;
    int32_t R  = ksize[0];
    int32_t S  = ksize[1];
    int32_t sh = stride[0];
    int32_t sw = stride[1];
    int32_t ph = padding[0];
    int32_t pw = padding[1];

    if(n >= N)
        return;

    FLOAT_ACCUM grad = 0;
    for(int32_t r = 0; r < R; ++r)
    {
        for(int32_t s = 0; s < S; ++s)
        {
            int32_t ohsh = h + ph - r;
            if(ohsh % sh != 0)
                continue;
            int32_t oh = ohsh / sh;
            if(oh < 0 || oh >= OH)
                continue;
            int32_t owsw = w + pw - s;
            if(owsw % sw != 0)
                continue;
            int32_t ow = owsw / sw;
            if(ow < 0 || ow >= OW)
                continue;

            int32_t hstart = oh * sh - ph;
            int32_t wstart = ow * sw - pw;
            int32_t hend   = min(hstart + R, H + ph);
            int32_t wend   = min(wstart + S, W + pw);

            const int32_t pool_size = (hend - hstart) * (wend - wstart);

            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend   = min(hend, H);
            wend   = min(wend, W);

            int32_t divide_factor;
            if(divisor_override != 0)
            {
                divide_factor = divisor_override;
            }
            else
            {
                if(count_include_pad)
                {
                    divide_factor = pool_size;
                }
                else
                {
                    divide_factor = (hend - hstart) * (wend - wstart);
                }
            }

            grad += CVT_FLOAT2ACCUM(output_grad[output_grad_tv.get_tensor_view_idx(
                        tensor_layout_t<4>(n, c, oh, ow))]) /
                    divide_factor;
        }
    }
    input_grad[input_grad_tv.get_tensor_view_idx(tensor_layout_t<4>(n, c, h, w))] =
        CVT_ACCUM2FLOAT(grad);
}

extern "C" __global__ void AvgPoolBackward2d(const INPUT_TYPE* __restrict__ output_grad,
                                             OUTPUT_TYPE* __restrict__ input_grad,
                                             size_t N,
                                             size_t C,
                                             size_t H,
                                             size_t W,
                                             size_t OH,
                                             size_t OW,
                                             int32_t* ksize,
                                             int32_t* stride,
                                             int32_t* padding,
                                             bool count_include_pad,
                                             int32_t divisor_override,
                                             tensor_view_t<4> output_grad_tv,
                                             tensor_view_t<4> input_grad_tv)
{
    avgPoolBackward2d<INPUT_TYPE, OUTPUT_TYPE>(output_grad,
                                               input_grad,
                                               N,
                                               C,
                                               H,
                                               W,
                                               OH,
                                               OW,
                                               ksize,
                                               stride,
                                               padding,
                                               count_include_pad,
                                               divisor_override,
                                               output_grad_tv,
                                               input_grad_tv);
}

template <typename TI, typename TO>
__device__ void avgPoolBackward3d(const TI* __restrict__ output_grad,
                                  TO* __restrict__ input_grad,
                                  size_t N,
                                  size_t C,
                                  size_t D,
                                  size_t H,
                                  size_t W,
                                  size_t OD,
                                  size_t OH,
                                  size_t OW,
                                  int32_t* ksize,
                                  int32_t* stride,
                                  int32_t* padding,
                                  bool count_include_pad,
                                  int32_t divisor_override,
                                  tensor_view_t<5> output_grad_tv,
                                  tensor_view_t<5> input_grad_tv)
{
    int32_t gid  = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t ncdh = gid / W, w = gid % W;
    int32_t ncd = ncdh / H, h = ncdh % H;
    int32_t nc = ncd / D, d = ncd % D;
    int32_t n = nc / C, c = nc % C;
    int32_t KD = ksize[0];
    int32_t R  = ksize[1];
    int32_t S  = ksize[2];
    int32_t sd = stride[0];
    int32_t sh = stride[1];
    int32_t sw = stride[2];
    int32_t pd = padding[0];
    int32_t ph = padding[1];
    int32_t pw = padding[2];

    if(n >= N)
        return;

    FLOAT_ACCUM grad = 0;
    for(int32_t kd = 0; kd < KD; ++kd)
    {
        for(int32_t r = 0; r < R; ++r)
        {
            for(int32_t s = 0; s < S; ++s)
            {
                int32_t odsd = d + pd - kd;
                if(odsd % sd != 0)
                    continue;
                int32_t od = odsd / sd;
                if(od < 0 || od >= OD)
                    continue;

                int32_t ohsh = h + ph - r;
                if(ohsh % sh != 0)
                    continue;
                int32_t oh = ohsh / sh;
                if(oh < 0 || oh >= OH)
                    continue;

                int32_t owsw = w + pw - s;
                if(owsw % sw != 0)
                    continue;
                int32_t ow = owsw / sw;
                if(ow < 0 || ow >= OW)
                    continue;

                int32_t dstart = od * sd - pd;
                int32_t hstart = oh * sh - ph;
                int32_t wstart = ow * sw - pw;
                int32_t dend   = min(dstart + KD, D + pd);
                int32_t hend   = min(hstart + R, H + ph);
                int32_t wend   = min(wstart + S, W + pw);

                const int32_t pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
                dstart                  = max(dstart, 0);
                hstart                  = max(hstart, 0);
                wstart                  = max(wstart, 0);
                dend                    = min(dend, D);
                hend                    = min(hend, H);
                wend                    = min(wend, W);
                int32_t divide_factor;
                if(divisor_override != 0)
                {
                    divide_factor = divisor_override;
                }
                else
                {
                    if(count_include_pad)
                    {
                        divide_factor = pool_size;
                    }
                    else
                    {
                        divide_factor = (dend - dstart) * (hend - hstart) * (wend - wstart);
                    }
                }
                grad += CVT_FLOAT2ACCUM(output_grad[output_grad_tv.get_tensor_view_idx(
                            tensor_layout_t<5>(n, c, od, oh, ow))]) /
                        divide_factor;
            }
        }
    }
    input_grad[input_grad_tv.get_tensor_view_idx(tensor_layout_t<5>(n, c, d, h, w))] =
        CVT_ACCUM2FLOAT(grad);
}

extern "C" __global__ void AvgPoolBackward3d(const INPUT_TYPE* __restrict__ output_grad,
                                             OUTPUT_TYPE* __restrict__ input_grad,
                                             size_t N,
                                             size_t C,
                                             size_t D,
                                             size_t H,
                                             size_t W,
                                             size_t OD,
                                             size_t OH,
                                             size_t OW,
                                             int32_t* ksize,
                                             int32_t* stride,
                                             int32_t* padding,
                                             bool count_include_pad,
                                             int32_t divisor_override,
                                             tensor_view_t<5> output_grad_tv,
                                             tensor_view_t<5> input_grad_tv)
{
    avgPoolBackward3d<INPUT_TYPE, OUTPUT_TYPE>(output_grad,
                                               input_grad,
                                               N,
                                               C,
                                               D,
                                               H,
                                               W,
                                               OD,
                                               OH,
                                               OW,
                                               ksize,
                                               stride,
                                               padding,
                                               count_include_pad,
                                               divisor_override,
                                               output_grad_tv,
                                               input_grad_tv);
}
