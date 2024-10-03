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

template <typename TI, typename TO>
__device__ void avgPoolForward2d(const TI* __restrict__ input,
                                 TO* __restrict__ output,
                                 int64_t N,
                                 int64_t C,
                                 int64_t H,
                                 int64_t W,
                                 int64_t OH,
                                 int64_t OW,
                                 int64_t R,
                                 int64_t S,
                                 int64_t sh,
                                 int64_t sw,
                                 int64_t ph,
                                 int64_t pw,
                                 bool count_include_pad,
                                 int64_t divisor_override,
                                 tensor_view_t<4> input_tv,
                                 tensor_view_t<4> output_tv)
{
    int64_t gid  = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t ncoh = gid / OW, ow = gid % OW;
    int64_t nc = ncoh / OH, oh = ncoh % OH;
    int64_t n = nc / C, c = nc % C;

    if(n >= N)
        return;

    FLOAT_ACCUM m = 0;
    for(int64_t r = 0; r < R; ++r)
    {
        for(int64_t s = 0; s < S; ++s)
        {
            // input idx : (n, c, h, w)
            int64_t h = oh * sh - ph + r;
            if(h < 0 || h >= H)
                continue;
            int64_t w = ow * sw - pw + s;
            if(w < 0 || w >= W)
                continue;
            // int64_t input_idx = ((n * C + c) * H + h) * W + w;
            m += CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({n, c, h, w})]);
        }
    }

    int64_t hstart = oh * sh - ph;
    int64_t wstart = ow * sw - pw;
    int64_t hend   = min(hstart + R, H + ph);
    int64_t wend   = min(wstart + S, W + pw);

    const int64_t pool_size = (hend - hstart) * (wend - wstart);

    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend   = min(hend, H);
    wend   = min(wend, W);

    int64_t divide_factor;
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

    output[output_tv.get_tensor_view_idx({n, c, oh, ow})] = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void AvgPoolForward2d(const INPUT_TYPE* __restrict__ input,
                                            OUTPUT_TYPE* __restrict__ output,
                                            int64_t N,
                                            int64_t C,
                                            int64_t H,
                                            int64_t W,
                                            int64_t OH,
                                            int64_t OW,
                                            int64_t R,
                                            int64_t S,
                                            int64_t sh,
                                            int64_t sw,
                                            int64_t ph,
                                            int64_t pw,
                                            bool count_include_pad,
                                            int64_t divisor_override,
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
                                              R,
                                              S,
                                              sh,
                                              sw,
                                              ph,
                                              pw,
                                              count_include_pad,
                                              divisor_override,
                                              input_tv,
                                              output_tv);
}

template <typename TI, typename TO>
__device__ void avgPoolForward3d(const TI* __restrict__ input,
                                 TO* __restrict__ output,
                                 int64_t N,
                                 int64_t C,
                                 int64_t D,
                                 int64_t H,
                                 int64_t W,
                                 int64_t OD,
                                 int64_t OH,
                                 int64_t OW,
                                 int64_t KD,
                                 int64_t R,
                                 int64_t S,
                                 int64_t sd,
                                 int64_t sh,
                                 int64_t sw,
                                 int64_t pd,
                                 int64_t ph,
                                 int64_t pw,
                                 bool count_include_pad,
                                 int64_t divisor_override,
                                 tensor_view_t<5> input_tv,
                                 tensor_view_t<5> output_tv)
{
    int64_t gid    = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t ncodoh = gid / OW, ow = gid % OW;
    int64_t ncod = ncodoh / OH, oh = ncodoh % OH;
    int64_t nc = ncod / OD, od = ncod % OD;
    int64_t n = nc / C, c = nc % C;

    if(n >= N)
        return;
    FLOAT_ACCUM sum = 0;
    for(int64_t kd = 0; kd < KD; ++kd)
    {
        for(int64_t r = 0; r < R; ++r)
        {
            for(int64_t s = 0; s < S; ++s)
            {
                // input idx : (n, c, d, h, w)
                int64_t d = od * sd - pd + kd;
                if(d < 0 || d >= D)
                    continue;
                int64_t h = oh * sh - ph + r;
                if(h < 0 || h >= H)
                    continue;
                int64_t w = ow * sw - pw + s;
                if(w < 0 || w >= W)
                    continue;
                // int64_t input_idx = ((n * C + c) * H + h) * W + w;
                sum += CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({n, c, d, h, w})]);
            }
        }
    }
    int64_t dstart = od * sd - pd;
    int64_t hstart = oh * sh - ph;
    int64_t wstart = ow * sw - pw;
    int64_t dend   = min(dstart + KD, D + pd);
    int64_t hend   = min(hstart + R, H + ph);
    int64_t wend   = min(wstart + S, W + pw);

    const int64_t pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
    dstart                  = max(dstart, 0);
    hstart                  = max(hstart, 0);
    wstart                  = max(wstart, 0);
    dend                    = min(dend, D);
    hend                    = min(hend, H);
    wend                    = min(wend, W);

    int64_t divide_factor;
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
    FLOAT_ACCUM val                                           = sum / divide_factor;
    output[output_tv.get_tensor_view_idx({n, c, od, oh, ow})] = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void AvgPoolForward3d(const INPUT_TYPE* __restrict__ input,
                                            OUTPUT_TYPE* __restrict__ output,
                                            int64_t N,
                                            int64_t C,
                                            int64_t D,
                                            int64_t H,
                                            int64_t W,
                                            int64_t OD,
                                            int64_t OH,
                                            int64_t OW,
                                            int64_t KD,
                                            int64_t R,
                                            int64_t S,
                                            int64_t sd,
                                            int64_t sh,
                                            int64_t sw,
                                            int64_t pd,
                                            int64_t ph,
                                            int64_t pw,
                                            bool count_include_pad,
                                            int64_t divisor_override,
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
                                              KD,
                                              R,
                                              S,
                                              sd,
                                              sh,
                                              sw,
                                              pd,
                                              ph,
                                              pw,
                                              count_include_pad,
                                              divisor_override,
                                              input_tv,
                                              output_tv);
}

template <typename TI, typename TO>
__device__ void avgPoolBackward2d(const TI* __restrict__ output_grad,
                                  TO* __restrict__ input_grad,
                                  int64_t N,
                                  int64_t C,
                                  int64_t H,
                                  int64_t W,
                                  int64_t OH,
                                  int64_t OW,
                                  int64_t R,
                                  int64_t S,
                                  int64_t sh,
                                  int64_t sw,
                                  int64_t ph,
                                  int64_t pw,
                                  bool count_include_pad,
                                  int64_t divisor_override,
                                  tensor_view_t<4> output_grad_tv,
                                  tensor_view_t<4> input_grad_tv)
{
    int64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t nch = gid / W, w = gid % W;
    int64_t nc = nch / H, h = nch % H;
    int64_t n = nc / C, c = nc % C;

    if(n >= N)
        return;

    FLOAT_ACCUM grad = 0;
    for(int64_t r = 0; r < R; ++r)
    {
        for(int64_t s = 0; s < S; ++s)
        {
            int64_t ohsh = h + ph - r;
            if(ohsh % sh != 0)
                continue;
            int64_t oh = ohsh / sh;
            if(oh < 0 || oh >= OH)
                continue;
            int64_t owsw = w + pw - s;
            if(owsw % sw != 0)
                continue;
            int64_t ow = owsw / sw;
            if(ow < 0 || ow >= OW)
                continue;

            int64_t hstart = oh * sh - ph;
            int64_t wstart = ow * sw - pw;
            int64_t hend   = min(hstart + R, H + ph);
            int64_t wend   = min(wstart + S, W + pw);

            const int64_t pool_size = (hend - hstart) * (wend - wstart);

            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend   = min(hend, H);
            wend   = min(wend, W);

            int64_t divide_factor;
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

            grad +=
                CVT_FLOAT2ACCUM(output_grad[output_grad_tv.get_tensor_view_idx({n, c, oh, ow})]) /
                divide_factor;
        }
    }
    input_grad[input_grad_tv.get_tensor_view_idx({n, c, h, w})] = CVT_ACCUM2FLOAT(grad);
}

extern "C" __global__ void AvgPoolBackward2d(const INPUT_TYPE* __restrict__ output_grad,
                                             OUTPUT_TYPE* __restrict__ input_grad,
                                             int64_t N,
                                             int64_t C,
                                             int64_t H,
                                             int64_t W,
                                             int64_t OH,
                                             int64_t OW,
                                             int64_t R,
                                             int64_t S,
                                             int64_t sh,
                                             int64_t sw,
                                             int64_t ph,
                                             int64_t pw,
                                             bool count_include_pad,
                                             int64_t divisor_override,
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
                                               R,
                                               S,
                                               sh,
                                               sw,
                                               ph,
                                               pw,
                                               count_include_pad,
                                               divisor_override,
                                               output_grad_tv,
                                               input_grad_tv);
}

template <typename TI, typename TO>
__device__ void avgPoolBackward3d(const TI* __restrict__ output_grad,
                                  TO* __restrict__ input_grad,
                                  int64_t N,
                                  int64_t C,
                                  int64_t D,
                                  int64_t H,
                                  int64_t W,
                                  int64_t OD,
                                  int64_t OH,
                                  int64_t OW,
                                  int64_t KD,
                                  int64_t R,
                                  int64_t S,
                                  int64_t sd,
                                  int64_t sh,
                                  int64_t sw,
                                  int64_t pd,
                                  int64_t ph,
                                  int64_t pw,
                                  bool count_include_pad,
                                  int64_t divisor_override,
                                  tensor_view_t<5> output_grad_tv,
                                  tensor_view_t<5> input_grad_tv)
{
    int64_t gid  = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t ncdh = gid / W, w = gid % W;
    int64_t ncd = ncdh / H, h = ncdh % H;
    int64_t nc = ncd / D, d = ncd % D;
    int64_t n = nc / C, c = nc % C;

    if(n >= N)
        return;

    FLOAT_ACCUM grad = 0;
    for(int64_t kd = 0; kd < KD; ++kd)
    {
        for(int64_t r = 0; r < R; ++r)
        {
            for(int64_t s = 0; s < S; ++s)
            {
                int64_t odsd = d + pd - kd;
                if(odsd % sd != 0)
                    continue;
                int64_t od = odsd / sd;
                if(od < 0 || od >= OD)
                    continue;

                int64_t ohsh = h + ph - r;
                if(ohsh % sh != 0)
                    continue;
                int64_t oh = ohsh / sh;
                if(oh < 0 || oh >= OH)
                    continue;

                int64_t owsw = w + pw - s;
                if(owsw % sw != 0)
                    continue;
                int64_t ow = owsw / sw;
                if(ow < 0 || ow >= OW)
                    continue;

                int64_t dstart = od * sd - pd;
                int64_t hstart = oh * sh - ph;
                int64_t wstart = ow * sw - pw;
                int64_t dend   = min(dstart + KD, D + pd);
                int64_t hend   = min(hstart + R, H + ph);
                int64_t wend   = min(wstart + S, W + pw);

                const int64_t pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
                dstart                  = max(dstart, 0);
                hstart                  = max(hstart, 0);
                wstart                  = max(wstart, 0);
                dend                    = min(dend, D);
                hend                    = min(hend, H);
                wend                    = min(wend, W);
                int64_t divide_factor;
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
                grad += CVT_FLOAT2ACCUM(
                            output_grad[output_grad_tv.get_tensor_view_idx({n, c, od, oh, ow})]) /
                        divide_factor;
            }
        }
    }
    input_grad[input_grad_tv.get_tensor_view_idx({n, c, d, h, w})] = CVT_ACCUM2FLOAT(grad);
}

extern "C" __global__ void AvgPoolBackward3d(const INPUT_TYPE* __restrict__ output_grad,
                                             OUTPUT_TYPE* __restrict__ input_grad,
                                             int64_t N,
                                             int64_t C,
                                             int64_t D,
                                             int64_t H,
                                             int64_t W,
                                             int64_t OD,
                                             int64_t OH,
                                             int64_t OW,
                                             int64_t KD,
                                             int64_t R,
                                             int64_t S,
                                             int64_t sd,
                                             int64_t sh,
                                             int64_t sw,
                                             int64_t pd,
                                             int64_t ph,
                                             int64_t pw,
                                             bool count_include_pad,
                                             int64_t divisor_override,
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
                                               KD,
                                               R,
                                               S,
                                               sd,
                                               sh,
                                               sw,
                                               pd,
                                               ph,
                                               pw,
                                               count_include_pad,
                                               divisor_override,
                                               output_grad_tv,
                                               input_grad_tv);
}
