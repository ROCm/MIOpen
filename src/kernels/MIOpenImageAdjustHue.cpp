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

template <typename TIO = FLOAT_ACCUM>
__device__ TIO clamp(TIO val, TIO min, TIO max)
{
    val = val < min ? min : val;
    val = val > max ? max : val;
    return val;
}

template <typename TIO = FLOAT_ACCUM>
__device__ void convertRGBToHSV(const TIO r, const TIO g, const TIO b, TIO* h, TIO* s, TIO* v)
{
    TIO minc = fmin(r, fmin(g, b));
    TIO maxc = fmax(r, fmax(g, b));

    *v = maxc;

    TIO cr   = maxc - minc;
    bool eqc = (cr == 0);

    *s = cr / (eqc ? static_cast<TIO>(1.0) : maxc);

    TIO cr_divisor = eqc ? static_cast<TIO>(1.0) : cr;
    TIO rc         = (maxc - r) / cr_divisor;
    TIO gc         = (maxc - g) / cr_divisor;
    TIO bc         = (maxc - b) / cr_divisor;

    TIO hr = (maxc == r) * (bc - gc);
    TIO hg = ((maxc == g) & (maxc != r)) * (static_cast<TIO>(2.0) + rc - bc);
    TIO hb = ((maxc != g) & (maxc != r)) * (static_cast<TIO>(4.0) + gc - rc);

    *h =
        fmod((hr + hg + hb) / static_cast<TIO>(6.0) + static_cast<TIO>(1.0), static_cast<TIO>(1.0));
}

template <typename TIO = FLOAT_ACCUM>
__device__ void convertHSVToRGB(const TIO h, const TIO s, const TIO v, TIO* r, TIO* g, TIO* b)
{
    TIO i      = floor(h * static_cast<TIO>(6.0));
    TIO f      = (h * static_cast<TIO>(6.0)) - i;
    int i_case = (static_cast<int>(i) + 6) % 6;

    TIO p = clamp(v * (static_cast<TIO>(1.0) - s), static_cast<TIO>(0.0), static_cast<TIO>(1.0));
    TIO q =
        clamp(v * (static_cast<TIO>(1.0) - s * f), static_cast<TIO>(0.0), static_cast<TIO>(1.0));
    TIO t = clamp(v * (static_cast<TIO>(1.0) - s * (static_cast<TIO>(1.0) - f)),
                  static_cast<TIO>(0.0),
                  static_cast<TIO>(1.0));

    switch(i_case)
    {
    case 0:
        *r = v;
        *g = t;
        *b = p;
        break;
    case 1:
        *r = q;
        *g = v;
        *b = p;
        break;
    case 2:
        *r = p;
        *g = v;
        *b = t;
        break;
    case 3:
        *r = p;
        *g = q;
        *b = v;
        break;
    case 4:
        *r = t;
        *g = p;
        *b = v;
        break;
    case 5:
        *r = v;
        *g = p;
        *b = q;
        break;
    default: return;
    }
}

template <typename TIO>
__device__ void DeviceImageAdjustHue(const TIO* input,
                                     TIO* output,
                                     float hue_factor,
                                     size_t N,
                                     tensor_view_t<4> input_tv,
                                     tensor_view_t<4> output_tv)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= N)
        return;

    tensor_layout_t<4> input_layout(input_tv, gid);
    auto n = input_layout.layout[0];
    auto c = input_layout.layout[1];
    auto h = input_layout.layout[2];
    auto w = input_layout.layout[3];

    n = n * 3 + c;

    TIO r = input[input_tv.get_tensor_view_idx({n, 0, h, w})];
    TIO g = input[input_tv.get_tensor_view_idx({n, 1, h, w})];
    TIO b = input[input_tv.get_tensor_view_idx({n, 2, h, w})];

    FLOAT_ACCUM fr = CVT_FLOAT2ACCUM(r);
    FLOAT_ACCUM fg = CVT_FLOAT2ACCUM(g);
    FLOAT_ACCUM fb = CVT_FLOAT2ACCUM(b);

    FLOAT_ACCUM hue, sat, val;
    convertRGBToHSV(fr, fg, fb, &hue, &sat, &val);
    hue = fmod(hue + hue_factor, 1.0f);
    convertHSVToRGB(hue, sat, val, &fr, &fg, &fb);

    output[output_tv.get_tensor_view_idx({n, 0, h, w})] = CVT_ACCUM2FLOAT(fr);
    output[output_tv.get_tensor_view_idx({n, 1, h, w})] = CVT_ACCUM2FLOAT(fg);
    output[output_tv.get_tensor_view_idx({n, 2, h, w})] = CVT_ACCUM2FLOAT(fb);
}

template <typename TIO>
__device__ void DeviceImageAdjustHueContiguous(
    const TIO* input, TIO* output, float hue_factor, size_t N, size_t c_stride)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= N)
        return;

    const size_t n   = gid / c_stride;
    const size_t idx = gid % c_stride;

    size_t pixel_idx = idx + n * c_stride * 3;

    TIO r = input[pixel_idx];
    TIO g = input[pixel_idx + c_stride];
    TIO b = input[pixel_idx + c_stride * 2];

    FLOAT_ACCUM fr = CVT_FLOAT2ACCUM(r);
    FLOAT_ACCUM fg = CVT_FLOAT2ACCUM(g);
    FLOAT_ACCUM fb = CVT_FLOAT2ACCUM(b);

    FLOAT_ACCUM hue, sat, val;

    convertRGBToHSV(fr, fg, fb, &hue, &sat, &val);
    hue = fmod(hue + hue_factor, 1.0f);
    convertHSVToRGB(hue, sat, val, &fr, &fg, &fb);

    output[pixel_idx]                = CVT_ACCUM2FLOAT(fr);
    output[pixel_idx + c_stride]     = CVT_ACCUM2FLOAT(fg);
    output[pixel_idx + c_stride * 2] = CVT_ACCUM2FLOAT(fb);
}

// Trampolines
extern "C" __global__ void ImageAdjustHue(const DTYPE* input,
                                          DTYPE* output,
                                          float hue_factor,
                                          size_t N,
                                          size_t c_stride,
                                          tensor_view_t<4> input_tv,
                                          tensor_view_t<4> output_tv)
{
    DeviceImageAdjustHue<DTYPE>(input, output, hue_factor, N, c_stride, input_tv, output_tv);
}

extern "C" __global__ void ImageAdjustHueContiguous(
    const DTYPE* input, DTYPE* output, float hue_factor, size_t N, size_t c_stride)
{
    DeviceImageAdjustHueContiguous<DTYPE>(input, output, hue_factor, N, c_stride);
}
