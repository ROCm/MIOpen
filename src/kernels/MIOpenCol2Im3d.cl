/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#include "float_types.h"

#if MIOPEN_USE_FP16
#define ACCUMULATOR_NEEDS_CONVERSION 1
#elif MIOPEN_USE_BFP16
#define ACCUMULATOR_NEEDS_CONVERSION 1
#elif MIOPEN_USE_FP32
#define ACCUMULATOR_NEEDS_CONVERSION 0
#endif

#ifndef MIOPEN_USE_64BIT_INDEX
#error "MIOPEN_USE_64BIT_INDEX must be defined"
#endif

__kernel void Col2Im3dU(global _FLOAT* col,
                        const uint col_d,
                        const uint col_h,
                        const uint col_w,
                        const uint wei_d,
                        const uint wei_h,
                        const uint wei_w,
                        const uint pad_d,
                        const uint pad_h,
                        const uint pad_w,
                        const uint stride_d,
                        const uint stride_h,
                        const uint stride_w,
                        const uint dilation_d,
                        const uint dilation_h,
                        const uint dilation_w,
                        const uint depth,
                        const uint height,
                        const uint width,
                        global _FLOAT* im,
                        const unsigned long im_offset)
{
    global _FLOAT* im_off = im + im_offset;
    uint gid              = (uint)get_global_id(0);

    uint im_ch = gid / (width * height * depth);
    uint itmp  = gid % (width * height * depth);
    uint im_d  = itmp / (width * height);
    itmp       = itmp % (width * height);
    uint im_h  = itmp / width;
    uint im_w  = itmp % width;

    im_d += pad_d;
    im_h += pad_h;
    im_w += pad_w;

    uint start_d = (im_d < dilation_d * (wei_d - 1) + 1)
                       ? 0
                       : (im_d - (dilation_d * (wei_d - 1) + 1)) / stride_d + 1;
    uint end_d   = min(col_d, im_d / stride_d + 1);

    uint start_h = (im_h < dilation_h * (wei_h - 1) + 1)
                       ? 0
                       : (im_h - (dilation_h * (wei_h - 1) + 1)) / stride_h + 1;
    uint end_h   = min(col_h, im_h / stride_h + 1);

    uint start_w = (im_w < dilation_w * (wei_w - 1) + 1)
                       ? 0
                       : (im_w - (dilation_w * (wei_w - 1) + 1)) / stride_w + 1;
    uint end_w   = min(col_w, im_w / stride_w + 1);

#if MIOPEN_USE_64BIT_INDEX
    ulong ch_offset = (ulong)im_ch * col_d * col_w * col_h * wei_d * wei_w * wei_h;
#else
    uint ch_offset = im_ch * col_d * col_w * col_h * wei_d * wei_w * wei_h;
#endif

    col += ch_offset;

    _FLOAT_ACCUM tmp = (_FLOAT_ACCUM)0;

    for(uint cz = start_d; cz < end_d; cz++)
    {
        for(uint cy = start_h; cy < end_h; cy++)
        {
            for(uint cx = start_w; cx < end_w; cx++)
            {
                if((im_d - cz * stride_d) % dilation_d == 0 &&
                   (im_h - cy * stride_h) % dilation_h == 0 &&
                   (im_w - cx * stride_w) % dilation_w == 0)
                {
                    uint z = (im_d - cz * stride_d) / dilation_d;
                    uint y = (im_h - cy * stride_h) / dilation_h;
                    uint x = (im_w - cx * stride_w) / dilation_w;

#if MIOPEN_USE_64BIT_INDEX
                    ulong col_off =
                        ((((((ulong)z * wei_h) + y) * wei_w + x) * col_d + cz) * col_h + cy) *
                            col_w +
                        cx;
#else
                    uint col_off =
                        (((((z * wei_h) + y) * wei_w + x) * col_d + cz) * col_h + cy) * col_w + cx;
#endif

                    tmp += CVT_FLOAT2ACCUM(col[col_off]);
                }
            }
        }
    }
#if ACCUMULATOR_NEEDS_CONVERSION
    im_off[gid] = tmp > CVT_FLOAT2ACCUM(MAX_VAL) ? MAX_VAL : CVT_ACCUM2FLOAT(tmp);
#else
    im_off[gid] = CVT_ACCUM2FLOAT(tmp);
#endif
}
