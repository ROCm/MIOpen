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

__kernel void Col2Im3d(global _FLOAT* col,
                       const int col_d,
                       const int col_h,
                       const int col_w,
                       const int wei_d,
                       const int wei_h,
                       const int wei_w,
                       const int pad_d,
                       const int pad_h,
                       const int pad_w,
                       const int stride_d,
                       const int stride_h,
                       const int stride_w,
                       const int dilation_d,
                       const int dilation_h,
                       const int dilation_w,
                       const int depth,
                       const int height,
                       const int width,
                       global _FLOAT* im,
                       const int im_offset)
{
    global _FLOAT* im_off = im + im_offset;
    int gid               = (int)get_global_id(0);

    int im_ch = gid / (width * height * depth);
    int itmp  = gid % (width * height * depth);
    int im_d  = itmp / (width * height);
    itmp      = itmp % (width * height);
    int im_h  = itmp / width;
    int im_w  = itmp % width;

    im_d += pad_d;
    im_h += pad_h;
    im_w += pad_w;

    int start_d = (im_d < dilation_d * (wei_d - 1) + 1)
                      ? 0
                      : (im_d - (dilation_d * (wei_d - 1) + 1)) / stride_d + 1;
    int end_d = min(col_d, im_d / stride_d + 1);

    int start_h = (im_h < dilation_h * (wei_h - 1) + 1)
                      ? 0
                      : (im_h - (dilation_h * (wei_h - 1) + 1)) / stride_h + 1;
    int end_h = min(col_h, im_h / stride_h + 1);

    int start_w = (im_w < dilation_w * (wei_w - 1) + 1)
                      ? 0
                      : (im_w - (dilation_w * (wei_w - 1) + 1)) / stride_w + 1;
    int end_w = min(col_w, im_w / stride_w + 1);

    int ch_offset = im_ch * col_d * col_w * col_h * wei_d * wei_w * wei_h;
    col += ch_offset;

    _FLOAT_ACCUM tmp = (_FLOAT_ACCUM)0;

    for(int cz = start_d; cz < end_d; cz++)
    {
        for(int cy = start_h; cy < end_h; cy++)
        {
            for(int cx = start_w; cx < end_w; cx++)
            {
                if((im_d - cz * stride_d) % dilation_d == 0 &&
                   (im_h - cy * stride_h) % dilation_h == 0 &&
                   (im_w - cx * stride_w) % dilation_w == 0)
                {
                    int z = (im_d - cz * stride_d) / dilation_d;
                    int y = (im_h - cy * stride_h) / dilation_h;
                    int x = (im_w - cx * stride_w) / dilation_w;

                    int col_off =
                        (((((z * wei_h) + y) * wei_w + x) * col_d + cz) * col_h + cy) * col_w + cx;

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
