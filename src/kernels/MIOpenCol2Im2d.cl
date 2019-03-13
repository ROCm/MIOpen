/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#if MIOPEN_USE_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half data_t;
typedef float accumulator_t;
#define ACCUMULATOR_NEEDS_CONVERSION 1
#ifndef HALF_MAX
#define MAX_VAL 65504 /* max value */
#else
#define MAX_VAL HALF_MAX
#endif
#elif MIOPEN_USE_FP32
typedef float data_t;
typedef float accumulator_t;
#define ACCUMULATOR_NEEDS_CONVERSION 0
#ifndef FLT_MAX
#define MAX_VAL 3.402823466e+38F /* max value */
#else
#define MAX_VAL FLT_MAX
#endif
#endif

__kernel void Col2Im2d(global data_t* col,
                       const int col_h,
                       const int col_w,
                       const int wei_h,
                       const int wei_w,
                       const int pad_h,
                       const int pad_w,
                       const int stride_h,
                       const int stride_w,
                       const int dilation_h,
                       const int dilation_w,
                       const int height,
                       const int width,
                       global data_t* im,
                       const int im_offset)
{
    global data_t* im_off = im + im_offset;
    int gid               = (int)get_global_id(0);

    int im_ch  = gid / (width * height);
    int im_pix = gid % (width * height);
    int im_h   = (im_pix / width) + pad_h;
    int im_w   = (im_pix % width) + pad_w;

    int start_h = (im_h < dilation_h * (wei_h - 1) + 1)
                      ? 0
                      : (im_h - (dilation_h * (wei_h - 1) + 1)) / stride_h + 1;
    int end_h   = min(col_h, im_h / stride_h + 1);
    int start_w = (im_w < dilation_w * (wei_w - 1) + 1)
                      ? 0
                      : (im_w - (dilation_w * (wei_w - 1) + 1)) / stride_w + 1;
    int end_w = min(col_w, im_w / stride_w + 1);

    int ch_offset = im_ch * col_w * col_h * wei_w * wei_h;
    col += ch_offset;

    accumulator_t tmp = (accumulator_t)0;
    for(int cy = start_h; cy < end_h; cy++)
    {
        for(int cx = start_w; cx < end_w; cx++)
        {
            if((im_h - cy * stride_h) % dilation_h == 0 && (im_w - cx * stride_w) % dilation_w == 0)
            {
                int col_off_y = cy + (((im_h - cy * stride_h) / dilation_h) * wei_w * col_h);
                int col_off_x = cx + (((im_w - cx * stride_w) / dilation_w) * col_w * col_h);

                tmp += (accumulator_t)(col[col_off_y * col_w + col_off_x]);
            }
        }
    }
#if ACCUMULATOR_NEEDS_CONVERSION
    im_off[gid] = tmp > ((accumulator_t)MAX_VAL) ? MAX_VAL : (data_t)tmp;
#else
    im_off[gid] = tmp;
#endif
}
