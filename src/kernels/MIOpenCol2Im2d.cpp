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
#include "float_types.h"
#include "miopen_cstdint.hpp"

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

#if MIOPEN_USE_64BIT_INDEX
using index_t = uint64_t;
#else
using index_t = uint32_t;
#endif

extern "C" __global__ void Col2Im2dU(FLOAT* col,
                                     const unsigned int col_h,
                                     const unsigned int col_w,
                                     const unsigned int wei_h,
                                     const unsigned int wei_w,
                                     const unsigned int pad_h,
                                     const unsigned int pad_w,
                                     const unsigned int stride_h,
                                     const unsigned int stride_w,
                                     const unsigned int dilation_h,
                                     const unsigned int dilation_w,
                                     const unsigned int channels,
                                     const unsigned int height,
                                     const unsigned int width,
                                     FLOAT* im,
                                     const unsigned int im_offset)
{
    FLOAT* im_off            = im + im_offset;
    unsigned int gid         = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int global_size = channels * height * width;

    unsigned int im_ch  = gid / (width * height);
    unsigned int im_pix = gid % (width * height);
    unsigned int im_h   = (im_pix / width) + pad_h;
    unsigned int im_w   = (im_pix % width) + pad_w;

    if(gid >= global_size)
        return;

    unsigned int start_h = (im_h < dilation_h * (wei_h - 1) + 1)
                               ? 0
                               : (im_h - (dilation_h * (wei_h - 1) + 1)) / stride_h + 1;
    unsigned int end_h   = min(col_h, im_h / stride_h + 1);
    unsigned int start_w = (im_w < dilation_w * (wei_w - 1) + 1)
                               ? 0
                               : (im_w - (dilation_w * (wei_w - 1) + 1)) / stride_w + 1;
    unsigned int end_w   = min(col_w, im_w / stride_w + 1);

    index_t ch_offset = (index_t)(im_ch * col_w * col_h) * (index_t)(wei_w * wei_h);
    col += ch_offset;

    FLOAT_ACCUM tmp = (FLOAT_ACCUM)0;
    for(unsigned int cy = start_h; cy < end_h; cy++)
    {
        for(unsigned int cx = start_w; cx < end_w; cx++)
        {
            if((im_h - cy * stride_h) % dilation_h == 0 && (im_w - cx * stride_w) % dilation_w == 0)
            {
                index_t col_off_y = cy + (((im_h - cy * stride_h) / dilation_h) * wei_w * col_h);
                index_t col_off_x = cx + (((im_w - cx * stride_w) / dilation_w) * col_w * col_h);

                tmp += CVT_FLOAT2ACCUM(col[col_off_y * col_w + col_off_x]);
            }
        }
    }

#if ACCUMULATOR_NEEDS_CONVERSION
    im_off[gid] = tmp > CVT_FLOAT2ACCUM(MAX_VAL) ? MAX_VAL : CVT_ACCUM2FLOAT(tmp);
#else
    im_off[gid] = tmp;
#endif
}
