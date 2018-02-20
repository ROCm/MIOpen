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
#define PPCAT_NX(A, B) A##B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define TWO 2
#define FOUR 4
#define EIGHT 8

#if MIOPEN_USE_FP16 == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT half
#ifndef HALF_MAX
#define MAX_VAL 65504 /* max value */
#else
#define MAX_VAL HALF_MAX
#endif
#endif
#if MIOPEN_USE_FP32 == 1
#define _FLOAT float
#ifndef FLT_MAX
#define MAX_VAL 3.402823466e+38F /* max value */
#else
#define MAX_VAL FLT_MAX
#endif
#endif

#define _FLOAT2 PPCAT(_FLOAT, TWO)
#define _FLOAT4 PPCAT(_FLOAT, FOUR)
#define _FLOAT8 PPCAT(_FLOAT, EIGHT)

/* Simple GPU implementation - number of threads launced == sizeof im2col buffer
 * Each thread writes one pixel of output. First (out_h*out_w) threads write to
 * the first line (row) of the im2col output.
 *
 * kernel void Im2Col(global _FLOAT* im, size_t im_offset,
 * 		const int h, const int w,
 * 		const int wei_h, const int wei_w,
 * 		const int out_h, const int out_w,
 * 		const int pad_h, const int pad_w,
 * 		const int stride_h, const int stride_w,
 * 		global _FLOAT* col)
 * {
 * 	int tid = get_global_id(0);
 *  // which row of the output to write to
 * 	int col_row = tid / (out_h * out_w);
 *
 *  // which pixel from the image and which channel to read from
 * 	int im_x = col_row % wei_w; // used to compute im_off_w
 * 	int im_y = (col_row / wei_w) % wei_h; // used to compute im_off_y
 * 	int im_c = col_row / (wei_w * wei_h); // im_c is the img channel
 *
 * 	int out_x = tid % out_w;
 * 	int out_y = (tid / out_w) % out_h;
 *
 *  // take the strides and padding into account while reading from the image
 * 	int im_off_h = out_y * stride_h - pad_h + im_y;
 * 	int im_off_w = out_x * stride_w - pad_w + im_x;
 *
 * 	global _FLOAT *im_off = (global _FLOAT *)&im[im_offset];
 *
 * 	if(im_off_h >= 0 && im_off_h < h && im_off_w >= 0 && im_off_w < w) {
 * 		col[col_row*out_h*out_w + out_y*out_w + out_x] = im_off[im_c*h*w + im_off_h*w +
 * im_off_w];
 * 	}
 * 	else {
 * 		col[col_row*out_h*out_w + out_y*out_w + out_x] = 0.;
 * 	}
 * }
 */

kernel void Im2Col(const int data_size_off,
                   global _FLOAT* im,
                   size_t im_offset,
                   const int h,
                   const int w,
                   const int wei_h,
                   const int wei_w,
                   const int out_h,
                   const int out_w,
                   const int pad_h,
                   const int pad_w,
                   const int stride_h,
                   const int stride_w,
                   const int dilation_h,
                   const int dilation_w,
                   global _FLOAT* col)
{
#define THREADS_PER_CH (256 / NUM_CH_PER_WG)

#if USE_IM_OFF_GUARD
#define IM_OFF_GUARD(idx) (idx) < data_size_off ? im_off[(idx)] : 0
#else
#define IM_OFF_GUARD(idx) im_off[idx]
#endif

    global _FLOAT* im_off = im + im_offset;
    int lid               = get_local_id(0);
    int gid               = get_group_id(0);

#if NUM_IM_BLKS == 1 && STRIDE_GT_1 == 0

    // Load image into LDS
    local _FLOAT local_im[LOCAL_MEM_SIZE];

    int witem_ch = lid / THREADS_PER_CH;

    int im_lid = lid;
    while(im_lid < NUM_CH_PER_WG * h * w)
    {
        local_im[im_lid] = IM_OFF_GUARD((gid * NUM_CH_PER_WG) * h * w + im_lid);
        im_lid += 256;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // where will each thread to col
    int witem_ch_offset = witem_ch * h * w;

    if(lid % THREADS_PER_CH < out_h * out_w)
    {
        int inner_lid = lid % THREADS_PER_CH;
        int out_x     = inner_lid % out_w;
        int out_y     = inner_lid / out_w;

        int col_x = out_y * out_w + out_x;
        int col_y = (gid * NUM_CH_PER_WG + witem_ch) * out_h * out_w * wei_h * wei_w;

        for(int y = 0; y < wei_h; y++)
        {
            for(int x = 0; x < wei_w; x++)
            {
                int im_off_h = out_y * stride_h - pad_h + y * dilation_h;
                int im_off_w = out_x * stride_w - pad_w + x * dilation_w;
                if(im_off_h >= 0 && im_off_h < h && im_off_w >= 0 && im_off_w < w)
                    col[col_y + col_x + (y * wei_w + x) * out_h * out_w] =
                        local_im[witem_ch_offset + (im_off_h)*w + im_off_w];
                else
                    col[col_y + col_x + (y * wei_w + x) * out_h * out_w] = 0;
            }
        }
    }

#else  // NUM_IM_BLKS > 1 || STRIDE_GT_1 1

    local _FLOAT local_im[LOCAL_MEM_SIZE];

    int wg_ch = gid / NUM_IM_BLKS;

    int im_x = ((gid % NUM_IM_BLKS) % NUM_IM_BLKS_X) * TILE_SZ_X;
    int im_y = ((gid % NUM_IM_BLKS) / NUM_IM_BLKS_X) * TILE_SZ_Y;

    int out_cols_wg = im_x + TILE_SZ_X <= out_w ? TILE_SZ_X : out_w - im_x;
    int out_rows_wg = im_y + TILE_SZ_Y <= out_h ? TILE_SZ_Y : out_h - im_y;

    int im_cols_wg = (TILE_SZ_X - 1) * stride_w + (wei_w - 1) * dilation_w + 1;
    int inner_lid  = lid;

    while(inner_lid < LOCAL_MEM_SIZE)
    {
        int row_to_use = inner_lid / im_cols_wg;
        int col_to_use = inner_lid % im_cols_wg;
        int lm_offset  = row_to_use * im_cols_wg + col_to_use;
        if(im_y * stride_h + row_to_use >= pad_h && im_y * stride_h + row_to_use < h + pad_h &&
           im_x * stride_w + col_to_use >= pad_w && im_x * stride_w + col_to_use < w + pad_w)
        {
            int im_off_h        = im_y * stride_h + row_to_use - pad_h;
            int im_off_w        = im_x * stride_w + col_to_use - pad_w;
            local_im[lm_offset] = IM_OFF_GUARD(wg_ch * h * w + im_off_h * w + im_off_w);
        }
        else
            local_im[lm_offset] = 0;

        inner_lid += 256;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    inner_lid = lid;
    while(inner_lid < out_cols_wg * out_rows_wg)
    {
        int out_x = inner_lid % out_cols_wg;
        int out_y = inner_lid / out_cols_wg;

        int col_x = (im_y + out_y) * out_w + im_x + out_x;
        int col_y = (gid / NUM_IM_BLKS) * out_h * out_w * wei_h * wei_w;

        for(int y = 0; y < wei_h; y++)
        {
            for(int x = 0; x < wei_w; x++)
            {
                int im_off_h = out_y * stride_h + y * dilation_h;
                int im_off_w = out_x * stride_w + x * dilation_w;
                col[col_y + col_x + (y * wei_w + x) * out_h * out_w] =
                    local_im[(im_off_h)*im_cols_wg + im_off_w];
            }
        }
        inner_lid += 256;
    }
#endif // NUM_IM_BLKS && STRIDE_GT_1
}
