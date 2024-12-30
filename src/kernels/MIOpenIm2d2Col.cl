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

#ifndef MIOPEN_USE_FP32
#define MIOPEN_USE_FP32 0
#endif

#ifndef MIOPEN_USE_FP16
#define MIOPEN_USE_FP16 0
#endif

#ifndef MIOPEN_USE_BFP16
#define MIOPEN_USE_BFP16 0
#endif

#ifndef MIOPEN_USE_INT8
#define MIOPEN_USE_INT8 0
#endif

#ifndef MIOPEN_USE_INT32
#define MIOPEN_USE_INT32 0
#endif

#ifndef MIOPEN_USE_FP8
#define MIOPEN_USE_FP8 0
#endif

#ifndef MIOPEN_USE_BFP8
#define MIOPEN_USE_BFP8 0
#endif

#if MIOPEN_USE_INT8 || MIOPEN_USE_FP8 || MIOPEN_USE_BFP8
typedef char data_t;
#elif MIOPEN_USE_INT32
typedef int data_t;
#elif(MIOPEN_USE_FP16 || MIOPEN_USE_BFP16)
// As the half type degrades the performance, use short instead of half in the
// im2col, which has no match op. May change back to half when compile can
// deliver equal performance as short
typedef short data_t;
#elif MIOPEN_USE_FP32
typedef float data_t;
#endif

/* Simple GPU implementation - number of threads launced == sizeof im2col buffer
 * Each thread writes one pixel of output. First (out_h*out_w) threads write to
 * the first line (row) of the im2col output.
 *
 * kernel void Im2Col(global data_t* im, int im_offset,
 * 		const int h, const int w,
 * 		const int wei_h, const int wei_w,
 * 		const int out_h, const int out_w,
 * 		const int pad_h, const int pad_w,
 * 		const int stride_h, const int stride_w,
 * 		global data_t* col)
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
 * 	global data_t *im_off = (global data_t *)&im[im_offset];
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

#ifdef USE_LARGE_BUFFER_INDEX
typedef long index_t;
#else
typedef int index_t;
#endif

kernel void Im2d2Col_v2(const int data_size_off,
                        global data_t* im,
                        const ulong im_offset,
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
                        global data_t* col,
                        const int num_ch_per_wg,
                        const int num_im_blks_x,
                        const int num_im_blks,
                        const int tile_sz_x,
                        const int tile_sz_y)
{
    /// NUM_CH_PER_WG {1;4}
    /// THREADS_PER_CH {256; 64}
    (void)num_ch_per_wg;
    (void)num_im_blks_x;
    (void)num_im_blks;
    (void)tile_sz_x;
    (void)tile_sz_y;

#if USE_IM_OFF_GUARD
#define IM_OFF_GUARD(idx) (idx) < data_size_off ? im_off[(idx)] : 0
#else
#define IM_OFF_GUARD(idx) im_off[idx]
#endif

    global data_t* im_off = im + im_offset;

#ifndef EXTREME_LARGE

    int lid = get_local_id(0);
    /// tile_sz_x = {32,16,8,4,2,1}, tile_sz_y = {8,4,2,1}
    /// NUM_IM_BLKS_X = out_w / tile_sz_x
    /// NUM_IM_BLKS = NUM_IM_BLKS_X * out_h / tile_sz_y => out_w * out_h
    /// c * NUM_IM_BLKS => c * out_w * out_h
    index_t gid = get_group_id(0);

#if NUM_IM_BLKS_EQ_1 == 1 && STRIDE_GT_1 == 0
    // This does not need to be a division and should be a right shift
    const int threads_per_ch = 256 / num_ch_per_wg;

    // Load image into LDS
    /// max (LOCAL_MEM_SIZE) = 65536
    local data_t local_im[LOCAL_MEM_SIZE];

    /// witem_ch [0;4)
    int witem_ch = lid / threads_per_ch;

    int im_lid = lid;
    /// h*w < LOCAL_MEM_SIZE/witem_ch
    int gid_stride = num_ch_per_wg * h * w;
    while(im_lid < gid_stride)
    {
        /// gid = max(1, (c_pack / NUM_CH_PER_WG)) => c
        /// max (c * LOCAL_MEM_SIZE) => 65536 * c
        index_t im_off_id = gid * gid_stride + im_lid;
        local_im[im_lid]  = IM_OFF_GUARD(im_off_id);
        im_lid += 256;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // where will each thread to col
    /// should fit in LDS size => witem_ch_offset < LOCAL_MEM_SIZE
    /// h*w < LOCAL_MEM_SIZE/witem_ch
    int witem_ch_offset = witem_ch * h * w;
    /// if (NUM_IM_BLKS == 1) => (out_h < 8 && out_w < 32)
    ///      => out_hw_stride < 256
    int out_hw_stride = out_h * out_w;
    if(lid % threads_per_ch < out_hw_stride)
    {
        /// lid[0, 255] % THREADS_PER_CH {256; 64} =>
        /// max(inner_lid)=255; max(out_x)=max(out_y)=255
        int inner_lid = lid % threads_per_ch;
        int out_x     = inner_lid % out_w;
        int out_y     = inner_lid / out_w;

        /// out_w < 32; out_y < 255; out_x < 255
        /// col_x < 2 080 800
        int col_x = out_y * out_w + out_x;
        /// gid = c = group_cnt-1; NUM_CH_PER_WG{1,4}; out_hw_stride < 256;
        /// EXTREME_LARGE==0
        /// => wei_h * wei_w * type_size * NUM_CH_PER_WG < max (LOCAL_MEM_SIZE)
        /// gid * out_hw_stride * LOCAL_MEM_SIZE => c * 256 * 65536
        index_t col_y = ((index_t)gid * num_ch_per_wg + witem_ch) * out_hw_stride * wei_h * wei_w;

        for(int y = 0; y < wei_h; y++)
        {
            for(int x = 0; x < wei_w; x++)
            {
                /// max(im_off_h)*w <= max(LOCAL_MEM_SIZE); max(im_off_w) <= max(LOCAL_MEM_SIZE);
                int im_off_h = out_y * stride_h - pad_h + y * dilation_h;
                int im_off_w = out_x * stride_w - pad_w + x * dilation_w;
                /// y * wei_w * type_size * NUM_CH_PER_WG < max (LOCAL_MEM_SIZE)
                int im_off_wei_hw = y * wei_w + x;
                // col_x + (im_off_wei_hw * out_hw_stride) => 2 080 800 + 65536 * 255
                index_t col_off = col_y + col_x + im_off_wei_hw * out_hw_stride;
                if(im_off_h >= 0 && im_off_h < h && im_off_w >= 0 && im_off_w < w)
                    col[col_off] = local_im[witem_ch_offset + (im_off_h)*w + im_off_w];
                else
                    col[col_off] = 0;
            }
        }
    }

#else  // NUM_IM_BLKS > 1 || STRIDE_GT_1 1

    local data_t local_im[LOCAL_MEM_SIZE];

    int wg_ch = gid / num_im_blks;
    /// TILE_SZ_X = 32, TILE_SZ_Y = 8;
    /// gid = c * NUM_IM_BLKS => im_x = NUM_IM_BLKS*TILE_SZ_X = NUM_IM_BLKS*32
    /// = NUM_IM_BLKS*32 = out_w * out_h / 8
    int im_x = ((gid % num_im_blks) % num_im_blks_x) * tile_sz_x; /// < out_w
    int im_y = ((gid % num_im_blks) / num_im_blks_x) * tile_sz_y; /// < out_h

    int out_cols_wg = (im_x + tile_sz_x) <= out_w ? tile_sz_x : (out_w - im_x); /// < out_w
    int out_rows_wg = (im_y + tile_sz_y) <= out_h ? tile_sz_y : (out_h - im_y); /// < out_h

    int im_cols_wg = (tile_sz_x - 1) * stride_w + (wei_w - 1) * dilation_w + 1;

    int inner_lid = lid;

    while(inner_lid < LOCAL_MEM_SIZE)
    {
        /// < 256
        int row_to_use = inner_lid / im_cols_wg;
        int col_to_use = inner_lid % im_cols_wg;
        /// max = LOCAL_MEM_SIZE + im_cols_wg
        int lm_offset = row_to_use * im_cols_wg + col_to_use;

        /// out_h*stride_h+256
        int im_y_off = im_y * stride_h + row_to_use;
        /// out_w*stride_w+256
        int im_x_off = im_x * stride_w + col_to_use;

        if(im_y_off >= pad_h && im_y_off < h + pad_h && im_x_off >= pad_w && im_x_off < w + pad_w)
        {
            int im_off_h        = im_y_off - pad_h;
            int im_off_w        = im_x_off - pad_w;
            index_t im_off_id   = (index_t)wg_ch * h * w + im_off_h * w + im_off_w;
            local_im[lm_offset] = IM_OFF_GUARD(im_off_id);
        }
        else
            local_im[lm_offset] = 0;

        inner_lid += 256;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    inner_lid = lid;
    while(inner_lid < out_cols_wg * out_rows_wg)
    {
        int out_x = inner_lid % out_cols_wg; /// < 256
        int out_y = inner_lid / out_cols_wg; /// < 256

        index_t col_x = (index_t)(im_y + out_y) * out_w + im_x + out_x; /// out_h * out_w
        /// c * out_h * out_w * wei_h * wei_w
        index_t col_y = (gid / num_im_blks) * out_h * out_w * wei_h * wei_w;

        for(int y = 0; y < wei_h; y++)
        {
            for(int x = 0; x < wei_w; x++)
            {
                int im_off_h    = out_y * stride_h + y * dilation_h;
                int im_off_w    = out_x * stride_w + x * dilation_w;
                index_t col_off = col_y + col_x + ((index_t)y * wei_w + x) * out_h * out_w;
                col[col_off]    = local_im[(im_off_h)*im_cols_wg + im_off_w];
            }
        }
        inner_lid += 256;
    }
#endif // NUM_IM_BLKS && STRIDE_GT_1
#else

    index_t tid = get_global_id(0);
    while(tid < (index_t)out_h * out_w * wei_w * wei_h * NUM_CH_TOTAL)
    {
        // which row of the output to write to
        index_t col_row = tid / ((index_t)out_h * out_w); // wei_w * wei_h * NUM_CH_TOTAL

        // which pixel from the image and which channel to read from
        int im_x = col_row % wei_w;                    // used to compute im_off_w
        int im_y = (col_row / wei_w) % wei_h;          // used to compute im_off_y
        int im_c = col_row / ((index_t)wei_w * wei_h); // im_c is the img channel

        int out_x = tid % out_w;
        int out_y = (tid / out_w) % out_h;

        // take the strides and padding into account while reading from the image
        int im_off_h = out_y * stride_h - pad_h + im_y * dilation_h;
        int im_off_w = out_x * stride_w - pad_w + im_x * dilation_w;

        index_t col_off = col_row * out_h * out_w + (index_t)out_y * out_w + out_x;

        if(im_off_h >= 0 && im_off_h < h && im_off_w >= 0 && im_off_w < w)
        {
            col[col_off] = IM_OFF_GUARD((index_t)im_c * h * w + im_off_h * w + im_off_w);
        }
        else
        {
            col[col_off] = 0.;
        }
        tid += get_global_size(0);
    }
#endif
}
