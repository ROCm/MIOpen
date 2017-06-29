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
__kernel void Col2Im(global float* col,
                     const int col_h,
                     const int col_w,
                     const int wei_h,
                     const int wei_w,
                     const int pad_h,
                     const int pad_w,
                     const int stride_h,
                     const int stride_w,
                     const int height,
                     const int width,
                     global float* im,
                     size_t im_offset)
{
    global float* im_off = im + im_offset;
    int gid              = (int)get_global_id(0);

    int im_ch  = gid / (width * height);
    int im_pix = gid % (width * height);
    int im_h   = (im_pix / width) + pad_h;
    int im_w   = (im_pix % width) + pad_w;

    int start_h = (im_h < wei_h) ? 0 : (im_h - wei_h) / stride_h + 1;
    int end_h   = min(col_h, im_h / stride_h + 1);
    int start_w = (im_w < wei_w) ? 0 : (im_w - wei_w) / stride_w + 1;
    int end_w   = min(col_w, im_w / stride_w + 1);

    int ch_offset = im_ch * col_w * col_h * wei_w * wei_h;
    col += ch_offset;

    float tmp = 0.0f;
    for(int cy = start_h; cy < end_h; cy++)
    {
        for(int cx = start_w; cx < end_w; cx++)
        {
            int col_off_y = cy + ((im_h - cy * stride_h) * wei_w * col_h);
            int col_off_x = cx + ((im_w - cx * stride_w) * col_w * col_h);

            tmp += col[col_off_y * col_w + col_off_x];
        }
    }
    im_off[gid] = tmp;
}
