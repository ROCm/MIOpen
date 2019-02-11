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
#ifndef GUARD_MIOPEN_UTIL_DRIVER_HPP
#define GUARD_MIOPEN_UTIL_DRIVER_HPP

// Tin is data type of input data,
// Tout is data type of output data
template <typename Tin, typename Tout>
void Im2ColCPU(std::vector<Tin>& in,
               const size_t in_offset,
               const int in_c,
               const int in_h,
               const int in_w,
               const int wei_h,
               const int wei_w,
               const int out_h,
               const int out_w,
               const int pad_h,
               const int pad_w,
               const int stride_h,
               const int stride_w,
               std::vector<Tout>& col)
{
    int col_m = in_c * wei_h * wei_w;

    auto in_iter = in.begin() + in_offset;

    for(int n = 0; n < col_m; n++)
    {
        int x  = n % wei_w;
        int y  = (n / wei_w) % wei_h;
        int ch = n / (wei_w * wei_h);

        for(int h = 0; h < out_h; h++)
        {
            for(int w = 0; w < out_w; w++)
            {
                int in_off_h = h * stride_h - pad_h + y;
                int in_off_w = w * stride_w - pad_w + x;

                if(in_off_h >= 0 && in_off_h < in_h && in_off_w >= 0 && in_off_w < in_w)
                    col[n * out_h * out_w + h * out_w + w] =
                        static_cast<Tout>(in_iter[ch * in_h * in_w + in_off_h * in_w + in_off_w]);
                else
                    col[n * out_h * out_w + h * out_w + w] = static_cast<Tout>(0);
            }
        }
    }
}

// Tin is data type of input data,
// Tout is data type of output data
template <typename Tin, typename Tout>
void Col2ImCPU(std::vector<Tin> data_col,
               const int channels,
               const int height,
               const int width,
               const int ksize,
               const int pad,
               const int stride,
               std::vector<Tout> data_im)
{
    memset(data_im, 0, sizeof(Tout) * height * width * channels);
    int height_col   = (height + 2 * pad - ksize) / stride + 1;
    int width_col    = (width + 2 * pad - ksize) / stride + 1;
    height_col       = (height_col < 0) ? 1 : height_col;
    width_col        = (width_col < 0) ? 1 : width_col;
    int channels_col = channels * ksize * ksize;
    for(int c = 0; c < channels_col; ++c)
    {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im     = c / ksize / ksize;
        for(int h = 0; h < height_col; ++h)
        {
            for(int w = 0; w < width_col; ++w)
            {
                int h_pad = h * stride - pad + h_offset;
                int w_pad = w * stride - pad + w_offset;
                if(h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                {
                    data_im[(c_im * height + h_pad) * width + w_pad] +=
                        static_cast<Tout>(data_col[(c * height_col + h) * width_col + w]);
                }
            }
        }
    }
}

#endif // GUARD_MIOPEN_UTIL_DRIVER_HPP
