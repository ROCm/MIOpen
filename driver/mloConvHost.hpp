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

#ifndef MLO_CONVHOST_H_
#define MLO_CONVHOST_H_

#include <cmath>
#include <iomanip>
#include <iostream>

template <typename _T>
double CalcErr(_T c_val, _T g_val)
{
    double err = 0;
    if(sizeof(_T) == 4)
    {
        int* c_uval = reinterpret_cast<int*>(&c_val);
        int* g_uval = reinterpret_cast<int*>(&g_val);
        err         = static_cast<double>(std::abs(*c_uval - *g_uval));
    }
    else if(sizeof(_T) == 8)
    {
        int64_t* c_uval = reinterpret_cast<int64_t*>(&c_val);
        int64_t* g_uval = reinterpret_cast<int64_t*>(&g_val);
        err             = static_cast<double>(std::abs(*c_uval - *g_uval));
    }

    //		double delta = abs(c_val - g_val);
    //	double nextafter_delta = nextafterf(min(abs(c_val), abs(g_val)), (_T)INFINITY) -
    // min(abs(c_val), abs(g_val));
    //		err = delta / nextafter_delta;
    return err;
}

//#if 0 // disable functions
#if 1
////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////
#define ADNN_MM_TRANSPOSE 1
template <typename Dtype>
void ADNN_mm_cpu(const Dtype* a_ptr,
                 size_t a_cols,
                 size_t a_rows,
                 size_t a_stride,
                 int a_flags,
                 const Dtype* b_ptr,
                 size_t b_cols,
                 size_t b_rows,
                 size_t b_stride,
                 int b_flags,
                 Dtype* c_ptr,
                 size_t c_cols,
                 size_t c_rows,
                 size_t c_stride,
                 int /*c_flags*/,
                 double d_alpha,
                 double d_beta)
{
    // mA

    // mB

    // mC
    Dtype alpha = Dtype(d_alpha);
    Dtype beta  = Dtype(d_beta);
    if((!(a_flags & ADNN_MM_TRANSPOSE) && !(b_flags & ADNN_MM_TRANSPOSE) &&
        ((a_cols != b_rows) || (a_rows != c_rows) || (b_cols != c_cols))) ||
       ((a_flags & ADNN_MM_TRANSPOSE) && (b_flags & ADNN_MM_TRANSPOSE) &&
        ((a_rows != b_cols) || (a_cols != c_rows) || (b_rows != c_cols))) ||
       ((a_flags & ADNN_MM_TRANSPOSE) && !(b_flags & ADNN_MM_TRANSPOSE) &&
        ((a_rows != b_rows) || (a_cols != c_rows) || (b_cols != c_cols))) ||
       (!(a_flags & ADNN_MM_TRANSPOSE) && (b_flags & ADNN_MM_TRANSPOSE) &&
        ((a_cols != b_cols) || (a_rows != c_rows) || (b_rows != c_cols))))
    {
        printf("MM_CPU ERROR; %zd %zd   %zd %zd   %zd %zd\n",
               a_cols,
               a_rows,
               b_cols,
               b_rows,
               c_rows,
               c_cols);
        return;
    }

    size_t inner_loop = (!(a_flags & ADNN_MM_TRANSPOSE)) ? a_cols : a_rows;

    if(!(a_flags & ADNN_MM_TRANSPOSE) && !(b_flags & ADNN_MM_TRANSPOSE))
    {
        for(size_t n = 0; n < c_rows; ++n)
        {
            for(size_t k = 0; k < c_cols; ++k)
            {
                Dtype mm_e = 0;
                for(size_t m = 0; m < inner_loop; ++m)
                {
                    mm_e += a_ptr[n * a_stride + m] * b_ptr[m * b_stride + k];
                }
                c_ptr[n * c_stride + k] = beta * c_ptr[n * c_stride + k] + alpha * mm_e;
            }
        }
    }
    else if((a_flags & ADNN_MM_TRANSPOSE) && !(b_flags & ADNN_MM_TRANSPOSE))
    {
        for(size_t n = 0; n < c_rows; ++n)
        {
            for(size_t k = 0; k < c_cols; ++k)
            {

                Dtype mm_e = 0;
                for(size_t m = 0; m < inner_loop; ++m)
                {
                    mm_e += a_ptr[m * a_stride + n] * b_ptr[m * b_stride + k];
#if 0
					if (
						(n == 0 && k == 33
						|| n == 1 && k == 32
						|| n == 3 && k == 1
						|| n == 4 && k == 0

						)
						&& a_ptr[m*a_stride + n] * b_ptr[m*b_stride + k] != 0
						)
					{
						printf("C:mm:%d %d %d   %11.9f %11.9f %11.9f %11.9f\n",
							n, k, m,
							mm_e, a_ptr[m*a_stride + n], b_ptr[m*b_stride + k], a_ptr[m*a_stride + n] * b_ptr[m*b_stride + k]);
					}
#endif
                }
                c_ptr[n * c_stride + k] = beta * c_ptr[n * c_stride + k] + alpha * mm_e;
            }
        }
    }
    else if(!(a_flags & ADNN_MM_TRANSPOSE) && (b_flags & ADNN_MM_TRANSPOSE))
    {
        for(size_t n = 0; n < c_rows; ++n)
        {
            for(size_t k = 0; k < c_cols; ++k)
            {
                Dtype mm_e = 0;

                for(size_t m = 0; m < inner_loop; ++m)
                {
                    mm_e += a_ptr[n * a_stride + m] * b_ptr[k * b_stride + m];
#if 0
					if (n == 0 && k == 6 && a_ptr[n*a_stride + m] * b_ptr[k*b_stride + m] != 0)
					{
						printf("%4d  %11.9f %11.9f %11.9f\n", m, mm_e, a_ptr[n*a_stride + m], b_ptr[k*b_stride + m]);
					}
#endif
                }
                c_ptr[n * c_stride + k] = beta * c_ptr[n * c_stride + k] + alpha * mm_e;
            }
        }
    }
    else
    {
        for(size_t n = 0; n < c_rows; ++n)
        {
            for(size_t k = 0; k < c_cols; ++k)
            {
                Dtype mm_e = 0;
                for(size_t m = 0; m < inner_loop; ++m)
                {
                    c_ptr[n * c_stride + k] += a_ptr[m * a_stride + n] * b_ptr[k * b_stride + m];
                }
                c_ptr[n * c_stride + k] = beta * c_ptr[n * c_stride + k] + alpha * mm_e;
            }
        }
    }
}

template <typename Dtype>
void ADNN_im2col_cpu(const Dtype* data_im,
                     const int channels,
                     const int height,
                     const int width,
                     const int ksize_h,
                     const int ksize_w,
                     const int pad,
                     const int stride,
                     Dtype* data_col,
                     int stride_col = 0)
{
    int height_col   = (height + 2 * pad - ksize_h) / stride + 1;
    int width_col    = (width + 2 * pad - ksize_w) / stride + 1;
    height_col       = (height_col < 0) ? 1 : height_col;
    width_col        = (width_col < 0) ? 1 : width_col;
    stride_col       = (stride_col == 0) ? height_col * width_col : stride_col;
    int channels_col = channels * ksize_h * ksize_w;
    for(int c = 0; c < channels_col; ++c)
    {
        int w_offset = c % ksize_w;
        int h_offset = (c / ksize_w) % ksize_h;
        int c_im     = c / ksize_h / ksize_w;
        for(int h = 0; h < height_col; ++h)
        {
            for(int w = 0; w < width_col; ++w)
            {
                int h_pad = h * stride - pad + h_offset;
                int w_pad = w * stride - pad + w_offset;
                if(h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                {
                    data_col[c * stride_col + h * width_col + w] =
                        data_im[(c_im * height + h_pad) * width + w_pad];
                }
                else
                {
                    data_col[c * stride_col + h * width_col + w] = 0;
                }
            }
        }
    }
}

template <typename Dtype>
void ADNN_col2im_cpu(const Dtype* data_col,
                     const int channels,
                     const int height,
                     const int width,
                     const int ksize_h,
                     const int ksize_w,
                     const int pad,
                     const int stride,
                     Dtype* data_im)
{
    memset(data_im, 0, sizeof(Dtype) * height * width * channels);
    int height_col   = (height + 2 * pad - ksize_h) / stride + 1;
    int width_col    = (width + 2 * pad - ksize_w) / stride + 1;
    height_col       = (height_col < 0) ? 1 : height_col;
    width_col        = (width_col < 0) ? 1 : width_col;
    int channels_col = channels * ksize_h * ksize_w;
    for(int c = 0; c < channels_col; ++c)
    {
        int w_offset = c % ksize_w;
        int h_offset = (c / ksize_w) % ksize_h;
        int c_im     = c / ksize_h / ksize_w;
        for(int h = 0; h < height_col; ++h)
        {
            for(int w = 0; w < width_col; ++w)
            {
                int h_pad = h * stride - pad + h_offset;
                int w_pad = w * stride - pad + w_offset;
                if(h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                {
                    data_im[(c_im * height + h_pad) * width + w_pad] +=
                        data_col[(c * height_col + h) * width_col + w];
#if 0
					if (c_im == 3 && h_pad == 30 && w_pad == 23)
					{
						printf("C:c2i: %d %d   %d %d %d %d    %14.12f %14.12f\n", c, h * width_col + w, w, h, w_pad, h_pad, data_im[(c_im * height + h_pad) * width + w_pad], data_col[(c * height_col + h) * width_col + w]);
					}
#endif
                }
            }
        }
    }
}

template <typename _T>
int mloConvForwarDirectOnHost(
    _T padding_value, // padding value
    int kernel_size1, // kernel 1 dim
    int pad1,         // padding size
    int stride1,      // scale factor
    int kernel_size0, // kernel 1 dim
    int pad0,         // padding size
    int stride0,      // scale factor
    int n_batchs,
    int n_outputs,
    int n_inputs,
    int top_height,
    int top_width,
    int top_batch_stride,
    int top_channel_stride,
    int top_stride,
    int bot_height,
    int bot_width,
    int bot_batch_stride,
    int bot_channel_stride,
    int bot_stride,
    int weights_stride,
    const _T* bot_ptr, // input "tensor" - batch x channels (input images, feature maps, slices) x
                       // width x height
    _T* top_ptr, // output "te4nsor"  - batch x channels (output images, feature maps, slices) x
                 // width (scaled) x height (scaled)
    const _T*
        weights_ptr, // weights n output channels x n input channels x filter size_y x filter size_x
    const _T* bias_ptr = NULL // bias, NULL if no bias
    )
{
    int ret                   = 0;
    const _T* run_bot_ptr     = bot_ptr;
    _T* run_top_ptr           = top_ptr;
    const _T* run_weights_ptr = weights_ptr;

    // over all batches
    for(int b = 0; b < n_batchs;
        b++, run_bot_ptr += bot_batch_stride, run_top_ptr += top_batch_stride)
    {
        run_weights_ptr = weights_ptr;
        // over all output channels
        for(int o = 0; o < n_outputs; o++)
        {
            // sum up convolutions
            // over output image (scaled input)
            for(int j = 0; j < top_height; j++)
            {
                for(int i = 0; i < top_width; i++)
                {
                    // over all input channels
                    _T accum = 0;
                    for(int c = 0; c < n_inputs; c++)
                    {
                        // do convolution with kernel kernel_size x kerenl_size
                        // with padding - left, right, top, bottom = pad, and value = 0
                        for(int k_j = 0; k_j < kernel_size1; k_j++)
                        {

                            int in_y = (j * stride1 + k_j - pad1);
                            for(int k_i = 0; k_i < kernel_size0; k_i++)
                            {
                                int in_x    = (i * stride0 + k_i - pad0);
                                _T data_val = padding_value;
                                if(!(in_y < 0 || in_x < 0 || in_y >= bot_height ||
                                     in_x >= bot_width))
                                {
                                    int in_data_off =
                                        c * bot_channel_stride + in_y * bot_stride + in_x;
                                    data_val = run_bot_ptr[in_data_off];
                                }

                                _T wei_val = run_weights_ptr[o * weights_stride +
                                                             c * kernel_size1 * kernel_size0 +
                                                             k_j * kernel_size0 + k_i];

                                accum += data_val * wei_val;
#if 0
								if (b == 0 && o == 0 && j == 1 && i == 0)
								{
									printf("c: %f %f %f\n",
										accum/* + bias_ptr[o]*/,
										data_val,
										wei_val
										);
								}
#endif
                            }
                        }
                    }

                    _T final_val = (bias_ptr) ? accum + bias_ptr[o] : accum;
                    run_top_ptr[o * top_channel_stride + j * top_stride + i] = final_val;
                }
            }
        }
    }

    return (ret);
}

template <typename _T>
int mloBackwardMMOnHost(int kernel_size_h,
                        int kernel_size_w,
                        int pad,
                        int stride,
                        const _T* weights_ptr,
                        int weights_height,
                        int weights_width,
                        int weights_stride,
                        const _T* top_df_ptr,
                        int top_height,
                        int top_width,
                        int outputs,
                        int batch_sz,
                        int top_df_batch_stride,
                        int top_df_channel_stride,
                        int /*top_df_stride*/,
                        _T* bot_df_ptr,
                        int bot_height,
                        int bot_width,
                        int inputs,
                        int bot_df_batch_stride,
                        int /*bot_df_channel_stride*/,
                        int /*bot_df_stride*/

                        )
{

    int col_we_df_width     = top_width * top_height;
    int col_we_df_height    = weights_width; // - bias
    int col_we_batch_stride = col_we_df_width * col_we_df_height;
    int col_we_stride       = col_we_df_width;
    _T* col_we_df_ptr       = new _T[col_we_batch_stride * batch_sz];

    assert(col_we_df_ptr);

    for(int b = 0; b < batch_sz; ++b)
    {
        ADNN_mm_cpu<_T>(weights_ptr,
                        weights_width,
                        weights_height,
                        weights_stride,
                        ADNN_MM_TRANSPOSE,
                        &_T(&top_df_ptr[top_df_batch_stride * b]),
                        top_width * top_height,
                        outputs,
                        top_df_channel_stride,
                        0,
                        &col_we_df_ptr[col_we_batch_stride * b],
                        col_we_df_width,
                        col_we_df_height,
                        col_we_stride,
                        0,
                        1,
                        0); //- bias

        ADNN_col2im_cpu<_T>(&col_we_df_ptr[col_we_batch_stride * b],
                            inputs,
                            bot_height,
                            bot_width,
                            kernel_size_h,
                            kernel_size_w,
                            pad,
                            stride,
                            &bot_df_ptr[bot_df_batch_stride * b]);
    }
    if(col_we_df_ptr)
    {
        delete[] col_we_df_ptr;
    }

    return (0);
}

template <typename _T>
int mloBackwardDirectOnHost(
    _T /*padding_value*/, // padding value
    // TO DO: check top, bot dim are equal
    int kernel_size1, // kernel 1 dim
    int pad1,         // padding size
    int stride1,      // scale factor
    int kernel_size0, // kernel 1 dim
    int pad0,         // padding size
    int stride0,      // scale factor
    int n_batchs,
    int n_outputs,
    int n_inputs,
    int top_height,
    int top_width,
    int top_batch_stride,
    int top_channel_stride,
    int top_stride,
    int bot_width,
    int bot_height,
    int bot_batch_stride,
    int bot_channel_stride,
    int bot_stride,
    int weights_stride,
    _T* bot_ptr, // input "tensor" - batch x channels (input images, feature maps, slices) x width x
                 // height
    const _T* top_ptr, // output "te4nsor"  - batch x channels (output images, feature maps, slices)
                       // x width (scaled) x height (scaled)
    const _T*
        weights_ptr // weights n output channels x n input channels x filter size_y x filter size_x
    )
{
    int ret                   = 0;
    _T* run_bot_ptr           = bot_ptr;
    const _T* run_top_ptr     = top_ptr;
    const _T* run_weights_ptr = weights_ptr;

    // over all batches
    for(int b = 0; b < n_batchs;
        b++, run_bot_ptr += bot_batch_stride, run_top_ptr += top_batch_stride)
    {
        run_weights_ptr = weights_ptr;
        // over all output channels
        for(int c = 0; c < n_inputs; ++c)
        {
            // sum up convolutions
            for(int o = 0; o < n_outputs; ++o)
            {

                for(int j = 0; j < top_height; ++j)
                {

                    for(int i = 0; i < top_width; ++i)
                    {

                        int out_data_off = o * top_channel_stride + j * top_stride + i;
                        _T data_val      = run_top_ptr[out_data_off];
                        // over all input channels
                        _T accum = 0;
                        // do convolution with kernel kernel_size x kerenl_size
                        // with padding - left, right, top, bottom = pad, and value = 0
                        for(int k_j = 0; k_j < kernel_size1; ++k_j)
                        {
                            int bot_y = (j * stride1 + k_j - pad1);
                            //	int top_y = (j + kernel_size1 - 1 - k_j);
                            for(int k_i = 0; k_i < kernel_size0; ++k_i)
                            {
                                // int top_x = (i + kernel_size0 - 1 - k_i);
                                int bot_x = (i * stride0 + k_i - pad0);
                                if(!(bot_y < 0 || bot_x < 0 || bot_y >= bot_height ||
                                     bot_x >= bot_width))
                                {
                                    _T wei_val = run_weights_ptr[o * weights_stride +
                                                                 c * kernel_size1 * kernel_size0 +
                                                                 k_j * kernel_size0 + k_i];

                                    int bot_data_off =
                                        c * bot_channel_stride + bot_y * bot_stride + bot_x;
                                    run_bot_ptr[bot_data_off] += data_val * wei_val;

#if 0
									if (b == 0 && o == 0 && bot_y == 0 && bot_x == 0)
									{
										printf("c: %d %d %d %d %d %d  %f %f %f\n",
											bot_data_off,
											k_j,
											k_i,
											j,
											i,
											out_data_off,
											run_bot_ptr[bot_data_off],
											data_val,
											wei_val
										);
									}
#endif
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return (ret);
}

template <typename _T>
void mloPrepad(int o,
               int c,
               int j,
               int i,
               _T padding_value, // padding value
               int pad0,         // padding size
               int pad1,         // padding size
               int bot_batch_stride,
               int bot_channel_stride,
               int bot_stride,
               int bot_height,
               int bot_width,
               int new_bot_batch_stride,
               int new_bot_channel_stride,
               int new_bot_stride,
               int /*new_bot_height*/,
               int /*new_bot_width*/,
               _T* new_bot_ptr,
               const _T* bot_ptr // input "tensor" - batch x channels (input images, feature maps,
                                 // slices) x width x height
               )
{
    int src_j   = j - pad1;
    int src_i   = i - pad0;
    int src_off = o * bot_batch_stride + c * bot_channel_stride + src_j * bot_stride + src_i;
    int dst_off = o * new_bot_batch_stride + c * new_bot_channel_stride + j * new_bot_stride + i;

    if(src_i >= 0 && src_i < bot_width && src_j >= 0 && src_j < bot_height)
    {
        new_bot_ptr[dst_off] = bot_ptr[src_off];
    }
    else
    {
        new_bot_ptr[dst_off] = padding_value;
    }
}

template <typename _T>
int mloDirectSPHostPrepad(_T padding_value, // padding value
                          int pad0,         // padding size
                          int pad1,         // padding size
                          int n_batchs,
                          int n_inputs,
                          int bot_batch_stride,
                          int bot_channel_stride,
                          int bot_stride,
                          int bot_height,
                          int bot_width,
                          int new_bot_batch_stride,
                          int new_bot_channel_stride,
                          int new_bot_stride,
                          int new_bot_height,
                          int new_bot_width,
                          _T* new_bot_ptr,
                          const _T* bot_ptr // input "tensor" - batch x channels (input images,
                                            // feature maps, slices) x width x height
                          )
{
    int ret = 0;
    for(int o = 0; o < n_batchs; ++o)
    {
        for(int c = 0; c < n_inputs; ++c)
        {
            for(int j = 0; j < new_bot_height; ++j)
            {
                for(int i = 0; i < new_bot_width; ++i)
                {
                    mloPrepad<_T>(o,
                                  c,
                                  j,
                                  i,
                                  padding_value, // padding value
                                  pad0,          // padding size
                                  pad1,          // padding size
                                  bot_batch_stride,
                                  bot_channel_stride,
                                  bot_stride,
                                  bot_height,
                                  bot_width,
                                  new_bot_batch_stride,
                                  new_bot_channel_stride,
                                  new_bot_stride,
                                  new_bot_height,
                                  new_bot_width,
                                  new_bot_ptr,
                                  bot_ptr);
                }
            }
        }
    }

    return (ret);
}

/*
weihts interleave
n inputs
        n output blocks (1 block == n output tiles)
                rows 1 * n output tiles
                rows 2 * n output tiles
                rows 3 * n output tiles
*/

template <typename _T>
void mloInterleavelWeightsInOutputs(int c,
                                    int o_block,
                                    int o,
                                    int k_j,
                                    int k_i,
                                    int kernel_size0, // kernel 1 dim
                                    int kernel_size1, // kernel 1 dim
                                    int n_outputs,
                                    int MLO_N_OUT_TILES,
                                    int n_inputs,
                                    const _T* wei_ptr,
                                    _T* new_wei_ptr)
{
    int src_off = (o_block * MLO_N_OUT_TILES + o) * n_inputs * kernel_size1 * kernel_size0 +
                  c * kernel_size1 * kernel_size0 + k_j * kernel_size0 + k_i;
    int dst_off = c * n_outputs * kernel_size1 * kernel_size0 +
                  o_block * MLO_N_OUT_TILES * kernel_size1 * kernel_size0 +
                  MLO_N_OUT_TILES * k_j * kernel_size0 + o * kernel_size0 + k_i;
    new_wei_ptr[dst_off] = wei_ptr[src_off];
}

template <typename _T>
void mloDirectSPHostIntlWeights(bool forward,     // forwad = 1, backward = 0
                                int kernel_size0, // kernel 1 dim
                                int kernel_size1, // kernel 1 dim
                                int n_outputs,
                                int n_inputs,
                                int MLO_N_OUT_TILES,
                                const _T* wei_ptr,
                                _T* new_wei_ptr)
{
    if(forward)
    {
        // interleave all
        // outputs per input
        int o_loops = (n_outputs + MLO_N_OUT_TILES - 1) / MLO_N_OUT_TILES;

        for(int c = 0; c < n_inputs; ++c)
        {
            for(int o_block = 0; o_block < o_loops; ++o_block)
            {
                for(int o = 0; o < MLO_N_OUT_TILES && o_block * MLO_N_OUT_TILES + o < n_outputs;
                    ++o)
                {
                    for(int k_j = 0; k_j < kernel_size1; ++k_j)
                    {
                        for(int k_i = 0; k_i < kernel_size0; ++k_i)
                        {
                            mloInterleavelWeightsInOutputs(c,
                                                           o_block,
                                                           o,
                                                           k_j,
                                                           k_i,
                                                           kernel_size0, // kernel 1 dim
                                                           kernel_size1, // kernel 1 dim
                                                           n_outputs,
                                                           MLO_N_OUT_TILES,
                                                           n_inputs,
                                                           wei_ptr,
                                                           new_wei_ptr);
                        }
                    }
                }
            }
        }
    }
}

template <typename _T>
int mloDirectSPConvHost5x5(int MLO_GRP_SZ1,
                           int MLO_GRP_SZ0,
                           int MLO_OUT_TILE1,
                           int MLO_OUT_TILE0,
                           int MLO_N_OUT_TILES,
                           int MLO_KERNEL_SZ1,   // kernel 1 dim
                           int MLO_FILTER_SIZE1, // kernel 1 dim
                           int MLO_N_BATCHS,
                           int MLO_N_OUTPUTS,
                           int MLO_TOP_BATCH_STRIDE,
                           int MLO_TOP_CHANNEL_STRIDE,
                           int MLO_TOP_STRIDE,
                           int MLO_TOP_HEIGHT,
                           int MLO_TOP_WIDTH,
                           int MLO_N_INPUTS,
                           int MLO_BOT_BATCH_STRIDE,
                           int MLO_BOT_CHANNEL_STRIDE,
                           int MLO_BOT_STRIDE,
                           int /*MLO_BOT_HEIGHT*/,
                           int /*MLO_BOT_WIDTH*/,
                           const _T* bot_ptr, // input "tensor" - batch x channels (input images,
                                              // feature maps, slices) x width x height
                           // interleaved weights
                           const _T* wei_ptr, // weights n output channels x n input channels x
                                              // filter size_y x filter size_x
                           _T* top_ptr, // output "te4nsor"  - batch x channels (output images,
                                        // feature maps, slices) x width (scaled) x height (scaled)
                           const _T* /*bias_ptr = NULL*/ // bias, NULL if no bias

                           )
{
    int j_loops =
        (MLO_TOP_HEIGHT + MLO_GRP_SZ1 * MLO_OUT_TILE1 - 1) / (MLO_GRP_SZ1 * MLO_OUT_TILE1);
    int i_loops = (MLO_TOP_WIDTH + MLO_GRP_SZ0 * MLO_OUT_TILE0 - 1) / (MLO_GRP_SZ0 * MLO_OUT_TILE0);
    int o_loops = (MLO_N_OUTPUTS + MLO_N_OUT_TILES - 1) / MLO_N_OUT_TILES;

    _T* bot_stage = new _T[MLO_OUT_TILE1 * (MLO_OUT_TILE0 + MLO_FILTER_SIZE1 - 1)];
    _T* wei_stage = new _T[MLO_FILTER_SIZE1];
    _T* out_tiles = new _T[MLO_OUT_TILE1 * MLO_OUT_TILE0 * MLO_N_OUT_TILES];

    for(int g2 = 0; g2 < o_loops * MLO_N_BATCHS; ++g2)
    {
        for(int g1 = 0; g1 < j_loops; ++g1)
        {
            for(int g0 = 0; g0 < i_loops; ++g0)
            {
                for(int l1 = 0;
                    l1 < MLO_GRP_SZ1 && (g1 * MLO_GRP_SZ1 + l1) * MLO_OUT_TILE1 < MLO_TOP_HEIGHT;
                    ++l1)
                {
                    for(int l0 = 0;
                        l0 < MLO_GRP_SZ0 && (g0 * MLO_GRP_SZ0 + l0) * MLO_OUT_TILE1 < MLO_TOP_WIDTH;
                        ++l0)
                    {
                        // KERNEL
                        int glbl1   = (g1 * MLO_GRP_SZ1 + l1) * MLO_OUT_TILE1;
                        int glbl0   = (g0 * MLO_GRP_SZ0 + l0) * MLO_OUT_TILE0;
                        int o_block = g2 / MLO_N_BATCHS;
                        int b       = g2 - o_block * MLO_N_BATCHS; // batch
                        // position of on the map of the top-left input pixel
                        // bot stride may include additional padding zeros from prepadding
                        int bot_off = b * MLO_BOT_BATCH_STRIDE + MLO_BOT_STRIDE * glbl1 + glbl0;
                        // weight are interleaved
                        int o_base  = o_block * MLO_N_OUT_TILES;
                        int wei_off = o_base * MLO_KERNEL_SZ1 * MLO_FILTER_SIZE1;
                        for(int ii = 0; ii < MLO_OUT_TILE1 * MLO_OUT_TILE0 * MLO_N_OUT_TILES; ++ii)
                        {
                            out_tiles[ii] = 0;
                        }
                        // the only place where we jump
                        for(int c = 0; c < MLO_N_INPUTS; ++c,
                                bot_off += MLO_BOT_CHANNEL_STRIDE,
                                wei_off += MLO_KERNEL_SZ1 * MLO_FILTER_SIZE1 * MLO_N_OUTPUTS)
                        {
                            int bot_off1 = bot_off;

                            // read first MLO_OUT_TILE1 - 1 lines of input
                            for(int j = 0; j < MLO_OUT_TILE1 - 1; ++j, bot_off1 += MLO_BOT_STRIDE)
                            {
                                for(int i = 0; i < (MLO_OUT_TILE0 + MLO_FILTER_SIZE1 - 1); ++i)
                                {
                                    bot_stage[j * (MLO_OUT_TILE0 + MLO_FILTER_SIZE1 - 1) + i] =
                                        bot_ptr[bot_off1 + i];
                                }
                            }
                            // now all weights are sequencially located
                            // see transformed layout
                            int wei_off1 = wei_off;
                            for(int k_j = 0; k_j < MLO_KERNEL_SZ1;
                                ++k_j, bot_off1 += MLO_BOT_STRIDE)
                            {
                                // insertn new line
                                for(int i = 0; i < (MLO_OUT_TILE0 + MLO_FILTER_SIZE1 - 1); ++i)
                                {
                                    bot_stage[(MLO_OUT_TILE1 - 1) *
                                                  (MLO_OUT_TILE0 + MLO_FILTER_SIZE1 - 1) +
                                              i] = bot_ptr[bot_off1 + i];
                                }
                                // loop over outputs

                                for(int o_i = 0; o_i < MLO_N_OUT_TILES; ++o_i)
                                {

                                    // read filter coeff row
                                    for(int w_i = 0; w_i < MLO_FILTER_SIZE1; ++w_i)
                                    {
                                        // moving along the weights
                                        wei_stage[w_i] = wei_ptr[wei_off1++];
                                    }
                                    // convolve
                                    for(int k_i = 0; k_i < MLO_FILTER_SIZE1; ++k_i)
                                    {
                                        for(int m = 0; m < MLO_OUT_TILE1; ++m)
                                        {
                                            for(int l = 0; l < MLO_OUT_TILE0; ++l)
                                            {
                                                out_tiles[o_i * MLO_OUT_TILE1 * MLO_OUT_TILE0 +
                                                          m * MLO_OUT_TILE0 + l] +=
                                                    bot_stage[m * (MLO_OUT_TILE0 +
                                                                   MLO_FILTER_SIZE1 - 1) +
                                                              k_i + l] *
                                                    wei_stage[k_i];
#if 0
												if (o_i == 1 && l1 == 0 && l0 == 0 && g0==0 && g1==0 && g2==0)
												{
													printf("ek: %f %f %f\n",
														out_tiles[o_i * MLO_OUT_TILE1 * MLO_OUT_TILE0 + m*MLO_OUT_TILE0 + l],
														bot_stage[m * (MLO_OUT_TILE0 + MLO_FILTER_SIZE1 - 1) + k_i + l],
														wei_stage[k_i]
														);
												}
#endif
                                            }
                                        }
                                    }

                                } // for (int o_i = 0; o_i < MLO_N_OUT_TILES; ++o_i, wei_off1 +=
                                  // MLO_FILTER_SIZE1)

                                // move data up
                                for(int up = 0; up < MLO_OUT_TILE1 - 1; ++up)
                                {
                                    for(int r = 0; r < (MLO_OUT_TILE0 + MLO_FILTER_SIZE1 - 1); ++r)
                                    {
                                        bot_stage[up * (MLO_OUT_TILE0 + MLO_FILTER_SIZE1 - 1) + r] =
                                            bot_stage[(up + 1) *
                                                          (MLO_OUT_TILE0 + MLO_FILTER_SIZE1 - 1) +
                                                      r];
                                    }
                                }

                            } // for (int k_j = 0; k_j < MLO_KERNEL_SZ1; ++k_j, bot_off1 +=
                              // MLO_BOT_STRIDE, wei_off += MLO_FILTER_SIZE1 * MLO_N_OUTPUTS *
                              // MLO_N_INPUTS)
                        }     // for (int c = 0; c < MLO_N_INPUTS; ++c, bot_off +=
                              // MLO_BOT_CHANNEL_STRIDE)

                        // output
                        int out_off = b * MLO_TOP_BATCH_STRIDE + o_base * MLO_TOP_CHANNEL_STRIDE +
                                      glbl1 * MLO_TOP_STRIDE + glbl0;
                        for(int o = 0; o < MLO_N_OUT_TILES && o_base + o < MLO_N_OUTPUTS;
                            ++o, out_off += MLO_TOP_CHANNEL_STRIDE)
                        {
                            int out_off1 = out_off;
                            for(int j = 0; j < MLO_OUT_TILE1; ++j, out_off1 += MLO_TOP_STRIDE)
                            {
                                for(int i = 0; i < MLO_OUT_TILE0; ++i)
                                {
                                    top_ptr[out_off1 + i] =
                                        out_tiles[o * MLO_OUT_TILE1 * MLO_OUT_TILE0 +
                                                  j * MLO_OUT_TILE0 + i];
                                }
                            }
                        }

                    } // for (int l0 = 0; l0 < MLO_SPC_GRP0 && (g0*MLO_SPC_GRP0 + l0) *
                      // MLO_OUT_TILE1 < MLO_TOP_WIDTH; ++l0)
                }
            }
        }
    }

    delete[] bot_stage;
    delete[] out_tiles;
    delete[] wei_stage;

    return (0);
}

template <typename _T>
int mloDirectSPHost(

    int MLO_SPC_GRP1,
    int MLO_SPC_GRP0,
    int MLO_OUT_TILE1,
    int MLO_OUT_TILE0,
    int MLO_N_OUT_TILES,

    bool forward,
    bool do_input_copy,
    _T padding_value, // padding value
    // TO DO: check top, bot dim are equal
    int kernel_size0, // kernel 1 dim
    int pad0,         // padding size
    int /*stride0*/,  // scale factor
    int kernel_size1, // kernel 1 dim
    int pad1,         // padding size
    int /*stride1*/,  // scale factor
    int n_batchs,
    int n_outputs,
    int top_batch_stride,
    int top_channel_stride,
    int top_stride,
    int top_height,
    int top_width,
    int n_inputs,
    int bot_batch_stride,
    int bot_channel_stride,
    int bot_stride,
    int bot_height,
    int bot_width,
    int new_bot_batch_stride,
    int new_bot_channel_stride,
    int new_bot_stride,
    int new_bot_height,
    int new_bot_width,
    _T* new_bot_ptr,
    const _T* bot_ptr, // input "tensor" - batch x channels (input images, feature maps, slices) x
                       // width x height

    const _T*
        wei_ptr, // weights n output channels x n input channels x filter size_y x filter size_x
    _T* new_wei_ptr,
    _T* top_ptr, // output "te4nsor"  - batch x channels (output images, feature maps, slices) x
                 // width (scaled) x height (scaled)
    const _T* bias_ptr = NULL // bias, NULL if no bias

    )
{
    int ret = 0;
    if(do_input_copy)
    {
        mloDirectSPHostPrepad<_T>(padding_value, // padding value
                                  pad0,          // padding size
                                  pad1,          // padding size
                                  n_batchs,
                                  n_inputs,
                                  bot_batch_stride,
                                  bot_channel_stride,
                                  bot_stride,
                                  bot_height,
                                  bot_width,
                                  new_bot_batch_stride,
                                  new_bot_channel_stride,
                                  new_bot_stride,
                                  new_bot_height,
                                  new_bot_width,
                                  new_bot_ptr,
                                  bot_ptr // input "tensor" - batch x channels (input images,
                                          // feature maps, slices) x width x height
                                  );
    }

    mloDirectSPHostIntlWeights<_T>(forward,      // forwad = 1, backward = 0
                                   kernel_size0, // kernel 1 dim
                                   kernel_size1, // kernel 1 dim
                                   n_outputs,
                                   n_inputs,
                                   MLO_N_OUT_TILES,
                                   wei_ptr,
                                   new_wei_ptr);

    mloDirectSPConvHost5x5<_T>(MLO_SPC_GRP1,
                               MLO_SPC_GRP0,
                               MLO_OUT_TILE1,
                               MLO_OUT_TILE0,
                               MLO_N_OUT_TILES,
                               kernel_size1,
                               kernel_size0,
                               n_batchs,
                               n_outputs,
                               top_batch_stride,
                               top_channel_stride,
                               top_stride,
                               top_height,
                               top_width,
                               n_inputs,
                               new_bot_batch_stride,
                               new_bot_channel_stride,
                               new_bot_stride,
                               new_bot_height,
                               new_bot_width,

                               new_bot_ptr,
                               new_wei_ptr,
                               top_ptr, // output "te4nsor"  - batch x channels (output images,
                                        // feature maps, slices) x width (scaled) x height (scaled)
                               bias_ptr);
    return (ret);
}

#endif // disable functions

template <typename _T>
bool mloVerify(int n_batchs,
               int n_channels,
               int height,
               int width,
               int c_batch_stride,
               int c_channel_stride,
               int c_stride,
               int g_batch_stride,
               int g_channel_stride,
               int g_stride,
               const _T* c_ptr,
               const _T* g_ptr,
               double eps,
               double max_abs_diff,
               double max_sqr,
               bool get_error_pos
               //	int dir,
               //	std::string name
               )

{

    double sqr_accum = 0;
    _T c_val_err = 0, g_val_err = 0;
    double max_err = max_abs_diff;
    int max_b = 0, max_c = 0, max_i = 0, max_j = 0;

    for(int b = 0; b < n_batchs; ++b)
    {
        for(int c = 0; c < n_channels; ++c)
        {
            for(int j = 0; j < height; ++j)
            {
                for(int i = 0; i < width; ++i)
                {
                    _T c_val = c_ptr[b * c_batch_stride + c * c_channel_stride + j * c_stride + i];
                    _T g_val = g_ptr[b * g_batch_stride + c * g_channel_stride + j * g_stride + i];

                    sqr_accum += (c_val - g_val) * (c_val - g_val);
                    double err = std::abs(c_val - g_val);
                    if(err > max_err)
                    {
                        max_err   = err;
                        c_val_err = c_val;
                        g_val_err = g_val;
                        max_b     = b;
                        max_c     = c;
                        max_i     = i;
                        max_j     = j;
                    }
                }
            }
        }
    }

    sqr_accum =
        std::sqrt(sqr_accum / (static_cast<double>(n_batchs * n_channels * height * width)));

    bool match = true;

    if(sqr_accum > max_sqr || std::isnan(sqr_accum) || !std::isfinite(sqr_accum))
    {
        std::cout << "Sqr error : " << std::fixed << std::setw(15) << std::setprecision(13)
                  << sqr_accum << " Max err: " << std::fixed << std::setw(15)
                  << std::setprecision(13) << max_err << " at " << max_b << ", " << max_c << ", "
                  << max_j << ", " << max_i << " c_v = " << std::fixed << std::setw(14)
                  << std::setprecision(12) << c_val_err << " vs g_v = " << std::fixed
                  << std::setw(14) << std::setprecision(12) << g_val_err << std::endl;

        if(get_error_pos)
        {

            for(int b = 0; b < n_batchs && match; ++b)
            {
                for(int c = 0; c < n_channels && match; ++c)
                {
                    for(int j = 0; j < height && match; ++j)
                    {
                        for(int i = 0; i < width && match; ++i)
                        {
                            _T c_val =
                                c_ptr[b * c_batch_stride + c * c_channel_stride + j * c_stride + i];
                            _T g_val =
                                g_ptr[b * g_batch_stride + c * g_channel_stride + j * g_stride + i];

                            double err = CalcErr<_T>(c_val, g_val);
                            if((err > eps && std::abs(c_val - g_val) > max_abs_diff) ||
                               std::isnan(c_val) || std::isnan(g_val) || !std::isfinite(c_val) ||
                               !std::isfinite(g_val))
                            {
                                std::cout << "Difference : " << err << " too large at " << b << ","
                                          << c << ", " << j << "," << i << " c_v = " << std::fixed
                                          << std::setw(14) << std::setprecision(12) << c_val
                                          << " vs g_v = " << std::fixed << std::setw(14)
                                          << std::setprecision(12) << g_val << std::endl;
                                match = false;
                            }
                        }
                    }
                }
            }
        }
    }

    return (match);
}

#endif
