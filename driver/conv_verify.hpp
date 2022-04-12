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
#ifndef GUARD_MIOPEN_CONV_VERIFY_HPP
#define GUARD_MIOPEN_CONV_VERIFY_HPP

#include <cassert>

template <typename _Tgpu /* the data type used in GPU computations (usually half) */,
          typename _Tcheck /* the data type used in CPU checkings (usually double) */>
void RunBackwardWeightsCPUVerify(std::vector<_Tcheck>& dwei_host,
                                 std::vector<_Tgpu>& in,
                                 std::vector<_Tgpu>& dout,
                                 const int in_n,
                                 const int in_c,
                                 const int in_h,
                                 const int in_w,
                                 const int in_nstride,
                                 const int in_cstride,
                                 const int in_hstride,
                                 const int in_wstride,
                                 const int wei_n,
                                 const int wei_c,
                                 const int wei_h,
                                 const int wei_w,
                                 const int wei_nstride,
                                 const int wei_cstride,
                                 const int wei_hstride,
                                 const int wei_wstride,
                                 const int out_n,
                                 const int out_c,
                                 const int out_h,
                                 const int out_w,
                                 const int out_nstride,
                                 const int out_cstride,
                                 const int out_hstride,
                                 const int out_wstride,
                                 const int stride_h,
                                 const int stride_w,
                                 const int pad_h,
                                 const int pad_w,
                                 const int dilation_h,
                                 const int dilation_w
                                 //	, miopenConvolutionMode_t mode
)
{
    assert(in_wstride == 1);
    assert(wei_wstride == 1);
    assert(out_wstride == 1);
#ifdef NDEBUG
    (void)in_wstride;  // -warn
    (void)wei_wstride; // -warn
    (void)out_wstride; // -warn
#endif
    std::vector<_Tcheck> t_wei(wei_n * wei_c * wei_h * wei_w, static_cast<_Tcheck>(0));
    for(int o = 0; o < out_n; o++) // mini-batch size
    {
        for(int w = 0; w < out_c; w++) // out_channels (num filters)
        {
            for(int k = 0; k < in_c; k++) // in_channels (RGB)
            {
                for(int x = 0; x < wei_h; x++) // filter height
                {
                    for(int y = 0; y < wei_w; y++) // filter width
                    {
                        for(int i = 0; i < out_h; i++) // output height
                        {
                            for(int j = 0; j < out_w; j++) // output width
                            {
                                int in_i = x * dilation_h + i * stride_h - pad_h; // vertical
                                int in_j = y * dilation_w + j * stride_w - pad_w; // horizontal

                                if((in_i >= 0) && (in_i < in_h) && (in_j >= 0) && (in_j < in_w))
                                {
                                    t_wei[w * wei_nstride + k * wei_cstride + x * wei_hstride +
                                          y] +=
                                        static_cast<_Tcheck>(in[o * in_nstride + k * in_cstride +
                                                                in_i * in_hstride + in_j]) *
                                        static_cast<_Tcheck>(
                                            dout[o * out_nstride + w * out_cstride +
                                                 i * out_hstride + j]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for(size_t i = 0; i < wei_n * wei_c * wei_h * wei_w; ++i)
    {
        dwei_host[i] = t_wei[i];
    }
#ifdef BACKWARD_WRW_VERIFY_DIRECT_2

    {
        assert(stride_h == 1);
        assert(stride_w == 1);

        std::fill(dwei_host.begin(), dwei_host.end(), (static_cast<_Tcheck>(0));

        int batch_sz              = out_n;
        int outputs               = out_c;
        int inputs                = in_c;
        int top_df_batch_stride   = out_nstride;
        int top_df_channel_stride = out_cstride;
        int top_df_stride         = out_hstride;
        int bot_batch_stride      = in_nstride;
        int bot_channel_stride    = in_cstride;
        int weights_df_v2_stride  = wei_nstride;
        int bot_stride            = in_hstride;

        int filter_size_w = wei_w;
        int filter_size_h = wei_h;
        int kernel_sz    = filter_size_w * filter_size_h;

        int top_height = out_h;
        int top_width  = out_w;
        int bot_height = in_h;
        int bot_width  = in_w;

        for(int b = 0; b < batch_sz; ++b)
        {
            for(int o = 0; o < outputs; ++o)
            {
                for(int c = 0; c < inputs; ++c)
                {
                    int top_df_off = b * top_df_batch_stride + o * top_df_channel_stride;
                    int bot_off    = b * bot_batch_stride + c * bot_channel_stride;
                    int we_off     = o * weights_df_v2_stride + c * kernel_sz;

                    for(int j = 0, c_j = j - pad_h; j < top_height; ++j, ++c_j)
                    {

                        for(int i = 0, c_i = i - pad_w; i < top_width; i++, ++c_i)
                        {
                            _Tcheck top_val =
                                static_cast<_Tcheck>(dout[top_df_off + j * top_df_stride + i]);

                            for(int k = 0, c_j = j - pad_h; k < filter_size_h; ++k, ++c_j)
                            {

                                for(int l = 0, c_i = i - pad_w; l < filter_size_w; ++l, ++c_i)
                                {

                                    _Tcheck bot_val =
                                        (c_j >= 0 && c_j < bot_height && c_i >= 0 &&
                                         c_i < bot_width)
                                            ? static_cast<_Tcheck>(
                                                  in[bot_off + c_j * bot_stride + c_i])
                                            : static_cast<_Tcheck>(0);

                                    dwei_host[we_off + k * filter_size_w + l] += bot_val * top_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

#endif

//#ifdef BACKWARD_WRW_VERIFY_GEMM
#if 0
    {
        assert(stride_h == stride_w);
        assert(pad_h == pad_w);

        std::fill(dwei_host.begin(), dwei_host.end(), static_cast<_Tcheck>(0));

        int batch_sz = out_n;
        int outputs = out_c;
        int inputs = in_c;

        int bot_batch_stride = in_c*in_h*in_w;
        int bot_channel_stride = in_h*in_w;
        int bot_stride = in_w;
        int bot_height = in_h;
        int bot_width = in_w;

        int top_width = out_w;
        int top_height = out_h;
        int top_df_channel_stride = top_width * top_height;
        int top_df_batch_stride = top_df_channel_stride * out_c;

        int weights_width = wei_w * wei_h * wei_c;
        int weights_height = wei_n;
        int weights_df_v_stride = weights_width;
//        int kernel_size = wei_w;

        int pad = pad_w;
        int stride = stride_w;

        // allocate raw data for in, dout, dwei for using im2col/gemm aDNN functions
        _Tcheck * weights_df_v_ptr = new _Tcheck[weights_width * weights_height];
        _Tcheck * top_df_ptr = new _Tcheck[out_n*out_c*out_h*out_w];
        _Tcheck * bot_ptr = new _Tcheck[in_n*in_c*in_h*in_w];

        // copy input (in) into packed
        for (int n = 0; n < in_n; n++)
        {
            for (int c = 0; c < in_c; c++)
            {
                for (int h = 0; h < in_h; h++)
                {
                    for (int w = 0; w < in_w; w++)
                    {
//                        if (mode == miopenTranspose)
//                            bot_ptr[n*in_c*in_h*in_w + c*in_h*in_w + h*in_w + w] = static_cast<_Tcheck>(dout[n*in_nstride + c*in_cstride + h*in_hstride + w]);
//                        else
                        bot_ptr[n*in_c*in_h*in_w + c*in_h*in_w + h*in_w + w] = static_cast<_Tcheck>(in[n*in_nstride + c*in_cstride + h*in_hstride + w]);
                    }
                }
            }
        }

        // copy delta out (dout) into packed
        for (int n = 0; n < out_n; n++)
        {
            for (int c = 0; c < out_c; c++)
            {
                for (int h = 0; h < out_h; h++)
                {
                    for (int w = 0; w < out_w; w++)
                    {
//                        if (mode == miopenTranspose)
//                            top_df_ptr[n*out_c*out_h*out_w + c*out_h*out_w + h*out_w + w] = in[n*out_nstride + c*out_cstride + h*out_hstride + w];
//                        else
                        top_df_ptr[n*out_c*out_h*out_w + c*out_h*out_w + h*out_w + w] = static_cast<_Tcheck>(dout[n*out_nstride + c*out_cstride + h*out_hstride + w]);
                    }
                }
            }
        }

        int im2col_batch_stride = weights_width * top_width * top_height;
        _Tcheck * im2col_ptr = new _Tcheck[im2col_batch_stride * batch_sz];

#define ADNN_MM_TRANSPOSE 1
        memset(im2col_ptr, 0, im2col_batch_stride * batch_sz * sizeof(_Tcheck));
        memset(weights_df_v_ptr, 0, weights_width * weights_height * sizeof(_Tcheck));
        for (int b = 0; b < batch_sz; ++b)
        {
            ADNN_im2col_cpu<_Tcheck>((const _Tcheck*)&bot_ptr[bot_batch_stride * b], inputs,
                bot_height, bot_width, wei_h, wei_w, pad,
                stride, &im2col_ptr[im2col_batch_stride * b]);
            // sum up over mini-batch
            ADNN_mm_cpu<_Tcheck>((const _Tcheck*)&top_df_ptr[top_df_batch_stride * b], top_width * top_height, outputs, top_df_channel_stride, 0,
                (const _Tcheck *)&im2col_ptr[im2col_batch_stride * b], top_width * top_height, weights_width, top_width * top_height, ADNN_MM_TRANSPOSE,
                weights_df_v_ptr, weights_width, weights_height, weights_df_v_stride, 0,
                1, 1);

        }

        // read back packed delta weight
        for (int n = 0; n < wei_n; n++)
        {
            for (int c = 0; c < wei_c; c++)
            {
                for (int h = 0; h < wei_h; h++)
                {
                    for (int w = 0; w < wei_w; w++)
                    {
                        dwei_host[n*wei_nstride + c*wei_cstride + h*wei_hstride + w] = weights_df_v_ptr[n*wei_c*wei_h*wei_w + c*wei_h*wei_w + h*wei_w + w];
                    }
                }
            }
        }

        delete[] im2col_ptr;
        delete[] weights_df_v_ptr;
        delete[] top_df_ptr;
        delete[] bot_ptr;

    }
#else
    (void)in_n;  // -warning
    (void)wei_c; // -warning
    (void)wei_n; // -warning
#endif
}

#endif // GUARD_MIOPEN_CONV_VERIFY_HPP
