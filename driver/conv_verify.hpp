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

template <typename T>
void RunBackwardWeightsCPUVerify(std::vector<T>& dwei_host,
                                 std::vector<T>& in,
                                 std::vector<T>& dout,
                                 int in_n,
                                 int in_c,
                                 int in_h,
                                 int in_w,
                                 int in_nstride,
                                 int in_cstride,
                                 int in_hstride,
                                 int in_wstride,
                                 int wei_n,
                                 int wei_c,
                                 int wei_h,
                                 int wei_w,
                                 int wei_nstride,
                                 int wei_cstride,
                                 int wei_hstride,
                                 int wei_wstride,
                                 int out_n,
                                 int out_c,
                                 int out_h,
                                 int out_w,
                                 int out_nstride,
                                 int out_cstride,
                                 int out_hstride,
                                 int out_wstride,
                                 int u,
                                 int v,
                                 int pad_h,
                                 int pad_w
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
#if 1
    std::vector<double> t_wei(wei_n * wei_c * wei_h * wei_w, 0);
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
                                int in_i = x + i * u - pad_h;
                                int in_j = y + j * v - pad_w;

                                if((in_i >= 0) && (in_i < in_h) && (in_j >= 0) && (in_j < in_w))
                                {
                                    t_wei[w * wei_nstride + k * wei_cstride + x * wei_hstride +
                                          y] +=
                                        static_cast<double>(in[o * in_nstride + k * in_cstride +
                                                               in_i * in_hstride + in_j]) *
                                        static_cast<double>(dout[o * out_nstride + w * out_cstride +
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
        dwei_host[i] = static_cast<T>(t_wei[i]);
    }
#endif
#ifdef BACKWARD_WRW_VERIFY_DIRECT_2

    {
        assert(u == 1);
        assert(v == 1);

        std::fill(dwei_host.begin(), dwei_host.end(), (T)0);

        int pad0                  = pad_w;
        int pad1                  = pad_h;
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

        int kernel_size0 = wei_w;
        int kernel_size1 = wei_h;
        int kernel_sz    = kernel_size0 * kernel_size1;

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

                    for(int j = 0, c_j = j - pad1; j < top_height; ++j, ++c_j)
                    {

                        for(int i = 0, c_i = i - pad0; i < top_width; i++, ++c_i)
                        {
                            float top_val = dout[top_df_off + j * top_df_stride + i];

                            for(int k = 0, c_j = j - pad1; k < kernel_size1; ++k, ++c_j)
                            {

                                for(int l = 0, c_i = i - pad0; l < kernel_size0; ++l, ++c_i)
                                {

                                    float bot_val = (c_j >= 0 && c_j < bot_height && c_i >= 0 &&
                                                     c_i < bot_width)
                                                        ? in[bot_off + c_j * bot_stride + c_i]
                                                        : 0;

                                    dwei_host[we_off + k * kernel_size0 + l] += bot_val * top_val;
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
		assert(u == v);
		assert(pad_h == pad_w);

		std::fill(dwei_host.begin(), dwei_host.end(), (T)0);

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
//		int kernel_size = wei_w;

		int pad = pad_w;
		int stride = v;

		// allocate raw data for in, dout, dwei for using im2col/gemm aDNN functions
		T * weights_df_v_ptr = new T[weights_width * weights_height];
		T * top_df_ptr = new T[out_n*out_c*out_h*out_w];
		T * bot_ptr = new T[in_n*in_c*in_h*in_w];

		// copy input (in) into packed
		for (int n = 0; n < in_n; n++)
		{
			for (int c = 0; c < in_c; c++)
			{
				for (int h = 0; h < in_h; h++)
				{
					for (int w = 0; w < in_w; w++)
					{
//						if (mode == miopenTranspose)
//						    bot_ptr[n*in_c*in_h*in_w + c*in_h*in_w + h*in_w + w] = dout[n*in_nstride + c*in_cstride + h*in_hstride + w];
//						else
						    bot_ptr[n*in_c*in_h*in_w + c*in_h*in_w + h*in_w + w] = in[n*in_nstride + c*in_cstride + h*in_hstride + w];
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
//						if (mode == miopenTranspose)
//						    top_df_ptr[n*out_c*out_h*out_w + c*out_h*out_w + h*out_w + w] = in[n*out_nstride + c*out_cstride + h*out_hstride + w];
//						else
						    top_df_ptr[n*out_c*out_h*out_w + c*out_h*out_w + h*out_w + w] = dout[n*out_nstride + c*out_cstride + h*out_hstride + w];
					}
				}
			}
		}

		int im2col_batch_stride = weights_width * top_width * top_height;
		T * im2col_ptr = new T[im2col_batch_stride * batch_sz];

#define ADNN_MM_TRANSPOSE 1
		memset(im2col_ptr, 0, im2col_batch_stride * batch_sz * sizeof(T));
		memset(weights_df_v_ptr, 0, weights_width * weights_height * sizeof(T));
		for (int b = 0; b < batch_sz; ++b)
		{
			ADNN_im2col_cpu<T>((const T*)&bot_ptr[bot_batch_stride * b], inputs,
				bot_height, bot_width, wei_h, wei_w, pad,
				stride, &im2col_ptr[im2col_batch_stride * b]);
			// sum up over mini-batch
			ADNN_mm_cpu<T>((const T*)&top_df_ptr[top_df_batch_stride * b], top_width * top_height, outputs, top_df_channel_stride, 0,
				(const T *)&im2col_ptr[im2col_batch_stride * b], top_width * top_height, weights_width, top_width * top_height, ADNN_MM_TRANSPOSE,
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
