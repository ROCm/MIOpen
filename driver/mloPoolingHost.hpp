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

#ifndef MLO_POOLINGHOST_H_
#define MLO_POOLINGHOST_H_

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
#endif

#include <cmath>
#include <cstring>
#include <iomanip>

#if 0
template<typename _T>
double CalcErr( _T c_val, _T g_val)
{
	double err = 0;
	if (sizeof(_T) == 4)
	{
		int * c_uval = (int *)&c_val;
		int * g_uval = (int *)&g_val;
		err = (double)std::abs(*c_uval - *g_uval);
	}
	else if (sizeof(_T) == 8)
	{
		int64_t * c_uval = (int64_t *)&c_val;
		int64_t * g_uval = (int64_t *)&g_val;
		err = (double)std::abs(*c_uval - *g_uval);

	}

	//		double delta = abs(c_val - g_val);
	//	double nextafter_delta = nextafterf(min(abs(c_val), abs(g_val)), (_T)INFINITY) - min(abs(c_val), abs(g_val));
	//		err = delta / nextafter_delta;
	return err;
}
#endif

////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////
#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F /* max value */
#endif

#ifndef MLO_POOLING_OP_MAX
#define MLO_POOLING_OP_MAX 0
#define MLO_POOLING_OP_AVE 1
#define MLO_POOLING_OP_STC 2
#endif

template <typename _T>
bool mloPoolingForwardRunHostAndVerify(int pooling_method,
                                       int pad1,
                                       int stride1,
                                       int kernel_size1,
                                       int pad0,
                                       int stride0,
                                       int kernel_size0,
                                       int n_batchs,
                                       int n_outputs,
                                       int bot_height,
                                       int bot_width,
                                       int bot_stride,
                                       int bot_channel_stride,
                                       int bot_batch_stride,
                                       int top_height,
                                       int top_width,
                                       int top_stride,
                                       int top_channel_stride,
                                       int top_batch_stride,
                                       const _T* bot_ptr,
                                       const _T* top_ptr,
                                       bool do_backward,
                                       size_t* mask_ptr,
                                       uint8_t* mask_gpu,
                                       double allowedEps)
{

    bool match = true;

    // c-emulator
    _T res = 0;

    for(int b = 0; b < n_batchs && match; b++)
    {
        for(int o = 0; o < n_outputs && match; o++)
        {
            for(int j = 0; j < top_height && match; j++)
            {
                for(int i = 0; i < top_width && match; i++)
                {
                    // c-emulator
                    if(pooling_method == MLO_POOLING_OP_MAX)
                    {
                        res = -FLT_MAX;
                    }
                    else if(pooling_method == MLO_POOLING_OP_AVE)
                    {
                        res = 0;
                    }

                    int hstart           = j * stride1 - pad1;
                    int wstart           = i * stride0 - pad0;
                    int hend             = std::min(hstart + kernel_size1, bot_height + pad1);
                    int wend             = std::min(wstart + kernel_size0, bot_width + pad0);
                    int pool_size        = (hend - hstart) * (wend - wstart);
                    hstart               = std::max(hstart, 0);
                    wstart               = std::max(wstart, 0);
                    hend                 = std::min(hend, bot_height);
                    wend                 = std::min(wend, bot_width);
                    size_t res_index     = 0;
                    size_t res_index_gpu = 0;
                    bool found           = false;
                    for(int h = hstart; h < hend; ++h)
                    {
                        for(int w = wstart; w < wend; ++w)
                        {
                            if(pooling_method == MLO_POOLING_OP_MAX)
                            {
                                size_t bot_index = b * bot_batch_stride + o * bot_channel_stride +
                                                   h * bot_stride + w;
                                if(bot_ptr[bot_index] > res)
                                {
                                    res           = bot_ptr[bot_index];
                                    res_index     = bot_index;
                                    res_index_gpu = ((h - j * stride1 + pad1) * kernel_size0) +
                                                    (w - i * stride0 + pad0);
                                    found = true;
                                }
                            }
                            else if(pooling_method == MLO_POOLING_OP_AVE)
                            {
#if 0
								if (j == 0 && i == 6)
								{

									printf("c: %d %f %f\n",
										b*bot_batch_stride + o * bot_channel_stride + h * bot_stride + w,
										res,
										bot_ptr[b*bot_batch_stride + o * bot_channel_stride + h * bot_stride + w]
										);
								}
#endif
                                res += bot_ptr[b * bot_batch_stride + o * bot_channel_stride +
                                               h * bot_stride + w];
                            }
                            else
                            {
                                std::cout << "ERROR: unknown operator : layer: pooling."
                                          << std::endl;
                                match = false;
                                continue;
                            }
                        }
                    }
                    // special index value is used to mark top points which has no associated bottom
                    // points
                    if(!found)
                    {
                        res_index     = std::numeric_limits<size_t>::max();
                        res_index_gpu = std::numeric_limits<uint8_t>::max();
                    }
                    if(pooling_method == MLO_POOLING_OP_MAX)
                    {
                        mask_ptr[b * top_batch_stride + o * top_channel_stride + j * top_stride +
                                 i] = res_index;
                        if(do_backward)
                        {
                            uint8_t mg = mask_gpu[b * top_batch_stride + o * top_channel_stride +
                                                  j * top_stride + i];
                            if(mg != res_index_gpu)
                            {
                                std::cout << "Mask mistmatch, gpu " << mg << " cpu "
                                          << res_index_gpu << "(" << res_index << ")" << std::endl;
                                match = false;
                            }
                        }
                    }
                    if(pooling_method == MLO_POOLING_OP_AVE)
                    {
                        res /= pool_size;
                    }
                    _T c_val = res;
                    _T g_val =
                        top_ptr[b * top_batch_stride + o * top_channel_stride + j * top_stride + i];
                    double err = CalcErr<_T>(c_val, g_val);
                    if(err > allowedEps || std::isnan(c_val) || std::isnan(g_val) ||
                       !std::isfinite(c_val) || !std::isfinite(g_val))
                    {
                        std::cout << "Difference " << err << " too large at " << b << ", " << o
                                  << ", " << j << ", " << i << " c_v = " << c_val
                                  << " vs g_val = " << g_val << std::endl;
                        match = false;
                    }
                }
            }
        }
    }

    return (match);
}

template <typename _T>
int mloPoolingBackwardRunHost(int pooling_method,
                              int kernel_size1,
                              int pad1,
                              int stride1,
                              int kernel_size0,
                              int pad0,
                              int stride0,

                              _T* bot_df_v_ptr, // the code assumes that bot_df_v_ptr was zeroed
                              const _T* top_df_ptr,
                              const size_t* mask_ptr,

                              int bot_df_v_batch_stride,
                              int bot_df_v_channel_stride,
                              int bot_df_v_stride,
                              int bot_width,
                              int bot_height,
                              int n_outputs,
                              int n_batchs,

                              int top_df_batch_stride,
                              int top_df_channel_stride,
                              int top_df_stride,
                              int top_width,
                              int top_height)
{

    int ret = 0;

    for(int b = 0; b < n_batchs; b++)
    {
        for(int o = 0; o < n_outputs; o++)
        {
            int bot_df_v_off = b * bot_df_v_batch_stride + o * bot_df_v_channel_stride;
            int top_df_off   = b * top_df_batch_stride + o * top_df_channel_stride;

            if(pooling_method == MLO_POOLING_OP_MAX)
            {
                for(int j = 0; j < top_height; j++)
                {
                    for(int i = 0; i < top_width; i++)
                    {
                        size_t top_idx = top_df_off + j * top_df_stride + i;
                        size_t bot_idx = mask_ptr[top_idx];
                        // skip top points that don't have associated bottom points
                        if(bot_idx == std::numeric_limits<size_t>::max())
                            continue;
                        bot_df_v_ptr[bot_idx] += top_df_ptr[top_idx];
                    }
                }
            }
            else if(pooling_method == MLO_POOLING_OP_AVE)
            {

                for(int j = 0; j < bot_height; j++)
                {
                    for(int i = 0; i < bot_width; i++)
                    {
                        // c-emulator
                        bot_df_v_ptr[bot_df_v_off + j * bot_df_v_stride + i] = 0;
                        int h                                                = j + pad1;
                        int w                                                = i + pad0;
                        int phstart = (h < kernel_size1) ? 0 : (h - kernel_size1) / stride1 + 1;
                        int phend   = std::min(h / stride1 + 1, top_height);
                        int pwstart = (w < kernel_size0) ? 0 : (w - kernel_size0) / stride0 + 1;
                        int pwend   = std::min(w / stride0 + 1, top_width);
                        _T gradient = 0;
                        for(int ph = phstart; ph < phend; ++ph)
                        {
                            for(int pw = pwstart; pw < pwend; ++pw)
                            {
                                // figure out the pooling size
                                int hstart    = ph * stride1 - pad1;
                                int wstart    = pw * stride0 - pad0;
                                int hend      = std::min(hstart + kernel_size1, bot_height + pad1);
                                int wend      = std::min(wstart + kernel_size0, bot_width + pad0);
                                int pool_size = (hend - hstart) * (wend - wstart);
                                gradient +=
                                    top_df_ptr[top_df_off + ph * top_df_stride + pw] / pool_size;
                            }
                        }
                        bot_df_v_ptr[bot_df_v_off + j * bot_df_v_stride + i] = gradient;
                    }
                }
            }
            else
            {
                std::cout << "ERROR: unknown operator : layer: pooling back-propagation."
                          << std::endl;
                continue;
            }
        }
    }
    return (ret);
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif
