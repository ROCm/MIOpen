/**********************************************************************
Copyright (c)2016 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/


#ifndef MLO_POOLINGHOST_H_
#define MLO_POOLINGHOST_H_

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
#define FLT_MAX         3.402823466e+38F        /* max value */
#endif

#ifndef MLO_POOLING_OP_MAX
#define MLO_POOLING_OP_MAX			0
#define MLO_POOLING_OP_AVE			1
#define MLO_POOLING_OP_STC			2
#endif

template<typename _T>
int mloPoolingForwardRunHostAndVerify(
	int pooling_method,
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
	int	top_batch_stride,
	const _T * bot_ptr,
	const _T * top_ptr,
	double allowedEps
	)
{

	int ret = 0;
	int match = 1;

	// c-emulator
	_T res = 0;

	for (int b = 0; b < n_batchs && match; b++)
	{
		for (int o = 0; o < n_outputs && match; o++)
		{
			for (int j = 0; j < top_height && match; j++)
			{
				for (int i = 0; i < top_width && match; i++)
				{
					// c-emulator
					if (pooling_method == MLO_POOLING_OP_MAX)
					{
						res = -FLT_MAX;
					}
					else if (pooling_method == MLO_POOLING_OP_AVE)
					{
						res = 0;
					}

					int hstart = j * stride1 - pad1;
					int wstart = i * stride0 - pad0;
					int hend = std::min(hstart + kernel_size1, bot_height + pad1);
					int wend = std::min(wstart + kernel_size0, bot_width + pad0);
					int pool_size = (hend - hstart) * (wend - wstart);
					hstart = std::max(hstart, 0);
					wstart = std::max(wstart, 0);
					hend = std::min(hend, bot_height);
					wend = std::min(wend, bot_width);
					for (int h = hstart; h < hend; ++h) {
						for (int w = wstart; w < wend; ++w) {
							if (pooling_method == MLO_POOLING_OP_MAX)
							{
								res = std::max(res, bot_ptr[b*bot_batch_stride + o * bot_channel_stride + h * bot_stride + w]);
							}
							else if (pooling_method == MLO_POOLING_OP_AVE)
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
								res +=
									bot_ptr[b*bot_batch_stride + o * bot_channel_stride + h * bot_stride + w];
							}
							else
							{
								std::cout << "ERROR: unknown operator : layer: pooling." << std::endl;
								match = 0;
								continue;
							}
						}
					}
					if (pooling_method == MLO_POOLING_OP_AVE)
					{
						res /= pool_size;
					}
					_T c_val = res;
					_T g_val = top_ptr[b*top_batch_stride + o * top_channel_stride + j * top_stride + i];
					double err = CalcErr<_T>(c_val, g_val);
					if (err > allowedEps || std::isnan(c_val) || std::isnan(g_val) || !std::isfinite(c_val) || !std::isfinite(g_val))
					{
						std::cout << "Difference " << err << " too large at " << b << ", " << o << ", " << j << ", " << i << " c_v = " << c_val << " vs g_val = " << g_val << std::endl;
						match = 0;
					}
				}
			}
		}
	}


	if (match)
	{
		ret = match;
	}
	else
	{
		ret = -1;
	}

	return(ret);

}

template<typename _T>
int mloPoolingBackwardRunHost(
	int pooling_method,
	int kernel_size1,
	int pad1,
	int stride1,
	int kernel_size0,
	int pad0,
	int stride0,

	_T * bot_df_v_ptr,
	const _T * top_df_ptr,
	const _T * bot_ptr,
	const _T * top_ptr,

	int bot_df_v_batch_stride,
	int bot_df_v_channel_stride,
	int bot_df_v_stride,
	int bot_batch_stride,
	int bot_channel_stride,
	int bot_stride,
	int bot_width,
	int bot_height,
	int n_outputs,
	int n_batchs,

	int top_df_batch_stride,
	int top_df_channel_stride,
	int top_df_stride,
	int top_width,
	int top_height,
	int top_batch_stride,
	int top_channel_stride,
	int top_stride

	)
{
	
	int ret = 0;

	for (int b = 0; b < n_batchs; b++)
	{
		for (int o = 0; o < n_outputs; o++)
		{
			int  bot_off = b * bot_batch_stride + o * bot_channel_stride;
			int  bot_df_v_off = b * bot_df_v_batch_stride + o * bot_df_v_channel_stride;
			int  top_df_off = b * top_df_batch_stride + o * top_df_channel_stride;
			int  top_off = b * top_batch_stride + o * top_channel_stride;

			if (pooling_method == MLO_POOLING_OP_MAX)
			{
				memset(&bot_df_v_ptr[bot_df_v_off], 0, bot_height * bot_df_v_stride * sizeof(_T));
				for (int j = 0; j < top_height; j++)
				{
					for (int i = 0; i < top_width; i++)
					{
						int hstart = j * stride1 - pad1;
						int wstart = i * stride0 - pad0;
						int hend = std::min(hstart + kernel_size1, bot_height);
						int wend = std::min(wstart + kernel_size0, bot_width);
						hstart = std::max(hstart, 0);
						wstart = std::max(wstart, 0);
						for (int h = hstart; h < hend; ++h) {
							for (int w = wstart; w < wend; ++w) {
								bot_df_v_ptr[bot_df_v_off + h * bot_df_v_stride + w] +=
									top_df_ptr[top_df_off + j * top_df_stride + i] *
									(bot_ptr[bot_off + h * bot_stride + w] ==
									top_ptr[top_off + j * top_stride + i]);
#if 0
								if (b == 0 && o == 5 && w == 17 && h == 0)
								{
									printf("C:max: %d %d   %13.11f  %13.11f  %13.11f %13.11f\n",
										i, j,
										bot_df_v_ptr[bot_df_v_off + h * bot_df_v_stride + w],
										top_df_ptr[top_df_off + j * top_df_stride + i],
										bot_ptr[bot_off + h * bot_stride + w],
										top_ptr[top_off + j * top_stride + i]
										);
								}
#endif
							}
						}

					}
				}

			}
			else if (pooling_method == MLO_POOLING_OP_AVE)
			{

				for (int j = 0; j < bot_height; j++)
				{
					for (int i = 0; i < bot_width; i++)
					{
						// c-emulator
						bot_df_v_ptr[bot_df_v_off + j * bot_df_v_stride + i] = 0;
						int h = j + pad1;
						int w = i + pad0;
						int phstart = (h < kernel_size1) ? 0 : (h - kernel_size1) / stride1 + 1;
						int phend = std::min(h / stride1 + 1, top_height);
						int pwstart = (w < kernel_size0) ? 0 : (w - kernel_size0) / stride0 + 1;
						int pwend = std::min(w / stride0 + 1, top_width);
						_T gradient = 0;
						for (int ph = phstart; ph < phend; ++ph) {
							for (int pw = pwstart; pw < pwend; ++pw) {
								// figure out the pooling size
								int hstart = ph * stride1 - pad1;
								int wstart = pw * stride0 - pad0;
								int hend = std::min(hstart + kernel_size1, bot_height + pad1);
								int wend = std::min(wstart + kernel_size0, bot_width + pad0);
								int pool_size = (hend - hstart) * (wend - wstart);
								gradient += top_df_ptr[top_df_off + ph * top_df_stride + pw] / pool_size;

#if 0
								if (b == 0 && o == 3 && i == 6 && j == 0)
								{
									printf("C:com: %10.8f %10.8f %10.8f %d\n", gradient, top_ptr[top_off + ph * top_stride + pw] / pool_size, top_ptr[top_off + ph * top_stride + pw], pool_size);
								}

#endif
							}
						}
						bot_df_v_ptr[bot_df_v_off + j * bot_df_v_stride + i] = gradient;
					}
				}
			}
			else
			{
				std::cout << "ERROR: unknown operator : layer: pooling back-propagation." << std::endl;
				continue;
			}


		}
	}
	return(ret);
}

#endif
