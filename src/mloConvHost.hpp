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


#ifndef MLO_CONVHOST_H_
#define MLO_CONVHOST_H_

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


template<typename _T>
int mloConvForwarDirectOnHost(
	_T padding_value,        // padding value
	// TO DO: check top, bot dim are equal
	int kernel_size0,   // kernel 1 dim 
	int pad0,               // padding size
	int stride0,    // scale factor
	int kernel_size1,   // kernel 1 dim 
	int pad1,               // padding size
	int stride1,    // scale factor
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
	const _T * bot_ptr,			// input "tensor" - batch x channels (input images, feature maps, slices) x width x height
	_T * top_ptr,	// output "te4nsor"  - batch x channels (output images, feature maps, slices) x width (scaled) x height (scaled)
	const _T * weights_ptr,    // weights n output channels x n input channels x filter size_y x filter size_x
	const _T * bias_ptr          // bias
	)
{
	int ret = 0;
	const _T * run_bot_ptr = bot_ptr;
	_T * run_top_ptr = top_ptr;
	const _T * run_weights_ptr = weights_ptr;

	// over all batches
	for (int b = 0; b < n_batchs; b++, run_bot_ptr += bot_batch_stride, run_top_ptr += top_batch_stride)
	{
		run_weights_ptr = weights_ptr;
		// over all output channels
		for (int o = 0; o < n_outputs; o++)
		{
			// sum up convolutions
			// over output image (scaled input)
			for (int j = 0; j < top_height; j++)
			{
				for (int i = 0; i < top_width; i++)
				{
					// over all input channels
					_T accum = 0;
					for (int c = 0; c < n_inputs; c++)
					{
						// do convolution with kernel kernel_size x kerenl_size
						// with padding - left, right, top, bottom = pad, and value = 0
						for (int k_j = 0; k_j < kernel_size1; k_j++)
						{

							int in_y = (j*stride1 + k_j - pad1);
							for (int k_i = 0; k_i < kernel_size0; k_i++)
							{
								int in_x = (i*stride0 + k_i - pad0);
								_T data_val = padding_value;
								if (!(in_y < 0 || in_x < 0 || in_y >= bot_height || in_x >= bot_width))
								{
									int in_data_off = c*bot_channel_stride + in_y * bot_stride + in_x;
									data_val = run_bot_ptr[in_data_off];
								}

								_T wei_val = run_weights_ptr[o*weights_stride + c*kernel_size1 *kernel_size0 + k_j*kernel_size0 + k_i];

								accum += data_val * wei_val;
#if 0
								if (b == 0 && o == 0 && j == 2 && i == 0)
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

					run_top_ptr[o*top_channel_stride + j*top_stride + i] = accum + bias_ptr[o]; // + bias

				}

			}

		}
	}

	return(ret);
}




template<typename _T>
int mloBackwardDirectOnHost(
	_T padding_value,        // padding value
	// TO DO: check top, bot dim are equal
	int kernel_size0,   // kernel 1 dim 
	int pad0,               // padding size
	int stride0,    // scale factor
	int kernel_size1,   // kernel 1 dim 
	int pad1,               // padding size
	int stride1,    // scale factor
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
	_T * bot_ptr,			// input "tensor" - batch x channels (input images, feature maps, slices) x width x height
	const _T * top_ptr,	// output "te4nsor"  - batch x channels (output images, feature maps, slices) x width (scaled) x height (scaled)
	const _T * weights_ptr    // weights n output channels x n input channels x filter size_y x filter size_x
	)
{
	int ret = 0;
	_T * run_bot_ptr = bot_ptr;
	const _T * run_top_ptr = top_ptr;
	const _T * run_weights_ptr = weights_ptr;

	// over all batches
	for (int b = 0; b < n_batchs; b++, run_bot_ptr += bot_batch_stride, run_top_ptr += top_batch_stride)
	{
		run_weights_ptr = weights_ptr;
		// over all output channels
		for (int c = 0; c < n_inputs; ++c)
		{
			// sum up convolutions
			for (int o = 0; o < n_outputs; ++o)
			{

				for (int j = 0; j < top_height; ++j)
				{

					for (int i = 0; i < top_width; ++i)
					{

						int out_data_off = o*top_channel_stride + j * top_stride + i;
						_T data_val = run_top_ptr[out_data_off];
						// over all input channels
						_T accum = 0;
						// do convolution with kernel kernel_size x kerenl_size
						// with padding - left, right, top, bottom = pad, and value = 0
						for (int k_j = 0; k_j < kernel_size1; ++k_j)
						{
							int bot_y = (j*stride1 + k_j - pad1);
							//									int top_y = (j + kernel_size1 - 1 - k_j);
							for (int k_i = 0; k_i < kernel_size0; ++k_i)
							{
								//										int top_x = (i + kernel_size0 - 1 - k_i);
								int bot_x = (i*stride0 + k_i - pad0);
								if (!(bot_y < 0 || bot_x < 0 || bot_y >= bot_height || bot_x >= bot_width))
								{
									_T wei_val = run_weights_ptr[o*weights_stride + c*kernel_size1 *kernel_size0 + k_j*kernel_size0 + k_i];


									int bot_data_off = c*bot_channel_stride + bot_y * bot_stride + bot_x;
									run_bot_ptr[bot_data_off] += data_val * wei_val;

								}

#if 0
								if (b == 0 && o == 1 && j == 2 && i == 2)
								{
									printf("c: %f %f %f\n",
										accum,
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


	return(ret);

}


template<typename _T>
bool mloVerifyConv(
	int n_batchs,
	int n_channels, 
	int height,
	int width,
	int c_batch_stride,
	int c_channel_stride,
	int c_stride,
	int g_batch_stride,
	int g_channel_stride,
	int g_stride,
	const _T *c_ptr,
	const _T *g_ptr,
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

	for (int b = 0; b < n_batchs; ++b)
	{
		for (int c = 0; c < n_channels; ++c)
		{
			for (int j = 0; j < height; ++j)
			{
				for (int i = 0; i < width; ++i)
				{
					_T c_val = c_ptr[b*c_batch_stride + c*c_channel_stride + j*c_stride + i];
					_T g_val = g_ptr[b*g_batch_stride + c*g_channel_stride + j*g_stride + i];

					sqr_accum += (c_val - g_val) * (c_val - g_val);
					double err = CalcErr<_T>(c_val, g_val);
					if (err > max_err)
					{
						max_err = err;
						c_val_err = c_val;
						g_val_err = g_val;
						max_b = b;
						max_c = c;
						max_i = i;
						max_j = j;
					}

				}
			}
		}
	}

	sqr_accum = sqrt(sqr_accum / ((double)n_batchs * n_channels*height *width));

	bool match = true;

	if (sqr_accum > max_sqr || std::isnan(sqr_accum) || !std::isfinite(sqr_accum))
	{
		std::cout << "Sqr error : " << std::fixed << std::setw(15) << std::setprecision(13) << sqr_accum <<
			" Max err: " << std::fixed << std::setw(15) << std::setprecision(13) << max_err << " at " << max_b << ", " << max_c << ", " << max_j << ", " << max_i
			<< " c_v = " << std::fixed << std::setw(14) << std::setprecision(12) << c_val_err
			<< " vs g_v = " << std::fixed << std::setw(14) << std::setprecision(12) << g_val_err
			<< std::endl;


		if (get_error_pos)
		{

			for (int b = 0; b < n_batchs && match; ++b)
			{
				for (int c = 0; c < n_channels && match; ++c)
				{
					for (int j = 0; j < height && match; ++j)
					{
						for (int i = 0; i < width && match; ++i)
						{
							_T c_val = c_ptr[b*c_batch_stride + c*c_channel_stride + j*c_stride + i];
							_T g_val = g_ptr[b*g_batch_stride + c*g_channel_stride + j*g_stride + i];


							double err = CalcErr<_T>(c_val, g_val);
							if ((err > eps && std::abs(c_val - g_val) > max_abs_diff) || std::isnan(c_val) || std::isnan(g_val) || !std::isfinite(c_val) || !std::isfinite(g_val))
							{
								std::cout << "Difference : " << err << " too large at " << b << "," << c << ", " << j << "," << i <<
									" c_v = " << std::fixed << std::setw(14) << std::setprecision(12) << c_val <<
									" vs g_v = " << std::fixed << std::setw(14) << std::setprecision(12) << g_val << std::endl;
								match = false;
							}
						}
					}
				}
			}
		}
	}

	return(match);
}


#endif
