/*
 * Copyright (c) 2016 AMD Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 */


#define _FLOAT					float
#define _FLOAT2					float2
#define _FLOAT4					float4
#define _FLOAT8					float8

#ifndef FLT_MAX
#define FLT_MAX         3.402823466e+38F        /* max value */
#endif



#define MLO_N_OUT_HORIZ_READS (MLO_IN_WIDTH)
#define MLO_PER_WAVE_READ ((MLO_IN_WIDTH + MLO_N_WAVES - 1) / MLO_N_WAVES)
#define MLO_N_OUT_HORIZ_PIX_READS (MLO_PER_WAVE_READ * MLO_N_WAVES)
#define MLO_N_OUT_VERTICAL_READS (MLO_FILTER_SIZE1)


#define MLO_IN_VERT_READS (1)
// there is an assumption that the scanline fits into LDS
#define MLO_N_IN_HORIZ_PIX_READS (MLO_IN_WIDTH) 
#define MLO_N_IN_HORIZ_READS ((MLO_N_IN_HORIZ_PIX_READS + MLO_READ_UNIT - 1) / MLO_READ_UNIT)
#define MLO_IN_N_PIXS_OFF  (MLO_N_IN_HORIZ_READS*MLO_READ_UNIT - MLO_N_IN_HORIZ_PIX_READS)
#define MLO_IN_LCL_WIDTH (MLO_N_IN_HORIZ_READS * MLO_READ_UNIT)
#define MLO_IN_LCL_HEIGHT MLO_IN_VERT_READS
#define MLO_IN_LCL_SZ (MLO_IN_LCL_WIDTH * MLO_IN_LCL_HEIGHT)
#define MLO_TOTAL_IN_LCL_SZ (MLO_N_LCL_BATCHS*MLO_N_LCL_IN_MAPS*MLO_IN_LCL_SZ)

#define MLO_LCL_SZ (MLO_TOTAL_IN_LCL_SZ)


#define MLO_HW_WAVE_ID_SETTING 0

#if MLO_HW_WAVE_ID_SETTING
extern __attribute__((const)) uint __hsail_get_dynwave_id(void);
inline int getWaveId()
{
	int wave_id = 0;

	wave_id = __hsail_get_dynwave_id();
	wave_id = wave_id & MLO_N_PHYS_WAVES_MASK;
	return(wave_id);
}
#else
inline int getWaveId()
{
	int wave_id = 0;

	wave_id = (get_local_id(0) >> MLO_LG2_PHYS_WAVE_SZ);

	return(wave_id);
}
#endif

inline int gePhysLocalId()
{
	int lcl_wave_id = get_local_id(0) - ((get_local_id(0) >> MLO_LG2_PHYS_WAVE_SZ) << MLO_LG2_PHYS_WAVE_SZ);
	return(lcl_wave_id);
}

inline int iDiv(int v, int d)
{
	int r = (int)((float)v / d + 0.00001f);
	return(r);
}

inline int iMod(int v, int u, int d)
{
	int r = v - mul24((int)u, (int)d);
	return(r);
}

inline void ReduceKernel(__local _FLOAT * lcl_blob, _FLOAT *weights_accum, int lcl_id, int scan_lcl, int sum_stride, int unit_len, bool debug)
{
// read first half
	if (scan_lcl < (sum_stride >> 1))
	{
		for (int i = 0; i < unit_len; ++i)
		{
			weights_accum[i] = lcl_blob[(lcl_id + scan_lcl) * unit_len + i];

		}

	}
// add second half
// appload accumulated value so far
	for (int j = (sum_stride >> 1); j > 0; j >>= 1)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (scan_lcl < j)
		{
			for (int i = 0; i < unit_len; ++i)
			{
				weights_accum[i] += lcl_blob[(lcl_id + j) * unit_len + i];

				lcl_blob[lcl_id * unit_len + i] = weights_accum[i];
			}

		}
	}
}


inline void ReduceKernel64(__local _FLOAT * lcl_blob, _FLOAT *weights_accum, int lcl_id, int scan_lcl, int sum_stride, int unit_len, bool debug)
{
	// read first half
	if (scan_lcl < (sum_stride >> 1))
	{
		for (int i = 0; i < unit_len; ++i)
		{
			weights_accum[i] = lcl_blob[(lcl_id + scan_lcl) * unit_len + i];

		}

	}
	// add second half
	// appload accumulated value so far
	for (int j = (sum_stride >> 1); j > 0; j >>= 1)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (scan_lcl < j)
		{
			for (int i = 0; i < unit_len; ++i)
			{
				weights_accum[i] += lcl_blob[(lcl_id + j) * unit_len + i];

				lcl_blob[lcl_id * unit_len + i] = weights_accum[i];
			}

		}
	}
}


inline void  Kahan_summation(_FLOAT *sum, _FLOAT * c, _FLOAT v)
{
	_FLOAT y = v - *c;    //So far, so good: c is zero.
	_FLOAT t = *sum + y;         //Alas, sum is big, y small, so low-order digits of y are lost.
	*c = (t - *sum) - y;   //(t - sum) recovers the high-order part of y; subtracting y recovers -(low part of y)
	*sum = t;             //Algebraically, c should always be zero. Beware eagerly optimising compilers!
}

inline void  Kahan_summation_tricked(_FLOAT *sum, _FLOAT * c, _FLOAT v, _FLOAT mod)
{
	_FLOAT y = v - *c;    //So far, so good: c is zero.
	_FLOAT t = *sum + y;         //Alas, sum is big, y small, so low-order digits of y are lost.
	*c = (t - *sum) * mod - y;   //(t - sum) recovers the high-order part of y; subtracting y recovers -(low part of y)
	*sum = t;             //Algebraically, c should always be zero. Beware eagerly optimising compilers!
}


inline void Kahan_summation2(_FLOAT *sum, _FLOAT *c, _FLOAT *v, int n)
{
	for (int i = 0; i < n; ++i)
	{
		_FLOAT y = v[i] - c[i];    //So far, so good: c is zero.
		_FLOAT t = sum[i] + y;         //Alas, sum is big, y small, so low-order digits of y are lost.
		c[i] = (t - sum[i]) - y;   //(t - sum) recovers the high-order part of y; subtracting y recovers -(low part of y)
		sum[i] = t;             //Algebraically, c should always be zero. Beware eagerly optimising compilers!
	}
}

/*********************************************************************************************************
// wrw algorithm for large filters
// idea:
// split output line line on number of spans by number of waves
// read MLO_FILTER_SIZE1 number of such spans into SGPS (for example 5 *7 = 35)
// read 1 input line for 64 maps into LDS
//
// alg
//
// accumulate 1 scan of data per wave per each input map per wk-item

**********************************************************************************************************/

__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2)))
__kernel void MLOpenCvBwdWrW(
	const __global _FLOAT * top_df,
	const __global _FLOAT * bot,
	__global _FLOAT * weights_df,
#if MLO_CONV_BIAS
	__global _FLOAT * bias_df,
#endif
	_FLOAT padding_val
)
{


	// input/output tiles + reduce buffer

	__local _FLOAT lcl[(MLO_LCL_SZ)];
	__local _FLOAT * lcl_bot = lcl;


	// guarnteeing an uniformity over a wave
	int wave_id = getWaveId();
	int lcl_id = get_local_id(0);
	int lcl_wv_id = gePhysLocalId();



	int c_idx_base = get_group_id(1); // input map index base

	int o_idx_base = iDiv(get_group_id(2), (MLO_BATCH_SZ / (MLO_N_BATCH_LOOPS*MLO_N_LCL_BATCHS))); // output map index base
	int ib_base = iMod(get_group_id(2), o_idx_base, (MLO_BATCH_SZ / (MLO_N_BATCH_LOOPS*MLO_N_LCL_BATCHS)));

	int ib = ib_base*MLO_N_LCL_BATCHS;

	int c_idx = c_idx_base * MLO_N_LCL_IN_MAPS; // input map index

	int o_idx = o_idx_base * (MLO_N_LCL_OUT_MAPS * MLO_OUT_STACKS); // output map index

	int gbl_in_off = c_idx * MLO_IN_CHANNEL_STRIDE + ib * MLO_IN_BATCH_STRIDE;
	int gbl_out_off = o_idx * MLO_OUT_CHANNEL_STRIDE + ib * MLO_OUT_BATCH_STRIDE;


	__private int top_wave_base[MLO_N_WAVES];

	for (int i = 0; i < MLO_N_WAVES; ++i)
	{
		top_wave_base[i] = i * MLO_PER_WAVE_READ;
	}

	__private _FLOAT bot_dat[MLO_PER_WAVE_READ*MLO_FILTER_SIZE1];

	for (int i = 0; i < MLO_PER_WAVE_READ*MLO_FILTER_SIZE1; ++i)
	{
		bot_dat[i] = 0;
	}

	__private _FLOAT pvt_accum[(MLO_FILTER_SIZE1*MLO_FILTER_SIZE0)];

	for (int i = 0; i < (MLO_FILTER_SIZE1*MLO_FILTER_SIZE0); ++i)
	{
		pvt_accum[i] = 0;
	}


	// zero out LDS
	for (int i = lcl_id; i < (MLO_LCL_SZ); i += MLO_GRP_SZ)
	{
		lcl[i] = 0;
	}

//	barrier(CLK_LOCAL_MEM_FENCE);


#if 0
	if (wei_tl_idx == 4 && o_idx == 0 && c_idx == 0 && (dat_tl_idx == 0 || dat_tl_idx == 1))
	{
		printf("G:p: %d %d %d %d %d\n",
			lcl_id,
			wei_tl_idx1,
			wei_tl_idx0,
			wei_tl1,
			wei_tl0
		);
	}

#endif



	// over all batches

	for (int b = 0;
		b < MLO_N_BATCH_LOOPS;
		++b,
		gbl_in_off += MLO_N_LCL_BATCHS*MLO_IN_BATCH_STRIDE,
		gbl_out_off += MLO_N_LCL_BATCHS*MLO_OUT_BATCH_STRIDE
		)
	{
		int in_y = 0;
		int out_y = 0;


		int gbl_in_scan_off = gbl_in_off;
		int gbl_out_scan_off = gbl_out_off;
		// over all out blocks
		// processing per MLO_N_ALIGNED_OUT_SCAN_BLK output scans


//		barrier(CLK_LOCAL_MEM_FENCE);

		// prefetch output
		for (int j = 0; j < MLO_FILTER_SIZE1 - 1; ++j)
		{
			for (int i = 0; i < MLO_PER_WAVE_READ; ++i)
			{
				bot_dat[j*MLO_PER_WAVE_READ + i] = top_df[gbl_out_scan_off + j * MLO_IN_STRIDE + top_wave_base[wave_id] + i];
			}
		}

		// prolog
		// handling padding

		int sc = 0;
// pad0
		for (; sc < 1; ++sc, gbl_out_scan_off += MLO_OUT_CHANNEL_STRIDE, gbl_in_scan_off += MLO_IN_CHANNEL_STRIDE)
		{
			barrier(CLK_LOCAL_MEM_FENCE);

			for (int p4 = lcl_id; p4 < MLO_N_LCL_IN_MAPS * MLO_N_IN_HORIZ_READS * MLO_IN_VERT_READS;
				p4 += MLO_GRP_SZ)
			{
				__private _FLOAT in_rd_data[MLO_READ_UNIT];

				int c_scan = sc;

				int c = iDiv(p4, (MLO_N_IN_HORIZ_READS * MLO_IN_VERT_READS));
				int c_pix4 = iMod(p4, c, (MLO_N_IN_HORIZ_READS*MLO_IN_VERT_READS));

//				if (c_idx + c < MLO_N_INPUTS)

				{
//					c_scan = iDiv(p4_t, MLO_N_IN_HORIZ_READS);

//					int c_pix4 = iMod(p4_t, c_scan, MLO_N_IN_HORIZ_READS);


#if MLO_IN_N_PIXS_OFF > 0

					if (c_pix4 == MLO_N_IN_HORIZ_READS - 1)
					{
						for (int i = 0; i < MLO_IN_N_PIXS_OFF; ++i)
						{

							in_rd_data[i] = bot[gbl_in_scan_off + c*MLO_IN_CHANNEL_STRIDE + c_scan * MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT + i];
						}

						for (int i = MLO_READ_UNIT - 1; i >= MLO_READ_UNIT - MLO_IN_N_PIXS_OFF; --i)
						{
							in_rd_data[i] = 0;
						}

					}
					else
#endif
					{
						*(MLO_READ_TYPE*)in_rd_data = *(MLO_READ_TYPE*)&bot[gbl_in_scan_off + c*MLO_IN_CHANNEL_STRIDE + c_scan * MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT];
					}


					for (int i = 0; i < MLO_READ_UNIT; ++i)
					{
						int lcl_in_off = c*MLO_IN_LCL_SZ + (c_scan + MLO_FILTER_PAD1)*MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 + c_pix4*MLO_READ_UNIT + i;
						lcl_bot[lcl_in_off] = in_rd_data[i];
#if 0
						if (c_idx + c == 1 && p4_t == 0)
						{
							printf("K:g: %d %f\n",
								lcl_in_off,
								lcl_bot[lcl_in_off]
							);
						}
#endif
					}
				}

			} // for (int p4 = lcl_id; p4 < MLO_N_LCL_IN_MAPS * MLO_N_IN_HORIZ_READS * MLO_IN_VERT_READS;

			barrier(CLK_LOCAL_MEM_FENCE);

// processing
			for (int l = MLO_FILTER_PAD1; l >= 0; --l)
			{

				for (int m = 0; m < MLO_PER_WAVE_READ; ++m)
				{
					for (int n = 0; n < MLO_FILTER_SIZE0; ++n)
					{
						pvt_accum[l*MLO_FILTER_SIZE0 + n]
							// each wk-item process an input
							= lcl_bot[lcl_wv_id * MLO_IN_LCL_WIDTH + wave_id * MLO_PER_WAVE_READ + n + m]
							* bot_dat[l * MLO_PER_WAVE_READ + m];
					}

				}

			}

		}



// pad1
		for (; sc < 2; ++sc, gbl_out_scan_off += MLO_OUT_CHANNEL_STRIDE, gbl_in_scan_off += MLO_IN_CHANNEL_STRIDE)
		{
			barrier(CLK_LOCAL_MEM_FENCE);

			for (int p4 = lcl_id; p4 < MLO_N_LCL_IN_MAPS * MLO_N_IN_HORIZ_READS * MLO_IN_VERT_READS;
				p4 += MLO_GRP_SZ)
			{
				__private _FLOAT in_rd_data[MLO_READ_UNIT];

				int c_scan = sc;
				int c = iDiv(p4, (MLO_N_IN_HORIZ_READS * MLO_IN_VERT_READS));
				int c_pix4 = iMod(p4, c, (MLO_N_IN_HORIZ_READS*MLO_IN_VERT_READS));

				//if (c_idx + c < MLO_N_INPUTS)

				{
					//					c_scan = iDiv(p4_t, MLO_N_IN_HORIZ_READS);

					//					int c_pix4 = iMod(p4_t, c_scan, MLO_N_IN_HORIZ_READS);


#if MLO_IN_N_PIXS_OFF > 0

					if (c_pix4 == MLO_N_IN_HORIZ_READS - 1)
					{
						for (int i = 0; i < MLO_IN_N_PIXS_OFF; ++i)
						{

							in_rd_data[i] = bot[gbl_in_scan_off + c*MLO_IN_CHANNEL_STRIDE + c_scan * MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT + i];
						}

						for (int i = MLO_READ_UNIT - 1; i >= MLO_READ_UNIT - MLO_IN_N_PIXS_OFF; --i)
						{
							in_rd_data[i] = 0;
						}

					}
					else
#endif
					{
						*(MLO_READ_TYPE*)in_rd_data = *(MLO_READ_TYPE*)&bot[gbl_in_scan_off + c*MLO_IN_CHANNEL_STRIDE + c_scan * MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT];
					}


					for (int i = 0; i < MLO_READ_UNIT; ++i)
					{
						int lcl_in_off = c*MLO_IN_LCL_SZ + (c_scan + MLO_FILTER_PAD1)*MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 + c_pix4*MLO_READ_UNIT + i;
						lcl_bot[lcl_in_off] = in_rd_data[i];
#if 0
						if (c_idx + c == 1 && p4_t == 0)
						{
							printf("K:g: %d %f\n",
								lcl_in_off,
								lcl_bot[lcl_in_off]
							);
						}
#endif
					}
				}

			} // for (int p4 = lcl_id; p4 < MLO_N_LCL_IN_MAPS * MLO_N_IN_HORIZ_READS * MLO_IN_VERT_READS;

			barrier(CLK_LOCAL_MEM_FENCE);

			  // processing
			for (int l = MLO_FILTER_PAD1 + 1; l >= 0; --l)
			{

				for (int m = 0; m < MLO_PER_WAVE_READ; ++m)
				{
					for (int n = 0; n < MLO_FILTER_SIZE0; ++n)
					{
						pvt_accum[l*MLO_FILTER_SIZE0 + n]
							// each wk-item process an input
							= lcl_bot[lcl_wv_id * MLO_IN_LCL_WIDTH + wave_id * MLO_PER_WAVE_READ + n + m]
							* bot_dat[l * MLO_PER_WAVE_READ + m];
					}

				}

			}

		}

// generic


		// move in the last output scan


		for (; sc < MLO_OUT_HEIGHT - MLO_FILTER_PAD1; ++sc, gbl_out_scan_off += MLO_OUT_CHANNEL_STRIDE, gbl_in_scan_off += MLO_IN_CHANNEL_STRIDE)
		{

			for (int i = 0; i < MLO_PER_WAVE_READ; ++i)
			{
				bot_dat[(MLO_FILTER_SIZE1 - 1) *MLO_PER_WAVE_READ + i] = top_df[gbl_out_scan_off + (MLO_FILTER_SIZE1 - 1) * MLO_IN_STRIDE + top_wave_base[wave_id] + i];
			}


			barrier(CLK_LOCAL_MEM_FENCE);

			for (int p4 = lcl_id; p4 < MLO_N_LCL_IN_MAPS * MLO_N_IN_HORIZ_READS * MLO_IN_VERT_READS;
				p4 += MLO_GRP_SZ)
			{
				__private _FLOAT in_rd_data[MLO_READ_UNIT];

				int c_scan = sc;
				int c = iDiv(p4, (MLO_N_IN_HORIZ_READS * MLO_IN_VERT_READS));
				int c_pix4 = iMod(p4, c, (MLO_N_IN_HORIZ_READS*MLO_IN_VERT_READS));

				//if (c_idx + c < MLO_N_INPUTS)

				{
					//					c_scan = iDiv(p4_t, MLO_N_IN_HORIZ_READS);

					//					int c_pix4 = iMod(p4_t, c_scan, MLO_N_IN_HORIZ_READS);


#if MLO_IN_N_PIXS_OFF > 0

					if (c_pix4 == MLO_N_IN_HORIZ_READS - 1)
					{
						for (int i = 0; i < MLO_IN_N_PIXS_OFF; ++i)
						{

							in_rd_data[i] = bot[gbl_in_scan_off + c*MLO_IN_CHANNEL_STRIDE + c_scan * MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT + i];
						}

						for (int i = MLO_READ_UNIT - 1; i >= MLO_READ_UNIT - MLO_IN_N_PIXS_OFF; --i)
						{
							in_rd_data[i] = 0;
						}

					}
					else
#endif
					{
						*(MLO_READ_TYPE*)in_rd_data = *(MLO_READ_TYPE*)&bot[gbl_in_scan_off + c*MLO_IN_CHANNEL_STRIDE + c_scan * MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT];
					}


					for (int i = 0; i < MLO_READ_UNIT; ++i)
					{
						int lcl_in_off = c*MLO_IN_LCL_SZ + (c_scan + MLO_FILTER_PAD1)*MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 + c_pix4*MLO_READ_UNIT + i;
						lcl_bot[lcl_in_off] = in_rd_data[i];
#if 0
						if (c_idx + c == 1 && p4_t == 0)
						{
							printf("K:g: %d %f\n",
								lcl_in_off,
								lcl_bot[lcl_in_off]
							);
						}
#endif
					}
				}

			} // for (int p4 = lcl_id; p4 < MLO_N_LCL_IN_MAPS * MLO_N_IN_HORIZ_READS * MLO_IN_VERT_READS;

			  // processing
			for (int l = MLO_FILTER_SIZE1 - 1; l >= 0; --l)
			{

				for (int m = 0; m < MLO_PER_WAVE_READ; ++m)
				{
					for (int n = 0; n < MLO_FILTER_SIZE0; ++n)
					{
						pvt_accum[l*MLO_FILTER_SIZE0 + n]
							// each wk-item process an input
							= lcl_bot[lcl_wv_id * MLO_IN_LCL_WIDTH + wave_id * MLO_PER_WAVE_READ + n + m]
							* bot_dat[l * MLO_PER_WAVE_READ + m];
					}

				}

			}
// move up output
			for (int j = 0; j < MLO_FILTER_SIZE1 - 1; ++j)
			{
				for (int i = 0; i < MLO_PER_WAVE_READ; ++i)
				{
					bot_dat[j*MLO_PER_WAVE_READ + i] = bot_dat[(j+1)*MLO_PER_WAVE_READ + i];
				}
			}


			barrier(CLK_LOCAL_MEM_FENCE);


		}

// epilog 
// handling padding
// pad1
		for (; sc < MLO_OUT_HEIGHT - MLO_FILTER_PAD1 + 1; ++sc, gbl_out_scan_off += MLO_OUT_CHANNEL_STRIDE, gbl_in_scan_off += MLO_IN_CHANNEL_STRIDE)
		{



			barrier(CLK_LOCAL_MEM_FENCE);

			for (int p4 = lcl_id; p4 < MLO_N_LCL_IN_MAPS * MLO_N_IN_HORIZ_READS * MLO_IN_VERT_READS;
				p4 += MLO_GRP_SZ)
			{
				__private _FLOAT in_rd_data[MLO_READ_UNIT];

				int c_scan = sc;
				int c = iDiv(p4, (MLO_N_IN_HORIZ_READS * MLO_IN_VERT_READS));
				int c_pix4 = iMod(p4, c, (MLO_N_IN_HORIZ_READS*MLO_IN_VERT_READS));

				//if (c_idx + c < MLO_N_INPUTS)

				{
					//					c_scan = iDiv(p4_t, MLO_N_IN_HORIZ_READS);

					//					int c_pix4 = iMod(p4_t, c_scan, MLO_N_IN_HORIZ_READS);


#if MLO_IN_N_PIXS_OFF > 0

					if (c_pix4 == MLO_N_IN_HORIZ_READS - 1)
					{
						for (int i = 0; i < MLO_IN_N_PIXS_OFF; ++i)
						{

							in_rd_data[i] = bot[gbl_in_scan_off + c*MLO_IN_CHANNEL_STRIDE + c_scan * MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT + i];
						}

						for (int i = MLO_READ_UNIT - 1; i >= MLO_READ_UNIT - MLO_IN_N_PIXS_OFF; --i)
						{
							in_rd_data[i] = 0;
						}

					}
					else
#endif
					{
						*(MLO_READ_TYPE*)in_rd_data = *(MLO_READ_TYPE*)&bot[gbl_in_scan_off + c*MLO_IN_CHANNEL_STRIDE + c_scan * MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT];
					}


					for (int i = 0; i < MLO_READ_UNIT; ++i)
					{
						int lcl_in_off = c*MLO_IN_LCL_SZ + (c_scan + MLO_FILTER_PAD1)*MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 + c_pix4*MLO_READ_UNIT + i;
						lcl_bot[lcl_in_off] = in_rd_data[i];
#if 0
						if (c_idx + c == 1 && p4_t == 0)
						{
							printf("K:g: %d %f\n",
								lcl_in_off,
								lcl_bot[lcl_in_off]
							);
						}
#endif
					}
				}

			} // for (int p4 = lcl_id; p4 < MLO_N_LCL_IN_MAPS * MLO_N_IN_HORIZ_READS * MLO_IN_VERT_READS;

			  // processing
			for (int l = MLO_FILTER_SIZE1 - 1; l >= MLO_FILTER_PAD1 - 1; --l)
			{

				for (int m = 0; m < MLO_PER_WAVE_READ; ++m)
				{
					for (int n = 0; n < MLO_FILTER_SIZE0; ++n)
					{
						pvt_accum[l*MLO_FILTER_SIZE0 + n]
							// each wk-item process an input
							= lcl_bot[lcl_wv_id * MLO_IN_LCL_WIDTH + wave_id * MLO_PER_WAVE_READ + n + m]
							* bot_dat[l * MLO_PER_WAVE_READ + m];
					}

				}

			} // for (int l = MLO_FILTER_SIZE1 - 1; l >= MLO_FILTER_PAD1 - 1; --l)


			barrier(CLK_LOCAL_MEM_FENCE);


		} // for (; sc < MLO_OUT_HEIGHT - MLO_FILTER_PAD1 + 1; ++sc, gbl_out_scan_off += MLO_OUT_CHANNEL_STRIDE, gbl_in_scan_off += MLO_IN_CHANNEL_STRIDE)



// pad0
		for (; sc < MLO_OUT_HEIGHT; ++sc, gbl_out_scan_off += MLO_OUT_CHANNEL_STRIDE, gbl_in_scan_off += MLO_IN_CHANNEL_STRIDE)
		{



			barrier(CLK_LOCAL_MEM_FENCE);

			for (int p4 = lcl_id; p4 < MLO_N_LCL_IN_MAPS * MLO_N_IN_HORIZ_READS * MLO_IN_VERT_READS;
				p4 += MLO_GRP_SZ)
			{
				__private _FLOAT in_rd_data[MLO_READ_UNIT];

				int c_scan = sc;
				int c = iDiv(p4, (MLO_N_IN_HORIZ_READS * MLO_IN_VERT_READS));
				int c_pix4 = iMod(p4, c, (MLO_N_IN_HORIZ_READS*MLO_IN_VERT_READS));

				//if (c_idx + c < MLO_N_INPUTS)

				{
					//					c_scan = iDiv(p4_t, MLO_N_IN_HORIZ_READS);

					//					int c_pix4 = iMod(p4_t, c_scan, MLO_N_IN_HORIZ_READS);


#if MLO_IN_N_PIXS_OFF > 0

					if (c_pix4 == MLO_N_IN_HORIZ_READS - 1)
					{
						for (int i = 0; i < MLO_IN_N_PIXS_OFF; ++i)
						{

							in_rd_data[i] = bot[gbl_in_scan_off + c*MLO_IN_CHANNEL_STRIDE + c_scan * MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT + i];
						}

						for (int i = MLO_READ_UNIT - 1; i >= MLO_READ_UNIT - MLO_IN_N_PIXS_OFF; --i)
						{
							in_rd_data[i] = 0;
						}

					}
					else
#endif
					{
						*(MLO_READ_TYPE*)in_rd_data = *(MLO_READ_TYPE*)&bot[gbl_in_scan_off + c*MLO_IN_CHANNEL_STRIDE + c_scan * MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT];
					}


					for (int i = 0; i < MLO_READ_UNIT; ++i)
					{
						int lcl_in_off = c*MLO_IN_LCL_SZ + (c_scan + MLO_FILTER_PAD1)*MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 + c_pix4*MLO_READ_UNIT + i;
						lcl_bot[lcl_in_off] = in_rd_data[i];
#if 0
						if (c_idx + c == 1 && p4_t == 0)
						{
							printf("K:g: %d %f\n",
								lcl_in_off,
								lcl_bot[lcl_in_off]
							);
						}
#endif
					}
				}

			} // for (int p4 = lcl_id; p4 < MLO_N_LCL_IN_MAPS * MLO_N_IN_HORIZ_READS * MLO_IN_VERT_READS;

			  // processing
			for (int l = MLO_FILTER_SIZE1 - 1; l >= MLO_FILTER_PAD1; --l)
			{

				for (int m = 0; m < MLO_PER_WAVE_READ; ++m)
				{
					for (int n = 0; n < MLO_FILTER_SIZE0; ++n)
					{
						pvt_accum[l*MLO_FILTER_SIZE0 + n]
							// each wk-item process an input
							= lcl_bot[lcl_wv_id * MLO_IN_LCL_WIDTH + wave_id * MLO_PER_WAVE_READ + n + m]
							* bot_dat[l * MLO_PER_WAVE_READ + m];
					}

				}

			}


			barrier(CLK_LOCAL_MEM_FENCE);


		} // for (; sc < MLO_OUT_HEIGHT - MLO_FILTER_PAD1 + 2; ++sc, gbl_out_scan_off += MLO_OUT_CHANNEL_STRIDE, gbl_in_scan_off += MLO_IN_CHANNEL_STRIDE)


	} // 	for (int b = 0;

// final summation
	for (int l = 0; l < MLO_FILTER_SIZE1; ++l)
	{
		for (int n = 0; n < MLO_FILTER_SIZE0 && wave_id > 0; ++n)
		{
			lcl[((wave_id - 1) * MLO_HW_WAVE_SZ + lcl_wv_id) * MLO_FILTER_SIZE0 + n] =
				pvt_accum[l*MLO_FILTER_SIZE0 + n];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		for(int w = 0; w < MLO_N_WAVES && wave_id == 0; ++w)
		{
	
			for (int n = 0; n < MLO_FILTER_SIZE0; ++n)
			{
				pvt_accum[l*MLO_FILTER_SIZE0 + n]
					+= lcl[(w* MLO_HW_WAVE_SZ + lcl_wv_id) * MLO_FILTER_SIZE0 + n];
					
			}

		}
	}

// output 
// inputs are outputs
	int c = lcl_wv_id;

	int wei_df_off = ((ib * MLO_N_OUTPUTS + o_idx) * (int)MLO_WEI_BATCH_STRIDE)
		// this input channel
		+ mul24((c_idx + c), (int)MLO_WEI_CHANNEL_STRIDE);
	if (wave_id == 0)
	{
		for (int i = 0; i < (MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0); ++i)
		{
			weights_df[wei_df_off + i] = pvt_accum[i];
		}

	}


}


// final reduction kernel
// add filters over batches
__attribute__((reqd_work_group_size(MLO_UT_GRP_SZ0, 1, 1)))
__kernel void MLOpenCvBwdWrW_rdc(
	const __global _FLOAT * weight_df_tmp,
	__global _FLOAT * weights_df
)
{
	int gbl_id = get_global_id(0);
	int wei_idx0 = gbl_id * MLO_UT_READ_UNIT;

	int wei_blk_idx = iDiv(wei_idx0, MLO_WEI_CHANNEL_STRIDE);
	int wei_idx = iMod(wei_idx0, wei_blk_idx, MLO_WEI_CHANNEL_STRIDE);

	_FLOAT pvt_accum_wei[MLO_UT_READ_UNIT];
	for (int i = 0; i < MLO_UT_READ_UNIT; ++i)
	{
		pvt_accum_wei[i] = 0;
	}

	int batch_loop = (MLO_BATCH_SZ + (MLO_N_BATCH_LOOPS*MLO_N_LCL_BATCHS) - 1) / (MLO_N_BATCH_LOOPS*MLO_N_LCL_BATCHS);
	for (int i = 0; i < batch_loop; ++i)
	{
		*(MLO_UT_READ_TYPE*)pvt_accum_wei
			+= *(MLO_UT_READ_TYPE*)&weight_df_tmp[(wei_blk_idx * MLO_WEI_CHANNEL_STRIDE + i* MLO_N_OUTPUTS*MLO_WEI_BATCH_STRIDE)  + wei_idx];
	}

	*(MLO_UT_READ_TYPE*)&weights_df[wei_idx0] = *(MLO_UT_READ_TYPE*)pvt_accum_wei;

}