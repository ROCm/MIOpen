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



#define MLO_N_OUT_HORIZ_READS ((MLO_OUT_WIDTH + MLO_IN_TILE0 - 1) / MLO_IN_TILE0)
#define MLO_N_SPANS_PER_SCAN (MLO_N_OUT_HORIZ_READS)
#define MLO_N_OUT_HORIZ_PIX_READS (MLO_PER_WAVE_READ * MLO_IN_TILE0)
#define MLO_N_OUT_VERTICAL_READS (MLO_FILTER_SIZE1)



#define MLO_IN_VERT_READS (MLO_IN_HEIGHT)
// there is an assumption that the scanline fits into LDS
#define MLO_N_IN_HORIZ_PIX_READS (MLO_IN_WIDTH) 
#define MLO_N_IN_HORIZ_READS ((MLO_N_IN_HORIZ_PIX_READS + MLO_READ_UNIT - 1) / MLO_READ_UNIT)
#define MLO_IN_N_PIXS_OFF  (MLO_N_IN_HORIZ_READS*MLO_READ_UNIT - MLO_N_IN_HORIZ_PIX_READS)
#define MLO_IN_LCL_WIDTH (MLO_N_IN_HORIZ_READS * MLO_READ_UNIT + 2* MLO_FILTER_PAD0)
#define MLO_IN_LCL_HEIGHT MLO_IN_VERT_READS
#define MLO_IN_LCL_SZ (MLO_IN_LCL_WIDTH * MLO_IN_LCL_HEIGHT)
#define MLO_TOTAL_IN_LCL_SZ (MLO_N_LCL_BATCHS*MLO_N_LCL_IN_MAPS*MLO_IN_LCL_SZ)

#define MLO_WEI_LCL_SZ (MLO_GRP_SZ * MLO_FILTER_SIZE0)
#if MLO_TOTAL_IN_LCL_SZ > MLO_WEI_LCL_SZ
#define MLO_LCL_SZ (MLO_TOTAL_IN_LCL_SZ)
#else
#define MLO_LCL_SZ (MLO_WEI_LCL_SZ)
#endif


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

inline void readInput(int lcl_id, int gbl_in_scan_off, const __global _FLOAT * bot, __local _FLOAT *lcl_bot)
{
	for (int p4 = lcl_id; p4 < MLO_N_LCL_IN_MAPS * MLO_N_IN_HORIZ_READS * MLO_IN_VERT_READS;
		p4 += MLO_GRP_SZ)
	{
		__private _FLOAT in_rd_data[MLO_READ_UNIT];
// TODO : more than 1 input
		int c = 0;
		int c_scan = iDiv(p4, (MLO_N_IN_HORIZ_READS));

		int c_pix4 = iMod(p4, c_scan, (MLO_N_IN_HORIZ_READS));

//		if (c < MLO_N_INPUTS)

		{

#if MLO_IN_N_PIXS_OFF > 0

			if (c_pix4 == MLO_N_IN_HORIZ_READS - 1)
			{
				for (int i = 0; i < MLO_IN_N_PIXS_OFF; ++i)
				{

					in_rd_data[i] = bot[gbl_in_scan_off + c*MLO_IN_CHANNEL_STRIDE + c_scan* MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT + i];
				}

				for (int i = MLO_READ_UNIT - 1; i >= MLO_READ_UNIT - MLO_IN_N_PIXS_OFF; --i)
				{
					in_rd_data[i] = 0;
				}

			}
			else
#endif
			{
				*(MLO_READ_TYPE*)in_rd_data = *(MLO_READ_TYPE*)&bot[gbl_in_scan_off + c*MLO_IN_CHANNEL_STRIDE + c_scan* MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT];
			}

// stack of inputs, each has 1 line
			for (int i = 0; i < MLO_READ_UNIT; ++i)
			{
				int lcl_in_off = c*MLO_IN_LCL_SZ + c_scan* MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 + c_pix4*MLO_READ_UNIT + i;
				lcl_bot[lcl_in_off] = in_rd_data[i];
#if 0
				if (c == 0 && c_scan ==1 && c_pix4 == 0)
				{
					printf("K:g: %d %d %d %d %f\n",
						lcl_id,
						c_scan,
						i,
						lcl_in_off,
						lcl_bot[lcl_in_off]
					);
				}
#endif
			}
		}

	} // for (int p4 = lcl_id; p4 < MLO_N_LCL_IN_MAPS * MLO_N_IN_HORIZ_READS * MLO_IN_VERT_READS;

	barrier(CLK_LOCAL_MEM_FENCE);


}

inline void Processing(int sc, int sc_lcl_off, int top_lim, int bot_lim, __private _FLOAT * pvt_accum, __local _FLOAT * lcl_bot, __private _FLOAT * top_dat)
{
	for (int l = top_lim; l >= bot_lim; --l)
	{
		for (int m = 0; m < MLO_IN_TILE0; ++m)
		{
			for (int n = 0; n < MLO_FILTER_SIZE0; ++n)
			{
				_FLOAT bot_val = lcl_bot[sc_lcl_off + n + m];
				_FLOAT top_val = top_dat[(top_lim - l) * MLO_IN_TILE0 + m];
				pvt_accum[l*MLO_FILTER_SIZE0 + n]
					// each wk-item process an input
					+= bot_val * top_val;
#if 0
				if (/*bot_val * top_val != 0 && */get_global_id(1) == 0 && get_global_id(2) == 0 && get_local_id(0) == 0 && l == 0 && n == 0)
				{
					printf("G: %d %d  %f %f %f %f\n",
						sc,
						sc_lcl_off,
						pvt_accum[l*MLO_FILTER_SIZE0 + n],
						bot_val * top_val,
						bot_val,
						top_val
					);
				}
#endif

			}

		}

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
// 1 span per wk_item, total scanline with MLO_N_SPANS_PER_SCAN spans 
// TODO: more than 1 input
	int o = iDiv(lcl_id, MLO_N_SPANS_PER_SCAN);
	bool scan_lead = (o*MLO_N_SPANS_PER_SCAN == lcl_id);
	int spn = iMod(lcl_id, o, MLO_N_SPANS_PER_SCAN);
#define MLO_TOP_DAT_SZ (MLO_IN_TILE0 * MLO_FILTER_SIZE1)
	int lcl_bot_off = spn * MLO_IN_TILE0;
	int out_wk_item_off = o * MLO_OUT_CHANNEL_STRIDE + lcl_bot_off;
	gbl_out_off += out_wk_item_off;

	__private _FLOAT top_dat[MLO_TOP_DAT_SZ];

	for (int i = 0; i < MLO_TOP_DAT_SZ; ++i)
	{
		top_dat[i] = 0;
	}

#define MLO_ACCUM_SZ (MLO_FILTER_SIZE1*MLO_FILTER_SIZE0)

	__private _FLOAT pvt_accum[MLO_ACCUM_SZ];

	for (int i = 0; i < MLO_ACCUM_SZ; ++i)
	{
		pvt_accum[i] = 0;
	}


	// zero out LDS
	for (int i = lcl_id; i < (MLO_LCL_SZ); i += MLO_GRP_SZ)
	{
		lcl[i] = 0;
	}

//	barrier(CLK_LOCAL_MEM_FENCE);





	// over all batches

	for (int b = 0;
		b < MLO_N_BATCH_LOOPS;
		++b,
		gbl_in_off += MLO_N_LCL_BATCHS*MLO_IN_BATCH_STRIDE,
		gbl_out_off += MLO_N_LCL_BATCHS*MLO_OUT_BATCH_STRIDE
		)
	{
		int in_y = 0;
	//	int out_y = 0;


		int gbl_in_scan_off = gbl_in_off;
		int gbl_out_scan_off = gbl_out_off;
		// over all out blocks
		// processing per MLO_N_ALIGNED_OUT_SCAN_BLK output scans

		barrier(CLK_LOCAL_MEM_FENCE);

		// read input line
		readInput(lcl_id, gbl_in_scan_off, bot, lcl_bot);


		// prefetch output
		for (int j = 0; j < MLO_FILTER_SIZE1 - 1; ++j, gbl_out_scan_off += MLO_OUT_STRIDE)
		{
			int top_df_off = gbl_out_scan_off;
			_FLOAT mask = 1;
#if MLO_FILTER_SIZE1 - 1 > MLO_OUT_HEIGHT
			top_df_off = (j < MLO_OUT_HEIGHT) ? top_df_off : 0;
			mask = (j < MLO_OUT_HEIGHT) ? 1 : 0;
#endif
			for (int i = 0; i < MLO_IN_TILE0; ++i)
			{
				_FLOAT top_val = top_df[top_df_off/* + j * MLO_OUT_STRIDE*/ + i] * mask;
				top_dat[j*MLO_IN_TILE0 + i] = top_val;
			}
		}


		barrier(CLK_LOCAL_MEM_FENCE);

		// prolog
		// handling padding

		int sc = 0;
		int sc_lcl_off = lcl_bot_off;
// pad0

// processing
		Processing(sc, sc_lcl_off, MLO_FILTER_PAD1, 0, pvt_accum, lcl_bot, top_dat);
	//	gbl_out_scan_off += MLO_OUT_STRIDE;
		sc++;
		sc_lcl_off += MLO_IN_LCL_WIDTH;


// pad1
		Processing(sc, sc_lcl_off, MLO_FILTER_PAD1 + 1, 0, pvt_accum, lcl_bot, top_dat);
	//	gbl_out_scan_off += MLO_OUT_STRIDE;
		sc++;
		sc_lcl_off += MLO_IN_LCL_WIDTH;

// generic

		for (; sc < MLO_OUT_HEIGHT - MLO_FILTER_PAD1; ++sc, gbl_out_scan_off += MLO_OUT_STRIDE, sc_lcl_off += MLO_IN_LCL_WIDTH)
		{

			int top_df_off = gbl_out_scan_off;
			_FLOAT mask = 1;

#if MLO_FILTER_SIZE1 > MLO_OUT_HEIGHT
			top_df_off = ((sc + MLO_FILTER_PAD1) < MLO_OUT_HEIGHT) ? top_df_off : 0;
			mask = ((sc + MLO_FILTER_PAD1) < MLO_OUT_HEIGHT) ? 1 : 0;
#endif
			// move in the last output scans
				for (int i = 0; i < MLO_IN_TILE0; ++i)
				{
					top_dat[(MLO_FILTER_SIZE1 -1) *MLO_IN_TILE0 + i] = top_df[top_df_off + i] * mask;
				}


			  // processing
			Processing(sc, sc_lcl_off, MLO_FILTER_SIZE1 - 1, 0, pvt_accum, lcl_bot, top_dat);
// move up output
// !!!! 2 is seleted because compiler cannot handle register allocation properly
			for (int j = 0; j < MLO_FILTER_SIZE1 - 1; ++j)
			{
				for (int i = 0; i < MLO_IN_TILE0; ++i)
				{
					top_dat[j*MLO_IN_TILE0 + i] = top_dat[(j+1)*MLO_IN_TILE0 + i];
				}
			}


		}

// epilog 
// handling padding
// pad1
		for (; sc < MLO_OUT_HEIGHT - MLO_FILTER_PAD1 + 1; ++sc, sc_lcl_off += MLO_IN_LCL_WIDTH)
		{


			  // processing
			Processing(sc, sc_lcl_off, MLO_FILTER_SIZE1 - 1, MLO_FILTER_PAD1 - 1, pvt_accum, lcl_bot, top_dat);
			  // move up output
			for (int j = 0; j < MLO_FILTER_SIZE1 - 2; ++j)
			{
				for (int i = 0; i < MLO_IN_TILE0; ++i)
				{
					top_dat[j*MLO_IN_TILE0 + i] = top_dat[(j + 1)*MLO_IN_TILE0 + i];
				}
			}


		} // for (; sc < MLO_OUT_HEIGHT - MLO_FILTER_PAD1 + 1; ++sc, gbl_out_scan_off += MLO_OUT_CHANNEL_STRIDE, gbl_in_scan_off += MLO_IN_CHANNEL_STRIDE)



// pad0
		for (; sc < MLO_OUT_HEIGHT; ++sc,sc_lcl_off += MLO_IN_LCL_WIDTH)
		{
			// processing
			Processing(sc, sc_lcl_off, MLO_FILTER_SIZE1 - 1, MLO_FILTER_PAD1, pvt_accum, lcl_bot, top_dat);
		} // for (; sc < MLO_OUT_HEIGHT - MLO_FILTER_PAD1 + 2; ++sc, gbl_out_scan_off += MLO_OUT_CHANNEL_STRIDE, gbl_in_scan_off += MLO_IN_CHANNEL_STRIDE)


	} // 	for (int b = 0;


	barrier(CLK_LOCAL_MEM_FENCE);


// final summation
	for (int l = 0; l < MLO_FILTER_SIZE1; ++l)
	{
		for (int n = 0; n < MLO_FILTER_SIZE0; ++n)
		{
			lcl[lcl_id * MLO_FILTER_SIZE0 + n] =
				pvt_accum[l*MLO_FILTER_SIZE0 + n];
#if 0
			if (lcl_wv_id == 0 && l == 2 && n == 3)
			{
				printf("G:s1: %f\n",
					pvt_accum[l*MLO_FILTER_SIZE0 + n]
				);
			}
#endif

		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if (scan_lead)
		{
			for (int s = 0; s < MLO_N_SPANS_PER_SCAN - 1; ++s)
			{

				for (int n = 0; n < MLO_FILTER_SIZE0; ++n)
				{
					pvt_accum[l*MLO_FILTER_SIZE0 + n]
						+= lcl[(lcl_id + s + 1) * MLO_FILTER_SIZE0 + n];
#if 0
					if (lcl_wv_id == 0 && l == 2 && n == 3)
					{
						printf("G:s2: %f %f\n",
							pvt_accum[l*MLO_FILTER_SIZE0 + n],
							lcl[(w* MLO_HW_WAVE_SZ + lcl_wv_id) * MLO_FILTER_SIZE0 + n]
						);
					}
#endif

				}

			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

// output 
// inputs are outputs
// TODO : for more than 1 input
	int c = 0;

	int wei_df_off = ((ib * MLO_N_OUTPUTS + o_idx + o) * (int)MLO_WEI_BATCH_STRIDE)
		// this input channel
		+ mul24((c_idx + c), (int)MLO_WEI_CHANNEL_STRIDE);
	if (scan_lead && o_idx + o < MLO_N_OUTPUTS)
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