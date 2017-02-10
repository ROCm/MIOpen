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



#define MLO_HW_WAVE_ID_SETTING 0

#if MLO_HW_WAVE_ID_SETTING
extern __attribute__((const)) uint __hsail_get_dynwave_id(void);
static inline int getWaveId()
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

inline void ReduceKernel(__local _FLOAT * lcl_blob, __private _FLOAT *weights_accum, int lcl_id, int scan_lcl, int sum_stride, int unit_len, bool debug)
{

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



static inline void  Kahan_summation(__private _FLOAT *sum, __private _FLOAT * c, _FLOAT v)
{
	_FLOAT y = v - *c;    //So far, so good: c is zero.
	_FLOAT t = *sum + y;         //Alas, sum is big, y small, so low-order digits of y are lost.
	*c = (t - *sum) - y;   //(t - sum) recovers the high-order part of y; subtracting y recovers -(low part of y)
	*sum = t;             //Algebraically, c should always be zero. Beware eagerly optimising compilers!
}

/*
	group cooperative read
	read by MLO_READ_UNIT
	handle out of range both horizontally and vertically (by fixed number of veryical reads)

	no guard against number of inputs
*/
static inline void readData(int n, int gbl_data_off, int gbl_data_stride,  const __global _FLOAT * g_data, __private _FLOAT * p_data, bool last_pixel)
{
	for (int j = 0; j < n; ++j)
	{


#if MLO_N_PIXS_OFF > 0

		if (last_pixel)
		{
			for (int i = 0; i < MLO_N_PIXS_OFF; ++i)
			{

				p_data[j*MLO_READ_UNIT + i] = g_data[gbl_data_off + j*gbl_data_stride + i];
			}

			for (int i = MLO_IN_N_PIXS_OFF; i < MLO_READ_UNIT; ++i)
			{
				p_data[j*MLO_READ_UNIT + i] = 0;
			}

		}
		else
#endif
		{
			*(MLO_READ_TYPE*)&p_data[j*MLO_READ_UNIT] = *(__global MLO_READ_TYPE*)&g_data[gbl_data_off + j*gbl_data_stride];
		}



	} // for (int j = 0; j < n; ++j)


}

/*
	core processing loop
	bot - input, from local (1 span)
	top - output diff, from global (array of spans, filters vertical size)

	loop over filter vertical size

*/

static inline void Processing(int k, int c, __private _FLOAT * pvt_accum, __private _FLOAT * bot_dat, __private _FLOAT * top_dat)
{
	for (int j = 0; j < k; ++j)
	{
		for (int i = 0; i < c; ++i)
		{
			for (int n = 0; n < MLO_READ_UINT; ++n)
			{
				pvt_accum[j*MLO_N_IN_MAPS + i] +=
					bot_dat[i*MLO_READ_UINT + n] * top_dat[j*MLO_READ_UINT + n];
			}

		}

	}

}

/*********************************************************************************************************
// wrw algorithm for large filters
// idea:
// split output scan-line on number of spans by the  MLO_IN_TILE0 (2 for example)
// 1 scan-line has ((MLO_OUT_WIDTH + MLO_IN_TILE0 - 1/MLO_IN_TILE0) spans
// group will process MLO_GRP_SZ/((MLO_OUT_WIDTH + MLO_IN_TILE0 - 1/MLO_IN_TILE0) output maps

// alg
// load a block of input map (or full map) into LDS
// loop
// read MLO_FILTER_SIZE1 number of spans from output map into VGPRs (for example 5 *2 = 10)
// read 1 input line for  maps into LDS
// accumulate

// accumulate all spans at the end
// start new loop for the next batch (if defined)
// write out 


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
	// reduction memory.
	// ceil pow2 of the number of wk-items keeping the map
#define MLO_LCL_MEM_SZ (MLO_POW2_MAP_WK_SZ * MLO_OUT_STACKS)

	__local _FLOAT red_mem[MLO_LCL_MEM_SZ];

	// guarnteeing an uniformity over a wave
	int wave_id = getWaveId();
	int lcl_id = get_local_id(0);
	int lcl_wv_id = gePhysLocalId();

	int m_idx_base = get_group_id(0); // map index base

	int ib = get_group_id(1); // batch id

	int c_idx = m_idx_base * (MLO_N_LCL_IN_MAPS * MLO_OUT_STACKS); // input map index

	int o_idx = m_idx_base * (MLO_N_LCL_OUT_MAPS * MLO_OUT_STACKS); // output map index

	int gbl_in_off = c_idx * MLO_IN_CHANNEL_STRIDE + ib * MLO_IN_BATCH_STRIDE;
	int gbl_out_off = o_idx * MLO_OUT_CHANNEL_STRIDE + ib * MLO_OUT_BATCH_STRIDE;

// map id inside group
	int m_idx = iDiv(lcl_id, MLO_MAP_WK_SZ);
// read pixel inside the map
	int p4 = iMod(lcl_id, m_idx, MLO_MAP_WK_SZ);

	bool last_pixel = (p4 == MLO_MAP_WK_SZ - 1);
	bool out_of_range = (m_idx >= MLO_OUT_STACKS);
	bool out_of_range_in = (m_idx + c_idx >= MLO_N_INPUTS) && out_of_range;
	bool out_of_range_out = (m_idx + o_idx >= MLO_N_OUTPUTS) && out_of_range;

	gbl_in_off += m_idx * MLO_IN_CHANNEL_STRIDE + p4*MLO_READ_UNIT;
	gbl_out_off += m_idx * MLO_IN_CHANNEL_STRIDE + p4*MLO_READ_UNIT;

// read guards
	gbl_in_off = (out_of_range_in) ? 0 : gbl_in_off;
	gbl_out_off = (out_of_range_out) ? 0 : gbl_out_off;


#define MLO_TOP_DAT_SZ (MLO_N_LCL_OUT_MAPS * MLO_READ_UNIT)

	__private _FLOAT top_dat[MLO_TOP_DAT_SZ];

	for (int i = 0; i < MLO_TOP_DAT_SZ; ++i)
	{
		top_dat[i] = 0;
	}

#define MLO_BOT_DAT_SZ (MLO_N_LCL_IN_MAPS * MLO_READ_UNIT)

	__private _FLOAT bot_dat[MLO_BOT_DAT_SZ];

	for (int i = 0; i < MLO_BOT_DAT_SZ; ++i)
	{
		bot_dat[i] = 0;
	}

#define MLO_ACCUM_SZ (MLO_N_LCL_OUT_MAPS* MLO_N_LCL_IN_MAPS)

	__private _FLOAT pvt_accum[MLO_ACCUM_SZ];

	for (int i = 0; i < MLO_ACCUM_SZ; ++i)
	{
		pvt_accum[i] = 0;
	}

	for (int i = lcl_id; i < MLO_LCL_MEM_SZ; i += MLO_GRP_SZ)
	{
		red_mem[i] = 0;
	}
	// over all batches

	for (int b = 0;
		b < MLO_N_BATCH_LOOPS;
		++b,
		gbl_in_off += MLO_N_LCL_BATCHS*MLO_IN_BATCH_STRIDE,
		gbl_out_off += MLO_N_LCL_BATCHS*MLO_OUT_BATCH_STRIDE
		)
	{


		int gbl_in_scan_off = gbl_in_off;
		int gbl_out_scan_off = gbl_out_off;


		// read input maps
		readData(MLO_N_LCL_IN_MAPS, gbl_in_scan_off, MLO_IN_CHANNEL_STRIDE, bot, bot_dat, last_pixel);

		// read output maps
		readData(MLO_N_LCL_OUT_MAPS, gbl_out_scan_off, MLO_OUT_CHANNEL_STRIDE, top_df, top_dat, last_pixel);

		Processing(MLO_N_LCL_OUT_MAPS, MLO_N_LCL_IN_MAPS, pvt_accum, bot_dat, top_dat);




// final summation over each filter row
	for (int l = 0; l < MLO_ACCUM_SZ; ++l)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
// write data
		rad_mem[m_idx * MLO_POW2_MAP_WK_SZ + p4] = pvt_accum[l];

// barrier inside
		ReduceKernel(&rad_mem[m_idx * MLO_POW2_MAP_WK_SZ], &pvt_accum[l], lcl_id, p4, MLO_POW2_MAP_WK_SZ, 1, false);

	}


	barrier(CLK_LOCAL_MEM_FENCE);

// output 
// inputs are outputs
// TODO : for more than 1 input
	int wei_df_off = ((ib * MLO_N_OUTPUTS + o_idx + m_idx) * (int)MLO_WEI_BATCH_STRIDE) + ((c_idx + m_idx) * MLO_WEI_CHANNEL_STRIDE);
	
	
	for (int o = 0; o < MLO_N_LCL_OUT_MAPS && p4==0 && m_idx < MLO_OUT_STACKS && o_idx + m_idx + o*MLO_OUT_STACKS < MLO_N_OUTPUTS; ++o)
	{
		for (int c = 0; c < MLO_N_LCL_IN_MAPS && c_idx + m_idx + c*MLO_OUT_STACKS < MLO_N_INPUTS; ++c)
		{
			weights_df[wei_df_off + o*MLO_WEI_BATCH_STRIDE + c*MLO_WEI_CHANNEL_STRIDE] = pvt_accum[o*MLO_N_LCL_IN_MAPS + c];
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
			+= *(__global MLO_UT_READ_TYPE*)&weight_df_tmp[(wei_blk_idx * MLO_WEI_CHANNEL_STRIDE + i* MLO_N_OUTPUTS*MLO_WEI_BATCH_STRIDE)  + wei_idx];
	}

	*(__global MLO_UT_READ_TYPE*)&weights_df[wei_idx0] = *(MLO_UT_READ_TYPE*)pvt_accum_wei;

}
