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
static inline void readData(int n, int gbl_data_off, int gbl_data_stride, int map_stride, int map_base,  int map_limit, const __global _FLOAT * g_data, __private _FLOAT * p_data, bool last_pixel)
{
	for (int j = 0; j < n; ++j)
	{
		int gbl_data_off0 = (j*map_stride + map_base < map_limit) ? gbl_data_off + j*map_stride*gbl_data_stride : 0;

#if MLO_N_PIXS_OFF > 0

		if (last_pixel)
		{
			for (int i = 0; i < MLO_N_PIXS_OFF; ++i)
			{

				p_data[j*MLO_READ_UNIT + i] = g_data[gbl_data_off0 + i];
			}

			for (int i = MLO_IN_N_PIXS_OFF; i < MLO_READ_UNIT; ++i)
			{
				p_data[j*MLO_READ_UNIT + i] = 0;
			}

		}
		else
#endif
		{
			for (int i = 0; i < MLO_READ_UNIT; ++i)
			{
				p_data[j*MLO_READ_UNIT + i] = g_data[gbl_data_off0 + i];
			}
		}



	} // for (int j = 0; j < n; ++j)

}

static inline void readDataFlex(int n, int gbl_data_off, int gbl_data_stride, int map_stride, int map_base,  int map_limit, const __global _FLOAT * g_data, __local _FLOAT * l_data)
{


	for( int l = get_local_id(0); l < n*map_stride* MLO_MAP_WK_SZ; l += MLO_GRP_SZ)
	{

	    int r = 0;
		int k0 = l;
#if MLO_N_LCL_OUT_MAPS > 1
		r = iDiv(l, map_stride* MLO_MAP_WK_SZ);  // maps row
		k0 = iMod(l, r, map_stride* MLO_MAP_WK_SZ);
#endif
		int k = iDiv(k0, MLO_MAP_WK_SZ);  // map column
		int p4 = iMod(k0, k, MLO_MAP_WK_SZ); // pixel block

		bool last_pixel = (p4 == MLO_MAP_WK_SZ -1);

		int gbl_data_off0 = (r*map_stride + k + map_base < map_limit) ? gbl_data_off + (r*map_stride + k)*gbl_data_stride + p4*MLO_READ_UNIT : 0;
		__private _FLOAT p_data[MLO_READ_UNIT];

#if MLO_N_PIXS_OFF > 0

		if (last_pixel)
		{
			for (int i = 0; i < MLO_N_PIXS_OFF; ++i)
			{

				p_data[i] = g_data[gbl_data_off0 + i];
			}

			for (int i = MLO_IN_N_PIXS_OFF; i < MLO_READ_UNIT; ++i)
			{
				p_data[i] = 0;
			}

		}
		else
#endif
		{
//			*(MLO_READ_TYPE*)&p_data[j*MLO_READ_UNIT] = *(__global MLO_READ_TYPE*)&g_data[gbl_data_off + j*gbl_data_stride*MLO_OUT_STACKS];
			for (int i = 0; i < MLO_READ_UNIT; ++i)
			{
				p_data[i] = g_data[gbl_data_off0 + i];
			}
		}

		for(int i = 0; i < MLO_READ_UNIT; ++i)
		{
			l_data[(k*MLO_MAP_WK_SZ + p4)*MLO_READ_UNIT + i] = p_data[i];
		}

	}

}



/*********************************************************************************************************
// wrw algorithm for large filters
// idea:
// read MLO_OUT_STACKS per group, MLO_N_LCL_IN_MAPS per wk_item input maps
// read MLO_OUT_STACKS per group, MLO_N_LCL_OUT_MAPS per wk_item output maps

// alg
// loop in MLO_N_LCL_OUT_MAPS
// load MLO_OUT_STACKS of output into LDS
loop in MLO_OUT_STACKS
// convolve with MLO_N_LCL_IN_MAPS per wk-item

// reduce

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
#if 0 //(MLO_POW2_MAP_WK_SZ * MLO_OUT_STACKS) > (MLO_OUT_STACKS * MLO_MAP_WK_SZ * MLO_READ_UNIT)
#define MLO_LCL_MEM_SZ (MLO_POW2_MAP_WK_SZ * MLO_OUT_STACKS)
//#else
#define MLO_LCL_MEM_SZ (MLO_OUT_STACKS * MLO_MAP_WK_SZ * MLO_READ_UNIT)
#endif

	__local _FLOAT lcl_mem[MLO_LCL_MEM_SZ];
	__local _FLOAT * red_mem = lcl_mem;
	__local _FLOAT * proc_mem = lcl_mem;

	// guarnteeing an uniformity over a wave
	int wave_id = getWaveId();
	int lcl_id = get_local_id(0);
	int lcl_wv_id = gePhysLocalId();

	int k_idx = get_group_id(0) * (MLO_OUT_STACKS * MLO_N_LCL_OUT_MAPS); // output map index base

	int c_idx = get_group_id(1) * (MLO_IN_STACKS * MLO_N_LCL_IN_MAPS); // input map index based

	int ib = get_group_id(2); // batch id


	int gbl_in_off = c_idx * MLO_IN_CHANNEL_STRIDE + ib * MLO_IN_BATCH_STRIDE;
	int gbl_out_off = k_idx * MLO_OUT_CHANNEL_STRIDE + ib * MLO_OUT_BATCH_STRIDE;

	// map id inside group
	int m_idx = iDiv(lcl_id, MLO_MAP_WK_SZ);
	// read pixel inside the map
	int p4 = iMod(lcl_id, m_idx, MLO_MAP_WK_SZ);

	bool last_pixel = (p4 == MLO_MAP_WK_SZ - 1);
//	bool out_of_range = false; // (m_idx >= MLO_OUT_STACKS);
//	bool out_of_range_in = (m_idx + c_idx >= MLO_N_INPUTS);
//	bool out_of_range_out = (m_idx + k_idx >= MLO_N_OUTPUTS);

	gbl_in_off += m_idx * MLO_IN_CHANNEL_STRIDE + p4*MLO_READ_UNIT;
//	gbl_out_off += m_idx * MLO_OUT_CHANNEL_STRIDE + p4*MLO_READ_UNIT;

	// read guards
//	gbl_in_off = (out_of_range_in || out_of_range) ? 0 : gbl_in_off;
//	gbl_out_off = (out_of_range_out || out_of_range) ? 0 : gbl_out_off;


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

#define MLO_ACCUM_SZ (MLO_N_LCL_OUT_MAPS* MLO_N_LCL_IN_MAPS * MLO_OUT_STACKS)

	__private _FLOAT pvt_accum[MLO_ACCUM_SZ];

	for (int i = 0; i < MLO_ACCUM_SZ; ++i)
	{
		pvt_accum[i] = 0;
	}

	for (int i = lcl_id; i < MLO_LCL_MEM_SZ; i += MLO_GRP_SZ)
	{
		lcl_mem[i] = 0;
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
		readData(MLO_N_LCL_IN_MAPS, gbl_in_scan_off, MLO_IN_CHANNEL_STRIDE, MLO_IN_STACKS, (m_idx + c_idx), MLO_N_INPUTS, bot, bot_dat, last_pixel);

		// read output maps
		readDataFlex(MLO_N_LCL_OUT_MAPS, gbl_out_scan_off, MLO_OUT_CHANNEL_STRIDE, MLO_OUT_STACKS, (k_idx), MLO_N_OUTPUTS, top_df, proc_mem);

		for (int k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
		{

			barrier(CLK_LOCAL_MEM_FENCE);
#if 0
	// move 1 set of output maps into LDS
			for (int i = 0; i < MLO_READ_UNIT; ++i)
			{
				proc_mem[lcl_id*MLO_READ_UNIT + i] = top_dat[k*MLO_READ_UNIT + i];
			}

			barrier(CLK_LOCAL_MEM_FENCE);
#endif
			/*
			core processing loop
			bot - input
			top - output diff

			do convolution with all available input maps

			*/

	
			for (int n = 0; n < MLO_OUT_STACKS; ++n)
			{
				__private _FLOAT pvt_top[MLO_READ_UNIT];
				for (int i = 0; i < MLO_READ_UNIT; ++i)
				{
					pvt_top[i] = proc_mem[(n*MLO_MAP_WK_SZ + p4)*MLO_READ_UNIT + i]; //top_df[gbl_out_scan_off + (k*MLO_OUT_STACKS + n0)*MLO_IN_CHANNEL_STRIDE + i]; //proc_mem[(n*MLO_MAP_WK_SZ + p4)*MLO_READ_UNIT + i];
				}

				for (int c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
				{
					for (int i = 0; i < MLO_READ_UNIT; ++i)
					{
						pvt_accum[(k*MLO_OUT_STACKS + n)*MLO_N_LCL_IN_MAPS + c]
							+= bot_dat[c*MLO_READ_UNIT + i] * pvt_top[i];
#if 0

						if (k_idx + k*MLO_OUT_STACKS + n == 0 && c_idx + c * MLO_IN_STACKS + m_idx == 16 && m_idx < MLO_IN_STACKS && c_idx + m_idx + c*MLO_IN_STACKS < MLO_N_INPUTS && k_idx + k*MLO_OUT_STACKS + n < MLO_N_OUTPUTS)
						{
							printf("K:s: %d %d %d %d %d %d %d %d %f %f %f %f\n",
								MLO_OUT_STACKS,
								n,
								m_idx,
								gbl_out_scan_off,
								get_group_id(1),
								get_local_id(0),
								k,
								c,
								pvt_accum[(n* MLO_N_LCL_OUT_MAPS + k)*MLO_N_LCL_IN_MAPS + c],
								bot_dat[c*MLO_READ_UNIT + i] * pvt_top[i],
								bot_dat[c*MLO_READ_UNIT + i],
								pvt_top[i]
							);
						}
#endif

					}
					
				}
//				Processing(MLO_N_LCL_OUT_MAPS, o_idx, m_idx, MLO_N_LCL_IN_MAPS, c_idx, m_idx, pvt_accum, bot_dat, top_dat, (p4 == 0 && (lcl_id == 0 || lcl_id == 1)));
			}
		}


	}



	barrier(CLK_LOCAL_MEM_FENCE);


#if MLO_MAP_WK_SZ > 8
// transpose
	int red_base_off = m_idx * MLO_MAP_WK_SZ*MLO_ACCUM_SZ;
	for (int l = 0; l < MLO_ACCUM_SZ; ++l)
	{
		// write data
		red_mem[red_base_off + p4* MLO_ACCUM_SZ + l] = pvt_accum[l];

	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// do final summation

	__private _FLOAT final_sum[1] = { 0 };
	int new_m_idx = iDiv(lcl_id, MLO_ACCUM_SZ);
	int new_p4 = iMod(lcl_id, new_m_idx, MLO_ACCUM_SZ);
	int new_red_base_off = new_m_idx * MLO_MAP_WK_SZ*MLO_ACCUM_SZ;

	for (int s = 0; s < MLO_MAP_WK_SZ; ++s)
	{
		final_sum[0] += red_mem[new_red_base_off + s*MLO_ACCUM_SZ + new_p4];
	}

	// write out 
	// inputs are outputs
	int wei_df_off = ((ib * MLO_N_OUTPUTS + k_idx) * (int)MLO_WEI_BATCH_STRIDE) + ((c_idx + new_m_idx) * MLO_WEI_CHANNEL_STRIDE);

	int n = iDiv(new_p4, MLO_N_LCL_IN_MAPS);
	int c = iMod(new_p4, n, MLO_N_LCL_IN_MAPS);
	int k = 0;

	if (new_m_idx < MLO_IN_STACKS &&k_idx + k*MLO_OUT_STACKS + n < MLO_N_OUTPUTS && c_idx + new_m_idx + c*MLO_IN_STACKS < MLO_N_INPUTS)
	{
		weights_df[wei_df_off + (k*MLO_OUT_STACKS + n)*MLO_WEI_BATCH_STRIDE + c*MLO_IN_STACKS*MLO_WEI_CHANNEL_STRIDE] = final_sum[0];
	}
#else

#if 0 //MLO_IN_WIDTH > 24
	for (int i = lcl_id; i < MLO_LCL_MEM_SZ; i += MLO_GRP_SZ)
	{
		red_mem[i] = 0;
	}

	int red_base_off = (m_idx >= MLO_OUT_STACKS) ? MLO_LCL_MEM_SZ : m_idx * MLO_POW2_MAP_WK_SZ;
	// final summation over each filter row
	for (int l = 0; l < MLO_ACCUM_SZ; ++l)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		// write data
		red_mem[red_base_off + p4] = pvt_accum[l];

		// barrier inside
		ReduceKernel(&red_mem[red_base_off], &pvt_accum[l], p4, p4, MLO_POW2_MAP_WK_SZ, 1, false);

	}


	barrier(CLK_LOCAL_MEM_FENCE);
#else

	for (int l = 0; l < MLO_ACCUM_SZ; ++l)
	{
		lcl_mem[lcl_id] = pvt_accum[l];
		barrier(CLK_LOCAL_MEM_FENCE);
		if (p4 == 0)
		{
			for (int i = 1; i < MLO_MAP_WK_SZ; ++i)
			{
				pvt_accum[l] += lcl_mem[lcl_id + i];
			}

		}	
		barrier(CLK_LOCAL_MEM_FENCE);
	}


#endif


	// write out 
	// inputs are outputs
	int wei_df_off = ((ib * MLO_N_OUTPUTS + k_idx) * (int)MLO_WEI_BATCH_STRIDE) + ((c_idx + m_idx) * MLO_WEI_CHANNEL_STRIDE);


	for (int n = 0; n < MLO_OUT_STACKS && p4 == 0 && m_idx < MLO_IN_STACKS; ++n)
	{
		for (int k = 0; k < MLO_N_LCL_OUT_MAPS && k_idx + k*MLO_OUT_STACKS + n < MLO_N_OUTPUTS; ++k)
		{
			for (int c = 0; c < MLO_N_LCL_IN_MAPS && c_idx + m_idx + c*MLO_IN_STACKS < MLO_N_INPUTS; ++c)
			{
				weights_df[wei_df_off + (k*MLO_OUT_STACKS + n)*MLO_WEI_BATCH_STRIDE + c*MLO_IN_STACKS*MLO_WEI_CHANNEL_STRIDE] = pvt_accum[(k*MLO_OUT_STACKS + n)*MLO_N_LCL_IN_MAPS + c];

			}

		}
	}

#endif
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
