/*
 * Copyright (c) 2017 AMD Inc.
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




__attribute__((always_inline))
int iDiv(int v, int d)
{
	int r = (int)((float)v / d + 0.00001f);
	return(r);
}

__attribute__((always_inline))
int iMod(int v, int u, int d)
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



/*********************************************************************************************************
// wrw algorithm for large filters
// idea:
// read MLO_OUT_STACKS output maps into LDS
// read MLO_N_LCL_IN_MAPS per wk_item input maps

// alg


// convolve 
// reduce with transform

// write out 


**********************************************************************************************************/

__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2)))
__kernel void MIOpenCvBwdWrWSmap(
	const __global _FLOAT * __restrict top_df,
	const __global _FLOAT * __restrict bot,
	__global _FLOAT * __restrict weights_df,
	_FLOAT padding_val
)
{
	// reduction memory.

	__local _FLOAT lcl_mem[MLO_LCL_MEM_SZ];
	__local _FLOAT * red_mem = lcl_mem;

	int lcl_id = get_local_id(0);

	int k_idx = get_group_id(0) * (MLO_N_LCL_OUT_MAPS); // output map index base

	int c_idx = get_group_id(1) * (MLO_N_LCL_IN_MAPS * MLO_N_MAPS_PER_GROUP); // input map index based

	int ib = get_group_id(2); // batch id


	int gbl_in_off = c_idx * MLO_IN_CHANNEL_STRIDE + ib * MLO_IN_BATCH_STRIDE;
	int gbl_out_off = k_idx * MLO_OUT_CHANNEL_STRIDE + ib * MLO_OUT_BATCH_STRIDE;


#define MLO_TOP_DAT_SZ (MLO_N_LCL_OUT_MAPS* MLO_READ_UNIT)

	__private _FLOAT top_dat[MLO_TOP_DAT_SZ];


#define MLO_BOT_DAT_SZ (MLO_N_LCL_IN_MAPS * MLO_READ_UNIT)

	__private _FLOAT bot_dat[MLO_BOT_DAT_SZ];


#define MLO_ACCUM_SZ (MLO_N_LCL_OUT_MAPS* MLO_N_LCL_IN_MAPS)

	__private _FLOAT pvt_accum[MLO_ACCUM_SZ];

	for (int i = 0; i < MLO_ACCUM_SZ; ++i)
	{
		pvt_accum[i] = 0;
	}

	for (int i = lcl_id; i < MLO_LCL_MEM_SZ; i += MLO_GRP_SZ)
	{
		lcl_mem[i] = 0;
	}

// map id inside the group, super-pixel inside the map
#if (MLO_MAP_WK_SZ &  (MLO_MAP_WK_SZ - 1))

	int m_id = iDiv(lcl_id, MLO_MAP_WK_SZ);  // map
	int p4 = iMod(lcl_id, m_id, MLO_MAP_WK_SZ); // pixel
#else
	int m_id = ((uint)lcl_id / MLO_MAP_WK_SZ);  // map
	int p4 = ((uint)lcl_id & (MLO_MAP_WK_SZ - 1)); // pixel

#endif

	gbl_in_off += p4 * MLO_READ_UNIT;
//	gbl_out_off += p4 * MLO_READ_UNIT;
// input is kept in registers at the start
	gbl_in_off += m_id * MLO_IN_CHANNEL_STRIDE;

	bool last_pixel = (p4 == MLO_MAP_WK_SZ - 1);
// inside input range
	bool inside_map_range_input = ((c_idx + m_id) < MLO_N_INPUTS && m_id < MLO_N_MAPS_PER_GROUP);
	bool inside_range_input = (p4 < MLO_MAP_WK_SZ &&  inside_map_range_input);

	// inside output range
	bool inside_range_output = (p4 < MLO_MAP_WK_SZ);

	for (int b = 0; b < MLO_BATCH_SZ; ++b, gbl_in_off += MLO_IN_BATCH_STRIDE, gbl_out_off += MLO_OUT_BATCH_STRIDE)
	{

		// read all inputs into registers
		int bot_off = gbl_in_off;

#if MLO_N_PIXS_OFF > 0

			if (last_pixel)
			{
				for (int c = 0; c < MLO_N_LCL_IN_MAPS; ++c, bot_off += MLO_N_MAPS_PER_GROUP*MLO_IN_CHANNEL_STRIDE)
				{

					// reading in order per group and jump over maps been read
					// read arbitrary data but inside the range

					bool inside_range_input2 = inside_range_input && ((c_idx + m_id + c*MLO_N_MAPS_PER_GROUP) < MLO_N_INPUTS);

					bot_off = (inside_range_input2) ? bot_off : 0;

					for (int i = 0; i < MLO_N_PIXS_OFF; ++i)
					{
						bot_dat[c*MLO_READ_UNIT + i] = bot[bot_off + i];
					}
					for (int i = MLO_N_PIXS_OFF; i < MLO_READ_UNIT; ++i)
					{
						bot_dat[c*MLO_READ_UNIT + i] = 0;
					}


				}

			}
			else
#endif
			{
				// check 
				for (int c = 0; c < MLO_N_LCL_IN_MAPS; ++c, bot_off += MLO_N_MAPS_PER_GROUP*MLO_IN_CHANNEL_STRIDE)
				{
					// reading in order per group and jump over maps been read
					// read arbitrary data but inside the range

					bool inside_range_input2 = inside_range_input && ((c_idx + m_id + c*MLO_N_MAPS_PER_GROUP) < MLO_N_INPUTS);

					bot_off = (inside_range_input2) ? bot_off : 0;

					for (int i = 0; i < MLO_READ_UNIT; ++i)
					{
						bot_dat[c*MLO_READ_UNIT + i] = bot[bot_off + i];
					}

				}

			} // if (last_pixel)

//			int top_off = (inside_range_output) ? gbl_out_off : 0;

		int top_off = gbl_out_off;

		// read all outputs
		// assum division by MLO_N_LCL_OUT
		for (int kb = 0; kb < MLO_N_LCL_OUT; kb++, top_off += MLO_OUT_LCL_BLK * MLO_OUT_CHANNEL_STRIDE)
		{

			barrier(CLK_LOCAL_MEM_FENCE);

			for (int p = lcl_id; p < MLO_OUT_LCL_BLK * MLO_MAP_WK_SZ; p += MLO_GRP_SZ)
			{
#if (MLO_MAP_WK_SZ & (MLO_MAP_WK_SZ - 1))
				int m = iDiv(p, MLO_MAP_WK_SZ);
				int pm = iMod(p, m, MLO_MAP_WK_SZ);
#else
				int m = ((uint)p / MLO_MAP_WK_SZ);
				int pm = ((uint)p & (MLO_MAP_WK_SZ - 1));
#endif


				int top_off1 = top_off + m * MLO_OUT_CHANNEL_STRIDE + pm*MLO_READ_UNIT;
#if MLO_N_OUT_MAPS_ALIGNED == 0
				top_off1 = (k_idx + m < MLO_N_OUTPUTS) ? top_off1 : 0;
#endif
#if MLO_N_PIXS_OFF > 0
				if (pm == MLO_MAP_WK_SZ - 1)
				{
					for (int i = 0; i < MLO_N_PIXS_OFF; ++i)
					{
						lcl_mem[p*MLO_READ_UNIT + i] = top_df[top_off1 + i];
					}
					for (int i = MLO_N_PIXS_OFF; i < MLO_READ_UNIT; ++i)
					{
						lcl_mem[p*MLO_READ_UNIT + i] = 0;
					}


				}
				else

#endif
				{
					for (int i = 0; i < MLO_READ_UNIT; ++i)
					{
						lcl_mem[p*MLO_READ_UNIT + i] = top_df[top_off1 + i];

					}
				}


			} // for(int p = 0; p < MLO_OUT_LCL_BLK * MLO_MAP_WK_SZ; p += MLO_GRP_SZ)

			barrier(CLK_LOCAL_MEM_FENCE);

			for (int k = kb*MLO_OUT_LCL_BLK; k < (kb + 1)*MLO_OUT_LCL_BLK; ++k)
			{
				// processing
				for (int c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
				{
					for (int i = 0; i < MLO_READ_UNIT; ++i)
					{
						_FLOAT bot_val = bot_dat[c*MLO_READ_UNIT + i];
						_FLOAT top_val = lcl_mem[((k - kb*MLO_OUT_LCL_BLK) * MLO_MAP_WK_SZ + p4)*MLO_READ_UNIT + i];
						pvt_accum[k * MLO_N_LCL_IN_MAPS + c] += top_val * bot_val;
#if 0
						if (top_val * bot_val !=0  &&  k == 0 && c == 0 && m_id == 0 && get_group_id(0) == 0 && get_group_id(1) == 1 && lcl_id ==0 )
						{

							printf("K:c: %d %d %d %d %f %f %f %f\n",
								c_idx + m_id + c*MLO_N_MAPS_PER_GROUP,
								lcl_id,
								c*MLO_READ_UNIT + i,
								((k - kb*MLO_OUT_LCL_BLK) * MLO_MAP_WK_SZ + p4)*MLO_READ_UNIT + i,
								pvt_accum[k * MLO_N_LCL_IN_MAPS + c],
								top_val * bot_val,
								bot_val,
								top_val

							);
						}

#endif

					}
				}
			}

		} // for(int kb = 0; kb < MLO_N_LCL_OUT; kb++, top_off += MLO_OUT_LCL_BLK * MLO_OUT_CHANNEL_STRIDE)
	} // for (int b = 0; b < MLO_BATCH_SZ; ++b, gbl_in_off += MLO_IN_BATCH_STRIDE, gbl_out_off += MLO_OUT_BATCH_STRIDE)


	// write out 
	// inputs are outputs
	int wei_df_off = ((ib * MLO_N_OUTPUTS + k_idx) * (int)MLO_WEI_BATCH_STRIDE) + (c_idx + m_id) * MLO_WEI_CHANNEL_STRIDE;


#if 1
// transpose data and usm it up using MLO_REDUC_LOOP_STEP wk-items from each small map 
	__private _FLOAT final_sum[(MLO_ACCUM_SZ / MLO_REDUC_LOOP_STEP)];
//	if (inside_range_input)
	{
		for (int r = 0; r < (MLO_ACCUM_SZ / MLO_REDUC_LOOP_STEP); ++r)
		{
			final_sum[r] = 0;

			barrier(CLK_LOCAL_MEM_FENCE);

			for (int rr = 0; rr < MLO_REDUC_LOOP_STEP; ++rr)
			{
				lcl_mem[lcl_id*MLO_REDUC_LOOP_STEP + rr] = pvt_accum[r*MLO_REDUC_LOOP_STEP + rr];
			}
			barrier(CLK_LOCAL_MEM_FENCE);

			if (p4 < MLO_REDUC_LOOP_STEP)
			{
				for (int j = 0; j < MLO_MAP_WK_SZ; j++)
				{
					final_sum[r] += lcl_mem[(m_id*MLO_MAP_WK_SZ + j)*MLO_REDUC_LOOP_STEP + p4];
				}

			}
		}

		if (p4 < MLO_REDUC_LOOP_STEP)
		{
			for (int r = 0; r < (MLO_ACCUM_SZ / MLO_REDUC_LOOP_STEP); ++r)
			{
				int wei_idx = r* ( MLO_REDUC_LOOP_STEP) + p4;

#if (MLO_N_LCL_IN_MAPS & (MLO_N_LCL_IN_MAPS - 1))
				int k = iDiv(wei_idx, MLO_N_LCL_IN_MAPS);
				int c = iMod(wei_idx, k, MLO_N_LCL_IN_MAPS);
#else
				int k = ((uint)wei_idx / MLO_N_LCL_IN_MAPS);
				int c = ((uint)wei_idx & (MLO_N_LCL_IN_MAPS-1));
#endif


				if (m_id < MLO_N_MAPS_PER_GROUP && (c_idx + m_id + c*MLO_N_MAPS_PER_GROUP) < MLO_N_INPUTS
#if MLO_N_OUT_MAPS_ALIGNED == 0
					&& k_idx + k < MLO_N_OUTPUTS
#endif
					)
				{
					int wei_off = wei_df_off + k*MLO_WEI_BATCH_STRIDE + c*MLO_N_MAPS_PER_GROUP*MLO_WEI_CHANNEL_STRIDE;
					weights_df[wei_off] = final_sum[r];
				}


			} // for (int r = 0; r < (MLO_ACCUM_SZ / MLO_REDUC_LOOP_STEP); ++r)
		} // if (p4 < MLO_REDUC_LOOP_STEP)
	} // if (inside_range_input)
#else
	for (int r = 0; r < (MLO_ACCUM_SZ / MLO_REDUC_LOOP_STEP); ++r)
	{
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int rr = 0; rr < MLO_REDUC_LOOP_STEP; ++rr)
		{
			lcl_mem[lcl_id*MLO_REDUC_LOOP_STEP + rr] = pvt_accum[r*MLO_REDUC_LOOP_STEP + rr];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		if (p4 == 0 && inside_range_input)
		{
			for (int j = 1; j < MLO_MAP_WK_SZ; j++)
			{
				for (int rr = 0; rr < MLO_REDUC_LOOP_STEP; ++rr)
				{
					pvt_accum[r*MLO_REDUC_LOOP_STEP + rr] += lcl_mem[(lcl_id + j)*MLO_REDUC_LOOP_STEP + rr];
				}
			}
		}
	}



	if (p4 == 0)
	{
		for (int kb = 0; kb < MLO_N_LCL_OUT; kb++)
		{
			for (int k = kb*MLO_OUT_LCL_BLK; k < (kb + 1)*MLO_OUT_LCL_BLK; ++k)
			{
				for (int c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
				{
					if ( (c_idx + m_id + c*MLO_N_MAPS_PER_GROUP) < MLO_N_INPUTS
#if MLO_N_OUT_MAPS_ALIGNED == 0
						  && k_idx + k < MLO_N_OUTPUTS
#endif
						)
					{
						int wei_off = wei_df_off + k*MLO_WEI_BATCH_STRIDE + c*MLO_N_MAPS_PER_GROUP*MLO_WEI_CHANNEL_STRIDE;
						weights_df[wei_off] = pvt_accum[k*MLO_N_LCL_IN_MAPS + c];
					}

				}

			}
		}

	}

#endif


}

