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
#define _LEN_OF_TYPE          MLO_READ_UNIT

#ifndef FLT_MAX
#define FLT_MAX         3.402823466e+38F        /* max value */
#endif
// calculating the size of the area for weights prefetch

#if MLO_N_MAPS_PERGROUP > 1
#define MLO_WEIGHTS_PER_LOOP_MAX 8
#else
#define MLO_WEIGHTS_PER_LOOP_MAX 16
#endif
#if ((MLO_N_MAPS_PERGROUP*MLO_N_LCL_IN_MAPS) < MLO_N_INPUTS)
#define MLO_LCL_IN_ROW (MLO_N_MAPS_PERGROUP*MLO_N_LCL_IN_MAPS)
#else
#define MLO_LCL_IN_ROW (MLO_N_INPUTS)
#endif

#define MLO_WEIGHTS_PER_LOOP_TMP ((MLO_N_INPUTS + MLO_LCL_IN_ROW - 1)/MLO_LCL_IN_ROW)

#if (MLO_WEIGHTS_PER_LOOP_TMP < MLO_WEIGHTS_PER_LOOP_MAX)
#define MLO_WEIGHTS_PER_LOOP (MLO_WEIGHTS_PER_LOOP_TMP)
#else
#define MLO_WEIGHTS_PER_LOOP (MLO_WEIGHTS_PER_LOOP_MAX)
#endif
#define MLO_LCL_WEIGHTS_ROW (MLO_WEIGHTS_PER_LOOP * MLO_LCL_IN_ROW)
#define MLO_WEIGHTS_ROW (MLO_LCL_WEIGHTS_ROW* MLO_WEI_CHANNEL_STRIDE)

// size of the area for weights prefetch
#define MLO_WEIGHTS_LCL_SZ (MLO_WEIGHTS_ROW * MLO_N_LCL_OUT_MAPS)

// size of area for exchanging partial sums
#define MLO_EXCHNGE_SZ4 (MLO_MAP_SZ4*MLO_EXCHANGE_STEP * MLO_N_MAPS_PERGROUP)


#if MLO_N_MAPS_PERGROUP > 1 && ((MLO_EXCHNGE_SZ4 * _LEN_OF_TYPE) > MLO_WEIGHTS_LCL_SZ)
#define MLO_LCL_MEM_SZ (MLO_EXCHNGE_SZ4 * _LEN_OF_TYPE)
#else
#define MLO_LCL_MEM_SZ MLO_WEIGHTS_LCL_SZ
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

/*
Layout:
assuming NCHW data layout.

Data:
data has been fetch by 4 floats sequentially.
MLO_MAP_SZ4 = (map_width*map_height + 3)/4.
in case of total size not a multiple of 4 the the last pixel has a special treatment.
There are 2 cases:
MLO_N_MAPS_PERGROUP == 1
and
MLO_N_MAPS_PERGROUP > 1, when MLO_MAP_SZ4 <= GPROUP_SIZE/2, in other words when more than 1 map can be held by a group.
Case MLO_N_MAPS_PERGROUP == 1:
Data, by 4 floats, may come from MLO_N_LCL_IN_MAPS sequential input maps from MLO_N_LCL_BATCHS neighboring batches.
Weigts:
on each MLO_WEIGHTS_PER_LOOP input loop set of weight are prefetched for another MLO_WEIGHTS_PER_LOOP loops.
Each input map contributes to partial sums of MLO_N_LCL_OUT_MAPS output maps.
Case MLO_N_MAPS_PERGROUP > 1:
Similar to a previous case.
The difference is that several input sequential input maps are kept by group.
Each taking part in the calculation of partial sums of the same MLO_N_LCL_OUT_MAPS output maps.
After completion of the main MLO_IN_LOOP loop partial sums have been summed up in parallel.

*/

__kernel void MLOpenConv1x1(
       const __global _FLOAT * __restrict in_ptr,
       const __global _FLOAT * __restrict wei_ptr,
#if MLO_CONV_BIAS
       const __global _FLOAT * __restrict bias,
#endif
 	  __global _FLOAT * __restrict out_ptr,
	   _FLOAT dummy_val // nothing
	   )
{
// KERNEL
// private buffers
	__private _FLOAT in_stage[MLO_N_LCL_BATCHS][MLO_N_LCL_IN_MAPS][MLO_READ_UNIT];
	__private _FLOAT wei_stage;
	__private _FLOAT out_tiles[MLO_N_LCL_BATCHS][MLO_N_LCL_OUT_MAPS][MLO_READ_UNIT];
	__local _FLOAT lcl_wei_stage[MLO_LCL_MEM_SZ];

#if MLO_N_MAPS_PERGROUP > 1
	__local _FLOAT * lcl_out_stage = lcl_wei_stage;

#endif

	int lcl_id0 = get_local_id(0);
	int in_map_id = 0; // map
	int pix_id = get_global_id(0);  // inside map
	in_map_id = pix_id / MLO_MAP_SZ4; // mad id inside group
	int out_grp_block = get_group_id(1); // block of outputs for the entire group
	int out_block = out_grp_block;
	int batch_block = get_group_id(2); // block of batchs
// multipe maps per group

	pix_id = (pix_id - in_map_id * MLO_MAP_SZ4);  // pixel inside map

	int in_map_off_id = (in_map_id >= MLO_N_MAPS_PERGROUP) ? MLO_N_MAPS_PERGROUP - 1 : in_map_id;

	int in_off = batch_block * MLO_N_LCL_BATCHS * MLO_IN_BATCH_STRIDE
		+ in_map_off_id * MLO_IN_CHANNEL_STRIDE
				+ pix_id * MLO_READ_UNIT;

	int wei_off = out_grp_block * MLO_N_LCL_OUT_MAPS * MLO_WEI_BSTRIDE;
	for (int j = 0; j < MLO_N_LCL_BATCHS; ++j)
	{
		for (int i = 0; i < MLO_N_LCL_OUT_MAPS; ++i)
		{
			for (int k = 0; k < MLO_READ_UNIT; ++k)
			{
				out_tiles[j][i][k] = 0;
			}
		}
	}
// over all input maps; with step == MLO_N_LCL_IN_MAPS * MLO_N_MAPS_PERGROUP; MLO_IN_LOOP
	for (int c = 0, wc = 0; c < MLO_IN_LOOP; ++c,
		in_off += MLO_IN_CHANNEL_STRIDE*MLO_N_LCL_IN_MAPS * MLO_N_MAPS_PERGROUP
		)
	{

		barrier(CLK_LOCAL_MEM_FENCE);
		// read array of weights
		if (wc - c == 0)
		{
			for (int w = lcl_id0; w < MLO_WEIGHTS_LCL_SZ ; w += MLO_GRP_SZ0)
			{

#if (MLO_WEIGHTS_ROW) & (MLO_WEIGHTS_ROW - 1)

				int oi = iDiv(w, MLO_WEIGHTS_ROW);
				int lwi = iMod(w, oi, MLO_WEIGHTS_ROW);
#else
				int oi = w / MLO_WEIGHTS_ROW;
				int lwi = w & (MLO_WEIGHTS_ROW - 1);

#endif

				int wi = mad24(wc, (int)(MLO_N_LCL_IN_MAPS * MLO_N_MAPS_PERGROUP*MLO_WEI_CHANNEL_STRIDE),lwi);

				// out of range check
				int wei_off_r = wei_off + mad24(oi, MLO_WEI_BSTRIDE, wi);
				wei_off_r = (wei_off_r < MLO_N_OUTPUTS *MLO_N_INPUTS) ? wei_off_r : 0;
				_FLOAT wei_val = wei_ptr[wei_off_r];
				wei_val = (wei_off_r < MLO_N_OUTPUTS *MLO_N_INPUTS) ? wei_val : 0;
				lcl_wei_stage[mad24(oi, MLO_WEIGHTS_ROW, lwi)] = wei_val;

			}
			wc += MLO_WEIGHTS_PER_LOOP;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		int wei_indx = c - wc + MLO_WEIGHTS_PER_LOOP;
		// read data
		// over all local batchs
		int in_off1 = in_off;
		for (int ib = 0; ib < MLO_N_LCL_BATCHS
			; ++ib, in_off1 += MLO_IN_BATCH_STRIDE)
		{
			int in_off2 = in_off1;
			// lcl in maps (in data tiles) is has the stride = MLO_N_MAPS_PERGROUP
			for (int ilc = 0; ilc < MLO_N_LCL_IN_MAPS; ++ilc, in_off2 += MLO_IN_CHANNEL_STRIDE * MLO_N_MAPS_PERGROUP)
			{
				for (int i = 0; i < MLO_READ_UNIT; ++i)
				{
					in_stage[ib][ilc][i] = 0;
				}

				if (c*MLO_N_LCL_IN_MAPS * MLO_N_MAPS_PERGROUP + in_map_id + ilc* MLO_N_MAPS_PERGROUP < MLO_N_INPUTS
#if MLO_BATCH_ALIGNED == 0
					&& (batch_block*MLO_N_LCL_BATCHS + ib < MLO_BATCH_SZ)
#endif
				)
				{
#if MLO_C1x1_PIXLEFT > 0
					// if the last one
					if (pix_id == MLO_MAP_SZ4 - 1)
					{

						for (int i = 0; i < MLO_C1x1_PIXLEFT; ++i)
						{
							in_stage[ib][ilc][i] = in_ptr[in_off2 + i];
						}
					}
					else

#endif
					{
						for (int i = 0; i < MLO_READ_UNIT; ++i)
						{
							in_stage[ib][ilc][i] = in_ptr[in_off2 + i];
						}
					}
				}
			}
		}


		// convolve
		for (int olc = 0, lcl_wei_off = wei_indx*MLO_N_LCL_IN_MAPS * MLO_N_MAPS_PERGROUP*MLO_WEI_CHANNEL_STRIDE; olc < MLO_N_LCL_OUT_MAPS; ++olc, lcl_wei_off += MLO_WEIGHTS_ROW)
		{
			// lcl in maps (in data tiles) is has the stride = MLO_N_MAPS_PERGROUP, weights are mapped accordingly
			int lcl_wei_off1 = lcl_wei_off;
			for (int ilc = 0; ilc < MLO_N_LCL_IN_MAPS; ++ilc, lcl_wei_off1 += MLO_N_MAPS_PERGROUP*MLO_WEI_CHANNEL_STRIDE)
			{
				// read weights
				int lcl_wei_off2 = lcl_wei_off1 + mul24(in_map_id, (int)MLO_WEI_CHANNEL_STRIDE);
				wei_stage = lcl_wei_stage[lcl_wei_off2];
				for (int ib = 0; ib < MLO_N_LCL_BATCHS; ++ib)
				{
					for (int i = 0; i < MLO_READ_UNIT; ++i)
					{
						out_tiles[ib][olc][i] += in_stage[ib][ilc][i] * wei_stage;
					}
				}
			}
		}


	}

// out of range check
	if (in_map_id >= MLO_N_MAPS_PERGROUP || in_map_id*MLO_N_LCL_IN_MAPS >= MLO_N_INPUTS)
	{
		return;
	}

	out_block = out_grp_block * MLO_N_LCL_OUT_MAPS;
	int out_off = batch_block * MLO_N_LCL_BATCHS * MLO_OUT_BATCH_STRIDE
		+ out_block *  MLO_OUT_CHANNEL_STRIDE
		+ pix_id * MLO_READ_UNIT;

// small groups
#if MLO_N_MAPS_PERGROUP > 1

	// calculate reduction over all partial sums
	// MLO_N_LCL_OUT_MAPS is multiple of MLO_EXCHANGE_STEP
	// write data into local memory

	for (int ib = 0; ib < MLO_N_LCL_BATCHS; ++ib)
	{
		for (int t = 0, p = 0; t < MLO_N_LCL_OUT_MAPS; t += MLO_EXCHANGE_STEP)
		{

			barrier(CLK_LOCAL_MEM_FENCE);

			if (lcl_id0 < MLO_MAP_SZ4 * MLO_N_MAPS_PERGROUP)
			{
				for (int om = 0; om < MLO_EXCHANGE_STEP; ++om)
				{
					int lcl_off = (om * MLO_MAP_SZ4*MLO_N_MAPS_PERGROUP + in_map_id*MLO_MAP_SZ4 + pix_id) * MLO_READ_UNIT;
					for (int i = 0; i < MLO_READ_UNIT; ++i)
					{
						lcl_out_stage[lcl_off + i] = out_tiles[ib][t + om][i];
					}
				}

			}
			barrier(CLK_LOCAL_MEM_FENCE);

			// sum partial sum
			// MLO_N_MAPS_PERGROUP >= MLO_EXCHANGE_STEP
			// in_map_id is an index of the output map now.
			if (in_map_id < MLO_EXCHANGE_STEP)
			{
				_FLOAT sum[MLO_READ_UNIT];
				for (int i = 0; i < MLO_READ_UNIT; ++i)
				{
					sum[i] = 0;
				}

				for (int s = 0; s < MLO_N_MAPS_PERGROUP; ++s)
				{
					int imp = in_map_id + s;
					imp = (imp >= MLO_N_MAPS_PERGROUP) ? imp - MLO_N_MAPS_PERGROUP : imp;
					int lcl_off = (in_map_id* MLO_MAP_SZ4*MLO_N_MAPS_PERGROUP + imp*MLO_MAP_SZ4 + pix_id) * MLO_READ_UNIT;
					for (int i = 0; i < MLO_READ_UNIT; ++i)
					{
						sum[i] += lcl_out_stage[lcl_off + i];
					}

				}



				// write it out
				int olc = t + in_map_id;

				if (true 
#if MLO_BATCH_ALIGNED == 0
					&& (batch_block*MLO_N_LCL_BATCHS + ib < MLO_BATCH_SZ)
#endif
#if MLO_OUTPUTS_ALIGNED == 0
					&& out_block + olc < MLO_N_OUTPUTS
#endif
					)
				{
				
					int out_off2 = out_off + ib * MLO_OUT_BATCH_STRIDE + olc * MLO_OUT_CHANNEL_STRIDE;

					_FLOAT  bias_val = 0;
#if MLO_CONV_BIAS
					bias_val = bias[out_block* MLO_N_LCL_OUT_MAPS + olc];
#endif
#if MLO_C1x1_PIXLEFT > 0

					// if the last one
					if (pix_id == MLO_MAP_SZ4 - 1)
					{
						for (int i = 0; i < MLO_C1x1_PIXLEFT; ++i)
						{
							out_ptr[out_off2 + i] = sum[i]
#if MLO_CONV_BIAS
								+ bias_val
#endif
								;

						}

					}
					else
#endif
					{

						for (int i = 0; i < MLO_READ_UNIT; ++i)
						{
							out_ptr[out_off2 + i] = sum[i]
#if MLO_CONV_BIAS
								+ bias_val
#endif
								;
						}
					}


				} //if (true

			} // if (in_map_id < MLO_EXCHANGE_STEP)

		} // for (int t = 0, p = 0; t < MLO_N_LCL_OUT_MAPS; t += MLO_EXCHANGE_STEP)

	} // 	for (int ib = 0; ib < MLO_N_LCL_BATCHS; ++ib)



#else // #if MLO_N_MAPS_PERGROUP > 1




	int out_off1 = out_off;
	for (int ib = 0; ib < MLO_N_LCL_BATCHS
		; ++ib, out_off1 += MLO_OUT_BATCH_STRIDE)
	{

		int out_off2 = out_off1;
		for (int olc = 0; olc < MLO_N_LCL_OUT_MAPS
			; ++olc, out_off2 += MLO_OUT_CHANNEL_STRIDE)
		{

		    if ( true
#if MLO_BATCH_ALIGNED == 0
			&& (batch_block*MLO_N_LCL_BATCHS + ib < MLO_BATCH_SZ)
#endif
#if MLO_OUTPUTS_ALIGNED == 0
			&& (out_block + olc < MLO_N_OUTPUTS)
#endif
			)
			{
				_FLOAT  bias_val = 0;
#if MLO_CONV_BIAS
				bias_val = bias[out_block* MLO_N_LCL_OUT_MAPS + olc];
#endif
#if MLO_C1x1_PIXLEFT > 0

			// if the last one
				if (pix_id == MLO_MAP_SZ4 - 1)
				{
					for (int i = 0; i < MLO_C1x1_PIXLEFT; ++i)
					{
						out_ptr[out_off2 + i] = out_tiles[ib][olc][i]
#if MLO_CONV_BIAS
							+ bias_val
#endif
						;

					}

				}
				else
#endif
				{
					for (int i = 0; i < MLO_READ_UNIT; ++i)
					{

						out_ptr[out_off2 + i] = out_tiles[ib][olc][i]
#if MLO_CONV_BIAS
						+ bias_val
#endif
						;
					}
				}
			}
		}

	}


#endif // #if MLO_N_MAPS_PERGROUP > 1

}
