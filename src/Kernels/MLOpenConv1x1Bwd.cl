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

#define DBG_OUT_OF_RNGE 0

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
#if MLO_N_MAPS_PERGROUP > 1
	__local  _FLOAT lcl_out_stage[MLO_MAP_SZ4*MLO_EXCHANGE_STEP * MLO_N_MAPS_PERGROUP * MLO_READ_UNIT];

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

	int wei_off = out_grp_block * MLO_N_LCL_OUT_MAPS *
#if MLO_DIR_FORWARD==1
		MLO_WEI_BSTRIDE
#else
		MLO_WEI_CHANNEL_STRIDE
#endif
		;
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
	for (int c = 0; c < MLO_IN_LOOP; ++c,
		in_off += MLO_IN_CHANNEL_STRIDE*MLO_N_LCL_IN_MAPS * MLO_N_MAPS_PERGROUP,
		wei_off += MLO_N_LCL_IN_MAPS* MLO_N_MAPS_PERGROUP *
#if MLO_DIR_FORWARD==1
		MLO_WEI_CHANNEL_STRIDE
#else
		MLO_WEI_BSTRIDE
#endif
		)
	{
		// read data
		// over all local batchs
		int in_off1 = in_off;
		for (int ib = 0; ib < MLO_N_LCL_BATCHS
#if MLO_BATCH_ALIGNED == 0
			&& (batch_block*MLO_N_LCL_BATCHS + ib < MLO_BATCH_SZ)
#endif
			; ++ib, in_off1 += MLO_IN_BATCH_STRIDE)
		{
			int in_off2 = in_off1;
			// lcl in maps (in data tiles) is has the stride = MLO_N_MAPS_PERGROUP
			for (int ilc = 0; ilc < MLO_N_LCL_IN_MAPS; ++ilc, in_off2 += MLO_IN_CHANNEL_STRIDE * MLO_N_MAPS_PERGROUP)
			{
				// read data

				for (int i = 0; i < MLO_READ_UNIT; ++i)
				{
					in_stage[ib][ilc][i] = 0;
				}


				if (c*MLO_N_LCL_IN_MAPS * MLO_N_MAPS_PERGROUP + in_map_id + ilc* MLO_N_MAPS_PERGROUP < MLO_N_INPUTS)
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
		int wei_off1 = wei_off + in_map_off_id *
#if MLO_DIR_FORWARD==1
			MLO_WEI_CHANNEL_STRIDE
#else
			MLO_WEI_BSTRIDE
#endif
			;
		for (int olc = 0; olc < MLO_N_LCL_OUT_MAPS; ++olc, wei_off1 += 
#if MLO_DIR_FORWARD==1
			MLO_WEI_BSTRIDE
#else
			MLO_WEI_CHANNEL_STRIDE
#endif
			)
		{
			int wei_off2 = wei_off1;
			// lcl in maps (in data tiles) is has the stride = MLO_N_MAPS_PERGROUP, weights are mapped accordingly

			for (int ilc = 0; ilc < MLO_N_LCL_IN_MAPS; ++ilc, wei_off2 += MLO_N_MAPS_PERGROUP * 
#if MLO_DIR_FORWARD==1
				MLO_WEI_CHANNEL_STRIDE
#else
				MLO_WEI_BSTRIDE
#endif
				)
			{
				// read weights
				int wei_off_r = (wei_off2 < MLO_N_INPUTS * MLO_N_OUTPUTS) ? wei_off2 : 0;

				wei_stage = wei_ptr[wei_off_r];
				wei_stage = (wei_off2 < MLO_N_INPUTS * MLO_N_OUTPUTS) ? wei_stage : 0;
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

	if (in_map_id >= MLO_N_MAPS_PERGROUP || in_map_id*MLO_N_LCL_IN_MAPS >= MLO_N_INPUTS)
	{
		return;
	}

	out_block = out_grp_block * MLO_N_LCL_OUT_MAPS;
	int out_off = batch_block * MLO_N_LCL_BATCHS * MLO_OUT_BATCH_STRIDE
		+ out_block *  MLO_OUT_CHANNEL_STRIDE
		+ pix_id * MLO_READ_UNIT;

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
			// in_map_id now is an index of the output map
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


#else




	int out_off1 = out_off;
	for (int ib = 0; ib < MLO_N_LCL_BATCHS
		; ++ib, out_off1 += MLO_OUT_BATCH_STRIDE)
	{


#if MLO_BATCH_ALIGNED == 0
		if (batch_block*MLO_N_LCL_BATCHS + ib < MLO_BATCH_SZ)
#endif
		{
			int out_off2 = out_off1;
			for (int olc = 0; olc < MLO_N_LCL_OUT_MAPS
				; ++olc, out_off2 += MLO_OUT_CHANNEL_STRIDE)
			{


#if MLO_OUTPUTS_ALIGNED == 0
				if (out_block + olc < MLO_N_OUTPUTS)

#endif
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
	}


#endif

}
