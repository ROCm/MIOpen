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

/*
Layout:


*/

__kernel void MLOpenConv1x1PS(
       const __global _FLOAT * restrict in_ptr,
       const __global _FLOAT * restrict wei_ptr,
#if MLO_CONV_BIAS
       const __global _FLOAT * bias,
#endif
 	  __global _FLOAT *out_ptr,
	   _FLOAT dummy_val // nothing
	   )
{
// KERNEL
// private buffers
	__private _FLOAT4 in_stage[MLO_N_LCL_BATCHS][MLO_N_LCL_IN_MAPS];
	__private _FLOAT wei_stage[MLO_N_LCL_OUT_MAPS][MLO_N_LCL_IN_MAPS];
	__private _FLOAT4 out_tiles[MLO_N_LCL_BATCHS][MLO_N_LCL_OUT_MAPS];
#if MLO_N_MAPS_PERGROUP > 1
	__local _FLOAT4 lcl_out_stage[MLO_MAP_SZ4*MLO_EXCHANGE_STEP * MLO_N_MAPS_PERGROUP];

#endif

	int lcl_id0 = get_local_id(0);
	int in_map_id = 0; // map
	int pix_id = get_global_id(0);  // inside map
	in_map_id = pix_id / MLO_MAP_SZ4; // mad id inside group
	int out_grp_block = get_group_id(1); // block of outputs for the entire group
	int out_block = out_grp_block;
	int batch_block = get_group_id(2); // block of batchs
// multipe maps per group
//#if MLO_N_MAPS_PERGROUP > 1
	pix_id = (pix_id - in_map_id * MLO_MAP_SZ4);  // pixel inside map
//#endif
	int in_map_off_id = (in_map_id >= MLO_N_MAPS_PERGROUP) ? MLO_N_MAPS_PERGROUP - 1 : in_map_id;

	int in_off = batch_block * MLO_N_LCL_BATCHS * MLO_IN_BATCH_STRIDE
		+ in_map_off_id * MLO_IN_CHANNEL_STRIDE
				+ pix_id * 4;

	int wei_off = out_grp_block * MLO_N_LCL_OUT_MAPS * MLO_WEI_BSTRIDE;
	for (int j = 0; j < MLO_N_LCL_BATCHS; ++j)
	{
		for (int i = 0; i < MLO_N_LCL_OUT_MAPS; ++i)
		{
			out_tiles[j][i] = 0;
		}
	}
// over all input maps; with step == MLO_N_LCL_IN_MAPS * MLO_N_MAPS_PERGROUP; MLO_IN_LOOP
	for (int c = 0; c < MLO_IN_LOOP; ++c,
		in_off += MLO_IN_CHANNEL_STRIDE*MLO_N_LCL_IN_MAPS * MLO_N_MAPS_PERGROUP,
		wei_off += MLO_N_LCL_IN_MAPS* MLO_N_MAPS_PERGROUP * MLO_WEI_CHANNEL_STRIDE
		//#if MLO_DIR_FORWARD==0
		//			* MLO_N_OUTPUTS
		//#endif
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
				//				in_stage[ib][ilc] = 0;
				//				if (c*MLO_N_LCL_IN_MAPS * MLO_N_MAPS_PERGROUP + in_map_id + ilc* MLO_N_MAPS_PERGROUP < MLO_N_INPUTS)
				{

					in_stage[ib][ilc] = *(_FLOAT4*)&in_ptr[in_off2];
					in_stage[ib][ilc] = (c*MLO_N_LCL_IN_MAPS * MLO_N_MAPS_PERGROUP + in_map_id + ilc* MLO_N_MAPS_PERGROUP < MLO_N_INPUTS) ? in_stage[ib][ilc] : 0;
#if !MLO_DIVBY4

					// if the last one
					if (pix_id == MLO_MAP_SZ4 - 1)
					{
						for (int j = 3; j >= MLO_C1x1_PIXLEFT; --j)
						{
							((_FLOAT*)&in_stage[ib][ilc])[j] = 0;
						}
					}

#endif
				}

			}
		}

		// convolve
		int wei_off1 = wei_off + in_map_off_id * MLO_WEI_CHANNEL_STRIDE;
		for (int olc = 0; olc < MLO_N_LCL_OUT_MAPS; ++olc, wei_off1 += MLO_WEI_BSTRIDE)
		{
			int wei_off2 = wei_off1;
			// lcl in maps (in data tiles) is has the stride = MLO_N_MAPS_PERGROUP, weights are mapped accordingly

			for (int ilc = 0; ilc < MLO_N_LCL_IN_MAPS; ++ilc, wei_off2 += MLO_WEI_CHANNEL_STRIDE * MLO_N_MAPS_PERGROUP)
			{
				// read weights
				wei_stage[olc][ilc] = wei_ptr[wei_off2];
				for (int ib = 0; ib < MLO_N_LCL_BATCHS; ++ib)
				{
					out_tiles[ib][olc] += in_stage[ib][ilc] * (_FLOAT4)wei_stage[olc][ilc];
#if 0
					if (get_group_id(0) == 0 && get_group_id(1) == 0 && get_group_id(2) == 0 && in_map_id == 0 && pix_id == 0 && olc == 0 && ib == 0)
					{
						printf("k:c: %d %d %d %d %d  %11.10f %11.10f %f %f\n",
							c,
							wei_off2,
							in_map_off_id,
							olc,
							ilc,
							out_tiles[ib][olc].s0,
							in_stage[ib][ilc].s0 * wei_stage[olc][ilc],
							in_stage[ib][ilc].s0,
							wei_stage[olc][ilc]
							);
					}
#endif

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
		+ pix_id * 4;

#if MLO_N_MAPS_PERGROUP > 1
	// calculate reduction over all partial sums
	// MLO_N_LCL_OUT_MAPS is multiple of MLO_EXCHANGE_STEP
	// write data into local memory

	for (int ib = 0; ib < MLO_N_LCL_BATCHS; ++ib)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int t = 0, p = 0; t < MLO_N_LCL_OUT_MAPS; t += MLO_EXCHANGE_STEP)
		{

			if (lcl_id0 < MLO_MAP_SZ4 * MLO_N_MAPS_PERGROUP)
			{
				for (int om = 0; om < MLO_EXCHANGE_STEP; ++om)
				{
					lcl_out_stage[om * MLO_MAP_SZ4*MLO_N_MAPS_PERGROUP + in_map_id*MLO_MAP_SZ4 + pix_id]
						= out_tiles[ib][t + om];
				}

			}
			barrier(CLK_LOCAL_MEM_FENCE);

			// sum partial sum
			// MLO_N_MAPS_PERGROUP >= MLO_EXCHANGE_STEP
			// in_map_id now is an index of the output map
			if (in_map_id < MLO_EXCHANGE_STEP)
			{
				_FLOAT4 sum = 0;
				for (int s = 0; s < MLO_N_MAPS_PERGROUP; ++s)
				{
					int imp = in_map_id + s;
					imp = (imp >= MLO_N_MAPS_PERGROUP) ? imp - MLO_N_MAPS_PERGROUP : imp;
					int lcl_off = in_map_id* MLO_MAP_SZ4*MLO_N_MAPS_PERGROUP // output map offset
						+ s*MLO_MAP_SZ4 + pix_id;
					sum += lcl_out_stage[lcl_off];
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
#if !MLO_DIVBY4

					// if the last one
					if (pix_id == MLO_MAP_SZ4 - 1)
					{
						for (int j = 0; j < MLO_C1x1_PIXLEFT; ++j)
						{
							out_ptr[out_off2 + j] = ((_FLOAT*)&sum)[j] + bias_val;

						}

					}
					else
#endif
					{

						*((_FLOAT4*)&out_ptr[out_off2]) = (sum + (_FLOAT4)bias_val);
					}


#if 0
					if (get_group_id(0) == 0 && get_group_id(1) == 0  && get_group_id(2) == 0 && in_map_id == 0 && pix_id == 0 && olc == 0 && ib ==0)
					{
						printf("k:o: %d %d %d %d %d %d   %11.10f %f %11.10f\n",
							out_off2,
							get_group_id(0),
							get_group_id(1),
							in_map_id,
							pix_id,
							olc,
							out_ptr[out_off2],
							((_FLOAT*)&sum)[0],
							bias_val
							);
					}
#endif

				} //if (true

			} // if (in_map_id < MLO_EXCHANGE_STEP)

		} // for (int t = 0, p = 0; t < MLO_N_LCL_OUT_MAPS; t += MLO_EXCHANGE_STEP)

	} // 	for (int ib = 0; ib < MLO_N_LCL_BATCHS; ++ib)


#else




	int out_off1 = out_off;
	for (int ib = 0; ib < MLO_N_LCL_BATCHS
#if MLO_BATCH_ALIGNED == 0
		&& (batch_block*MLO_N_LCL_BATCHS + ib < MLO_BATCH_SZ)
#endif
		; ++ib, out_off1 += MLO_OUT_BATCH_STRIDE)
	{




		int out_off2 = out_off1;
		for (int olc = 0; olc < MLO_N_LCL_OUT_MAPS
#if MLO_OUTPUTS_ALIGNED == 0
			&& out_block + olc < MLO_N_OUTPUTS
#endif
			; ++olc, out_off2 += MLO_OUT_CHANNEL_STRIDE)
		{

			_FLOAT  bias_val = 0;
#if MLO_CONV_BIAS
			bias_val = bias[out_block* MLO_N_LCL_OUT_MAPS + olc];
#endif
#if !MLO_DIVBY4

			// if the last one
			if (pix_id == MLO_MAP_SZ4 - 1)
			{
				for (int j = 0; j < MLO_C1x1_PIXLEFT; ++j)
				{
					out_ptr[out_off2 + j] = ((_FLOAT*)&out_tiles[ib][olc])[j] + bias_val;

				}

			}
			else
#endif
			{

				*((_FLOAT4*)&out_ptr[out_off2]) = (out_tiles[ib][olc] + (_FLOAT4)bias_val);
			}


#if 0
			if (get_group_id(0) == 0 && get_group_id(1) == 0 && get_group_id(2) == 0 && in_map_id == 0 && pix_id == 0 && olc == 0)
			{
				printf("k:o: %d %d %d %d %d %d   %11.10f %f %11.10f\n",
					out_off2,
					get_group_id(0),
					get_group_id(1),
					in_map_id,
					pix_id,
					olc,
					out_ptr[out_off2],
					((_FLOAT*)&out_tiles[ib][olc])[0],
					bias_val
					);
			}
#endif

		}

	}


#endif

}
