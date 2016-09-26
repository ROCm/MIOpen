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

__kernel void MLOpenConv1x1(
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
	__local _FLOAT lcl_wei_stage[MLO_N_MAPS_PERGROUP*MLO_N_LCL_OUT_MAPS][MLO_N_LCL_IN_MAPS*MLO_N_MAPS_PERGROUP];
#if MLO_N_MAPS_PERGROUP > 1
	__local _FLOAT4 lcl_in_stage[MLO_GRP_SZ0];

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
	out_block = out_grp_block * MLO_N_MAPS_PERGROUP + in_map_id;
//#endif

	int in_off = batch_block * MLO_N_LCL_BATCHS * MLO_IN_BATCH_STRIDE
		+ in_map_id * MLO_IN_CHANNEL_STRIDE
				+ pix_id * 4;

	int wei_off = out_grp_block * MLO_N_MAPS_PERGROUP * MLO_N_LCL_OUT_MAPS * MLO_WEI_BSTRIDE;
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
		wei_off += MLO_N_LCL_IN_MAPS * MLO_N_MAPS_PERGROUP * MLO_WEI_CHANNEL_STRIDE
//#if MLO_DIR_FORWARD==0
//			* MLO_N_OUTPUTS
//#endif
		)
	{
		barrier(CLK_LOCAL_MEM_FENCE);

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
				in_stage[ib][ilc] = 0;
				if (c*MLO_N_LCL_IN_MAPS * MLO_N_MAPS_PERGROUP + in_map_id + ilc* MLO_N_MAPS_PERGROUP < MLO_N_INPUTS)
				{

					in_stage[ib][ilc] = *(_FLOAT4*)&in_ptr[in_off2];

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
				/*
				if(c*MLO_N_LCL_IN_MAPS * MLO_N_MAPS_PERGROUP + in_map_id + ilc* MLO_N_MAPS_PERGROUP >= MLO_N_INPUTS)
				{
					in_stage[ib][ilc] = 0;
				}
				*/

			}
		}

// read weights
// KCRS weights layout
		int wei_off1 = wei_off;

		for (int l = lcl_id0; l < MLO_N_MAPS_PERGROUP * MLO_N_LCL_OUT_MAPS * MLO_N_MAPS_PERGROUP * MLO_N_LCL_IN_MAPS; l += MLO_GRP_SZ0)
		{
			int o = (int)(((float)l + 0.000001f) / (MLO_N_LCL_IN_MAPS*MLO_N_MAPS_PERGROUP));
			int i = -mad24(o,(int)(MLO_N_LCL_IN_MAPS*MLO_N_MAPS_PERGROUP),-l);
			lcl_wei_stage[o][i] = wei_ptr[wei_off1 + mad24(o, (int)MLO_WEI_BSTRIDE,i)];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

// convolve
		for (int olc = 0; olc < MLO_N_LCL_OUT_MAPS; ++olc)
		{
			// lcl in maps (in data tiles) is has the stride = MLO_N_MAPS_PERGROUP, weights are mapped accordingly

			for (int ilc = 0; ilc < MLO_N_LCL_IN_MAPS; ++ilc)
			{
				// read weights
				wei_stage[olc][ilc] = lcl_wei_stage[mad24(in_map_id, (int)MLO_N_LCL_OUT_MAPS, olc)][mad24(in_map_id, (int)MLO_N_LCL_IN_MAPS, ilc)];
				for (int ib = 0; ib < MLO_N_LCL_BATCHS;  ++ib)
				{
					out_tiles[ib][olc] += in_stage[ib][ilc] * (_FLOAT4)wei_stage[olc][ilc];
#if 0
					if (get_group_id(0) == 312 && get_group_id(1) == 0 && in_map_id == 0 && pix_id == 80065 && olc == 0)
					{
						printf("k:c: 0 %d %d %d %d  %11.10f %11.10f %f %f\n",
							wei_off2,
							get_local_id(0),
							olc,
							ilc,
							out_tiles[ib][olc].s2,
							in_stage[ib][ilc].s2 * wei_stage[olc][ilc],
							in_stage[ib][ilc].s2,
							wei_stage[olc][ilc]
							);
					}
#endif

				}
			}
		}

#if MLO_N_MAPS_PERGROUP > 1
// exchange inputs with other MLO_N_MAPS_PERGROUP - 1 maps 
		for (int im = 1; im < MLO_N_MAPS_PERGROUP; ++im)
		{

// mov to different output block
			// exchange data
//			if(pix_id < MLO_MAP_SZ4)
			{
				for(int ib = 0; ib < MLO_N_LCL_BATCHS; ++ib)
				{
					for(int ilc = 0; ilc < MLO_N_LCL_IN_MAPS; ++ilc)
					{
						barrier(CLK_LOCAL_MEM_FENCE);
						lcl_in_stage[lcl_id0] = in_stage[ib][ilc];
						int lcl_off = lcl_id0 + MLO_MAP_SZ4;
						lcl_off = (lcl_off >= MLO_N_MAPS_PERGROUP * MLO_MAP_SZ4) ? lcl_off - MLO_N_MAPS_PERGROUP * MLO_MAP_SZ4 : lcl_off;
						barrier(CLK_LOCAL_MEM_FENCE);
						in_stage[ib][ilc] = lcl_in_stage[lcl_off];
					}
				}
			}
			int imp = in_map_id + im;
			imp = (imp >= MLO_N_MAPS_PERGROUP) ? imp - MLO_N_MAPS_PERGROUP :  imp;


			// convolve
//			int wei_off1 = wei_off + in_map_id * MLO_WEI_BSTRIDE + imp * MLO_WEI_CHANNEL_STRIDE;
			for (int olc = 0; olc < MLO_N_LCL_OUT_MAPS; ++olc/*, wei_off1 += MLO_WEI_BSTRIDE*/)
			{
				int wei_off2 = wei_off1;
				for (int ilc = 0; ilc < MLO_N_LCL_IN_MAPS; ++ilc/*, wei_off2 += MLO_WEI_CHANNEL_STRIDE * MLO_N_MAPS_PERGROUP*/)
				{
					// read weights
//					wei_stage[olc][ilc] = wei_ptr[wei_off2];
					wei_stage[olc][ilc] = lcl_wei_stage[mad24(in_map_id, (int)MLO_N_LCL_OUT_MAPS, olc)][mad24(imp, (int)MLO_N_LCL_IN_MAPS, ilc)];
					for (int ib = 0; ib < MLO_N_LCL_BATCHS;++ib)
					{
						out_tiles[ib][olc] += in_stage[ib][ilc] * (_FLOAT4)wei_stage[olc][ilc];
#if 0
						if (get_group_id(0) == 0 && get_group_id(1) == 0 && in_map_id == 4 && pix_id == 27 && olc == 1 )
						{
							printf("k:c: %d %d %d %d %d  %11.10f %11.10f %f %f\n",
								im,
								wei_off2,
								get_local_id(0),
								olc,
								ilc,
								out_tiles[ib][olc].s1,
								in_stage[ib][ilc].s1*wei_stage[olc][ilc],
								in_stage[ib][ilc].s1,
								wei_stage[olc][ilc]
								);
					}
#endif

					}
				}
			}
	

		}


#endif

	}

	if (in_map_id >= MLO_N_MAPS_PERGROUP || in_map_id*MLO_N_LCL_IN_MAPS >= MLO_N_INPUTS)
	{
		return;
	}

	out_block = out_grp_block * MLO_N_MAPS_PERGROUP + in_map_id;
	int out_off = batch_block * MLO_N_LCL_BATCHS * MLO_OUT_BATCH_STRIDE
		+ out_block * MLO_N_LCL_OUT_MAPS * MLO_OUT_CHANNEL_STRIDE
		+ pix_id * 4;


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
			&& out_block* MLO_N_LCL_OUT_MAPS + olc < MLO_N_OUTPUTS
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
			if (out_block* MLO_N_LCL_OUT_MAPS + olc == 0 && pix_id == (444*720 +582) / 4)
			{
				printf("k:o: %d %d %d %d %d %d   %11.10f %f %11.10f\n",
					out_off2 + 2,
					get_group_id(0),
					get_group_id(1),
					in_map_id,
					pix_id,
					olc,
					out_ptr[out_off2 + 2],
					((_FLOAT*)&out_tiles[ib][olc])[2],
					bias_val
					);
			}
#endif

		}

	}

}
