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

#ifndef MLO_FILTER_STRIDE0
#define MLO_FILTER_STRIDE0 1
#endif
#ifndef MLO_FILTER_STRIDE1
#define MLO_FILTER_STRIDE1 1
#endif

#define MLO_FILTER_SZ (MLO_FILTER_SIZE1*MLO_FILTER_SIZE0)

#define MLO_GRP_SZ0  (MLO_GRP_TILE0*MLO_GRP_TILE1)
#define MLO_GRP_SZ1 1
#define MLO_GRP_SZ2 1
#define MLO_GRP_SZ (MLO_GRP_SZ0*MLO_GRP_SZ1*MLO_GRP_SZ2)
#define MLO_N_PROC_WAVES ((MLO_GRP_SZ + MLO_N_READ_PROCS - 1)/MLO_N_READ_PROCS)
#define MLO_OUT_TILE_SZ (MLO_OUT_TILE1*MLO_OUT_TILE0)
#define MLO_ALU_TILE_SZ (MLO_ALU_VTILE1*MLO_ALU_VTILE0)


#if MLO_IN_TILE0 < MLO_OUT_WIDTH || MLO_IN_TILE1 < MLO_OUT_HEIGHT
#define MLO_LARGE_MAP 1
#else
#define MLO_LARGE_MAP 0
#endif


#if (MLO_IN_WIDTH == MLO_OUT_WIDTH && (MLO_IN_WIDTH / MLO_IN_TILE0 ) * MLO_IN_TILE0 == MLO_IN_WIDTH && MLO_IN_HEIGHT == MLO_OUT_HEIGHT && (MLO_IN_HEIGHT / MLO_IN_TILE1 ) * MLO_IN_TILE1 == MLO_IN_HEIGHT)
#define MLO_OUT_ALIGNED 1
#else
#define MLO_OUT_ALIGNED 0
#endif


#define MLO_N_ALUTILES_TOTAL ((MLO_GRP_TILE0*MLO_GRP_TILE1)/(MLO_ALU_TILE_SZ))
#define MLO_N_ALUTILES_PERSTACK (MLO_N_ALUTILES_TOTAL/MLO_N_STACKS)
#define MLO_ALUTILES_STACK_SZ (MLO_N_ALUTILES_PERSTACK*MLO_ALU_TILE_SZ)
#define MLO_N_IN_TILES_TOTAL (MLO_N_IN_TILES_PERSTACK*MLO_N_STACKS)
/*
#define MLO_N_OUT_TILES_PERSTACK (MLO_N_OUT_TILES*MLO_N_ALUTILES_PERSTACK)
#if MLO_N_OUT_TILES_PERSTACK > MLO_N_OUTPUTS
#undef MLO_N_OUT_TILES_PERSTACK
#define MLO_N_OUT_TILES_PERSTACK MLO_N_OUTPUTS
#endif 
*/
#define MLO_N_OUT_TILE_BLOCKS0 ((MLO_OUT_WIDTH+MLO_IN_TILE0-1)/MLO_IN_TILE0)
#define MLO_N_OUT_TILE_BLOCKS1 ((MLO_OUT_HEIGHT+MLO_IN_TILE1-1)/MLO_IN_TILE1)
#define MLO_N_IN_PACKS  ((MLO_N_INPUTS+MLO_N_IN_TILES_PERSTACK-1)/MLO_N_IN_TILES_PERSTACK)

#define MLO_N_IN_READ (MLO_N_IN_PACKS*MLO_N_IN_TILES_PERSTACK)
#if MLO_N_IN_READ == MLO_N_INPUTS
#define MLO_INPUTS_ALIGNED 1
#else
#define MLO_INPUTS_ALIGNED 0
#endif

#define MLO_N_OUT_PACKS  (MLO_N_OUTPUTS/MLO_N_OUT_TILES_PERSTACK)
#if MLO_N_OUT_PACKS*MLO_N_OUT_TILES_PERSTACK == MLO_N_OUTPUTS && MLO_N_OUT_TILES_PERSTACK != MLO_N_OUTPUTS
#define MLO_OUTPUTS_ALIGNED 1
#else
#define MLO_OUTPUTS_ALIGNED 0
#endif

#define MLO_N_BATCH_PACKS (MLO_BATCH_SZ/MLO_N_STACKS)
#if MLO_N_BATCH_PACKS*MLO_N_STACKS == MLO_BATCH_SZ && MLO_N_STACKS != MLO_BATCH_SZ
#define MLO_BATCH_ALIGNED 1
#else
#define MLO_BATCH_ALIGNED 0
#endif



#if MLO_DIR_FORWARD == 1
#define MLO_IN_LCL_WIDTH  ((MLO_IN_TILE0 - 1) * MLO_FILTER_STRIDE0 + MLO_FILTER_SIZE0)  // here we use kernel size. it's important when padding == 0  2* MLO_FILTER_PAD0
#define MLO_IN_LCL_HEIGHT ((MLO_IN_TILE1 - 1) * MLO_FILTER_STRIDE1 + MLO_FILTER_SIZE1)
#else
#define MLO_IN_LCL_WIDTH  ((MLO_IN_TILE0 + MLO_FILTER_SIZE0 - 1 + MLO_FILTER_STRIDE0 - 1) / MLO_FILTER_STRIDE0) // here we use kernel size. it's important when padding == 0  2* MLO_FILTER_PAD0
#define MLO_IN_LCL_HEIGHT ((MLO_IN_TILE1 + MLO_FILTER_SIZE1 - 1 + MLO_FILTER_STRIDE1 - 1) / MLO_FILTER_STRIDE1)
#endif
#define MLO_IN_LCL_TILE_SZ (MLO_IN_LCL_WIDTH*MLO_IN_LCL_HEIGHT)
#define MLO_IN_LCL_PERSTACK_SZ (MLO_IN_LCL_TILE_SZ*MLO_N_IN_TILES_PERSTACK)
#define MLO_IN_LCL_SZ (MLO_IN_LCL_PERSTACK_SZ*MLO_N_STACKS)

#define MLO_WEIGHTS_SZ (MLO_N_OUT_TILES_PERSTACK*MLO_N_IN_TILES_PERSTACK*MLO_FILTER_SZ)

#define MLO_PVT_ACCUM_DATA_SZ (MLO_N_OUT_TILES * MLO_OUT_TILE_SZ)
#if MLO_DIR_FORWARD == 1
#define MLO_PVT_IN_WIDTH  ((MLO_OUT_TILE0 - 1) * MLO_FILTER_STRIDE0 + MLO_FILTER_SIZE0)
#define MLO_PVT_IN_HEIGHT ((MLO_OUT_TILE1 - 1) * MLO_FILTER_STRIDE1 + 1)
#else
#define MLO_PVT_IN_WIDTH  ((MLO_OUT_TILE0 + MLO_FILTER_SIZE0 - 1 + MLO_FILTER_STRIDE0 - 1) / MLO_FILTER_STRIDE0)
#define MLO_PVT_IN_HEIGHT ((MLO_OUT_TILE1 +                        MLO_FILTER_STRIDE1 - 1) / MLO_FILTER_STRIDE1)
#endif

#define MLO_LCL_WEIGHTS 1


static inline void calculateXYPos(int linPos, int width, int *x, int *y)
{
	(*y) = (int)((float)linPos / (float)width + 0.00001f);
	(*x) = linPos - mul24((*y), width); 
}

static inline int calculateOffset(int stride, int x, int y)
{
	int ret = y * stride + x;
	return(ret);
}

static inline void readDataElem(int linPos,__local _FLOAT *lcl_data, int lcl_base, int lcl_height, int lcl_width, int lcl_stride, int lcl_y, int lcl_x,
					 const __global _FLOAT * gbl_data, int gbl_base, int gbl_height, int gbl_width, int gbl_stride, int gbl_y, int gbl_x,
					 bool vis,
					 bool debug)
{
	int x, y;
	calculateXYPos(linPos, lcl_width, &x, &y);
	int g_x = x + gbl_x;
	int g_y = y + gbl_y;
	int gbl_off0 = calculateOffset(gbl_stride, g_x, g_y);
	int gbl_off = gbl_off0 + gbl_base;

	int l_x = x + lcl_x;
	int l_y = y + lcl_y;
	int lcl_off = lcl_base +
#if MLO_LARGE_MAP == 1
		linPos
#else
		mad24(l_y, lcl_stride, l_x);
#endif
	 ;

#if MLO_LARGE_MAP == 1
	 vis &= (g_x >= 0 && g_x < gbl_width && g_y >= 0 && g_y < gbl_height);
#endif
	 gbl_off = (vis) ? gbl_off : 0;
	 _FLOAT gbl_val = gbl_data[gbl_off];
	 gbl_val = (vis) ? gbl_val : 0;

	lcl_data[lcl_off] = gbl_val;

}


static inline void readData(int lcl_id, int size, int lcl_p_stride, __local _FLOAT *lcl_data, int lcl_base, int lcl_height, int lcl_width, int lcl_stride, int lcl_y, int lcl_x,
					 const __global _FLOAT * gbl_data, int gbl_base, int gbl_height, int gbl_width, int gbl_stride, int gbl_y, int gbl_x,
					 bool vis,
					 bool debug
					 )
{
	
	for(int i = lcl_id; i < size; i+= lcl_p_stride)
	{
		readDataElem(i, lcl_data, lcl_base, lcl_height, lcl_width, lcl_stride, lcl_y, lcl_x,
					 gbl_data, gbl_base, gbl_height, gbl_width, gbl_stride, gbl_y, gbl_x,
					 vis,
					 debug);
	}

}

static inline void loadData(int lcl_id, int lcl_p_stride,
					__local _FLOAT * lcl_data,
					int lcl_off, int lcl_size, int lcl_height, int lcl_width, int lcl_stride, int lcl_bot_y, int lcl_bot_x,
					const __global _FLOAT * gbl_data,
					int gbl_off, int gbl_size, int gbl_height, int glb_width, int gbl_stride, int gbl_bot_y, int gbl_bot_x,
					int buf_block_ind, int max_n_bufs, int lcl_n_bufs,
					bool debug)
{


	for(int c = 0; c < lcl_n_bufs;
		 ++c, lcl_off += lcl_size, gbl_off += gbl_size )
	{
		bool vis = (buf_block_ind + c <  max_n_bufs);
		readData(lcl_id, lcl_size, lcl_p_stride,
				 lcl_data, lcl_off, lcl_height, lcl_width, lcl_stride, lcl_bot_y, lcl_bot_x,
				 gbl_data, gbl_off, gbl_height, glb_width, gbl_stride, gbl_bot_y, gbl_bot_x,
				 vis,
				 (debug));
	}

}


static inline void Conv(int o_map_base,
				int in_stg_off,
				__private _FLOAT *pvt_in_stage, __local _FLOAT * lcl_indata,
				__private _FLOAT *pvt_wei_stage, __local _FLOAT * lcl_wei,
				__private _FLOAT *pvt_accum
				 )
{
// convolution

		// over all inputs in stack
		int in_stg_off1 = in_stg_off;
		for(int i_c = 0; i_c < MLO_N_IN_TILES_PERSTACK; ++i_c, in_stg_off1 += MLO_IN_LCL_TILE_SZ)
		{
        // preload input		
			int wei_stg_base_off = mad24(o_map_base, (int)(MLO_N_IN_TILES_PERSTACK*MLO_FILTER_SZ), mul24(i_c,(int)MLO_FILTER_SZ));
			int in_stg_off2 = in_stg_off1;
			for(int j = 0; j < MLO_PVT_IN_HEIGHT-1; ++j, in_stg_off2+=MLO_IN_LCL_WIDTH
					)
			{
				for(int i = 0; i < MLO_PVT_IN_WIDTH; ++i)
				{
					pvt_in_stage[j*MLO_PVT_IN_WIDTH + i] = lcl_indata[in_stg_off2 + i];
				}
			}

		// over filter rows
#if MLO_DIR_FORWARD == 1
			for(int k = 0; k < MLO_FILTER_SIZE1; ++k, in_stg_off2+=MLO_IN_LCL_WIDTH
			)
#else
			for(int k = 0; k < MLO_FILTER_SIZE1; ++k, in_stg_off2 += (((k + (MLO_FILTER_SIZE1%MLO_OUT_TILE1)) % MLO_FILTER_STRIDE1) ? 0 : MLO_IN_LCL_WIDTH)
			)
#endif
			{	
				int k_act = 0;
#if MLO_DIR_FORWARD==1
				k_act = k;
#else
// load filter in reverse order
				k_act = MLO_FILTER_SIZE1 - 1 - k;
#endif
		// load next input row
				for(int i_pvt = 0; i_pvt < MLO_PVT_IN_WIDTH; ++i_pvt)
				{
					pvt_in_stage[(MLO_PVT_IN_HEIGHT-1)*MLO_PVT_IN_WIDTH + i_pvt] = lcl_indata[in_stg_off2 + i_pvt];
				}
				
		// over all outputs
				for(int o_c = 0; o_c < MLO_N_OUT_TILES; ++o_c)
				{
					int wei_stg_off = wei_stg_base_off + o_c * MLO_N_IN_TILES_PERSTACK*MLO_FILTER_SZ + k_act * MLO_FILTER_SIZE0;
					for(int i = 0; i < MLO_FILTER_SIZE0; ++i)
					{
						pvt_wei_stage[i] =  lcl_wei[wei_stg_off + i]; //(float)o_c/(float)MLO_N_OUT_TILES + (float)(i+k)/9;  
					}
				

		// actual conv

					for( int j = 0; j < MLO_OUT_TILE1; ++j)
					{
#if MLO_DIR_FORWARD == 0
						if (((j + k + 1 + (MLO_FILTER_SIZE1 % MLO_FILTER_STRIDE1)) % MLO_FILTER_STRIDE1) == 0)
#endif
						for(int i = 0; i < MLO_OUT_TILE0; ++i)
						{
							for(int l = 0; l < MLO_FILTER_SIZE0; ++l)
							{

								int l_act = 0;
#if MLO_DIR_FORWARD==1
								l_act = l;

#else
// in reverse horizontal and vertical orders
								l_act = MLO_FILTER_SIZE0 - 1 - l;

#endif

#if MLO_DIR_FORWARD == 1
								pvt_accum[(o_c * MLO_OUT_TILE1 + j) * MLO_OUT_TILE0 + i]
								     += pvt_in_stage[j * MLO_PVT_IN_WIDTH * MLO_FILTER_STRIDE1 + i * MLO_FILTER_STRIDE0 + l] * pvt_wei_stage[l_act];
#else
								if (((i + l + 1 + (MLO_FILTER_SIZE0 % MLO_FILTER_STRIDE0)) % MLO_FILTER_STRIDE0) == 0)
								{
									pvt_accum[(o_c * MLO_OUT_TILE1 + j) * MLO_OUT_TILE0 + i]
										+= pvt_in_stage[(j / MLO_FILTER_STRIDE1) * MLO_PVT_IN_WIDTH + (i + l) / MLO_FILTER_STRIDE0] * pvt_wei_stage[l_act];
								}
#endif
							}
						}

					}



				} // for(int o_c = 0; o_c < MLO_N_OUT_TILES; ++o_c)

               // move data up 
				for(int j = 0; j < MLO_PVT_IN_HEIGHT-1; ++j)
				{
					for(int i = 0; i < MLO_PVT_IN_WIDTH; ++i)
					{
						pvt_in_stage[j*MLO_PVT_IN_WIDTH + i] = pvt_in_stage[(j+1)*MLO_PVT_IN_WIDTH + i];
					}
				}


			} // for(int k = 0; k < MLO_FILER_SIZE1; ++k,in_stg_off2+=MLO_IN_LCL_WIDTH)		
		
		} // for(int i_c = 0; i_c < MLO_N_IN_TILES_PERSTACK; ++i_c, in_stg_off1 += MLO_IN_LCL_PERSTACK_SZ)

}

__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2)))
__kernel void MLOpenConvUni(
       const __global _FLOAT * in,
       const __global _FLOAT * weights,
#if MLO_CONV_BIAS
       const __global _FLOAT * bias,
#endif
 	  __global _FLOAT *out,
	   _FLOAT padding_val
	   )
{
	__local _FLOAT lcl_indata[MLO_IN_LCL_SZ];
	__local _FLOAT lcl_wei[MLO_WEIGHTS_SZ];
	__private _FLOAT pvt_accum[MLO_PVT_ACCUM_DATA_SZ];
	__private _FLOAT pvt_in_stage[MLO_PVT_IN_HEIGHT * MLO_PVT_IN_WIDTH];
	__private _FLOAT pvt_wei_stage[MLO_FILTER_SIZE0];


	int grp_id0 = get_group_id(0);
	int y_tile_blk = (int)((float)grp_id0 / (float)MLO_N_OUT_TILE_BLOCKS0 + 0.00001f);
	int x_tile_blk = -mad24(y_tile_blk, (int)MLO_N_OUT_TILE_BLOCKS0, -grp_id0);
	int o_pack = get_group_id(1); // block of outputs
	int b_pack = get_group_id(2); // batch block

	int lcl_id = get_local_id(0);
	int stack = (int)((float)lcl_id/(float)MLO_ALUTILES_STACK_SZ + 0.00001f);  // stack
	int alu_stack_id = -mad24(stack, (int)MLO_ALUTILES_STACK_SZ, -lcl_id);  // alu index in stack
// ALU plane inside stack
	int alu_out_plane_id = (int)((float)alu_stack_id / (float)MLO_ALU_TILE_SZ + 0.00001f);  // alu output plane index
	int alu_out_id = -mad24(alu_out_plane_id, (int)MLO_ALU_TILE_SZ, -alu_stack_id); // alu index inside an ALU output plane
// pos inside ALU tile
	int alu_tl1 = (int)((float)alu_out_id/(float)MLO_ALU_VTILE0 + 0.00001f);
	int alu_tl0 = -mad24(alu_tl1, (int)MLO_ALU_VTILE0, -alu_out_id);

	int o_map_plane = o_pack * MLO_N_OUT_TILES_PERSTACK; // first output maps index per full ALU plane stack
	int o_map_base = alu_out_plane_id*MLO_N_OUT_TILES;  // local output map offset
	int o_map = o_map_plane + o_map_base; // output map index per ALU plane
	int b_index = b_pack * MLO_N_STACKS;

	int wave_id = (int)((float)lcl_id / (float)MLO_N_READ_PROCS + 0.00001f);
	int wave_lcl_id = -mad24(wave_id, (int)MLO_N_READ_PROCS, -lcl_id);

#if MLO_DIR_FORWARD == 1
	int x_grp = x_tile_blk * MLO_IN_TILE0 * MLO_FILTER_STRIDE0;
	int y_grp = y_tile_blk * MLO_IN_TILE1 * MLO_FILTER_STRIDE1;

	int x_in_grp = x_grp - MLO_FILTER_PAD0;
	int y_in_grp = y_grp - MLO_FILTER_PAD1;

	int x_in_lcl = alu_tl0 * MLO_OUT_TILE0 * MLO_FILTER_STRIDE0;
	int y_in_lcl = alu_tl1 * MLO_OUT_TILE1 * MLO_FILTER_STRIDE1;
#else
	int x_grp = x_tile_blk * (MLO_IN_TILE0 / MLO_FILTER_STRIDE0);
	int y_grp = y_tile_blk * (MLO_IN_TILE1 / MLO_FILTER_STRIDE1);
	int x_in_grp = x_grp - (MLO_FILTER_PAD0 / MLO_FILTER_STRIDE0);
	int y_in_grp = y_grp - (MLO_FILTER_PAD1 / MLO_FILTER_STRIDE1);
	int x_in_lcl = alu_tl0 * (MLO_OUT_TILE0 / MLO_FILTER_STRIDE0);
	int y_in_lcl = alu_tl1 * (MLO_OUT_TILE1 / MLO_FILTER_STRIDE1);
#endif

// base offset to read data from local input data
	int in_stg_off = stack*MLO_IN_LCL_PERSTACK_SZ + (y_in_lcl) * MLO_IN_LCL_WIDTH + x_in_lcl;

    int in_off = b_index * MLO_IN_BATCH_STRIDE;

	
#if MLO_DIR_FORWARD==1
	int wei_off = mul24(o_map_plane, MLO_N_INPUTS * MLO_FILTER_SZ);
#else
	int wei_off = mul24(o_map_plane,MLO_FILTER_SZ);
#endif
	

#if MLO_LARGE_MAP == 0
	for(int i = lcl_id; i < MLO_IN_LCL_SZ;  i += MLO_GRP_SZ)
	{
		lcl_indata[i] = 0;
	}
#endif

	for(int i = 0; i < MLO_PVT_ACCUM_DATA_SZ; ++i)
	{
		pvt_accum[i] = 0;
	}

	for(int ic = 0; ic < MLO_N_INPUTS; ic += MLO_N_IN_TILES_PERSTACK, in_off += MLO_IN_CHANNEL_STRIDE*MLO_N_IN_TILES_PERSTACK,
				wei_off += MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ
#if MLO_DIR_FORWARD==0
							* MLO_N_OUTPUTS
#endif
	)
	{
		barrier(CLK_LOCAL_MEM_FENCE);

// small map has been read in full continiously into the lDS buffer within padded rect,
// padding has been done on initilization.
// large map calculates padding on the fly and fills it with 0.


#if 1 // all inputs

#if MLO_LARGE_MAP == 1
		int in_lcl_off1 = 0;
		int in_off1 = in_off;
		for(int i_b = 0; i_b < MLO_N_STACKS; ++i_b, in_off1 += MLO_IN_BATCH_STRIDE, in_lcl_off1 += MLO_IN_LCL_PERSTACK_SZ)
		{
			bool vis = true;
#if MLO_BATCH_ALIGNED == 0
			vis &= (b_index + i_b < MLO_BATCH_SZ);
#endif

// over all inputs in stack
			int in_off2 = in_off1;
			int in_lcl_off2 = in_lcl_off1;
			for(int i_c = 0; i_c < MLO_N_IN_TILES_PERSTACK; ++i_c, in_off2 += MLO_IN_CHANNEL_STRIDE, in_lcl_off2 += MLO_IN_LCL_TILE_SZ)
			{
#if MLO_INPUTS_ALIGNED == 0
				vis &= (ic + i_c < MLO_N_INPUTS);
#endif
		
				int elem_id = lcl_id;
				int lcl_p_stride = MLO_GRP_SZ0;
				int lcl_base = 0;
				int lcl_y = 0;
				int lcl_x = 0;
				int gbl_base = 0;

				readData(elem_id, (MLO_IN_LCL_HEIGHT * MLO_IN_LCL_WIDTH), lcl_p_stride,
						&lcl_indata[in_lcl_off2], lcl_base, MLO_IN_LCL_HEIGHT, MLO_IN_LCL_WIDTH, MLO_IN_LCL_WIDTH, lcl_y, lcl_x,
						&in[in_off2], gbl_base, MLO_IN_HEIGHT, MLO_IN_WIDTH, MLO_IN_STRIDE, y_in_grp, x_in_grp,
						vis,
						true
					 );
			}

		}
#else
		for(int i = wave_id; i < MLO_N_IN_TILES_TOTAL;  i += MLO_N_PROC_WAVES)
		{
			int i_b = (int)((float)i/ (float)MLO_N_IN_TILES_PERSTACK + 0.00001f);
			int i_c = i - mul24(i_b, (int)MLO_N_IN_TILES_PERSTACK);

			bool vis = true;

#if MLO_BATCH_ALIGNED == 0
			vis &= (b_index + i_b < MLO_BATCH_SZ);
#endif

#if MLO_INPUTS_ALIGNED == 0
			vis &= (ic + i_c < MLO_N_INPUTS);
#endif
			int in_off2 = in_off + i_b * MLO_IN_BATCH_STRIDE + i_c * MLO_IN_CHANNEL_STRIDE;
			int in_lcl_off2 = i_b * MLO_IN_LCL_PERSTACK_SZ + i_c * MLO_IN_LCL_TILE_SZ;

			int elem_id = wave_lcl_id;
			int lcl_p_stride = MLO_N_READ_PROCS;
			int lcl_base = 0;
#if MLO_DIR_FORWARD == 1
			int lcl_y = MLO_FILTER_PAD1;
			int lcl_x = MLO_FILTER_PAD0;
#else
			int lcl_y = (MLO_FILTER_PAD1 / MLO_FILTER_STRIDE0);
			int lcl_x = (MLO_FILTER_PAD0 / MLO_FILTER_STRIDE1);
#endif
			int gbl_base = 0;

			readData(elem_id, (MLO_IN_HEIGHT * MLO_IN_WIDTH), lcl_p_stride,
						&lcl_indata[in_lcl_off2], lcl_base, MLO_IN_HEIGHT, MLO_IN_WIDTH, MLO_IN_LCL_WIDTH, lcl_y, lcl_x,
						&in[in_off2], gbl_base, MLO_IN_HEIGHT, MLO_IN_WIDTH, MLO_IN_STRIDE, y_grp, x_grp,
						vis,
						true
					 );
		}
#endif





// read inputs and weights
// put weights into LDS 

#if 1  // only weights



		for(int i = lcl_id; i < MLO_WEIGHTS_SZ; i += MLO_GRP_SZ)
		{
#if MLO_DIR_FORWARD==1
// here is [tops][bottoms]
			int lcl_o = (int)((float)i/(float)(MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ) + 0.00001f);
			int gbl_i = i - mul24(lcl_o, (int)(MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ));
			lcl_wei[i] = weights[wei_off + lcl_o * MLO_N_INPUTS * MLO_FILTER_SZ + gbl_i];
#else
// outputs are botoms(inputs))
// inputs are tops(outputs)
			int lcl_o = (int)((float)i/ (float)(MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ) + 0.00001f);
			int gbl_i = i - mul24(lcl_o, (int)(MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ));
			int lcl_c = (int)((float)gbl_i/ (float)MLO_FILTER_SZ + 0.00001f);
			int lcl_i = gbl_i - mul24(lcl_c, (int)MLO_FILTER_SZ);

			int lcl_we_off = mad24(mad24(lcl_c, (int)MLO_N_IN_TILES_PERSTACK, lcl_o), (int)MLO_FILTER_SZ, lcl_i);
			int gbl_we_off = mad24(mad24(lcl_o, (int)MLO_N_OUTPUTS, lcl_c), (int)MLO_FILTER_SZ, wei_off + lcl_i);
			bool within_range = gbl_we_off < (MLO_N_OUTPUTS*MLO_N_INPUTS*MLO_FILTER_SZ);
			gbl_we_off = (within_range) ? gbl_we_off : 0;
			_FLOAT wei = weights[gbl_we_off];
			wei = (within_range) ? wei : 0;
			lcl_wei[lcl_we_off] 
				= wei;

#endif

		}

#endif

// over all batch stacks

#endif  // all input

		barrier(CLK_LOCAL_MEM_FENCE);

// convolution
		Conv(o_map_base,
			in_stg_off,
			pvt_in_stage, lcl_indata,
			pvt_wei_stage, lcl_wei,
			pvt_accum
			);


//		barrier(CLK_LOCAL_MEM_FENCE);	
	}
// write results out
#if MLO_DIR_FORWARD == 1
#if MLO_FILTER_STRIDE0 == 1
	int x_out_grp = x_grp;
#else
	int x_out_grp = x_tile_blk * MLO_IN_TILE0;
#endif
#if MLO_FILTER_STRIDE1 == 1
	int y_out_grp = y_grp;
#else
	int y_out_grp = y_tile_blk * MLO_IN_TILE1;
#endif
#else
	int x_out_grp = x_grp * MLO_FILTER_STRIDE0;
	int y_out_grp = y_grp * MLO_FILTER_STRIDE1;
#endif
	int x_out_lcl = alu_tl0 * MLO_OUT_TILE0;
	int y_out_lcl = alu_tl1 * MLO_OUT_TILE1;


    int out_off = (b_index + stack) * MLO_OUT_BATCH_STRIDE + o_map * MLO_OUT_CHANNEL_STRIDE + (y_out_grp + y_out_lcl) * MLO_OUT_STRIDE + x_out_grp + x_out_lcl;
// over all local stacks
#if MLO_BATCH_ALIGNED == 0
	if (b_index + stack < MLO_BATCH_SZ)
#endif
	{


// over all local outputs
		int out_off1 = out_off;
		for(int o = 0; o < MLO_N_OUT_TILES

						; ++o, out_off1 += MLO_OUT_CHANNEL_STRIDE
						)
		{
// over output tile


			int out_off2 = out_off1;
			for( int j = 0; j < MLO_OUT_TILE1; ++j, out_off2 += MLO_OUT_STRIDE)
			{
				for(int i = 0; i < MLO_OUT_TILE0; ++i)
				{
				if (true 
#if MLO_OUT_ALIGNED == 0
					&& y_out_grp + y_out_lcl + j < MLO_OUT_HEIGHT &&  x_out_grp + x_out_lcl + i < MLO_OUT_WIDTH
#endif
#if MLO_OUTPUTS_ALIGNED == 0
						&& o_map + o < MLO_N_OUTPUTS
#endif
					)
					{
						out[out_off2 + i] = pvt_accum[o*MLO_OUT_TILE_SZ + j * MLO_OUT_TILE0 + i]
#if MLO_CONV_BIAS
						+ bias_val = bias[o_map + o];
#endif
						 ;

					}

				}
			}

		}
	}


}
