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


//#define MLO_HW_WAVE_SZ  64

// #define MLO_DIR_FORWARD 0
//#define MLO_N_OUTPUTS  8
//#define MLO_N_INPUTS   8
//#define MLO_BATCH_SZ   10

//#define MLO_OUT_WIDTH 32    
//#define MLO_OUT_HEIGHT 32    

//#define MLO_OUT_STRIDE (MLO_OUT_WIDTH)
//#define MLO_OUT_CHANNEL_STRIDE (MLO_OUT_WIDTH*MLO_OUT_HEIGHT)
//#define MLO_OUT_BATCH_STRIDE (MLO_OUT_STRIDE*MLO_N_OUTPUTS)

//#define MLO_IN_WIDTH 32    
//#define MLO_IN_HEIGHT 32    
// temp
//#define MLO_IN_STRIDE (MLO_IN_WIDTH)
//#define MLO_IN_CHANNEL_STRIDE (MLO_IN_WIDTH*MLO_IN_HEIGHT)
//#define MLO_IN_BATCH_STRIDE (MLO_IN_STRIDE*MLO_N_INPUTS)


//#define MLO_FILER_PAD0 1
//#define MLO_FILER_STRIDE0 1
//#define MLO_FILER_SIZE0 3
//#define MLO_FILER_PAD1 1
//#define MLO_FILER_STRIDE1 1
//#define MLO_FILER_SIZE1 3
#define MLO_FILTER_SZ (MLO_FILTER_SIZE1*MLO_FILTER_SIZE0)

//#define MLO_GRP_TILE0 16
//#define MLO_GRP_TILE1 16

#define MLO_GRP_SZ0  (MLO_GRP_TILE0*MLO_GRP_TILE1)
#define MLO_GRP_SZ1 1
#define MLO_GRP_SZ2 1
#define MLO_GRP_SZ (MLO_GRP_SZ0*MLO_GRP_SZ1*MLO_GRP_SZ2)
#define MLO_N_PROC_WAVES ((MLO_GRP_SZ + MLO_N_READ_PROCS - 1)/MLO_N_READ_PROCS)

// input tile size
//#define MLO_IN_TILE0 16  // size of input data per ALU plane
//#define MLO_IN_TILE1 16

//#define MLO_OUT_TILE0  4      // size of ouptput tile per wk-item (ALU))
//#define MLO_OUT_TILE1  4
#define MLO_OUT_TILE_SZ (MLO_OUT_TILE1*MLO_OUT_TILE0)

//#define MLO_N_STACKS		2		// n of stacks per group
//#define MLO_N_OUT_TILES		4       // per wkitem (ALU), they stacked - it's different output
//#define MLO_N_IN_TILES_TOTAL 8



//#define MLO_ALU_VTILE0 (MLO_IN_TILE0/MLO_OUT_TILE0)      // size of ALU plane
//#define MLO_ALU_VTILE1 (MLO_IN_TILE1/MLO_OUT_TILE1)
#define MLO_ALU_TILE_SZ (MLO_ALU_VTILE1*MLO_ALU_VTILE0)


#if MLO_IN_TILE0 < MLO_IN_WIDTH || MLO_IN_TILE1 < MLO_IN_HEIGHT
#define MLO_LARGE_MAP 1
#else
#define MLO_LARGE_MAP 0
#endif


#if (MLO_IN_WIDTH == MLO_OUT_WIDTH && (MLO_IN_WIDTH + MLO_IN_TILE0 - 1) / MLO_IN_TILE0 ) * MLO_IN_TILE0 == MLO_IN_WIDTH && MLO_IN_HEIGHT == MLO_OUT_HEIGHT && ((MLO_IN_HEIGHT + MLO_IN_TILE1 - 1) / MLO_IN_TILE1 ) * MLO_IN_TILE1 == MLO_IN_HEIGHT
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

#define MLO_N_OUT_PACKS  ((MLO_N_OUTPUTS+MLO_N_OUT_TILES_PERSTACK-1)/MLO_N_OUT_TILES_PERSTACK)
#if MLO_N_OUT_PACKS*MLO_N_OUT_TILES_PERSTACK == MLO_N_OUTPUTS && MLO_N_OUT_TILES_PERSTACK != MLO_N_OUTPUTS
#define MLO_OUTPUTS_ALIGNED 1
#else
#define MLO_OUTPUTS_ALIGNED 0
#endif

#define MLO_N_BATCH_PACKS ((MLO_BATCH_SZ + MLO_N_STACKS - 1)/MLO_N_STACKS)
#if MLO_N_BATCH_PACKS*MLO_N_STACKS == MLO_BATCH_SZ && MLO_N_STACKS != MLO_BATCH_SZ
#define MLO_BATCH_ALIGNED 1
#else
#define MLO_BATCH_ALIGNED 0
#endif



#define MLO_IN_LCL_WIDTH (MLO_IN_TILE0 + MLO_FILTER_SIZE0 - 1)  // here we use kernel size. it's important when padding == 0 
#define MLO_IN_LCL_HEIGHT (MLO_IN_TILE1 + MLO_FILTER_SIZE1 - 1)
#define MLO_IN_LCL_TILE_SZ (MLO_IN_LCL_WIDTH*MLO_IN_LCL_HEIGHT)
#define MLO_IN_LCL_PERSTACK_SZ (MLO_IN_LCL_TILE_SZ*MLO_N_IN_TILES_PERSTACK)
#define MLO_IN_LCL_SZ (MLO_IN_LCL_PERSTACK_SZ*MLO_N_STACKS)

#define MLO_WEIGHTS_SZ (MLO_N_OUT_TILES_PERSTACK*MLO_N_IN_TILES_PERSTACK*MLO_FILTER_SZ)

#define MLO_PVT_ACCUM_DATA_SZ (MLO_N_OUT_TILES * MLO_OUT_TILE_SZ)
#define MLO_PVT_IN_WIDTH (MLO_FILTER_SIZE0 + MLO_OUT_TILE0 - 1)
#define MLO_PVT_IN_HEIGHT (MLO_OUT_TILE1)

#define MLO_LCL_WEIGHTS 1


inline void calculateXYPos(int linPos, int width, int *x, int *y)
{
	(*y) = linPos / width;
	(*x) = linPos - (*y) * width; 
}

inline int calculateOffset(int stride, int x, int y)
{
	int ret = y * stride + x;
	return(ret);
}

inline void readDataElem(int linPos,__local _FLOAT *lcl_data, int lcl_base, int lcl_height, int lcl_width, int lcl_stride, int lcl_y, int lcl_x,
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

	_FLOAT gbl_val = gbl_data[gbl_off];
#if MLO_LARGE_MAP == 1
	vis &= (g_x >= 0 && g_x < gbl_width && g_y >= 0 && g_y < gbl_height);
#endif
     gbl_val =  (vis) ? gbl_val : 0;

	lcl_data[lcl_off] = gbl_val;

#if 0
	if ( debug && get_group_id(0) == 0 && l_x < 9 && l_y < 1)
	{
		printf("K:in: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d  %f\n", linPos, lcl_width, x, y, l_y, l_x, lcl_off, g_y, g_x, gbl_y, gbl_x, gbl_off,
		 gbl_height, gbl_width, gbl_stride, gbl_val);
	}
#endif
}


inline void readData(int lcl_id, int size, int lcl_p_stride, __local _FLOAT *lcl_data, int lcl_base, int lcl_height, int lcl_width, int lcl_stride, int lcl_y, int lcl_x,
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

inline void loadData(int lcl_id, int lcl_p_stride,
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


inline void Conv(int o_map_base,
				int in_stg_off,
				 _FLOAT *pvt_in_stage, __local _FLOAT * lcl_indata,
				 _FLOAT *pvt_wei_stage, __local _FLOAT * lcl_wei,
				 _FLOAT *pvt_accum
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
			for(int k = 0; k < MLO_FILTER_SIZE1; ++k, in_stg_off2+=MLO_IN_LCL_WIDTH
			)
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
						pvt_wei_stage[i] = lcl_wei[wei_stg_off + i];
					}
				

		// actual conv

					for( int j = 0; j < MLO_OUT_TILE1; ++j)
					{
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

								pvt_accum[(o_c * MLO_OUT_TILE1 + j) * MLO_OUT_TILE0 + i]
									 += pvt_in_stage[j * MLO_PVT_IN_WIDTH + i + l] * pvt_wei_stage[l_act];
#if 0 //MLO_DIR_FORWARD==1
								if  (get_local_id(0) == 1 && get_group_id(0) == 0 /*&& alu_out_plane_id == 0 && alu_out_id == 0 */ && (i ==2 /*|| i == 1*/)&& j == 0)
								{
									printf("K: oc=%d ic=%d k=%d l=%d j=%d i=%d ai=%d di=%d  %f %f %f\n",
									o_c,
									i_c,
									k_act,
									l_act,
									j,
									i,
									(o_c * MLO_OUT_TILE1 + j) * MLO_OUT_TILE0 + i,
									j * MLO_PVT_IN_WIDTH + i + l,
									pvt_accum[(o_c * MLO_OUT_TILE1 + j) * MLO_OUT_TILE0 + i],								
									pvt_in_stage[j * MLO_PVT_IN_WIDTH + i + l],
									pvt_wei_stage[l_act]
									);
								}
#endif

							}


							mem_fence(CLK_LOCAL_MEM_FENCE);


						}

//						mem_fence(CLK_LOCAL_MEM_FENCE);
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

//				mem_fence(CLK_LOCAL_MEM_FENCE);


			} // for(int k = 0; k < MLO_FILER_SIZE1; ++k,in_stg_off2+=MLO_IN_LCL_WIDTH)		
		
		} // for(int i_c = 0; i_c < MLO_N_IN_TILES_PERSTACK; ++i_c, in_stg_off1 += MLO_IN_LCL_PERSTACK_SZ)

}

__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2)))
__kernel void aDNNConvUni(
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
	_FLOAT pvt_accum[MLO_PVT_ACCUM_DATA_SZ];
	_FLOAT pvt_in_stage[MLO_PVT_IN_HEIGHT * MLO_PVT_IN_WIDTH];
	_FLOAT pvt_wei_stage[MLO_FILTER_SIZE0];


	int grp_id0 = get_group_id(0);
	int y_tile_blk = grp_id0 / MLO_N_OUT_TILE_BLOCKS0;
	int x_tile_blk = -mad24(y_tile_blk, (int)MLO_N_OUT_TILE_BLOCKS0, -grp_id0);
	int o_pack = get_group_id(1); // block of outputs
	int b_pack = get_group_id(2); // batch block

	int lcl_id = get_local_id(0);
	int stack = lcl_id/MLO_ALUTILES_STACK_SZ;  // stack
	int alu_stack_id = -mad24(stack, (int)MLO_ALUTILES_STACK_SZ, -lcl_id);  // alu index in stack
// ALU plane inside stack
	int alu_out_plane_id = alu_stack_id / MLO_ALU_TILE_SZ;  // alu output plane index
	int alu_out_id = -mad24(alu_out_plane_id, (int)MLO_ALU_TILE_SZ, -alu_stack_id); // alu index inside an ALU output plane
// pos inside ALU tile
	int alu_tl1 = alu_out_id/MLO_ALU_VTILE0;
	int alu_tl0 = -mad24(alu_tl1, (int)MLO_ALU_VTILE0, -alu_out_id);

	int o_map_plane = o_pack * MLO_N_OUT_TILES_PERSTACK; // first output maps index per full ALU plane stack
	int o_map_base = alu_out_plane_id*MLO_N_OUT_TILES;  // local output map offset
	int o_map = o_map_plane + o_map_base; // output map index per ALU plane
	int b_index = b_pack * MLO_N_STACKS;

	int wave_id = lcl_id / MLO_N_READ_PROCS;
	int wave_lcl_id = -mad24(wave_id, (int)MLO_N_READ_PROCS, -lcl_id);

	int x_grp = x_tile_blk * MLO_IN_TILE0;
	int y_grp = y_tile_blk * MLO_IN_TILE1;

// TO DO: scale
	int x_in_grp = x_grp - MLO_FILTER_PAD0;
	int y_in_grp = y_grp - MLO_FILTER_PAD1;

	int x_in_lcl = alu_tl0 * MLO_OUT_TILE0;
	int y_in_lcl = alu_tl1 * MLO_OUT_TILE1;

// base offset to read data from local input data
	int in_stg_off = stack*MLO_IN_LCL_PERSTACK_SZ + (y_in_lcl) * MLO_IN_LCL_WIDTH + x_in_lcl;

    int in_off = b_index * MLO_IN_BATCH_STRIDE;

#if 0
	if (lcl_id == 0 )
	{
			printf("K:srt: %d %d %d\n",
				MLO_IN_LCL_SZ,
				MLO_WEIGHTS_SZ,
				(MLO_WEIGHTS_SZ + MLO_IN_LCL_SZ) * 4
			);
	}
#endif
#if 0
	if (lcl_id == 0 )
	{
			printf("K:srt: %d %d %d %d %d %d %d %d %d %d %d %d\n",
				grp_id0,
				alu_out_plane_id,
				alu_out_id,
				MLO_N_OUT_TILE_BLOCKS0,
				y_grp,
				x_grp,

// TO DO: scale
				y_out_grp,
				x_out_grp,

				y_out_lcl,
				x_out_lcl,

				MLO_OUT_HEIGHT,
				MLO_OUT_WIDTH

			);
	}
#endif

	
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
		//(MLO_N_STACKS * MLO_N_OUT_TILES_PERSTACK)
			int i_b = i / MLO_N_IN_TILES_PERSTACK;
			int i_c = -mad24(i_b, (int)MLO_N_IN_TILES_PERSTACK, -i);

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
			int lcl_y = MLO_FILTER_PAD1;
			int lcl_x = MLO_FILTER_PAD0;
			int gbl_base = 0;

#if 0
		if  (elem_id==0)
		{
			printf("K:in: %d %d %d %d %d %d %d\n",
				in_off,
			    i_b,
				MLO_IN_BATCH_STRIDE,
				i_c,
				MLO_IN_CHANNEL_STRIDE,
				in_off2,
				MLO_N_IN_TILES_TOTAL
				);
		}
#endif
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
			int lcl_o = (int)floor((float)i/(float)(MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ));
			int gbl_i = -mad24(lcl_o, (int)(MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ), -i);
			lcl_wei[i] = weights[wei_off + lcl_o * MLO_N_INPUTS * MLO_FILTER_SZ + gbl_i];
#else
// outputs are botoms(inputs))
// inputs are tops(outputs)
			int lcl_o = (int)floor((float)i/(float)(MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ));
			int gbl_i = -mad24(lcl_o, (int)(MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ), -i);
			int lcl_c = (int)(floor)((float)gbl_i / (float)MLO_FILTER_SZ);
			int lcl_i = -mad24(lcl_c, (int)MLO_FILTER_SZ, -gbl_i);

			int lcl_we_off = mad24(mad24(lcl_c, (int)MLO_N_IN_TILES_PERSTACK, lcl_o), (int)MLO_FILTER_SZ, lcl_i);
			int gbl_we_off = mad24(mad24(lcl_o, (int)MLO_N_OUTPUTS, lcl_c), (int)MLO_FILTER_SZ, wei_off + lcl_i);
			lcl_wei[lcl_we_off /*(lcl_c * MLO_N_IN_TILES_PERSTACK + lcl_o) * MLO_FILTER_SZ + lcl_i*/] 
				= weights[gbl_we_off /*wei_off + (lcl_o * MLO_N_OUTPUTS  + lcl_c)* MLO_FILTER_SZ + lcl_i*/];

#if 0
//			if ( i == 0 )
			{
				printf("K:w: %d %d %d %d %d %d %d %d %f\n",
						MLO_WEIGHTS_SZ,
						i,
						lcl_o,
						gbl_i,
						lcl_c,
						lcl_i,
						(lcl_c * MLO_N_IN_TILES_PERSTACK + lcl_o) * MLO_FILTER_SZ + lcl_i,
						wei_off + (lcl_o * MLO_N_OUTPUTS  + lcl_c)* MLO_FILTER_SZ + lcl_i,
						lcl_wei[(lcl_c * MLO_N_IN_TILES_PERSTACK + lcl_o) * MLO_FILTER_SZ + lcl_i]
				);
			}

#endif
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
	int x_out_grp = x_grp;
	int y_out_grp = y_grp;
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
#if MLO_OUTPUTS_ALIGNED == 0
						&& o_map + o < MLO_N_OUTPUTS
#endif
						; ++o, out_off1 += MLO_OUT_CHANNEL_STRIDE
						)
		{
// over output tile
			_FLOAT  bias_val = 0;
#if MLO_CONV_BIAS
			bias_val = bias[o_map + o];
#endif
			int out_off2 = out_off1;
			for( int j = 0; j < MLO_OUT_TILE1; ++j, out_off2 += MLO_OUT_STRIDE)
			{
				for(int i = 0; i < MLO_OUT_TILE0; ++i)
				{
#if MLO_OUT_ALIGNED == 0
					if ( y_out_grp + y_out_lcl + j < MLO_OUT_HEIGHT &&  x_out_grp + x_out_lcl + i < MLO_OUT_WIDTH)
#endif
					{
						out[out_off2 + i] = pvt_accum[o*MLO_OUT_TILE_SZ + j * MLO_OUT_TILE0 + i] + bias_val;
#if 0
						if ( out_off2 + i == 12 /*y_out_grp + y_out_lcl + j == 2 && x_out_grp + x_out_lcl + i == 0*/)
						{
							printf("K:out: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d    %f %f %f\n",
								grp_id0,
								lcl_id,
								alu_out_plane_id,
								alu_out_id,
								b_index + stack,
								o,
								o_map,
								out_off,
								out_off1,
								out_off2,
								y_out_grp,
								y_out_lcl,
								x_out_grp,
								x_out_lcl,
								j,
								i,
								pvt_accum[o*MLO_OUT_TILE_SZ + j * MLO_OUT_TILE0 + i],
								 bias_val,
								out[out_off2 + i]
								);
						}
#endif
					}

				}
			}

		}
	}


}
