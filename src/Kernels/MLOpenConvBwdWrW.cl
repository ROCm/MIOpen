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


#define MLO_CONVBWD_GROUP_SZ2 1
#define MLO_CONVBWD_N_SCAN_PERGROUP (MLO_CONVBWD_N_SCANPERIN * MLO_CONVBWD_N_ACCUM_SCAN)



inline void calculateXYPos(int linPos, int width, int *x, int *y)
{
	(*y) = linPos /width;
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
	calculateXYPos(linPos, lcl_stride, &x, &y);
	int g_x = x + gbl_x;
	int g_y = y + gbl_y;
	int gbl_off0 = calculateOffset(gbl_stride, g_x, g_y);
	int gbl_off = gbl_off0 + gbl_base;

	int lcl_off = lcl_base + linPos;

	_FLOAT gbl_val = gbl_data[gbl_off];

	vis &= (g_x >= 0 && g_x < gbl_width && g_y >= 0 && g_y < gbl_height);
     gbl_val =  (vis) ? gbl_val : 0;

	lcl_data[lcl_off] = gbl_val;

#if 0
	if ( debug && (linPos >= 0 && linPos < 40) )
	{
		printf("K:%d %d %f\n", y, x, gbl_val);
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

inline void dWeightsHoriz(_FLOAT * horiz_weights_accum, _FLOAT * bot, _FLOAT d_top, bool debug)
{
	for(int i = 0; i < MLO_CONV_KERNEL_SZ0; ++i)
	{
	   horiz_weights_accum[i] = bot[i] * d_top;
#if 0
	if (debug)
	{
		printf("K:b-t:%f %f\n",
		bot[i],
		d_top
		);
	}
#endif
	}
}

inline void dWeightsAccum(_FLOAT * weights_accum, _FLOAT * bias_accum, _FLOAT * bot_stage, __local _FLOAT *lcl_bot, __local _FLOAT*lcl_top_df, int top_lcl_sz,
							bool debug)
{
	_FLOAT d_top[MLO_CONVBWD_N_OUTSPERIN * MLO_CONVBWD_N_ACCUM_SCAN];
// the first 2 pixels from bottom has been loaded at setup

// get next data
	for(int j = 0; j < MLO_CONV_KERNEL_SZ1 + MLO_CONVBWD_N_ACCUM_SCAN - 1; ++j)
	{
// shift previous to the right
		for(int i = 0; i < MLO_CONV_KERNEL_SZ0 - 1; ++i)
		{
		   bot_stage[j*MLO_CONV_KERNEL_SZ0 + i] = bot_stage[j*MLO_CONV_KERNEL_SZ0 + i + 1];
		}

// next bot pixel
	    _FLOAT val = lcl_bot[j * MLO_CONVBWD_BOT_DATA_WIDTH];		
		bot_stage[j*MLO_CONV_KERNEL_SZ0 + MLO_CONV_KERNEL_SZ0 - 1] = val;
	}

// next top pixels
	for(int j = 0; j < MLO_CONVBWD_N_OUTSPERIN; ++j)
	{
		for(int i = 0; i < MLO_CONVBWD_N_ACCUM_SCAN; ++i)
		{
		    _FLOAT top_df_val = lcl_top_df[j*top_lcl_sz + i * MLO_CONVBWD_TOP_DATA_WIDTH];
			d_top[j*MLO_CONVBWD_N_ACCUM_SCAN + i] = top_df_val;

// bias df calculation

			bias_accum[j] += top_df_val;
#if 0
		if ( debug && j == 0)
		{
			printf("K:%d %f %f\n",
				j,
				bias_accum[j],
				top_df_val
		   );
		}
#endif

		}
	}
// accum
	_FLOAT horiz_accum[MLO_CONV_KERNEL_SZ0];
// over tops

	for(int l = 0; l < MLO_CONVBWD_N_OUTSPERIN; ++l)
	{
	// over all accumulating scans per wk-item
		for(int i = 0; i < MLO_CONVBWD_N_ACCUM_SCAN; i++)
		{
// over all kernel scans
			for (int j = 0; j < MLO_CONV_KERNEL_SZ1; ++j)
			{
    // grad wrt W
				dWeightsHoriz(horiz_accum, &bot_stage[(j + i) * MLO_CONV_KERNEL_SZ0], d_top[l*MLO_CONVBWD_N_ACCUM_SCAN + i], (debug && (l == 0)));
	// acumulate into proper scan elements
				for(int k = 0; k < MLO_CONV_KERNEL_SZ0; ++k)
				{
					weights_accum[(l*MLO_CONV_KERNEL_SZ1 + j)*MLO_CONV_KERNEL_SZ0 + k] += horiz_accum[k];

				}
// reducing reg pressure
//				mem_fence(CLK_LOCAL_MEM_FENCE);

			}
		}
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
				 (debug && (c==0)));
	}

}


inline void ReduceKernel(__local _FLOAT * lcl_blob, _FLOAT *weights_accum, int lcl_id, int scan_lcl, int sum_stride, int unit_len, bool debug)
{
	for(int j = (sum_stride>>1); j > 0; j >>= 1)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if ( scan_lcl < j )
		{
			for(int i = 0; i < unit_len; ++i)
			{

				weights_accum[i] += lcl_blob[(lcl_id + j) * unit_len + i];

				lcl_blob[lcl_id * unit_len + i] = weights_accum[i];
			}

		}
	}
}

__attribute__((reqd_work_group_size(MLO_CONVBWD_GROUP_SZ0,MLO_CONVBWD_GROUP_SZ1,MLO_CONVBWD_GROUP_SZ2)))
__kernel void MLOpenConvBwdWrW(
       const __global _FLOAT * top_df,
       const __global _FLOAT * bot,
// need another pass to sum over the batch and inputs scan blocks
       __global _FLOAT * weights_df,
       __global _FLOAT * bias_df,
	   _FLOAT padding_val
	   )
{
// bot data
	__local _FLOAT lcl_blob[MLO_CONVBWD_LCL_MEMSZ];
	__local _FLOAT *lcl_bot_data = lcl_blob;
	__local _FLOAT *lcl_top_df_data = lcl_bot_data + MLO_CONVBWD_N_INS*MLO_CONVBWD_BOT_DATA_WIDTH * MLO_CONVBWD_BOT_DATA_HEIGHT;

	_FLOAT weights_accum[MLO_CONVBWD_N_OUTSPERIN * MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1];
	_FLOAT bias_accum[MLO_CONVBWD_N_OUTSPERIN];
	_FLOAT bot_stage[(MLO_CONV_KERNEL_SZ1 + MLO_CONVBWD_N_ACCUM_SCAN - 1) * MLO_CONV_KERNEL_SZ0];

	int lcl_id0 = get_local_id(0);
	int lcl_id = get_local_id(1) * MLO_CONVBWD_GROUP_SZ0 + get_local_id(0);
	int scan_grp = get_group_id(0);
	int x_grp = 0;
	int y_grp = scan_grp * MLO_CONVBWD_N_SCAN_PERGROUP;  // y position inside the top image
	int oo = (get_global_id(1)/(MLO_CONVBWD_N_OUTS/MLO_CONVBWD_N_OUTSPERIN)) * MLO_CONVBWD_N_OUTS; // output block
	int o_lcl = get_local_id(1) * MLO_CONVBWD_N_OUTSPERIN; // output per input
	int o = oo + o_lcl;
	int cc = get_group_id(2) /  MLO_CONV_BATCH_SZ;              // input block index
	int b = get_group_id(2) - cc * MLO_CONV_BATCH_SZ;
	cc *= MLO_CONVBWD_N_INS; // group input
	int c_lcl = lcl_id0 / MLO_CONVBWD_N_SCANPERIN;
	int scan_lcl =  lcl_id0 - c_lcl * MLO_CONVBWD_N_SCANPERIN;
	int c = cc + c_lcl;

// input channel offset
	int gbl_bot_off = b * MLO_CONV_BOT_BATCH_STRIDE + cc * MLO_CONV_BOT_CHANNEL_STRIDE;
// output msp offset
	int gbl_top_off = b * MLO_CONVBWD_TOPDF_BATCH_STRIDE + oo * MLO_CONVBWD_TOPDF_CHANNEL_STRIDE;
	int lcl_bot_y = 0;
	int lcl_bot_x = 0;

	int lcl_bot_data_off = 0;
	int lcl_top_df_data_off = 0;

// outer loop over batch
// accumulate all df wrt W and wrt Bias
// for the same inputs/outputs paires

	int lcl_top_y = scan_lcl * MLO_CONVBWD_N_ACCUM_SCAN;
	int lcl_top_x = 0;
	int lcl_top_off = 0;
	int lcl_top_off2 = (o_lcl * MLO_CONVBWD_TOP_DATA_HEIGHT + lcl_top_y) * MLO_CONVBWD_TOP_DATA_WIDTH + lcl_top_x;
	int lcl_bot_off = 0;
	int lcl_bot_off2 = (c_lcl * MLO_CONVBWD_BOT_DATA_HEIGHT + lcl_top_y) *  MLO_CONVBWD_BOT_DATA_WIDTH + lcl_top_x;

	int lcl_top_size = MLO_CONVBWD_TOP_DATA_WIDTH * MLO_CONVBWD_TOP_DATA_HEIGHT;
	int gbl_top_size = MLO_CONV_TOP_WIDTH * MLO_CONV_TOP_HEIGHT;
	int y_grp_top = y_grp;
	int x_grp_top = x_grp;
	int y_grp_top_lcl = 0;
	int x_grp_top_lcl = 0;
	int x_top = 0;
	int y_top = y_grp_top + lcl_top_y;

	int lcL_bot_size = MLO_CONVBWD_BOT_DATA_WIDTH * MLO_CONVBWD_BOT_DATA_HEIGHT;
	int gbl_bot_size = MLO_CONV_BOT_WIDTH * MLO_CONV_BOT_HEIGHT;
    int y_grp_bot_lcl = 0;
	int x_grp_bot_lcl = 0;

	for(int i = lcl_id; i < MLO_CONVBWD_N_INS*MLO_CONVBWD_BOT_DATA_WIDTH * MLO_CONVBWD_BOT_DATA_HEIGHT; i += MLO_CONVBWD_GROUP_SZ)
	{
		lcl_bot_data[i] = 0;
	}

	for(int i = lcl_id; i < MLO_CONVBWD_N_OUTS * MLO_CONVBWD_TOP_DATA_WIDTH * MLO_CONVBWD_TOP_DATA_HEIGHT; i += MLO_CONVBWD_GROUP_SZ)
	{
		lcl_top_df_data[i] = 0;
	}

	for(int i = 0; i < MLO_CONVBWD_N_OUTSPERIN * MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1; ++i)
	{
		weights_accum[i] = 0;
	}

	for(int i = 0; i < MLO_CONVBWD_N_OUTSPERIN; ++i)
	{
		bias_accum[i] = 0;
	}

//	for (int b = 0; b < 1 /*MLO_CONV_BATCH_SZ*/; ++b, gbl_bot_off += MLO_CONV_BOT_BATCH_STRIDE, gbl_top_off += MLO_CONVBWD_TOPDF_BATCH_STRIDE)
	{
//		barrier(CLK_LOCAL_MEM_FENCE);

// move along scanlines

// accum coord

// moving along the full scan of input/output imgs
		for(int sl = 0; sl < MLO_CONVBWD_N_SCANLOOPS; ++sl, x_grp_top += MLO_CONVBWD_SCAN_STEP)
		{

			barrier(CLK_LOCAL_MEM_FENCE);

// top
			loadData(lcl_id, MLO_CONVBWD_GROUP_SZ,
					lcl_top_df_data, lcl_top_off, lcl_top_size, MLO_CONVBWD_TOP_DATA_HEIGHT, MLO_CONVBWD_TOP_DATA_WIDTH, MLO_CONVBWD_TOP_DATA_WIDTH, y_grp_top_lcl, x_grp_top_lcl,
					top_df, gbl_top_off, gbl_top_size, MLO_CONV_TOP_HEIGHT, MLO_CONV_TOP_WIDTH, MLO_CONVBWD_TOPDF_STRIDE, y_grp_top, x_grp_top,
					oo, MLO_CONV_N_OUTPUTS, MLO_CONVBWD_N_OUTS,
					false);

// bot
			loadData(lcl_id, MLO_CONVBWD_GROUP_SZ,
					 lcl_bot_data, lcl_bot_off, lcL_bot_size, MLO_CONVBWD_BOT_DATA_HEIGHT, MLO_CONVBWD_BOT_DATA_WIDTH, MLO_CONVBWD_BOT_DATA_WIDTH, y_grp_bot_lcl, x_grp_bot_lcl,
					 bot, gbl_bot_off, gbl_bot_size, MLO_CONV_BOT_HEIGHT, MLO_CONV_BOT_WIDTH, MLO_CONV_BOT_STRIDE,  y_grp_top - MLO_CONV_KERNEL_PAD1, x_grp_top - MLO_CONV_KERNEL_PAD0,
					 cc, MLO_CONV_N_INPUTS, MLO_CONVBWD_N_INS,
					 false //sl==1
					 );



			barrier(CLK_LOCAL_MEM_FENCE);

// fill the first kernel_sz0 - 1 horiz (scan) slots
			for(int j = 0; j < MLO_CONV_KERNEL_SZ1 + MLO_CONVBWD_N_ACCUM_SCAN - 1; ++j)
			{
				bot_stage[j*MLO_CONV_KERNEL_SZ0] = 0;

				for(int i = 0; i < MLO_CONV_KERNEL_SZ0 - 1; ++i)
				{
// padding substructed above
					_FLOAT val = lcl_bot_data[lcl_bot_off2 + j * MLO_CONVBWD_BOT_DATA_WIDTH + i];
					bot_stage[j*MLO_CONV_KERNEL_SZ0 + i + 1] = val;
				}
			}

			int lcl_bot_off3 = lcl_bot_off2 + MLO_CONV_KERNEL_SZ0 - 1;
// inner loop over scan step		
			for(int sp = 0; sp < MLO_CONVBWD_SCAN_STEP; ++sp)
			{			
				bool visTopX = (x_top < MLO_CONV_TOP_WIDTH);
				dWeightsAccum(weights_accum, bias_accum, bot_stage,
					 &lcl_bot_data[lcl_bot_off3 + sp],
					 &lcl_top_df_data[lcl_top_off2 + sp], MLO_CONVBWD_TOP_DATA_WIDTH * MLO_CONVBWD_TOP_DATA_HEIGHT,
					 (lcl_id <4)
					 );

			}




		} // sl

	} // b


#if 0
	if (lcl_id == 0)
	{
		for(int k = 0; k < MLO_CONVBWD_N_OUTSPERIN; ++k)
		{
			for(int i = 0; i < MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1; ++i)
			{
				if ( k == 0)
				{
					printf("K:%d %12.10f\n",
						 lcl_id,
						 weights_accum[k * MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1 + i]
						);
				}
			}
		}
	}
#endif

// reduce over all outputs over scans
	for(int k = 0; k < MLO_CONVBWD_N_OUTSPERIN && o + k < MLO_CONV_N_OUTPUTS; ++k)
	{
		barrier(CLK_LOCAL_MEM_FENCE);

		int n_input_scan_blocks = get_num_groups(0);
		int input_scan_block_idx = get_group_id(0);            

// bias
// do it once per input
// use only fisrt MLO_CONVBWD_N_SCANPERIN wk-items for bias reduce
		if ( cc == 0 && lcl_id >= get_local_id(1) * MLO_CONVBWD_GROUP_SZ0 && lcl_id < get_local_id(1) * MLO_CONVBWD_GROUP_SZ0 + MLO_CONVBWD_N_SCANPERIN)
		{
			lcl_blob[lcl_id] = 0;

			lcl_blob[get_local_id(1) * MLO_CONVBWD_N_SCANPERIN + scan_lcl] = bias_accum[k];

		
			ReduceKernel(lcl_blob, &bias_accum[k], (get_local_id(1) * MLO_CONVBWD_N_SCANPERIN) + scan_lcl, scan_lcl, MLO_CONVBWD_N_SCANPERIN, 1, false);
					
		}


		barrier(CLK_LOCAL_MEM_FENCE);

// do it once per input
		if ( cc == 0 && lcl_id == get_local_id(1) * MLO_CONVBWD_GROUP_SZ0)
		{
			int bias_off = ((o + k)  * MLO_CONV_BATCH_SZ + b )* n_input_scan_blocks + input_scan_block_idx;
			bias_df[bias_off] = bias_accum[k];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

// weights

		for(int i = 0; i < MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1; ++i)
		{
			lcl_blob[lcl_id*MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1 + i] = weights_accum[k * MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1 + i];
		}
	
		ReduceKernel(lcl_blob, &weights_accum[k * MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1], lcl_id, scan_lcl, MLO_CONVBWD_N_SCANPERIN, (MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1), false);

		barrier(CLK_LOCAL_MEM_FENCE);


		if ( scan_lcl == 0 && c < MLO_CONV_N_INPUTS)
		{

// replaced with log sum

#if 0
			for(int j = 0; j < MLO_CONVBWD_N_SCANPERIN - 1; ++j)
			{
				for(int i = 0; i < MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1; ++i)
				{
					weights_accum[k * MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1 + i] += lcl_blob[(lcl_id + j + 1) * MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1 + i];
				}
			}
#endif

			int weight_off = ((((o + k) * MLO_CONV_N_INPUTS + c) * MLO_CONV_BATCH_SZ  + b )* n_input_scan_blocks + input_scan_block_idx) *MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1;
			for(int i = 0; i < MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1; ++i)
			{
				weights_df[weight_off + i] = weights_accum[k * MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1 + i];
			}

		}



	}



}




#define MLO_CONVBSUM_GRP_SZ1 1
#define MLO_CONVBSUM_GRP_SZ2 1


__attribute__((reqd_work_group_size(MLO_CONVBSUM_GRP_SZ0,MLO_CONVBSUM_GRP_SZ1,MLO_CONVBSUM_GRP_SZ2)))
__kernel void MLOpenConvBwdWrW_rdc(
       const __global _FLOAT * weights_df_t,
       const __global _FLOAT * bias_df_t,
       __global _FLOAT * weights_df,
       __global _FLOAT * bias_df
	   )
{

	__local _FLOAT lcl_we_accum[MLO_CONVBSUM_GRP_SZ0 * MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1];
	int lcl_id = get_local_id(0);  // 
	int o = get_global_id(1); // o
	int c = get_global_id(2);   // c

	_FLOAT weights_accum[MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1];

	for(int i = 0; i < MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1; ++i)
	{
		weights_accum[i] = 0;
	}
	int in_off = (o * MLO_CONV_N_INPUTS + c) * MLO_CONV_BATCH_SZ * MLO_CONVBWD_N_GRPS_PERHEIGHT * MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1;
	for(int j = lcl_id; j < MLO_CONV_BATCH_SZ * MLO_CONVBWD_N_GRPS_PERHEIGHT; j += get_local_size(0))
	{

		for(int i = 0; i < MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1; ++i)
		{
		
		    weights_accum[i] += weights_df_t[in_off + j * MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1 + i];


		}
	}



	for(int i = 0; i < MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1; ++i)
	{
		
		lcl_we_accum[lcl_id * MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1 + i] = weights_accum[i];

	}
	
// barrier inside
	ReduceKernel(lcl_we_accum, weights_accum, lcl_id, lcl_id, MLO_CONVBSUM_GRP_SZ0, MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1, false);


	barrier(CLK_LOCAL_MEM_FENCE);

	if ( lcl_id == 0)
	{
// TO DO: replace with log sum
#if 0
		for(int j = lcl_id; j < MLO_CONVBSUM_GRP_SZ0 - 1; ++j)
		{

			for(int i = 0; i < MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1; ++i)
			{

				weights_accum[i] += lcl_we_accum[(j + 1) * MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1 + i];


			}
		}
#endif

		int out_off = (o * MLO_CONV_N_INPUTS + c) * MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1;

		for(int i = 0; i < MLO_CONV_KERNEL_SZ0 * MLO_CONV_KERNEL_SZ1; ++i)
		{
		
		    weights_df[out_off + i] = weights_accum[i];

		}

	}

// bias

	_FLOAT bias_accum = 0;
	int bias_in_off = o * MLO_CONV_BATCH_SZ * MLO_CONVBWD_N_GRPS_PERHEIGHT;

// all 256 wk- items; barrier is fine
	if (c == 0)
	{
		for(int j = lcl_id; j < MLO_CONV_BATCH_SZ * MLO_CONVBWD_N_GRPS_PERHEIGHT; j += get_local_size(0))
		{

			bias_accum += bias_df_t[bias_in_off + j];
		}


		lcl_we_accum[lcl_id] = bias_accum;
#if 0
		if ( lcl_id < 16)
		{
			printf("K:%d %f\n",
				lcl_id,
				lcl_we_accum[lcl_id]
		   );
		}
#endif
// barrier inside
		ReduceKernel(lcl_we_accum, &bias_accum, lcl_id, lcl_id, MLO_CONVBSUM_GRP_SZ0, 1, false);


		if ( lcl_id == 0)
		{
			bias_df[o] = bias_accum;
		}
	}

}


