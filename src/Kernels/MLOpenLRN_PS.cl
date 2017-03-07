/*
 * Copyright (c) 2015 AMD Inc.
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


#define MLO_LRN_GROUP_SZ2 1
#define MLO_LRN_STRIDE 1

#define MLO_LRN_LEFT_PAD0 (((MLO_LRN_PAD0 + MLO_READ_UNIT - 1) / MLO_READ_UNIT) * MLO_READ_UNIT)
#define MLO_LRN_RIGHT_SIDE (((MLO_LRN_GROUP_SZ0 *MLO_LRN_N_HORIZ_OUT_PIX + MLO_LRN_PAD0 + MLO_READ_UNIT - 1)/MLO_READ_UNIT)*MLO_READ_UNIT)
#define MLO_LRN_LCL_DATA_WIDTH (MLO_LRN_LEFT_PAD0 + MLO_LRN_RIGHT_SIDE)
#define MLO_LCL_READ4 (MLO_LRN_LCL_DATA_WIDTH/MLO_READ_UNIT)
#define MLO_LRN_LCL_DATA_HEIGHT  (MLO_LRN_GROUP_SZ1 * MLO_LRN_N_VERT_OUT_PIX + MLO_LRN_KERNEL_SZ - 1)
#define MLO_LRN_GROUP_SZ (MLO_LRN_GROUP_SZ2 * MLO_LRN_GROUP_SZ1 * MLO_LRN_GROUP_SZ0)
//#define MLO_LRN_PREPAD_SZ (MLO_LRN_KERNEL_SZ - 1)/2

static inline int iDiv(int v, int d)
{
	int r = (int)((float)v / d + 0.00001f);
	return(r);
}

static inline int iMod(int v, int u, int d)
{
	int r = v - mul24((int)u, (int)d);
	return(r);
}
__attribute__((reqd_work_group_size(MLO_LRN_GROUP_SZ0,MLO_LRN_GROUP_SZ1,MLO_LRN_GROUP_SZ2)))
__kernel void MLOpenLRNWithinChannel_PS(
       const __global _FLOAT * bot,
       __global _FLOAT * top,
#if MLO_LRN_DO_SCALE
	   __global _FLOAT * scale,
#endif
	   _FLOAT alphaoverarea,
	   _FLOAT alpha,
	   _FLOAT beta,
	   _FLOAT K
	   )
{

// IT's taken from POOLING AVE with stride = 1'
		__local _FLOAT bot_data[MLO_LRN_LCL_DATA_WIDTH * MLO_LRN_LCL_DATA_HEIGHT];
		int x = get_group_id(0) * MLO_LRN_GROUP_SZ0 * MLO_LRN_N_HORIZ_OUT_PIX;
		int y = get_group_id(1) * MLO_LRN_GROUP_SZ1 * MLO_LRN_N_VERT_OUT_PIX;
		int lcl_id0 = get_local_id(0);
		int lcl_id1 = get_local_id(1);
		int lcl_id = (lcl_id1 << MLO_LRN_GROUP_LG2SZ0) + lcl_id0;
		int ob = get_global_id(2); // output * batch_sz
		int o = iDiv(ob,MLO_LRN_BATCH_SZ);
		int b = iMod(ob, o, MLO_LRN_BATCH_SZ);
		int bot_x = x;
		int bot_y = y;
		int bot_off = mul24(b, (int)MLO_LRN_BOT_BATCH_STRIDE) + mul24(o,(int)MLO_LRN_BOT_CHANNEL_STRIDE);

// load tile
		for( int b_j = lcl_id1; b_j < MLO_LRN_LCL_DATA_HEIGHT; b_j += MLO_LRN_GROUP_SZ1)
		{	
			int bot_y_act = bot_y + b_j - MLO_LRN_PAD1;

			bool invisibleY = (bot_y_act < 0) || (bot_y_act >= MLO_LRN_BOT_HEIGHT);

			int bot_y_off = mul24(bot_y_act, (int) MLO_LRN_BOT_STRIDE);

			int lcl_off_v = mul24(b_j, (int)MLO_LRN_LCL_DATA_WIDTH);

			for (int b_i = lcl_id0; b_i < MLO_LCL_READ4; b_i += MLO_LRN_GROUP_SZ0)
			{

				int bot_x_act = bot_x + (b_i * MLO_READ_UNIT) - MLO_LRN_LEFT_PAD0;



				_FLOAT bot_val4[MLO_READ_UNIT];
				

				bool invisibleX;
				for (int i = 0; i < MLO_READ_UNIT; ++i)
				{

					int bot_off_x = bot_off + bot_y_off + bot_x_act + i;

					invisibleX = (bot_x_act + i < 0) || (bot_x_act + i >= MLO_LRN_BOT_WIDTH);

					bot_off_x = (invisibleX || invisibleY) ? 0 : bot_off_x;

					_FLOAT bot_val = bot[bot_off_x];
					// since we need a sum of squares for the normalization
					// we do the square at the input
					bot_val *= bot_val;

					bot_val = (invisibleX || invisibleY) ? 0 : bot_val;

					bot_data[lcl_off_v + (b_i * MLO_READ_UNIT) + i] = bot_val;
				}
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		_FLOAT partial_sum_x[MLO_LRN_N_HORIZ_OUT_PIX - 1];  // horizontal partial sum
		_FLOAT partial_sum_xy[MLO_LRN_N_VERT_OUT_PIX - 1][MLO_LRN_N_HORIZ_OUT_PIX]; // horizontal-vertical partial sums.
		_FLOAT accum[MLO_LRN_N_VERT_OUT_PIX][MLO_LRN_N_HORIZ_OUT_PIX]; // accumulator

		int top_y = mad24(lcl_id1, (int)MLO_LRN_N_VERT_OUT_PIX, y);
		int top_x = mad24(lcl_id0, (int)MLO_LRN_N_HORIZ_OUT_PIX, x);

		int lcl_y = mul24(lcl_id1, (int)MLO_LRN_N_VERT_OUT_PIX);
		int lcl_x = mad24(lcl_id0, (int)(MLO_LRN_N_HORIZ_OUT_PIX), (int)(MLO_LRN_LEFT_PAD0 - MLO_LRN_PAD0));
		int lcl_off = mad24(lcl_y, MLO_LRN_LCL_DATA_WIDTH, lcl_x);

		for (int j = 0; j < MLO_LRN_N_VERT_OUT_PIX; ++j)
		{
			for (int i = 0; i < MLO_LRN_N_HORIZ_OUT_PIX; ++i)
			{
				accum[j][i] = 0;
			}
		}
		for (int j = 0; j < MLO_LRN_N_VERT_OUT_PIX - 1; ++j)
		{
			for (int i = 0; i < MLO_LRN_N_HORIZ_OUT_PIX; ++i)
			{
				partial_sum_xy[j][i] = 0;
			}
		}


// running window  summation
		_FLOAT mov_accum;
		int jj = 0;
		int ii = 0;

// first to get vertica partial sums 
		for (; jj < (int)(MLO_LRN_N_VERT_OUT_PIX-1); ++jj)
		{
			for (ii = 0; ii < (int)(MLO_LRN_N_HORIZ_OUT_PIX - 1); ++ii)
			{

				_FLOAT bot_val = bot_data[lcl_off + jj*MLO_LRN_LCL_DATA_WIDTH + ii];
				_FLOAT accum_tmp = bot_val;
// save horizontal partial sums
				partial_sum_x[ii] = accum_tmp;
// accumulate in vert-horizontal(0)
				partial_sum_xy[jj][0] += accum_tmp;

			}

			for (; ii < (int)MLO_LRN_KERNEL_SZ0; ++ii)
			{

				_FLOAT bot_val = bot_data[lcl_off + jj*MLO_LRN_LCL_DATA_WIDTH + ii];
				_FLOAT accum_tmp = bot_val;
				// accumulate in vert horizontal(0)
				partial_sum_xy[jj][0] += accum_tmp;
			}

// running horizontal window				

			for (; ii < (int)(MLO_LRN_KERNEL_SZ0 + MLO_LRN_N_HORIZ_OUT_PIX - 1); ++ii)
			{

				_FLOAT bot_val = bot_data[lcl_off + jj*MLO_LRN_LCL_DATA_WIDTH + ii];
				_FLOAT accum_tmp = bot_val;
// calculate all vertical-horizontal partial sums
				partial_sum_xy[jj][ii - MLO_LRN_KERNEL_SZ0 + 1] = partial_sum_xy[jj][ii - MLO_LRN_KERNEL_SZ0] + (accum_tmp - partial_sum_x[ii - MLO_LRN_KERNEL_SZ0]);

			}

// put into accumulator[0][i] 
// whatever has been accumulated so far
			for (int i = 0; i < MLO_LRN_N_HORIZ_OUT_PIX; ++i)
			{
				accum[0][i] += partial_sum_xy[jj][i];

			}

		}

// calculate row 0 accumulators
		for (; jj < (int)MLO_LRN_KERNEL_SZ1; ++jj)
		{
			mov_accum = 0;

			for (ii = 0; ii < (int)(MLO_LRN_N_HORIZ_OUT_PIX - 1); ++ii)
			{

				_FLOAT bot_val = bot_data[lcl_off + jj*MLO_LRN_LCL_DATA_WIDTH + ii];
				_FLOAT accum_tmp = bot_val;
				partial_sum_x[ii] = accum_tmp;
				mov_accum += accum_tmp;
			}

			for (; ii < (int)MLO_LRN_KERNEL_SZ0; ++ii)
			{

				_FLOAT bot_val = bot_data[lcl_off + jj*MLO_LRN_LCL_DATA_WIDTH + ii];
				_FLOAT accum_tmp = bot_val;
				mov_accum += accum_tmp;
			}

			accum[0][0] += mov_accum;
// running horizontal window				

			for (; ii < (int)(MLO_LRN_KERNEL_SZ0 + MLO_LRN_N_HORIZ_OUT_PIX - 1); ++ii)
			{

				_FLOAT bot_val = bot_data[lcl_off + jj*MLO_LRN_LCL_DATA_WIDTH + ii];
				_FLOAT accum_tmp = bot_val;
// running horizontal window				
				mov_accum += (accum_tmp - partial_sum_x[ii - MLO_LRN_KERNEL_SZ0]);
				accum[0][ii - MLO_LRN_KERNEL_SZ0 + 1] += mov_accum;

			}

		}


// accumulate all other rows besides 0
		for (; jj < (int)(MLO_LRN_KERNEL_SZ1 + MLO_LRN_N_VERT_OUT_PIX - 1); ++jj)
		{
// first running horizontal winodw as before
			mov_accum = 0;
			for (ii = 0; ii < (int)(MLO_LRN_N_HORIZ_OUT_PIX - 1); ++ii)
			{

				_FLOAT bot_val = bot_data[lcl_off + jj*MLO_LRN_LCL_DATA_WIDTH + ii];
				_FLOAT accum_tmp = bot_val;
				partial_sum_x[ii] = accum_tmp;
				accum[jj - MLO_LRN_KERNEL_SZ1 + 1][0] += accum_tmp;
			}
			for (; ii < (int)MLO_LRN_KERNEL_SZ0; ++ii)
			{

				_FLOAT bot_val = bot_data[lcl_off + jj*MLO_LRN_LCL_DATA_WIDTH + ii];
				_FLOAT accum_tmp = bot_val;
				accum[jj - MLO_LRN_KERNEL_SZ1 + 1][0] += accum_tmp;
			}
// running horizontal window				

			int ii1 = ii;
			for (; ii < (int)(MLO_LRN_KERNEL_SZ0 + MLO_LRN_N_HORIZ_OUT_PIX - 1); ++ii)
			{

				_FLOAT bot_val = bot_data[lcl_off + jj*MLO_LRN_LCL_DATA_WIDTH + ii];
				_FLOAT accum_tmp = bot_val;
				// 
				accum[jj - MLO_LRN_KERNEL_SZ1 + 1][ii - MLO_LRN_KERNEL_SZ0 + 1] = accum[jj - MLO_LRN_KERNEL_SZ1 + 1][ii - MLO_LRN_KERNEL_SZ0] + accum_tmp;
				accum[jj - MLO_LRN_KERNEL_SZ1 + 1][ii - MLO_LRN_KERNEL_SZ0 + 1] 
					-= partial_sum_x[ii - MLO_LRN_KERNEL_SZ0];

			}

// finally running vertical window				

			for (ii = ii1; ii < (int)(MLO_LRN_KERNEL_SZ0 + MLO_LRN_N_HORIZ_OUT_PIX - 1); ++ii)
			{

				// finish horizontal summation
				// add/substarct vertical patial sum
				accum[jj - MLO_LRN_KERNEL_SZ1 + 1][ii - MLO_LRN_KERNEL_SZ0 + 1] += accum[jj - MLO_LRN_KERNEL_SZ1][ii - MLO_LRN_KERNEL_SZ0 + 1];
				accum[jj - MLO_LRN_KERNEL_SZ1 + 1][ii - MLO_LRN_KERNEL_SZ0 + 1] -= partial_sum_xy[jj - MLO_LRN_KERNEL_SZ1][ii - MLO_LRN_KERNEL_SZ0 + 1];

			}
			accum[jj - MLO_LRN_KERNEL_SZ1 + 1][0] -= partial_sum_xy[jj - MLO_LRN_KERNEL_SZ1][0];
			accum[jj - MLO_LRN_KERNEL_SZ1 + 1][0] += accum[jj - MLO_LRN_KERNEL_SZ1][0];

		}

// normalization
		_FLOAT prv_scale[MLO_LRN_N_VERT_OUT_PIX][MLO_LRN_N_HORIZ_OUT_PIX];
		_FLOAT adj_alphaoverarea = alphaoverarea;
		for (int k = 0; k < MLO_LRN_N_VERT_OUT_PIX; k++)
		{

//			int hstart = y + lcl_id1 * MLO_LRN_N_VERT_OUT_PIX  + k - MLO_LRN_PAD1;
//			int hend = min(hstart + MLO_LRN_KERNEL_SZ, MLO_LRN_BOT_HEIGHT + MLO_LRN_PAD1);

			for(int l = 0; l < MLO_LRN_N_HORIZ_OUT_PIX; l++)
			{

//				int wstart = x + lcl_id0 * MLO_LRN_N_HORIZ_OUT_PIX + l - MLO_LRN_PAD0;
//				int wend = min(wstart + MLO_LRN_KERNEL_SZ, MLO_LRN_BOT_WIDTH + MLO_LRN_PAD0);
//				int adj_area_size = (hend - hstart) * (wend - wstart);
//				adj_alphaoverarea = alpha / adj_area_size;

				prv_scale[k][l]  = K + accum[k][l] * adj_alphaoverarea ;

			}
		}


		int top_off = b * MLO_LRN_TOP_BATCH_STRIDE + o * MLO_LRN_TOP_CHANNEL_STRIDE + top_y * MLO_LRN_TOP_STRIDE + top_x;
		int scale_off = b * MLO_LRN_SCALE_BATCH_STRIDE + o * MLO_LRN_SCALE_CHANNEL_STRIDE + top_y * MLO_LRN_SCALE_STRIDE + top_x;

// final output

		for (int k = 0; k < MLO_LRN_N_VERT_OUT_PIX
#if MLO_OUT_VERT_ALIGNED == 0
			&& (top_y + k < MLO_LRN_TOP_HEIGHT)
#endif
			; k++)
		{
			for (int l = 0; l < MLO_LRN_N_HORIZ_OUT_PIX
#if MLO_OUT_HORIZ_ALIGNED == 0
				&& (top_x + l < MLO_LRN_TOP_WIDTH)
#endif
				; l++)
			{
				_FLOAT s;
				_FLOAT bot_val;
				s = native_exp((_FLOAT)-beta * native_log(prv_scale[k][l]));
				//					s = pow(prv_scale[k][l], -beta);
				_FLOAT tmp = bot_data[lcl_off + mad24((k + MLO_LRN_PAD1), (int)MLO_LRN_LCL_DATA_WIDTH, (l + MLO_LRN_PAD0))];

// do a square root to get back to the raw input
				bot_val = native_sqrt(tmp);
#if MLO_LRN_DO_SCALE
				scale[scale_off + k * MLO_LRN_SCALE_STRIDE +l] = prv_scale[k][l];
#endif
				top[top_off + k * MLO_LRN_TOP_STRIDE + l] = bot_val * s;

			}
		}

}


__attribute__((reqd_work_group_size(MLO_LRN_GROUP_SZ0,MLO_LRN_GROUP_SZ1,MLO_LRN_GROUP_SZ2)))
__kernel void MLOpenLRNWithinChannelBwd(
       const __global _FLOAT * top,
	   const __global _FLOAT *	bot,
       const __global _FLOAT * top_df,
	   const __global _FLOAT *	scale,
       __global _FLOAT * bot_df,
	   _FLOAT ratio, //2. * alpha * beta / local_area
	   _FLOAT alpha,
	   _FLOAT beta
	   )
{
		__local _FLOAT top_df_data[MLO_LRN_LCL_DATA_WIDTH * MLO_LRN_LCL_DATA_HEIGHT];
		__local _FLOAT ratio_data[MLO_LRN_LCL_DATA_WIDTH * MLO_LRN_LCL_DATA_HEIGHT];
		int x = get_group_id(0) * MLO_LRN_GROUP_SZ0 * MLO_LRN_N_HORIZ_OUT_PIX;
		int y = get_group_id(1) * MLO_LRN_GROUP_SZ1 * MLO_LRN_N_VERT_OUT_PIX;
		int lcl_id0 = get_local_id(0);
		int lcl_id1 = get_local_id(1);
		int lcl_id = (lcl_id1 << MLO_LRN_GROUP_LG2SZ0) + lcl_id0;
		int ob = get_global_id(2); // output * batch_sz
		int o = ob / MLO_LRN_BATCH_SZ;
		int b = ob - o * MLO_LRN_BATCH_SZ;
		int top_x = x;
		int top_y = y;
		int top_df_off = b * MLO_LRN_TOPDF_BATCH_STRIDE + o * MLO_LRN_TOPDF_CHANNEL_STRIDE;
		int scale_off = b * MLO_LRN_SCALE_BATCH_STRIDE + o * MLO_LRN_SCALE_CHANNEL_STRIDE;
		int bot_x = x + lcl_id0 * MLO_LRN_N_HORIZ_OUT_PIX;
		int bot_y = y + lcl_id1 * MLO_LRN_N_VERT_OUT_PIX;

		_FLOAT prv_exp_scale[MLO_LRN_N_VERT_OUT_PIX][MLO_LRN_N_HORIZ_OUT_PIX];
//		_FLOAT prv_top_df[MLO_LRN_N_VERT_OUT_PIX][MLO_LRN_N_HORIZ_OUT_PIX];


		// load top_diff and scale tiles
		for( int b_j = lcl_id1; b_j < MLO_LRN_LCL_DATA_HEIGHT; b_j += MLO_LRN_GROUP_SZ1)
		{	
			int top_y_act = top_y + b_j - MLO_LRN_PAD;

			bool invisibleY = (top_y_act < 0) || (top_y_act >= MLO_LRN_TOP_HEIGHT);


			int top_df_y_off = top_y_act * MLO_LRN_TOPDF_STRIDE;
			int scale_y_off = top_y_act * MLO_LRN_SCALE_STRIDE;

			int lcl_off_v = b_j * MLO_LRN_LCL_DATA_WIDTH;

			for(int b_i = lcl_id0; b_i < MLO_LRN_LCL_DATA_WIDTH; b_i += MLO_LRN_GROUP_SZ0)
			{

				int top_x_act = top_x + b_i - MLO_LRN_PAD;

				bool invisibleX = (top_x_act < 0) || (top_x_act >= MLO_LRN_TOP_WIDTH);
			
				int top_df_off_x = (invisibleX || invisibleY) ?  0 : top_df_off + top_df_y_off + top_x_act;
				int scale_off_x = (invisibleX || invisibleY) ? 0 : scale_off + scale_y_off + top_x_act;

				_FLOAT top_df_val = top_df[top_df_off_x];
				_FLOAT scale_val = scale[scale_off_x];

				top_df_val = (invisibleX || invisibleY)?
							0 :
							top_df_val;
				scale_val = (invisibleX || invisibleY)?
							1.f :
							scale_val;

								
				top_df_data[lcl_off_v + b_i] = top_df_val;
				ratio_data[lcl_off_v + b_i] = scale_val;
				
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		// actual top_diffs and scales
		for(int j = 0; j < MLO_LRN_N_VERT_OUT_PIX; ++j)
		{
			int lcl_off_v = (lcl_id1 * MLO_LRN_N_VERT_OUT_PIX + MLO_LRN_PAD + j) *  MLO_LRN_LCL_DATA_WIDTH;
			for(int i = 0; i < MLO_LRN_N_HORIZ_OUT_PIX; i++)
			{
				_FLOAT scale = ratio_data[lcl_off_v + lcl_id0 * MLO_LRN_N_HORIZ_OUT_PIX + MLO_LRN_PAD + i];
				prv_exp_scale[j][i]= native_exp(-beta * native_log(scale));
//				prv_exp_scale[j][i]= pow(scale, -beta);
			}
		}

// read top and load ratio tile
		int top_off = b * MLO_LRN_TOP_BATCH_STRIDE + o * MLO_LRN_TOP_CHANNEL_STRIDE;
		for( int b_j = lcl_id1; b_j < MLO_LRN_LCL_DATA_HEIGHT; b_j += MLO_LRN_GROUP_SZ1)
		{	
			int top_y_act = top_y + b_j - MLO_LRN_PAD;

			bool invisibleY = (top_y_act < 0) || (top_y_act >= MLO_LRN_TOP_HEIGHT);

			int top_y_off = top_y_act * MLO_LRN_TOP_STRIDE;

			int lcl_off_v = b_j * MLO_LRN_LCL_DATA_WIDTH;

			for(int b_i = lcl_id0; b_i < MLO_LRN_LCL_DATA_WIDTH; b_i += MLO_LRN_GROUP_SZ0)
			{

				int top_x_act = top_x + b_i - MLO_LRN_PAD;

				bool invisibleX = (top_x_act < 0) || (top_x_act >= MLO_LRN_TOP_WIDTH);
			
				int top_off_x = (invisibleX || invisibleY) ? 0 : top_off + top_y_off + top_x_act;

				_FLOAT top_val = top[top_off_x];

				top_val = (invisibleX || invisibleY) ? 0 : top_val;

				_FLOAT top_df_val = top_df_data[lcl_off_v + b_i];

				_FLOAT scale_val = ratio_data[lcl_off_v + b_i];

	// scale val is not 0							
				_FLOAT 	ratio_dta = 
						(top_df_val * top_val) / scale_val;
	// replacing scale with ratio
				ratio_data[lcl_off_v + b_i] = ratio_dta;
				
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);


// caculate bot diff
		_FLOAT prv_bot_diff[MLO_LRN_N_VERT_OUT_PIX][MLO_LRN_N_HORIZ_OUT_PIX];

		for(int j = 0; j < MLO_LRN_N_VERT_OUT_PIX; ++j)
		{
			int v_off_v =  (lcl_id1 * MLO_LRN_N_VERT_OUT_PIX + j);
			int hstart = y + v_off_v - MLO_LRN_PAD;
			int hend = min(hstart + MLO_LRN_KERNEL_SZ, MLO_LRN_TOP_HEIGHT + MLO_LRN_PAD);

		// accum offset, vertical
//			int lcl_a_off_v = v_off_v *  MLO_LRN_LCL_DATA_WIDTH;
		// value offset, vertical
			int lcl_v_off_v = (v_off_v + MLO_LRN_PAD) *  MLO_LRN_LCL_DATA_WIDTH;
			for(int i = 0; i < MLO_LRN_N_HORIZ_OUT_PIX; i++)
			{
				_FLOAT prv_ratio_accum = 0;
				int v_off_h = lcl_id0 * MLO_LRN_N_HORIZ_OUT_PIX + i;

				int wstart = x + v_off_h - MLO_LRN_PAD;
				int wend = min(wstart + MLO_LRN_KERNEL_SZ, MLO_LRN_TOP_WIDTH + MLO_LRN_PAD);

				int adj_area_size = (hend - hstart) * (wend - wstart);

		// accum offset, horiz
				int lcl_a_off_h = v_off_h;
		//	value offset, horiz
				int lcl_v_off_h = lcl_a_off_h  + MLO_LRN_PAD;

				for(int k = 0; k < MLO_LRN_KERNEL_SZ; k++)
				{ 
					for(int l = 0; l < MLO_LRN_KERNEL_SZ; l++)
					{
						prv_ratio_accum += ratio_data[(v_off_v + k) * MLO_LRN_LCL_DATA_WIDTH + lcl_a_off_h + l];

					}
				}

				_FLOAT top_df_val = top_df_data[lcl_v_off_v + lcl_v_off_h];
				_FLOAT bot_dta = bot[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE *o + MLO_LRN_BOT_STRIDE * (y + v_off_v) + x + v_off_h];
				_FLOAT adj_ratio = 2.f * alpha * beta / adj_area_size;
				_FLOAT prv_accum_ratio =
					adj_ratio * bot_dta * prv_ratio_accum;
				prv_bot_diff[j][i] = 
					prv_exp_scale[j][i] * top_df_val - prv_accum_ratio;
			}
		}


		for( int j = 0; j < MLO_LRN_N_VERT_OUT_PIX; j++)
		{
			for(int i = 0; i < MLO_LRN_N_HORIZ_OUT_PIX; i++)
			{
				if (bot_y + j < MLO_LRN_BOT_HEIGHT && bot_x + i < MLO_LRN_BOT_WIDTH)
				{	

					bot_df[MLO_LRN_BOTDF_BATCH_STRIDE * b + MLO_LRN_BOTDF_CHANNEL_STRIDE *o + MLO_LRN_BOTDF_STRIDE * (bot_y + j) + bot_x + i] = prv_bot_diff[j][i];
				}
			}
		}


}

#if (MLO_LRN_N_INPUTS + 2* MLO_LRN_PAD - 1 < MLO_LRN_KERNEL_SZ || MLO_LRN_N_OUTPUTS + 2* MLO_LRN_PAD - 1 < MLO_LRN_KERNEL_SZ)
#define MLO_LOW_CHNL_COUNT 1
#else
#define MLO_LOW_CHNL_COUNT 0
#endif
__attribute__((reqd_work_group_size(MLO_LRN_GROUP_SZ0,MLO_LRN_GROUP_SZ1,MLO_LRN_GROUP_SZ2)))
__kernel void MLOpenLRNAcrossChannels4(
       const __global _FLOAT * bottom,
       __global _FLOAT * top,
#if MLO_LRN_DO_SCALE
	   __global _FLOAT *scale,
#endif
	   _FLOAT alphaoverarea,
	   _FLOAT alpha,
	   _FLOAT beta,
	   _FLOAT K
	   )
{
	
		int pix_id = get_global_id(0); // 
		int b = get_global_id(2); // batch 
		MLO_READ_TYPE accum = 0;
		MLO_READ_TYPE bot_in2[MLO_LRN_KERNEL_SZ];
		int c_i = 0, c_o = 0;
		for (int i = 0; i < MLO_LRN_KERNEL_SZ; ++i)
		{
			bot_in2[i] = 0;
		}

		int top_off = 0;
		int scale_off = 0;

		for( c_i = 0; c_i < MLO_LRN_PAD
#if MLO_LOW_CHNL_COUNT
			&& (c_i < MLO_LRN_N_INPUTS)
#endif
			; c_i++)
		{

			MLO_READ_TYPE prv_in;
#if MLO_C1x1_PIXLEFT > 0
			// if the last one
			if (pix_id == MLO_MAP_SZ4 - 1)
			{
				prv_in = 0;

				for (int j = 0; j < MLO_C1x1_PIXLEFT;++j)
				{
					((_FLOAT*)&prv_in)[j] = bottom[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE *c_i + (pix_id * MLO_READ_UNIT) + j];
				}
			}
			else
#endif
			{
				prv_in = *(__global MLO_READ_TYPE*)&bottom[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE *c_i + (pix_id * MLO_READ_UNIT)];
			}

			bot_in2[c_i] = prv_in * prv_in;
			accum = 
				accum + bot_in2[c_i];
//				fma(bot_in[c_i + MLO_LRN_PAD], bot_in[c_i + MLO_LRN_PAD], accum);

		}

		for( ; c_i < MLO_LRN_KERNEL_SZ
#if MLO_LOW_CHNL_COUNT
			&& (c_i < MLO_LRN_N_INPUTS)
#endif
			; c_i++, c_o++)
		{
			MLO_READ_TYPE prv_in;
#if MLO_C1x1_PIXLEFT > 0
			// if the last one
			if (pix_id == MLO_MAP_SZ4 - 1)
			{
				prv_in = 0;

				for (int j = 0; j < MLO_C1x1_PIXLEFT; ++j)
				{
					((_FLOAT*)&prv_in)[j] = bottom[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE *c_i + (pix_id * MLO_READ_UNIT) + j];
				}
			}
			else
#endif
			{
				prv_in = *(__global MLO_READ_TYPE*)&bottom[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE *c_i + (pix_id * MLO_READ_UNIT)];
			}

			bot_in2[c_i] = prv_in * prv_in;
			accum =
				accum + bot_in2[c_i];

			top_off = b * MLO_LRN_TOP_BATCH_STRIDE + c_o * MLO_LRN_TOP_CHANNEL_STRIDE + (pix_id * MLO_READ_UNIT);
			scale_off = b * MLO_LRN_SCALE_BATCH_STRIDE + c_o * MLO_LRN_SCALE_CHANNEL_STRIDE + (pix_id * MLO_READ_UNIT);
			MLO_READ_TYPE prv_scale = 
				((MLO_READ_TYPE)K + accum * (MLO_READ_TYPE)alphaoverarea);
//				fma(accum,alphaoverarea, 1.f);


			MLO_READ_TYPE exp_scale = 
				native_exp((MLO_READ_TYPE)-beta * native_log(prv_scale));
//				pow(prv_scale,-beta);
			//bug
			//	MLO_READ_TYPE prv_out = native_sqrt(bot_in2[c_o]);
			MLO_READ_TYPE prv_out = bot_in2[c_o];
			prv_out = native_sqrt(prv_out);
			MLO_READ_TYPE out_val = prv_out * exp_scale;
#if MLO_LOW_CHNL_COUNT
			if (c_o < MLO_LRN_N_OUTPUTS)
#endif
			{


#if MLO_C1x1_PIXLEFT > 0

				// if the last one
				if (pix_id == MLO_MAP_SZ4 - 1)
				{
					for (int j = 0; j < MLO_C1x1_PIXLEFT; ++j)
					{
						top[top_off + j] = ((_FLOAT*)&out_val)[j];
#if MLO_LRN_DO_SCALE
						scale[scale_off + j] = ((_FLOAT*)&prv_scale)[j];
#endif

					}

				}
				else
#endif
				{

					*((__global MLO_READ_TYPE*)&top[top_off]) = out_val;
#if MLO_LRN_DO_SCALE
					*((__global MLO_READ_TYPE*)&scale[scale_off]) = prv_scale;
#endif
				}


			}
		}

		for( ; c_i < MLO_LRN_N_CHANNELS
#if MLO_LOW_CHNL_COUNT
			&& (c_i < MLO_LRN_N_INPUTS)
#endif
			; c_i++, c_o++)
		{

			MLO_READ_TYPE prv_in;
#if MLO_C1x1_PIXLEFT > 0
			// if the last one
			if (pix_id == MLO_MAP_SZ4 - 1)
			{
				prv_in = 0;

				for (int j = 0; j < MLO_C1x1_PIXLEFT; ++j)
				{
					((_FLOAT*)&prv_in)[j] = bottom[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE *c_i + (pix_id * MLO_READ_UNIT) + j];
				}
			}
			else
#endif
			{
				prv_in = *(__global MLO_READ_TYPE*)&bottom[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE *c_i + (pix_id * MLO_READ_UNIT)];
			}

			MLO_READ_TYPE prv_bot_in2 = prv_in * prv_in;
			accum = 
				accum + prv_bot_in2;

			accum =
				accum - bot_in2[0];
//				fma(-bot_in[0], bot_in[0], accum);

			for( int i = 0; i < MLO_LRN_KERNEL_SZ - 1; i++)
			{
				bot_in2[i] = bot_in2[i+1];
			}

			bot_in2[MLO_LRN_KERNEL_SZ - 1] = prv_bot_in2;
		

			top_off = b * MLO_LRN_TOP_BATCH_STRIDE + c_o * MLO_LRN_TOP_CHANNEL_STRIDE + (pix_id * MLO_READ_UNIT);
			scale_off = b * MLO_LRN_SCALE_BATCH_STRIDE + c_o * MLO_LRN_SCALE_CHANNEL_STRIDE + (pix_id * MLO_READ_UNIT);
			MLO_READ_TYPE prv_scale =
				((MLO_READ_TYPE)K + accum * (MLO_READ_TYPE)alphaoverarea);
			//				fma(accum,alphaoverarea, 1.f);


			MLO_READ_TYPE exp_scale =
				native_exp((MLO_READ_TYPE)-beta * native_log(prv_scale));
			//				pow(prv_scale,-beta);
			//bug
			//			MLO_READ_TYPE prv_out = native_sqrt(bot_in2[MLO_LRN_PAD]);
			MLO_READ_TYPE prv_out = bot_in2[MLO_LRN_PAD];
			prv_out = native_sqrt(prv_out);
			MLO_READ_TYPE out_val = prv_out * exp_scale;

#if MLO_LOW_CHNL_COUNT
			if (c_o < MLO_LRN_N_OUTPUTS)
#endif
			{


#if MLO_C1x1_PIXLEFT > 0

				// if the last one
				if (pix_id == MLO_MAP_SZ4 - 1)
				{
					for (int j = 0; j < MLO_C1x1_PIXLEFT; ++j)
					{
						top[top_off + j] = ((_FLOAT*)&out_val)[j];
#if MLO_LRN_DO_SCALE
						scale[scale_off + j] = ((_FLOAT*)&prv_scale)[j];
#endif

					}

				}
				else
#endif
				{

					*((__global MLO_READ_TYPE*)&top[top_off]) = out_val;
#if MLO_LRN_DO_SCALE
					*((__global MLO_READ_TYPE*)&scale[scale_off]) = prv_scale;
#endif
				}


			}

		}

		for(  ; c_i < MLO_LRN_N_CHANNELS + MLO_LRN_PAD; c_i++, c_o++)
		{

			accum = 
				accum - bot_in2[0];
//				fma(-bot_in[0], bot_in[0], accum);

			for( int i = 0; i < MLO_LRN_KERNEL_SZ - 1; i++)
			{
				bot_in2[i] = bot_in2[i+1];
			}

			top_off = b * MLO_LRN_TOP_BATCH_STRIDE + c_o * MLO_LRN_TOP_CHANNEL_STRIDE + (pix_id * MLO_READ_UNIT);
			scale_off = b * MLO_LRN_SCALE_BATCH_STRIDE + c_o * MLO_LRN_SCALE_CHANNEL_STRIDE + (pix_id * MLO_READ_UNIT);
			MLO_READ_TYPE prv_scale =
				((MLO_READ_TYPE)K + accum * (MLO_READ_TYPE)alphaoverarea);
			//				fma(accum,alphaoverarea, 1.f);


			MLO_READ_TYPE exp_scale =
				native_exp((MLO_READ_TYPE)-beta * native_log(prv_scale));
			//				pow(prv_scale,-beta);
//bug
//			MLO_READ_TYPE prv_out = native_sqrt(bot_in2[MLO_LRN_PAD]);
			MLO_READ_TYPE prv_out = bot_in2[MLO_LRN_PAD];
			prv_out = native_sqrt(prv_out);

			MLO_READ_TYPE out_val = prv_out * exp_scale;
#if MLO_LOW_CHNL_COUNT
			if (c_o < MLO_LRN_N_OUTPUTS)
#endif
			{


#if MLO_C1x1_PIXLEFT > 0

				// if the last one
				if (pix_id == MLO_MAP_SZ4 - 1)
				{
					for (int j = 0; j < MLO_C1x1_PIXLEFT; ++j)
					{
						top[top_off + j] = ((_FLOAT*)&out_val)[j];
#if MLO_LRN_DO_SCALE
						scale[scale_off + j] = ((_FLOAT*)&prv_scale)[j];
#endif

					}

				}
				else
#endif
				{

					*((__global MLO_READ_TYPE*)&top[top_off]) = out_val;
#if MLO_LRN_DO_SCALE
					*((__global MLO_READ_TYPE*)&scale[scale_off]) = prv_scale;
#endif
				}


			}
		}

}


__attribute__((reqd_work_group_size(MLO_LRN_GROUP_SZ0,MLO_LRN_GROUP_SZ1,MLO_LRN_GROUP_SZ2)))
__kernel void MLOpenLRNAcrossChannelsBwd1(
       const __global _FLOAT * top,
	   const __global _FLOAT *	bot,
       const __global _FLOAT * top_df,
	   const __global _FLOAT *	scale,
       __global _FLOAT * bot_df,
	   _FLOAT ratio, //2. * alpha * beta / local_area
	   _FLOAT alpha,
	   _FLOAT beta
	   )
{
	
		int x = get_global_id(0); // channel x
		int y = get_global_id(1); // channel y
		int b = get_global_id(2); // batch 
		_FLOAT accum_ratio = 0;
		_FLOAT top_df_in[MLO_LRN_KERNEL_SZ];
		_FLOAT scale_in[MLO_LRN_KERNEL_SZ];
		_FLOAT ratio_dta[MLO_LRN_KERNEL_SZ];
		int c_i = 0, c_o = 0;
		int bot_df_off = 0;

		for (int i = 0; i < MLO_LRN_KERNEL_SZ; ++i)
		{
			top_df_in[i] = 0;
			scale_in[i] = 0;
			ratio_dta[i] = 0;
		}

		for( c_i = 0; c_i < MLO_LRN_PAD
#if MLO_LOW_CHNL_COUNT
			&& (c_i < MLO_LRN_N_OUTPUTS)
#endif
			; c_i++)
		{
			
			top_df_in[c_i] = top_df[MLO_LRN_TOPDF_BATCH_STRIDE * b + MLO_LRN_TOPDF_CHANNEL_STRIDE *c_i + MLO_LRN_TOPDF_STRIDE * y + x];
			scale_in[c_i] = scale[MLO_LRN_SCALE_BATCH_STRIDE * b + MLO_LRN_SCALE_CHANNEL_STRIDE * c_i + MLO_LRN_SCALE_STRIDE * y + x];
			_FLOAT top_dta = top[MLO_LRN_TOP_BATCH_STRIDE *b  + MLO_LRN_TOP_CHANNEL_STRIDE * c_i + MLO_LRN_TOP_STRIDE * y + x];

			ratio_dta[c_i] = (top_df_in[c_i] * top_dta) / scale_in[c_i];


 			accum_ratio = 
				accum_ratio + ratio_dta[c_i] ;


		}

		for( ; c_i < MLO_LRN_KERNEL_SZ
#if MLO_LOW_CHNL_COUNT
			&& (c_i < MLO_LRN_N_OUTPUTS)
#endif
			; c_i++, c_o++)
		{
			top_df_in[c_i] = top_df[MLO_LRN_TOPDF_BATCH_STRIDE * b + MLO_LRN_TOPDF_CHANNEL_STRIDE *c_i + MLO_LRN_TOPDF_STRIDE * y + x];
			scale_in[c_i] = scale[MLO_LRN_SCALE_BATCH_STRIDE * b + MLO_LRN_SCALE_CHANNEL_STRIDE * c_i + MLO_LRN_SCALE_STRIDE * y + x];
			_FLOAT top_dta = top[MLO_LRN_TOP_BATCH_STRIDE *b  + MLO_LRN_TOP_CHANNEL_STRIDE * c_i + MLO_LRN_TOP_STRIDE * y + x];
			ratio_dta[c_i] = (top_df_in[c_i] * top_dta) / scale_in[c_i];


 			accum_ratio = 
				accum_ratio + ratio_dta[c_i] ;
#if MLO_LOW_CHNL_COUNT
			if (c_o < MLO_LRN_N_INPUTS)
#endif
			{
				_FLOAT bot_dta = bot[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE *c_o + MLO_LRN_BOT_STRIDE * y + x];

				_FLOAT prv_scale = scale_in[c_o];


				_FLOAT exp_scale =
					native_exp(-beta * native_log(prv_scale));
//					pow(prv_scale, -beta);

				_FLOAT prv_accum_ratio =
					ratio * bot_dta * accum_ratio;

				_FLOAT out_val = top_df_in[c_o] * exp_scale - prv_accum_ratio;

				bot_df_off = MLO_LRN_BOTDF_BATCH_STRIDE * b + MLO_LRN_BOTDF_CHANNEL_STRIDE * c_o + MLO_LRN_BOTDF_STRIDE * y + x;

				bot_df[bot_df_off] = out_val;
			}

		}

		for( ; c_i < MLO_LRN_N_CHANNELS
#if MLO_LOW_CHNL_COUNT
			&& (c_i < MLO_LRN_N_OUTPUTS)
#endif
			; c_i++, c_o++)
		{


			_FLOAT prv_top_df_in = top_df[MLO_LRN_TOPDF_BATCH_STRIDE * b + MLO_LRN_TOPDF_CHANNEL_STRIDE *c_i + MLO_LRN_TOPDF_STRIDE * y + x];
			_FLOAT prv_scale_in = scale[MLO_LRN_SCALE_BATCH_STRIDE * b + MLO_LRN_SCALE_CHANNEL_STRIDE * c_i + MLO_LRN_SCALE_STRIDE * y + x];
			_FLOAT top_dta = top[MLO_LRN_TOP_BATCH_STRIDE *b  + MLO_LRN_TOP_CHANNEL_STRIDE * c_i + MLO_LRN_TOP_STRIDE * y + x];
			_FLOAT prv_ratio_dta = prv_top_df_in * top_dta / prv_scale_in;



 			accum_ratio = 
				accum_ratio + prv_ratio_dta ;

			accum_ratio =
				accum_ratio - ratio_dta[0];


			for( int i = 0; i < MLO_LRN_KERNEL_SZ - 1; i++)
			{
				top_df_in[i] = top_df_in[i+1];
				scale_in[i] = scale_in[i+1];
				ratio_dta[i] = ratio_dta[i+1];
			}

			top_df_in[MLO_LRN_KERNEL_SZ - 1] = prv_top_df_in;
			scale_in[MLO_LRN_KERNEL_SZ - 1] = prv_scale_in;
			ratio_dta[MLO_LRN_KERNEL_SZ - 1]= prv_ratio_dta;
			
#if MLO_LOW_CHNL_COUNT
			if (c_o < MLO_LRN_N_INPUTS)
#endif
			{
				_FLOAT bot_dta = bot[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE *c_o + MLO_LRN_BOT_STRIDE * y + x];

				_FLOAT prv_scale = scale_in[MLO_LRN_PAD];


				_FLOAT exp_scale =
					native_exp(-beta * native_log(prv_scale));
				//				pow(prv_scale,-beta);

				_FLOAT prv_accum_ratio =
					ratio * bot_dta * accum_ratio;

				_FLOAT out_val = top_df_in[MLO_LRN_PAD] * exp_scale - prv_accum_ratio;

				bot_df_off = MLO_LRN_BOTDF_BATCH_STRIDE * b + MLO_LRN_BOTDF_CHANNEL_STRIDE * c_o + MLO_LRN_BOTDF_STRIDE * y + x;

				bot_df[bot_df_off] = out_val;
			}



		}

		for (; c_i < MLO_LRN_N_CHANNELS + MLO_LRN_PAD; c_i++, c_o++)
		{


			accum_ratio =
				accum_ratio - ratio_dta[0];



			for (int i = 0; i < MLO_LRN_KERNEL_SZ - 1; i++)
			{
				top_df_in[i] = top_df_in[i + 1];
				scale_in[i] = scale_in[i + 1];
				ratio_dta[i] = ratio_dta[i + 1];

			}

#if MLO_LOW_CHNL_COUNT
			if (c_o < MLO_LRN_N_INPUTS)
#endif
			{
				_FLOAT bot_dta = bot[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE *c_o + MLO_LRN_BOT_STRIDE * y + x];

				_FLOAT prv_scale = scale_in[MLO_LRN_PAD];


				_FLOAT exp_scale =
					native_exp(-beta * native_log(prv_scale));
				//				pow(prv_scale,-beta);

				_FLOAT prv_accum_ratio =
					ratio * bot_dta * accum_ratio;

				_FLOAT out_val = top_df_in[MLO_LRN_PAD] * exp_scale - prv_accum_ratio;

				bot_df_off = MLO_LRN_BOTDF_BATCH_STRIDE * b + MLO_LRN_BOTDF_CHANNEL_STRIDE * c_o + MLO_LRN_BOTDF_STRIDE * y + x;

				bot_df[bot_df_off] = out_val;
			}
		}

}


