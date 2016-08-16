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

#define MLO_LRN_LCL_DATA_WIDTH (MLO_LRN_GROUP_SZ0 *MLO_LRN_N_HORIZ_OUT_PIX + MLO_LRN_KERNEL_SZ - 1)
#define MLO_LRN_LCL_DATA_HEIGHT  (MLO_LRN_GROUP_SZ1 * MLO_LRN_N_VERT_OUT_PIX + MLO_LRN_KERNEL_SZ - 1)
#define MLO_LRN_GROUP_SZ (MLO_LRN_GROUP_SZ2 * MLO_LRN_GROUP_SZ1 * MLO_LRN_GROUP_SZ0)
//#define MLO_LRN_PREPAD_SZ (MLO_LRN_KERNEL_SZ - 1)/2

__attribute__((reqd_work_group_size(MLO_LRN_GROUP_SZ0,MLO_LRN_GROUP_SZ1,MLO_LRN_GROUP_SZ2)))
__kernel void MLOpenLRNWithinChannel(
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
		int o = ob / MLO_LRN_BATCH_SZ;
		int b = ob - o * MLO_LRN_BATCH_SZ;
		int bot_x = x;
		int bot_y = y;
		int bot_off = b * MLO_LRN_BOT_BATCH_STRIDE + o * MLO_LRN_BOT_CHANNEL_STRIDE;


		_FLOAT prv_scale[MLO_LRN_N_VERT_OUT_PIX][MLO_LRN_N_HORIZ_OUT_PIX];


// load tile
		for( int b_j = lcl_id1; b_j < MLO_LRN_LCL_DATA_HEIGHT; b_j += MLO_LRN_GROUP_SZ1)
		{	
			int bot_y_act = bot_y + b_j - MLO_LRN_PAD;

			bool invisibleY = (bot_y_act < 0) || (bot_y_act >= MLO_LRN_BOT_HEIGHT);

			bot_y_act = (bot_y_act < 0) ? 0 : (bot_y_act >= MLO_LRN_BOT_HEIGHT) ? MLO_LRN_BOT_HEIGHT - 1: bot_y_act;

			int bot_y_off = bot_y_act * MLO_LRN_BOT_STRIDE;

			int lcl_off_v = b_j * MLO_LRN_LCL_DATA_WIDTH;

			for(int b_i = lcl_id0; b_i < MLO_LRN_LCL_DATA_WIDTH; b_i += MLO_LRN_GROUP_SZ0)
			{

				int bot_x_act = bot_x + b_i - MLO_LRN_PAD;

				bool invisibleX = (bot_x_act < 0) || (bot_x_act >= MLO_LRN_BOT_WIDTH);
			
				bot_x_act = (bot_x_act < 0) ? 0 : (bot_x_act >= MLO_LRN_BOT_WIDTH) ? MLO_LRN_BOT_WIDTH - 1: bot_x_act;

				_FLOAT bot_val = bot[bot_off + bot_y_off + bot_x_act];

				bot_val = (invisibleX || invisibleY)?
							0 :
							bot_val;

				bot_val = bot_val;
								
				bot_data[lcl_off_v + b_i] = bot_val;
				
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);


		int top_y = (y + lcl_id1 * MLO_LRN_N_VERT_OUT_PIX);
		int top_x = (x + lcl_id0 * MLO_LRN_N_HORIZ_OUT_PIX);

		int lcl_y = lcl_id1 * MLO_LRN_N_VERT_OUT_PIX;
		int lcl_x = lcl_id0 * MLO_LRN_N_HORIZ_OUT_PIX;
		int lcl_off = lcl_y * MLO_LRN_LCL_DATA_WIDTH + lcl_x;
		
		for( int k = 0; k < MLO_LRN_N_VERT_OUT_PIX; k++)
		{

			int hstart = y + lcl_id1 * MLO_LRN_N_VERT_OUT_PIX  + k - MLO_LRN_PAD;
			int hend = min(hstart + MLO_LRN_KERNEL_SZ, MLO_LRN_BOT_HEIGHT + MLO_LRN_PAD);

			for(int l = 0; l < MLO_LRN_N_HORIZ_OUT_PIX; l++)
			{

				int wstart = x + lcl_id0 * MLO_LRN_N_HORIZ_OUT_PIX + l - MLO_LRN_PAD;
				int wend = min(wstart + MLO_LRN_KERNEL_SZ, MLO_LRN_BOT_WIDTH + MLO_LRN_PAD);

				int adj_area_size = (hend - hstart) * (wend - wstart);

				_FLOAT accum = 0;
				for( int j = 0; j < MLO_LRN_KERNEL_SZ; j++)
				{
					for(int i = 0; i < MLO_LRN_KERNEL_SZ; i++)
					{

						_FLOAT bot_val =  bot_data[lcl_off + (k +j)*MLO_LRN_LCL_DATA_WIDTH + (l +i)];
						accum   += bot_val * bot_val;

					}
				}

				_FLOAT adj_alphaoverarea = alpha / adj_area_size;
				prv_scale[k][l]  = K + accum * adj_alphaoverarea ;

#if 0
				if ( top_x + l == 7 && top_y + k == 0 && o == 0)
				{
					printf("K:lrn: %13.11f %13.11f %13.11f\n", prv_scale[k][l], accum, adj_alphaoverarea);
				}
#endif
			}
		}

		int top_off = b * MLO_LRN_TOP_BATCH_STRIDE + o * MLO_LRN_TOP_CHANNEL_STRIDE + top_y * MLO_LRN_TOP_STRIDE + top_x;
		int scale_off = b * MLO_LRN_SCALE_BATCH_STRIDE + o * MLO_LRN_SCALE_CHANNEL_STRIDE + top_y * MLO_LRN_SCALE_STRIDE + top_x;


		for( int k = 0; k < MLO_LRN_N_VERT_OUT_PIX; k++)
		{
			for(int l = 0; l < MLO_LRN_N_HORIZ_OUT_PIX; l++)
			{
				_FLOAT s;
				_FLOAT bot_val;

				if (top_y + k < MLO_LRN_TOP_HEIGHT && top_x + l < MLO_LRN_TOP_WIDTH)
				{	

//					s = native_exp((_FLOAT)-beta * native_log(prv_scale[k][l]));
					s = pow(prv_scale[k][l], -beta);
					bot_val =  bot_data[lcl_off + (k +MLO_LRN_PAD)*MLO_LRN_LCL_DATA_WIDTH + (l + MLO_LRN_PAD)];
#if MLO_LRN_DO_SCALE
					scale[scale_off + k * MLO_LRN_SCALE_STRIDE +l] = prv_scale[k][l];
#endif
					top[top_off + k * MLO_LRN_TOP_STRIDE +l] = bot_val * s;
				}
#if 0
				if ( top_x + l == 9 && top_y + k == 4 && o == 0)
				{
					printf("K:lrn: %13.11f %13.11f %13.11f %13.11f\n", top[top_off + k * MLO_LRN_TOP_STRIDE +l], bot_val, s, prv_scale[k][l]);
				}
#endif

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

//			top_y_act = (bot_y_act < 0) ? 0 : (bot_y_act >= MLO_LRN_BOT_HEIGHT) ? MLO_LRN_BOT_HEIGHT - 1: bot_y_act;

			int top_df_y_off = top_y_act * MLO_LRN_TOPDF_STRIDE;
			int scale_y_off = top_y_act * MLO_LRN_SCALE_STRIDE;

			int lcl_off_v = b_j * MLO_LRN_LCL_DATA_WIDTH;

			for(int b_i = lcl_id0; b_i < MLO_LRN_LCL_DATA_WIDTH; b_i += MLO_LRN_GROUP_SZ0)
			{

				int top_x_act = top_x + b_i - MLO_LRN_PAD;

				bool invisibleX = (top_x_act < 0) || (top_x_act >= MLO_LRN_TOP_WIDTH);
			
//				bot_x_act = (bot_x_act < 0) ? 0 : (bot_x_act >= MLO_LRN_BOT_WIDTH) ? MLO_LRN_BOT_WIDTH - 1: bot_x_act;

				_FLOAT top_df_val = top_df[top_df_off + top_df_y_off + top_x_act];
				_FLOAT scale_val = scale[scale_off + scale_y_off + top_x_act];

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
//				prv_exp_scale[j][i]= native_exp(-beta * native_log(scale));
				prv_exp_scale[j][i]= pow(scale, -beta);
#if 0
				if (o==0 && b==0 && (bot_x + i) ==2 && (bot_y + j) == 0)
				{
					printf("K:scl: %d %d  %11.9f %11.9f %11.9f\n",
					i, j,
					prv_exp_scale[j][i],
					beta, scale
					);
				}
#endif

//				_FLOAT top_df_val = top_df_data[lcl_off_v + lcl_id0 * MLO_LRN_N_HORIZ_OUT_PIX + MLO_LRN_PAD + i];
//				prv_top_df[j][i] = top_df_val;
			}
		}

// read top and load ratio tile
		int top_off = b * MLO_LRN_TOP_BATCH_STRIDE + o * MLO_LRN_TOP_CHANNEL_STRIDE;
		for( int b_j = lcl_id1; b_j < MLO_LRN_LCL_DATA_HEIGHT; b_j += MLO_LRN_GROUP_SZ1)
		{	
			int top_y_act = top_y + b_j - MLO_LRN_PAD;

			bool invisibleY = (top_y_act < 0) || (top_y_act >= MLO_LRN_TOP_HEIGHT);

//			top_y_act = (bot_y_act < 0) ? 0 : (bot_y_act >= MLO_LRN_BOT_HEIGHT) ? MLO_LRN_BOT_HEIGHT - 1: bot_y_act;

			int top_y_off = top_y_act * MLO_LRN_TOP_STRIDE;

			int lcl_off_v = b_j * MLO_LRN_LCL_DATA_WIDTH;

			for(int b_i = lcl_id0; b_i < MLO_LRN_LCL_DATA_WIDTH; b_i += MLO_LRN_GROUP_SZ0)
			{

				int top_x_act = top_x + b_i - MLO_LRN_PAD;

				bool invisibleX = (top_x_act < 0) || (top_x_act >= MLO_LRN_TOP_WIDTH);
			
//				bot_x_act = (bot_x_act < 0) ? 0 : (bot_x_act >= MLO_LRN_BOT_WIDTH) ? MLO_LRN_BOT_WIDTH - 1: bot_x_act;

				_FLOAT top_val = top[top_off + top_y_off + top_x_act];

				top_val = (invisibleX || invisibleY) ? 0 : top_val;

				_FLOAT top_df_val = top_df_data[lcl_off_v + b_i];

				_FLOAT scale_val = ratio_data[lcl_off_v + b_i];

	// scale val is not 0							
				_FLOAT 	ratio_dta = 
						(top_df_val * top_val) / scale_val;
#if 0
				if (o==0 && b==0 && (b_i >=0 && b_i < 8) && (b_j == 0))
				{
					printf("K:rdt: %d %d  %11.9f %11.9f %11.9f %11.9f\n",
					b_i, b_j,
					top_df_val, top_val,scale_val,
					ratio_dta
					);
				}
#endif
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
#if 0
				if (o==0 && b==0 && (bot_x + i) ==2 && (bot_y + j) == 0)
				{
					printf("K:prv: %d %d  %d %d  %11.9f %11.9f\n",
					l, k,
					(v_off_v + k) * MLO_LRN_LCL_DATA_WIDTH + lcl_a_off_h + l,
					MLO_LRN_LCL_DATA_WIDTH * MLO_LRN_LCL_DATA_HEIGHT,
					prv_ratio_accum,
					ratio_data[(v_off_v + k)  * MLO_LRN_LCL_DATA_WIDTH + lcl_a_off_h + l]	
					);
				}
#endif

					}
				}

				_FLOAT top_df_val = top_df_data[lcl_v_off_v + lcl_v_off_h];
				_FLOAT bot_dta = bot[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE *o + MLO_LRN_BOT_STRIDE * (y + v_off_v) + x + v_off_h];
				_FLOAT adj_ratio = 2.f * alpha * beta / adj_area_size;
				_FLOAT prv_accum_ratio =
					adj_ratio * bot_dta * prv_ratio_accum;
				prv_bot_diff[j][i] = 
					prv_exp_scale[j][i] * top_df_val - prv_accum_ratio;
#if 0
				if (o==0 && b==0 && (bot_x + i) ==2 && (bot_y + j) == 0)
				{
					printf("K:lrn: %11.9f %11.9f %11.9f\n%11.9f %11.9f %11.9f\n%11.9f\n",
					adj_ratio, bot_dta, prv_ratio_accum,
					prv_exp_scale[j][i], top_df_val, prv_accum_ratio,
					prv_bot_diff[j][i]
					);
				}
#endif
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


__attribute__((reqd_work_group_size(MLO_LRN_GROUP_SZ0,MLO_LRN_GROUP_SZ1,MLO_LRN_GROUP_SZ2)))
__kernel void MLOpenLRNAcrossChannels1(
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
	
		int x = get_global_id(0); // channel x
		int y = get_global_id(1); // channel y
		int b = get_global_id(2); // batch 
		_FLOAT accum = 0;
		_FLOAT bot_in[MLO_LRN_KERNEL_SZ];
		_FLOAT bot_in2[MLO_LRN_KERNEL_SZ];
		int c_i = 0, c_o = 0;


		int top_off = 0;
		int scale_off = 0;

		for( c_i = 0; c_i < MLO_LRN_PAD; c_i++)
		{			
			bot_in[c_i] = bottom[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE *c_i + MLO_LRN_BOT_STRIDE * y + x];
			bot_in2[c_i] = bot_in[c_i] * bot_in[c_i];
			accum = 
				accum + bot_in2[c_i];
//				fma(bot_in[c_i + MLO_LRN_PAD], bot_in[c_i + MLO_LRN_PAD], accum);

		}

		for( ; c_i < MLO_LRN_KERNEL_SZ; c_i++, c_o++)
		{
			bot_in[c_i] = bottom[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE *c_i + MLO_LRN_BOT_STRIDE * y + x];
			bot_in2[c_i] = bot_in[c_i] * bot_in[c_i];
			accum = 
				accum + bot_in2[c_i];

			top_off = b * MLO_LRN_TOP_BATCH_STRIDE + c_o * MLO_LRN_TOP_CHANNEL_STRIDE + y * MLO_LRN_TOP_STRIDE +x;
			scale_off = b * MLO_LRN_SCALE_BATCH_STRIDE + c_o * MLO_LRN_SCALE_CHANNEL_STRIDE + y * MLO_LRN_SCALE_STRIDE +x;
			_FLOAT prv_scale = 
				(K + accum * alphaoverarea);
//				fma(accum,alphaoverarea, 1.f);


			_FLOAT exp_scale = 
				native_exp(-beta * native_log(prv_scale));
//				pow(prv_scale,-beta);

			_FLOAT out_val = bot_in[c_o] * exp_scale;
#if MLO_LRN_DO_SCALE
			scale[scale_off] = prv_scale;
#endif
			top[top_off] = out_val;
		}

		for( ; c_i < MLO_LRN_N_CHANNELS; c_i++, c_o++)
		{

			_FLOAT prv_bot_in = bottom[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE *c_i + MLO_LRN_BOT_STRIDE * y + x];
			_FLOAT prv_bot_in2 = prv_bot_in * prv_bot_in;
			accum = 
				accum + prv_bot_in2;

			accum =
				accum - bot_in2[0];
//				fma(-bot_in[0], bot_in[0], accum);

			for( int i = 0; i < MLO_LRN_KERNEL_SZ - 1; i++)
			{
				bot_in[i] = bot_in[i+1];
				bot_in2[i] = bot_in2[i+1];
			}

			bot_in[MLO_LRN_KERNEL_SZ - 1] = prv_bot_in;
			bot_in2[MLO_LRN_KERNEL_SZ - 1] = prv_bot_in2;
		

			top_off = b * MLO_LRN_TOP_BATCH_STRIDE + c_o * MLO_LRN_TOP_CHANNEL_STRIDE + y * MLO_LRN_TOP_STRIDE +x;
			scale_off = b * MLO_LRN_SCALE_BATCH_STRIDE + c_o * MLO_LRN_SCALE_CHANNEL_STRIDE + y * MLO_LRN_SCALE_STRIDE +x;
			_FLOAT prv_scale = 
				(K + accum * alphaoverarea);
//				fma(accum,alphaoverarea, 1.f);


			_FLOAT exp_scale = 
				native_exp(-beta * native_log(prv_scale));
//				pow(prv_scale,-beta);

			_FLOAT out_val = bot_in[MLO_LRN_PAD] * exp_scale;
#if MLO_LRN_DO_SCALE
			scale[scale_off] = prv_scale;
#endif
			top[top_off] = out_val;


		}

		for(  ; c_i < MLO_LRN_N_CHANNELS + MLO_LRN_PAD; c_i++, c_o++)
		{

			accum = 
				accum - bot_in2[0];
//				fma(-bot_in[0], bot_in[0], accum);

			for( int i = 0; i < MLO_LRN_KERNEL_SZ - 1; i++)
			{
				bot_in[i] = bot_in[i+1];
				bot_in2[i] = bot_in2[i+1];
			}

			top_off = b * MLO_LRN_TOP_BATCH_STRIDE + c_o * MLO_LRN_TOP_CHANNEL_STRIDE + y * MLO_LRN_TOP_STRIDE + x;
			scale_off = b * MLO_LRN_SCALE_BATCH_STRIDE + c_o * MLO_LRN_SCALE_CHANNEL_STRIDE + y * MLO_LRN_SCALE_STRIDE +x;

			_FLOAT prv_scale = 
				(K + accum * alphaoverarea);
//				fma(accum,alphaoverarea, 1.f);

			_FLOAT exp_scale = 
//				native_exp(-beta * native_log(prv_scale));
				pow(prv_scale,-beta);

			_FLOAT out_val = bot_in[MLO_LRN_PAD] * exp_scale;
#if MLO_LRN_DO_SCALE
			scale[scale_off] = prv_scale;
#endif
			top[top_off] = out_val;

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

		for( c_i = 0; c_i < MLO_LRN_PAD; c_i++)
		{
			
			top_df_in[c_i] = top_df[MLO_LRN_TOPDF_BATCH_STRIDE * b + MLO_LRN_TOPDF_CHANNEL_STRIDE *c_i + MLO_LRN_TOPDF_STRIDE * y + x];
			scale_in[c_i] = scale[MLO_LRN_SCALE_BATCH_STRIDE * b + MLO_LRN_SCALE_CHANNEL_STRIDE * c_i + MLO_LRN_SCALE_STRIDE * y + x];
			_FLOAT top_dta = top[MLO_LRN_TOP_BATCH_STRIDE *b  + MLO_LRN_TOP_CHANNEL_STRIDE * c_i + MLO_LRN_TOP_STRIDE * y + x];
			ratio_dta[c_i] = 
						(top_df_in[c_i] * top_dta) /scale_in[c_i];

#if 0
			if ( x == 5 && y == 11/* && c_o==12 */&& b==10)
			{
				printf("K:a %d %f %f\n",
					c_i,
					accum_ratio,
					ratio_dta[c_i]

				);

			}

#endif


 			accum_ratio = 
				accum_ratio + ratio_dta[c_i] ;


		}

		for( ; c_i < MLO_LRN_KERNEL_SZ; c_i++, c_o++)
		{
			top_df_in[c_i] = top_df[MLO_LRN_TOPDF_BATCH_STRIDE * b + MLO_LRN_TOPDF_CHANNEL_STRIDE *c_i + MLO_LRN_TOPDF_STRIDE * y + x];
			scale_in[c_i] = scale[MLO_LRN_SCALE_BATCH_STRIDE * b + MLO_LRN_SCALE_CHANNEL_STRIDE * c_i + MLO_LRN_SCALE_STRIDE * y + x];
			_FLOAT top_dta = top[MLO_LRN_TOP_BATCH_STRIDE *b  + MLO_LRN_TOP_CHANNEL_STRIDE * c_i + MLO_LRN_TOP_STRIDE * y + x];
			ratio_dta[c_i] = 
						(top_df_in[c_i] * top_dta) /scale_in[c_i];

#if 0
			if ( x == 5 && y == 11/* && c_o==12 */&& b==10)
			{
				printf("K:a %d %f %f\n",
					c_i,
					accum_ratio,
					ratio_dta[c_i]

				);

			}

#endif

 			accum_ratio = 
				accum_ratio + ratio_dta[c_i] ;

			_FLOAT bot_dta = bot[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE *c_o + MLO_LRN_BOT_STRIDE * y + x];

			_FLOAT prv_scale = scale_in[c_o];


			_FLOAT exp_scale = 
//				native_exp(-beta * native_log(prv_scale));
				pow(prv_scale,-beta);

			_FLOAT prv_accum_ratio =
				ratio * bot_dta * accum_ratio;

			_FLOAT out_val = top_df_in[c_o] * exp_scale - prv_accum_ratio;

			bot_df_off = MLO_LRN_BOTDF_BATCH_STRIDE * b + MLO_LRN_BOTDF_CHANNEL_STRIDE * c_o + MLO_LRN_BOTDF_STRIDE * y + x;

			bot_df[bot_df_off] = out_val;


		}

		for( ; c_i < MLO_LRN_N_CHANNELS; c_i++, c_o++)
		{


			_FLOAT prv_top_df_in = top_df[MLO_LRN_TOPDF_BATCH_STRIDE * b + MLO_LRN_TOPDF_CHANNEL_STRIDE *c_i + MLO_LRN_TOPDF_STRIDE * y + x];
			_FLOAT prv_scale_in = scale[MLO_LRN_SCALE_BATCH_STRIDE * b + MLO_LRN_SCALE_CHANNEL_STRIDE * c_i + MLO_LRN_SCALE_STRIDE * y + x];
			_FLOAT top_dta = top[MLO_LRN_TOP_BATCH_STRIDE *b  + MLO_LRN_TOP_CHANNEL_STRIDE * c_i + MLO_LRN_TOP_STRIDE * y + x];
			_FLOAT prv_ratio_dta= 
						prv_top_df_in * top_dta / prv_scale_in;

#if 0
			if ( x == 5 && y == 11/* && c_o==12 */&& b==10)
			{
				printf("K:a %d %f %f\n",
					c_i,
					accum_ratio,
					prv_ratio_dta

				);

			}

#endif



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
			
#if 0
			if ( x == 3 && y == 0/* && c_o==12 */&& b==3)
			{
				printf("K: %d %16.12f %16.12f %16.12f %16.12f %16.12f\n",
					c_i,
					accum_ratio,
					prv_ratio_dta,
					prv_top_df_in,
					top_dta,
					prv_scale_in
				);

			}

#endif


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

		for(  ; c_i < MLO_LRN_N_CHANNELS + MLO_LRN_PAD; c_i++, c_o++)
		{


			accum_ratio =
				accum_ratio - ratio_dta[0];



			for( int i = 0; i < MLO_LRN_KERNEL_SZ - 1; i++)
			{
				top_df_in[i] = top_df_in[i+1];
				scale_in[i] = scale_in[i+1];
				ratio_dta[i] = ratio_dta[i+1];

			}


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

 #if 0
			if ( x == 5 && y == 11/* && c_o==12 */&& b== 10)
			{
				printf("K: %d %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f\n",
					c_i,
					bot_df[bot_df_off],
					top_df_in[MLO_LRN_PAD],
					exp_scale,
					prv_scale,
					-prv_accum_ratio,
					bot_dta,
					accum_ratio
				);

			}

#endif

		}

}







#if 0
__attribute__((reqd_work_group_size(MLO_LRN_GROUP_SZ0,MLO_LRN_GROUP_SZ1,MLO_LRN_GROUP_SZ2)))
__kernel void aDNNLRNAcrossChannels4(
       const __global _FLOAT * bottom,
       __global _FLOAT * top,
	   int bot_width,
	   int bot_height,
	   int bot_stride,
	   int bot_channel_stride,
	   int bot_batch_stride,
	   int top_stride,
	   int top_channel_stride,
	   int top_batch_stride,
	   _FLOAT padding_value,
	   _FLOAT alphaoverarea,
	   _FLOAT beta
	   )
{
	
		int x = get_global_id(0); // channel x
		int y = get_global_id(1); // channel y
		int b = get_global_id(2); // batch 
		_FLOAT4 accum = 0;
		_FLOAT4 bot_in[MLO_LRN_LOCALAREA_SZ];
		int c_i = 0, c_o = 0;
		for( ; c_i < MLO_LRN_LOCALAREA_PAD_SZ; c_i++)
		{
			bot_in[c_i] = (_FLOAT4)padding_value;
			accum += bot_in[c_i] * bot_in[c_i];
		}


		for( ; c_i < MLO_LRN_LOCALAREA_SZ; c_i++)
		{
			
			bot_in[c_i] = *(__global _FLOAT4*)&bottom[bot_batch_stride * b + bot_channel_stride *c_i + bot_stride * y + x*MLO_LRN_N_HORIZ_OUT_PIX];
			accum += bot_in[c_i] * bot_in[c_i];

		}

		int top_off = 0;

		for( ; c_i < MLO_LRN_N_CHANNELS + MLO_LRN_LOCALAREA_PAD_SZ; c_i++)
		{
			top_off = b * top_batch_stride + c_o++ * top_channel_stride + top_stride *y + x*MLO_LRN_N_HORIZ_OUT_PIX;
			_FLOAT4 a = ((_FLOAT4)1.f + accum * alphaoverarea);
			_FLOAT4 scale = native_exp((_FLOAT4)-beta * native_log(a));
			_FLOAT4 out_val = bot_in[MLO_LRN_LOCALAREA_PAD_SZ + 1] * scale;
			*(__global _FLOAT4*)&top[top_off] = out_val;

			accum -= bot_in[0] * bot_in[0];
			for( int i = 0; i < MLO_LRN_LOCALAREA_SZ - 1; i++)
			{
				bot_in[i] = bot_in[i+1];
			}

			bot_in[MLO_LRN_LOCALAREA_SZ - 1] = *(__global _FLOAT4*)&bottom[bot_batch_stride * b + bot_channel_stride *c_i + bot_stride * y + x*MLO_LRN_N_HORIZ_OUT_PIX];
			accum += bot_in[MLO_LRN_LOCALAREA_SZ - 1] * bot_in[MLO_LRN_LOCALAREA_SZ - 1];
		}

		for(  ; c_i < MLO_LRN_N_CHANNELS + MLO_LRN_LOCALAREA_PAD_SZ * 2; c_i++)
		{
			top_off = b * top_batch_stride + c_o++ * top_channel_stride + top_stride *y + x*MLO_LRN_N_HORIZ_OUT_PIX;

			_FLOAT4 a = ((_FLOAT4)1.f + accum * alphaoverarea);
			_FLOAT4 scale = native_exp((_FLOAT4)-beta * native_log(a));
			_FLOAT4 out_val = bot_in[MLO_LRN_LOCALAREA_PAD_SZ + 1] * scale;
			*(__global _FLOAT4*)&top[top_off] = out_val;

			accum -= bot_in[0] * bot_in[0];
			for( int i = 0; i < MLO_LRN_LOCALAREA_SZ - 1; i++)
			{
				bot_in[i] = bot_in[i+1];
			}

			bot_in[MLO_LRN_LOCALAREA_SZ - 1] = (_FLOAT4)padding_value;
			accum += bot_in[MLO_LRN_LOCALAREA_SZ - 1] * bot_in[MLO_LRN_LOCALAREA_SZ - 1];
		}

}


#define MLO_LRN_LCL_STRIDE (MLO_LRN_GROUP_SZ0)
__attribute__((reqd_work_group_size(MLO_LRN_GROUP_SZ0,MLO_LRN_GROUP_SZ1,MLO_LRN_GROUP_SZ2)))
__kernel void aDNNLRNWithinChannel4_3x3(
       const __global _FLOAT * bot,
       __global _FLOAT * top,
	   int bot_width,
	   int bot_height,
	   int bot_stride,
	   int bot_channel_stride,
	   int bot_batch_stride,
	   int top_stride,
	   int top_channel_stride,
	   int top_batch_stride,
	   _FLOAT padding_value,
	   _FLOAT alphaoverarea,
	   _FLOAT beta
	   )
{

	__local _FLOAT4 lcl_sum[ MLO_LRN_LCL_STRIDE *(MLO_LRN_GROUP_SZ1 + MLO_LRN_LOCALAREA_PAD_SZ * 2)];
	int x = get_global_id(0); // channel x
	int lcl_x = get_local_id(0);
	int grp_x = get_group_id(0);
	int y = get_global_id(1); // channel y
	int lcl_y = get_local_id(1);
	int grp_y = get_group_id(1);
	int cb= get_global_id(2); // channels * batch 
	int b = (int)((float)cb / MLO_LRN_N_CHANNELS);
	int c = cb - b * MLO_LRN_N_CHANNELS;
	int bot_off= 0;
	_FLOAT accum[4] = {0,0,0,0};
	_FLOAT4 accum4 = 0;
	_FLOAT4 data4;

// Y padding should not affect the DATA being read.
// the DATA is going to be scaled by the final sum.
// we needd only top and borttom part of accum, since the middle has been calculated already. Again it should not be affected by Y padding.

	int x_group_off = grp_x * MLO_LRN_GROUP_SZ0 * MLO_LRN_N_HORIZ_OUT_PIX - MLO_LRN_LOCALAREA_PAD_SZ;

	int x_off = x_group_off + lcl_x;

	int y_group_off = grp_y * MLO_LRN_GROUP_SZ1 * MLO_LRN_N_VERT_OUT_PIX - MLO_LRN_GROUP_SZ1;

	int y_off = y_group_off + lcl_y ;
	while (y_off - y_group_off < MLO_LRN_GROUP_SZ1 + MLO_LRN_LOCALAREA_PAD_SZ * 2)
	{

		bool top_padding = (y_off < 0);
		bool bot_padding = (y_off >= bot_height);
		if ( top_padding )
		{
			lcl_sum[(y_off + MLO_LRN_LOCALAREA_PAD_SZ) *MLO_LRN_LCL_STRIDE + lcl_x] = (_FLOAT4)padding_value;
		}
		if ( bot_padding )
		{

			lcl_sum[(bot_height - y_off + MLO_LRN_LOCALAREA_SZ + MLO_LRN_LOCALAREA_PAD_SZ) *MLO_LRN_LCL_STRIDE + lcl_x] = (_FLOAT4)padding_value;
		}
		if ( top_padding || bot_padding )
		{
			y_off += MLO_LRN_GROUP_SZ1;
			continue;
		}
// top/left with padding
// padding should not exeed 4 - 9x9 local area
		for(int i = 0; i < MLO_LRN_LOCALAREA_PAD_SZ; i++)
		{
			bool left_padding = (x_off < 0);
			x_off = (left_padding) ? 0 : x_off;

			bot_off = b * bot_batch_stride + c*bot_channel_stride + y_off* bot_stride + x_off;

			_FLOAT val = bot[bot_off];
		 
			val = (left_padding) ? padding_value : val;

			for(int j = 0; j < i; j++)
			{
				accum[j] += val * val;
			}
			x_off++;
		}
// assuming  - data is alwways group_size*4 aligned horizontally an dvertically
// read 4 per wk-item
		bot_off = b * bot_batch_stride + c*bot_channel_stride + y_off* bot_stride + x_off;
		data4 = *(__global _FLOAT4 *)&bot[bot_off];
		_FLOAT data[4] = {data4.s0, data4.s1, data4.s2, data4.s3};
	// checlk LOOP HERE
// left padding
		for (int j = 0; j < MLO_LRN_LOCALAREA_SZ - MLO_LRN_LOCALAREA_PAD_SZ; j++)
		{
			accum[0] += data[j] * data[j];
		}
// inside
		for ( int i = 1; i < 3; i++)
		{
			for (int j = i - 1; j < i - 1 + MLO_LRN_LOCALAREA_SZ; j++)
			{
				accum[i] += data[j] * data[j];
			}
		}
// right padding 

		for (int j = 2; j < 2 + MLO_LRN_LOCALAREA_SZ - MLO_LRN_LOCALAREA_PAD_SZ; j++)
		{
			accum[3] += data[j] * data[j];
		}

		x_off += MLO_LRN_N_HORIZ_OUT_PIX;
		for(int i = 0; i < MLO_LRN_LOCALAREA_PAD_SZ; i++)
		{

			x_off += i;
			bool right_padding = x_off >= bot_width;
			x_off = (right_padding) ? 0 : x_off;
			bot_off = b * bot_batch_stride + c*bot_channel_stride + y_off* bot_stride + x_off;

			_FLOAT val = bot[bot_off];
		 
			val = (right_padding ) ? padding_value : val;

			for(int j = MLO_LRN_N_HORIZ_OUT_PIX - (MLO_LRN_LOCALAREA_SZ - MLO_LRN_LOCALAREA_PAD_SZ) + i; j < MLO_LRN_N_HORIZ_OUT_PIX; j++)
			{
				accum[j] += val * val;
			} 

		}

// write into local
		accum4.s0 = accum[0];
		accum4.s1 = accum[1]; 
		accum4.s2 = accum[2];
		accum4.s3 = accum[3];
		lcl_sum[(y_off - y_group_off) * MLO_LRN_LCL_STRIDE + lcl_x] = accum4;
// bottom padding
		y_off += MLO_LRN_GROUP_SZ1;

	}

	barrier(CLK_LOCAL_MEM_FENCE);
// final summ
	for(int j = lcl_y; j < lcl_y + MLO_LRN_LOCALAREA_PAD_SZ; j++)
	{
		accum4 += lcl_sum[j * MLO_LRN_LCL_STRIDE + lcl_x];
	}
	for(int j = lcl_y + MLO_LRN_LOCALAREA_PAD_SZ + 1; j < lcl_y + MLO_LRN_LOCALAREA_SZ; j++)
	{
		accum4 += lcl_sum[j * MLO_LRN_LCL_STRIDE + lcl_x];
	}
// done summation
	mem_fence(CLK_LOCAL_MEM_FENCE);
// calculate scale
	_FLOAT4 a = ((_FLOAT4)1.f + accum4 * alphaoverarea);
	_FLOAT4 scale = native_exp((_FLOAT4)-beta * native_log(a));

	int top_off = b * top_batch_stride + c*top_channel_stride + y* bot_stride + x;
	*(__global _FLOAT4 *)&top[top_off] = data4 * scale; 

}

#endif

