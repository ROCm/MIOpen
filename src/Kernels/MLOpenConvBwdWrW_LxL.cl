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

#define MLO_N_OUT_HORIZ_READS (MLO_ALIGNED_OUT_SCAN_LN)
#define MLO_OUT_HORIX_PIX_SZ (MLO_N_OUT_HORIZ_READS * MLO_READ_UNIT)
#define MLO_OUT_BLK_GRP_PIX_SZ (MLO_OUT_HORIX_PIX_SZ * MLO_N_ALIGNED_OUT_SCAN_BLK)
#define MLO_OUT_BLK_GRP_WK_SZ (MLO_OUT_BLK_GRP_PIX_SZ / MLO_READ_UNIT)
#define MLO_OUT_LCL_SZ (MLO_OUT_BLK_GRP_PIX_SZ)
// LDS OUT SIZE
#define MLO_TOTAL_OUT_LCL_SZ (MLO_N_LCL_OUT_MAPS*MLO_OUT_LCL_SZ)

// input size depends on output scan length and
// number of output scans 
// this number is constrained by amount or LDS and size of register file.
// TO DO:: CHECK PADDING!!!
#define MLO_IN_LCL_HEIGHT ((MLO_N_ALIGNED_OUT_SCAN_BLK-1)*MLO_FILTER_STRIDE1 + MLO_FILTER_SIZE1)
#define MLO_N_IN_HORIZ_READS ((((MLO_OUT_HORIX_PIX_SZ-1)*MLO_FILTER_STRIDE0 + MLO_FILTER_SIZE0) + MLO_READ_UNIT - 1) / MLO_READ_UNIT)
#define MLO_IN_LCL_WIDTH (MLO_N_IN_HORIZ_READS * MLO_READ_UNIT)
#define MLO_IN_BLK_GRP_PIX_SZ (MLO_IN_LCL_WIDTH * MLO_IN_LCL_HEIGHT)
#define MLO_IN_BLK_GRP_WK_SZ (MLO_IN_BLK_GRP_PIX_SZ/MLO_READ_UNIT)
#define MLO_IN_LCL_SZ (MLO_IN_BLK_GRP_PIX_SZ)
// LDS IN SIZE
#define MLO_TOTAL_IN_LCL_SZ (MLO_N_LCL_IN_MAPS*MLO_IN_LCL_SZ)
#define MLO_IN_VERT_READS (MLO_GRP_SZ/MLO_N_IN_HORIZ_READS)

#define MLO_WEI_WKITEM 2
#define MLO_WEI_BLK_SZ0 (MLO_FILTER_SIZE0/MLO_WEI_WKITEM)
#define MLO_WEI_BLK_SZ (MLO_FILTER_SIZE1*MLO_WEI_BLK_SZ0)
#define MLO_N_WEI_BLK (MLO_GRP_SZ/ MLO_WEI_BLK_SZ)
#define MLO_OUT_WEI_SCAN_BLK ((MLO_OUT_HORIX_PIX_SZ + MLO_N_WEI_BLK - 1) / MLO_N_WEI_BLK)


#if MLO_OUT_SCAN_NOT_DIVBY4
__constant _FLOAT cnst_out_r_mask[MLO_READ_UNIT] =
{
#if MLO_OUT_N_PIXS_OFF == 1
	1, 1, 1, 0
#elif MLO_OUT_N_PIXS_OFF == 2
	1, 1, 0, 0
#else
	1, 0, 0, 0
#endif
};
#endif


#define MLO_HW_WAVE_ID_SETTING 0

#if MLO_HW_WAVE_ID_SETTING
extern __attribute__((const)) uint __hsail_get_dynwave_id(void);
inline int getWaveId()
{
	int wave_id = 0;

	wave_id = __hsail_get_dynwave_id();
	wave_id = wave_id & MLO_N_PHYS_WAVES_MASK;
	return(wave_id);
}
#else
inline int getWaveId()
{
	int wave_id = 0;

	wave_id = (get_local_id(0) >> MLO_LG2_PHYS_WAVE_SZ);

	return(wave_id);
}
#endif


inline int iDiv(int v, int d)
{
	int r = (int)((float)v / d + 0.00001f);
	return(r);
}

inline int iMod(int v, int u, int d)
{
	int r = v - mul24((int)u,(int)d);
	return(r);
}

inline void ReduceKernel(__local _FLOAT * lcl_blob, _FLOAT *weights_accum, int lcl_id, int scan_lcl, int sum_stride, int unit_len, bool debug)
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


/*********************************************************************************************************
// wrw algorithm
// data layout.
// registers:
// per wk_item:
// MLO_READ_UNIT x MLO_N_LCL_OUT_MAPS of dy
// group:
// MLO_N_OUT_BLK_GRP blocks of MLO_N_OUT_BLK_GRP_PIX_SZ of dy
// 
// LDS:
// data buffer
// MLO_N_LCL_IN_MAPS blocks of MLO_IN_LCL_SZ of x

// dw local buffer keeps 1 set of dweghts calculated at each iteration
	MLO_TEMP_WEI_BUFFER_SZ

// 
// inputs:
// read into LDS
// MLO_N_IN_BLK_GRP groups
// output
// fetch 
// MLO_N_OUT_BLK_GRP groups

// alg
// for each batch
//		until end of output map (MLO_N_OUT_BLK)
//			load input map block in LDS
//			load output maps into registers
//          for k in filter size1
//				for l in filter size 0
//					at for each out/in pixel pair op,ip in wk_item
//                  calculate the contribution to a dweights by formula ip.x (input pixel.x) = (op.x * stride.x - pad .x + l) % (filter size 0), ip.y = (op.y * stride.y - pad.y + k) % filter size1,
//					wei[k,l] = in[ip.y, ip.x]*op[op.y, op.x]
//                  lcl_wei[MLO_OUT_BLK_GRP_PIX_SZ * out.grp + [op.y, op.x]] = we[k,l]
//                  accumulate the weights in the wk_item with wei_accum +=  lcl_wei[accum_id], accum_id = MLO_OUT_BLK_GRP*MLO_N_LCL_OUT_MAPS*map_id + k*(filter size 0) + l
//	write out accululate weights

// group layout
// 0 - input channel index
// 1 - output channel index
// 2 - 1

// loop over batches
**********************************************************************************************************/

__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2)))
__kernel void MLOpenCvBwdWrW(
	const __global _FLOAT * top_df,
	const __global _FLOAT * bot,
	__global _FLOAT * weights_df,
#if MLO_CONV_BIAS
	__global _FLOAT * bias_df,
#endif
	_FLOAT padding_val
)
{
// input/output tiles

	__local _FLOAT lcl[(MLO_TOTAL_IN_LCL_SZ + MLO_TOTAL_OUT_LCL_SZ + 2)];
	__local _FLOAT * lcl_bot = lcl;
	__local _FLOAT * lcl_top = lcl + MLO_TOTAL_IN_LCL_SZ;




	// guarnteeing an uniformity over a wave
	int wave_id = getWaveId();



	int c_idx_base = get_group_id(1); // input map index base

// register memory
	int o_idx_base = get_group_id(2); // output map index base



	int c_idx = c_idx_base * MLO_N_LCL_IN_MAPS; // input map index

	int o_idx = o_idx_base * (MLO_N_OUT_BLK_GRP * MLO_N_LCL_OUT_MAPS); // output map index

	int lcl_id = get_local_id(0);



// output global read
//	int o_blk_idx = iDiv(lcl_id, MLO_OUT_LCL_SZ);
//	int o_pix_X4 = iMod(lcl_id, o_blk_idx, MLO_OUT_LCL_SZ);
	int o_scan = iDiv(lcl_id, MLO_N_OUT_HORIZ_READS);
	int o_pix4 = iMod(lcl_id, o_scan, MLO_N_OUT_HORIZ_READS);

// input global read
	int i_blk_idx = 0; //iDiv(lcl_id, MLO_IN_LCL_SZ);
	//int c_pix_X4 = iMod(lcl_id, i_blk_idx, MLO_IN_LCL_SZ);
	int c_scan0 = iDiv(lcl_id, MLO_N_IN_HORIZ_READS);
	int c_pix4 = iMod(lcl_id,c_scan0, MLO_N_IN_HORIZ_READS);

	int w_blk_idx = iDiv(lcl_id, MLO_WEI_BLK_SZ);
	int w_idx = iMod(lcl_id, w_blk_idx, MLO_WEI_BLK_SZ);
	int w_y = iDiv(w_idx, MLO_WEI_BLK_SZ0);
	int w_x0 = iMod(w_idx, w_y, MLO_WEI_BLK_SZ0);

	__private _FLOAT pvt_accum[(MLO_N_OUT_BLK_GRP * MLO_N_LCL_OUT_MAPS * MLO_WEI_WKITEM)];


	int gbl_in_off = c_idx * MLO_IN_CHANNEL_STRIDE;
	int gbl_out_off = o_idx * MLO_OUT_CHANNEL_STRIDE;

	for (int i = 0; i < (MLO_N_OUT_BLK_GRP * MLO_N_LCL_OUT_MAPS * MLO_WEI_WKITEM); ++i)
	{
		pvt_accum[i] = 0;
	}


// zero out LDS
	for (int i = 0; i <  (MLO_TOTAL_IN_LCL_SZ + MLO_TOTAL_OUT_LCL_SZ + 2) / 2; i += MLO_GRP_SZ)
	{
		*(_FLOAT2*)&lcl[2 * i] = 0;
	}

#if MLO_OUT_SCAN_NOT_DIVBY4
	 _FLOAT out_r_mask[MLO_READ_UNIT] = {1, 1, 1, 1};

	if(o_pix4 == (MLO_N_OUT_HORIZ_READS-1))
	{
		for(int i = 0; i < MLO_READ_UNIT; ++i)
		{
			out_r_mask[i] = cnst_out_r_mask[i];
		}
	}
#endif
	barrier(CLK_LOCAL_MEM_FENCE);
// zero out pvt accum
	// over all batches

#if 0
	if ( lcl_id == 0 && c_idx ==0 && o_idx == 0)
	{
		printf("K:s:%d %d\n",
			MLO_TOTAL_IN_LCL_SZ*4,
			MLO_TOTAL_OUT_LCL_SZ
		);
	}
#endif
#if 1
	for (int b = 0;
		b < MLO_N_BATCH_LOOPS;
		++b,
		gbl_in_off += MLO_N_LCL_BATCHS * MLO_IN_BATCH_STRIDE,
		gbl_out_off += MLO_N_LCL_BATCHS * MLO_OUT_BATCH_STRIDE
		)
	{
		int in_y = 0;
		int out_y = 0;
// prefetch MLO_FILTER_STRIDE1 - MLO_FILTER_PAD1
// LDS UPDATE !!!!
// input scan
		__private _FLOAT in_rd_data[MLO_READ_UNIT];

		int gbl_in_scan_off = 0;
		int gbl_out_scan_off = 0;

		
		
		for(int c_scan = c_scan0; c_scan < MLO_FILTER_SIZE1 - MLO_FILTER_PAD1 - 1; c_scan += MLO_IN_VERT_READS)
		{
// TODO :: multiply on right mask if needed
			*(MLO_READ_TYPE*)in_rd_data = *(MLO_READ_TYPE*)&bot[gbl_in_scan_off + c_scan * MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT];
// TO DO:: CHECK HEIGHT
// TO DO:: CHECK PADDING
			*(MLO_READ_TYPE*)&lcl_bot[(c_scan + MLO_FILTER_PAD1)*MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 + c_pix4*MLO_READ_UNIT] = *(MLO_READ_TYPE*)in_rd_data;
		}

		in_y +=  MLO_FILTER_SIZE1 - MLO_FILTER_PAD1 - 1;

// TO DO: HANDLE PADDING
// over all out blocks
		for (int ob = 0; ob < MLO_N_OUT_BLK; ++ob, in_y += (MLO_IN_LCL_HEIGHT - MLO_FILTER_SIZE1 + 1), out_y += MLO_N_ALIGNED_OUT_SCAN_BLK)
		{

			barrier(CLK_LOCAL_MEM_FENCE);



			gbl_in_scan_off = gbl_in_off + mul24(in_y, MLO_IN_STRIDE);


// fetch input: (MLO_IN_LCL_HEIGHT - MLO_FILTER_SIZE1 + 1)
               // TODO:: HANDLE multiple INPUTS 
			   // overshot i shadled by out put
			
			for (int c_scan = c_scan0;i_blk_idx < MLO_N_LCL_IN_MAPS && in_y + c_scan < MLO_IN_HEIGHT && c_scan <  (MLO_IN_LCL_HEIGHT - MLO_FILTER_SIZE1 + 1); c_scan += MLO_IN_VERT_READS)
			{

					*(MLO_READ_TYPE*)in_rd_data = *(MLO_READ_TYPE*)&bot[gbl_in_scan_off + c_scan * MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT];
					*(MLO_READ_TYPE*)&lcl_bot[(c_scan + MLO_FILTER_SIZE1 - 1)*MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 + c_pix4*MLO_READ_UNIT] = *(MLO_READ_TYPE*)in_rd_data;
							
			}


			// ouput scan
			__private _FLOAT out_rd_data[MLO_N_LCL_OUT_MAPS * MLO_READ_UNIT];



			gbl_out_scan_off = gbl_out_off + mul24(out_y, MLO_OUT_STRIDE);

	
			
// over all outputs groups
			
			int gbl_out_scan_off1 = gbl_out_scan_off;
			for(int og = 0; og < MLO_N_OUT_BLK_GRP; ++og, gbl_out_scan_off1 += MLO_OUT_LCL_SZ*MLO_OUT_CHANNEL_STRIDE )
			{

// fetch output

				for(int o = 0; o < MLO_N_LCL_OUT_MAPS
						&&  ( o_idx + o < MLO_N_OUTPUTS && out_y + o_scan < MLO_OUT_HEIGHT && o_scan < MLO_N_ALIGNED_OUT_SCAN_BLK)
						; ++o)
				{
						*(MLO_READ_TYPE*)&out_rd_data[o*MLO_READ_UNIT]
							= *(MLO_READ_TYPE*)&top_df[gbl_out_scan_off + o*MLO_OUT_CHANNEL_STRIDE + o_scan * MLO_OUT_STRIDE + o_pix4*MLO_READ_UNIT];
#if MLO_OUT_SCAN_NOT_DIVBY4
	 					*(MLO_READ_TYPE*)&out_rd_data[o*MLO_READ_UNIT] *= *(MLO_READ_TYPE*)out_r_mask;
#endif
						*(MLO_READ_TYPE*)&lcl_top[o * MLO_OUT_LCL_SZ + o_scan * MLO_OUT_HORIX_PIX_SZ + o_pix4*MLO_READ_UNIT] = *(MLO_READ_TYPE*)&out_rd_data[o*MLO_READ_UNIT];
				}
				
// process	
// algorithm

#if 1
				for(int j = 0; j < MLO_N_ALIGNED_OUT_SCAN_BLK; ++j)		
				{	
					_FLOAT i_vals[MLO_WEI_WKITEM];
					for(int w = 0; w < MLO_WEI_WKITEM - 1; ++w)
					{
							int w_x = w_x0 + w*MLO_FILTER_STRIDE0;
							int i_off = (j*MLO_FILTER_STRIDE1 + w_y) * MLO_IN_LCL_WIDTH + (w_blk_idx*MLO_OUT_WEI_SCAN_BLK + 0) * MLO_FILTER_STRIDE0 + w_x;
							i_vals[w] = lcl_bot[i_off];
													
					}

					for(int i = 0; i < MLO_OUT_WEI_SCAN_BLK; ++i)
					{

//						for(int w = 0; w < MLO_WEI_WKITEM; ++w)
						{
							int w_x = w_x0 + (MLO_WEI_WKITEM-1)* MLO_FILTER_STRIDE0;
							int i_off = (j*MLO_FILTER_STRIDE1 + w_y) * MLO_IN_LCL_WIDTH + (w_blk_idx*MLO_OUT_WEI_SCAN_BLK + i) * MLO_FILTER_STRIDE0 + w_x;
							i_vals[(MLO_WEI_WKITEM-1)] = lcl_bot[i_off];
													
						}


						for(int o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)
						{
							_FLOAT o_val 
								= lcl_top[o*MLO_OUT_LCL_SZ + j * MLO_N_OUT_HORIZ_READS + (w_blk_idx*MLO_OUT_WEI_SCAN_BLK + i)];

							for(int w = 0; w < MLO_WEI_WKITEM; ++w)
							{

								pvt_accum[(og * MLO_N_LCL_OUT_MAPS + o) * MLO_WEI_WKITEM + w] += i_vals[w] * o_val;
							}
						}

						for(int w = 0; w < MLO_WEI_WKITEM - 1; ++w)
						{
							i_vals[w] = i_vals[w + 1];
													
						}

					}
				}
#else
						for(int o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)
						{
							for(int w = 0; w < MLO_WEI_WKITEM; ++w)
							{
								pvt_accum[(og * MLO_N_LCL_OUT_MAPS + o) * MLO_WEI_WKITEM + w] = lcl_top[(og * MLO_N_LCL_OUT_MAPS + o)] * lcl_bot[w];
							}
						}
#endif			
			

			} // for(; og < (MLO_N_OUT_BLK_GRP; ++og )

			barrier(CLK_LOCAL_MEM_FENCE);

// move data up
			for(int c_scan = c_scan0; c_scan < MLO_FILTER_SIZE1 -1; c_scan += MLO_IN_VERT_READS)
			{
					*(MLO_READ_TYPE*)in_rd_data = *(MLO_READ_TYPE*)&lcl_bot[(c_scan +  (MLO_IN_LCL_HEIGHT - MLO_FILTER_SIZE1 + 1))*(MLO_IN_LCL_WIDTH) + MLO_FILTER_PAD0 + c_pix4*MLO_READ_UNIT];
					*(MLO_READ_TYPE*)&lcl_bot[c_scan*(MLO_IN_LCL_WIDTH) + MLO_FILTER_PAD0 + c_pix4*MLO_READ_UNIT] = *(MLO_READ_TYPE*)in_rd_data;
	
			}


		} // for (int ob = 0; ob < MLO_N_OUT_BLK; ++ob, in_y += (MLO_IN_LCL_HEIGHT - MLO_FILTER_SIZE1 + 1), out_y += MLO_N_ALIGNED_OUT_SCAN_BLK)
	} // for (int b = 0;


#endif

#if 1
// save in lcl
	for(int og = 0; og < MLO_N_OUT_BLK_GRP; ++og)
	{
		for(int o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)
		{
			for(int w = 0; w < MLO_WEI_WKITEM; ++w)
			{
				int wei_lcl_off = ((og * MLO_N_LCL_OUT_MAPS  + o) *  MLO_N_WEI_BLK  + w_blk_idx) * (MLO_FILTER_SIZE1*(MLO_FILTER_SIZE0/MLO_WEI_WKITEM)) + w_y * (MLO_FILTER_SIZE0/MLO_WEI_WKITEM) + w;
				lcl_bot[wei_lcl_off] = pvt_accum[(og * MLO_N_LCL_OUT_MAPS + o) * MLO_WEI_WKITEM + w];

			}
		}
	
	}
#endif


// send it out
	int wei_df_off = mul24(o_idx, (int)MLO_WEI_BATCH_STRIDE)
		// this input channel
						+ mul24(c_idx, (int)MLO_WEI_CHANNEL_STRIDE);
// write out from 0th wk_item
	if (lcl_id < (MLO_N_OUT_BLK_GRP * MLO_N_LCL_OUT_MAPS *MLO_FILTER_SIZE1*MLO_FILTER_SIZE0))
	{

#if 1
		int o = iDiv(lcl_id, (MLO_FILTER_SIZE1*MLO_FILTER_SIZE0));
		int wei_i = iMod(lcl_id, o, (MLO_FILTER_SIZE1*MLO_FILTER_SIZE0));

		weights_df[wei_df_off + o * MLO_WEI_BATCH_STRIDE + wei_i] = lcl_bot[lcl_id];
#else
		_FLOAT t_accum = 0;

		for(int og = 0; og < MLO_N_OUT_BLK_GRP; ++og)
		{
			for(int o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)
			{
				for(int w = 0; w < MLO_WEI_WKITEM; ++w)
				{
					t_accum += pvt_accum[(og * MLO_N_LCL_OUT_MAPS + o) * MLO_WEI_WKITEM + w];
				}
			}
		}

		weights_df[lcl_id] = t_accum;
#endif


	}



}


