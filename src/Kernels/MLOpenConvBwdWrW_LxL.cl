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

// number of filter taps in the processing wk_item
//#define MLO_WEI_WKITEM 5


#define MLO_N_OUT_HORIZ_READS (MLO_ALIGNED_OUT_SCAN_LN)
#define MLO_OUT_HORIZ_PIX_SZ (MLO_N_OUT_HORIZ_READS * MLO_READ_UNIT)

// n of of filter blks
#define MLO_WEI_BLK_SZ0 ((MLO_FILTER_SIZE0 + MLO_WEI_WKITEM -1)/MLO_WEI_WKITEM)
#define MLO_WEI_BLK_SZ (MLO_FILTER_SIZE1*MLO_WEI_BLK_SZ0)
// n of filter tiles in the group grid
#define MLO_N_WEI_BLK (MLO_GRP_SZ/ MLO_WEI_BLK_SZ)
// n of steps per scan line to be made in the inner loop
// extended scan to deal with overshot in the inner loop
#define MLO_OUT_WEI_EXT_SCAN_BLK ((MLO_OUT_HORIZ_PIX_SZ + MLO_N_WEI_BLK - 1) / MLO_N_WEI_BLK)

#define MLO_OUT_WEI_SCAN_BLK (MLO_OUT_WEI_EXT_SCAN_BLK)

// max loop over virtual blocks for small images
#define MLO_MAX_WEI_BLK_LOOP_TMP ((MLO_OUT_HORIZ_PIX_SZ + MLO_OUT_WEI_SCAN_BLK - 1)/MLO_OUT_WEI_SCAN_BLK)
#if MLO_MAX_WEI_BLK_LOOP_TMP < MLO_N_WEI_BLK
#define MLO_MAX_WEI_BLK_LOOP MLO_MAX_WEI_BLK_LOOP_TMP
#else
#define MLO_MAX_WEI_BLK_LOOP MLO_N_WEI_BLK
#endif

#define MLO_WEI_BLKS_SZ (MLO_MAX_WEI_BLK_LOOP * MLO_WEI_BLK_SZ *MLO_WEI_WKITEM)
#define MLO_WEI_BLKS_LCL_SZ (MLO_WEI_BLKS_SZ * MLO_N_LCL_OUT_MAPS)

#define MLO_OUT_BLK_GRP_PIX_SZ (MLO_OUT_HORIZ_PIX_SZ * MLO_N_ALIGNED_OUT_SCAN_BLK)
#define MLO_OUT_BLK_GRP_WK_SZ (MLO_OUT_BLK_GRP_PIX_SZ / MLO_READ_UNIT)

#define MLO_OUT_HORIZ_PIX_EXT_SZ (MLO_OUT_WEI_EXT_SCAN_BLK * MLO_MAX_WEI_BLK_LOOP)
#define MLO_OUT_BLK_GRP_EXT_PIX_SZ (MLO_OUT_HORIZ_PIX_EXT_SZ * MLO_N_ALIGNED_OUT_SCAN_BLK)
#define MLO_OUT_LCL_SZ (MLO_OUT_BLK_GRP_EXT_PIX_SZ)
// LDS OUT SIZE
#define MLO_TOTAL_OUT_LCL_SZ (MLO_N_LCL_BATCHS*MLO_N_LCL_OUT_MAPS*MLO_OUT_LCL_SZ)
#if ((MLO_OUT_HEIGHT/MLO_N_ALIGNED_OUT_SCAN_BLK)*MLO_N_ALIGNED_OUT_SCAN_BLK == MLO_OUT_HEIGHT)
#define MLO_BLK_ALIGNED 1
#else
#define MLO_BLK_ALIGNED 0
#endif


// input size depends on output scan length and
// number of output scans 
// this number is constrained by amount or LDS and size of register file.
// TO DO:: CHECK PADDING!!!
#define MLO_IN_LCL_HEIGHT ((MLO_N_ALIGNED_OUT_SCAN_BLK-1)*MLO_FILTER_STRIDE1 + MLO_FILTER_SIZE1)
#define MLO_N_IN_HORIZ_READS ((((MLO_OUT_HORIZ_PIX_SZ-1)*MLO_FILTER_STRIDE0 + MLO_FILTER_SIZE0) + MLO_READ_UNIT - 1) / MLO_READ_UNIT)
#define MLO_IN_LCL_WIDTH (MLO_N_IN_HORIZ_READS * MLO_READ_UNIT)
#define MLO_IN_BLK_GRP_PIX_SZ (MLO_IN_LCL_WIDTH * MLO_IN_LCL_HEIGHT)
#define MLO_IN_BLK_GRP_WK_SZ (MLO_IN_BLK_GRP_PIX_SZ/MLO_READ_UNIT)
#define MLO_IN_LCL_SZ (MLO_IN_BLK_GRP_PIX_SZ)
// LDS IN SIZE
#define MLO_TOTAL_IN_LCL_SZ (MLO_N_LCL_BATCHS*MLO_N_LCL_IN_MAPS*MLO_IN_LCL_SZ)
#define MLO_IN_VERT_READS (MLO_GRP_SZ/MLO_N_IN_HORIZ_READS)

#if (MLO_TOTAL_OUT_LCL_SZ + MLO_TOTAL_IN_LCL_SZ) > (MLO_WEI_BLKS_LCL_SZ)
#define MLO_LCL_SZ (MLO_TOTAL_OUT_LCL_SZ + MLO_TOTAL_IN_LCL_SZ)
#else
#define MLO_LCL_SZ (MLO_WEI_BLKS_LCL_SZ)
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
	int r = v - mul24((int)u, (int)d);
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
//			load output maps in LDS
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

	__local _FLOAT lcl[(MLO_LCL_SZ)];
	__local _FLOAT * lcl_bot = lcl;
	__local _FLOAT * lcl_top = lcl + MLO_TOTAL_IN_LCL_SZ;




	// guarnteeing an uniformity over a wave
	int wave_id = getWaveId();



	int c_idx_base = get_group_id(1); // input map index base


	int o_idx_base = iDiv(get_group_id(2), (MLO_BATCH_SZ / (MLO_N_BATCH_LOOPS*MLO_N_LCL_BATCHS))); // output map index base
	int ib_base = iMod(get_group_id(2), o_idx_base, (MLO_BATCH_SZ / (MLO_N_BATCH_LOOPS*MLO_N_LCL_BATCHS)));

	int ib = ib_base*MLO_N_LCL_BATCHS;

	int c_idx = c_idx_base * MLO_N_LCL_IN_MAPS; // input map index

	int o_idx = o_idx_base * (MLO_N_OUT_BLK_GRP * MLO_N_LCL_OUT_MAPS); // output map index

	int lcl_id = get_local_id(0);




	int w_blk_idx = iDiv(lcl_id, MLO_WEI_BLK_SZ);
	int w_idx = iMod(lcl_id, w_blk_idx, MLO_WEI_BLK_SZ);
	int w_y = iDiv(w_idx, MLO_WEI_BLK_SZ0);
	int w_x0 = iMod(w_idx, w_y, MLO_WEI_BLK_SZ0);



	__private _FLOAT pvt_accum[(MLO_N_OUT_BLK_GRP * MLO_N_LCL_OUT_MAPS * MLO_WEI_WKITEM)];


	int gbl_in_off = c_idx * MLO_IN_CHANNEL_STRIDE + ib * MLO_IN_BATCH_STRIDE;
	int gbl_out_off = o_idx * MLO_OUT_CHANNEL_STRIDE + ib * MLO_OUT_BATCH_STRIDE;

	for (int i = 0; i < (MLO_N_OUT_BLK_GRP * MLO_N_LCL_OUT_MAPS * MLO_WEI_WKITEM); ++i)
	{
		pvt_accum[i] = 0;
	}


	// zero out LDS
	for (int i = lcl_id; i < (MLO_LCL_SZ); i += MLO_GRP_SZ)
	{
		lcl[i] = 0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	// zero out pvt accum
	// over all batches

#if 0
	if (lcl_id == 0)
	{
		printf("K:i: %d %d %d %d %d\n",
			ib,
			gbl_in_off,
			gbl_out_off,
			MLO_N_BATCH_LOOPS,
			MLO_N_LCL_BATCHS
		);
	}
#endif

#if 1

	for (int b = 0;
		b < MLO_N_BATCH_LOOPS;
		++b,
		gbl_in_off += MLO_N_LCL_BATCHS*MLO_IN_BATCH_STRIDE,
		gbl_out_off += MLO_N_LCL_BATCHS*MLO_OUT_BATCH_STRIDE
		)
	{
		int in_y = 0;
		int out_y = 0;
		// prefetch MLO_FILTER_STRIDE1 - MLO_FILTER_PAD1
		// LDS UPDATE !!!!
		// input scan
		__private _FLOAT in_rd_data[MLO_READ_UNIT];

		int gbl_in_scan_off = gbl_in_off;
		int gbl_out_scan_off = gbl_out_off;
		for (int lb = 0; lb < MLO_N_LCL_BATCHS; ++lb)
		{
			int c_scan = 0;

			for (int p4 = lcl_id;  p4 < MLO_N_IN_HORIZ_READS * (MLO_FILTER_SIZE1 - MLO_FILTER_STRIDE1) && in_y + c_scan < MLO_IN_HEIGHT;
				//				c_scan = iDiv(p4, MLO_N_IN_HORIZ_READS),
				p4 += MLO_GRP_SZ)
			{

				c_scan = iDiv(p4, MLO_N_IN_HORIZ_READS);
				int c_pix4 = iMod(p4, c_scan, MLO_N_IN_HORIZ_READS);
				// TODO :: multiply on right mask if needed
				*(MLO_READ_TYPE*)in_rd_data = *(MLO_READ_TYPE*)&bot[lb*MLO_IN_BATCH_STRIDE + gbl_in_scan_off + c_scan * MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT];
				// TO DO:: CHECK HEIGHT
				// TO DO:: CHECK PADDING
				//			*(MLO_READ_TYPE*)&lcl_bot[(c_scan + MLO_FILTER_PAD1)*MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 + c_pix4*MLO_READ_UNIT] = *(MLO_READ_TYPE*)in_rd_data;
				for (int i = 0; i < MLO_READ_UNIT; ++i)
				{
					lcl_bot[lb*MLO_IN_LCL_SZ + (c_scan + MLO_FILTER_PAD1)*MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 + c_pix4*MLO_READ_UNIT + i] = in_rd_data[i];
				}

#if 0
				if (c_idx == 0 && o_idx == 0 && (c_pix4 == 0 || c_pix4 == 1))
				{
					printf("K:i: %d %d %d %d %d %d %d     %f %f %f %f\n",
						lcl_id,
						lb,
						p4,
						c_scan,
						in_y + c_scan,
						lb*MLO_IN_BATCH_STRIDE + gbl_in_scan_off + c_scan * MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT,
						lb*MLO_IN_LCL_SZ + (c_scan + MLO_FILTER_PAD1)*MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 + c_pix4*MLO_READ_UNIT,
						in_rd_data[0],
						in_rd_data[1],
						in_rd_data[2],
						in_rd_data[3]
					);
				}
#endif

			}
		}

		in_y += MLO_FILTER_SIZE1 - MLO_FILTER_STRIDE1;

		// TO DO: HANDLE PADDING
		// over all out blocks
		for (int ob = 0; ob < MLO_N_OUT_BLK; ++ob, in_y += (MLO_IN_LCL_HEIGHT - MLO_FILTER_SIZE1 + MLO_FILTER_STRIDE1), out_y += MLO_N_ALIGNED_OUT_SCAN_BLK)
		{
#if MLO_BLK_ALIGNED == 0
			for (int i = lcl_id; ob == MLO_N_OUT_BLK - 1 && i < MLO_TOTAL_OUT_LCL_SZ; i += MLO_GRP_SZ)
			{
				lcl_top[i] = 0;
			}
#endif


			barrier(CLK_LOCAL_MEM_FENCE);


			// fetch input: (MLO_IN_LCL_HEIGHT - MLO_FILTER_SIZE1 + 1)
			// TODO:: HANDLE multiple INPUTS 
			// overshot i shadled by out put			
			gbl_in_scan_off = gbl_in_off + mul24(in_y, MLO_IN_STRIDE);
			for (int lb = 0; lb < MLO_N_LCL_BATCHS; ++lb)
			{
				int c_scan = 0;
				for (int p4 = lcl_id; p4 < MLO_N_IN_HORIZ_READS * (MLO_IN_LCL_HEIGHT - MLO_FILTER_SIZE1 + MLO_FILTER_STRIDE1) && in_y + c_scan < MLO_IN_HEIGHT;
					p4 += MLO_GRP_SZ)
				{
					c_scan = iDiv(p4, MLO_N_IN_HORIZ_READS);
					int c_pix4 = iMod(p4, c_scan, MLO_N_IN_HORIZ_READS);

					*(MLO_READ_TYPE*)in_rd_data = *(MLO_READ_TYPE*)&bot[lb*MLO_IN_BATCH_STRIDE + gbl_in_scan_off + c_scan * MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT];
					//					*(MLO_READ_TYPE*)&lcl_bot[(c_scan + MLO_FILTER_SIZE1 - 1)*MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 + c_pix4*MLO_READ_UNIT] = *(MLO_READ_TYPE*)in_rd_data;
					for (int i = 0; i < MLO_READ_UNIT; ++i)
					{
						lcl_bot[lb*MLO_IN_LCL_SZ + (c_scan + MLO_FILTER_SIZE1 - MLO_FILTER_STRIDE1)*MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 + c_pix4*MLO_READ_UNIT + i] = in_rd_data[i];
					}

#if 0
					if (c_idx == 1 && o_idx == 0 && in_y + c_scan == 0)
					{
						printf("K:i: %d %d %f %f\n",
							lcl_id,
							gbl_in_scan_off + c_scan * MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT,
							in_rd_data[0],
							in_rd_data[2]
						);
					}
#endif

				}

			}


			gbl_out_scan_off = gbl_out_off + mul24(out_y, MLO_OUT_STRIDE);

			// over all outputs groups
			int gbl_out_scan_off1 = gbl_out_scan_off;
			for (int og = 0; og < MLO_N_OUT_BLK_GRP
				; ++og, gbl_out_scan_off1 += MLO_N_LCL_OUT_MAPS*MLO_OUT_CHANNEL_STRIDE)
			{
				//				barrier(CLK_LOCAL_MEM_FENCE);

				// fetch output
				// ouput scan
				__private _FLOAT out_rd_data[MLO_READ_UNIT];

				for (int ob = 0; ob < MLO_N_LCL_BATCHS; ++ob)
				{
					for (int oo_p4 = lcl_id; oo_p4 < (MLO_N_LCL_OUT_MAPS*MLO_N_ALIGNED_OUT_SCAN_BLK*MLO_N_OUT_HORIZ_READS);
						oo_p4 += MLO_GRP_SZ)
					{
						int o = iDiv(oo_p4, (MLO_N_ALIGNED_OUT_SCAN_BLK*MLO_N_OUT_HORIZ_READS));
						int o_pX4 = iMod(oo_p4, o, (MLO_N_ALIGNED_OUT_SCAN_BLK*MLO_N_OUT_HORIZ_READS));
						int o_scan = iDiv(o_pX4, MLO_N_OUT_HORIZ_READS);
						int o_pix4 = iMod(o_pX4, o_scan, MLO_N_OUT_HORIZ_READS);
						if (out_y + o_scan < MLO_OUT_HEIGHT
							&& o_idx + og *MLO_N_LCL_OUT_MAPS + o < MLO_N_OUTPUTS
							)
						{
							*(MLO_READ_TYPE*)out_rd_data
								= *(MLO_READ_TYPE*)&top_df[ob*MLO_OUT_BATCH_STRIDE + gbl_out_scan_off1 + o*MLO_OUT_CHANNEL_STRIDE + o_scan * MLO_OUT_STRIDE + o_pix4*MLO_READ_UNIT];
#if MLO_OUT_SCAN_NOT_DIVBY4
							if (o_pix4 == (MLO_N_OUT_HORIZ_READS - 1))
							{
								for (int i = MLO_READ_UNIT - 1; i >= MLO_READ_UNIT - MLO_OUT_N_PIXS_OFF; --i)
								{
									out_rd_data[i] = 0;
								}
							}
#endif
							// write with MLO_OUT_HORIZ_PIX_EXT_SZ stride
							//						*(MLO_READ_TYPE*)&lcl_top[o * MLO_OUT_LCL_SZ + o_scan * MLO_OUT_HORIZ_PIX_EXT_SZ + o_pix4*MLO_READ_UNIT] = *(MLO_READ_TYPE*)out_rd_data;
							//
							lcl_top[(ob*MLO_N_LCL_OUT_MAPS + o) * MLO_OUT_LCL_SZ + o_scan * MLO_OUT_HORIZ_PIX_EXT_SZ + o_pix4*MLO_READ_UNIT + 0] = out_rd_data[0];
							lcl_top[(ob*MLO_N_LCL_OUT_MAPS + o) * MLO_OUT_LCL_SZ + o_scan * MLO_OUT_HORIZ_PIX_EXT_SZ + o_pix4*MLO_READ_UNIT + 1] = out_rd_data[1];
							lcl_top[(ob*MLO_N_LCL_OUT_MAPS + o) * MLO_OUT_LCL_SZ + o_scan * MLO_OUT_HORIZ_PIX_EXT_SZ + o_pix4*MLO_READ_UNIT + 2] = out_rd_data[2];
							lcl_top[(ob*MLO_N_LCL_OUT_MAPS + o) * MLO_OUT_LCL_SZ + o_scan * MLO_OUT_HORIZ_PIX_EXT_SZ + o_pix4*MLO_READ_UNIT + 3] = out_rd_data[3];
							//						for (int i = 0; i < MLO_READ_UNIT; ++i)
							//						{
							//							lcl_top[o * MLO_OUT_LCL_SZ + o_scan * MLO_OUT_HORIZ_PIX_EXT_SZ + o_pix4*MLO_READ_UNIT + i] = out_rd_data[o*MLO_READ_UNIT + i];
							//						}

#if 0
							if (c_idx == 0 && o_idx == 0 && og == 0 && o == 0/* && out_y + o_scan == 0*/)
							{
								printf("K:o: %d %d %d %f %f %f\n",
									lcl_id,
									gbl_out_scan_off + o*MLO_OUT_CHANNEL_STRIDE + o_scan * MLO_OUT_STRIDE + o_pix4*MLO_READ_UNIT,
									o * MLO_OUT_LCL_SZ + o_scan * MLO_OUT_HORIZ_PIX_EXT_SZ + o_pix4*MLO_READ_UNIT,
									lcl_top[o * MLO_OUT_LCL_SZ + o_scan * MLO_OUT_HORIZ_PIX_EXT_SZ + o_pix4*MLO_READ_UNIT + 0],
									lcl_top[o * MLO_OUT_LCL_SZ + o_scan * MLO_OUT_HORIZ_PIX_EXT_SZ + o_pix4*MLO_READ_UNIT + 1],
									lcl_top[o * MLO_OUT_LCL_SZ + o_scan * MLO_OUT_HORIZ_PIX_EXT_SZ + o_pix4*MLO_READ_UNIT + 2]
								);
							}
#endif

						}
					}
				}
				barrier(CLK_LOCAL_MEM_FENCE);


				// process	
				// algorithm

#if 1
				//				if (w_blk_idx < MLO_MAX_WEI_BLK_LOOP)
				{
					for (int j = 0; j < MLO_N_ALIGNED_OUT_SCAN_BLK; ++j)
					{

						_FLOAT i_vals[MLO_N_LCL_BATCHS*MLO_WEI_WKITEM];

						for (int lb = 0; lb < MLO_N_LCL_BATCHS; ++lb)
						{

							for (int w = 0; w < MLO_WEI_WKITEM - 1; ++w)
							{
								int w_x = w_x0 + w*MLO_WEI_BLK_SZ0;
								int i_off = lb*MLO_IN_LCL_SZ + (j*MLO_FILTER_STRIDE1 + w_y) * MLO_IN_LCL_WIDTH + (w_blk_idx*MLO_OUT_WEI_SCAN_BLK + 0) * MLO_FILTER_STRIDE0 + w_x;
								_FLOAT i_val = lcl_bot[i_off];

								i_vals[lb*MLO_WEI_WKITEM + w] = i_val;

							}
						}

						// if we overshoot the scanline
						// out data will be 0 by initial setting
						for (int i = 0; i < MLO_OUT_WEI_SCAN_BLK; ++i)
						{

							//						for(int w = 0; w < MLO_WEI_WKITEM; ++w)

							for (int lb = 0; lb < MLO_N_LCL_BATCHS; ++lb)
							{
								int w_x = w_x0 + (MLO_WEI_WKITEM - 1)* MLO_WEI_BLK_SZ0;
								int i_off = lb*MLO_IN_LCL_SZ + (j*MLO_FILTER_STRIDE1 + w_y) * MLO_IN_LCL_WIDTH + (w_blk_idx*MLO_OUT_WEI_SCAN_BLK + i) * MLO_FILTER_STRIDE0 + w_x;
								_FLOAT i_val = lcl_bot[i_off];

								i_vals[lb*MLO_WEI_WKITEM + (MLO_WEI_WKITEM - 1)] = i_val;

							}


							for (int o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)
							{
								// read with MLO_OUT_HORIX_PIX_EXT_SZ stride
								for (int ob = 0; ob < MLO_N_LCL_BATCHS; ++ob)
								{
									_FLOAT o_val
										= lcl_top[(ob*MLO_N_LCL_OUT_MAPS + o)*MLO_OUT_LCL_SZ + j * MLO_OUT_HORIZ_PIX_EXT_SZ + (w_blk_idx*MLO_OUT_WEI_SCAN_BLK + i)];

									int w = 0;
									_FLOAT i_val;
									for (/*int w = 0*/; w < MLO_WEI_WKITEM; ++w)
									{


										for (int lb = 0; lb < MLO_N_LCL_BATCHS; ++lb)
										{
											i_val = i_vals[lb*MLO_WEI_WKITEM + w];
#if 0
											pvt_accum[(og * MLO_N_LCL_OUT_MAPS + o) * MLO_WEI_WKITEM + w] = fma(i_val, o_val, pvt_accum[(og * MLO_N_LCL_OUT_MAPS + o) * MLO_WEI_WKITEM + w]);
#else
											pvt_accum[(og * MLO_N_LCL_OUT_MAPS + o) * MLO_WEI_WKITEM + w] += i_val * o_val;
#endif


#if 0
											int w_x = w_x0 + w*MLO_WEI_BLK_SZ0;
											if (c_idx == 0 && o_idx == 0 && og == 0 && o == 0 && w_y == 0 && w_x == 0 && w_blk_idx < MLO_MAX_WEI_BLK_LOOP /* && lcl_id < MLO_OUT_WEI_SCAN_BLK * MLO_MAX_WEI_BLK_LOOP * MLO_WEI_BLK_SZ0 * MLO_WEI_WKITEM*/)
											{
												int i_off = lb*MLO_IN_LCL_SZ + (j*MLO_FILTER_STRIDE1 + w_y) * MLO_IN_LCL_WIDTH + (w_blk_idx*MLO_OUT_WEI_SCAN_BLK + i) * MLO_FILTER_STRIDE0 + w_x;
												printf("K:s: %d %d %d %d %d %d %d %f %f %f\n",
													lcl_id,
													lb,
													j,
													i,
													w,
													i_off,
													(ob*MLO_N_LCL_OUT_MAPS + o)*MLO_OUT_LCL_SZ + j * MLO_OUT_HORIZ_PIX_EXT_SZ + (w_blk_idx*MLO_OUT_WEI_SCAN_BLK + i),
													pvt_accum[(og * MLO_N_LCL_OUT_MAPS + o) * MLO_WEI_WKITEM + w],
													i_val,
													o_val
												);
											}
#endif
										}
									} // for (/*int w = 0*/; w < MLO_WEI_WKITEM; ++w)
								}
							} // for (int o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)

							for (int lb = 0; lb < MLO_N_LCL_BATCHS; ++lb)
							{

								for (int w = 0; w < MLO_WEI_WKITEM - 1; ++w)
								{
									i_vals[lb*MLO_WEI_WKITEM + w] = i_vals[lb*MLO_WEI_WKITEM + w + 1];

								}
							}

						}
					}
				}
#else
				for (int o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)
				{
					for (int w = 0; w < MLO_WEI_WKITEM; ++w)
					{
						pvt_accum[(og * MLO_N_LCL_OUT_MAPS + o) * MLO_WEI_WKITEM + w] = lcl_top[(og * MLO_N_LCL_OUT_MAPS + o)] * lcl_bot[w];
					}
				}
#endif			


			} // for(; og < (MLO_N_OUT_BLK_GRP; ++og )

			barrier(CLK_LOCAL_MEM_FENCE);

			// move data up
			for (int lb = 0; lb < MLO_N_LCL_BATCHS; ++lb)
			{
				int c_scan = 0;
				for (int p4 = lcl_id; p4 < MLO_N_IN_HORIZ_READS * (MLO_FILTER_SIZE1 - MLO_FILTER_STRIDE1);
					p4 += MLO_GRP_SZ, c_scan = iDiv(p4, MLO_N_IN_HORIZ_READS))
				{
					int c_pix4 = iMod(p4, c_scan, MLO_N_IN_HORIZ_READS);

					*(MLO_READ_TYPE*)in_rd_data = *(MLO_READ_TYPE*)&lcl_bot[lb*MLO_IN_LCL_SZ + (c_scan + (MLO_IN_LCL_HEIGHT - MLO_FILTER_SIZE1 + MLO_FILTER_STRIDE1))*(MLO_IN_LCL_WIDTH)+MLO_FILTER_PAD0 + c_pix4*MLO_READ_UNIT];
					*(MLO_READ_TYPE*)&lcl_bot[lb*MLO_IN_LCL_SZ + c_scan*(MLO_IN_LCL_WIDTH)+MLO_FILTER_PAD0 + c_pix4*MLO_READ_UNIT] = *(MLO_READ_TYPE*)in_rd_data;

				}
			}


		} // for (int ob = 0; ob < MLO_N_OUT_BLK; ++ob, in_y += (MLO_IN_LCL_HEIGHT - MLO_FILTER_SIZE1 + 1), out_y += MLO_N_ALIGNED_OUT_SCAN_BLK)
	} // for (int b = 0;


#endif


	int oo = iDiv(lcl_id, MLO_WEI_CHANNEL_STRIDE);
	int wei_i = iMod(lcl_id, oo, MLO_WEI_CHANNEL_STRIDE);
	int wei_i_y = iDiv(wei_i, (MLO_FILTER_SIZE0));
	int wei_i_x = iMod(wei_i, wei_i_y, (MLO_FILTER_SIZE0));
	// send it out
	// inputs are outputs
	int wei_df_off = ((ib * MLO_N_OUTPUTS + o_idx) * (int)MLO_WEI_BATCH_STRIDE)
		// this input channel
		+ mul24(c_idx, (int)MLO_WEI_CHANNEL_STRIDE);

	int wei_lcl_off = 0;
	_FLOAT final_sum = 0;


	// save in lcl and orgnize in a proper order
	// outputs
	//	filter size1
	//	  filter size0
	// TO DO DEPENDING ON THE GROUP SIZE
	for (int og = 0; og < MLO_N_OUT_BLK_GRP; ++og)
	{
		for (int o = 0; w_blk_idx < MLO_MAX_WEI_BLK_LOOP/* && lcl_id < MLO_OUT_WEI_SCAN_BLK * MLO_MAX_WEI_BLK_LOOP * MLO_WEI_BLK_SZ0 * MLO_WEI_WKITEM*/ && o < MLO_N_LCL_OUT_MAPS; ++o)
		{
			int w = 0;
			for (; w < MLO_WEI_WKITEM; ++w)
			{
				// save "virtual" filter table
				int w_x = w_x0 + w*MLO_WEI_BLK_SZ0;
				wei_lcl_off = ((o * MLO_MAX_WEI_BLK_LOOP + w_blk_idx) * MLO_FILTER_SIZE1 + w_y) * (MLO_WEI_BLK_SZ0 *MLO_WEI_WKITEM) + w_x;
				lcl_bot[wei_lcl_off] = pvt_accum[(og * MLO_N_LCL_OUT_MAPS + o) * MLO_WEI_WKITEM + w];
#if 0
				// wei_lcl_off == 1020 || wei_lcl_off == 1080 || wei_lcl_off == 1140)
				if (c_idx == 0 && o_idx == 0 && og == 0 && o == 0 && w_y == 1 && w_x == 1/* && lcl_id < MLO_OUT_WEI_SCAN_BLK * MLO_MAX_WEI_BLK_LOOP * MLO_WEI_BLK_SZ0 * MLO_WEI_WKITEM*/)
				{
					printf("K:u: %d %d %d %f %f\n",
						lcl_id,
						w_blk_idx,
						wei_lcl_off,
						lcl_bot[wei_lcl_off],
						pvt_accum[(og * MLO_N_LCL_OUT_MAPS + o) * MLO_WEI_WKITEM + w]
					);
				}
#endif

			}

		}

		barrier(CLK_LOCAL_MEM_FENCE);
#if 1	
		// read into real filter table

		if (lcl_id < (MLO_N_LCL_OUT_MAPS *MLO_FILTER_SIZE1*MLO_FILTER_SIZE0))
		{
			final_sum = 0;
			for (int i = 0; i < MLO_MAX_WEI_BLK_LOOP; ++i)
			{
				final_sum += lcl_bot[((oo * MLO_MAX_WEI_BLK_LOOP + i) * MLO_FILTER_SIZE1 + wei_i_y) * (MLO_WEI_BLK_SZ0 *MLO_WEI_WKITEM) + wei_i_x];
#if 0
				if (c_idx == 0 && o_idx == 0 && og == 0 && oo == 0 && wei_i_y == 0 && wei_i_x == 0)
				{
					printf("K:fs: %d %d %d  %f %f\n",
						lcl_id,
						i,
						((oo * MLO_MAX_WEI_BLK_LOOP + i) * MLO_FILTER_SIZE1 + wei_i_y) * (MLO_WEI_BLK_SZ0 *MLO_WEI_WKITEM) + wei_i_x, final_sum,
						lcl_bot[((oo * MLO_MAX_WEI_BLK_LOOP + i) * MLO_FILTER_SIZE1 + wei_i_y) * (MLO_WEI_BLK_SZ0 *MLO_WEI_WKITEM) + wei_i_x]
					);
				}
#endif
				}

#if 1

			weights_df[wei_df_off + (og *  MLO_N_LCL_OUT_MAPS + oo) * MLO_WEI_BATCH_STRIDE + wei_i] = final_sum; //lcl_bot[lcl_id]; //

#if 0
			if (og == 0 && oo == 0 && wei_i == 0 && c_idx == 0)
			{

				printf("K:o: %d %d %d %d %d %d %f\n",
					ib,
					o_idx,
					c_idx,
					(og *  MLO_N_LCL_OUT_MAPS + oo),
					wei_df_off,
					wei_df_off + (og *  MLO_N_LCL_OUT_MAPS + oo) * MLO_WEI_BATCH_STRIDE + wei_i,
					weights_df[wei_df_off + (og *  MLO_N_LCL_OUT_MAPS + oo) * MLO_WEI_BATCH_STRIDE + wei_i]
				);
			}
#endif
#else
			_FLOAT t_accum = 0;

			for (int og = 0; og < MLO_N_OUT_BLK_GRP; ++og)
			{
				for (int o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)
				{
					for (int w = 0; w < MLO_WEI_WKITEM; ++w)
					{
						t_accum += pvt_accum[(og * MLO_N_LCL_OUT_MAPS + o) * MLO_WEI_WKITEM + w];
					}
				}
			}

			weights_df[lcl_id] = t_accum;
#endif


	}
#endif
		barrier(CLK_LOCAL_MEM_FENCE);

	} // for(int og = 0; og < MLO_N_OUT_BLK_GRP; ++og)



}


__attribute__((reqd_work_group_size(MLO_UT_GRP_SZ0, 1, 1)))
__kernel void MLOpenCvBwdWrW_rdc(
	const __global _FLOAT * weight_df_tmp,
	__global _FLOAT * weights_df
)
{
	int gbl_id = get_global_id(0);
	int wei_idx0 = gbl_id * MLO_UT_READ_UNIT;

	int wei_blk_idx = iDiv(wei_idx0, MLO_WEI_CHANNEL_STRIDE);
	int wei_idx = iMod(wei_idx0, wei_blk_idx, MLO_WEI_CHANNEL_STRIDE);

	_FLOAT pvt_accum_wei[MLO_UT_READ_UNIT] = { 0,0 };

	int batch_loop = (MLO_BATCH_SZ + (MLO_N_BATCH_LOOPS*MLO_N_LCL_BATCHS) - 1) / (MLO_N_BATCH_LOOPS*MLO_N_LCL_BATCHS);
	for (int i = 0; i < batch_loop; ++i)
	{
		*(MLO_UT_READ_TYPE*)pvt_accum_wei
			+= *(MLO_UT_READ_TYPE*)&weight_df_tmp[(wei_blk_idx * MLO_WEI_CHANNEL_STRIDE + i* MLO_N_OUTPUTS*MLO_WEI_BATCH_STRIDE)  + wei_idx];
#if 0
		if (wei_blk_idx == 0 && wei_idx == 0)
		{
			printf("k:a: %d  %f\n",
				(wei_blk_idx * MLO_WEI_CHANNEL_STRIDE + i* MLO_N_OUTPUTS*MLO_WEI_BATCH_STRIDE) + wei_idx,
				pvt_accum_wei[0]
				);
		}
#endif

	}
	*(MLO_UT_READ_TYPE*)&weights_df[wei_idx0] = *(MLO_UT_READ_TYPE*)pvt_accum_wei;

}