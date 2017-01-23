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

#define MLO_N_OUT_VERTICAL_READS (MLO_OUT_HEIGHT)

// weights tiles verticall/horiz
#define MLO_WEI_TILE_SZ1 ((MLO_FILTER_SIZE1 + MLO_OUT_TILE1 -1)/MLO_OUT_TILE1)
#define MLO_WEI_TILE_SZ0 ((MLO_FILTER_SIZE0 + MLO_OUT_TILE0 -1)/MLO_OUT_TILE0)

// weight tile in wk-items
#define MLO_WEI_TILE_SZ (MLO_WEI_TILE_SZ1*MLO_WEI_TILE_SZ0)

// n accum tiles (wk_items) along x
#define MLO_N_ACCUM0 ((MLO_OUT_WIDTH +  MLO_IN_TILE0 -1) /  MLO_IN_TILE0)
// n accum tiles (wk-items) along y
#define MLO_N_ACCUM1 ((MLO_N_OUT_VERTICAL_READS +  MLO_IN_TILE1 -1) /  MLO_IN_TILE1)
// vert processing area
#define MLO_N_OUT_VERT_PROCS (MLO_N_ACCUM1 *MLO_IN_TILE1)

// total number of weight processing blocks
#define MLO_N_WEI_TILES (MLO_N_ACCUM1 * MLO_N_ACCUM0 * MLO_WEI_TILE_SZ)

// total number of prcessing elements(wk-items)
#define MLO_WEI_TILES_SZ (MLO_N_WEI_TILES * MLO_OUT_TILE1 * MLO_OUT_TILE0)

#define MLO_WEI_BLKS_LCL_SZ (MLO_WEI_TILES_SZ * MLO_N_LCL_OUT_MAPS * MLO_N_LCL_IN_MAPS * MLO_OUT_STACKS)


//#if MLO_OUT_HORIZ_PIX_SZ >  (MLO_N_ACCUM0 * MLO_IN_TILE0)
#define MLO_OUT_HORIZ_PIX_EXT_SZ (MLO_OUT_HORIZ_PIX_SZ)
//#else
//#define MLO_OUT_HORIZ_PIX_EXT_SZ (MLO_N_ACCUM0 * MLO_IN_TILE0)
//#endif


//#define MLO_OUT_BLK_GRP_EXT_PIX_SZ (MLO_OUT_HORIZ_PIX_EXT_SZ * MLO_N_OUT_VERT_PROCS)
#define MLO_OUT_LCL_SZ (MLO_OUT_HORIZ_PIX_EXT_SZ * MLO_N_OUT_VERT_PROCS)
// LDS OUT SIZE
#define MLO_TOTAL_OUT_LCL_SZ (MLO_N_LCL_BATCHS*MLO_N_LCL_OUT_MAPS*MLO_OUT_STACKS*MLO_OUT_LCL_SZ)

#if (MLO_N_OUT_VERT_PROCS == MLO_N_OUT_VERTICAL_READS)
#define MLO_BLK_ALIGNED 1
#else
#define MLO_BLK_ALIGNED 0
#endif


#define MLO_IN_VERT_READS (MLO_IN_HEIGHT)
// count weight tile vertical border (bottom)
#define MLO_IN_VERT_PROCS (MLO_IN_VERT_READS + MLO_WEI_TILE_SZ1 * MLO_OUT_TILE1 - MLO_FILTER_SIZE1)
#define MLO_IN_LCL_HEIGHT (MLO_IN_VERT_PROCS + 2 * MLO_FILTER_PAD1)
// there is an assumption that the scanline fits into LDS
#define MLO_N_IN_HORIZ_PIX_READS (MLO_IN_WIDTH) //((MLO_OUT_WIDTH-1)*MLO_FILTER_STRIDE0 + MLO_FILTER_SIZE0 - 2 * MLO_FILTER_PAD0
// count weight tile horizontal border (right)
#define MLO_IN_HORIZ_PROC (MLO_N_IN_HORIZ_PIX_READS + MLO_WEI_TILE_SZ0 * MLO_OUT_TILE0 - MLO_FILTER_SIZE0)


#define MLO_N_IN_HORIZ_READS ((MLO_N_IN_HORIZ_PIX_READS + MLO_READ_UNIT - 1) / MLO_READ_UNIT)
#define MLO_IN_N_PIXS_OFF  (MLO_N_IN_HORIZ_READS*MLO_READ_UNIT - MLO_N_IN_HORIZ_PIX_READS)

#if MLO_IN_HORIZ_PROC > (MLO_N_IN_HORIZ_READS * MLO_READ_UNIT)
#define  MLO_IN_LCL_HORIZ_SZ MLO_IN_HORIZ_PROC
#else
#define  MLO_IN_LCL_HORIZ_SZ (MLO_N_IN_HORIZ_READS * MLO_READ_UNIT)
#endif
// assum the input scan + 2 pads fit into LDS
#define MLO_IN_LCL_WIDTH (MLO_IN_LCL_HORIZ_SZ + 2 * MLO_FILTER_PAD0)


#define MLO_IN_LCL_SZ (MLO_IN_LCL_WIDTH * MLO_IN_LCL_HEIGHT)
// LDS IN SIZE
#define MLO_TOTAL_IN_LCL_SZ (MLO_N_LCL_BATCHS*MLO_N_LCL_IN_MAPS*MLO_IN_LCL_SZ)

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

inline int gePhysLocalId()
{
	int lcl_wave_id = get_local_id(0) - ((get_local_id(0) >> MLO_LG2_PHYS_WAVE_SZ) << MLO_LG2_PHYS_WAVE_SZ);
	return(lcl_wave_id);
}

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
// read first half
	if (scan_lcl < (sum_stride >> 1))
	{
		for (int i = 0; i < unit_len; ++i)
		{
			weights_accum[i] = lcl_blob[(lcl_id + scan_lcl) * unit_len + i];

		}

	}
// add second half
// appload accumulated value so far
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


inline void ReduceKernel64(__local _FLOAT * lcl_blob, _FLOAT *weights_accum, int lcl_id, int scan_lcl, int sum_stride, int unit_len, bool debug)
{
	// read first half
	if (scan_lcl < (sum_stride >> 1))
	{
		for (int i = 0; i < unit_len; ++i)
		{
			weights_accum[i] = lcl_blob[(lcl_id + scan_lcl) * unit_len + i];

		}

	}
	// add second half
	// appload accumulated value so far
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


inline void  Kahan_summation(_FLOAT *sum, _FLOAT * c, _FLOAT v)
{
	_FLOAT y = v - *c;    //So far, so good: c is zero.
	_FLOAT t = *sum + y;         //Alas, sum is big, y small, so low-order digits of y are lost.
	*c = (t - *sum) - y;   //(t - sum) recovers the high-order part of y; subtracting y recovers -(low part of y)
	*sum = t;             //Algebraically, c should always be zero. Beware eagerly optimising compilers!
}

inline void  Kahan_summation_tricked(_FLOAT *sum, _FLOAT * c, _FLOAT v, _FLOAT mod)
{
	_FLOAT y = v - *c;    //So far, so good: c is zero.
	_FLOAT t = *sum + y;         //Alas, sum is big, y small, so low-order digits of y are lost.
	*c = (t - *sum) * mod - y;   //(t - sum) recovers the high-order part of y; subtracting y recovers -(low part of y)
	*sum = t;             //Algebraically, c should always be zero. Beware eagerly optimising compilers!
}


inline void Kahan_summation2(_FLOAT *sum, _FLOAT *c, _FLOAT *v, int n)
{
	for (int i = 0; i < n; ++i)
	{
		_FLOAT y = v[i] - c[i];    //So far, so good: c is zero.
		_FLOAT t = sum[i] + y;         //Alas, sum is big, y small, so low-order digits of y are lost.
		c[i] = (t - sum[i]) - y;   //(t - sum) recovers the high-order part of y; subtracting y recovers -(low part of y)
		sum[i] = t;             //Algebraically, c should always be zero. Beware eagerly optimising compilers!
	}
}

/*********************************************************************************************************
// wrw algorithm for large filters
// idea:
// split filter taps into sub-tiles along x and y axis with number of tap groups muliples of stride or 1
// for example
// the 5x10 filter has been split into 10 sub-tiles 1x5 each, 1 tap in y direction and 5 taps in x direction.
// those horizontal taps are 0, 2, 4, 6, 8 and 1, 3, 5, 7, 9
// a single vertical tap is 0 or 1 or 2 or 3 or 4.
// one may say sub-tiles are indexed by a vertical tap.
// the partial sum has been calculated into those 10 sub-tiles in parallel.
// the full filter has been calulated by reducing all sub-tiles into a single filter per group.
// teh accumulation has been done over all pixels of several outputs being shared with a single input.
// the accuulation has been done per batch.
//
// the total reduction over all batches has been doesn a separete kerenel.
//
// alg
//
//		until end of output map (MLO_N_OUT_BLK)
//			load input map block in LDS
//			load output maps in LDS
//          for j in output scans
//				for i in output scan interval
//                  accumulate the weights into sub-tiles
//
//		reduce sub-tiles into a single filter for each output
//		write accululated weights
//
// group layout
// 0 - n waves * wave size (n_waves has been defined by host)
// 1 - input channel index
// 2 - output channel/batch index
//
//
// for each batch
//	 accumulate all weights per input/output pair


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
#if 1

	// input/output tiles + reduce buffer

	__local _FLOAT lcl[(MLO_LCL_SZ)];
	__local _FLOAT * lcl_bot = lcl;
	__local _FLOAT * lcl_top = lcl + MLO_TOTAL_IN_LCL_SZ;


	// guarnteeing an uniformity over a wave
	int wave_id = getWaveId();
	int lcl_id = get_local_id(0);

	int dat_tl_idx = iDiv(lcl_id, MLO_WEI_TILE_SZ);
	int wei_tl_idx = iMod(lcl_id, dat_tl_idx, MLO_WEI_TILE_SZ);

// weight grid tile
	int wei_tl_idx1 = iDiv(wei_tl_idx, MLO_WEI_TILE_SZ0);
	int wei_tl_idx0 = iMod(wei_tl_idx, wei_tl_idx1, MLO_WEI_TILE_SZ0);
	int wei_tl1 = wei_tl_idx1 * MLO_OUT_TILE1;
	int wei_tl0 = wei_tl_idx0 * MLO_OUT_TILE0;

// in/out grid tile
	int dat_tl_idx1 = iDiv(dat_tl_idx, MLO_N_ACCUM0);
	int dat_tl_idx0 = iMod(dat_tl_idx, dat_tl_idx1, MLO_N_ACCUM0);

	int dat_tl1 = dat_tl_idx1 * MLO_IN_TILE1;
	int dat_tl0 = dat_tl_idx0 * MLO_IN_TILE0;


// processor vertical pos
	int lcl_bot_off = (dat_tl1 + wei_tl1) * MLO_IN_LCL_WIDTH + (dat_tl0 + wei_tl0);

// top tile vertical/horizontal positions
	int lcl_top_off = (dat_tl1) * (MLO_OUT_HORIZ_PIX_EXT_SZ) + (dat_tl0);


	int c_idx_base = get_group_id(1); // input map index base

	int o_idx_base = iDiv(get_group_id(2), (MLO_BATCH_SZ / (MLO_N_BATCH_LOOPS*MLO_N_LCL_BATCHS))); // output map index base
	int ib_base = iMod(get_group_id(2), o_idx_base, (MLO_BATCH_SZ / (MLO_N_BATCH_LOOPS*MLO_N_LCL_BATCHS)));

	int ib = ib_base*MLO_N_LCL_BATCHS;

	int c_idx = c_idx_base * MLO_N_LCL_IN_MAPS; // input map index

	int o_idx = o_idx_base * (MLO_N_LCL_OUT_MAPS * MLO_OUT_STACKS); // output map index

	int gbl_in_off = c_idx * MLO_IN_CHANNEL_STRIDE + ib * MLO_IN_BATCH_STRIDE;
	int gbl_out_off = o_idx * MLO_OUT_CHANNEL_STRIDE + ib * MLO_OUT_BATCH_STRIDE;

	__private _FLOAT pvt_accum[(MLO_OUT_TILE1 *  MLO_OUT_TILE0 * MLO_N_LCL_OUT_MAPS * MLO_N_LCL_IN_MAPS)];

	for (int i = 0; i < (MLO_OUT_TILE1 *  MLO_OUT_TILE0 * MLO_N_LCL_OUT_MAPS * MLO_N_LCL_IN_MAPS); ++i)
	{
		pvt_accum[i] = 0;
	}


	// zero out LDS
	for (int i = lcl_id; i < (MLO_LCL_SZ); i += MLO_GRP_SZ)
	{
		lcl[i] = 0;
	}

//	barrier(CLK_LOCAL_MEM_FENCE);


#if 0
	if (wei_tl_idx == 4 && o_idx == 0 && c_idx == 0 && (dat_tl_idx == 0 || dat_tl_idx == 1))
	{
		printf("G:p: %d %d %d %d %d\n",
			lcl_id,
			wei_tl_idx1,
			wei_tl_idx0,
			wei_tl1,
			wei_tl0
		);
	}

#endif


#if 1
	// over all batches

	for (int b = 0;
		b < MLO_N_BATCH_LOOPS;
		++b,
		gbl_in_off += MLO_N_LCL_BATCHS*MLO_IN_BATCH_STRIDE,
		gbl_out_off += MLO_N_LCL_BATCHS*MLO_OUT_BATCH_STRIDE
		)
	{
		int in_y = 0;
		int out_y = 0;

		// prefetch MLO_FILTER_STRIDE1 - MLO_FILTER_PAD1 input scans
		__private _FLOAT in_rd_data[MLO_READ_UNIT];

		int gbl_in_scan_off = gbl_in_off;
		int gbl_out_scan_off = gbl_out_off;
		// over all out blocks
		// processing per MLO_N_ALIGNED_OUT_SCAN_BLK output scans


		barrier(CLK_LOCAL_MEM_FENCE);




		for (int p4 = lcl_id; p4 < MLO_N_LCL_IN_MAPS * MLO_N_IN_HORIZ_READS * MLO_IN_VERT_READS;
			p4 += MLO_GRP_SZ)
		{
			int c_scan = 0;

			int c = 0;
			int p4_t = p4;
#if MLO_N_LCL_IN_MAPS > 1
			c = iDiv(p4, (MLO_N_IN_HORIZ_READS * MLO_IN_VERT_READS));
			p4_t = iMod(p4, c, (MLO_N_IN_HORIZ_READS*MLO_IN_VERT_READS));
//			if (c_idx + c < MLO_N_INPUTS)
#endif

			{
				c_scan = iDiv(p4_t, MLO_N_IN_HORIZ_READS);

				int c_pix4 = iMod(p4_t, c_scan, MLO_N_IN_HORIZ_READS);


#if MLO_IN_N_PIXS_OFF > 0

				if (c_pix4 == MLO_N_IN_HORIZ_READS - 1)
				{
					for (int i = 0; i < MLO_IN_N_PIXS_OFF; ++i)
					{

						in_rd_data[i] = bot[gbl_in_scan_off + c*MLO_IN_CHANNEL_STRIDE + c_scan * MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT + i];
					}

					for (int i = MLO_READ_UNIT - 1; i >= MLO_READ_UNIT - MLO_IN_N_PIXS_OFF; --i)
					{
						in_rd_data[i] = 0;
					}

				}
				else
#endif
				{
					*(MLO_READ_TYPE*)in_rd_data = *(MLO_READ_TYPE*)&bot[gbl_in_scan_off + c*MLO_IN_CHANNEL_STRIDE + c_scan * MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT];
				}


				for (int i = 0; i < MLO_READ_UNIT; ++i)
				{
					int lcl_in_off = c*MLO_IN_LCL_SZ + (c_scan + MLO_FILTER_PAD1)*MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 + c_pix4*MLO_READ_UNIT + i;
					lcl_bot[lcl_in_off] = in_rd_data[i];
#if 0
					if (c_idx + c == 1 && p4_t == 0)
					{
						printf("K:g: %d %f\n",
							lcl_in_off,
							lcl_bot[lcl_in_off]
						);
					}
#endif
				}
			}

		} // for (int p4 = lcl_id; p4 < MLO_N_LCL_IN_MAPS * MLO_N_IN_HORIZ_READS * MLO_IN_VERT_READS;


		// MLO_N_LCL_OUT_MAPS number is restricted by LDS size

		// fetch output. MLO_N_ALIGNED_OUT_SCAN_BLK output scans, each of size MLO_N_OUT_HORIZ_READS

		__private _FLOAT out_rd_data[MLO_READ_UNIT];

		for (int o_p4 = lcl_id; o_p4 < (MLO_N_LCL_OUT_MAPS* MLO_OUT_STACKS*MLO_N_OUT_VERTICAL_READS*MLO_N_OUT_HORIZ_READS);
			o_p4 += MLO_GRP_SZ)
		{
			int o = iDiv(o_p4, (MLO_N_OUT_VERTICAL_READS*MLO_N_OUT_HORIZ_READS));
			int o_pX4 = iMod(o_p4, o, (MLO_N_OUT_VERTICAL_READS*MLO_N_OUT_HORIZ_READS));
			int o_scan = iDiv(o_pX4, MLO_N_OUT_HORIZ_READS);
			int o_pix4 = iMod(o_pX4, o_scan, MLO_N_OUT_HORIZ_READS);

			// scan has been fetch by 4
			// here the non-multiple of 4 scan has been handled
			// also makes sure the input garbage hs been multipled by 0
			if (o_idx + o < MLO_N_OUTPUTS)
			{
#if MLO_OUT_N_PIXS_OFF > 0
				if (o_pix4 == (MLO_N_OUT_HORIZ_READS - 1))
				{
					for (int i = 0; i < MLO_OUT_N_PIXS_OFF; ++i)
					{
						out_rd_data[i] = top_df[gbl_out_scan_off + o*MLO_OUT_CHANNEL_STRIDE + o_scan * MLO_OUT_STRIDE + o_pix4*MLO_READ_UNIT + i];
					}

					for (int i = MLO_READ_UNIT - 1; i >= MLO_READ_UNIT - MLO_OUT_N_PIXS_OFF; --i)
					{
						out_rd_data[i] = 0;
					}
				}
				else
#endif
				{
					*(MLO_READ_TYPE*)out_rd_data
						= *(MLO_READ_TYPE*)&top_df[gbl_out_scan_off + o*MLO_OUT_CHANNEL_STRIDE + o_scan * MLO_OUT_STRIDE + o_pix4*MLO_READ_UNIT];
				}


			}
			// write into LDS with MLO_OUT_HORIZ_PIX_EXT_SZ stride to zero out weights block overshoot
									//						*(MLO_READ_TYPE*)&lcl_top[o * MLO_OUT_LCL_SZ + o_scan * MLO_OUT_HORIZ_PIX_EXT_SZ + o_pix4*MLO_READ_UNIT] = *(MLO_READ_TYPE*)out_rd_data;
									//

			for (int i = 0; i < MLO_READ_UNIT; ++i)
			{
				lcl_top[o * MLO_OUT_LCL_SZ + o_scan * MLO_OUT_HORIZ_PIX_EXT_SZ + o_pix4*MLO_READ_UNIT + i] = out_rd_data[i];
			}

		} //	for (int oo_p4 = lcl_id; oo_p4 < (MLO_N_LCL_OUT_MAPS*MLO_N_ALIGNED_OUT_SCAN_BLK*MLO_N_OUT_HORIZ_READS); oo_p4 += MLO_GRP_SZ)

		barrier(CLK_LOCAL_MEM_FENCE);



		// process	
		// algorithm

#if 1
		__private _FLOAT bot_data[MLO_N_LCL_IN_MAPS*MLO_IN_TILE1 *(MLO_IN_TILE0 + MLO_OUT_TILE0 - 1)];
// prefetch
		for (int c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
		{

			for (int j = 0; j < (MLO_IN_TILE1 - 1); ++j)
			{
				for (int i = 0; i < (MLO_IN_TILE0 + MLO_OUT_TILE0 - 1); ++i)
				{
					int bot_lcl_off = c*MLO_IN_LCL_SZ + lcl_bot_off + j * MLO_IN_LCL_WIDTH + i;
					bot_data[(c*MLO_IN_TILE1 + j) *  (MLO_IN_TILE0 + MLO_OUT_TILE0 - 1) + i] = lcl_bot[bot_lcl_off];
				}
			}
		}




		for (int j = 0; j < MLO_OUT_TILE1; ++j)
		{
			// the next bot line 
			for (int c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
			{

				for (int i = 0; i < (MLO_IN_TILE0 + MLO_OUT_TILE0 - 1); ++i)
				{
					int bot_lcl_off = c*MLO_IN_LCL_SZ + lcl_bot_off + ((MLO_OUT_TILE1 - 1) + j) * MLO_IN_LCL_WIDTH + i;
					bot_data[(c*MLO_IN_TILE1 + (MLO_IN_TILE1 - 1)) *  (MLO_IN_TILE0 + MLO_OUT_TILE0 - 1) + i] = lcl_bot[bot_lcl_off];
				}
			}

// over all output
			for (int k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
			{
// read output scanline
				__private _FLOAT top_data[MLO_IN_TILE0];

				for (int c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
				{
					for (int l = 0; l < MLO_IN_TILE1; ++l)
					{
						for (int m = 0; m < MLO_IN_TILE0; ++m)
						{
							for (int i = 0; i < MLO_OUT_TILE0; ++i)
							{
								_FLOAT bot_val = bot_data[(c*MLO_IN_TILE1 + (l + j))*(MLO_IN_TILE0 + MLO_OUT_TILE0 - 1) + (m + i)];


								// read output scanline
								int top_lcl_off = k*MLO_OUT_LCL_SZ + lcl_top_off + l * MLO_OUT_HORIZ_PIX_EXT_SZ + m;
								top_data[m] = lcl_top[top_lcl_off];
								_FLOAT top_val = top_data[m];

								pvt_accum[k* (MLO_OUT_TILE1 *  MLO_OUT_TILE0 * MLO_N_LCL_IN_MAPS)
									+ c* (MLO_OUT_TILE1 *  MLO_OUT_TILE0)
									+ j*MLO_OUT_TILE0 + i]

									+= bot_val * top_val;

#if 0
								if (bot_val * top_val != 0 && wei_tl1 + j == 1 && wei_tl0 + i == 2 && o_idx + k == 0 && c_idx + c == 0)
								{
									printf("G:a: %d %d %d  %f %f %f %f\n",
										lcl_id,
										dat_tl_idx,
										wei_tl_idx,
										pvt_accum[k* (MLO_OUT_TILE1 *  MLO_OUT_TILE0 * MLO_N_LCL_IN_MAPS)
										+ c* (MLO_OUT_TILE1 *  MLO_OUT_TILE0)
										+ j*MLO_OUT_TILE0 + i],
										bot_val * top_val,
										bot_val,
										top_val
										);
								}

#endif

							} // for (int i = 0; i < MLO_OUT_TILE0; ++i)
						} // for (int m = 0; m < MLO_IN_TILE0; ++m)
					} // for (int l = 0; l < MLO_IN_TILE1; ++l)
				} // for (int c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
			} // for (int k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)

// move up
			for (int c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
			{
				for (int j = 0; j < (MLO_IN_TILE1 - 1); ++j)
				{
					for (int i = 0; i < (MLO_IN_TILE0 + MLO_OUT_TILE0 - 1); ++i)
					{
						bot_data[(c*MLO_IN_TILE1 + j) *  (MLO_IN_TILE0 + MLO_OUT_TILE0 - 1) + i]
							= bot_data[(c*MLO_IN_TILE1 + j + 1) *  (MLO_IN_TILE0 + MLO_OUT_TILE0 - 1) + i];
					}
				}
			}
		}

#else

		for (int k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
		{
			for (int c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
			{
				for (int l = 0; l < MLO_OUT_TILE1; ++l)
				{
					for (int m = 0; m < MLO_OUT_TILE0; ++m)
					{
						pvt_accum[k* (MLO_OUT_TILE1 *  MLO_OUT_TILE0 * MLO_N_LCL_IN_MAPS)
							+ c* (MLO_OUT_TILE1 *  MLO_OUT_TILE0)
							+ l*MLO_OUT_TILE0 + m]

							= lcl_bot[k + c]
							* lcl_top[l+m];
					}
				}
			}

		}


#endif // processing



	} // for (int b = 0;


#endif // over all batches


	barrier(CLK_LOCAL_MEM_FENCE);


// save in lcl and organize in a proper order
	// outputs
	//	  input (if available)
	//		 filter size1
	//			filter size0

	if (lcl_id < MLO_N_WEI_TILES)
	{
		for (int k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
		{
			for (int c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
			{
				for (int l = 0; l < MLO_OUT_TILE1; ++l)
				{
					for (int m = 0; m < MLO_OUT_TILE0; ++m)
					{
						// weights value location
						// all 64 of them together for easy summation.
						int wei_lcl_off = k*(MLO_WEI_TILES_SZ * MLO_N_ACCUM0 * MLO_N_ACCUM1 * MLO_N_LCL_IN_MAPS) // * MLO_OUT_STACKS)
							+ c*MLO_WEI_TILES_SZ * MLO_N_ACCUM0 * MLO_N_ACCUM1
							+ lcl_id * MLO_N_ACCUM0 * MLO_N_ACCUM1
							+ (wei_tl1 + l * MLO_WEI_TILE_SZ0 * MLO_OUT_TILE0)
							+ wei_tl0 + m;
						lcl[wei_lcl_off] =
							pvt_accum[k* (MLO_OUT_TILE1 *  MLO_OUT_TILE0 * MLO_N_LCL_IN_MAPS)
							+ c* (MLO_OUT_TILE1 *  MLO_OUT_TILE0)
							+ l*MLO_OUT_TILE0 + m];
					}
				}

			}

		}

	}

	barrier(CLK_LOCAL_MEM_FENCE);

		// send it out
		// inputs are outputs
		int wei_df_off = ((ib * MLO_N_OUTPUTS + o_idx) * (int)MLO_WEI_BATCH_STRIDE)
			// this input channel
			+ mul24(c_idx, (int)MLO_WEI_CHANNEL_STRIDE);

		for (int l = lcl_id; l < MLO_OUT_STACKS * MLO_N_LCL_OUT_MAPS * MLO_N_LCL_IN_MAPS* MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0; l += MLO_GRP_SZ)
		{

			int k = iDiv(l, MLO_N_LCL_IN_MAPS *MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0);
			int c_w = iMod(l, k, MLO_N_LCL_IN_MAPS *MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0);
			int c = iDiv(c_w, MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0);
			int wei_i = iMod(c_w, c, MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0);
			int wei_i1 = iDiv(wei_i, MLO_FILTER_SIZE0);
			int wei_i0 = iMod(wei_i, wei_i1, MLO_FILTER_SIZE0);

			wei_df_off += k * MLO_WEI_BATCH_STRIDE + c *MLO_WEI_CHANNEL_STRIDE + wei_i;

			_FLOAT final_sum = 0;
			for (int j = 0; j < MLO_N_ACCUM1 * MLO_N_ACCUM0; ++j)
			{
				int lcl_off = k*(MLO_WEI_TILES_SZ * MLO_N_ACCUM0 * MLO_N_ACCUM1 * MLO_N_LCL_IN_MAPS) // * MLO_OUT_STACKS)
					+ c*MLO_WEI_TILES_SZ * MLO_N_ACCUM0 * MLO_N_ACCUM1
					+ j * MLO_WEI_TILES_SZ
					+ wei_i1 * MLO_WEI_TILE_SZ0 * MLO_OUT_TILE0
					+ wei_i0;
				final_sum += lcl[lcl_off];
			}


			weights_df[wei_df_off] = final_sum;

		}


#endif

}


// final reduction kernel
// add filters over batches
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

	_FLOAT pvt_accum_wei[MLO_UT_READ_UNIT];
	for (int i = 0; i < MLO_UT_READ_UNIT; ++i)
	{
		pvt_accum_wei[i] = 0;
	}

	int batch_loop = (MLO_BATCH_SZ + (MLO_N_BATCH_LOOPS*MLO_N_LCL_BATCHS) - 1) / (MLO_N_BATCH_LOOPS*MLO_N_LCL_BATCHS);
	for (int i = 0; i < batch_loop; ++i)
	{
		*(MLO_UT_READ_TYPE*)pvt_accum_wei
			+= *(MLO_UT_READ_TYPE*)&weight_df_tmp[(wei_blk_idx * MLO_WEI_CHANNEL_STRIDE + i* MLO_N_OUTPUTS*MLO_WEI_BATCH_STRIDE)  + wei_idx];
	}

	*(MLO_UT_READ_TYPE*)&weights_df[wei_idx0] = *(MLO_UT_READ_TYPE*)pvt_accum_wei;

}