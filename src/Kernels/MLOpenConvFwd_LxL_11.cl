/*
 * Copyright (c) 2017 AMD Inc.
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



// filter size for all filters with small n of input maps (first layer)
// split a long filter by stride

#define MLO_N_FILTER_SPLITS1 ((MLO_FILTER_SIZE1 + MLO_FILTER_STRIDE1 - 1)/ MLO_FILTER_STRIDE1)
#ifndef MLO_OUT_PIX_TILE0
#define MLO_N_FILTER_SPLITS0 ((MLO_FILTER_SIZE0 + MLO_FILTER_STRIDE0 - 1)/ MLO_FILTER_STRIDE0)
#define MLO_OUT_PIX_TILE0 MLO_N_FILTER_SPLITS0
#endif
// processing arrangement
// generate full output width
// extent1 == MLO_GRP_SZ / MLO_PROCESING_WIDTH
#ifndef MLO_OUT_EXTENT1
#define MLO_PROCESSING_WIDTH  ((MLO_OUT_WIDTH + MLO_OUT_PIX_TILE0 - 1) / MLO_OUT_PIX_TILE0)
#define MLO_OUT_EXTENT1 (MLO_GRP_SZ / MLO_PROCESSING_WIDTH)
#endif


#define MLO_WEI_LCL_WIDTH MLO_FILTER_SIZE0 //(MLO_N_FILTER_SPLITS0*MLO_FILTER_STRIDE0)
#define MLO_WEI_EXTENT1 MLO_N_FILTER_SPLITS1
#define MLO_WEI_SZ (MLO_WEI_EXTENT1*MLO_WEI_LCL_WIDTH)
// LDS size
#define MLO_WEI_LCL_SZ (MLO_WEI_SZ * MLO_OUT_STACKS * MLO_N_LCL_OUT_MAPS* MLO_N_LCL_IN_MAPS)


#define MLO_IN_LCL_HEIGHT (MLO_OUT_EXTENT1 + MLO_N_FILTER_SPLITS1 - 1)
// there is an assumption that the scanline fits into LDS
#define MLO_N_IN_HORIZ_PIX_READS (MLO_IN_WIDTH)
#define MLO_N_IN_HORIZ_READS ((MLO_N_IN_HORIZ_PIX_READS + MLO_READ_UNIT - 1) / MLO_READ_UNIT)
#define MLO_IN_N_PIXS_OFF  (MLO_N_IN_HORIZ_PIX_READS - (MLO_N_IN_HORIZ_PIX_READS  / MLO_READ_UNIT)*MLO_READ_UNIT)

#define MLO_IN_LCL_WIDTH (MLO_N_IN_HORIZ_READS * MLO_READ_UNIT + 2 * MLO_FILTER_PAD0)
#define MLO_IN_LCL_SZ (MLO_IN_LCL_WIDTH*MLO_IN_LCL_HEIGHT)
// LDS size
#define MLO_TOTAL_IN_LCL_SZ (MLO_N_LCL_BATCHS*MLO_IN_LCL_SZ* MLO_N_LCL_IN_MAPS)

//#if (MLO_WEI_LCL_SZ + MLO_TOTAL_IN_LCL_SZ) > (MLO_N_PARTIAL_SUMS *MLO_OUT_WIDTH*MLO_OUT_EXTENT1)
#define MLO_LCL_MEM_SZ (MLO_WEI_LCL_SZ + MLO_TOTAL_IN_LCL_SZ)
//#else
//#define MLO_LCL_MEM_SZ (MLO_N_PARTIAL_SUMS *MLO_OUT_WIDTH*MLO_OUT_EXTENT1)
#//endif

// number of loops to flush put full output map
#define MLO_N_OUT_BLKS 1   //((MLO_OUT_HEIGHT + (MLO_OUT_PIX_TILE1*MLO_N_OUT_FOLDS1) -1) / (MLO_OUT_PIX_TILE1*MLO_N_OUT_FOLDS1))

#define MLO_HW_WAVE_ID_SETTING 0

#if MLO_HW_WAVE_ID_SETTING
extern __attribute__((const)) uint __hsail_get_dynwave_id(void);
static inline int getWaveId()
{
	int wave_id = 0;

	wave_id = __hsail_get_dynwave_id();
	wave_id = wave_id & MLO_N_PHYS_WAVES_MASK;
	return(wave_id);
}
#else
static inline int getWaveId()
{
	int wave_id = 0;

	wave_id = (get_local_id(0) >> MLO_LG2_PHYS_WAVE_SZ);

	return(wave_id);
}
#endif

static inline int gePhysLocalId()
{
	int lcl_wave_id = get_local_id(0) - ((get_local_id(0) >> MLO_LG2_PHYS_WAVE_SZ) << MLO_LG2_PHYS_WAVE_SZ);
	return(lcl_wave_id);
}

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

static inline void ReduceKernel(__local _FLOAT * lcl_blob, _FLOAT *weights_accum, int lcl_id, int scan_lcl, int sum_stride, int unit_len, bool debug)
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

static inline void  Kahan_summation(_FLOAT *sum, _FLOAT * c, _FLOAT v)
{
	_FLOAT y = v - *c;    //So far, so good: c is zero.
	_FLOAT t = *sum + y;         //Alas, sum is big, y small, so low-order digits of y are lost.
	*c = (t - *sum) - y;   //(t - sum) recovers the high-order part of y; subtracting y recovers -(low part of y)
	*sum = t;             //Algebraically, c should always be zero. Beware eagerly optimising compilers!
}

static inline void  Kahan_summation_tricked(_FLOAT *sum, _FLOAT * c, _FLOAT v, _FLOAT mod)
{
	_FLOAT y = v - *c;    //So far, so good: c is zero.
	_FLOAT t = *sum + y;         //Alas, sum is big, y small, so low-order digits of y are lost.
	*c = (t - *sum) * mod - y;   //(t - sum) recovers the high-order part of y; subtracting y recovers -(low part of y)
	*sum = t;             //Algebraically, c should always be zero. Beware eagerly optimising compilers!
}


static inline void Kahan_summation2(_FLOAT *sum, _FLOAT *c, _FLOAT *v, int n)
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
// process 3 output pixel per wk-item, 19 wk-items per output scan,
// 13 output sacn-line per group of 256
// read (13+2) input scan-lines 4 scan-lines apart from 2 batches
// convolve with 3 filters rows 4 rowes apart from 4(8) filter banks.


**********************************************************************************************************/

__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2)))
__kernel void MLOpenCvFwd(
	const __global _FLOAT * bot,
	const __global _FLOAT * weights,
#if MLO_CONV_BIAS == 1
	const __global _FLOAT * bias,
#endif
	__global _FLOAT *top,
	_FLOAT padding_val
)
{

	__local _FLOAT lcl_mem[MLO_LCL_MEM_SZ];
	__local _FLOAT * bot_mem = lcl_mem;
	__local _FLOAT * wei_mem = lcl_mem + MLO_TOTAL_IN_LCL_SZ;

	int wave_id = getWaveId();
	int lcl_id = get_local_id(0);
	int lcl_wv_id = gePhysLocalId();


	
	int ob = get_group_id(0);  // output map extent id

	int k_idx = get_group_id(1) * (MLO_OUT_STACKS * MLO_N_LCL_OUT_MAPS); // input map index based

	int c_idx = 0;

	int ib_idx = get_group_id(2)*MLO_N_LCL_BATCHS; // batch idx

	int ib = ib_idx;


	int gbl_in_off = /*c_idx * MLO_IN_CHANNEL_STRIDE + */ib * MLO_IN_BATCH_STRIDE;
	int gbl_wei_off = k_idx * MLO_WEI_BATCH_STRIDE;
	int out_y = ob*MLO_OUT_EXTENT1;
	gbl_in_off += out_y*MLO_FILTER_STRIDE1 * MLO_IN_STRIDE;

#define MLO_ACCUM_SZ (MLO_OUT_PIX_TILE1 * MLO_OUT_PIX_TILE0 * MLO_N_LCL_OUT_MAPS* MLO_N_LCL_IN_MAPS*MLO_N_LCL_BATCHS)

	__private _FLOAT pvt_accum[MLO_ACCUM_SZ];


	// zero out LDS
	for (int i = lcl_id; i < (MLO_LCL_MEM_SZ); i += MLO_GRP_SZ)
	{
		lcl_mem[i] = 0;
	}

// processing arrangement
	int ex_row = iDiv(lcl_id, MLO_PROCESSING_WIDTH);
// 
	int ex_col = iMod(lcl_id, ex_row, MLO_PROCESSING_WIDTH);
	int ex_pix = ex_col * MLO_OUT_PIX_TILE0;


	// over all batches

	for (int b = 0;
		b < MLO_N_BATCH_LOOPS;
		b += MLO_N_LCL_BATCHS,
		gbl_in_off += MLO_N_LCL_BATCHS*MLO_IN_BATCH_STRIDE
		)
	{

		barrier(CLK_LOCAL_MEM_FENCE);




		// prefetch MLO_FILTER_STRIDE1 - MLO_FILTER_PAD1 input scans
		__private _FLOAT in_rd_data[MLO_READ_UNIT];

		int gbl_in_scan_off0 = gbl_in_off;

		// generate pixels all MLO_OUT_STACKS * MLO_N_LCL_OUT_MAPS output maps
//		for (int ob = 0; ob < MLO_N_OUT_BLKS; ++ob, in_y0 += (MLO_OUT_PIX_TILE1 *MLO_FILTER_STRIDE1*MLO_N_OUT_FOLDS1), gbl_in_scan_off0 += (MLO_OUT_PIX_TILE1 *MLO_FILTER_STRIDE1*MLO_N_OUT_FOLDS1) * MLO_IN_CHANNEL_STRIDE, out_y += MLO_OUT_PIX_TILE1 *MLO_N_OUT_FOLDS1)
		{


			for (int i = 0; i < MLO_ACCUM_SZ; ++i)
			{
				pvt_accum[i] = 0;
			}


			// all input maps
			for (int c = 0, gbl_in_scan_off = gbl_in_scan_off0; c < MLO_N_INPUTS; ++c, gbl_in_scan_off += MLO_IN_CHANNEL_STRIDE)
			{
				for (int f_s = 0; f_s < MLO_FILTER_STRIDE1; ++f_s)
				{

					barrier(CLK_LOCAL_MEM_FENCE);

					// read weights by stride
					for (int w = lcl_id; w < MLO_WEI_LCL_SZ; w += MLO_GRP_SZ)
					{
						int k = iDiv(w, MLO_WEI_SZ);
						int t0 = iMod(w, k, MLO_WEI_SZ);
						int j = iDiv(t0, MLO_FILTER_SIZE0);
						int i = iMod(t0, j, MLO_FILTER_SIZE0);
						int wei_off = gbl_wei_off + k*MLO_WEI_BATCH_STRIDE + c*MLO_WEI_CHANNEL_STRIDE;

						if ((j*MLO_FILTER_STRIDE1 + f_s)*MLO_FILTER_SIZE0 + i < MLO_WEI_CHANNEL_STRIDE)
						{
							wei_mem[k*MLO_WEI_SZ + j*MLO_WEI_LCL_WIDTH + i] = weights[wei_off + (j*MLO_FILTER_STRIDE1 + f_s)*MLO_FILTER_SIZE0 + i];
#if 0
								if (ob==0 && k == 1)
								{
									printf("G:w: %d %d %d %d   %f %f\n",
//										lcl_id,
//										w,
//										f_s,
//										j,
//										i,
//										k_idx,
										k*MLO_WEI_SZ + j*MLO_WEI_LCL_WIDTH + i,
										gbl_wei_off,
										wei_off + (j*MLO_FILTER_STRIDE1 + f_s)*MLO_FILTER_SIZE0 + i,
										weights[wei_off + (j*MLO_FILTER_STRIDE1 + f_s)*MLO_FILTER_SIZE0 + i],
										wei_mem[k*MLO_WEI_SZ + j*MLO_WEI_LCL_WIDTH + i]
									);
								}

#endif

						}
						else
						{
							wei_mem[k*MLO_WEI_SZ + j*MLO_WEI_LCL_WIDTH + i] = 0;
						}
					}

					int n_reads = MLO_IN_LCL_HEIGHT; // ((ob == 0 && (f_s < MLO_FILTER_PAD1)) || (ob == get_local_size(0) - 1 && (MLO_FILTER_STRIDE1 - f_s) < MLO_FILTER_PAD1)) ? MLO_IN_LCL_HEIGHT - 1 : MLO_IN_LCL_HEIGHT;
					int lcl_scan = 0; // (ob == 0 && (f_s < MLO_FILTER_PAD1)) ? 1 : 0;

					// fetch input by stride
					for (int p4 = lcl_id, c_scan = 0;  p4 < MLO_N_IN_HORIZ_READS * n_reads * MLO_N_LCL_BATCHS;
						p4 += MLO_GRP_SZ)
					{
						int b = 0;
						int t0 = p4;
#if MLO_N_LCL_BATCHS > 1
						b = iDiv(p4, MLO_N_IN_HORIZ_READS * n_reads);
						t0 = iMod(p4, b, MLO_N_IN_HORIZ_READS * n_reads);
#endif
						c_scan = iDiv(t0, MLO_N_IN_HORIZ_READS);
						int c_pix4 = iMod(t0, c_scan, MLO_N_IN_HORIZ_READS);
						int in_scan = (c_scan + lcl_scan) * MLO_FILTER_STRIDE1 + f_s - MLO_FILTER_PAD1;

						for (int i = 0; i < MLO_READ_UNIT; ++i)
						{
							in_rd_data[i] = 0;
						}

						if (0 <= out_y*MLO_FILTER_STRIDE1 + in_scan && out_y*MLO_FILTER_STRIDE1 + in_scan < MLO_IN_HEIGHT)
						{

							int gbl_off = gbl_in_scan_off + b*MLO_IN_BATCH_STRIDE + in_scan * MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT;
							// still problems with unaligned LDS access
#if MLO_IN_N_PIXS_OFF > 0
							if (c_pix4 == MLO_N_IN_HORIZ_READS - 1)
							{
								int i = 0;
								for (; i < MLO_IN_N_PIXS_OFF; ++i)
								{
									in_rd_data[i] = bot[gbl_off + i];
								}
//								for (; i < MLO_READ_UNIT; ++i)
//								{
//									in_rd_data[i] = 0;
//								}

							}
							else
#endif
							{
	
								for (int i = 0; i < MLO_READ_UNIT; ++i)
								{
									in_rd_data[i] = bot[gbl_off + i];
								}
							}

						}
						int lcl_off = (lcl_scan + c_scan)*MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 + c_pix4*MLO_READ_UNIT;
						for (int i = 0; i < MLO_READ_UNIT; ++i)
						{
							bot_mem[lcl_off + i] = in_rd_data[i];
						}

					}

					barrier(CLK_LOCAL_MEM_FENCE);

					// convolution
					// along vertical filter
					for (int m = 0; m < MLO_N_FILTER_SPLITS1; ++m)
					{
#if 0
						// select all vertical scans that matches the vertical filter tap 
						__private _FLOAT in_vals[MLO_N_LCL_BATCHS * ((MLO_OUT_PIX_TILE0 - 1)*MLO_FILTER_STRIDE0 + MLO_FILTER_SIZE0)];
						// read input values for this filter phase
						for (int bb = 0; bb < MLO_N_LCL_BATCHS; ++bb)
						{
							for (int i = 0; i < ((MLO_OUT_PIX_TILE0 - 1)*MLO_FILTER_STRIDE0 + MLO_FILTER_SIZE0); ++i)
							{
								in_vals[bb* ((MLO_OUT_PIX_TILE0 - 1)*MLO_FILTER_STRIDE0 + MLO_FILTER_SIZE0) + i]
									= bot_mem[bb*MLO_IN_LCL_SZ + (ex_row + m) * MLO_IN_LCL_WIDTH + ex_pix*MLO_FILTER_STRIDE0 + i];
							}

						}
#endif
// only for 11 
						__private _FLOAT wei_vals[MLO_N_LCL_OUT_MAPS*MLO_N_FILTER_SPLITS0];
						// first 2 splits
						int l;
						for (l = 0; l <  MLO_FILTER_STRIDE0 - 1; ++l)
						{
// read all weights
							for (int k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
							{
								for (int i = 0; i < MLO_N_FILTER_SPLITS0; ++i)
								{
									wei_vals[k*MLO_N_FILTER_SPLITS0 + i]
										= wei_mem[k*MLO_WEI_SZ + m*MLO_WEI_LCL_WIDTH + i*MLO_FILTER_STRIDE0 + l];
								}
							}

							// convolve 
							for (int bb = 0; bb < MLO_N_LCL_BATCHS; ++bb)
							{

								__private _FLOAT in_vals[MLO_N_LCL_BATCHS * (MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 -1)];
								for (int i = 0; i < (MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 1); ++i)
								{
									in_vals[bb*(MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 1) + i]
										= bot_mem[bb*MLO_IN_LCL_SZ + (ex_row + m) * MLO_IN_LCL_WIDTH + ex_pix*MLO_FILTER_STRIDE0 + i*MLO_FILTER_STRIDE0 + l];
								}

								for (int k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
								{
									for (int n = 0; n < MLO_OUT_PIX_TILE0; ++n)
									{

										for (int i = 0; i <  MLO_N_FILTER_SPLITS0; ++i)
										{
											_FLOAT in_val = in_vals[bb* (MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 1) + n + i];
											_FLOAT wei_val = wei_vals[k*MLO_N_FILTER_SPLITS0 + i];
											pvt_accum[(bb*MLO_N_LCL_OUT_MAPS + k) * MLO_OUT_PIX_TILE0 + n]
												+= wei_val * in_val;
#if 0
											if (wei_val * in_val != 0 && ib+b+bb == 0 && k_idx+k == 1 && out_y + ex_row == 0 && ex_pix + n == 0)
											{
												printf("G:c: %d %d %d %d %d %d %d %d %d %d %d %d  %f %f %f %f\n",
													f_s,
													out_y,
													ex_row,
													ex_pix,
													m,
													n,
													l,
													i,
													(out_y + ex_row)*MLO_FILTER_STRIDE1 + m*MLO_FILTER_STRIDE1 + f_s - MLO_FILTER_PAD1, // actual input vertical position
													(ex_pix + n)*MLO_FILTER_STRIDE0 + l*MLO_FILTER_STRIDE0 + i - MLO_FILTER_PAD0, // actual input horiz pos (assuming full scan is inside LDS)
													m*MLO_FILTER_STRIDE1 + f_s, // actual filter vet pos
													l*MLO_FILTER_STRIDE0 + i, // actual filter horiz pos
													pvt_accum[(bb*MLO_N_LCL_OUT_MAPS + k) * MLO_OUT_PIX_TILE0 + n],
													wei_val * in_val,
													wei_val,
													in_val
												);
											}

#endif

										}
									}
								}
							} // b
						} // l
// 3d
						{
// read all weights
							for (int k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
							{
								for (int i = 0; i < MLO_N_FILTER_SPLITS0; ++i)
								{
									wei_vals[k*MLO_N_FILTER_SPLITS0 + i]
										= wei_mem[k*MLO_WEI_SZ + m*MLO_WEI_LCL_WIDTH + i*MLO_FILTER_STRIDE0 + l];
								}
							}

							// convolve 
							for (int bb = 0; bb < MLO_N_LCL_BATCHS; ++bb)
							{

								__private _FLOAT in_vals[MLO_N_LCL_BATCHS * (MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 -1)];
								for (int i = 0; i < (MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 1); ++i)
								{
									in_vals[bb*(MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 1) + i]
										= bot_mem[bb*MLO_IN_LCL_SZ + (ex_row + m) * MLO_IN_LCL_WIDTH + ex_pix*MLO_FILTER_STRIDE0 + i*MLO_FILTER_STRIDE0 + l];
								}

								for (int k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
								{
									for (int n = 0; n < MLO_OUT_PIX_TILE0; ++n)
									{

										for (int i = 0; i <  MLO_N_FILTER_SPLITS0; ++i)
										{
											_FLOAT in_val = in_vals[bb* (MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 1) + n + i];
											_FLOAT wei_val = wei_vals[k*MLO_N_FILTER_SPLITS0 + i];
											pvt_accum[(bb*MLO_N_LCL_OUT_MAPS + k) * MLO_OUT_PIX_TILE0 + n]
												+= wei_val * in_val;
#if 0
											if (wei_val * in_val != 0 && ib+b+bb == 0 && k_idx+k == 1 && out_y + ex_row == 0 && ex_pix + n == 0)
											{
												printf("G:c: %d %d %d %d %d %d %d %d %d %d %d %d  %f %f %f %f\n",
													f_s,
													out_y,
													ex_row,
													ex_pix,
													m,
													n,
													l,
													i,
													(out_y + ex_row)*MLO_FILTER_STRIDE1 + m*MLO_FILTER_STRIDE1 + f_s - MLO_FILTER_PAD1, // actual input vertical position
													(ex_pix + n)*MLO_FILTER_STRIDE0 + l*MLO_FILTER_STRIDE0 + i - MLO_FILTER_PAD0, // actual input horiz pos (assuming full scan is inside LDS)
													m*MLO_FILTER_STRIDE1 + f_s, // actual filter vet pos
													l*MLO_FILTER_STRIDE0 + i, // actual filter horiz pos
													pvt_accum[(bb*MLO_N_LCL_OUT_MAPS + k) * MLO_OUT_PIX_TILE0 + n],
													wei_val * in_val,
													wei_val,
													in_val
												);
											}

#endif

										}
									}
								}
							} // b
						} // l


					} // m

				} // f_s

			} // c


			barrier(CLK_LOCAL_MEM_FENCE);

			for (int bb = 0; bb < MLO_N_LCL_BATCHS && (out_y + ex_row) < MLO_OUT_HEIGHT; ++bb)
			{
				for (int k = 0; k < MLO_N_LCL_OUT_MAPS && (k_idx + k) < MLO_N_OUTPUTS; ++k)
				{
					// write out 
					// inputs are outputs
					int out_off = (ib + b + bb) * MLO_OUT_BATCH_STRIDE + (k_idx + k) * MLO_OUT_CHANNEL_STRIDE + (out_y + ex_row) *MLO_OUT_STRIDE + ex_pix;
					for (int i = 0; i < MLO_OUT_PIX_TILE0 && ex_pix + i < MLO_OUT_WIDTH; ++i)
					{
						top[out_off + i] = pvt_accum[(bb*MLO_N_LCL_OUT_MAPS + k) * MLO_OUT_PIX_TILE0 + i];
					}
				}
			}

		} // ob

	} // b
}
