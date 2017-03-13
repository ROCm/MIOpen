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



#define MLO_N_OUT_HORIZ_READS ((MLO_OUT_WIDTH + MLO_IN_TILE0 - 1) / MLO_IN_TILE0)
#define MLO_N_SPANS_PER_SCAN (MLO_N_OUT_HORIZ_READS)
#define MLO_N_OUT_HORIZ_PIX_READS (MLO_N_OUT_HORIZ_READS * MLO_IN_TILE0)
#define MLO_OUT_N_PIXS_OFF (MLO_OUT_WIDTH - ((MLO_OUT_WIDTH / MLO_IN_TILE0)*MLO_IN_TILE0))
#define MLO_N_OUT_VERTICAL_READS (MLO_FILTER_SIZE1)



#define MLO_IN_VERT_READS (MLO_IN_HEIGHT)
// there is an assumption that the scanline fits into LDS
#define MLO_N_IN_HORIZ_PIX_READS (MLO_IN_WIDTH) 
#define MLO_N_IN_HORIZ_READS ((MLO_N_IN_HORIZ_PIX_READS + MLO_READ_UNIT - 1) / MLO_READ_UNIT)
#define MLO_IN_N_PIXS_OFF  (MLO_N_IN_HORIZ_PIX_READS - (MLO_N_IN_HORIZ_PIX_READS/MLO_READ_UNIT)*MLO_READ_UNIT)
#define MLO_IN_LCL_WIDTH (MLO_N_IN_HORIZ_READS * MLO_READ_UNIT + 2* MLO_FILTER_PAD0)
#define MLO_IN_LCL_HEIGHT MLO_IN_VERT_READS
#define MLO_IN_LCL_SZ (MLO_IN_LCL_WIDTH * MLO_IN_LCL_HEIGHT)
#define MLO_TOTAL_IN_LCL_SZ (MLO_N_LCL_BATCHS*MLO_N_LCL_IN_MAPS*MLO_IN_LCL_SZ)

#define MLO_WEI_LCL_SZ (MLO_GRP_SZ * MLO_FILTER_SIZE0)
#if MLO_TOTAL_IN_LCL_SZ > MLO_WEI_LCL_SZ
#define MLO_LCL_SZ (MLO_TOTAL_IN_LCL_SZ)
#else
#define MLO_LCL_SZ (MLO_WEI_LCL_SZ)
#endif


#define MLO_HW_WAVE_ID_SETTING 1
#if MLO_HW_WAVE_ID_SETTING //&& MLO_COMPILER_AMD_OPENCL_HSAIL==1

// FIXME Conduct enabling from the host code.
extern __attribute__((const)) uint __hsail_get_dynwave_id(void);
#endif
static inline int getWaveId()
{
	int wave_id = 0;

#if MLO_HW_WAVE_ID_SETTING //&& MLO_COMPILER_AMD_OPENCL_HSAIL==1

	wave_id = __hsail_get_dynwave_id();
	wave_id &= MLO_N_WAVES_MASK;
#if 0
#elif MLO_HW_WAVE_ID_SETTING && MLO_COMPILER_AMD_OPENCL_LC==1 && MLO_GRP_SZ1==1 && MLO_GRP_SZ2==1 && (MLO_GRP_SZ % (1 << MLO_LG2_WAVE_SZ))==0
	// (local_id/wavesize) has the same value in all workitems.
	// Make it scalar to enable scalarization optimizations.
	wave_id = __llvm_amdgcn_readfirstlane((uint)(get_local_id(0) >> MLO_LG2_WAVE_SZ));
	// Alternate implementation:
	//__asm__ ("v_readfirstlane_b32 %0, %1" : "=s" (wave_id) : "v" ((int)(get_local_id(0) >> MLO_LG2_WAVE_SZ)) );
#endif
#else
	wave_id = (get_local_id(0) >> MLO_LG2_WAVE_SZ);
#endif
	return(wave_id);
}


inline int gePhysLocalId()
{
	int lcl_wave_id = get_local_id(0) - ((get_local_id(0) >> MLO_LG2_WAVE_SZ) << MLO_LG2_WAVE_SZ);
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

inline void ReduceKernel(__local _FLOAT * lcl_blob, __private _FLOAT *weights_accum, int lcl_id, int scan_lcl, int sum_stride, int unit_len, bool debug)
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



static inline void  Kahan_summation(__private _FLOAT *sum, __private _FLOAT * c, _FLOAT v)
{
	_FLOAT y = v - *c;    //So far, so good: c is zero.
	_FLOAT t = *sum + y;         //Alas, sum is big, y small, so low-order digits of y are lost.
	*c = (t - *sum) - y;   //(t - sum) recovers the high-order part of y; subtracting y recovers -(low part of y)
	*sum = t;             //Algebraically, c should always be zero. Beware eagerly optimising compilers!
}

/*********************************************************************************************************
// wrw algorithm for 7x7 and smallee
// idea:
// layout
// each wave keeps WAVE_SZ output maps
// 1 input map
// alg
// for  all batches

// read output maps in VGPRS 1 map per wk_item
// for number of waves
// read full input map in SGPRs
// convolve internal pixels
// convolve border pixels
// exchange input maps
// rof
// rof

// write out 


**********************************************************************************************************/

__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2)))
__kernel void MLOpenCvBwdWrW_7x7(
	const __global _FLOAT * top_df,
	__constant _FLOAT * bot,
	__global _FLOAT * weights_df,
#if MLO_CONV_BIAS
	__global _FLOAT * bias_df,
#endif
	_FLOAT padding_val
)
{


	// guarnteeing an uniformity over a wave
	int wave_id = getWaveId();
	int lcl_id = get_local_id(0);
	int lcl_wv_id = gePhysLocalId();



	int c_idx_base = get_group_id(0); // input map index base

	int o_idx_base = get_group_id(1); // output map index base

	int ib_base = get_group_id(2); //batch index

	int ib = ib_base*MLO_N_LCL_BATCHS;

	int c_idx = c_idx_base * MLO_N_WAVES* MLO_N_LCL_IN_MAPS; // input map index

	int o_idx = o_idx_base * MLO_N_WAVES * MLO_HW_WAVE_SZ* MLO_N_LCL_OUT_MAPS; // output map index

	int gbl_in_off = c_idx * MLO_IN_CHANNEL_STRIDE + ib * MLO_IN_BATCH_STRIDE;
	int gbl_out_off = o_idx * MLO_OUT_CHANNEL_STRIDE + ib * MLO_OUT_BATCH_STRIDE;




// out address based on wave and batch
	int out_map = lcl_wv_id;

	gbl_out_off +=  (wave_id * MLO_HW_WAVE_SZ  + out_map) * MLO_IN_CHANNEL_STRIDE;


#define MLO_DAT_SZ (MLO_IN_HEIGHT * MLO_IN_WIDTH)

	__private _FLOAT top_dat[MLO_DAT_SZ];

/*
	for (int i = 0; i < MLO_DAT_SZ; ++i)
	{
		top_dat[i] = 0;
	}

*/
	__private _FLOAT bot_dat[MLO_DAT_SZ];

/*
	for (int i = 0; i < MLO_DAT_SZ; ++i)
	{
		bot_dat[i] = 0;
	}
*/

#define MLO_ACCUM_SZ (MLO_N_WAVES * MLO_N_LCL_OUT_MAPS * MLO_FILTER_SIZE1*MLO_FILTER_SIZE0)

	__private _FLOAT pvt_accum[MLO_ACCUM_SZ];

	for (int i = 0; i < MLO_ACCUM_SZ; ++i)
	{
		pvt_accum[i] = 0;
	}


	// over all batches

	for (int b = 0;
		b < MLO_N_BATCH_LOOPS;
		++b,
		gbl_in_off += MLO_N_LCL_BATCHS*MLO_IN_BATCH_STRIDE,
		gbl_out_off += MLO_N_LCL_BATCHS*MLO_OUT_BATCH_STRIDE
		)
	{


		// read output map

		for (int j = 0; j < MLO_DAT_SZ; ++j)
		{
			top_dat[j] = top_df[gbl_out_off + j];
		}
		for (int w = 0; w < MLO_N_WAVES; ++w)
		{
			// in address based on wave_id

			int wave_id_r = w; //((wave_id + w) < MLO_N_WAVES) ?  (wave_id + w) : (wave_id + w) - MLO_N_WAVES;
			int gbl_in_off_r = gbl_in_off + wave_id_r * MLO_IN_CHANNEL_STRIDE;

			// read input map presumably to SGPRs

			for (int j = 0; j < MLO_DAT_SZ; ++j)
			{
				bot_dat[j] = bot[gbl_in_off_r + j];
			}

// convolve
// internals
			for (int m = 0; m < MLO_FILTER_SIZE1; ++m)
			{
				for (int l = 0; l < MLO_FILTER_SIZE0; ++l)
				{
					for (int j = MLO_FILTER_PAD1; j < MLO_IN_HEIGHT - MLO_FILTER_PAD1; ++j)
					{
						for (int i = MLO_FILTER_PAD0; i < MLO_IN_WIDTH - MLO_FILTER_PAD0; ++i)
						{

							pvt_accum[(wave_id_r * MLO_FILTER_SIZE1 + m) * MLO_FILTER_SIZE0 + l]
								+= top_dat[j*MLO_IN_WIDTH + i] * bot_dat[(j + m - MLO_FILTER_PAD1) *MLO_IN_WIDTH + i + l - MLO_FILTER_PAD0];
						}  // i
					} // j
 				} // l

			} // m


#if 1

// top
			for (int m = MLO_FILTER_PAD1; m < MLO_FILTER_SIZE1; ++m)
			{
				for (int l = 0; l < MLO_FILTER_SIZE0; ++l)
				{
					int j = 0;
					{
						for (int i = MLO_FILTER_PAD0; i < MLO_IN_WIDTH - MLO_FILTER_PAD0; ++i)
						{

							pvt_accum[(wave_id_r * MLO_FILTER_SIZE1 + m) * MLO_FILTER_SIZE0 + l]
								+= top_dat[j*MLO_IN_WIDTH + i] * bot_dat[(j + m - MLO_FILTER_PAD1)*MLO_IN_WIDTH + i + l - MLO_FILTER_PAD0];
						}  // i
					} // j
				} // l

			} // m

// bot
			for (int m = 0; m < MLO_FILTER_SIZE1 - MLO_FILTER_PAD1; ++m)
			{
				for (int l = 0; l < MLO_FILTER_SIZE0; ++l)
				{
					int j = MLO_IN_HEIGHT - 1;
					{
						for (int i = MLO_FILTER_PAD0; i < MLO_IN_WIDTH - MLO_FILTER_PAD0; ++i)
						{

							pvt_accum[(wave_id_r * MLO_FILTER_SIZE1 + m) * MLO_FILTER_SIZE0 + l]
								+= top_dat[j*MLO_IN_WIDTH + i] * bot_dat[(j + m)*MLO_IN_WIDTH + i + l - MLO_FILTER_PAD0];
						}  // i
					} // j
				} // l

			} // m


// left 
			for (int m = 0; m < MLO_FILTER_SIZE1; ++m)
			{
				for (int l = MLO_FILTER_PAD0; l < MLO_FILTER_SIZE0; ++l)
				{
					for (int j = MLO_FILTER_PAD1; j < MLO_IN_HEIGHT - MLO_FILTER_PAD1; ++j)
					{
						int i = 0;
						{
							pvt_accum[(wave_id_r * MLO_FILTER_SIZE1 + m) * MLO_FILTER_SIZE0 + l]
								+= top_dat[j*MLO_IN_WIDTH + i] * bot_dat[(j + m - MLO_FILTER_PAD1) *MLO_IN_WIDTH + i + l - MLO_FILTER_PAD0];
						}  // i
					} // j
				} // l

			} // m

// right
			for (int m = 0; m < MLO_FILTER_SIZE1; ++m)
			{
				for (int l = 0; l < MLO_FILTER_SIZE0 - MLO_FILTER_PAD0; ++l)
				{
					for (int j = MLO_FILTER_PAD1; j < MLO_IN_HEIGHT - MLO_FILTER_PAD1; ++j)
					{
						int i = MLO_IN_WIDTH -1;
						{
							pvt_accum[(wave_id_r * MLO_FILTER_SIZE1 + m) * MLO_FILTER_SIZE0 + l]
								+= top_dat[j*MLO_IN_WIDTH + i] * bot_dat[(j + m - MLO_FILTER_PAD1) *MLO_IN_WIDTH + i + l];
						}  // i
					} // j
				} // l

			} // m

// top - left
			for (int m = MLO_FILTER_PAD1; m < MLO_FILTER_SIZE1; ++m)
			{
				for (int l = MLO_FILTER_PAD0; l < MLO_FILTER_SIZE0; ++l)
				{
					int j = 0;
					{
						int i = 0;
						{

							pvt_accum[(wave_id_r * MLO_FILTER_SIZE1 + m) * MLO_FILTER_SIZE0 + l]
								+= top_dat[j*MLO_IN_WIDTH + i] * bot_dat[(j + m - MLO_FILTER_PAD1)*MLO_IN_WIDTH + i + l - MLO_FILTER_PAD0];
						}  // i
					} // j
				} // l

			} // m

// top - right
			for (int m = MLO_FILTER_PAD1; m < MLO_FILTER_SIZE1; ++m)
			{
				for (int l = 0; l < MLO_FILTER_SIZE0 - MLO_FILTER_PAD0; ++l)
				{
					int j = 0;
					{
						int i = MLO_IN_WIDTH - 1;
						{

							pvt_accum[(wave_id_r * MLO_FILTER_SIZE1 + m) * MLO_FILTER_SIZE0 + l]
								+= top_dat[j*MLO_IN_WIDTH + i] * bot_dat[(j + m - MLO_FILTER_PAD1)*MLO_IN_WIDTH + i + l];
						}  // i
					} // j
				} // l

			} // m

// bot - left
			for (int m = 0; m < MLO_FILTER_SIZE1 - MLO_FILTER_PAD1; ++m)
			{
				for (int l = MLO_FILTER_PAD0; l < MLO_FILTER_SIZE0; ++l)
				{
					int j = MLO_IN_WIDTH - 1;
					{
						int i = 0;
						{

							pvt_accum[(wave_id_r * MLO_FILTER_SIZE1 + m) * MLO_FILTER_SIZE0 + l]
								+= top_dat[j*MLO_IN_WIDTH + i] * bot_dat[(j + m)*MLO_IN_WIDTH + i + l - MLO_FILTER_PAD0];
						}  // i
					} // j
				} // l

			} // m

// bot - right
			for (int m = 0; m < MLO_FILTER_SIZE1 - MLO_FILTER_PAD1; ++m)
			{
				for (int l = 0; l < MLO_FILTER_SIZE0 - MLO_FILTER_PAD0; ++l)
				{
					int j = MLO_IN_HEIGHT - 1;
					{
						int i = MLO_IN_WIDTH - 1;
						{

							pvt_accum[(wave_id_r * MLO_FILTER_SIZE1 + m) * MLO_FILTER_SIZE0 + l]
								+= top_dat[j*MLO_IN_WIDTH + i] * bot_dat[(j + m)* MLO_IN_WIDTH + i + l];
						}  // i
					} // j
				} // l

			} // m


#endif



		} // w

	} // 	for (int b = 0;


// output 
// inputs are outputs
// TODO : for more than 1 input
	int c = 0;

	int wei_df_off = (wave_id * MLO_HW_WAVE_SZ + out_map) * MLO_WEI_BATCH_STRIDE + c_idx * MLO_WEI_CHANNEL_STRIDE;

	for (int w = 0; w < MLO_N_WAVES && (wave_id * MLO_HW_WAVE_SZ + out_map) < MLO_N_OUTPUTS && (c_idx + w) < MLO_N_INPUTS; ++w)
	{
		for (int i = 0; i < (MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0); ++i)
		{
			weights_df[wei_df_off + w * MLO_WEI_CHANNEL_STRIDE + i] = pvt_accum[w*(MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0) + i];
		}

	}


}


#if 0
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
			+= *(__global MLO_UT_READ_TYPE*)&weight_df_tmp[(wei_blk_idx * MLO_WEI_CHANNEL_STRIDE + i* MLO_N_OUTPUTS*MLO_WEI_BATCH_STRIDE)  + wei_idx];
	}

	*(__global MLO_UT_READ_TYPE*)&weights_df[wei_idx0] = *(MLO_UT_READ_TYPE*)pvt_accum_wei;

}

#endif