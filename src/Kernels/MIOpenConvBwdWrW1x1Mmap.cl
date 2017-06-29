/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#define _FLOAT float
#define _FLOAT2 float2
#define _FLOAT4 float4
#define _FLOAT8 float8

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F /* max value */
#endif

#define UNUSED __attribute__((__unused__))

#ifndef MLO_N_PIXS_OFF
#define MLO_N_PIXS_OFF 0
#endif

__attribute__((always_inline)) uint iDiv(uint v, uint d)
{
    uint r = (uint)((float)v * (1.0f / (float)d) + 0.00001f);
    return (r);
}

__attribute__((always_inline)) uint iMod(uint v, uint u, uint d)
{
    uint r = v - mul24((uint)u, (uint)d);
    return (r);
}

inline void ReduceKernel(__local _FLOAT* lcl_blob,
                         __private _FLOAT* weights_accum,
                         uint lcl_id,
                         uint scan_lcl,
                         uint sum_stride,
                         uint unit_len,
                         UNUSED bool debug)
{

    for(uint j = (sum_stride >> 1); j > 0; j >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if(scan_lcl < j)
        {
            for(uint i = 0; i < unit_len; ++i)
            {
                weights_accum[i] += lcl_blob[(lcl_id + j) * unit_len + i];

                lcl_blob[lcl_id * unit_len + i] = weights_accum[i];
            }
        }
    }
}

static inline void
Kahan_summation(__private _FLOAT* __restrict sum, __private _FLOAT* __restrict c, _FLOAT v)
{
    _FLOAT y = v - *c;   // So far, so good: c is zero.
    _FLOAT t = *sum + y; // Alas, sum is big, y small, so low-order digits of y are lost.
    *c       = (t - *sum) -
         y;   //(t - sum) recovers the high-order part of y; subtracting y recovers -(low part of y)
    *sum = t; // Algebraically, c should always be zero. Beware eagerly optimising compilers!
}

/*
        group cooperative read
        read by MLO_READ_UNIT
        handle out of range both horizontally and vertically (by fixed number of veryical reads)

        no guard against number of inputs
*/
__attribute__((always_inline)) void readData(uint n,
                                             int gbl_data_off,
                                             uint gbl_data_stride,
                                             uint map_stride,
                                             int map_base,
                                             int map_limit,
                                             const __global _FLOAT* __restrict g_data,
                                             __private _FLOAT* __restrict p_data,
                                             bool last_pixel)
{
    for(uint j = 0; j < n; ++j)
    {
        int gbl_data_off0 = (j * map_stride + map_base < map_limit)
                                ? gbl_data_off + j * map_stride * gbl_data_stride
                                : 0;
        const __global _FLOAT* g_data_p = &g_data[gbl_data_off0];

#if MLO_N_PIXS_OFF > 0

        if(last_pixel)
        {
            for(uint i = 0; i < MLO_N_PIXS_OFF; ++i)
            {

                p_data[j * MLO_READ_UNIT + i] = g_data_p[i];
            }

            for(uint i = MLO_IN_N_PIXS_OFF; i < MLO_READ_UNIT; ++i)
            {
                p_data[j * MLO_READ_UNIT + i] = 0;
            }
        }
        else
#else
        (void)last_pixel;
#endif
        {
            for(uint i = 0; i < MLO_READ_UNIT; ++i)
            {
                p_data[j * MLO_READ_UNIT + i] = g_data_p[i];
            }
        }

    } // for (int j = 0; j < n; ++j)
}

__attribute__((always_inline)) void readDataFlex(uint n,
                                                 uint gbl_data_off,
                                                 uint gbl_data_stride,
                                                 uint map_stride,
                                                 uint map_base,
                                                 uint map_limit,
                                                 const __global _FLOAT* __restrict g_data,
                                                 __local _FLOAT* __restrict l_data)
{
#ifdef __AMDGCN__
#pragma unroll 2
#endif
    for(uint l = get_local_id(0); l < n * map_stride * MLO_MAP_WK_SZ; l += MLO_GRP_SZ)
    {

        uint r  = 0;
        uint k0 = l;
#if MLO_N_LCL_OUT_MAPS > 1
        r  = iDiv(l, map_stride * MLO_MAP_WK_SZ); // maps row
        k0 = iMod(l, r, map_stride * MLO_MAP_WK_SZ);
#endif

#if(MLO_MAP_WK_SZ) & (MLO_MAP_WK_SZ - 1)

        uint k  = iDiv(k0, MLO_MAP_WK_SZ);    // map column
        uint p4 = iMod(k0, k, MLO_MAP_WK_SZ); // pixel block
#else
        uint k  = (k0 / MLO_MAP_WK_SZ);       // map column
        uint p4 = (k0 & (MLO_MAP_WK_SZ - 1)); // pixel block

#endif

        uint gbl_data_off0 =
            (r * map_stride + k + map_base < map_limit)
                ? gbl_data_off + (r * map_stride + k) * gbl_data_stride + p4 * MLO_READ_UNIT
                : 0;
        const __global _FLOAT* g_data_p = &g_data[gbl_data_off0];
        __private _FLOAT p_data[MLO_READ_UNIT];

#if MLO_N_PIXS_OFF > 0

        bool last_pixel = (p4 == MLO_MAP_WK_SZ - 1);
        if(last_pixel)
        {
            for(uint i = 0; i < MLO_N_PIXS_OFF; ++i)
            {

                p_data[i] = g_data_p[i];
            }

            for(uint i = MLO_IN_N_PIXS_OFF; i < MLO_READ_UNIT; ++i)
            {
                p_data[i] = 0;
            }
        }
        else
#endif
        {
            for(uint i = 0; i < MLO_READ_UNIT; ++i)
            {
                p_data[i] = g_data_p[i];
            }
        }

        for(uint i = 0; i < MLO_READ_UNIT; ++i)
        {
            l_data[(k * MLO_MAP_WK_SZ + p4) * MLO_READ_UNIT + i] = p_data[i];
        }
    }
}

/*********************************************************************************************************
// wrw algorithm for large filters
// idea:
// read MLO_OUT_STACKS output maps into LDS
// read MLO_IN_STACKS per group, MLO_N_LCL_IN_MAPS per wk_item input maps

// alg

// loop over MLO_OUT_STACKS of output

// convolve with MLO_N_LCL_IN_MAPS per wk-item

// reduce with transform

// write out


**********************************************************************************************************/

__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2))) __kernel void
MIOpenCvBwdWrW(const __global _FLOAT* __restrict top_df,
               const __global _FLOAT* __restrict bot,
               __global _FLOAT* __restrict weights_df,
               UNUSED _FLOAT padding_val)
{
    // reduction memory.

    __local _FLOAT lcl_mem[MLO_LCL_MEM_SZ];
#if MLO_MAP_WK_SZ > 8
    __local _FLOAT* red_mem = lcl_mem;
#endif
    __local _FLOAT* proc_mem = lcl_mem;

    uint lcl_id = get_local_id(0);

    uint k_idx = get_group_id(0) * (MLO_OUT_STACKS * MLO_N_LCL_OUT_MAPS); // output map index base

    uint c_idx = get_group_id(1) * (MLO_IN_STACKS * MLO_N_LCL_IN_MAPS); // input map index based

    uint ib = get_group_id(2); // batch id

    uint gbl_in_off  = c_idx * MLO_IN_CHANNEL_STRIDE + ib * MLO_IN_BATCH_STRIDE;
    uint gbl_out_off = k_idx * MLO_OUT_CHANNEL_STRIDE + ib * MLO_OUT_BATCH_STRIDE;

#if MLO_MAP_WK_SZ & (MLO_MAP_WK_SZ - 1)
    // map id inside group
    uint m_idx = iDiv(lcl_id, MLO_MAP_WK_SZ);
    // read pixel inside the map
    uint p4 = iMod(lcl_id, m_idx, MLO_MAP_WK_SZ);
#else
    uint m_idx  = lcl_id / MLO_MAP_WK_SZ;
    uint p4     = lcl_id & (MLO_MAP_WK_SZ - 1);
#endif

    bool last_pixel = (p4 == MLO_MAP_WK_SZ - 1);

    gbl_in_off += m_idx * MLO_IN_CHANNEL_STRIDE + p4 * MLO_READ_UNIT;

#define MLO_TOP_DAT_SZ (MLO_N_LCL_OUT_MAPS * MLO_READ_UNIT)

    __private _FLOAT top_dat[MLO_TOP_DAT_SZ];

    for(uint i = 0; i < MLO_TOP_DAT_SZ; ++i)
    {
        top_dat[i] = 0;
    }

#define MLO_BOT_DAT_SZ (MLO_N_LCL_IN_MAPS * MLO_READ_UNIT)

    __private _FLOAT bot_dat[MLO_BOT_DAT_SZ];

    for(uint i = 0; i < MLO_BOT_DAT_SZ; ++i)
    {
        bot_dat[i] = 0;
    }

#define MLO_ACCUM_SZ (MLO_N_LCL_OUT_MAPS * MLO_N_LCL_IN_MAPS * MLO_OUT_STACKS)

    __private _FLOAT pvt_accum[MLO_ACCUM_SZ];

    for(uint i = 0; i < MLO_ACCUM_SZ; ++i)
    {
        pvt_accum[i] = 0;
    }

    for(uint i = lcl_id; i < MLO_LCL_MEM_SZ; i += MLO_GRP_SZ)
    {
        lcl_mem[i] = 0;
    }
    // over all batches

    for(uint b = 0; b < MLO_N_BATCH_LOOPS; ++b,
             gbl_in_off += MLO_N_LCL_BATCHS * MLO_IN_BATCH_STRIDE,
             gbl_out_off += MLO_N_LCL_BATCHS * MLO_OUT_BATCH_STRIDE)
    {

        uint gbl_in_scan_off  = gbl_in_off;
        uint gbl_out_scan_off = gbl_out_off;

        // read input maps
        readData(MLO_N_LCL_IN_MAPS,
                 gbl_in_scan_off,
                 MLO_IN_CHANNEL_STRIDE,
                 MLO_IN_STACKS,
                 (m_idx + c_idx),
                 MLO_N_INPUTS,
                 bot,
                 bot_dat,
                 last_pixel);

        // read output maps into LDS
        readDataFlex(MLO_N_LCL_OUT_MAPS,
                     gbl_out_scan_off,
                     MLO_OUT_CHANNEL_STRIDE,
                     MLO_OUT_STACKS,
                     (k_idx),
                     MLO_N_OUTPUTS,
                     top_df,
                     proc_mem);

        for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
        {

            barrier(CLK_LOCAL_MEM_FENCE);

            /*
            core processing loop
            bot - input
            top - output diff

            do convolution with all available input maps

            */

            for(uint n = 0; n < MLO_OUT_STACKS; ++n)
            {
                __private _FLOAT pvt_top[MLO_READ_UNIT];
                for(uint i = 0; i < MLO_READ_UNIT; ++i)
                {
                    pvt_top[i] = proc_mem[(n * MLO_MAP_WK_SZ + p4) * MLO_READ_UNIT +
                                          i]; // top_df[gbl_out_scan_off + (k*MLO_OUT_STACKS +
                                              // n0)*MLO_IN_CHANNEL_STRIDE + i];
                                              // //proc_mem[(n*MLO_MAP_WK_SZ + p4)*MLO_READ_UNIT +
                                              // i];
                }

                for(uint c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
                {
                    for(uint i = 0; i < MLO_READ_UNIT; ++i)
                    {
                        pvt_accum[(k * MLO_OUT_STACKS + n) * MLO_N_LCL_IN_MAPS + c] +=
                            bot_dat[c * MLO_READ_UNIT + i] * pvt_top[i];
#if 0

						if (k_idx + k*MLO_OUT_STACKS + n == 0 && c_idx + c * MLO_IN_STACKS + m_idx == 16 && m_idx < MLO_IN_STACKS && c_idx + m_idx + c*MLO_IN_STACKS < MLO_N_INPUTS && k_idx + k*MLO_OUT_STACKS + n < MLO_N_OUTPUTS)
						{
							printf("K:s: %d %d %d %d %d %d %d %d %f %f %f %f\n",
								MLO_OUT_STACKS,
								n,
								m_idx,
								gbl_out_scan_off,
								get_group_id(1),
								get_local_id(0),
								k,
								c,
								pvt_accum[(n* MLO_N_LCL_OUT_MAPS + k)*MLO_N_LCL_IN_MAPS + c],
								bot_dat[c*MLO_READ_UNIT + i] * pvt_top[i],
								bot_dat[c*MLO_READ_UNIT + i],
								pvt_top[i]
							);
						}
#endif
                    }
                }
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

#if MLO_MAP_WK_SZ > 8
    // transpose
    uint red_base_off = m_idx * MLO_MAP_WK_SZ * MLO_ACCUM_SZ;
    for(uint l = 0; l < MLO_ACCUM_SZ; ++l)
    {
        // write data
        red_mem[red_base_off + p4 * MLO_ACCUM_SZ + l] = pvt_accum[l];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // do final summation

    __private _FLOAT final_sum[1] = {0};

#if MLO_ACCUM_SZ & (MLO_ACCUM_SZ - 1)
    uint new_m_idx = iDiv(lcl_id, MLO_ACCUM_SZ);
    uint new_p4    = iMod(lcl_id, new_m_idx, MLO_ACCUM_SZ);
#else
    uint new_m_idx = (lcl_id / MLO_ACCUM_SZ);
    uint new_p4    = (lcl_id & (MLO_ACCUM_SZ - 1));

#endif

    uint new_red_base_off = new_m_idx * MLO_MAP_WK_SZ * MLO_ACCUM_SZ;

    for(uint s = 0; s < MLO_MAP_WK_SZ; ++s)
    {
        final_sum[0] += red_mem[new_red_base_off + s * MLO_ACCUM_SZ + new_p4];
    }

    // write out
    // inputs are outputs
    uint wei_df_off = ((ib * MLO_N_OUTPUTS + k_idx) * (uint)MLO_WEI_BATCH_STRIDE) +
                      ((c_idx + new_m_idx) * MLO_WEI_CHANNEL_STRIDE);

#if MLO_N_LCL_IN_MAPS & (MLO_N_LCL_IN_MAPS - 1)

    uint n = iDiv(new_p4, MLO_N_LCL_IN_MAPS);
    uint c = iMod(new_p4, n, MLO_N_LCL_IN_MAPS);
#else

    uint n = (new_p4 / MLO_N_LCL_IN_MAPS);
    uint c = (new_p4 & (MLO_N_LCL_IN_MAPS - 1));
#endif

    uint k = 0;

    if(new_m_idx < MLO_IN_STACKS && k_idx + k * MLO_OUT_STACKS + n < MLO_N_OUTPUTS &&
       c_idx + new_m_idx + c * MLO_IN_STACKS < MLO_N_INPUTS)
    {
        weights_df[wei_df_off + (k * MLO_OUT_STACKS + n) * MLO_WEI_BATCH_STRIDE +
                   c * MLO_IN_STACKS * MLO_WEI_CHANNEL_STRIDE] = final_sum[0];
    }
#else

#if 0 // MLO_IN_WIDTH > 24
// logar reduction
	for (uint i = lcl_id; i < MLO_LCL_MEM_SZ; i += MLO_GRP_SZ)
	{
		red_mem[i] = 0;
	}

	int red_base_off = (m_idx >= MLO_OUT_STACKS) ? MLO_LCL_MEM_SZ : m_idx * MLO_POW2_MAP_WK_SZ;
	// final summation over each filter row
	for (uint l = 0; l < MLO_ACCUM_SZ; ++l)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		// write data
		red_mem[red_base_off + p4] = pvt_accum[l];

		// barrier inside
		ReduceKernel(&red_mem[red_base_off], &pvt_accum[l], p4, p4, MLO_POW2_MAP_WK_SZ, 1, false);

	}


	barrier(CLK_LOCAL_MEM_FENCE);
#else
    // direct reduction
    for(uint l = 0; l < MLO_ACCUM_SZ; ++l)
    {
        lcl_mem[lcl_id] = pvt_accum[l];
        barrier(CLK_LOCAL_MEM_FENCE);
        if(p4 == 0)
        {
            for(uint i = 1; i < MLO_MAP_WK_SZ; ++i)
            {
                pvt_accum[l] += lcl_mem[lcl_id + i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

#endif

    // write out
    // inputs are outputs
    uint wei_df_off = ((ib * MLO_N_OUTPUTS + k_idx) * (int)MLO_WEI_BATCH_STRIDE) +
                      ((c_idx + m_idx) * MLO_WEI_CHANNEL_STRIDE);

    if(p4 == 0)
    {
        for(uint n = 0; n < MLO_OUT_STACKS; ++n)
        {
            for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
            {
                for(uint c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
                {
                    if(m_idx < MLO_IN_STACKS && k_idx + k * MLO_OUT_STACKS + n < MLO_N_OUTPUTS &&
                       c_idx + m_idx + c * MLO_IN_STACKS < MLO_N_INPUTS)
                    {
                        weights_df[wei_df_off + (k * MLO_OUT_STACKS + n) * MLO_WEI_BATCH_STRIDE +
                                   c * MLO_IN_STACKS * MLO_WEI_CHANNEL_STRIDE] =
                            pvt_accum[(k * MLO_OUT_STACKS + n) * MLO_N_LCL_IN_MAPS + c];
                    }
                }
            }
        }
    }

#endif
}

// final reduction kernel
// add filters over batches
__attribute__((reqd_work_group_size(MLO_UT_GRP_SZ0, 1, 1))) __kernel void
MIOpenCvBwdWrW_rdc(const __global _FLOAT* __restrict weight_df_tmp,
                   __global _FLOAT* __restrict weights_df)
{
    uint gbl_id   = get_global_id(0);
    uint wei_idx0 = gbl_id * MLO_UT_READ_UNIT;

#if MLO_WEI_CHANNEL_STRIDE & (MLO_WEI_CHANNEL_STRIDE - 1)
    uint wei_blk_idx = iDiv(wei_idx0, MLO_WEI_CHANNEL_STRIDE);
    uint wei_idx     = iMod(wei_idx0, wei_blk_idx, MLO_WEI_CHANNEL_STRIDE);
#else
    uint wei_blk_idx = wei_idx0 / MLO_WEI_CHANNEL_STRIDE;
    uint wei_idx     = wei_idx0 & (MLO_WEI_CHANNEL_STRIDE - 1);
#endif

    _FLOAT pvt_accum_wei[MLO_UT_READ_UNIT];
    for(uint i = 0; i < MLO_UT_READ_UNIT; ++i)
    {
        pvt_accum_wei[i] = 0;
    }

    uint batch_loop = (MLO_BATCH_SZ + (MLO_N_BATCH_LOOPS * MLO_N_LCL_BATCHS) - 1) /
                      (MLO_N_BATCH_LOOPS * MLO_N_LCL_BATCHS);
    for(uint i = 0; i < batch_loop; ++i)
    {
        for(uint j = 0; j < MLO_UT_READ_UNIT; ++j)
        {
            pvt_accum_wei[j] += weight_df_tmp[(wei_blk_idx * MLO_WEI_CHANNEL_STRIDE +
                                               i * MLO_N_OUTPUTS * MLO_WEI_BATCH_STRIDE) +
                                              wei_idx + j];
        }
    }

    for(uint j = 0; j < MLO_UT_READ_UNIT; ++j)
    {
        weights_df[wei_idx0 + j] = pvt_accum_wei[j];
    }
}
