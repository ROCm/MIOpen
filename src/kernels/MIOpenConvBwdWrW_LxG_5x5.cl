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

#define PPCAT_NX(A, B) A##B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define TWO 2
#define FOUR 4
#define EIGHT 8

#if MIOPEN_USE_FP16 == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT half
#ifndef HALF_MAX
#define MAX_VAL 65504 /* max value */
#else
#define MAX_VAL HALF_MAX
#endif
#endif
#if MIOPEN_USE_FP32 == 1
#define _FLOAT float
#ifndef FLT_MAX
#define MAX_VAL 3.402823466e+38F /* max value */
#else
#define MAX_VAL FLT_MAX
#endif
#endif

#define _FLOAT2 PPCAT(_FLOAT, TWO)
#define _FLOAT4 PPCAT(_FLOAT, FOUR)
#define _FLOAT8 PPCAT(_FLOAT, EIGHT)

#define UNUSED __attribute__((__unused__))

#define DBG_OUT_OF_RNGE 0

#define MLO_N_OUT_HORIZ_READS ((MLO_OUT_WIDTH + MLO_IN_TILE0 - 1) / MLO_IN_TILE0)
#define MLO_N_SPANS_PER_SCAN (MLO_N_OUT_HORIZ_READS)
#define MLO_N_OUT_HORIZ_PIX_READS (MLO_N_OUT_HORIZ_READS * MLO_IN_TILE0)
#define MLO_OUT_N_PIXS_OFF (MLO_OUT_WIDTH - ((MLO_OUT_WIDTH / MLO_IN_TILE0) * MLO_IN_TILE0))
#define MLO_N_OUT_VERTICAL_READS (MLO_FILTER_SIZE1)

#define MLO_IN_VERT_READS (MLO_IN_HEIGHT)
// there is an assumption that the scanline fits into LDS
#define MLO_N_IN_HORIZ_PIX_READS (MLO_IN_WIDTH)
#define MLO_N_IN_HORIZ_READS ((MLO_N_IN_HORIZ_PIX_READS + MLO_READ_UNIT - 1) / MLO_READ_UNIT)
#define MLO_IN_N_PIXS_OFF \
    (MLO_N_IN_HORIZ_PIX_READS - (MLO_N_IN_HORIZ_PIX_READS / MLO_READ_UNIT) * MLO_READ_UNIT)
#define MLO_IN_LCL_WIDTH (MLO_N_IN_HORIZ_READS * MLO_READ_UNIT + 2 * MLO_FILTER_PAD0)
#define MLO_IN_LCL_HEIGHT MLO_IN_VERT_READS
#define MLO_IN_LCL_SZ (MLO_IN_LCL_WIDTH * MLO_IN_LCL_HEIGHT)
#define MLO_TOTAL_IN_LCL_SZ (MLO_N_LCL_BATCHS * MLO_N_LCL_IN_MAPS * MLO_IN_LCL_SZ)

#define MLO_WEI_LCL_SZ (MLO_GRP_SZ * MLO_FILTER_SIZE0)
#if MLO_TOTAL_IN_LCL_SZ > MLO_WEI_LCL_SZ
#define MLO_LCL_SZ (MLO_TOTAL_IN_LCL_SZ)
#else
#define MLO_LCL_SZ (MLO_WEI_LCL_SZ)
#endif

#include "math_ops.h"

/*
        group cooperative read
        read by MLO_READ_UNIT
        handle out of range both horizontally and vertically (by fixed number of veryical reads)

        no guard against number of inputs
*/
void readInput(int lcl_id,
               int gbl_in_scan_off,
               const __global _FLOAT* __restrict bot,
               __local _FLOAT* __restrict lcl_bot)
{
    for(int p4 = lcl_id; p4 < MLO_N_LCL_IN_MAPS * MLO_N_IN_HORIZ_READS * MLO_IN_VERT_READS;
        p4 += MLO_GRP_SZ)
    {
        __private _FLOAT in_rd_data[MLO_READ_UNIT];
        // TODO : more than 1 input
        int c      = 0;
        int c_scan = iDiv_legacy(p4, (MLO_N_IN_HORIZ_READS));

        int c_pix4 = iMod(p4, c_scan, (MLO_N_IN_HORIZ_READS));

        //		if (c < MLO_N_INPUTS)

        {
            int bot_off = gbl_in_scan_off + c * MLO_IN_CHANNEL_STRIDE + c_scan * MLO_IN_STRIDE +
                          c_pix4 * MLO_READ_UNIT;
#if MLO_IN_N_PIXS_OFF > 0

            if(c_pix4 == MLO_N_IN_HORIZ_READS - 1)
            {
                for(int i = 0; i < MLO_IN_N_PIXS_OFF; ++i)
                {

                    in_rd_data[i] = bot[bot_off + i];
#if DBG_OUT_OF_RNGE
                    if(bot_off + i >= MLO_IN_BATCH_STRIDE * MLO_BATCH_SZ)
                    {
                        printf("k:err:in-of-range\n");
                    }
#endif
                }

                for(int i = MLO_IN_N_PIXS_OFF; i < MLO_READ_UNIT; ++i)
                {
                    in_rd_data[i] = 0;
                }
            }
            else
#endif
            {

                for(int i = 0; i < MLO_READ_UNIT; ++i)
                {
                    in_rd_data[i] = bot[bot_off + i];
#if DBG_OUT_OF_RNGE
                    if(bot_off + i >= MLO_IN_BATCH_STRIDE * MLO_BATCH_SZ)
                    {
                        printf("k:err:in-of-range\n");
                    }
#endif
                }
            }

            // stack of inputs, each has 1 line
            for(int i = 0; i < MLO_READ_UNIT; ++i)
            {
                int lcl_in_off = c * MLO_IN_LCL_SZ + c_scan * MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 +
                                 c_pix4 * MLO_READ_UNIT + i;
                lcl_bot[lcl_in_off] = in_rd_data[i];
            }
        }

    } // for (int p4 = lcl_id; p4 < MLO_N_LCL_IN_MAPS * MLO_N_IN_HORIZ_READS * MLO_IN_VERT_READS;

    barrier(CLK_LOCAL_MEM_FENCE);
}

/*
        core processing loop
        bot - input, from local (1 span)
        top - output diff, from global (array of spans, filters vertical size)

        loop over filter vertical size

*/
void Processing(UNUSED int sc,
                int sc_lcl_off,
                int top_lim,
                int bot_lim,
                __private _FLOAT* __restrict pvt_accum,
                __local _FLOAT* __restrict lcl_bot,
                __private _FLOAT* __restrict top_dat)
{
    for(int l = top_lim; l >= bot_lim; --l)
    {
        for(int m = 0; m < MLO_IN_TILE0; ++m)
        {
            for(int n = 0; n < MLO_FILTER_SIZE0; ++n)
            {
                _FLOAT bot_val = lcl_bot[sc_lcl_off + n + m];
                _FLOAT top_val = top_dat[(top_lim - l) * MLO_IN_TILE0 + m];
                pvt_accum[l * MLO_FILTER_SIZE0 + n]
                    // each wk-item process an input
                    += bot_val * top_val;
#if 0
				if (bot_val * top_val != 0 && get_global_id(1) == 0 && get_global_id(2) == 0 && get_local_id(0) == 0 && l == 1 && n == 2)
				{
					printf("G: %d %d %f %f %f %f\n",
						sc,
						sc_lcl_off,
						pvt_accum[l*MLO_FILTER_SIZE0 + n],
						bot_val * top_val,
						bot_val,
						top_val
					);
				}
#endif
            }
        }
    }
}

void moveOutputUp(__private _FLOAT* __restrict top_dat)
{
    // move up output to reduce overfetch
    for(int j = 0; j < MLO_FILTER_SIZE1 - 1; ++j)
    {
        for(int i = 0; i < MLO_IN_TILE0; ++i)
        {
            int pvt_off_n      = j * MLO_IN_TILE0 + i;
            int pvt_off_o      = (j + 1) * MLO_IN_TILE0 + i;
            top_dat[pvt_off_n] = top_dat[pvt_off_o];
        }
    }
}

/*********************************************************************************************************
// wrw algorithm for large filters
// idea:
// split output scan-line on number of spans by the  MLO_IN_TILE0 (2 for example)
// 1 scan-line has ((MLO_OUT_WIDTH + MLO_IN_TILE0 - 1/MLO_IN_TILE0) spans
// group will process MLO_GRP_SZ/((MLO_OUT_WIDTH + MLO_IN_TILE0 - 1/MLO_IN_TILE0) output maps

// alg
// load a block of input map (or full map) into LDS
// loop
// read MLO_FILTER_SIZE1 number of spans from output map into VGPRs (for example 5 *2 = 10)
// read 1 input line for  maps into LDS
// accumulate

// accumulate all spans at the end
// start new loop for the next batch (if defined)
// write out


**********************************************************************************************************/

__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2))) __kernel void
MIOpenCvBwdWrW(const __global _FLOAT* __restrict top_df,
               const __global _FLOAT* __restrict bot,
               __global _FLOAT* __restrict weights_df,
#if MLO_CONV_BIAS
               __global _FLOAT* __restrict bias_df,
#endif
               UNUSED _FLOAT padding_val)
{

    // input/output tiles + reduce buffer

    __local _FLOAT lcl[(MLO_LCL_SZ)];
    __local _FLOAT* lcl_bot = lcl;

    int lcl_id = get_local_id(0);

    int c_idx_base = get_group_id(0); // input map index base

    int o_idx_base = get_group_id(1); // output map index base

    int ib_base = get_group_id(2);

    int ib = ib_base * MLO_N_LCL_BATCHS;

    int c_idx = c_idx_base * MLO_N_LCL_IN_MAPS; // input map index

    int o_idx = o_idx_base * (MLO_N_LCL_OUT_MAPS * MLO_OUT_STACKS); // output map index

    int gbl_in_off  = c_idx * MLO_IN_CHANNEL_STRIDE + ib * MLO_IN_BATCH_STRIDE;
    int gbl_out_off = o_idx * MLO_OUT_CHANNEL_STRIDE + ib * MLO_OUT_BATCH_STRIDE;
    // 1 span per wk_item, total scanline with MLO_N_SPANS_PER_SCAN spans
    // TODO: more than 1 input
    int o = iDiv_legacy(lcl_id, MLO_N_SPANS_PER_SCAN);
    //	bool scan_lead = (o*MLO_N_SPANS_PER_SCAN == lcl_id);
    int spn = iMod(lcl_id, o, MLO_N_SPANS_PER_SCAN);

    int lcl_bot_off     = spn * MLO_IN_TILE0;
    int out_wk_item_off = o * MLO_OUT_CHANNEL_STRIDE + lcl_bot_off;
    gbl_out_off += out_wk_item_off;
    // no output out of range
    gbl_out_off = (o_idx + o < MLO_N_OUTPUTS && o < MLO_OUT_STACKS) ? gbl_out_off : 0;

#define MLO_TOP_DAT_SZ (MLO_IN_TILE0 * MLO_FILTER_SIZE1)

    __private _FLOAT top_dat[MLO_TOP_DAT_SZ];

    for(int i = 0; i < MLO_TOP_DAT_SZ; ++i)
    {
        top_dat[i] = (_FLOAT)(0);
    }

#define MLO_ACCUM_SZ (MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0)

    __private _FLOAT pvt_accum[MLO_ACCUM_SZ];

    for(int i = 0; i < MLO_ACCUM_SZ; ++i)
    {
        pvt_accum[i] = (_FLOAT)(0);
    }

    // zero out LDS
    for(int i = lcl_id; i < (MLO_LCL_SZ); i += MLO_GRP_SZ)
    {
        lcl[i] = (_FLOAT)(0);
    }

    // over all batches

    for(int b = 0; b < MLO_N_BATCH_LOOPS; ++b,
            gbl_in_off += MLO_N_LCL_BATCHS * MLO_IN_BATCH_STRIDE,
            gbl_out_off += MLO_N_LCL_BATCHS * MLO_OUT_BATCH_STRIDE)
    {
        for(int i = 0; i < MLO_TOP_DAT_SZ; ++i)
        {
            top_dat[i] = (_FLOAT)(0);
        }

        int gbl_in_scan_off  = gbl_in_off;
        int gbl_out_scan_off = gbl_out_off;

        barrier(CLK_LOCAL_MEM_FENCE);

        // read input map
        readInput(lcl_id, gbl_in_scan_off, bot, lcl_bot);

        // prefetch output
        for(int j = 0; j < MLO_FILTER_SIZE1 - 1; ++j, gbl_out_scan_off += MLO_OUT_STRIDE)
        {
            int top_df_off = gbl_out_scan_off;
            _FLOAT mask    = (_FLOAT)(1);
#if MLO_IN_HEIGHT != MLO_OUT_HEIGHT || MLO_FILTER_SIZE1 - 1 > MLO_OUT_HEIGHT
            top_df_off = (j < MLO_OUT_HEIGHT) ? top_df_off : 0;
            mask       = (j < MLO_OUT_HEIGHT) ? 1 : (_FLOAT)(0);
#endif

#if MLO_OUT_N_PIXS_OFF > 0
            if(spn == MLO_N_SPANS_PER_SCAN - 1)
            {
                int i = 0;
                for(; i < MLO_OUT_N_PIXS_OFF; ++i)
                {
                    top_dat[j * MLO_IN_TILE0 + i] = top_df[top_df_off + i] * mask;
#if DBG_OUT_OF_RNGE
                    if(top_df_off + i >= MLO_OUT_BATCH_STRIDE * MLO_BATCH_SZ)
                    {
                        printf("k:err:out-of-range\n");
                    }
#endif
                }
                for(; i < MLO_IN_TILE0; ++i)
                {
                    top_dat[j * MLO_IN_TILE0 + i] = (_FLOAT)(0);
                }
            }
            else
#endif
            {
                for(int i = 0; i < MLO_IN_TILE0; ++i)
                {
                    top_dat[j * MLO_IN_TILE0 + i] = top_df[top_df_off + i] * mask;
#if DBG_OUT_OF_RNGE
                    if(top_df_off + i >= MLO_OUT_BATCH_STRIDE * MLO_BATCH_SZ)
                    {
                        printf("k:err:out-of-range\n");
                    }
#endif
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // prolog
        // handling padding

        int sc         = 0;
        int sc_lcl_off = lcl_bot_off;
        // prolog
        // handling padding

        // top padding
        for(; sc < MLO_FILTER_SIZE1 - MLO_FILTER_PAD1 - 1; ++sc, sc_lcl_off += MLO_IN_LCL_WIDTH)
        {
            Processing(sc, sc_lcl_off, sc + MLO_FILTER_PAD1, 0, pvt_accum, lcl_bot, top_dat);
        }

        // generic loop

        for(; sc < MLO_IN_HEIGHT - MLO_FILTER_PAD1;
            ++sc, gbl_out_scan_off += MLO_OUT_STRIDE, sc_lcl_off += MLO_IN_LCL_WIDTH)
        {

            int top_df_off = gbl_out_scan_off;
            _FLOAT mask    = (_FLOAT)(1);

#if MLO_IN_HEIGHT != MLO_OUT_HEIGHT || MLO_FILTER_SIZE1 > MLO_OUT_HEIGHT
            top_df_off = ((sc + MLO_FILTER_PAD1) < MLO_OUT_HEIGHT) ? top_df_off : 0;
            mask       = ((sc + MLO_FILTER_PAD1) < MLO_OUT_HEIGHT) ? (_FLOAT)(1) : (_FLOAT)(0);
#endif
// move in the last output scans
#if MLO_OUT_N_PIXS_OFF > 0
            if(spn == MLO_N_SPANS_PER_SCAN - 1)
            {
                int i = 0;
                for(; i < MLO_OUT_N_PIXS_OFF; ++i)
                {
                    top_dat[(MLO_FILTER_SIZE1 - 1) * MLO_IN_TILE0 + i] =
                        top_df[top_df_off + i] * mask;
#if DBG_OUT_OF_RNGE
                    if(top_df_off + i >= MLO_OUT_BATCH_STRIDE * MLO_BATCH_SZ)
                    {
                        printf("k:err:out-of-range\n");
                    }
#endif
                }
                for(; i < MLO_IN_TILE0; ++i)
                {
                    top_dat[(MLO_FILTER_SIZE1 - 1) * MLO_IN_TILE0 + i] = (_FLOAT)(0);
                }
            }
            else
#endif
            {
                for(int i = 0; i < MLO_IN_TILE0; ++i)
                {
                    top_dat[(MLO_FILTER_SIZE1 - 1) * MLO_IN_TILE0 + i] =
                        top_df[top_df_off + i] * mask;
#if DBG_OUT_OF_RNGE
                    if(top_df_off + i >= MLO_OUT_BATCH_STRIDE * MLO_BATCH_SZ)
                    {
                        printf("k:err:out-of-range\n");
                    }
#endif
                }
            }

            // processing
            Processing(sc, sc_lcl_off, MLO_FILTER_SIZE1 - 1, 0, pvt_accum, lcl_bot, top_dat);

            // move up output to reduce overfetch

            moveOutputUp(top_dat);
        } // for (; sc < MLO_IN_HEIGHT - MLO_FILTER_PAD1; ++sc, gbl_out_scan_off += MLO_OUT_STRIDE,
          // sc_lcl_off += MLO_IN_LCL_WIDTH)

        for(; sc < MLO_IN_HEIGHT; ++sc, sc_lcl_off += MLO_IN_LCL_WIDTH)
        {

            // processing
            Processing(sc,
                       sc_lcl_off,
                       MLO_FILTER_SIZE1 - 1,
                       (MLO_FILTER_PAD1 + 1 - (MLO_IN_HEIGHT - sc)),
                       pvt_accum,
                       lcl_bot,
                       top_dat);
            moveOutputUp(top_dat);

        } // for (; sc < MLO_IN_HEIGHT)

    } // 	for (int b = 0;

    // final summation over each filter row
    for(int l = 0; l < MLO_FILTER_SIZE1; ++l)
    {
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int n = 0; n < MLO_FILTER_SIZE0; ++n)
        {
            lcl[lcl_id * MLO_FILTER_SIZE0 + n] = pvt_accum[l * MLO_FILTER_SIZE0 + n];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if(spn == 0)
        {
            for(int s = 0; s < MLO_N_SPANS_PER_SCAN - 1; ++s)
            {

                for(int n = 0; n < MLO_FILTER_SIZE0; ++n)
                {
                    pvt_accum[l * MLO_FILTER_SIZE0 + n] +=
                        lcl[(lcl_id + s + 1) * MLO_FILTER_SIZE0 + n];
                }
            }
        }
    }

    // output
    // inputs are outputs
    // TODO : for more than 1 input
    int c = 0;

    int wei_df_off = ((ib * MLO_N_OUTPUTS + o_idx + o) * (int)MLO_WEI_BATCH_STRIDE)
                     // this input channel
                     + mul24((c_idx + c), (int)MLO_WEI_CHANNEL_STRIDE);
    if(spn == 0 && o_idx + o < MLO_N_OUTPUTS && o < MLO_OUT_STACKS)
    {
        for(int i = 0; i < (MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0); ++i)
        {
            weights_df[wei_df_off + i] = pvt_accum[i];
        }
    }
}

// final reduction kernel
// add filters over batches
__attribute__((reqd_work_group_size(MLO_UT_GRP_SZ0, 1, 1))) __kernel void
MIOpenCvBwdWrW_rdc(const __global _FLOAT* weight_df_tmp, __global _FLOAT* weights_df)
{
    int gbl_id   = get_global_id(0);
    int wei_idx0 = gbl_id * MLO_UT_READ_UNIT;

    int wei_blk_idx = iDiv_legacy(wei_idx0, MLO_WEI_CHANNEL_STRIDE);
    int wei_idx     = iMod(wei_idx0, wei_blk_idx, MLO_WEI_CHANNEL_STRIDE);

    _FLOAT pvt_accum_wei[MLO_UT_READ_UNIT];
    for(int i = 0; i < MLO_UT_READ_UNIT; ++i)
    {
        pvt_accum_wei[i] = (_FLOAT)(0);
    }

    int batch_loop = (MLO_BATCH_SZ + (MLO_N_BATCH_LOOPS * MLO_N_LCL_BATCHS) - 1) /
                     (MLO_N_BATCH_LOOPS * MLO_N_LCL_BATCHS);
    for(int i = 0; i < batch_loop; ++i)
    {
        for(int j = 0; j < MLO_UT_READ_UNIT; ++j)
        {
            pvt_accum_wei[j] += weight_df_tmp[(wei_blk_idx * MLO_WEI_CHANNEL_STRIDE +
                                               i * MLO_N_OUTPUTS * MLO_WEI_BATCH_STRIDE) +
                                              wei_idx + j];
        }
    }

    for(int j = 0; j < MLO_UT_READ_UNIT; ++j)
    {
        weights_df[wei_idx0 + j] = pvt_accum_wei[j];
    }
}
