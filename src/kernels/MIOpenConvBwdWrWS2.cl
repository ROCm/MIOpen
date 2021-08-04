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
#include "float_types.h"

#define UNUSED __attribute__((__unused__))

#if((MLO_BATCH_SZ % MLO_N_BATCH_LOOPS) == 0)
#define MLO_N_BATCH_LOOPS_PARTIAL 0
#else
#define MLO_N_BATCH_LOOPS_PARTIAL 1
#endif

#define MLO_N_OUT_HORIZ_READS (MLO_ALIGNED_OUT_SCAN_LN)
#define MLO_OUT_HORIZ_PIX_SZ (MLO_N_OUT_HORIZ_READS * MLO_READ_UNIT)

// MLO_WEI_BLK_SZ0 is the width of each filter tap
#define MLO_WEI_BLK_SZ0 ((MLO_FILTER_SIZE0 + MLO_WEI_WKITEM - 1) / MLO_WEI_WKITEM)
#define MLO_WEI_BLK_SZ (MLO_FILTER_SIZE1 * MLO_WEI_BLK_SZ0)
// n of filter tiles in the group grid
#define MLO_N_WEI_BLK (MLO_GRP_SZ / MLO_WEI_BLK_SZ)
// n of steps per scan line to be made in the inner loop
// extended scan to deal with overshot in the inner loop
#define MLO_OUT_WEI_EXT_SCAN_LOOP ((MLO_OUT_WIDTH + MLO_N_WEI_BLK - 1) / MLO_N_WEI_BLK)

#define MLO_OUT_WEI_SCAN_LOOP (MLO_OUT_WEI_EXT_SCAN_LOOP)

// MLO_WEI_WKITEM is the number of filter taps (in horizonotal direction)
// MLO_WEI_WKITEM number of input data are saved in register for each thread
// MLO_WEI_WKITEM_REUSE is number of input data that can be reused, when filter move horizontally
#if MLO_WEI_WKITEM > (MLO_FILTER_STRIDE0 / MLO_WEI_BLK_SZ0)
#define MLO_WEI_WKITEM_REUSE (MLO_WEI_WKITEM - (MLO_FILTER_STRIDE0 / MLO_WEI_BLK_SZ0))
#else
#define MLO_WEI_WKITEM_REUSE 0
#endif

#define MLO_MAX_WEI_BLK ((MLO_OUT_WIDTH + MLO_OUT_WEI_SCAN_LOOP - 1) / MLO_OUT_WEI_SCAN_LOOP)
#if MLO_MAX_WEI_BLK > MLO_N_WEI_BLK
#undef MLO_MAX_WEI_BLK
#define MLO_MAX_WEI_BLK MLO_N_WEI_BLK
#endif

#define MLO_WEI_BLKS_SZ (MLO_MAX_WEI_BLK * MLO_WEI_BLK_SZ * MLO_WEI_WKITEM)
#define MLO_WEI_BLKS_LCL_SZ (MLO_WEI_BLKS_SZ * MLO_N_LCL_OUT_MAPS)

#define MLO_OUT_BLK_GRP_PIX_SZ (MLO_OUT_HORIZ_PIX_SZ * MLO_N_ALIGNED_OUT_SCAN_BLK)
#define MLO_OUT_BLK_GRP_WK_SZ (MLO_OUT_BLK_GRP_PIX_SZ / MLO_READ_UNIT)

#if MLO_OUT_HORIZ_PIX_SZ < (MLO_OUT_WEI_EXT_SCAN_LOOP * MLO_MAX_WEI_BLK)
#define MLO_OUT_HORIZ_PIX_EXT_SZ (MLO_OUT_WEI_EXT_SCAN_LOOP * MLO_MAX_WEI_BLK)
#else
#define MLO_OUT_HORIZ_PIX_EXT_SZ MLO_OUT_HORIZ_PIX_SZ
#endif
#define MLO_OUT_BLK_GRP_EXT_PIX_SZ (MLO_OUT_HORIZ_PIX_EXT_SZ * MLO_N_ALIGNED_OUT_SCAN_BLK)
#define MLO_OUT_LCL_SZ (MLO_OUT_BLK_GRP_EXT_PIX_SZ)
// LDS OUT SIZE
#define MLO_TOTAL_OUT_LCL_SZ (MLO_N_LCL_OUT_MAPS * MLO_OUT_LCL_SZ)
#if((MLO_OUT_HEIGHT / MLO_N_ALIGNED_OUT_SCAN_BLK) * MLO_N_ALIGNED_OUT_SCAN_BLK == MLO_OUT_HEIGHT)
#define MLO_BLK_ALIGNED 1
#else
#define MLO_BLK_ALIGNED 0
#endif

// input size depends on output scan length and
// number of output scans
// this number is constrained by amount or LDS and size of register file.
// TO DO:: CHECK PADDING!!!
#define MLO_IN_LCL_HEIGHT ((MLO_N_ALIGNED_OUT_SCAN_BLK - 1) * MLO_FILTER_STRIDE1 + MLO_FILTER_SIZE1)
// there is an assumption that the scanline fits into LDS
#define MLO_N_IN_HORIZ_PIX_READS MLO_IN_WIDTH
#define MLO_N_IN_HORIZ_READS ((MLO_N_IN_HORIZ_PIX_READS + MLO_READ_UNIT - 1) / MLO_READ_UNIT)
#define MLO_IN_N_PIXS_OFF \
    (MLO_N_IN_HORIZ_PIX_READS - (MLO_N_IN_HORIZ_PIX_READS / MLO_READ_UNIT) * MLO_READ_UNIT)

// LDS IN SIZE
#define MLO_TOTAL_IN_LCL_SZ (MLO_N_LCL_IN_MAPS * MLO_IN_LCL_SZ)
#define MLO_IN_VERT_READS (MLO_GRP_SZ / MLO_N_IN_HORIZ_READS)

#if(MLO_TOTAL_OUT_LCL_SZ + MLO_TOTAL_IN_LCL_SZ) > (MLO_WEI_BLKS_LCL_SZ)
#define MLO_LCL_SZ (MLO_TOTAL_OUT_LCL_SZ + MLO_TOTAL_IN_LCL_SZ)
#else
#define MLO_LCL_SZ (MLO_WEI_BLKS_LCL_SZ)
#endif

#include "math_ops.h"

/*********************************************************************************************************
// wrw algorithm for large filters
// idea:
// split filter taps into sub-tiles along x and y axis with number of tap groups muliples of stride
or 1
// for example
// the 5x10 filter has been split into 10 sub-tiles 1x5 each, 2 tap in y direction and 5 taps in x
direction.
// those horizontal taps are 0, 2, 4, 6, 8 and 1, 3, 5, 7, 9
// a single vertical tap is 0 or 1 or 2 or 3 or 4.
// one may say sub-tiles are indexed by a vertical tap.
// the partial sum has been calculated into those 10 sub-tiles in parallel.
// the full filter has been calulated by reducing all sub-tiles into a single filter per group.
// teh accumulation has been done over all pixels of several outputs being shared with a single
input.
// the accuulation has been done per batch.
//
// the total reduction over all batches has been doesn a separete kerenel.
//
// alg
//
//		until end of output map (MLO_N_OUT_BLK)
//			load input map block in LDS
//			load output maps in LDS
//		for j in output scans
//			for i in output scan interval
//				accumulate the weights into sub-tiles
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
    __local _FLOAT* lcl_top = lcl + MLO_TOTAL_IN_LCL_SZ;

    uint c_idx_base = get_group_id(0); // input map index base

    uint o_idx_base = get_group_id(1); // output map index base
    uint ib_base    = get_group_id(2); // batch index base

    uint ib = ib_base * MLO_N_BATCH_LOOPS;

    uint c_idx = c_idx_base * MLO_N_LCL_IN_MAPS; // input map index

    uint o_idx = o_idx_base * (MLO_N_OUT_BLK_GRP * MLO_N_LCL_OUT_MAPS); // output map index

    uint gbl_in_off  = c_idx * MLO_IN_CHANNEL_STRIDE + ib * MLO_IN_BATCH_STRIDE;
    uint gbl_out_off = o_idx * MLO_OUT_CHANNEL_STRIDE + ib * MLO_OUT_BATCH_STRIDE;

    uint lcl_id = get_local_id(0);

// weight tile id
#if MLO_WEI_BLK_SZ & (MLO_WEI_BLK_SZ - 1)
    uint w_blk_idx = iDiv(lcl_id, MLO_WEI_BLK_SZ);
    uint w_idx     = iMod(lcl_id, w_blk_idx, MLO_WEI_BLK_SZ);
#else
    uint w_blk_idx = lcl_id / MLO_WEI_BLK_SZ;
    uint w_idx     = lcl_id & (MLO_WEI_BLK_SZ - 1);
#endif
#if MLO_WEI_BLK_SZ0 & (MLO_WEI_BLK_SZ0 - 1)
    uint w_y  = iDiv(w_idx, MLO_WEI_BLK_SZ0);
    uint w_x0 = iMod(w_idx, w_y, MLO_WEI_BLK_SZ0);
#else
    uint w_y       = w_idx / MLO_WEI_BLK_SZ0;
    uint w_x0      = w_idx & (MLO_WEI_BLK_SZ0 - 1);
#endif

    // only w_blk_idx_dummy < MLO_MAX_WEI_BLK will do useful core convolution computation
    uint w_blk_idx_dummy = w_blk_idx < MLO_MAX_WEI_BLK ? w_blk_idx : 0;

    __private _FLOAT_ACCUM pvt_accum[(MLO_N_OUT_BLK_GRP * MLO_N_LCL_OUT_MAPS * MLO_WEI_WKITEM)];

    for(uint i = 0; i < (MLO_N_OUT_BLK_GRP * MLO_N_LCL_OUT_MAPS * MLO_WEI_WKITEM); ++i)
    {
        pvt_accum[i] = (_FLOAT_ACCUM)0;
    }

    // zero out LDS
    for(uint i = lcl_id; i < (MLO_LCL_SZ); i += MLO_GRP_SZ)
    {
        lcl[i] = 0;
    }

    // over all batches
    for(uint b = 0;
#if MLO_N_BATCH_LOOPS_PARTIAL
        b < MLO_N_BATCH_LOOPS && ib + b < MLO_BATCH_SZ;
#else
        b < MLO_N_BATCH_LOOPS;
#endif
        ++b, gbl_in_off += MLO_IN_BATCH_STRIDE, gbl_out_off += MLO_OUT_BATCH_STRIDE)
    {

        barrier(CLK_LOCAL_MEM_FENCE);

        uint in_y  = 0;
        uint out_y = 0;

        __private _FLOAT in_rd_data[MLO_READ_UNIT];

        uint gbl_in_scan_off  = gbl_in_off;
        uint gbl_out_scan_off = gbl_out_off;

        // zero out input lower bound padding in LDS, and prefetch first few lines of input into LDS
        for(uint p4 = lcl_id; p4 < MLO_N_IN_HORIZ_READS * (MLO_FILTER_SIZE1 - MLO_FILTER_STRIDE1);
            p4 += MLO_GRP_SZ)
        {
#if MLO_N_IN_HORIZ_READS & (MLO_N_IN_HORIZ_READS - 1)
            uint c_scan = iDiv(p4, MLO_N_IN_HORIZ_READS);
            uint c_pix4 = iMod(p4, c_scan, MLO_N_IN_HORIZ_READS);
#else
            uint c_scan = p4 / MLO_N_IN_HORIZ_READS;
            uint c_pix4 = p4 & (MLO_N_IN_HORIZ_READS - 1);
#endif

            for(uint i = 0; i < MLO_READ_UNIT; ++i)
            {
                in_rd_data[i] = 0;
            }

            int c_scan_bot = (int)c_scan - MLO_FILTER_PAD1;

            if(c_scan_bot >= 0 && c_scan_bot < MLO_IN_HEIGHT)
            {
                uint bot_off =
                    gbl_in_scan_off + c_scan_bot * MLO_IN_STRIDE + c_pix4 * MLO_READ_UNIT;
                const __global _FLOAT* bot_p = &bot[bot_off];
// still problems with unaligned LDS access
#if MLO_IN_N_PIXS_OFF > 0
                if(c_pix4 == MLO_N_IN_HORIZ_READS - 1)
                {
                    uint i = 0;
                    for(; i < MLO_IN_N_PIXS_OFF; ++i)
                    {
                        in_rd_data[i] = bot_p[i];
                    }
                }
                else
#endif
                {
                    for(uint i = 0; i < MLO_READ_UNIT; ++i)
                    {
                        in_rd_data[i] = bot_p[i];
                    }
                }
            }
            for(uint i = 0; i < MLO_READ_UNIT; ++i)
            {
                lcl_bot[c_scan * MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 + c_pix4 * MLO_READ_UNIT + i] =
                    in_rd_data[i];
            }
        }

        in_y += MLO_FILTER_SIZE1 - MLO_FILTER_STRIDE1 - MLO_FILTER_PAD1;

        // TO DO: HANDLE PADDING
        // over all out blocks
        // processing per MLO_N_ALIGNED_OUT_SCAN_BLK output scans

        for(uint ob = 0; ob < MLO_N_OUT_BLK; ++ob,
                 in_y += (MLO_IN_LCL_HEIGHT - MLO_FILTER_SIZE1 + MLO_FILTER_STRIDE1),
                 out_y += MLO_N_ALIGNED_OUT_SCAN_BLK)
        {

            barrier(CLK_LOCAL_MEM_FENCE);

            // fetch input: (MLO_IN_LCL_HEIGHT - MLO_FILTER_SIZE1 + 1)
            // TODO:: HANDLE multiple INPUTS
            // an overshoot has to be handled by zero outing output
            gbl_in_scan_off = gbl_in_off + mul24(in_y, (uint)MLO_IN_STRIDE);
            for(uint p4 = lcl_id;
                p4 <
                MLO_N_IN_HORIZ_READS * (MLO_IN_LCL_HEIGHT - MLO_FILTER_SIZE1 + MLO_FILTER_STRIDE1);
                p4 += MLO_GRP_SZ)
            {
#if MLO_N_IN_HORIZ_READS & (MLO_N_IN_HORIZ_READS - 1)
                uint c_scan = iDiv(p4, MLO_N_IN_HORIZ_READS);
                uint c_pix4 = iMod(p4, c_scan, MLO_N_IN_HORIZ_READS);
#else
                uint c_scan = p4 / MLO_N_IN_HORIZ_READS;
                uint c_pix4 = p4 & (MLO_N_IN_HORIZ_READS - 1);
#endif

                // this effectively set upper and right boundary padding to 0 in LDS
                for(uint i = 0; i < MLO_READ_UNIT; ++i)
                {
                    in_rd_data[i] = 0;
                }

                // this effectively set upper boundary padding to 0 in LDS
                if(in_y + c_scan < MLO_IN_HEIGHT)
                {
                    uint bot_off =
                        gbl_in_scan_off + c_scan * MLO_IN_STRIDE + c_pix4 * MLO_READ_UNIT;
                    const __global _FLOAT* bot_p = &bot[bot_off];

#if MLO_IN_N_PIXS_OFF > 0
                    if(c_pix4 == MLO_N_IN_HORIZ_READS - 1)
                    {

                        uint i = 0;
                        for(; i < MLO_IN_N_PIXS_OFF; ++i)
                        {
                            in_rd_data[i] = bot_p[i];
                        }
                    }
                    else
#endif
                    {

                        for(uint i = 0; i < MLO_READ_UNIT; ++i)
                        {
                            in_rd_data[i] = bot_p[i];
                        }
                    }

                } // if (in_y + c_scan < MLO_IN_HEIGHT)

                for(uint i = 0; i < MLO_READ_UNIT; ++i)
                {
                    uint lcl_off =
                        (c_scan + MLO_FILTER_SIZE1 - MLO_FILTER_STRIDE1) * MLO_IN_LCL_WIDTH +
                        MLO_FILTER_PAD0 + c_pix4 * MLO_READ_UNIT;
                    lcl_bot[lcl_off + i] = in_rd_data[i];
                }
            }

            gbl_out_scan_off = gbl_out_off + mul24(out_y, (uint)MLO_OUT_STRIDE);

            // over all outputs groups
            // MLO_N_OUT_BLK_GRP outputs reuse the same input
            // each output blk is MLO_N_LCL_OUT_MAPS outputs
            // MLO_N_LCL_OUT_MAPS nuber is restricted by LDS size

            uint gbl_out_scan_off1 = gbl_out_scan_off;
            for(uint og = 0; og < MLO_N_OUT_BLK_GRP;
                ++og, gbl_out_scan_off1 += MLO_N_LCL_OUT_MAPS * MLO_OUT_CHANNEL_STRIDE)
            {
                barrier(CLK_LOCAL_MEM_FENCE);

                // fetch output. MLO_N_ALIGNED_OUT_SCAN_BLK output scans, each of size
                // MLO_N_OUT_HORIZ_READS

                __private _FLOAT out_rd_data[MLO_READ_UNIT];

                for(uint oo_p4 = lcl_id; oo_p4 < (MLO_N_LCL_OUT_MAPS * MLO_N_ALIGNED_OUT_SCAN_BLK *
                                                  MLO_N_OUT_HORIZ_READS);
                    oo_p4 += MLO_GRP_SZ)
                {
#if(MLO_N_ALIGNED_OUT_SCAN_BLK * MLO_N_OUT_HORIZ_READS) & \
    ((MLO_N_ALIGNED_OUT_SCAN_BLK * MLO_N_OUT_HORIZ_READS) - 1)
                    uint o = iDiv(oo_p4, (MLO_N_ALIGNED_OUT_SCAN_BLK * MLO_N_OUT_HORIZ_READS));
                    uint o_pX4 =
                        iMod(oo_p4, o, (MLO_N_ALIGNED_OUT_SCAN_BLK * MLO_N_OUT_HORIZ_READS));
#else
                    uint o     = oo_p4 / (MLO_N_ALIGNED_OUT_SCAN_BLK * MLO_N_OUT_HORIZ_READS);
                    uint o_pX4 = oo_p4 & ((MLO_N_ALIGNED_OUT_SCAN_BLK * MLO_N_OUT_HORIZ_READS) - 1);
#endif
#if MLO_N_OUT_HORIZ_READS & (MLO_N_OUT_HORIZ_READS - 1)
                    uint o_scan = iDiv(o_pX4, MLO_N_OUT_HORIZ_READS);
                    uint o_pix4 = iMod(o_pX4, o_scan, MLO_N_OUT_HORIZ_READS);
#else
                    uint o_scan = o_pX4 / MLO_N_OUT_HORIZ_READS;
                    uint o_pix4 = o_pX4 & (MLO_N_OUT_HORIZ_READS - 1);
#endif
                    for(uint i = 0; i < MLO_READ_UNIT; ++i)
                    {
                        out_rd_data[i] = 0;
                    }
                    uint top_df_off = gbl_out_scan_off1 + o * MLO_OUT_CHANNEL_STRIDE +
                                      o_scan * MLO_OUT_STRIDE + o_pix4 * MLO_READ_UNIT;
                    const __global _FLOAT* top_df_p = &top_df[top_df_off];

                    if(out_y + o_scan < MLO_OUT_HEIGHT &&
                       o_idx + og * MLO_N_LCL_OUT_MAPS + o < MLO_N_OUTPUTS)
                    {
// scan has been fetch by 4
// here the non-multiple of 4 scan has been handled
// also makes sure the input garbage hs been multipled by 0
#if MLO_OUT_N_PIXS_OFF > 0
                        if(o_pix4 == (MLO_N_OUT_HORIZ_READS - 1))
                        {
                            uint i = 0;
                            for(; i < MLO_OUT_N_PIXS_OFF; ++i)
                            {
                                out_rd_data[i] = top_df_p[i];
                            }
                        }

                        else
#endif

                        {
                            for(uint i = 0; i < MLO_READ_UNIT; ++i)
                            {
                                out_rd_data[i] = top_df_p[i];
                            }
                        }

                    } // if (out_y + o_scan < MLO_OUT_HEIGHT && o_idx + og *MLO_N_LCL_OUT_MAPS + o <
                      // MLO_N_OUTPUTS)

                    // write into LDS with MLO_OUT_HORIZ_PIX_EXT_SZ stride to zero out weights block
                    // overshoot

                    for(uint i = 0; i < MLO_READ_UNIT; ++i)
                    {
                        uint lcl_off = o * MLO_OUT_LCL_SZ + o_scan * MLO_OUT_HORIZ_PIX_EXT_SZ +
                                       o_pix4 * MLO_READ_UNIT;
                        lcl_top[lcl_off + i] = out_rd_data[i];
                    }

                } //	for (uint oo_p4 = lcl_id; oo_p4 <
                  //(MLO_N_LCL_OUT_MAPS*MLO_N_ALIGNED_OUT_SCAN_BLK*MLO_N_OUT_HORIZ_READS); oo_p4 +=
                  // MLO_GRP_SZ)

                barrier(CLK_LOCAL_MEM_FENCE);

                // process
                // algorithm

                // over all input scans in LDS
                for(uint j = 0; j < MLO_N_ALIGNED_OUT_SCAN_BLK; ++j)
                {
                    // prefetch proper inputs pixels.
                    // they are MLO_WEI_BLK_SZ0 apart taps of the filter

                    _FLOAT i_vals[MLO_WEI_WKITEM];

                    for(uint w = 0; w < MLO_WEI_WKITEM_REUSE; ++w)
                    {
                        uint w_x = w_x0 + w * MLO_WEI_BLK_SZ0;
                        uint i_off =
                            (j * MLO_FILTER_STRIDE1 + w_y) * MLO_IN_LCL_WIDTH +
                            (w_blk_idx_dummy * MLO_OUT_WEI_SCAN_LOOP + 0) * MLO_FILTER_STRIDE0 +
                            w_x;
                        _FLOAT i_val = lcl_bot[i_off];

                        i_vals[w] = i_val;
                    }

                    // if we overshoot the scanline
                    // out data will be 0 by initial setting
                    for(uint i = 0; i < MLO_OUT_WEI_SCAN_LOOP; ++i)
                    {

                        // read the current input pixel
                        for(uint w = MLO_WEI_WKITEM_REUSE; w < MLO_WEI_WKITEM; ++w)
                        {
                            uint w_x = w_x0 + w * MLO_WEI_BLK_SZ0;
                            uint i_off =
                                (j * MLO_FILTER_STRIDE1 + w_y) * MLO_IN_LCL_WIDTH +
                                (w_blk_idx_dummy * MLO_OUT_WEI_SCAN_LOOP + i) * MLO_FILTER_STRIDE0 +
                                w_x;
                            _FLOAT i_val = lcl_bot[i_off];

                            i_vals[w] = i_val;
                        }
                        // for each output accumulate a proper filter tap
                        for(uint o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)
                        {

                            // read with MLO_OUT_HORIX_PIX_EXT_SZ stride
                            _FLOAT o_val =
                                lcl_top[(o)*MLO_OUT_LCL_SZ + j * MLO_OUT_HORIZ_PIX_EXT_SZ +
                                        (w_blk_idx_dummy * MLO_OUT_WEI_SCAN_LOOP + i)];

                            for(uint w = 0; w < MLO_WEI_WKITEM; ++w)
                            {
                                _FLOAT i_val = i_vals[w];

                                pvt_accum[(og * MLO_N_LCL_OUT_MAPS + o) * MLO_WEI_WKITEM + w] +=
                                    CVT_FLOAT2ACCUM(i_val) * CVT_FLOAT2ACCUM(o_val);
                            } // for (/*uint w = 0*/; w < MLO_WEI_WKITEM; ++w)
                        }     // for (uint o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)

                        for(uint w = 0; w < MLO_WEI_WKITEM_REUSE; ++w)
                        {
                            i_vals[w] = i_vals[w + MLO_WEI_WKITEM - MLO_WEI_WKITEM_REUSE];
                        }
                    } // for (uint i = 0; i < MLO_OUT_WEI_SCAN_LOOP; ++i)
                }     // for (uint j = 0; j < MLO_N_ALIGNED_OUT_SCAN_BLK; ++j)

            } // for(; og < (MLO_N_OUT_BLK_GRP; ++og )

            // move the input data tail inside LDS to reduce mem bandwidth
            for(uint c_scan = 0; c_scan < MLO_FILTER_SIZE1 - MLO_FILTER_STRIDE1; ++c_scan)
            {
                barrier(CLK_LOCAL_MEM_FENCE);

                for(uint p4 = lcl_id; p4 < MLO_N_IN_HORIZ_READS; p4 += MLO_GRP_SZ)
                {

                    uint c_pix4 = p4;
                    for(uint i = 0; i < MLO_READ_UNIT; ++i)
                    {
                        lcl_bot[c_scan * (MLO_IN_LCL_WIDTH) + MLO_FILTER_PAD0 +
                                c_pix4 * MLO_READ_UNIT + i] =
                            lcl_bot[(c_scan +
                                     (MLO_IN_LCL_HEIGHT - MLO_FILTER_SIZE1 + MLO_FILTER_STRIDE1)) *
                                        (MLO_IN_LCL_WIDTH) +
                                    MLO_FILTER_PAD0 + c_pix4 * MLO_READ_UNIT + i];
                    }
                }
            }
        } // for (uint ob = 0; ob < MLO_N_OUT_BLK; ++ob, in_y += (MLO_IN_LCL_HEIGHT -
          // MLO_FILTER_SIZE1 + 1), out_y += MLO_N_ALIGNED_OUT_SCAN_BLK)
    }     // for (uint b = 0;

    // send it out

    uint wei_lcl_off       = 0;
    _FLOAT_ACCUM final_sum = 0;

    // save in lcl and orgnize in a proper order
    // outputs
    //	filter size1
    //	  filter size0

    // TO DO:: DEPENDING ON THE GROUP SIZE
    for(uint og = 0; og < MLO_N_OUT_BLK_GRP; ++og)
    {
        barrier(CLK_LOCAL_MEM_FENCE);

        // o_base may be larger than MLO_N_OUTPUTS, so o_number may be negative
        uint o_base   = o_idx + og * MLO_N_LCL_OUT_MAPS;
        uint o_number = o_base < MLO_N_OUTPUTS ? MLO_N_OUTPUTS - o_base : 0;
        o_number      = o_number < MLO_N_LCL_OUT_MAPS ? o_number : MLO_N_LCL_OUT_MAPS;

        for(uint o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)
        {
            if(o < o_number && w_blk_idx < MLO_MAX_WEI_BLK)
            {
                for(uint w = 0; w < MLO_WEI_WKITEM; ++w)
                {
                    // save "virtual" filter table
                    uint w_x    = w_x0 + w * MLO_WEI_BLK_SZ0;
                    wei_lcl_off = ((o * MLO_MAX_WEI_BLK + w_blk_idx) * MLO_FILTER_SIZE1 + w_y) *
                                      (MLO_WEI_BLK_SZ0 * MLO_WEI_WKITEM) +
                                  w_x;

                    lcl[wei_lcl_off] = CVT_ACCUM2FLOAT(
                        pvt_accum[(og * MLO_N_LCL_OUT_MAPS + o) * MLO_WEI_WKITEM + w]);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // read into real filter table
        for(uint l = lcl_id; l < (o_number * MLO_WEI_CHANNEL_STRIDE); l += MLO_GRP_SZ)
        {
#if MLO_WEI_CHANNEL_STRIDE & (MLO_WEI_CHANNEL_STRIDE - 1)
            uint oo    = iDiv(l, MLO_WEI_CHANNEL_STRIDE);
            uint wei_i = iMod(l, oo, MLO_WEI_CHANNEL_STRIDE);
#else
            uint oo      = l / MLO_WEI_CHANNEL_STRIDE;
            uint wei_i   = l & MLO_WEI_CHANNEL_STRIDE - 1;
#endif
#if(MLO_FILTER_SIZE0) & ((MLO_FILTER_SIZE0)-1)
            uint wei_i_y = iDiv(wei_i, (MLO_FILTER_SIZE0));
            uint wei_i_x = iMod(wei_i, wei_i_y, (MLO_FILTER_SIZE0));
#else
            uint wei_i_y = wei_i / (MLO_FILTER_SIZE0);
            uint wei_i_x = wei_i & ((MLO_FILTER_SIZE0)-1);
#endif
            // send it out
            // inputs are outputs
            uint wei_df_off = ((ib_base * MLO_N_OUTPUTS + o_idx) * (uint)MLO_WEI_BATCH_STRIDE)
                              // this input channel
                              + mul24(c_idx, (uint)MLO_WEI_CHANNEL_STRIDE);

            final_sum = 0;
            for(uint i = 0; i < MLO_MAX_WEI_BLK; ++i)
            {
                final_sum +=
                    CVT_FLOAT2ACCUM(lcl[((oo * MLO_MAX_WEI_BLK + i) * MLO_FILTER_SIZE1 + wei_i_y) *
                                            (MLO_WEI_BLK_SZ0 * MLO_WEI_WKITEM) +
                                        wei_i_x]);
            }

            uint wei_out_off =
                wei_df_off + (og * MLO_N_LCL_OUT_MAPS + oo) * MLO_WEI_BATCH_STRIDE + wei_i;

            weights_df[wei_out_off] = CVT_ACCUM2FLOAT(final_sum); // lcl[lcl_id]; //
        }

    } // for(uint og = 0; og < MLO_N_OUT_BLK_GRP; ++og)
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

    _FLOAT_ACCUM pvt_accum_wei[MLO_UT_READ_UNIT];
    for(uint i = 0; i < MLO_UT_READ_UNIT; ++i)
    {
        pvt_accum_wei[i] = CVT_FLOAT2ACCUM(0);
    }

    uint batch_loop = MLO_N_BATCH_BLKS;

    for(uint i = 0; i < batch_loop; ++i)
    {
        for(uint j = 0; j < MLO_UT_READ_UNIT; ++j)
        {
            pvt_accum_wei[j] +=
                CVT_FLOAT2ACCUM(weight_df_tmp[(wei_blk_idx * MLO_WEI_CHANNEL_STRIDE +
                                               i * MLO_N_OUTPUTS * MLO_WEI_BATCH_STRIDE) +
                                              wei_idx + j]);
        }
    }

    for(uint j = 0; j < MLO_UT_READ_UNIT; ++j)
    {
        weights_df[wei_idx0 + j] = CVT_ACCUM2FLOAT(pvt_accum_wei[j]);
    }
}
