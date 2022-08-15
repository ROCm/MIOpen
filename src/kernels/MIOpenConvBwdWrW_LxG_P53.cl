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

#if MLO_IN_TILE0 == 0
#error "Error: (MLO_IN_TILE0 == 0)"
#endif

#ifndef MLO_OUT_WIDTH_CHUNK
#error "MLO_OUT_WIDTH_CHUNK is not defined"
#endif

// MLO_OUT_WIDTH_CHUNK - The length of output row processed by workgroup.
// If the input row was split into chunks to be fit into LDS, the corresponding output row
// will be split into chunks so as to fit into private registers.

#define MLO_N_OUT_HORIZ_READS ((MLO_OUT_WIDTH_CHUNK + MLO_IN_TILE0 - 1) / MLO_IN_TILE0)
#define MLO_N_SPANS_PER_SCAN (MLO_N_OUT_HORIZ_READS)
#define MLO_N_OUT_HORIZ_PIX_READS (MLO_N_OUT_HORIZ_READS * MLO_IN_TILE0)
#define MLO_OUT_N_PIXS_OFF \
    (MLO_OUT_WIDTH_CHUNK - ((MLO_OUT_WIDTH_CHUNK / MLO_IN_TILE0) * MLO_IN_TILE0))
#define MLO_N_OUT_VERTICAL_READS (MLO_FILTER_SIZE1)
// won't run non-border blocks if  MLO_IN_N_VERT_LOOPS < 2
//

#define MLO_IN_VERT_READS MLO_IN_EXTENT1

#if MLO_IN_N_VERT_LOOPS >= 2
#define MLO_N_GENERIC_LOOPS (MLO_IN_N_VERT_LOOPS - 2)
#else
#define MLO_N_GENERIC_LOOPS 0
#endif

// there is an assumption that the horizontal scanline fits into LDS
// In case the entire input width can't fit into LDS, the horizontal
// scanline becomes the chunk of input calculated in solver based on available LDS.

#define MLO_N_IN_HORIZ_PIX_READS (MLO_IN_WIDTH_CHUNK)
#define MLO_N_IN_HORIZ_READS ((MLO_N_IN_HORIZ_PIX_READS + MLO_READ_UNIT - 1) / MLO_READ_UNIT)
#define MLO_IN_N_PIXS_OFF \
    (MLO_N_IN_HORIZ_PIX_READS - (MLO_N_IN_HORIZ_PIX_READS / MLO_READ_UNIT) * MLO_READ_UNIT)

#if MLO_IN_WIDTH_N_LOOPS > 1 && MLO_FILTER_PAD0 > 0
#define MLO_IN_LCL_PADDED_WIDTH (MLO_N_IN_HORIZ_READS * MLO_READ_UNIT + MLO_FILTER_PAD0)
#define MLO_IN_LCL_UNPADDED_WIDTH (MLO_N_IN_HORIZ_READS * MLO_READ_UNIT)
#else
#define MLO_IN_LCL_WIDTH (MLO_N_IN_HORIZ_READS * MLO_READ_UNIT + 2 * MLO_FILTER_PAD0)
#endif

#define MLO_IN_LCL_HEIGHT MLO_IN_VERT_READS

#if MLO_IN_WIDTH_N_LOOPS > 1 && MLO_FILTER_PAD0 > 0
#define MLO_IN_LCL_PADDED_SZ (MLO_IN_LCL_PADDED_WIDTH * MLO_IN_LCL_HEIGHT)
#define MLO_IN_LCL_UNPADDED_SZ (MLO_IN_LCL_UNPADDED_WIDTH * MLO_IN_LCL_HEIGHT)
#define MLO_IN_LCL_SZ (MLO_IN_LCL_PADDED_SZ) // Assuming padded > unpadded size
#else
#define MLO_IN_LCL_SZ (MLO_IN_LCL_WIDTH * MLO_IN_LCL_HEIGHT)
#endif

#define MLO_TOTAL_IN_LCL_SZ (MLO_N_LCL_BATCHS * MLO_N_LCL_IN_MAPS * MLO_IN_LCL_SZ)

#define MLO_WEI_LCL_SZ (MLO_GRP_SZ * MLO_FILTER_SIZE0)
#if MLO_TOTAL_IN_LCL_SZ > MLO_WEI_LCL_SZ
#define MLO_LCL_SZ (MLO_TOTAL_IN_LCL_SZ)
#else
#define MLO_LCL_SZ (MLO_WEI_LCL_SZ)
#endif

// if to read all of the number of MLO_N_LCL_IN_MAPS input channel or not
#define MLO_READ_PARTIAL_N_LCL_IN_MAPS (MLO_N_INPUTS % MLO_N_LCL_IN_MAPS != 0)

#include "math_ops.h"

/*
        group cooperative read
        read by MLO_READ_UNIT
        handle out of range both horizontally and vertically (by fixed number of veryical reads)

        no guard against number of inputs
*/
void readInput(uint lcl_id,
               uint gbl_in_scan_off,
#if !MLO_READ_PARTIAL_N_LCL_IN_MAPS
               UNUSED
#endif
                   uint n_in_map_reads,
               uint n_v_reads,
               const __global _FLOAT* __restrict bot,
               __local _FLOAT* __restrict lcl_bot,
               uint chunk_id)
{

#if MLO_IN_WIDTH_N_LOOPS > 1 && MLO_FILTER_PAD0 > 0
    uint pad_shift = 0;
    if(chunk_id == 0)
    {
        pad_shift = MLO_FILTER_PAD0;
    }
#else
    uint pad_shift = MLO_FILTER_PAD0;
#endif

    for(uint p4 = lcl_id; p4 < MLO_N_LCL_IN_MAPS * MLO_N_IN_HORIZ_READS * n_v_reads;
        p4 += MLO_GRP_SZ)
    {
        uint c    = 0;
        uint t_p4 = p4;
#if MLO_N_LCL_IN_MAPS > 1
        c    = p4 / (MLO_N_IN_HORIZ_READS * n_v_reads);
        t_p4 = iMod(p4, c, (MLO_N_IN_HORIZ_READS * n_v_reads));
#endif

        uint c_scan = t_p4 / (MLO_N_IN_HORIZ_READS);

#if MLO_N_IN_HORIZ_READS & (MLO_N_IN_HORIZ_READS - 1)
        uint c_pix4 = iMod(t_p4, c_scan, (MLO_N_IN_HORIZ_READS));
#else
        uint c_pix4 = t_p4 & (MLO_N_IN_HORIZ_READS - 1);
#endif

        uint bot_off = gbl_in_scan_off + c * MLO_IN_CHANNEL_STRIDE + c_scan * MLO_IN_STRIDE +
                       c_pix4 * MLO_READ_UNIT;
        const __global _FLOAT* bot_p = &bot[bot_off];

        __private _FLOAT in_rd_data[MLO_READ_UNIT];

        for(uint i = 0; i < MLO_READ_UNIT; ++i)
        {
            in_rd_data[i] = 0;
        }

#if MLO_READ_PARTIAL_N_LCL_IN_MAPS
        if(c < n_in_map_reads)
#endif
        {
#if MLO_IN_WIDTH_LAST_CHUNK_VALID_READ_UNITS > 0
            if(chunk_id == MLO_IN_WIDTH_N_LOOPS - 1 &&
               c_pix4 < MLO_IN_WIDTH_LAST_CHUNK_VALID_READ_UNITS)
            {
                if(c_pix4 == MLO_IN_WIDTH_LAST_CHUNK_VALID_READ_UNITS - 1)
                {
                    for(uint i = 0; i < MLO_IN_WIDTH_LAST_CHUNK_VALID_PIXELS_IN_LAST_READ_UNIT; ++i)
                    {
                        in_rd_data[i] = bot_p[i];
                    }
                }
                else
                {
                    for(uint i = 0; i < MLO_READ_UNIT; ++i)
                    {
                        in_rd_data[i] = bot_p[i];
                    }
                }
            }
            else if(chunk_id != MLO_IN_WIDTH_N_LOOPS - 1)
#endif
            {
#if MLO_IN_N_PIXS_OFF > 0
                if(c_pix4 == MLO_N_IN_HORIZ_READS - 1)
                {
                    for(uint i = 0; i < MLO_IN_N_PIXS_OFF; ++i)
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
        }

#if MLO_IN_WIDTH_N_LOOPS > 1 && MLO_FILTER_PAD0 > 0
        if(chunk_id == 0 || chunk_id == MLO_IN_WIDTH_N_LOOPS - 1)
        {
            for(uint i = 0; i < MLO_READ_UNIT; ++i)
            {
                int lcl_in_off = c * MLO_IN_LCL_PADDED_SZ + c_scan * MLO_IN_LCL_PADDED_WIDTH +
                                 pad_shift + c_pix4 * MLO_READ_UNIT + i;
                lcl_bot[lcl_in_off] = in_rd_data[i];
            }
        }
        else
        {
            for(uint i = 0; i < MLO_READ_UNIT; ++i)
            {
                int lcl_in_off = c * MLO_IN_LCL_UNPADDED_SZ + c_scan * MLO_IN_LCL_UNPADDED_WIDTH +
                                 pad_shift + c_pix4 * MLO_READ_UNIT + i;
                lcl_bot[lcl_in_off] = in_rd_data[i];
            }
        }
#else
        (void)chunk_id;
        for(uint i = 0; i < MLO_READ_UNIT; ++i)
        {
            int lcl_in_off = c * MLO_IN_LCL_SZ + c_scan * MLO_IN_LCL_WIDTH + pad_shift +
                             c_pix4 * MLO_READ_UNIT + i;
            lcl_bot[lcl_in_off] = in_rd_data[i];
        }
#endif
    } // for (int p4 = lcl_id; p4 < MLO_N_LCL_IN_MAPS * MLO_N_IN_HORIZ_READS * MLO_IN_VERT_READS;

    barrier(CLK_LOCAL_MEM_FENCE);
}

/*
        core processing loop
        bot - input, from local (1 span)
        top - output diff, from global (array of spans, filters vertical size)

        loop over filter vertical size
*/
void Processing(UNUSED uint sc,
                uint sc_lcl_off,
                uint top_lim,
                int bot_lim, // bot_lim could be negative at lower boundary padding
                __private _FLOAT_ACCUM* __restrict pvt_accum,
                __local _FLOAT* __restrict lcl_bot,
                __private _FLOAT* __restrict top_dat,
                uint chunk_id)
{
    for(int l = top_lim; l >= bot_lim; --l)
    {
        for(uint m = 0; m < MLO_IN_TILE0; ++m)
        {
            for(uint n = 0; n < MLO_FILTER_SIZE0; ++n)
            {
                for(uint c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
                {

#if MLO_IN_WIDTH_N_LOOPS > 1 && MLO_FILTER_PAD0 > 0
                    uint bot_off = sc_lcl_off + n + m;
                    if(chunk_id == 0 || chunk_id == MLO_IN_WIDTH_N_LOOPS - 1)
                    {
                        bot_off = bot_off + c * MLO_IN_LCL_PADDED_SZ;
                    }
                    else
                    {
                        bot_off = bot_off + c * MLO_IN_LCL_UNPADDED_SZ;
                    }
#else
                    (void)chunk_id;
                    uint bot_off = sc_lcl_off + c * MLO_IN_LCL_SZ + n + m;
#endif

                    _FLOAT bot_val = lcl_bot[bot_off];

                    for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
                    {
                        uint pvt_top_off =
                            k * MLO_IN_TILE0 * MLO_FILTER_SIZE1 + (top_lim - l) * MLO_IN_TILE0 + m;
                        uint pvt_accum_off =
                            (k * MLO_N_LCL_IN_MAPS + c) * MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0 +
                            l * MLO_FILTER_SIZE0 + n;

                        _FLOAT top_val = top_dat[pvt_top_off];
                        pvt_accum[pvt_accum_off]
                            // each wk-it process an input
                            += CVT_FLOAT2ACCUM(bot_val) * CVT_FLOAT2ACCUM(top_val);
                    }
                }
            }
        }
    }
}

void moveOutputUp(__private _FLOAT* __restrict top_dat)
{
    // move up output to reduce overfetch
    for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
    {
        for(uint j = 0; j < MLO_FILTER_SIZE1 - 1; ++j)
        {
            for(uint i = 0; i < MLO_IN_TILE0; ++i)
            {
                uint pvt_off_n = k * MLO_IN_TILE0 * MLO_FILTER_SIZE1 + j * MLO_IN_TILE0 + i;
                uint pvt_off_o = k * MLO_IN_TILE0 * MLO_FILTER_SIZE1 + (j + 1) * MLO_IN_TILE0 + i;
                top_dat[pvt_off_n] = top_dat[pvt_off_o];
            }
        }
    }
}

void zeroInitLDS(uint lcl_id, __local _FLOAT* __restrict lcl_bot)
{
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint i = lcl_id; i < (MLO_LCL_SZ); i += MLO_GRP_SZ)
    {
        lcl_bot[i] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

void spanReadingOutput(int spn,
                       int k,
                       int j,
                       int top_df_off,
                       _FLOAT mask,
                       __private _FLOAT* __restrict top_dat,
                       const __global _FLOAT* __restrict top_df,
                       uint chunk_id)
{
    int pvt_off                     = k * MLO_IN_TILE0 * MLO_FILTER_SIZE1 + j * MLO_IN_TILE0;
    const __global _FLOAT* top_df_p = &top_df[top_df_off];
#if MLO_OUT_WIDTH_LAST_CHUNK_VALID_SPANS > 0
    if(chunk_id == MLO_OUT_WIDTH_N_LOOPS - 1 && spn < MLO_OUT_WIDTH_LAST_CHUNK_VALID_SPANS)
    {
        if(spn == MLO_OUT_WIDTH_LAST_CHUNK_VALID_SPANS - 1)
        {
            uint i = 0;
            for(; i < MLO_OUT_WIDTH_LAST_CHUNK_VALID_PIXELS_IN_LAST_SPAN; ++i)
            {
                top_dat[pvt_off + i] = top_df_p[i] * mask;
            }
            for(; i < MLO_IN_TILE0; ++i)
            {
                top_dat[pvt_off + i] = 0;
            }
        }
        else
        {
            for(uint i = 0; i < MLO_IN_TILE0; ++i)
            {
                top_dat[pvt_off + i] = top_df_p[i] * mask;
            }
        }
    }
    else if(chunk_id != MLO_OUT_WIDTH_N_LOOPS - 1)
#endif
    {
#if MLO_OUT_N_PIXS_OFF > 0
        if(spn == MLO_N_SPANS_PER_SCAN - 1)
        {
            uint i = 0;
            for(; i < MLO_OUT_N_PIXS_OFF; ++i)
            {
                top_dat[pvt_off + i] = top_df_p[i] * mask;
            }
            for(; i < MLO_IN_TILE0; ++i)
            {
                top_dat[pvt_off + i] = 0;
            }
        }
        else
#else
        (void)spn;
#endif
        {
            (void)chunk_id;
            for(uint i = 0; i < MLO_IN_TILE0; ++i)
            {
                top_dat[pvt_off + i] = top_df_p[i] * mask;
            }
        }
    }
}

/*********************************************************************************************************
// wrw algorithm for large filters
// idea:
// split output scan-line on number of spans by the  MLO_IN_TILE0 (2 for example)
// 1 scan-line has ((MLO_OUT_WIDTH_CHUNK + MLO_IN_TILE0 - 1/MLO_IN_TILE0) spans
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

// kerenl handles 5x5, 3x3 with padding
// small images in 1 short- MLO_N_GENERIC_LOOPS == 0
// big images  in 2 blocks - MLO_IN_N_VERT_LOOPS == 2 or multiple blocks - MLO_IN_N_VERT_LOOPS > 2
// there are prolog and apilog that deal with top/bottom padding.
// left/right padding handles as a LDS border pixels zeroed at the beginning.

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

    __local _FLOAT lcl[(MLO_LCL_SZ) + 1];
    __local _FLOAT* lcl_bot = lcl;

    uint lcl_id = get_local_id(0);

    uint c_idx_base = get_group_id(0); // input map index base

    uint o_idx_base = get_group_id(1); // output map index base

    uint ib_base = get_group_id(2);

    uint ib = ib_base * (MLO_N_BATCH_LOOPS * MLO_N_LCL_BATCHS);

    uint c_idx = c_idx_base * MLO_N_LCL_IN_MAPS; // input map index

    uint o_idx = o_idx_base * (MLO_N_LCL_OUT_MAPS * MLO_OUT_STACKS); // output map index

#if MLO_READ_PARTIAL_N_LCL_IN_MAPS
    uint n_in_map_reads = MLO_N_INPUTS >= c_idx + MLO_N_LCL_IN_MAPS
                              ? MLO_N_LCL_IN_MAPS
                              : (MLO_N_INPUTS >= c_idx ? MLO_N_INPUTS - c_idx : 0);
#else
    uint n_in_map_reads = MLO_N_LCL_IN_MAPS;
#endif

    uint gbl_in_off  = c_idx * MLO_IN_CHANNEL_STRIDE + ib * MLO_IN_BATCH_STRIDE;
    uint gbl_out_off = o_idx * MLO_OUT_CHANNEL_STRIDE + ib * MLO_OUT_BATCH_STRIDE;
    // 1 span per wk_item, total scanline with MLO_N_SPANS_PER_SCAN spans
    // TODO: more than 1 input
    uint o = lcl_id / MLO_N_SPANS_PER_SCAN;
#if MLO_N_SPANS_PER_SCAN & (MLO_N_SPANS_PER_SCAN - 1)
    uint spn = iMod(lcl_id, o, MLO_N_SPANS_PER_SCAN);
#else
    uint spn            = lcl_id & (MLO_N_SPANS_PER_SCAN - 1);
#endif
    //	bool scan_lead = (o*MLO_N_SPANS_PER_SCAN == lcl_id);

    uint lcl_bot_off     = spn * MLO_IN_TILE0;
    uint out_wk_item_off = o * MLO_OUT_CHANNEL_STRIDE + lcl_bot_off;
    gbl_out_off += out_wk_item_off;
    // no output out of range
    gbl_out_off = (o_idx + o < MLO_N_OUTPUTS && o < MLO_OUT_STACKS) ? gbl_out_off : 0;

#define MLO_TOP_DAT_SZ (MLO_N_LCL_OUT_MAPS * MLO_IN_TILE0 * MLO_FILTER_SIZE1)

    __private _FLOAT top_dat[MLO_TOP_DAT_SZ];

    for(uint i = 0; i < MLO_TOP_DAT_SZ; ++i)
    {
        top_dat[i] = 0;
    }

#define MLO_ACCUM_SZ (MLO_N_LCL_OUT_MAPS * MLO_N_LCL_IN_MAPS * MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0)

    __private _FLOAT_ACCUM pvt_accum[MLO_ACCUM_SZ];

    for(uint i = 0; i < MLO_ACCUM_SZ; ++i)
    {
        pvt_accum[i] = (_FLOAT_ACCUM)0;
    }

    // zero out LDS
    for(uint i = lcl_id; i < (MLO_LCL_SZ); i += MLO_GRP_SZ)
    {
        lcl[i] = 0;
    }

    // over all batches
    uint bend = ib + MLO_N_BATCH_LOOPS * MLO_N_LCL_BATCHS;
    bend      = bend > MLO_BATCH_SZ ? MLO_BATCH_SZ : bend;

    for(uint b = ib; b < bend; ++b,
             gbl_in_off += MLO_N_LCL_BATCHS * MLO_IN_BATCH_STRIDE,
             gbl_out_off += MLO_N_LCL_BATCHS * MLO_OUT_BATCH_STRIDE)
    {
        barrier(CLK_LOCAL_MEM_FENCE);

        uint ichunk = 0;
#if MLO_IN_WIDTH_N_LOOPS > 1
        for(; ichunk < MLO_IN_WIDTH_N_LOOPS; ++ichunk)
#endif
        {
            uint filter_shift = 0;
#if MLO_IN_WIDTH_N_LOOPS > 1
            if(ichunk != 0)
            {
                filter_shift = MLO_FILTER_SIZE0 - 1;
            }
#endif
#if MLO_IN_WIDTH_N_LOOPS > 1 && MLO_FILTER_PAD0 > 0
            uint lcl_width = (ichunk == 0 || ichunk == MLO_IN_WIDTH_N_LOOPS - 1)
                                 ? MLO_IN_LCL_PADDED_WIDTH
                                 : MLO_IN_LCL_UNPADDED_WIDTH;
#else
            uint lcl_width = MLO_IN_LCL_WIDTH;
#endif
            // top border input block
            uint gbl_in_scan_off = gbl_in_off + ichunk * MLO_IN_WIDTH_CHUNK - ichunk * filter_shift;
            uint gbl_out_scan_off = gbl_out_off + ichunk * MLO_OUT_WIDTH_CHUNK;

            // read input map
            readInput(
                lcl_id, gbl_in_scan_off, n_in_map_reads, MLO_IN_VERT_READS, bot, lcl_bot, ichunk);

            // move input pointer
            gbl_in_scan_off += MLO_IN_STRIDE * MLO_IN_EXTENT1;

            for(uint i = 0; i < MLO_TOP_DAT_SZ; ++i)
            {
                top_dat[i] = 0;
            }

            // prefetch output
            uint gbl_out_scan_off1 = gbl_out_scan_off;
            for(uint k = 0; k < MLO_N_LCL_OUT_MAPS;
                ++k, gbl_out_scan_off1 += MLO_OUT_STACKS * MLO_OUT_CHANNEL_STRIDE)
            {
                for(uint j = 0; j < MLO_FILTER_SIZE1 - 1; ++j)
                {
                    // loop around all output maps
                    uint top_df_off = gbl_out_scan_off1 + j * MLO_OUT_STRIDE;
                    _FLOAT mask     = 1;
#if MLO_IN_HEIGHT != MLO_OUT_HEIGHT || MLO_FILTER_SIZE1 - 1 > MLO_OUT_HEIGHT
                    top_df_off = (j < MLO_OUT_HEIGHT) ? top_df_off : 0;
                    mask       = (j < MLO_OUT_HEIGHT) ? 1 : 0;
#endif

                    spanReadingOutput(spn, k, j, top_df_off, mask, top_dat, top_df, ichunk);
                }
            }

            gbl_out_scan_off += (MLO_FILTER_SIZE1 - 1) * MLO_OUT_STRIDE;

            uint sc         = 0;
            uint sc_lcl_off = lcl_bot_off;

            // prolog
            // handling padding

            // top padding
            for(; sc < MLO_FILTER_SIZE1 - MLO_FILTER_PAD1 - 1; ++sc, sc_lcl_off += lcl_width)
            {
                Processing(
                    sc, sc_lcl_off, sc + MLO_FILTER_PAD1, 0, pvt_accum, lcl_bot, top_dat, ichunk);
            }

#ifdef __AMDGCN__
#pragma unroll 2
#endif

#if MLO_IN_N_VERT_LOOPS == 1
            for(; sc < MLO_IN_HEIGHT + MLO_FILTER_PAD1 - MLO_FILTER_SIZE1 + 1;
#else
            for(; sc < MLO_IN_EXTENT1;
#endif
                ++sc, gbl_out_scan_off += MLO_OUT_STRIDE, sc_lcl_off += lcl_width)
            {

                for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
                {
                    uint top_df_off =
                        gbl_out_scan_off + k * MLO_OUT_STACKS * MLO_OUT_CHANNEL_STRIDE;
                    _FLOAT mask = 1;

#if MLO_IN_HEIGHT != MLO_OUT_HEIGHT || MLO_FILTER_SIZE1 - 1 > MLO_OUT_HEIGHT
                    top_df_off = ((sc + MLO_FILTER_PAD1) < MLO_OUT_HEIGHT) ? top_df_off : 0;
                    mask       = ((sc + MLO_FILTER_PAD1) < MLO_OUT_HEIGHT) ? 1 : 0;
#endif

                    spanReadingOutput(
                        spn, k, (MLO_FILTER_SIZE1 - 1), top_df_off, mask, top_dat, top_df, ichunk);
                }

                // processing
                Processing(
                    sc, sc_lcl_off, MLO_FILTER_SIZE1 - 1, 0, pvt_accum, lcl_bot, top_dat, ichunk);

                // move up output to reduce overfetch
                moveOutputUp(top_dat);
            }

            // non-border input blocks
            for(uint i_loop = 0; i_loop < MLO_N_GENERIC_LOOPS;
                ++i_loop, gbl_in_scan_off += MLO_IN_STRIDE * MLO_IN_EXTENT1)
            {
                barrier(CLK_LOCAL_MEM_FENCE);

                readInput(lcl_id,
                          gbl_in_scan_off,
                          n_in_map_reads,
                          MLO_IN_VERT_READS,
                          bot,
                          lcl_bot,
                          ichunk);

                // point to the start of the local buffer

                sc_lcl_off = lcl_bot_off;

                for(; sc < (i_loop + 2) * MLO_IN_EXTENT1;
                    ++sc, gbl_out_scan_off += MLO_OUT_STRIDE, sc_lcl_off += lcl_width)
                {

                    for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
                    {
                        uint top_df_off =
                            gbl_out_scan_off + k * MLO_OUT_STACKS * MLO_OUT_CHANNEL_STRIDE;
                        _FLOAT mask = 1;

#if MLO_IN_HEIGHT != MLO_OUT_HEIGHT
                        top_df_off = ((sc + MLO_FILTER_PAD1) < MLO_OUT_HEIGHT) ? top_df_off : 0;
                        mask       = ((sc + MLO_FILTER_PAD1) < MLO_OUT_HEIGHT) ? 1 : 0;
#endif

                        spanReadingOutput(spn,
                                          k,
                                          (MLO_FILTER_SIZE1 - 1),
                                          top_df_off,
                                          mask,
                                          top_dat,
                                          top_df,
                                          ichunk);
                    }

                    // processing
                    Processing(sc,
                               sc_lcl_off,
                               MLO_FILTER_SIZE1 - 1,
                               0,
                               pvt_accum,
                               lcl_bot,
                               top_dat,
                               ichunk);

                    // move up output to reduce overfetch
                    moveOutputUp(top_dat);
                }
            }

            // bottom border block

            for(int i_loop = 0; i_loop < (MLO_IN_N_VERT_LOOPS - MLO_N_GENERIC_LOOPS - 1);
                ++i_loop, gbl_in_scan_off += MLO_IN_STRIDE * MLO_IN_EXTENT1)
            {
                barrier(CLK_LOCAL_MEM_FENCE);

                // read 1 scan line less
                // padding processing takes care of the bottom border.

#define MLO_LAST_VERT_READS (MLO_IN_HEIGHT - MLO_IN_EXTENT1 * (MLO_IN_N_VERT_LOOPS - 1))

                readInput(lcl_id,
                          gbl_in_scan_off,
                          n_in_map_reads,
                          MLO_LAST_VERT_READS,
                          bot,
                          lcl_bot,
                          ichunk);

                // point to the start of the local buffer
                sc_lcl_off = lcl_bot_off;

#ifndef MLO_DISABLE_PRAGMA_UNROLL_COMPILER_SWDEV_200074_WORKAROUND
#pragma unroll 3
#endif
                for(; sc < MLO_IN_HEIGHT + MLO_FILTER_PAD1 - MLO_FILTER_SIZE1 + 1;
                    ++sc, gbl_out_scan_off += MLO_OUT_STRIDE, sc_lcl_off += lcl_width)
                {

                    for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
                    {
                        uint top_df_off =
                            gbl_out_scan_off + k * MLO_OUT_STACKS * MLO_OUT_CHANNEL_STRIDE;
                        _FLOAT mask = 1;

                        spanReadingOutput(spn,
                                          k,
                                          (MLO_FILTER_SIZE1 - 1),
                                          top_df_off,
                                          mask,
                                          top_dat,
                                          top_df,
                                          ichunk);
                    }

                    // processing
                    Processing(sc,
                               sc_lcl_off,
                               MLO_FILTER_SIZE1 - 1,
                               0,
                               pvt_accum,
                               lcl_bot,
                               top_dat,
                               ichunk);

                    // move up output to reduce overfetch
                    moveOutputUp(top_dat);
                }
            }

            // epilog
            // handling padding

            for(; sc < MLO_IN_HEIGHT; ++sc, sc_lcl_off += lcl_width)
            {

                // processing
                Processing(sc,
                           sc_lcl_off,
                           MLO_FILTER_SIZE1 - 1,
                           MLO_FILTER_SIZE1 - (MLO_IN_HEIGHT + MLO_FILTER_PAD1 - sc),
                           pvt_accum,
                           lcl_bot,
                           top_dat,
                           ichunk);

                // move up output to reduce overfetch
                moveOutputUp(top_dat);
            } // epilog handling loop

#if MLO_OUT_WIDTH_N_LOOPS > 1 && MLO_FILTER_PAD0 > 0
            zeroInitLDS(lcl_id, lcl_bot);
#endif
        } // row-level  loop
    }     // batch-level loop

    // final summation over all output maps and each filter row
    // this coudl be done with log but it negligeble anyway
    for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
    {
        for(uint c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
        {

            for(uint l = 0; l < MLO_FILTER_SIZE1; ++l)
            {
                barrier(CLK_LOCAL_MEM_FENCE);

                for(uint n = 0; n < MLO_FILTER_SIZE0; ++n)
                {
                    uint pvt_off =
                        (k * MLO_N_LCL_IN_MAPS + c) * MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0 +
                        l * MLO_FILTER_SIZE0 + n;

                    lcl[lcl_id * MLO_FILTER_SIZE0 + n] = CVT_ACCUM2FLOAT(pvt_accum[pvt_off]);
                }

                barrier(CLK_LOCAL_MEM_FENCE);

                if(spn == 0)
                {
                    for(uint s = 0; s < MLO_N_SPANS_PER_SCAN - 1; ++s)
                    {

                        for(uint n = 0; n < MLO_FILTER_SIZE0; ++n)
                        {
                            uint pvt_off =
                                (k * MLO_N_LCL_IN_MAPS + c) * MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0 +
                                l * MLO_FILTER_SIZE0 + n;
                            pvt_accum[pvt_off] +=
                                CVT_FLOAT2ACCUM(lcl[(lcl_id + s + 1) * MLO_FILTER_SIZE0 + n]);
                        }
                    }
                }
            }
        }
    }

    // output
    // inputs are outputs
    // TODO : for more than 1 input

    uint wei_df_off =
        (((ib / MLO_N_BATCH_LOOPS) * MLO_N_OUTPUTS + o_idx + o) * (uint)MLO_WEI_BATCH_STRIDE)
        // this input channel
        + mul24(c_idx, (uint)MLO_WEI_CHANNEL_STRIDE);

    for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
    {
        for(uint c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
        {
            if(spn == 0 && c < n_in_map_reads && o_idx + o + k * MLO_OUT_STACKS < MLO_N_OUTPUTS &&
               o < MLO_OUT_STACKS)
            {
                for(uint i = 0; i < (MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0); ++i)
                {
                    weights_df[wei_df_off + k * MLO_OUT_STACKS * MLO_WEI_BATCH_STRIDE +
                               c * MLO_WEI_CHANNEL_STRIDE + i] =
                        CVT_ACCUM2FLOAT(pvt_accum[(k * MLO_N_LCL_IN_MAPS + c) * MLO_FILTER_SIZE1 *
                                                      MLO_FILTER_SIZE0 +
                                                  i]);
                }
            }
        }
    }
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

    _FLOAT_ACCUM pvt_accum_wei[MLO_UT_READ_UNIT] = {0};
    //	for (uint i = 0; i < MLO_UT_READ_UNIT; ++i)
    //	{
    //		pvt_accum_wei[i] = 0;
    //	}

    int batch_loop = (MLO_BATCH_SZ + (MLO_N_BATCH_LOOPS * MLO_N_LCL_BATCHS) - 1) /
                     (MLO_N_BATCH_LOOPS * MLO_N_LCL_BATCHS);

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
