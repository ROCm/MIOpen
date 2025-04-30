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

#define DBG_OUT_OF_RNGE 0
#define DBG_PRINTF 0

// filter size for all filters with small n of input maps (first layer)
// split a long filter by stride

#ifndef MLO_N_FILTER_SPLITS1
#define MLO_N_FILTER_SPLITS1 ((MLO_FILTER_SIZE1 + MLO_FILTER_STRIDE1 - 1) / MLO_FILTER_STRIDE1)
#endif
#ifndef MLO_N_FILTER_SPLITS0
#define MLO_N_FILTER_SPLITS0 ((MLO_FILTER_SIZE0 + MLO_FILTER_STRIDE0 - 1) / MLO_FILTER_STRIDE0)
#endif
#ifndef MLO_OUT_PIX_TILE0
#define MLO_OUT_PIX_TILE0 MLO_N_FILTER_SPLITS0
#endif
// processing arrangement
// generate full output width
// extent1 == MLO_GRP_SZ / MLO_PROCESING_WIDTH
#ifndef MLO_OUT_EXTENT1
#define MLO_PROCESSING_WIDTH ((MLO_OUT_WIDTH + MLO_OUT_PIX_TILE0 - 1) / MLO_OUT_PIX_TILE0)
#define MLO_OUT_EXTENT1 (MLO_GRP_SZ / MLO_PROCESSING_WIDTH)
#endif

#define MLO_WEI_LCL_WIDTH MLO_FILTER_SIZE0 //(MLO_N_FILTER_SPLITS0*MLO_FILTER_STRIDE0)
#define MLO_WEI_EXTENT1 MLO_N_FILTER_SPLITS1
#define MLO_WEI_SZ (MLO_WEI_EXTENT1 * MLO_WEI_LCL_WIDTH)
// LDS size
#ifndef MLO_WEI_LCL_SZ
#define MLO_WEI_LCL_SZ (MLO_WEI_SZ * MLO_N_LCL_OUT_MAPS * MLO_N_LCL_IN_MAPS)
#endif

#ifndef MLO_IN_LCL_HEIGHT
#define MLO_IN_LCL_HEIGHT (MLO_OUT_EXTENT1 + MLO_N_FILTER_SPLITS1 - 1)
#endif
// there is an assumption that the scanline fits into LDS
#define MLO_N_IN_HORIZ_PIX_READS (MLO_IN_WIDTH)
#ifndef MLO_N_IN_HORIZ_READS
#define MLO_N_IN_HORIZ_READS ((MLO_N_IN_HORIZ_PIX_READS + MLO_READ_UNIT - 1) / MLO_READ_UNIT)
#endif

#define MLO_IN_N_PIXS_OFF \
    (MLO_N_IN_HORIZ_PIX_READS - (MLO_N_IN_HORIZ_PIX_READS / MLO_READ_UNIT) * MLO_READ_UNIT)

#define MLO_IN_LCL_WIDTH (MLO_N_IN_HORIZ_READS * MLO_READ_UNIT + 2 * MLO_FILTER_PAD0)
#define MLO_IN_LCL_SZ (MLO_IN_LCL_WIDTH * MLO_IN_LCL_HEIGHT)
// LDS size
#define MLO_TOTAL_IN_LCL_SZ (MLO_N_LCL_BATCHS * MLO_IN_LCL_SZ * MLO_N_LCL_IN_MAPS)

#ifndef MLO_LCL_MEM_SZ
#define MLO_LCL_MEM_SZ (MLO_WEI_LCL_SZ + MLO_TOTAL_IN_LCL_SZ)
#endif

// number of loops to flush put full output map
#define MLO_N_OUT_BLKS \
    1 //((MLO_OUT_HEIGHT + (MLO_OUT_PIX_TILE1*MLO_N_OUT_FOLDS1) -1) /
      //(MLO_OUT_PIX_TILE1*MLO_N_OUT_FOLDS1))

#define MLO_HW_WAVE_ID_SETTING 1

#if defined(__AMDGCN__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored \
    "-Wunknown-warning-option" // clang in ROCm 4.3 does not support "reserved-identifier".
#pragma clang diagnostic ignored "-Wreserved-identifier"
#pragma clang diagnostic pop // "-Wreserved-identifier"
#define uniform(x) __builtin_amdgcn_readfirstlane(x)
#else
#define uniform(x) (x)
#endif

uint getWaveId()
{
    uint wave_id = 0;

#if MLO_HW_WAVE_ID_SETTING && defined(__AMDGCN__)
    // (local_id/wavesize) has the same value in all workitems.
    // Make it scalar to enable scalarization optimizations.

    wave_id = __builtin_amdgcn_readfirstlane((uint)(get_local_id(0) >> MLO_LG2_WAVE_SZ));
    // Alternate implementation:
    //__asm__ ("v_readfirstlane_b32 %0, %1" : "=s" (wave_id) : "v" ((uint)(get_local_id(0) >>
    // MLO_LG2_WAVE_SZ)) );

#elif MLO_HW_WAVE_ID_SETTING
    // FIXME Conduct enabling from the host code.
    extern __attribute__((const)) uint __hsail_get_dynwave_id(void);
    wave_id = __hsail_get_dynwave_id();
    wave_id &= MLO_N_WAVES_MASK;

#else
    wave_id = (get_local_id(0) >> MLO_LG2_WAVE_SZ);
#endif
    return (wave_id);
}

uint getWaveLocalId()
{
    uint lcl_wave_id = get_local_id(0) & ((1 << MLO_LG2_WAVE_SZ) - 1);
    return (lcl_wave_id);
}

uint getLocalId(uint wave_id, uint wave_lcl_id)
{
    uint lcl_id = (wave_id << MLO_LG2_WAVE_SZ) + wave_lcl_id;
    return (lcl_id);
}

#include "math_ops.h"

void ReduceKernel(__local _FLOAT* lcl_blob,
                  _FLOAT* weights_accum,
                  uint lcl_id,
                  uint scan_lcl,
                  uint sum_stride,
                  uint unit_len,
                  UNUSED bool debug)
{
    for(uint j = (sum_stride >> 1); j > scan_lcl; j >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        for(uint i = 0; i < unit_len; ++i)
        {

            weights_accum[i] += lcl_blob[(lcl_id + j) * unit_len + i];

            lcl_blob[lcl_id * unit_len + i] = weights_accum[i];
        }
    }
}

#if MLO_DIR_FORWARD == 1

// TO DO: remove f_s and c from offest calculation
void fetchWeights(uint c,
                  uint k_idx,
                  uint f_s,
                  uint lcl_id,
                  uint wei_read,
                  uint gbl_wei_off,
                  __local _FLOAT* wei_mem,
                  const __global _FLOAT* weights)
{
    // read weights by stride
    for(uint w = lcl_id; w < (wei_read / MLO_FILTER_SIZE0) * MLO_N_LCL_OUT_MAPS; w += MLO_GRP_SZ)
    {
        uint k = iDiv_legacy(w, (wei_read / MLO_FILTER_SIZE0));
        uint j = iMod(w, k, (wei_read / MLO_FILTER_SIZE0));
        int wei_off =
            ((j * MLO_FILTER_STRIDE1 + f_s) < MLO_FILTER_SIZE1 && k_idx + k < MLO_N_OUTPUTS)
                ? gbl_wei_off + k * MLO_WEI_BATCH_STRIDE + c * MLO_WEI_CHANNEL_STRIDE +
                      (j * MLO_FILTER_STRIDE1 + f_s) * MLO_FILTER_SIZE0
                : 0;
        const __global _FLOAT* wei_p = &weights[wei_off];

        for(uint i = 0; i < MLO_FILTER_SIZE0; ++i)
        {
            _FLOAT weight                                       = wei_p[i];
            wei_mem[k * MLO_WEI_SZ + j * MLO_WEI_LCL_WIDTH + i] = weight;
#if DBG_OUT_OF_RNGE == 1
            if(wei_off + i >= MLO_N_OUTPUTS * MLO_N_INPUTS * MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0)
            {
                printf("K:err:weights out-of-range");
            }
#endif
        }
#if 0
		if (ob == 0 && k == 1)
		{
			printf("G:w: %d %d %d %d %f %f\n",
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
}

void fetchData(uint f_s,
               uint lcl_id,
               uint lcl_scan,
               uint n_reads,
               int in_y,
               uint gbl_in_scan_off,
               __local _FLOAT* bot_mem,
               const __global _FLOAT* bot)
{
    __private _FLOAT in_rd_data[MLO_READ_UNIT];

    for(uint p4 = lcl_id, c_scan = 0; p4 < MLO_N_IN_HORIZ_READS * n_reads * MLO_N_LCL_BATCHS;
        p4 += MLO_GRP_SZ)
    {
        uint b  = 0;
        uint t0 = p4;
#if MLO_N_LCL_BATCHS > 1
        b  = iDiv_legacy(p4, MLO_N_IN_HORIZ_READS * n_reads);
        t0 = iMod(p4, b, MLO_N_IN_HORIZ_READS * n_reads);
#endif
#if MLO_N_IN_HORIZ_READS & (MLO_N_IN_HORIZ_READS - 1)
        c_scan      = iDiv_legacy(t0, MLO_N_IN_HORIZ_READS);
        uint c_pix4 = iMod(t0, c_scan, MLO_N_IN_HORIZ_READS);
#else
        c_scan      = t0 / MLO_N_IN_HORIZ_READS;
        uint c_pix4 = t0 & (MLO_N_IN_HORIZ_READS - 1);
#endif
        int in_scan = (c_scan + lcl_scan) * MLO_FILTER_STRIDE1 + f_s;

        for(uint i = 0; i < MLO_READ_UNIT; ++i)
        {
            in_rd_data[i] = 0;
        }

        if(0 <= in_y + in_scan && in_y + in_scan < MLO_IN_HEIGHT)
        {

            int gbl_off = gbl_in_scan_off + b * MLO_IN_BATCH_STRIDE + in_scan * MLO_IN_STRIDE +
                          c_pix4 * MLO_READ_UNIT;
            const __global _FLOAT* bot_p = &bot[gbl_off];
// still problems with unaligned LDS access
#if MLO_IN_N_PIXS_OFF > 0
            if(c_pix4 == MLO_N_IN_HORIZ_READS - 1)
            {
                uint i = 0;
                for(; i < MLO_IN_N_PIXS_OFF; ++i)
                {
                    in_rd_data[i] = bot_p[i];
                }
                //								for (; i <
                // MLO_READ_UNIT;
                //++i)
                //								{
                //									in_rd_data[i]
                //=
                // 0;
                //								}
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
        int lcl_off =
            (lcl_scan + c_scan) * MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 + c_pix4 * MLO_READ_UNIT;
        for(uint i = 0; i < MLO_READ_UNIT; ++i)
        {
            bot_mem[lcl_off + i] = in_rd_data[i];
        }
    }
}

void Convolve(uint ex_row,
              uint ex_pix,
              uint l,
              uint m,
              uint wei_h,
              uint bot_h,
              __local _FLOAT* __restrict wei_mem,
              __local _FLOAT* __restrict bot_mem,
              __private _FLOAT_ACCUM* pvt_accum)
{
    // only for 11
    __private _FLOAT wei_vals[MLO_N_LCL_OUT_MAPS * MLO_N_FILTER_SPLITS0];
    __private _FLOAT in_vals[(MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 1)];

    // read all weights
    for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
    {
        for(uint i = 0; i < wei_h; ++i)
        {
            wei_vals[k * MLO_N_FILTER_SPLITS0 + i] =
                wei_mem[k * MLO_WEI_SZ + m * MLO_WEI_LCL_WIDTH + i * MLO_FILTER_STRIDE0 + l];
        }
    }

    // convolve
    for(uint i = 0; i < bot_h; ++i)
    {
        in_vals[i] = bot_mem[(ex_row + m) * MLO_IN_LCL_WIDTH + ex_pix * MLO_FILTER_STRIDE0 +
                             i * MLO_FILTER_STRIDE0 + l];
    }

    for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
    {
        for(uint n = 0; n < MLO_OUT_PIX_TILE0; ++n)
        {

            for(uint i = 0; i < wei_h; ++i)
            {
                _FLOAT in_val  = in_vals[n + i];
                _FLOAT wei_val = wei_vals[k * MLO_N_FILTER_SPLITS0 + i];
                pvt_accum[k * MLO_OUT_PIX_TILE0 + n] +=
                    CVT_FLOAT2ACCUM(wei_val) * CVT_FLOAT2ACCUM(in_val);
#if 0
				if (wei_val * in_val != 0 && ib + b + bb == 0 && k_idx + k == 1 && out_y + ex_row == 0 && ex_pix + n == 0)
				{
					printf("G:c: %d %d %d %d %d %d %d %d %d %d %d %d %f %f %f %f\n",
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
} // l

/*********************************************************************************************************
// frw algorithm for large filters
// idea:
// process 3 output pixel per wk-item, 19 wk-items per output scan,
// 13 output sacn-line per group of 256
// read (13+2) input scan-lines 4 scan-lines apart from 2 batches
// convolve with 3 filters rows 4 rowes apart from 4(8) filter banks.


**********************************************************************************************************/

__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2))) __kernel void
MIOpenCvFwd11x11(const __global _FLOAT* __restrict bot,
                 const __global _FLOAT* __restrict weights,
#if MLO_CONV_BIAS == 1
                 const __global _FLOAT* __restrict bias,
#endif
                 __global _FLOAT* __restrict top,
                 UNUSED _FLOAT padding_val)
{

    __local _FLOAT lcl_mem[MLO_LCL_MEM_SZ];
    __local _FLOAT* bot_mem = lcl_mem;
    __local _FLOAT* wei_mem = lcl_mem + MLO_TOTAL_IN_LCL_SZ;

    uint lcl_id = get_local_id(0);

    uint ob = get_group_id(0); // output map extent id

    uint k_idx = get_group_id(1) * (MLO_N_LCL_OUT_MAPS); // input map index based

    uint ib_idx = get_group_id(2) * MLO_N_LCL_BATCHS; // batch idx

    uint ib = ib_idx;

    int gbl_in_off   = /*c_idx * MLO_IN_CHANNEL_STRIDE + */ ib * MLO_IN_BATCH_STRIDE;
    uint gbl_wei_off = k_idx * MLO_WEI_BATCH_STRIDE;
    uint out_y       = ob * MLO_OUT_EXTENT1;
    int in_y         = out_y * MLO_FILTER_STRIDE1 - MLO_FILTER_PAD1;
    gbl_in_off += in_y * MLO_IN_STRIDE;

#define MLO_ACCUM_SZ \
    (MLO_OUT_PIX_TILE1 * MLO_OUT_PIX_TILE0 * MLO_N_LCL_OUT_MAPS * MLO_N_LCL_IN_MAPS)

    __private _FLOAT_ACCUM pvt_accum[MLO_ACCUM_SZ];

    // zero out LDS
    for(uint i = lcl_id; i < (MLO_LCL_MEM_SZ); i += MLO_GRP_SZ)
    {
        lcl_mem[i] = 0;
    }

// processing arrangement
#if MLO_PROCESSING_WIDTH & (MLO_PROCESSING_WIDTH - 1)
    uint ex_row = iDiv_legacy(lcl_id, MLO_PROCESSING_WIDTH);
    uint ex_col = iMod(lcl_id, ex_row, MLO_PROCESSING_WIDTH);
#else
    uint ex_row = lcl_id / MLO_PROCESSING_WIDTH;
    uint ex_col = lcl_id & (MLO_PROCESSING_WIDTH - 1);
#if MLO_PROCESSING_WIDTH >= 64
    ex_row = uniform(ex_row);
#endif
#endif
    uint ex_pix = ex_col * MLO_OUT_PIX_TILE0;

    // over all batches

    for(uint b = 0; b < MLO_N_BATCH_LOOPS; ++b, gbl_in_off += MLO_IN_BATCH_STRIDE)
    {

        int gbl_in_scan_off0 = gbl_in_off;

        // generate pixels from all MLO_N_LCL_OUT_MAPS output maps

        for(uint i = 0; i < MLO_ACCUM_SZ; ++i)
        {
            pvt_accum[i] = CVT_FLOAT2ACCUM(0);
        }

// all input maps
#ifdef __AMDGCN__
#pragma unroll 4
#endif
        for(uint c = 0, gbl_in_scan_off = gbl_in_scan_off0; c < MLO_N_INPUTS;
            ++c, gbl_in_scan_off += MLO_IN_CHANNEL_STRIDE)
        {
            uint f_s = 0;
            for(; f_s < MLO_FILTER_STRIDE1 - 1; ++f_s)
            {

                barrier(CLK_LOCAL_MEM_FENCE);

                // get a set of horizaontal taps
                fetchWeights(c, k_idx, f_s, lcl_id, MLO_WEI_SZ, gbl_wei_off, wei_mem, weights);

                // fetch a set of input scanlines

                uint n_reads = MLO_IN_LCL_HEIGHT; // ((ob == 0 && (f_s < MLO_FILTER_PAD1)) || (ob ==
                                                  // get_local_size(0) - 1 && (MLO_FILTER_STRIDE1 -
                                                  // f_s) < MLO_FILTER_PAD1)) ? MLO_IN_LCL_HEIGHT -
                                                  // 1 : MLO_IN_LCL_HEIGHT;
                uint lcl_scan = 0;                // (ob == 0 && (f_s < MLO_FILTER_PAD1)) ? 1 : 0;

                fetchData(f_s, lcl_id, lcl_scan, n_reads, in_y, gbl_in_scan_off, bot_mem, bot);

                barrier(CLK_LOCAL_MEM_FENCE);

// convolution
// along vertical filter
#pragma unroll
                for(uint m = 0; m < MLO_N_FILTER_SPLITS1; ++m)
                {

                    // first 3 splits
                    uint l;
                    for(l = 0; l < MLO_FILTER_STRIDE0 - 1; ++l)
                    {

                        Convolve(ex_row,
                                 ex_pix,
                                 l,
                                 m,
                                 (MLO_N_FILTER_SPLITS0),
                                 (MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 1),
                                 wei_mem,
                                 bot_mem,
                                 pvt_accum);
                    } // l
                      // 4th

                    Convolve(ex_row,
                             ex_pix,
                             l,
                             m,
                             (MLO_N_FILTER_SPLITS0 - 1),
                             (MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 2),
                             wei_mem,
                             bot_mem,
                             pvt_accum);

                } // m

            } // f_s

            // last f_s
            {

                barrier(CLK_LOCAL_MEM_FENCE);

#define MLO_WEI_READ ((MLO_N_FILTER_SPLITS1 - 1) * MLO_WEI_LCL_WIDTH)
                // fetch a set of weight vertical taps
                fetchWeights(c, k_idx, f_s, lcl_id, (MLO_WEI_READ), gbl_wei_off, wei_mem, weights);

                // fetch a set of input scanlines

                uint n_reads = MLO_IN_LCL_HEIGHT - 1; // ((ob == 0 && (f_s < MLO_FILTER_PAD1)) ||
                                                      // (ob == get_local_size(0) - 1 &&
                                                      // (MLO_FILTER_STRIDE1 - f_s) <
                                                      // MLO_FILTER_PAD1)) ? MLO_IN_LCL_HEIGHT - 1 :
                                                      // MLO_IN_LCL_HEIGHT;
                uint lcl_scan = 0; // (ob == 0 && (f_s < MLO_FILTER_PAD1)) ? 1 : 0;

                fetchData(f_s, lcl_id, lcl_scan, n_reads, in_y, gbl_in_scan_off, bot_mem, bot);

                barrier(CLK_LOCAL_MEM_FENCE);

// convolution
// along vertical filter
#pragma unroll
                for(uint m = 0; m < MLO_N_FILTER_SPLITS1 - 1; ++m)
                {

                    // first 3 splits
                    uint l;
                    for(l = 0; l < MLO_FILTER_STRIDE0 - 1; ++l)
                    {
                        Convolve(ex_row,
                                 ex_pix,
                                 l,
                                 m,
                                 (MLO_N_FILTER_SPLITS0),
                                 (MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 1),
                                 wei_mem,
                                 bot_mem,
                                 pvt_accum);

                    } // l
                      // 4th

                    Convolve(ex_row,
                             ex_pix,
                             l,
                             m,
                             (MLO_N_FILTER_SPLITS0 - 1),
                             (MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 2),
                             wei_mem,
                             bot_mem,
                             pvt_accum);

                } // m

            } // f_s

        } // c

        //			for (int bb = 0; bb < MLO_N_LCL_BATCHS && ex_row < MLO_OUT_EXTENT1
        //&&
        //(out_y + ex_row) < MLO_OUT_HEIGHT; ++bb)
        {
            for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
            {
                // write out
                // inputs are outputs
                uint out_off = (ib + b) * MLO_OUT_BATCH_STRIDE +
                               (k_idx + k) * MLO_OUT_CHANNEL_STRIDE +
                               (out_y + ex_row) * MLO_OUT_STRIDE + ex_pix;
                __global _FLOAT* top_p = &top[out_off];
                for(uint i = 0; i < MLO_OUT_PIX_TILE0; ++i)
                {
                    if((k_idx + k) < MLO_N_OUTPUTS && ex_row < MLO_OUT_EXTENT1 &&
                       (out_y + ex_row) < MLO_OUT_HEIGHT && ex_pix + i < MLO_OUT_WIDTH)
                    {
                        top_p[i] = CVT_ACCUM2FLOAT(pvt_accum[k * MLO_OUT_PIX_TILE0 + i]);
                    }
                }
            }
        }

    } // b
}

/*****************************************************
        2nd pass
******************************************************/
#undef MLO_LCL_MEM_SZ
#undef MLO_TOTAL_IN_LCL_SZ
#undef MLO_IN_LCL_SZ
#undef MLO_IN_LCL_HEIGHT
#undef MLO_OUT_EXTENT1
#undef MLO_N_LCL_BATCHS

#define MLO_N_LCL_BATCHS MLO_N_LCL_BATCHS_PASS2
#define MLO_OUT_EXTENT1 (MLO_LAST_OUT_EXTENT1)
#define MLO_IN_LCL_HEIGHT (MLO_OUT_EXTENT1 + MLO_N_FILTER_SPLITS1 - 1)
#define MLO_IN_LCL_SZ (MLO_IN_LCL_WIDTH * MLO_IN_LCL_HEIGHT)
// LDS size
#define MLO_TOTAL_IN_LCL_SZ (MLO_N_LCL_BATCHS * MLO_IN_LCL_SZ * MLO_N_LCL_IN_MAPS)
#define MLO_LCL_MEM_SZ (MLO_WEI_LCL_SZ + MLO_TOTAL_IN_LCL_SZ)

void fetchData2(uint ib,
                uint f_s,
                uint lcl_id,
                uint lcl_scan,
                uint n_reads,
                int in_y,
                int gbl_in_scan_off,
                __local _FLOAT* bot_mem,
                const __global _FLOAT* bot)
{
    __private _FLOAT in_rd_data[MLO_READ_UNIT];

    for(uint p4 = lcl_id, c_scan = 0; p4 < MLO_N_IN_HORIZ_READS * n_reads * MLO_N_LCL_BATCHS;
        p4 += MLO_GRP_SZ)
    {
        uint b  = 0;
        uint t0 = p4;
#if MLO_N_LCL_BATCHS > 1
        b  = iDiv_legacy(p4, MLO_N_IN_HORIZ_READS * n_reads);
        t0 = iMod(p4, b, MLO_N_IN_HORIZ_READS * n_reads);
#endif
#if MLO_N_IN_HORIZ_READS & (MLO_N_IN_HORIZ_READS - 1)
        c_scan      = iDiv_legacy(t0, MLO_N_IN_HORIZ_READS);
        uint c_pix4 = iMod(t0, c_scan, MLO_N_IN_HORIZ_READS);
#else
        c_scan = t0 / MLO_N_IN_HORIZ_READS;
        uint c_pix4 = t0 & (MLO_N_IN_HORIZ_READS - 1);
#endif
        int in_scan = (c_scan + lcl_scan) * MLO_FILTER_STRIDE1 + f_s;

        for(uint i = 0; i < MLO_READ_UNIT; ++i)
        {
            in_rd_data[i] = 0;
        }

        if(0 <= in_y + in_scan && in_y + in_scan < MLO_IN_HEIGHT && b < MLO_N_LCL_BATCHS &&
           (ib + b) < MLO_BATCH_SZ)
        {

            int gbl_off = gbl_in_scan_off + b * MLO_IN_BATCH_STRIDE + in_scan * MLO_IN_STRIDE +
                          c_pix4 * MLO_READ_UNIT;
            const __global _FLOAT* bot_p = &bot[gbl_off];
// still problems with unaligned LDS access
#if MLO_IN_N_PIXS_OFF > 0
            if(c_pix4 == MLO_N_IN_HORIZ_READS - 1)
            {
                uint i = 0;
                for(; i < MLO_IN_N_PIXS_OFF; ++i)
                {
                    in_rd_data[i] = bot_p[i];
                }
                //								for (; i <
                // MLO_READ_UNIT;
                //++i)
                //								{
                //									in_rd_data[i]
                //=
                // 0;
                //								}
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

        if(b < MLO_N_LCL_BATCHS)
        {
            int lcl_off = b * MLO_IN_LCL_SZ + (lcl_scan + c_scan) * MLO_IN_LCL_WIDTH +
                          MLO_FILTER_PAD0 + c_pix4 * MLO_READ_UNIT;
            for(uint i = 0; i < MLO_READ_UNIT; ++i)
            {
                bot_mem[lcl_off + i] = in_rd_data[i];
            }
        }
    }
}

void Convolve2(uint b,
               uint ex_row,
               uint ex_pix,
               uint l,
               uint m,
               uint wei_h,
               uint bot_h,
               __local _FLOAT* __restrict wei_mem,
               __local _FLOAT* __restrict bot_mem,
               __private _FLOAT_ACCUM* pvt_accum)
{
    // only for 11
    __private _FLOAT wei_vals[MLO_N_LCL_OUT_MAPS * MLO_N_FILTER_SPLITS0];
    __private _FLOAT in_vals[(MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 1)];

    // read all weights
    for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
    {
        for(uint i = 0; i < wei_h; ++i)
        {
            wei_vals[k * MLO_N_FILTER_SPLITS0 + i] =
                wei_mem[k * MLO_WEI_SZ + m * MLO_WEI_LCL_WIDTH + i * MLO_FILTER_STRIDE0 + l];
        }
    }

    // convolve
    for(uint i = 0; i < bot_h; ++i)
    {
        in_vals[i] = bot_mem[b * MLO_IN_LCL_SZ + (ex_row + m) * MLO_IN_LCL_WIDTH +
                             ex_pix * MLO_FILTER_STRIDE0 + i * MLO_FILTER_STRIDE0 + l];
    }

    for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
    {
        for(uint n = 0; n < MLO_OUT_PIX_TILE0; ++n)
        {

            for(uint i = 0; i < wei_h; ++i)
            {
                _FLOAT in_val  = in_vals[n + i];
                _FLOAT wei_val = wei_vals[k * MLO_N_FILTER_SPLITS0 + i];
                pvt_accum[k * MLO_OUT_PIX_TILE0 + n] +=
                    CVT_FLOAT2ACCUM(wei_val) * CVT_FLOAT2ACCUM(in_val);
#if 0
				if (wei_val * in_val != 0 && ib + b + bb == 0 && k_idx + k == 1 && out_y + ex_row == 0 && ex_pix + n == 0)
				{
					printf("G:c: %d %d %d %d %d %d %d %d %d %d %d %d %f %f %f %f\n",
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
} // l

__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2))) __kernel void
MIOpenCvFwd11x11_2(const __global _FLOAT* __restrict bot,
                   const __global _FLOAT* __restrict weights,
#if MLO_CONV_BIAS == 1
                   const __global _FLOAT* __restrict bias,
#endif
                   __global _FLOAT* __restrict top,
                   UNUSED _FLOAT padding_val)
{

    __local _FLOAT lcl_mem[MLO_LCL_MEM_SZ];
    __local _FLOAT* bot_mem = lcl_mem;
    __local _FLOAT* wei_mem = lcl_mem + MLO_TOTAL_IN_LCL_SZ;

    uint lcl_id = get_local_id(0);

    uint k_idx = get_group_id(1) * (MLO_N_LCL_OUT_MAPS); // input map index based

    uint ib_idx = get_group_id(2) * MLO_N_LCL_BATCHS; // batch idx

    uint ib = ib_idx;

    int gbl_in_off   = /*c_idx * MLO_IN_CHANNEL_STRIDE + */ ib * MLO_IN_BATCH_STRIDE;
    uint gbl_wei_off = k_idx * MLO_WEI_BATCH_STRIDE;

    // last extent
    // the major part of the output map has been processed in the previous pass to avoid the
    // granularity loss
    int out_y = MLO_OUT_HEIGHT - MLO_LAST_OUT_EXTENT1;

    int in_y = out_y * MLO_FILTER_STRIDE1 - MLO_FILTER_PAD1;
    gbl_in_off += in_y * MLO_IN_STRIDE;

#define MLO_ACCUM_SZ \
    (MLO_OUT_PIX_TILE1 * MLO_OUT_PIX_TILE0 * MLO_N_LCL_OUT_MAPS * MLO_N_LCL_IN_MAPS)

    __private _FLOAT_ACCUM pvt_accum[MLO_ACCUM_SZ];

    // zero out LDS
    for(uint i = lcl_id; i < (MLO_LCL_MEM_SZ); i += MLO_GRP_SZ)
    {
        lcl_mem[i] = 0;
    }

// processing arrangement
// batch
#if(MLO_PROCESSING_WIDTH * MLO_LAST_OUT_EXTENT1) & (MLO_PROCESSING_WIDTH * MLO_LAST_OUT_EXTENT1 - 1)
    uint bb = iDiv_legacy(lcl_id, (MLO_PROCESSING_WIDTH * MLO_LAST_OUT_EXTENT1));
    uint t0 = iMod(lcl_id, bb, (MLO_PROCESSING_WIDTH * MLO_LAST_OUT_EXTENT1));
#elif(MLO_PROCESSING_WIDTH * MLO_LAST_OUT_EXTENT1) != 0
    uint bb = lcl_id / (MLO_PROCESSING_WIDTH * MLO_LAST_OUT_EXTENT1);
    uint t0 = lcl_id & ((MLO_PROCESSING_WIDTH * MLO_LAST_OUT_EXTENT1) - 1);
#if(MLO_PROCESSING_WIDTH * MLO_LAST_OUT_EXTENT1) >= 64
    bb = uniform(bb);
#endif
#else
    uint bb = lcl_id;
    uint t0 = 0;
#endif
#if MLO_PROCESSING_WIDTH & (MLO_PROCESSING_WIDTH - 1)
    uint ex_row = iDiv_legacy(t0, MLO_PROCESSING_WIDTH);
    uint ex_col = iMod(t0, ex_row, MLO_PROCESSING_WIDTH);
#else
    uint ex_row = t0 / MLO_PROCESSING_WIDTH;
    uint ex_col = t0 & (MLO_PROCESSING_WIDTH - 1);
#endif
    uint ex_pix = ex_col * MLO_OUT_PIX_TILE0;

    // over all batches

    for(uint b = 0; b < MLO_N_BATCH_LOOPS; ++b, gbl_in_off += MLO_IN_BATCH_STRIDE)
    {

        int gbl_in_scan_off0 = gbl_in_off;

        // generate pixels from all MLO_N_LCL_OUT_MAPS output maps

        for(uint i = 0; i < MLO_ACCUM_SZ; ++i)
        {
            pvt_accum[i] = CVT_FLOAT2ACCUM(0);
        }

// all input maps
#ifdef __AMDGCN__
#pragma unroll 4
#endif
        for(uint c = 0, gbl_in_scan_off = gbl_in_scan_off0; c < MLO_N_INPUTS;
            ++c, gbl_in_scan_off += MLO_IN_CHANNEL_STRIDE)
        {
            uint f_s = 0;
            for(; f_s < MLO_FILTER_STRIDE1 - 1; ++f_s)
            {

                barrier(CLK_LOCAL_MEM_FENCE);

                // get a set of horizaontal taps
                fetchWeights(c, k_idx, f_s, lcl_id, MLO_WEI_SZ, gbl_wei_off, wei_mem, weights);

                // fetch a set of input scanlines

                uint n_reads = MLO_IN_LCL_HEIGHT; // ((ob == 0 && (f_s < MLO_FILTER_PAD1)) || (ob ==
                                                  // get_local_size(0) - 1 && (MLO_FILTER_STRIDE1 -
                                                  // f_s) < MLO_FILTER_PAD1)) ? MLO_IN_LCL_HEIGHT -
                                                  // 1 : MLO_IN_LCL_HEIGHT;
                uint lcl_scan = 0;                // (ob == 0 && (f_s < MLO_FILTER_PAD1)) ? 1 : 0;

                fetchData2(
                    (ib + b), f_s, lcl_id, lcl_scan, n_reads, in_y, gbl_in_scan_off, bot_mem, bot);

                barrier(CLK_LOCAL_MEM_FENCE);

// convolution
// along vertical filter
#pragma unroll
                for(uint m = 0; m < MLO_N_FILTER_SPLITS1; ++m)
                {

                    // first 3 splits
                    uint l;
                    for(l = 0; l < MLO_FILTER_STRIDE0 - 1; ++l)
                    {

                        Convolve2(bb,
                                  ex_row,
                                  ex_pix,
                                  l,
                                  m,
                                  (MLO_N_FILTER_SPLITS0),
                                  (MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 1),
                                  wei_mem,
                                  bot_mem,
                                  pvt_accum);
                    } // l
                      // 4th

                    Convolve2(bb,
                              ex_row,
                              ex_pix,
                              l,
                              m,
                              (MLO_N_FILTER_SPLITS0 - 1),
                              (MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 2),
                              wei_mem,
                              bot_mem,
                              pvt_accum);

                } // m

            } // f_s

            // last f_s
            {

                barrier(CLK_LOCAL_MEM_FENCE);

#define MLO_WEI_READ ((MLO_N_FILTER_SPLITS1 - 1) * MLO_WEI_LCL_WIDTH)
                // fetch a set of weight vertical taps

                fetchWeights(c, k_idx, f_s, lcl_id, (MLO_WEI_READ), gbl_wei_off, wei_mem, weights);

                // fetch a set of input scanlines

                uint n_reads = MLO_IN_LCL_HEIGHT - 1; // ((ob == 0 && (f_s < MLO_FILTER_PAD1)) ||
                                                      // (ob == get_local_size(0) - 1 &&
                                                      // (MLO_FILTER_STRIDE1 - f_s) <
                                                      // MLO_FILTER_PAD1)) ? MLO_IN_LCL_HEIGHT - 1 :
                                                      // MLO_IN_LCL_HEIGHT;
                uint lcl_scan = 0; // (ob == 0 && (f_s < MLO_FILTER_PAD1)) ? 1 : 0;

                fetchData2(
                    (ib + b), f_s, lcl_id, lcl_scan, n_reads, in_y, gbl_in_scan_off, bot_mem, bot);

                barrier(CLK_LOCAL_MEM_FENCE);

// convolution
// along vertical filter
#pragma unroll
                for(uint m = 0; m < MLO_N_FILTER_SPLITS1 - 1; ++m)
                {

                    // first 3 splits
                    uint l;
                    for(l = 0; l < MLO_FILTER_STRIDE0 - 1; ++l)
                    {
                        Convolve2(bb,
                                  ex_row,
                                  ex_pix,
                                  l,
                                  m,
                                  (MLO_N_FILTER_SPLITS0),
                                  (MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 1),
                                  wei_mem,
                                  bot_mem,
                                  pvt_accum);

                    } // l
                      // 4th

                    Convolve2(bb,
                              ex_row,
                              ex_pix,
                              l,
                              m,
                              (MLO_N_FILTER_SPLITS0 - 1),
                              (MLO_OUT_PIX_TILE0 + MLO_N_FILTER_SPLITS0 - 2),
                              wei_mem,
                              bot_mem,
                              pvt_accum);

                } // m

            } // f_s

        } // c

        for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
        {
            // write out
            // inputs are outputs
            uint out_off = (ib + bb + b) * MLO_OUT_BATCH_STRIDE +
                           (k_idx + k) * MLO_OUT_CHANNEL_STRIDE +
                           (out_y + ex_row) * MLO_OUT_STRIDE + ex_pix;
            __global _FLOAT* top_p = &top[out_off];
            for(uint i = 0; i < MLO_OUT_PIX_TILE0; ++i)
            {
                if((ib + bb + b) < MLO_BATCH_SZ && bb < MLO_N_LCL_BATCHS &&
                   (k_idx + k) < MLO_N_OUTPUTS && ex_row < MLO_LAST_OUT_EXTENT1 &&
                   (out_y + ex_row) < MLO_OUT_HEIGHT && ex_pix + i < MLO_OUT_WIDTH)
                {
                    top_p[i] = CVT_ACCUM2FLOAT(pvt_accum[k * MLO_OUT_PIX_TILE0 + i]);
                }

#if 0
				if (out_off + i == 0)
				{
					printf("G:p2:o: %d %d %d %d\n",
						lcl_id,
						out_y,
						ex_row,
						ex_pix
					);
				}
#endif
            }
        }

    } // b
}

#else

#define MLO_N_TILES1 \
    ((MLO_OUT_HEIGHT + MLO_OUT_PIX_TILE1 - 1 + 2 * MLO_FILTER_PAD1) / MLO_OUT_PIX_TILE1)
#define MLO_N_TILES0 \
    ((MLO_OUT_WIDTH + MLO_OUT_PIX_TILE0 - 1 + 2 * MLO_FILTER_PAD0) / MLO_OUT_PIX_TILE0)

void MoveWeightsIn(__local _FLOAT* lcl_mem,
                   uint lcl_wei_write_off,
                   const __global _FLOAT* weights,
                   uint gbl_wei_off,
                   uint lcl_id)
{
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint i = lcl_id; i < MLO_N_LCL_OUT_MAPS * MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0;
        i += MLO_GRP_SZ)
    {
        lcl_mem[lcl_wei_write_off + i] = weights[gbl_wei_off + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

#if defined(__AMDGCN__)

void MoveDataIn(_FLOAT proc_dat[MLO_IN_PIX_TILE1][MLO_IN_PIX_TILE0],
                __local _FLOAT* lcl_mem,
                const __global _FLOAT* bot,
                uint gbl_in_off,
                int grp_in_y,
                int grp_in_x,
                uint lcl_in_y,
                uint lcl_in_x)
{
    uint lcl_id = get_local_id(0);

    barrier(CLK_LOCAL_MEM_FENCE);

    // TODO: 	MLO_N_LCL_IN_MAPS

    for(uint i = lcl_id; i < MLO_N_IN_BWD_VERT_READS * MLO_N_IN_BWD_HORIZ_READS; i += MLO_GRP_SZ)
    {
        uint in_y = i / MLO_N_IN_BWD_HORIZ_READS;
        uint in_x = i % MLO_N_IN_BWD_HORIZ_READS;

        int in_run_y = grp_in_y + (int)in_y;
        int in_run_x = grp_in_x + (int)in_x;

        bool out_of_range =
            (in_run_y < 0 || in_run_y >= MLO_IN_HEIGHT || in_run_x < 0 || in_run_x >= MLO_IN_WIDTH);
        uint bot_off = (out_of_range) ? 0 : in_run_y * MLO_IN_STRIDE + in_run_x;

        _FLOAT dat = bot[gbl_in_off + bot_off];
        lcl_mem[i] = (out_of_range) ? 0 : dat;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint j = 0; j < MLO_IN_PIX_TILE1; ++j)
    {
        for(int i = 0; i < MLO_IN_PIX_TILE0; ++i)
        {
            uint lcl_off = (lcl_in_y + j) * MLO_N_IN_BWD_HORIZ_READS + lcl_in_x + i;
            proc_dat[j][i] = lcl_mem[lcl_off];
        }
    }
}

#else

void MoveDataIn(_FLOAT proc_dat[MLO_IN_PIX_TILE1][MLO_IN_PIX_TILE0],
                const __global _FLOAT* bot,
                uint gbl_in_off,
                const uint gbl_in_offs[MLO_IN_PIX_TILE1][MLO_IN_PIX_TILE0],
                const _FLOAT* mask_out_of_range)
{
    for(int j = MLO_IN_PIX_TILE1 - 1; j >= 0; --j)
    {
        for(int i = MLO_IN_PIX_TILE0 - 1; i >= 0; --i)
        {
            _FLOAT dat =
                bot[gbl_in_off + gbl_in_offs[j][i]] * mask_out_of_range[j * MLO_IN_PIX_TILE0 + i];
            proc_dat[j][i] = dat;
        }
    }
}
#endif

void Convolve(_FLOAT_ACCUM* pvt_accum,
              const _FLOAT proc_dat[MLO_IN_PIX_TILE1][MLO_IN_PIX_TILE0],
              const __local _FLOAT* lcl_mem,
              uint lcl_wei_read_off
#if DBG_PRINTF == 1
              ,
              int map_out_y,
              int map_out_x
#endif
)
{
    // convolve
    for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
    {
        int jj = MLO_N_FILTER_SPLITS1 - 1;
        for(; jj > 0; --jj)
        {
            for(uint j = 0; j < MLO_OUT_PIX_TILE1; ++j)
            {

                int ii = MLO_N_FILTER_SPLITS0 - 1;
                for(; ii > 0; --ii)
                {
                    for(uint i = 0; i < MLO_OUT_PIX_TILE0; ++i)
                    {
                        uint pvt_off = (k * MLO_OUT_PIX_TILE1 + j) * MLO_OUT_PIX_TILE0 + i;
                        uint y = ((MLO_N_FILTER_SPLITS1 - 1 - jj) * MLO_FILTER_STRIDE1 +
                                  j % MLO_FILTER_STRIDE1);
                        uint x = (MLO_N_FILTER_SPLITS0 - 1 - ii) * MLO_FILTER_STRIDE0 +
                                 i % MLO_FILTER_STRIDE0;
                        uint lcl_off = lcl_wei_read_off + k * MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0 +
                                       y * MLO_FILTER_SIZE0 + x;
                        pvt_accum[pvt_off] +=
                            CVT_FLOAT2ACCUM(proc_dat[j / MLO_FILTER_STRIDE1 + jj]
                                                    [i / MLO_FILTER_STRIDE0 + ii]) *
                            CVT_FLOAT2ACCUM(lcl_mem[lcl_off]);
#if 0

						if (k == 0 && map_out_x + (int)i == 0 && map_out_y + (int)j == 66)
						{
							printf("K:c0:%d %d %d %d %d %d %d %d    %9.7f %9.7f %9.7f %9.7f\n",
								jj,
								ii,
								y,
								x,
								pvt_off,
								j / MLO_FILTER_STRIDE1 + jj,
								i / MLO_FILTER_STRIDE0 + ii,
								lcl_off,
								pvt_accum[pvt_off],
								proc_dat[j / MLO_FILTER_STRIDE1 + jj][i / MLO_FILTER_STRIDE0 + ii] * lcl_mem[lcl_off],
								proc_dat[j / MLO_FILTER_STRIDE1 + jj][i / MLO_FILTER_STRIDE0 + ii],
								lcl_mem[lcl_off]
							);
						}
#endif
                    }
                }
                //					for (; ii >= 0; --ii)

                {
                    for(uint im = 0; im < MLO_TILE_REPLICATE0; ++im)
                    {
                        for(uint i = im * MLO_FILTER_STRIDE0; i < (im + 1) * MLO_FILTER_STRIDE0 - 1;
                            ++i)
                        {
                            uint pvt_off = (k * MLO_OUT_PIX_TILE1 + j) * MLO_OUT_PIX_TILE0 + i;
                            uint y = ((MLO_N_FILTER_SPLITS1 - 1 - jj) * MLO_FILTER_STRIDE1 +
                                      j % MLO_FILTER_STRIDE1);
                            uint x = (MLO_N_FILTER_SPLITS0 - 1 - ii) * MLO_FILTER_STRIDE0 +
                                     i % MLO_FILTER_STRIDE0;
                            uint lcl_off = lcl_wei_read_off +
                                           k * MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0 +
                                           y * MLO_FILTER_SIZE0 + x;
                            pvt_accum[pvt_off] +=
                                CVT_FLOAT2ACCUM(proc_dat[j / MLO_FILTER_STRIDE1 + jj]
                                                        [i / MLO_FILTER_STRIDE0 + ii]) *
                                CVT_FLOAT2ACCUM(lcl_mem[lcl_off]);
#if 0

							if (k == 0 && map_out_x + (int)i == 0 && map_out_y + (int)j == 66)
							{
								printf("K:c1:%d %d %d %d %d %d %d %d    %9.7f %9.7f %9.7f %9.7f\n",
									jj,
									ii,
									y,
									x,
									pvt_off,
									j / MLO_FILTER_STRIDE1 + jj,
									i / MLO_FILTER_STRIDE0 + ii,
									lcl_off,
									pvt_accum[pvt_off],
									proc_dat[j / MLO_FILTER_STRIDE1 + jj][i / MLO_FILTER_STRIDE0 + ii] * lcl_mem[lcl_off],
									proc_dat[j / MLO_FILTER_STRIDE1 + jj][i / MLO_FILTER_STRIDE0 + ii],
									lcl_mem[lcl_off]
								);
							}
#endif
                        } //  for (uint i = im*MLO_FILTER_STRIDE0; i < (im+1)*MLO_FILTER_STRIDE0 -
                          //  1; ++i)
                    }     // for (uint im = 0; im < MLO_TILE_REPLICATE0; ++im)
                }
            }
        }
        //			for (; jj > 0; --jj)
        {
            for(uint jm = 0; jm < MLO_TILE_REPLICATE1; ++jm)
            {
                for(uint j = jm * MLO_FILTER_STRIDE1; j < (jm + 1) * MLO_FILTER_STRIDE1 - 1; ++j)
                {

                    int ii = MLO_N_FILTER_SPLITS0 - 1;
                    for(; ii > 0; --ii)
                    {
                        for(uint i = 0; i < MLO_OUT_PIX_TILE0; ++i)
                        {
                            uint pvt_off = (k * MLO_OUT_PIX_TILE1 + j) * MLO_OUT_PIX_TILE0 + i;
                            uint y = ((MLO_N_FILTER_SPLITS1 - 1 - jj) * MLO_FILTER_STRIDE1 +
                                      j % MLO_FILTER_STRIDE1);
                            uint x = (MLO_N_FILTER_SPLITS0 - 1 - ii) * MLO_FILTER_STRIDE0 +
                                     i % MLO_FILTER_STRIDE0;
                            uint lcl_off = lcl_wei_read_off +
                                           k * MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0 +
                                           y * MLO_FILTER_SIZE0 + x;
                            pvt_accum[pvt_off] +=
                                CVT_FLOAT2ACCUM(proc_dat[j / MLO_FILTER_STRIDE1 + jj]
                                                        [i / MLO_FILTER_STRIDE0 + ii]) *
                                CVT_FLOAT2ACCUM(lcl_mem[lcl_off]);
#if 0

							if (k == 0 && map_out_x + (int)i == 0 && map_out_y + (int)j == 66)
							{
								printf("K:c0:%d %d %d %d %d %d    %f %f %f %f\n",
									jj,
									ii,
									y,
									x,
									pvt_off,
									lcl_off,
									pvt_accum[pvt_off],
									proc_dat[j / MLO_FILTER_STRIDE1 + jj][i / MLO_FILTER_STRIDE0 + ii] * lcl_mem[lcl_off],
									proc_dat[j / MLO_FILTER_STRIDE1 + jj][i / MLO_FILTER_STRIDE0 + ii],
									lcl_mem[lcl_off]
								);
							}
#endif

                        } // for (uint i = 0; i < MLO_OUT_PIX_TILE0; ++i)
                    }     // for (; ii > 0; --ii)
                          //					for (; ii >= 0; --ii)
                    {
                        for(uint im = 0; im < MLO_TILE_REPLICATE0; ++im)
                        {
                            for(uint i = im * MLO_FILTER_STRIDE0;
                                i < (im + 1) * MLO_FILTER_STRIDE0 - 1;
                                ++i)
                            {
                                uint pvt_off = (k * MLO_OUT_PIX_TILE1 + j) * MLO_OUT_PIX_TILE0 + i;
                                uint y = ((MLO_N_FILTER_SPLITS1 - 1 - jj) * MLO_FILTER_STRIDE1 +
                                          j % MLO_FILTER_STRIDE1);
                                uint x = (MLO_N_FILTER_SPLITS0 - 1 - ii) * MLO_FILTER_STRIDE0 +
                                         i % MLO_FILTER_STRIDE0;
                                uint lcl_off = lcl_wei_read_off +
                                               k * MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0 +
                                               y * MLO_FILTER_SIZE0 + x;
                                pvt_accum[pvt_off] +=
                                    CVT_FLOAT2ACCUM(proc_dat[j / MLO_FILTER_STRIDE1 + jj]
                                                            [i / MLO_FILTER_STRIDE0 + ii]) *
                                    CVT_FLOAT2ACCUM(lcl_mem[lcl_off]);
#if 0

								if (k == 0 && map_out_x + (int)i == 0 && map_out_y + (int)j == 66)
								{
									printf("K:c0:%d %d %d %d %d %d    %f %f %f %f\n",
										jj,
										ii,
										y,
										x,
										pvt_off,
										lcl_off,
										pvt_accum[pvt_off],
										proc_dat[j / MLO_FILTER_STRIDE1 + jj][i / MLO_FILTER_STRIDE0 + ii] * lcl_mem[lcl_off],
										proc_dat[j / MLO_FILTER_STRIDE1 + jj][i / MLO_FILTER_STRIDE0 + ii],
										lcl_mem[lcl_off]
									);
								}
#endif
                            } // for (uint i = im*MLO_FILTER_STRIDE0; i < (im +
                              // 1)*MLO_FILTER_STRIDE0 - 1; ++i)
                        }     // for (uint im = 0; im < MLO_TILE_REPLICATE0; ++im)
                    }
                } // for (uint j = 0; j < MLO_OUT_PIX_TILE1 - 1; ++j)
            }     // for (uint jm = 0; jm < MLO_TILE_REPLICATE1; ++jm)
        }         // for (; jj > 0; --jj)

    } // for (uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
}

/*********************************************************************************************************
// brw algorithm with stride 4
// idea:
// process 4x4 micro-tile
// loop with duble buffering
// read 3x3 input micro-tile into registers
// 11x11 filter into LDS


**********************************************************************************************************/

__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2))) __kernel void
MIOpenCvBwd11x11(const __global _FLOAT* __restrict bot,
                 const __global _FLOAT* __restrict weights,
#if MLO_CONV_BIAS == 1
                 const __global _FLOAT* __restrict bias,
#endif
                 __global _FLOAT* __restrict top,
                 UNUSED _FLOAT padding_val)
{
    // double buffering
    __local _FLOAT lcl_mem[MLO_LCL_BWD_MEM_SZ];

    uint lcl_wei_write_off = 0;
    uint lcl_wei_read_off = 0;

#undef MLO_ACCUM_SZ
#define MLO_ACCUM_SZ (MLO_OUT_PIX_TILE1 * MLO_OUT_PIX_TILE0 * MLO_N_LCL_OUT_MAPS)
    _FLOAT_ACCUM pvt_accum[MLO_ACCUM_SZ];

    for(uint i = 0; i < MLO_ACCUM_SZ; ++i)
    {
        pvt_accum[i] = CVT_FLOAT2ACCUM(0);
    }

    uint lcl_id = get_local_id(0);

    uint gbl_id = get_global_id(0); // id of the 4x4 micro-tile inside the output buffer

    uint tile_y = (gbl_id / MLO_N_TILES0);
    uint tile_x = (gbl_id % MLO_N_TILES0);

    uint k_idx = get_global_id(1) * MLO_N_LCL_OUT_MAPS; // output idx

    uint b_idx = get_global_id(2); // batch

    // prefetch input map
    _FLOAT proc_dat[MLO_IN_PIX_TILE1][MLO_IN_PIX_TILE0];

    uint gbl_in_off = b_idx * MLO_IN_BATCH_STRIDE;

    int map_out_y = (int)tile_y * MLO_OUT_PIX_TILE1 - MLO_FILTER_PAD1;

    int map_in_y = (map_out_y + MLO_OUT_PIX_TILE1 - 1) / MLO_FILTER_STRIDE1;

    int map_out_x = (int)tile_x * MLO_OUT_PIX_TILE0 - MLO_FILTER_PAD0;

    int map_in_x = (map_out_x + MLO_OUT_PIX_TILE0 - 1) / MLO_FILTER_STRIDE0;

    // stupid compiler
    uint out_of_range_y[MLO_IN_PIX_TILE1];
    uint out_of_range_x[MLO_IN_PIX_TILE0];
    _FLOAT mask_out_of_range[MLO_IN_PIX_TILE1 * MLO_IN_PIX_TILE0];

    uint gbl_in_offs[MLO_IN_PIX_TILE1][MLO_IN_PIX_TILE0];

    for(int j = MLO_IN_PIX_TILE1 - 1; j >= 0; --j)
    {
        out_of_range_y[j] = ((map_in_y + (j - MLO_IN_PIX_TILE1 + 1)) >= MLO_IN_HEIGHT ||
                             (map_in_y + (j - MLO_IN_PIX_TILE1 + 1)) < 0)
                                ? 0
                                : (uint)(-1);

        for(int i = MLO_IN_PIX_TILE0 - 1; i >= 0; --i)
        {
            out_of_range_x[i] = ((map_in_x + (i - MLO_IN_PIX_TILE0 + 1)) >= MLO_IN_WIDTH ||
                                 (map_in_x + (i - MLO_IN_PIX_TILE0 + 1)) < 0)
                                    ? 0
                                    : (uint)(-1);
            gbl_in_offs[j][i] = (((map_in_y + (j - MLO_IN_PIX_TILE1 + 1)) * MLO_IN_STRIDE +
                                  (map_in_x + (i - MLO_IN_PIX_TILE0 + 1))) &
                                 (out_of_range_x[i] & out_of_range_y[j]));

            mask_out_of_range[j * MLO_IN_PIX_TILE0 + i] =
                (out_of_range_x[i] & out_of_range_y[j]) ? 1 : 0;
        }
    }

#if defined(__AMDGCN__)
    uint grp_gbl_id = get_group_id(0) * MLO_GRP_SZ;
    uint grp_tile_y = (grp_gbl_id / MLO_N_TILES0);
    // adjust col pos to 0 to read from the beginning of the scan for easyie mapping
    uint grp_tile_x = (MLO_N_TILES0 < MLO_GRP_SZ) ? 0 : (grp_gbl_id % MLO_N_TILES0);

    int grp_out_y = (int)grp_tile_y * MLO_OUT_PIX_TILE1 - MLO_FILTER_PAD1;
    int grp_out_x = (int)grp_tile_x * MLO_OUT_PIX_TILE0 - MLO_FILTER_PAD0;

    int grp_in_y = (grp_out_y + MLO_OUT_PIX_TILE1 - 1) / MLO_FILTER_STRIDE1 - MLO_IN_PIX_TILE1 + 1;
    int grp_in_x = (grp_out_x + MLO_OUT_PIX_TILE0 - 1) / MLO_FILTER_STRIDE0 - MLO_IN_PIX_TILE0 + 1;

    uint lcl_in_y = (uint)((tile_y - grp_tile_y) * MLO_OUT_PIX_TILE1) / MLO_FILTER_STRIDE1;

    uint lcl_in_x = (uint)((tile_x - grp_tile_x) * MLO_OUT_PIX_TILE0) / MLO_FILTER_STRIDE0;
#endif

    uint gbl_wei_off = k_idx * MLO_WEI_CHANNEL_STRIDE;

    //#pragma unroll 2
    for(int c = 0; c < MLO_N_INPUTS;
        ++c, gbl_in_off += MLO_IN_CHANNEL_STRIDE, gbl_wei_off += MLO_WEI_BATCH_STRIDE)
    {

// move data in
#if defined(__AMDGCN__)
        MoveDataIn(proc_dat, lcl_mem, bot, gbl_in_off, grp_in_y, grp_in_x, lcl_in_y, lcl_in_x);
#endif
        // move next weights in

        MoveWeightsIn(lcl_mem, lcl_wei_write_off, weights, gbl_wei_off, lcl_id);

// move data in
#if !defined(__AMDGCN__)
        MoveDataIn(proc_dat, bot, gbl_in_off, gbl_in_offs, mask_out_of_range);
#endif
        // convolve

        Convolve(pvt_accum,
                 proc_dat,
                 lcl_mem,
                 lcl_wei_read_off
#if DBG_PRINTF == 1
                 ,
                 map_out_y,
                 map_out_x
#endif
        );

    } // for (int c = 0; c < MLO_N_INPUTS; ++c, gbl_in_off += MLO_IN_CHANNEL_STRIDE, gbl_wei_off +=
      // MLO_WEI_BATCH_STRIDE)

    // write out
    for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
    {
        // write out
        // inputs are outputs
        int out_off = b_idx * MLO_OUT_BATCH_STRIDE + (k_idx + k) * MLO_OUT_CHANNEL_STRIDE +
                      map_out_y * MLO_OUT_STRIDE + map_out_x;
        __global _FLOAT* top_p = &top[out_off];
        for(int j = 0; j < MLO_OUT_PIX_TILE1; ++j)
        {
            for(int i = 0; i < MLO_OUT_PIX_TILE0; ++i)
            {
                if((k_idx + k) < MLO_N_OUTPUTS && map_out_y + j < MLO_OUT_HEIGHT &&
                   map_out_y + j >= 0 && map_out_x + i < MLO_OUT_WIDTH && map_out_x + i >= 0)
                {
                    top_p[j * MLO_OUT_STRIDE + i] = CVT_ACCUM2FLOAT(
                        pvt_accum[(k * MLO_OUT_PIX_TILE1 + j) * MLO_OUT_PIX_TILE0 + i]);
                }
            }
        }
    }
}

#endif // MLO_DIR_FORWARD == 1
