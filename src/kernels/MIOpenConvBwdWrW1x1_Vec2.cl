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
#define INLINE __attribute__((always_inline))
#define IDIV(A, B) (iDiv(A, B))
#define IMOD(A, B, C) (iMod(A, B, C))
//#define IDIV(A,B) ((uint)((float)A * (1.0f / (float) B) + 0.00001f))
//#define IMOD(A,B,C) (A - mul24(B, (uint)C))

#define DBG_OUT_OF_RNGE 0

INLINE
uint iDiv(uint v, uint d)
{
    uint r = (uint)((float)v * (1.0f / (float)d) + 0.00001f);
    return (r);
}

INLINE
uint iMod(uint v, uint u, uint d)
{
    uint r = v - mul24(u, d);
    return (r);
}

INLINE
void ReduceKernel(__local _FLOAT* lcl_blob,
                  __private _FLOAT* weights_accum,
                  uint lcl_id,
                  uint scan_lcl,
                  uint sum_stride,
                  uint unit_len)
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

/*********************************************************************************************************
// wrw algorithm for large filters
// idea:
// read MLO_OUT_STACKS output maps into LDS
// read MLO_N_LCL_IN_MAPS per wk_item input maps

// alg


// convolve
// reduce with transform

// write out


**********************************************************************************************************/

#define MLO_TOP_DAT_SZ (MLO_N_LCL_OUT_MAPS * MLO_READ_UNIT)
#define MLO_BOT_DAT_SZ (MLO_N_LCL_IN_MAPS * MLO_READ_UNIT)
#define MLO_ACCUM_SZ (MLO_N_LCL_OUT_MAPS * MLO_N_LCL_IN_MAPS)

/*

        Small  maps

*/

__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2))) __kernel void
MIOpenCvBwdWrWSmap(const __global _FLOAT* __restrict top_df,
                   const __global _FLOAT* __restrict bot,
                   __global _FLOAT* __restrict weights_df,
                   UNUSED _FLOAT padding_val)
{
    // reduction memory.

    __local _FLOAT2 lcl_mem[MLO_LCL_MEM_SZ];

    uint lcl_id = get_local_id(0);

    uint k_idx = get_group_id(0) * (MLO_N_LCL_OUT_MAPS); // output map index base

    uint c_idx =
        get_group_id(1) * (MLO_N_LCL_IN_MAPS * MLO_N_MAPS_PER_GROUP); // input map index based

    uint ib = get_group_id(2); // batch id

    uint gbl_in_off  = c_idx * MLO_IN_CHANNEL_STRIDE + ib * MLO_IN_BATCH_STRIDE;
    uint gbl_out_off = k_idx * MLO_OUT_CHANNEL_STRIDE + ib * MLO_OUT_BATCH_STRIDE;

    __private _FLOAT2 bot_dat[MLO_BOT_DAT_SZ];

    __private _FLOAT2 pvt_accum[MLO_ACCUM_SZ] = {(MLO_ACCUM_SZ) * (_FLOAT2)(0)};

    //	for (uint i = 0; i < MLO_ACCUM_SZ; ++i)
    //	{
    //		pvt_accum[i] = (_FLOAT2) (0);
    //	}

    for(uint i = lcl_id; i < MLO_LCL_MEM_SZ; i += MLO_GRP_SZ)
    {
        lcl_mem[i] = (_FLOAT2)(0);
    }

// map id inside the group, super-pixel inside the map
#if(MLO_MAP_WK_SZ & (MLO_MAP_WK_SZ - 1))

    uint m_id = IDIV(lcl_id, MLO_MAP_WK_SZ);       // map
    uint p4   = IMOD(lcl_id, m_id, MLO_MAP_WK_SZ); // pixel
#else
    uint m_id                                          = ((uint)lcl_id / MLO_MAP_WK_SZ); // map
    uint p4                                            = ((uint)lcl_id & (MLO_MAP_WK_SZ - 1)); // pixel

#endif

    gbl_in_off += p4 * MLO_READ_UNIT;

    // input is kept in registers at the start
    gbl_in_off += m_id * MLO_IN_CHANNEL_STRIDE;
    uint2 gbl_in_offv2  = (uint2)(gbl_in_off, gbl_in_off + MLO_IN_BATCH_STRIDE);
    uint2 gbl_out_offv2 = (uint2)(gbl_out_off, gbl_out_off + MLO_OUT_BATCH_STRIDE);

// inside input range

#if MLO_N_IN_MAPS_ALIGNED == 0
    bool inside_map_range   = (p4 < MLO_MAP_WK_SZ);
    bool inside_range_input = inside_map_range & ((c_idx + m_id) < MLO_N_INPUTS);
#endif

#ifdef __AMDGCN__
#pragma unroll 2
#endif

    for(uint b = 0; b < MLO_BATCH_SZ; b += 2,
             gbl_in_offv2 += (uint2)(2 * MLO_IN_BATCH_STRIDE),
             gbl_out_offv2 += (uint2)(2 * MLO_OUT_BATCH_STRIDE))
    {
#if MLO_BATCH_SZ % 2 > 0
        bool IsLast = ((b + 1) >= MLO_BATCH_SZ);
#endif
        if(m_id < MLO_N_MAPS_PER_GROUP)
        {

            // read all inputs into registers
            uint2 bot_off = gbl_in_offv2;

#if MLO_N_PIXS_OFF > 0
            bool last_pixel = (p4 == MLO_MAP_WK_SZ - 1);

            if(last_pixel)
            {
                for(uint c = 0; c < MLO_N_LCL_IN_MAPS;
                    ++c, bot_off += (uint2)(MLO_N_MAPS_PER_GROUP * MLO_IN_CHANNEL_STRIDE))
                {

// reading in order per group and jump over maps been read
// read arbitrary data but inside the range
#if MLO_N_IN_MAPS_ALIGNED == 0
                    bot_off = (inside_range_input &&
                               ((c_idx + m_id + c * MLO_N_MAPS_PER_GROUP) < MLO_N_INPUTS))
                                  ? bot_off
                                  : (uint2)(0);
#endif

                    const __global _FLOAT* bot1X = &bot[bot_off.x];
                    const __global _FLOAT* bot1Y = &bot[bot_off.y];

                    for(uint i = 0; i < MLO_N_PIXS_OFF; ++i)
                    {
#if MLO_BATCH_SZ % 2 > 0
                        bot_dat[c * MLO_READ_UNIT + i] =
                            (_FLOAT2)(bot1X[i], (IsLast) ? (_FLOAT)0 : bot1Y[i]);
#else
                        bot_dat[c * MLO_READ_UNIT + i] = (_FLOAT2)(bot1X[i], bot1Y[i]);
#endif
#if DBG_OUT_OF_RNGE
                        if(bot_off + i >= MLO_IN_BATCH_STRIDE * MLO_BATCH_SZ)
                        {
                            printf("k:err:in-off-range\n");
                        }
#endif
                    }
                    for(uint i = MLO_N_PIXS_OFF; i < MLO_READ_UNIT; ++i)
                    {
                        bot_dat[c * MLO_READ_UNIT + i] = (_FLOAT2)(0);
                    }
                }
            }
            else
#endif
            {
                // check
                for(uint c = 0; c < MLO_N_LCL_IN_MAPS;
                    ++c, bot_off += (uint2)(MLO_N_MAPS_PER_GROUP * MLO_IN_CHANNEL_STRIDE))
                {
// reading in order per group and jump over maps been read
// read arbitrary data but inside the range

#if MLO_N_IN_MAPS_ALIGNED == 0
                    bot_off = (inside_range_input &&
                               ((c_idx + m_id + c * MLO_N_MAPS_PER_GROUP) < MLO_N_INPUTS))
                                  ? bot_off
                                  : (uint2)(0);
#endif
                    const __global _FLOAT* bot1X = &bot[bot_off.x];
                    const __global _FLOAT* bot1Y = &bot[bot_off.y];

                    for(uint i = 0; i < MLO_READ_UNIT; ++i)
                    {
#if MLO_BATCH_SZ % 2 > 0
                        bot_dat[c * MLO_READ_UNIT + i] =
                            (_FLOAT2)(bot1X[i], (IsLast) ? (_FLOAT)0 : bot1Y[i]);
#else
                        bot_dat[c * MLO_READ_UNIT + i] = (_FLOAT2)(bot1X[i], bot1Y[i]);
#endif
#if DBG_OUT_OF_RNGE
                        if(bot_off + i >= MLO_IN_BATCH_STRIDE * MLO_BATCH_SZ)
                        {
                            printf("k:err:in-off-range\n");
                        }
#endif
                    }
                }

            } // if (last_pixel)
        }

        uint2 top_off = gbl_out_offv2;

        // read all outputs
        // assum division by MLO_N_LCL_OUT
        for(uint kb = 0; kb < MLO_N_LCL_OUT;
            kb++, top_off += (uint2)(MLO_OUT_LCL_BLK * MLO_OUT_CHANNEL_STRIDE))
        {

            barrier(CLK_LOCAL_MEM_FENCE);

            for(uint p = lcl_id; p < MLO_OUT_LCL_BLK * MLO_MAP_WK_SZ; p += MLO_GRP_SZ)
            {
#if(MLO_MAP_WK_SZ & (MLO_MAP_WK_SZ - 1))
                uint m  = IDIV(p, MLO_MAP_WK_SZ);
                uint pm = IMOD(p, m, MLO_MAP_WK_SZ);
#else
                uint m                                 = ((uint)p / MLO_MAP_WK_SZ);
                uint pm                                = ((uint)p & (MLO_MAP_WK_SZ - 1));
#endif

                uint2 top_off1 = top_off + (uint2)(m * MLO_OUT_CHANNEL_STRIDE + pm * MLO_READ_UNIT);
#if MLO_N_OUT_MAPS_ALIGNED == 0
                top_off1 = (k_idx + m < MLO_N_OUTPUTS) ? top_off1 : (uint2)(0);
#endif

                const __global _FLOAT* top_df1X = &top_df[top_off1.x];
                const __global _FLOAT* top_df1Y = &top_df[top_off1.y];

#if MLO_N_PIXS_OFF > 0
                if(pm == MLO_MAP_WK_SZ - 1)
                {

                    for(uint i = 0; i < MLO_N_PIXS_OFF; ++i)
                    {
#if MLO_BATCH_SZ % 2 > 0
                        lcl_mem[p * MLO_READ_UNIT + i] =
                            (_FLOAT2)(top_df1X[i], (IsLast) ? (_FLOAT)0 : top_df1Y[i]);
#else
                        lcl_mem[p * MLO_READ_UNIT + i] = (_FLOAT2)(top_df1X[i], top_df1Y[i]);
#endif
#if DBG_OUT_OF_RNGE
                        if(top_off1 + i >= MLO_OUT_BATCH_STRIDE * MLO_BATCH_SZ)
                        {
                            printf("k:err:out-off-range\n");
                        }
#endif
                    }
                    for(uint i = MLO_N_PIXS_OFF; i < MLO_READ_UNIT; ++i)
                    {
                        lcl_mem[p * MLO_READ_UNIT + i] = (_FLOAT2)(0);
                    }
                }
                else

#endif
                {
                    for(uint i = 0; i < MLO_READ_UNIT; ++i)
                    {
#if MLO_BATCH_SZ % 2 > 0
                        lcl_mem[p * MLO_READ_UNIT + i] =
                            (_FLOAT2)(top_df1X[i], (IsLast) ? (_FLOAT)0 : top_df1Y[i]);
#else
                        lcl_mem[p * MLO_READ_UNIT + i] = (_FLOAT2)(top_df1X[i], top_df1Y[i]);
#endif
#if DBG_OUT_OF_RNGE
                        if(top_off1 + i >= MLO_OUT_BATCH_STRIDE * MLO_BATCH_SZ)
                        {
                            printf("k:err:out-off-range\n");
                        }
#endif
                    }
                }

            } // for(int p = 0; p < MLO_OUT_LCL_BLK * MLO_MAP_WK_SZ; p += MLO_GRP_SZ)

            barrier(CLK_LOCAL_MEM_FENCE);

            for(uint k = kb * MLO_OUT_LCL_BLK; k < (kb + 1) * MLO_OUT_LCL_BLK; ++k)
            {
                // processing
                for(uint c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
                {
                    for(uint i = 0; i < MLO_READ_UNIT; ++i)
                    {
                        _FLOAT2 bot_val = bot_dat[c * MLO_READ_UNIT + i];
                        _FLOAT2 top_val =
                            lcl_mem[((k - kb * MLO_OUT_LCL_BLK) * MLO_MAP_WK_SZ + p4) *
                                        MLO_READ_UNIT +
                                    i];
                        pvt_accum[k * MLO_N_LCL_IN_MAPS + c] += top_val * bot_val;
#if 0
						if (top_val * bot_val !=0  &&  k == 0 && c == 0 && m_id == 0 && get_group_id(0) == 0 && get_group_id(1) == 1 && lcl_id ==0 )
						{

							printf("K:c: %d %d %d %d %f %f %f %f\n",
								c_idx + m_id + c*MLO_N_MAPS_PER_GROUP,
								lcl_id,
								c*MLO_READ_UNIT + i,
								((k - kb*MLO_OUT_LCL_BLK) * MLO_MAP_WK_SZ + p4)*MLO_READ_UNIT + i,
								pvt_accum[k * MLO_N_LCL_IN_MAPS + c],
								top_val * bot_val,
								bot_val,
								top_val

							);
						}

#endif
                    }
                }
            }

        } // for(int kb = 0; kb < MLO_N_LCL_OUT; kb++, top_off += MLO_OUT_LCL_BLK *
          // MLO_OUT_CHANNEL_STRIDE)
    } // for (int b = 0; b < MLO_BATCH_SZ; ++b, gbl_in_off += MLO_IN_BATCH_STRIDE, gbl_out_off +=
      // MLO_OUT_BATCH_STRIDE)

    // FINAL REDUCTION

    // write out
    // inputs are outputs
    uint wei_df_off = ((ib * MLO_N_OUTPUTS + k_idx) * (int)MLO_WEI_BATCH_STRIDE) +
                      (c_idx + m_id) * MLO_WEI_CHANNEL_STRIDE;

#define MLO_N_FIRST_SPLITS (1 << (MLO_LG2_REDUC_ROUNDS - 1))
    // transpose data using MLO_REDUC_LOOP_STEP wk-items from each small map
    __private _FLOAT2 final_sum[(MLO_ACCUM_SZ / MLO_REDUC_LOOP_STEP)];

// final log reduction with the initail input not pow2
#if 1
    // first round
    // transpose and split into sub-group for logar summation
    for(uint r = 0; r < (MLO_ACCUM_SZ / MLO_REDUC_LOOP_STEP); ++r)
    {
        final_sum[r] = (_FLOAT2)(0);

        barrier(CLK_LOCAL_MEM_FENCE);
// write out only valid pixels
#if MLO_MAP_WK_SZ < MLO_GRP_SZ
        if(lcl_id < MLO_N_MAPS_PER_GROUP * MLO_MAP_WK_SZ)
#endif

        {

            for(uint rr = 0; rr < MLO_REDUC_LOOP_STEP; ++rr)
            {
                lcl_mem[lcl_id * MLO_REDUC_LOOP_STEP + rr] =
                    pvt_accum[r * MLO_REDUC_LOOP_STEP + rr];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if(p4 < (MLO_REDUC_LOOP_STEP << (MLO_LG2_REDUC_ROUNDS - 1)))
        {
// what split the pix belong to
#if(MLO_REDUC_LOOP_STEP & (MLO_REDUC_LOOP_STEP - 1))
            uint split     = IDIV(p4, MLO_REDUC_LOOP_STEP);
            uint split_pix = IMOD(p4, split, MLO_REDUC_LOOP_STEP);
#else
            uint split                                 = ((uint)p4 / MLO_REDUC_LOOP_STEP);
            uint split_pix                             = ((uint)p4 & (MLO_REDUC_LOOP_STEP - 1));
#endif

            for(uint j = 0; j < MLO_FIRST_ROUND; j++)
            {
#if MLO_FIRST_CAN_DIVIDE == 0
                if(split * MLO_FIRST_ROUND + j < MLO_MAP_WK_SZ)
#endif
                {
                    final_sum[r] += lcl_mem[(m_id * MLO_MAP_WK_SZ + split * MLO_FIRST_ROUND + j) *
                                                MLO_REDUC_LOOP_STEP +
                                            split_pix];
                }
            }
        }
    }

#if MLO_LG2_REDUC_ROUNDS > 1
    // log summation
    for(int rd = (MLO_LG2_REDUC_ROUNDS - 2); rd >= 0; --rd)
    {

        barrier(CLK_LOCAL_MEM_FENCE);

        if(p4 >= ((uint)MLO_REDUC_LOOP_STEP << (uint)rd) &&
           p4 < ((uint)MLO_REDUC_LOOP_STEP << (uint)(rd + 1)) && m_id < MLO_N_MAPS_PER_GROUP)
        {
            for(uint rr = 0; rr < (MLO_ACCUM_SZ / MLO_REDUC_LOOP_STEP); ++rr)
            {
                int base_off           = (rr * MLO_N_MAPS_PER_GROUP + m_id) * MLO_MAP_WK_SZ;
                lcl_mem[base_off + p4] = final_sum[rr];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if(p4 < ((uint)MLO_REDUC_LOOP_STEP << (uint)rd) && m_id < MLO_N_MAPS_PER_GROUP)
        {
            for(uint rr = 0; rr < (MLO_ACCUM_SZ / MLO_REDUC_LOOP_STEP); ++rr)
            {
                int base_off = (rr * MLO_N_MAPS_PER_GROUP + m_id) * MLO_MAP_WK_SZ;
                final_sum[rr] += lcl_mem[base_off + (MLO_REDUC_LOOP_STEP << rd) + p4];
            }
        }
    }

#endif

    if(p4 < MLO_REDUC_LOOP_STEP)
    {
        for(uint r = 0; r < (MLO_ACCUM_SZ / MLO_REDUC_LOOP_STEP); ++r)
        {
            uint wei_idx = r * MLO_REDUC_LOOP_STEP + p4;

#if(MLO_N_LCL_IN_MAPS & (MLO_N_LCL_IN_MAPS - 1))
            uint k = IDIV(wei_idx, MLO_N_LCL_IN_MAPS);
            uint c = IMOD(wei_idx, k, MLO_N_LCL_IN_MAPS);
#else
            uint k                                     = ((uint)wei_idx / MLO_N_LCL_IN_MAPS);
            uint c                                     = ((uint)wei_idx & (MLO_N_LCL_IN_MAPS - 1));
#endif

            if(m_id < MLO_N_MAPS_PER_GROUP
#if MLO_N_IN_MAPS_ALIGNED == 0
               &&
               (c_idx + m_id + c * MLO_N_MAPS_PER_GROUP) < MLO_N_INPUTS
#endif
#if MLO_N_OUT_MAPS_ALIGNED == 0
               &&
               k_idx + k < MLO_N_OUTPUTS
#endif
               )
            {
                uint wei_off = wei_df_off + k * MLO_WEI_BATCH_STRIDE +
                               c * MLO_N_MAPS_PER_GROUP * MLO_WEI_CHANNEL_STRIDE;
                weights_df[wei_off] = final_sum[r].x
#if MLO_BATCH_SZ != 1
                                      + final_sum[r].y
#endif
                    ;
            }

        } // for (int r = 0; r < (MLO_ACCUM_SZ / MLO_REDUC_LOOP_STEP); ++r)
    }     // if (p4 < MLO_REDUC_LOOP_STEP)

// naive reduction
#elif 1
    //	if (inside_range_input)
    {
        for(int r = 0; r < (MLO_ACCUM_SZ / MLO_REDUC_LOOP_STEP); ++r)
        {
            final_sum[r] = (_FLOAT2)(0);

            barrier(CLK_LOCAL_MEM_FENCE);

            for(int rr = 0; rr < MLO_REDUC_LOOP_STEP; ++rr)
            {
                lcl_mem[lcl_id * MLO_REDUC_LOOP_STEP + rr] =
                    pvt_accum[r * MLO_REDUC_LOOP_STEP + rr];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if(p4 < MLO_REDUC_LOOP_STEP)
            {
                for(int j = 0; j < MLO_MAP_WK_SZ; j++)
                {
                    final_sum[r] += lcl_mem[(m_id * MLO_MAP_WK_SZ + j) * MLO_REDUC_LOOP_STEP + p4];
                }
            }
        }

        if(p4 < MLO_REDUC_LOOP_STEP)
        {
            for(int r = 0; r < (MLO_ACCUM_SZ / MLO_REDUC_LOOP_STEP); ++r)
            {
                int wei_idx = r * (MLO_REDUC_LOOP_STEP) + p4;

#if(MLO_N_LCL_IN_MAPS & (MLO_N_LCL_IN_MAPS - 1))
                int k = IDIV(wei_idx, MLO_N_LCL_IN_MAPS);
                int c = IMOD(wei_idx, k, MLO_N_LCL_IN_MAPS);
#else
                int k = ((uint)wei_idx / MLO_N_LCL_IN_MAPS);
                int c = ((uint)wei_idx & (MLO_N_LCL_IN_MAPS - 1));
#endif

                if(m_id < MLO_N_MAPS_PER_GROUP &&
                   (c_idx + m_id + c * MLO_N_MAPS_PER_GROUP) < MLO_N_INPUTS
#if MLO_N_OUT_MAPS_ALIGNED == 0
                   &&
                   k_idx + k < MLO_N_OUTPUTS
#endif
                   )
                {
                    int wei_off = wei_df_off + k * MLO_WEI_BATCH_STRIDE +
                                  c * MLO_N_MAPS_PER_GROUP * MLO_WEI_CHANNEL_STRIDE;
                    weights_df[wei_off] = final_sum[r].x
#if MLO_BATCH_SZ != 1
                                          + final_sum[r].y
#endif
                        ;
                }

            } // for (int r = 0; r < (MLO_ACCUM_SZ / MLO_REDUC_LOOP_STEP); ++r)
        }     // if (p4 < MLO_REDUC_LOOP_STEP)
    }         // if (inside_range_input)

// verification
#else

    for(int r = 0; r < (MLO_ACCUM_SZ / MLO_REDUC_LOOP_STEP); ++r)
    {
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int rr = 0; rr < MLO_REDUC_LOOP_STEP; ++rr)
        {
            lcl_mem[lcl_id * MLO_REDUC_LOOP_STEP + rr] = pvt_accum[r * MLO_REDUC_LOOP_STEP + rr];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if(p4 == 0 && inside_range_input)
        {
            for(int j = 1; j < MLO_MAP_WK_SZ; j++)
            {
                for(int rr = 0; rr < MLO_REDUC_LOOP_STEP; ++rr)
                {
                    pvt_accum[r * MLO_REDUC_LOOP_STEP + rr] +=
                        lcl_mem[(lcl_id + j) * MLO_REDUC_LOOP_STEP + rr];
                }
            }
        }
    }

    if(p4 == 0)
    {
        for(int kb = 0; kb < MLO_N_LCL_OUT; kb++)
        {
            for(int k = kb * MLO_OUT_LCL_BLK; k < (kb + 1) * MLO_OUT_LCL_BLK; ++k)
            {
                for(int c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
                {
                    if((c_idx + m_id + c * MLO_N_MAPS_PER_GROUP) < MLO_N_INPUTS
#if MLO_N_OUT_MAPS_ALIGNED == 0
                       &&
                       k_idx + k < MLO_N_OUTPUTS
#endif
                       )
                    {
                        int wei_off = wei_df_off + k * MLO_WEI_BATCH_STRIDE +
                                      c * MLO_N_MAPS_PER_GROUP * MLO_WEI_CHANNEL_STRIDE;
                        weights_df[wei_off] = pvt_accum[k * MLO_N_LCL_IN_MAPS + c].x
#if MLO_BATCH_SZ != 1
                                              + pvt_accum[k * MLO_N_LCL_IN_MAPS + c].y
#endif
                            ;
                    }
                }
            }
        }
    }

#endif
}

/*

  Large maps

*/

#undef MLO_N_MAPS_PER_GROUP

__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2))) __kernel void
MIOpenCvBwdWrWLmap(const __global _FLOAT* __restrict top_df,
                   const __global _FLOAT* __restrict bot,
                   __global _FLOAT* __restrict weights_df,
                   UNUSED _FLOAT padding_val)
{
    // reduction memory.

    __local _FLOAT2 lcl_mem[MLO_LCL_MEM_SZ];

    uint lcl_id = get_local_id(0);

    uint k_idx = get_group_id(0) * (MLO_N_LCL_OUT_MAPS); // output map index base

    uint c_idx = get_group_id(1) * (MLO_N_LCL_IN_MAPS); // input map index based

    uint gbl_in_off0  = c_idx * MLO_IN_CHANNEL_STRIDE;
    uint gbl_out_off0 = k_idx * MLO_OUT_CHANNEL_STRIDE;

    __private _FLOAT2 top_dat[MLO_TOP_DAT_SZ];

    __private _FLOAT2 bot_dat[MLO_BOT_DAT_SZ];

    __private _FLOAT2 pvt_accum[MLO_ACCUM_SZ] = {(MLO_ACCUM_SZ) * (_FLOAT2)(0)};

    //	for (uint i = 0; i < MLO_ACCUM_SZ; ++i)
    //	{
    //		pvt_accum[i] = (_FLOAT2) (0);
    //	}

    for(uint i = lcl_id; i < MLO_LCL_MEM_SZ; i += MLO_GRP_SZ)
    {
        lcl_mem[i] = (_FLOAT2)(0);
    }

    for(uint pix4 = lcl_id; pix4 < MLO_MAP_WK_SZ * MLO_BATCH_SZ; pix4 += 2 * MLO_GRP_SZ)
    {

        bool IsLast   = ((pix4 + MLO_GRP_SZ) >= (MLO_MAP_WK_SZ * MLO_BATCH_SZ));
        uint2 pix4_v2 = (uint2)(pix4, pix4 + MLO_GRP_SZ);
#if(MLO_MAP_WK_SZ) & (MLO_MAP_WK_SZ - 1)

        uint2 b  = (uint2)(IDIV(pix4_v2.x, MLO_MAP_WK_SZ), IDIV(pix4_v2.y, MLO_MAP_WK_SZ)); // batch
        uint2 p4 = (uint2)(IMOD(pix4_v2.x, b.x, MLO_MAP_WK_SZ),
                           IMOD(pix4_v2.y, b.y, MLO_MAP_WK_SZ)); // pixel block
#else
        uint2 b  = pix4_v2 / (uint2)(MLO_MAP_WK_SZ);     // batch
        uint2 p4 = pix4_v2 & (uint2)(MLO_MAP_WK_SZ - 1); // pixel block

#endif
        uint2 gbl_in_off =
            (uint2)(gbl_in_off0) + b * (uint2)(MLO_IN_BATCH_STRIDE) + p4 * (uint2)(MLO_READ_UNIT);
        uint2 gbl_out_off =
            (uint2)(gbl_out_off0) + b * (uint2)(MLO_OUT_BATCH_STRIDE) + p4 * (uint2)(MLO_READ_UNIT);

// Set-up 1st vector element
#if MLO_N_PIXS_OFF > 0
        bool last_pixelX = (p4.x == MLO_MAP_WK_SZ - 1);

        if(last_pixelX)
        {
            for(uint i = 0; i < MLO_TOP_DAT_SZ; ++i)
            {
                top_dat[i].x = (_FLOAT)0;
            }
            for(uint i = 0; i < MLO_BOT_DAT_SZ; ++i)
            {
                bot_dat[i].x = (_FLOAT)0;
            }
            for(uint c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
            {
                uint bot_off = gbl_in_off.x + c * MLO_IN_CHANNEL_STRIDE;
#if MLO_N_IN_MAPS_ALIGNED == 0
                // reading garbage, will be thrown away on the way output
                bot_off = (c_idx + c < MLO_N_INPUTS) ? bot_off : (uint)0;
#endif

                const __global _FLOAT* bot1 = &bot[bot_off];

                for(uint i = 0; i < MLO_N_PIXS_OFF; ++i)
                {
                    bot_dat[c * MLO_READ_UNIT + i].x = bot1[i];
#if DBG_OUT_OF_RNGE
                    if(bot_off + i >= MLO_IN_BATCH_STRIDE * MLO_BATCH_SZ)
                    {
                        printf("k:err:in-off-range\n");
                    }
#endif
                }
            }
            for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
            {
                uint top_off = gbl_out_off.x + k * MLO_OUT_CHANNEL_STRIDE;

#if MLO_N_OUT_MAPS_ALIGNED == 0
                top_off = (k_idx + k < MLO_N_OUTPUTS) ? top_off : (uint)0;
#endif
                const __global _FLOAT* top_df1 = &top_df[top_off];

                for(uint i = 0; i < MLO_N_PIXS_OFF; ++i)
                {
                    top_dat[k * MLO_READ_UNIT + i].x = top_df1[i];
#if DBG_OUT_OF_RNGE
                    if(top_off + i >= MLO_OUT_BATCH_STRIDE * MLO_BATCH_SZ)
                    {
                        printf("k:err:out-off-range\n");
                    }
#endif
                }
            }
        }
        else
#endif
        {
            for(uint c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
            {
                uint bot_off = gbl_in_off.x + c * MLO_IN_CHANNEL_STRIDE;
#if MLO_N_IN_MAPS_ALIGNED == 0
                // reading garbage, will be thrown away on the way output
                bot_off = (c_idx + c < MLO_N_INPUTS) ? bot_off : (uint)0;
#endif
                const __global _FLOAT* bot1 = &bot[bot_off];

                for(uint i = 0; i < MLO_READ_UNIT; ++i)
                {
                    bot_dat[c * MLO_READ_UNIT + i].x = bot1[i];
#if DBG_OUT_OF_RNGE
                    if(bot_off + i >= MLO_IN_BATCH_STRIDE * MLO_BATCH_SZ)
                    {
                        printf("k:err:in-off-range\n");
                    }
#endif
                }
            }
            for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
            {
                uint top_off = gbl_out_off.x + k * MLO_OUT_CHANNEL_STRIDE;
#if MLO_N_OUT_MAPS_ALIGNED == 0
                top_off = (k_idx + k < MLO_N_OUTPUTS) ? top_off : (uint)0;
#endif
                const __global _FLOAT* top_df1 = &top_df[top_off];

                for(uint i = 0; i < MLO_READ_UNIT; ++i)
                {
                    top_dat[k * MLO_READ_UNIT + i].x = top_df1[i];
#if DBG_OUT_OF_RNGE
                    if(top_off + i >= MLO_OUT_BATCH_STRIDE * MLO_BATCH_SZ)
                    {
                        printf("k:err:out-off-range\n");
                    }
#endif
                }
            }
        }

// Set-up 2nd vector elemnt
#if MLO_N_PIXS_OFF > 0
        bool last_pixelY = (p4.y == MLO_MAP_WK_SZ - 1);

        if(last_pixelY)
        {
            for(uint i = 0; i < MLO_TOP_DAT_SZ; ++i)
            {
                top_dat[i].y = (_FLOAT)0;
            }
            for(uint i = 0; i < MLO_BOT_DAT_SZ; ++i)
            {
                bot_dat[i].y = (_FLOAT)0;
            }
            for(uint c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
            {
                uint bot_off = gbl_in_off.y + c * MLO_IN_CHANNEL_STRIDE;
#if MLO_N_IN_MAPS_ALIGNED == 0
                // reading garbage, will be thrown away on the way output
                bot_off = (c_idx + c < MLO_N_INPUTS) ? bot_off : (uint)0;
#endif

                const __global _FLOAT* bot1 = &bot[bot_off];

                for(uint i = 0; i < MLO_N_PIXS_OFF; ++i)
                {
                    //#if (MLO_MAP_WK_SZ * MLO_BATCH_SZ)%(2*MLO_GRP_SZ) <= MLO_GRP_SZ &&
                    //(MLO_MAP_WK_SZ * MLO_BATCH_SZ)%(2*MLO_GRP_SZ) > 0
                    bot_dat[c * MLO_READ_UNIT + i].y = (IsLast) ? (_FLOAT)0 : bot1[i];
                    //#else
                    //					bot_dat[c*MLO_READ_UNIT + i].y = bot1[i];
                    //#endif
                }
            }
            for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
            {
                uint top_off = gbl_out_off.y + k * MLO_OUT_CHANNEL_STRIDE;

#if MLO_N_OUT_MAPS_ALIGNED == 0
                top_off = (k_idx + k < MLO_N_OUTPUTS) ? top_off : (uint)0;
#endif
                const __global _FLOAT* top_df1 = &top_df[top_off];

                for(uint i = 0; i < MLO_N_PIXS_OFF; ++i)
                {
                    //#if (MLO_MAP_WK_SZ * MLO_BATCH_SZ)%(2*MLO_GRP_SZ) <= MLO_GRP_SZ &&
                    //(MLO_MAP_WK_SZ * MLO_BATCH_SZ)%(2*MLO_GRP_SZ) > 0
                    top_dat[k * MLO_READ_UNIT + i].y = (IsLast) ? (_FLOAT)0 : top_df1[i];
                    //#else
                    //					top_dat[k*MLO_READ_UNIT + i].y = top_df1[i];
                    //#endif
                }
            }
        }
        else
#endif
        {
            for(uint c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
            {
                uint bot_off = gbl_in_off.y + c * MLO_IN_CHANNEL_STRIDE;
#if MLO_N_IN_MAPS_ALIGNED == 0
                // reading garbage, will be thrown away on the way output
                bot_off = (c_idx + c < MLO_N_INPUTS) ? bot_off : (uint)0;
#endif
                const __global _FLOAT* bot1 = &bot[bot_off];

                for(uint i = 0; i < MLO_READ_UNIT; ++i)
                {
                    //#if (MLO_MAP_WK_SZ * MLO_BATCH_SZ)%(2*MLO_GRP_SZ) <= MLO_GRP_SZ &&
                    //(MLO_MAP_WK_SZ * MLO_BATCH_SZ)%(2*MLO_GRP_SZ) > 0
                    bot_dat[c * MLO_READ_UNIT + i].y = (IsLast) ? (_FLOAT)0 : bot1[i];
                    //#else
                    //					bot_dat[c*MLO_READ_UNIT + i].y = bot1[i];
                    //#endif
                }
            }
            for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
            {
                uint top_off = gbl_out_off.y + k * MLO_OUT_CHANNEL_STRIDE;
#if MLO_N_OUT_MAPS_ALIGNED == 0
                top_off = (k_idx + k < MLO_N_OUTPUTS) ? top_off : (uint)0;
#endif
                const __global _FLOAT* top_df1 = &top_df[top_off];

                for(uint i = 0; i < MLO_READ_UNIT; ++i)
                {
                    //#if (MLO_MAP_WK_SZ * MLO_BATCH_SZ)%(2*MLO_GRP_SZ) <= MLO_GRP_SZ &&
                    //(MLO_MAP_WK_SZ * MLO_BATCH_SZ)%(2*MLO_GRP_SZ) > 0
                    top_dat[k * MLO_READ_UNIT + i].y = (IsLast) ? (_FLOAT)0 : top_df1[i];
                    //#else
                    //					top_dat[k*MLO_READ_UNIT + i].y = top_df1[i];
                    //#endif
                }
            }
        }

        // processing
        for(uint k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
        {

            for(uint c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
            {
                for(uint i = 0; i < MLO_READ_UNIT; ++i)
                {
                    pvt_accum[k * MLO_N_LCL_IN_MAPS + c] +=
                        bot_dat[c * MLO_READ_UNIT + i] * top_dat[k * MLO_READ_UNIT + i];

#if 0
					if (get_group_id(0) == 1 && lcl_id == 0 && k == 0 )
					{
						printf("K:c: %f %f %f %f\n",
							pvt_accum[k * MLO_N_LCL_IN_MAPS + c],
							bot_dat[c*MLO_READ_UNIT + i] * top_dat[k*MLO_READ_UNIT + i],
							bot_dat[c*MLO_READ_UNIT + i],
							top_dat[k*MLO_READ_UNIT + i]
							);
					}
#endif
                }
            }
        }
    }

    // FINAL REDUCTION
    // write out
    // inputs are outputs
    uint wei_df_off = k_idx * MLO_WEI_BATCH_STRIDE + c_idx * MLO_WEI_CHANNEL_STRIDE;

    // transpose data using MLO_REDUC_LOOP_STEP wk-items from each small map
    __private _FLOAT2 final_sum[(MLO_ACCUM_SZ / MLO_REDUC_LOOP_STEP)];

// final log reduction with the initail input not pow2
#if 1

    // first round
    // transpose and split into sub-group for logar summation
    for(uint r = 0; r < (MLO_ACCUM_SZ / MLO_REDUC_LOOP_STEP); ++r)
    {
        final_sum[r] = (_FLOAT2)(0);

        barrier(CLK_LOCAL_MEM_FENCE);
        // write out only valid pixels

        for(uint rr = 0; rr < MLO_REDUC_LOOP_STEP; ++rr)
        {
            lcl_mem[lcl_id * MLO_REDUC_LOOP_STEP + rr] = pvt_accum[r * MLO_REDUC_LOOP_STEP + rr];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if(lcl_id < (MLO_REDUC_LOOP_STEP << (MLO_LG2_REDUC_ROUNDS - 1)))
        {
// what split the pix belong to
#if(MLO_REDUC_LOOP_STEP & (MLO_REDUC_LOOP_STEP - 1))
            uint split     = IDIV(lcl_id, MLO_REDUC_LOOP_STEP);
            uint split_pix = IMOD(lcl_id, split, MLO_REDUC_LOOP_STEP);
#else
            uint split                                 = ((uint)lcl_id / MLO_REDUC_LOOP_STEP);
            uint split_pix                             = ((uint)lcl_id & (MLO_REDUC_LOOP_STEP - 1));
#endif

            for(uint j = 0; j < MLO_FIRST_ROUND; j++)
            {
#if MLO_FIRST_CAN_DIVIDE == 0
                if(split * MLO_FIRST_ROUND + j < MLO_GRP_SZ)
#endif
                {
                    final_sum[r] +=
                        lcl_mem[(split * MLO_FIRST_ROUND + j) * MLO_REDUC_LOOP_STEP + split_pix];
                }
            }
        }
    }

    // log summation
    for(int rd = (MLO_LG2_REDUC_ROUNDS - 2); rd >= 0; --rd)
    {

        barrier(CLK_LOCAL_MEM_FENCE);

        if(lcl_id >= ((uint)MLO_REDUC_LOOP_STEP << (uint)rd) &&
           lcl_id < ((uint)MLO_REDUC_LOOP_STEP << (uint)(rd + 1)))
        {
            for(uint rr = 0; rr < (MLO_ACCUM_SZ / MLO_REDUC_LOOP_STEP); ++rr)
            {
                int base_off               = rr * MLO_GRP_SZ;
                lcl_mem[base_off + lcl_id] = final_sum[rr];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if(lcl_id < ((uint)MLO_REDUC_LOOP_STEP << (uint)rd))
        {
            for(uint rr = 0; rr < (MLO_ACCUM_SZ / MLO_REDUC_LOOP_STEP); ++rr)
            {
                int base_off = rr * MLO_GRP_SZ;
                final_sum[rr] += lcl_mem[base_off + (MLO_REDUC_LOOP_STEP << rd) + lcl_id];
            }
        }
    }

    if(lcl_id < MLO_REDUC_LOOP_STEP)
    {
        for(uint r = 0; r < (MLO_ACCUM_SZ / MLO_REDUC_LOOP_STEP); ++r)
        {
            uint wei_idx = r * MLO_REDUC_LOOP_STEP + lcl_id;

#if(MLO_N_LCL_IN_MAPS & (MLO_N_LCL_IN_MAPS - 1))
            uint k = IDIV(wei_idx, MLO_N_LCL_IN_MAPS);
            uint c = IMOD(wei_idx, k, MLO_N_LCL_IN_MAPS);
#else
            uint k                                     = ((uint)wei_idx / MLO_N_LCL_IN_MAPS);
            uint c                                     = ((uint)wei_idx & (MLO_N_LCL_IN_MAPS - 1));
#endif

            if(true
#if MLO_N_IN_MAPS_ALIGNED == 0
               &&
               c_idx + c < MLO_N_INPUTS
#endif
#if MLO_N_OUT_MAPS_ALIGNED == 0
               &&
               k_idx + k < MLO_N_OUTPUTS
#endif
               )
            {
                uint wei_off = wei_df_off + k * MLO_WEI_BATCH_STRIDE + c * MLO_WEI_CHANNEL_STRIDE;
                weights_df[wei_off] = final_sum[r].x + final_sum[r].y;
            }

        } // for (int r = 0; r < (MLO_ACCUM_SZ / MLO_REDUC_LOOP_STEP); ++r)
    }     // if (p4 < MLO_REDUC_LOOP_STEP)

// naive reduction
#elif 1
    //	if (inside_range_input)
    {
        for(int r = 0; r < (MLO_ACCUM_SZ / MLO_REDUC_LOOP_STEP); ++r)
        {
            final_sum[r] = (_FLOAT2)(0);

            barrier(CLK_LOCAL_MEM_FENCE);

            for(int rr = 0; rr < MLO_REDUC_LOOP_STEP; ++rr)
            {
                lcl_mem[lcl_id * MLO_REDUC_LOOP_STEP + rr] =
                    pvt_accum[r * MLO_REDUC_LOOP_STEP + rr];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if(lcl_id < MLO_REDUC_LOOP_STEP)
            {
                for(int j = 0; j < MLO_GRP_SZ; j++)
                {
                    final_sum[r] += lcl_mem[j * MLO_REDUC_LOOP_STEP + lcl_id];
                }
            }
        }

        if(lcl_id < MLO_REDUC_LOOP_STEP)
        {
            for(int r = 0; r < (MLO_ACCUM_SZ / MLO_REDUC_LOOP_STEP); ++r)
            {
                int wei_idx = r * (MLO_REDUC_LOOP_STEP) + lcl_id;

#if(MLO_N_LCL_IN_MAPS & (MLO_N_LCL_IN_MAPS - 1))
                int k = IDIV(wei_idx, MLO_N_LCL_IN_MAPS);
                int c = IMOD(wei_idx, k, MLO_N_LCL_IN_MAPS);
#else
                int k = ((uint)wei_idx / MLO_N_LCL_IN_MAPS);
                int c = ((uint)wei_idx & (MLO_N_LCL_IN_MAPS - 1));
#endif

                if(true
#if MLO_N_IN_MAPS_ALIGNED == 0
                   &&
                   c_idx + c < MLO_N_INPUTS
#endif
#if MLO_N_OUT_MAPS_ALIGNED == 0
                   &&
                   k_idx + k < MLO_N_OUTPUTS
#endif
                   )
                {
                    int wei_off =
                        wei_df_off + k * MLO_WEI_BATCH_STRIDE + c * MLO_WEI_CHANNEL_STRIDE;
                    weights_df[wei_off] = final_sum[r].x
#if(MLO_MAP_WK_SZ * MLO_BATCH_SZ) <= MLO_GRP_SZ
                                          + final_sum[r].y
#endif
                        ;
                }

            } // for (int r = 0; r < (MLO_ACCUM_SZ / MLO_REDUC_LOOP_STEP); ++r)
        }     // if (p4 < MLO_REDUC_LOOP_STEP)
    }         // if (inside_range_input)
              // verification
#else
    for(int r = 0; r < (MLO_ACCUM_SZ / MLO_REDUC_LOOP_STEP); ++r)
    {
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int rr = 0; rr < MLO_REDUC_LOOP_STEP; ++rr)
        {
            lcl_mem[lcl_id * MLO_REDUC_LOOP_STEP + rr] = pvt_accum[r * MLO_REDUC_LOOP_STEP + rr];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if(lcl_id == 0)
        {
            for(int j = 1; j < MLO_GRP_SZ; j++)
            {
                for(int rr = 0; rr < MLO_REDUC_LOOP_STEP; ++rr)
                {
                    pvt_accum[r * MLO_REDUC_LOOP_STEP + rr] +=
                        lcl_mem[(lcl_id + j) * MLO_REDUC_LOOP_STEP + rr];
                }
            }
        }
    }

    if(lcl_id == 0)
    {
        for(int k = 0; k < MLO_N_LCL_OUT_MAPS; ++k)
        {
            for(int c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
            {
                if(true
#if MLO_N_IN_MAPS_ALIGNED == 0
                   &&
                   c_idx + c < MLO_N_INPUTS
#endif
#if MLO_N_OUT_MAPS_ALIGNED == 0
                   &&
                   k_idx + k < MLO_N_OUTPUTS
#endif
                   )
                {
                    int wei_off =
                        wei_df_off + k * MLO_WEI_BATCH_STRIDE + c * MLO_WEI_CHANNEL_STRIDE;
                    weights_df[wei_off] = pvt_accum[k * MLO_N_LCL_IN_MAPS + c] + (IsLast)
                                              ? (_FLOAT)0
                                              : pvt_accum[k * MLO_N_LCL_IN_MAPS + c].y;
                }
            }
        }
    }

#endif
}
