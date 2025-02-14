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
#define _FLOAT_PREC half
#define MAX_VAL 65504 /* max value */
#else
#define MAX_VAL HALF_MAX
#endif
#endif
#if MIOPEN_USE_FP32 == 1
#define _FLOAT float
#define _FLOAT_PREC float
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

#ifndef MLO_FILTER_STRIDE0
#define MLO_FILTER_STRIDE0 1
#endif
#ifndef MLO_FILTER_STRIDE1
#define MLO_FILTER_STRIDE1 1
#endif

#define MLO_FILTER_SZ (MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0)

#define MLO_GRP_SZ0 (MLO_GRP_TILE0 * MLO_GRP_TILE1)
#define MLO_GRP_SZ1 1
#define MLO_GRP_SZ2 1
#define MLO_GRP_SZ (MLO_GRP_SZ0 * MLO_GRP_SZ1 * MLO_GRP_SZ2)
#define MLO_N_PROC_WAVES ((MLO_GRP_SZ + MLO_N_READ_PROCS - 1) / MLO_N_READ_PROCS)
#define MLO_OUT_TILE_SZ (MLO_OUT_TILE1 * MLO_OUT_TILE0)
#define MLO_ALU_TILE_SZ (MLO_ALU_VTILE1 * MLO_ALU_VTILE0)

#if MLO_IN_TILE0 < MLO_OUT_WIDTH || MLO_IN_TILE1 < MLO_OUT_HEIGHT
#define MLO_LARGE_MAP 1
#else
#define MLO_LARGE_MAP 0
#endif

#if(MLO_IN_WIDTH == MLO_OUT_WIDTH &&                                \
    (MLO_IN_WIDTH / MLO_IN_TILE0) * MLO_IN_TILE0 == MLO_IN_WIDTH && \
    MLO_IN_HEIGHT == MLO_OUT_HEIGHT &&                              \
    (MLO_IN_HEIGHT / MLO_IN_TILE1) * MLO_IN_TILE1 == MLO_IN_HEIGHT)
#define MLO_OUT_ALIGNED 1
#else
#define MLO_OUT_ALIGNED 0
#endif

#define MLO_N_ALUTILES_TOTAL ((MLO_GRP_TILE0 * MLO_GRP_TILE1) / (MLO_ALU_TILE_SZ))
#define MLO_N_ALUTILES_PERSTACK (MLO_N_ALUTILES_TOTAL / MLO_N_STACKS)
#define MLO_ALUTILES_STACK_SZ (MLO_N_ALUTILES_PERSTACK * MLO_ALU_TILE_SZ)
#define MLO_N_IN_TILES_TOTAL (MLO_N_IN_TILES_PERSTACK * MLO_N_STACKS)
/*
#define MLO_N_OUT_TILES_PERSTACK (MLO_N_OUT_TILES*MLO_N_ALUTILES_PERSTACK)
#if MLO_N_OUT_TILES_PERSTACK > MLO_N_OUTPUTS
#undef MLO_N_OUT_TILES_PERSTACK
#define MLO_N_OUT_TILES_PERSTACK MLO_N_OUTPUTS
#endif
*/
#define MLO_N_OUT_TILE_BLOCKS0 ((MLO_OUT_WIDTH + MLO_IN_TILE0 - 1) / MLO_IN_TILE0)
#define MLO_N_OUT_TILE_BLOCKS1 ((MLO_OUT_HEIGHT + MLO_IN_TILE1 - 1) / MLO_IN_TILE1)
#define MLO_N_IN_PACKS ((MLO_N_INPUTS + MLO_N_IN_TILES_PERSTACK - 1) / MLO_N_IN_TILES_PERSTACK)

#define MLO_N_IN_READ (MLO_N_IN_PACKS * MLO_N_IN_TILES_PERSTACK)
#if MLO_N_IN_READ == MLO_N_INPUTS
#define MLO_INPUTS_ALIGNED 1
#else
#define MLO_INPUTS_ALIGNED 0
#endif

#define MLO_N_OUT_PACKS (MLO_N_OUTPUTS / MLO_N_OUT_TILES_PERSTACK)
#if MLO_N_OUT_PACKS * MLO_N_OUT_TILES_PERSTACK == MLO_N_OUTPUTS && \
    MLO_N_OUT_TILES_PERSTACK != MLO_N_OUTPUTS
#define MLO_OUTPUTS_ALIGNED 1
#else
#define MLO_OUTPUTS_ALIGNED 0
#endif

#define MLO_N_BATCH_PACKS (MLO_BATCH_SZ / MLO_N_STACKS)
#if MLO_N_BATCH_PACKS * MLO_N_STACKS == MLO_BATCH_SZ && MLO_N_STACKS != MLO_BATCH_SZ
#define MLO_BATCH_ALIGNED 1
#else
#define MLO_BATCH_ALIGNED 0
#endif

#if MLO_DIR_FORWARD == 1
#define MLO_IN_LCL_WIDTH \
    ((MLO_IN_TILE0 - 1) * MLO_FILTER_STRIDE0 + MLO_FILTER_SIZE0) // here we use kernel size. it's
                                                                 // important when padding == 0  2*
                                                                 // MLO_FILTER_PAD0
#define MLO_IN_LCL_HEIGHT ((MLO_IN_TILE1 - 1) * MLO_FILTER_STRIDE1 + MLO_FILTER_SIZE1)
#else
#define MLO_IN_LCL_WIDTH                                              \
    ((MLO_IN_TILE0 + MLO_FILTER_SIZE0 - 1 + MLO_FILTER_STRIDE0 - 1) / \
     MLO_FILTER_STRIDE0) // here we use kernel size. it's important when padding == 0  2*
// MLO_FILTER_PAD0
#define MLO_IN_LCL_HEIGHT \
    ((MLO_IN_TILE1 + MLO_FILTER_SIZE1 - 1 + MLO_FILTER_STRIDE1 - 1) / MLO_FILTER_STRIDE1)
#endif
#define MLO_IN_LCL_TILE_SZ (MLO_IN_LCL_WIDTH * MLO_IN_LCL_HEIGHT)
#define MLO_IN_LCL_PERSTACK_SZ (MLO_IN_LCL_TILE_SZ * MLO_N_IN_TILES_PERSTACK)
#define MLO_IN_LCL_SZ (MLO_IN_LCL_PERSTACK_SZ * MLO_N_STACKS)

#define MLO_WEIGHTS_SZ (MLO_N_OUT_TILES_PERSTACK * MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ)

#define MLO_PVT_ACCUM_DATA_SZ (MLO_N_OUT_TILES * MLO_OUT_TILE_SZ)
#if MLO_DIR_FORWARD == 1
#define MLO_PVT_IN_WIDTH ((MLO_OUT_TILE0 - 1) * MLO_FILTER_STRIDE0 + MLO_FILTER_SIZE0)
#define MLO_PVT_IN_HEIGHT ((MLO_OUT_TILE1 - 1) * MLO_FILTER_STRIDE1 + 1)
#else
#define MLO_PVT_IN_WIDTH \
    ((MLO_OUT_TILE0 + MLO_FILTER_SIZE0 - 1 + MLO_FILTER_STRIDE0 - 1) / MLO_FILTER_STRIDE0)
#define MLO_PVT_IN_HEIGHT ((MLO_OUT_TILE1 + MLO_FILTER_STRIDE1 - 1) / MLO_FILTER_STRIDE1)
#endif

#define MLO_LCL_WEIGHTS 1

#define MLO_PADDING_SHIFT1 (MLO_FILTER_SIZE1 - MLO_FILTER_PAD1 - 1)
#define MLO_PADDING_SHIFT0 (MLO_FILTER_SIZE0 - MLO_FILTER_PAD0 - 1)

#define MLO_PADDING_FIX1 (MLO_FILTER_SIZE1 % MLO_OUT_TILE1)
#define MLO_PADDING_FIX0 (MLO_FILTER_SIZE0 % MLO_OUT_TILE0)

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

#include "math_ops.h"
#include "data_ops.h"

static inline void Conv(uint o_map_base,
                        uint in_stg_off,
                        __private _FLOAT* __restrict pvt_in_stage,
                        __local _FLOAT* __restrict lcl_indata,
                        __private _FLOAT* __restrict pvt_wei_stage,
                        __local _FLOAT* __restrict lcl_wei,
                        __private float* __restrict pvt_accum)
/*__private _FLOAT* __restrict pvt_accum)*/
{
    // convolution

    // over all inputs in stack
    uint in_stg_off1 = in_stg_off;
    for(uint i_c = 0; i_c < MLO_N_IN_TILES_PERSTACK; ++i_c, in_stg_off1 += MLO_IN_LCL_TILE_SZ)
    {
        // preload input
        uint wei_stg_base_off = mad24(o_map_base,
                                      (uint)(MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ),
                                      mul24(i_c, (uint)MLO_FILTER_SZ));
        uint in_stg_off2      = in_stg_off1;
        for(uint j = 0; j < MLO_PVT_IN_HEIGHT - 1; ++j,
#if MLO_DIR_FORWARD == 1
                 in_stg_off2 += MLO_IN_LCL_WIDTH
#else
                 in_stg_off2 += (((j - MLO_PADDING_SHIFT1 + MLO_PADDING_FIX1) % MLO_FILTER_STRIDE1)
                                     ? 0
                                     : MLO_IN_LCL_WIDTH)
#endif
        )
        {
            for(uint i = 0; i < MLO_PVT_IN_WIDTH; ++i)
            {
                pvt_in_stage[j * MLO_PVT_IN_WIDTH + i] = lcl_indata[in_stg_off2 + i];
            }
        }

// over filter rows
#ifdef __AMDGCN__
#if MLO_FILTER_SIZE1 < 6
#pragma unroll
#elif MLO_FILTER_SIZE1 < 9
#pragma unroll 2
#endif
#endif
#if MLO_DIR_FORWARD == 1
        for(uint k = 0; k < MLO_FILTER_SIZE1; ++k, in_stg_off2 += MLO_IN_LCL_WIDTH)
#else
        for(uint k = 0; k < MLO_FILTER_SIZE1; ++k,
                 in_stg_off2 += (((k - MLO_PADDING_SHIFT1 + MLO_PADDING_FIX1) % MLO_FILTER_STRIDE1)
                                     ? 0
                                     : MLO_IN_LCL_WIDTH))
#endif
        {
            uint k_act = 0;
#if MLO_DIR_FORWARD == 1
            k_act = k;
#else
            // load filter in reverse order
            k_act = MLO_FILTER_SIZE1 - 1 - k;
#endif
            // load next input row
            for(uint i_pvt = 0; i_pvt < MLO_PVT_IN_WIDTH; ++i_pvt)
            {
                pvt_in_stage[(MLO_PVT_IN_HEIGHT - 1) * MLO_PVT_IN_WIDTH + i_pvt] =
                    lcl_indata[in_stg_off2 + i_pvt];
            }

            // over all outputs
            for(uint o_c = 0; o_c < MLO_N_OUT_TILES; ++o_c)
            {
                uint wei_stg_off = wei_stg_base_off +
                                   o_c * MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ +
                                   k_act * MLO_FILTER_SIZE0;
                for(uint i = 0; i < MLO_FILTER_SIZE0; ++i)
                {
                    pvt_wei_stage[i] =
                        lcl_wei[wei_stg_off +
                                i]; //(float)o_c/(float)MLO_N_OUT_TILES + (float)(i+k)/9;
                }

                // actual conv

                for(uint j = 0; j < MLO_OUT_TILE1; ++j)
                {
#if MLO_DIR_FORWARD == 0
                    if(((j + k + 1 - MLO_PADDING_SHIFT1 + (MLO_FILTER_SIZE1 % MLO_FILTER_STRIDE1)) %
                        MLO_FILTER_STRIDE1) == 0)
#endif
                        for(uint i = 0; i < MLO_OUT_TILE0; ++i)
                        {
#if MLO_DIR_FORWARD == 1
                            /*_FLOAT sum = (_FLOAT)0;*/
                            float sum = (float)0;
#endif
                            for(uint l = 0; l < MLO_FILTER_SIZE0; ++l)
                            {

                                uint l_act = 0;
#if MLO_DIR_FORWARD == 1
                                l_act = l;

#else
                            // in reverse horizontal and vertical orders
                            l_act = MLO_FILTER_SIZE0 - 1 - l;

#endif

#if MLO_DIR_FORWARD == 1
                                // Directly accumulating to `pvt_accum` here sometimes results in
                                // validation error for half precision.
                                sum +=
                                    (float)(pvt_in_stage[j * MLO_PVT_IN_WIDTH * MLO_FILTER_STRIDE1 +
                                                         i * MLO_FILTER_STRIDE0 + l] *
                                            pvt_wei_stage[l_act]);
#else
                            if(((i + l + 1 - MLO_PADDING_SHIFT0 +
                                 (MLO_FILTER_SIZE0 % MLO_FILTER_STRIDE0)) %
                                MLO_FILTER_STRIDE0) == 0)
                            {
                                pvt_accum[(o_c * MLO_OUT_TILE1 + j) * MLO_OUT_TILE0 + i] +=
                                    pvt_in_stage[(j / MLO_FILTER_STRIDE1) * MLO_PVT_IN_WIDTH +
                                                 (i + l) / MLO_FILTER_STRIDE0] *
                                    pvt_wei_stage[l_act];
                            }
#endif
                            }
#if MLO_DIR_FORWARD == 1
                            // Only needed for forward to fix validation errors on half precision.
                            pvt_accum[(o_c * MLO_OUT_TILE1 + j) * MLO_OUT_TILE0 + i] += sum;
#endif
                        }
                }

            } // for(uint o_c = 0; o_c < MLO_N_OUT_TILES; ++o_c)

            // move data up
            for(uint j = 0; j < MLO_PVT_IN_HEIGHT - 1; ++j)
            {
                for(uint i = 0; i < MLO_PVT_IN_WIDTH; ++i)
                {
                    pvt_in_stage[j * MLO_PVT_IN_WIDTH + i] =
                        pvt_in_stage[(j + 1) * MLO_PVT_IN_WIDTH + i];
                }
            }

        } // for(uint k = 0; k < MLO_FILER_SIZE1; ++k,in_stg_off2+=MLO_IN_LCL_WIDTH)

    } // for(uint i_c = 0; i_c < MLO_N_IN_TILES_PERSTACK; ++i_c, in_stg_off1 +=
      // MLO_IN_LCL_PERSTACK_SZ)
}

#ifndef MLO_CONV_BIAS
#define MLO_CONV_BIAS 0
#endif

#if MIOPEN_USE_FP16 == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT half
#define EPSILON (_FLOAT)0.0001
#endif
#if MIOPEN_USE_FP32 == 1
#define _FLOAT float
#define EPSILON (_FLOAT)0.000001
#endif

#include "activation_functions.h"

// if the BN / Bias ops are not used define appropriate symbols

#if !defined SPATIAL_BN && !defined PERACT_BN
#define NO_BN
#endif

__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2))) __kernel void
MIOpenConvUniBatchNormActiv(
#ifdef MIOPEN_YES_ACTIV
    const _FLOAT alpha,
    const _FLOAT beta,
    const _FLOAT gamma,
#endif
#ifndef NO_BN
    double epsilon,
#endif
    const __global _FLOAT* __restrict in,
    __global _FLOAT* __restrict out,
    const __global _FLOAT* __restrict weights
#if MLO_CONV_BIAS
    ,
    const __global _FLOAT* __restrict conv_bias
#endif
#ifndef NO_BN
    ,
    const __global _FLOAT* __restrict bn_bias,
    const __global _FLOAT* __restrict scale,
    const __global _FLOAT* __restrict estimatedMean,
    const __global _FLOAT* __restrict estimatedVariance
#endif
)
{
    __local _FLOAT lcl_indata[MLO_IN_LCL_SZ];
    __local _FLOAT lcl_wei[MLO_WEIGHTS_SZ];
    __private float pvt_accum[MLO_PVT_ACCUM_DATA_SZ];
    __private _FLOAT pvt_in_stage[MLO_PVT_IN_HEIGHT * MLO_PVT_IN_WIDTH];
    __private _FLOAT pvt_wei_stage[MLO_FILTER_SIZE0];

    uint grp_id0 = get_group_id(0);
#if MLO_N_OUT_TILE_BLOCKS0 & (MLO_N_OUT_TILE_BLOCKS0 - 1)
    uint y_tile_blk = iDiv_legacy(grp_id0, MLO_N_OUT_TILE_BLOCKS0);
    uint x_tile_blk = iMod(grp_id0, y_tile_blk, MLO_N_OUT_TILE_BLOCKS0);
#else
    uint y_tile_blk       = grp_id0 / MLO_N_OUT_TILE_BLOCKS0;
    uint x_tile_blk       = grp_id0 & (MLO_N_OUT_TILE_BLOCKS0 - 1);
#endif
    uint o_pack = get_group_id(1); // block of outputs
    uint b_pack = get_group_id(2); // batch block

    uint lcl_id = get_local_id(0);
#if MLO_ALUTILES_STACK_SZ >= MLO_GRP_SZ
    uint stack        = 0;
    uint alu_stack_id = lcl_id;
#elif MLO_ALUTILES_STACK_SZ & (MLO_ALUTILES_STACK_SZ - 1)
    uint stack            = iDiv_legacy(lcl_id, MLO_ALUTILES_STACK_SZ); // stack
    uint alu_stack_id     = iMod(lcl_id, stack, MLO_ALUTILES_STACK_SZ); // alu index in stack
#else
    uint stack        = lcl_id / MLO_ALUTILES_STACK_SZ;       // stack
    uint alu_stack_id = lcl_id & (MLO_ALUTILES_STACK_SZ - 1); // alu index in stack
#if MLO_ALUTILES_STACK_SZ >= 64
    stack             = uniform(stack);
#endif
#endif
// ALU plane inside stack
#if MLO_ALU_TILE_SZ & (MLO_ALU_TILE_SZ - 1)
    uint alu_out_plane_id = iDiv_legacy(alu_stack_id, MLO_ALU_TILE_SZ); // alu output plane index
    uint alu_out_id       = iMod(
        alu_stack_id, alu_out_plane_id, MLO_ALU_TILE_SZ); // alu index inside an ALU output plane
#else
    uint alu_out_plane_id = alu_stack_id / MLO_ALU_TILE_SZ;             // alu output plane index
    uint alu_out_id = alu_stack_id & (MLO_ALU_TILE_SZ - 1); // alu index inside an ALU output plane
#endif
// pos inside ALU tile
#if MLO_ALU_VTILE0 & (MLO_ALU_VTILE0 - 1)
    uint alu_tl1 = iDiv_legacy(alu_out_id, MLO_ALU_VTILE0);
    uint alu_tl0 = iMod(alu_out_id, alu_tl1, MLO_ALU_VTILE0);
#else
    uint alu_tl1    = alu_out_id / MLO_ALU_VTILE0;
    uint alu_tl0    = alu_out_id & (MLO_ALU_VTILE0 - 1);
#endif

    uint o_map_plane =
        o_pack * MLO_N_OUT_TILES_PERSTACK; // first output maps index per full ALU plane stack
    uint o_map_base = alu_out_plane_id * MLO_N_OUT_TILES; // local output map offset
    uint o_map      = o_map_plane + o_map_base;           // output map index per ALU plane
    uint b_index    = b_pack * MLO_N_STACKS;

#if MLO_LARGE_MAP != 1
#if MLO_N_READ_PROCS >= MLO_GRP_SZ
    uint wave_id     = 0;
    uint wave_lcl_id = lcl_id;
#elif MLO_N_READ_PROCS & (MLO_N_READ_PROCS - 1)
    uint wave_id     = iDiv_legacy(lcl_id, MLO_N_READ_PROCS);
    uint wave_lcl_id = iMod(lcl_id, wave_id, MLO_N_READ_PROCS);
#else
    uint wave_id     = lcl_id / MLO_N_READ_PROCS;
    uint wave_lcl_id = lcl_id & (MLO_N_READ_PROCS - 1);
#if MLO_N_READ_PROCS >= 64
    wave_id          = uniform(wave_id);
#endif
#endif
#endif

#if MLO_DIR_FORWARD == 1
    uint x_grp = x_tile_blk * MLO_IN_TILE0 * MLO_FILTER_STRIDE0;
    uint y_grp = y_tile_blk * MLO_IN_TILE1 * MLO_FILTER_STRIDE1;
#if MLO_LARGE_MAP == 1
    uint x_in_grp = x_grp - MLO_FILTER_PAD0;
    uint y_in_grp = y_grp - MLO_FILTER_PAD1;
#endif
    uint x_in_lcl = alu_tl0 * MLO_OUT_TILE0 * MLO_FILTER_STRIDE0;
    uint y_in_lcl = alu_tl1 * MLO_OUT_TILE1 * MLO_FILTER_STRIDE1;
#else
    uint x_grp      = x_tile_blk * (MLO_IN_TILE0 / MLO_FILTER_STRIDE0);
    uint y_grp      = y_tile_blk * (MLO_IN_TILE1 / MLO_FILTER_STRIDE1);
#if MLO_LARGE_MAP == 1
    uint x_in_grp   = x_grp - (MLO_FILTER_PAD0 / MLO_FILTER_STRIDE0);
    uint y_in_grp   = y_grp - (MLO_FILTER_PAD1 / MLO_FILTER_STRIDE1);
#endif
    uint x_in_lcl   = alu_tl0 * (MLO_OUT_TILE0 / MLO_FILTER_STRIDE0);
    uint y_in_lcl   = alu_tl1 * (MLO_OUT_TILE1 / MLO_FILTER_STRIDE1);
#endif

    // base offset to read data from local input data
    uint in_stg_off = stack * MLO_IN_LCL_PERSTACK_SZ + (y_in_lcl)*MLO_IN_LCL_WIDTH + x_in_lcl;

    uint in_off = b_index * MLO_IN_BATCH_STRIDE;

#if MLO_DIR_FORWARD == 1
    uint wei_off = mul24(o_map_plane, (uint)(MLO_N_INPUTS * MLO_FILTER_SZ));
#else
    uint wei_off    = mul24(o_map_plane, (uint)MLO_FILTER_SZ);
#endif

#if MLO_LARGE_MAP == 0
    for(uint i = lcl_id; i < MLO_IN_LCL_SZ; i += MLO_GRP_SZ)
    {
        lcl_indata[i] = 0;
    }
#endif

    for(uint i = 0; i < MLO_PVT_ACCUM_DATA_SZ; ++i)
    {
        pvt_accum[i] = 0;
    }

    for(uint ic = 0; ic < MLO_N_INPUTS; ic += MLO_N_IN_TILES_PERSTACK,
             in_off += MLO_IN_CHANNEL_STRIDE * MLO_N_IN_TILES_PERSTACK,
             wei_off += MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ
#if MLO_DIR_FORWARD == 0
                                        * MLO_N_OUTPUTS
#endif
    )
    {
        barrier(CLK_LOCAL_MEM_FENCE);

        // small map has been read in full continiously into the lDS buffer within padded rect,
        // padding has been done on initilization.
        // large map calculates padding on the fly and fills it with 0.

#if 1 // all inputs

#if MLO_LARGE_MAP == 1
        uint in_lcl_off1 = 0;
        uint in_off1     = in_off;
        for(uint i_b = 0; i_b < MLO_N_STACKS;
            ++i_b, in_off1 += MLO_IN_BATCH_STRIDE, in_lcl_off1 += MLO_IN_LCL_PERSTACK_SZ)
        {
            bool vis = true;
#if MLO_BATCH_ALIGNED == 0
            vis &= (b_index + i_b < MLO_BATCH_SZ);
#endif

            // over all inputs in stack
            uint in_off2     = in_off1;
            uint in_lcl_off2 = in_lcl_off1;
            for(uint i_c = 0; i_c < MLO_N_IN_TILES_PERSTACK;
                ++i_c, in_off2 += MLO_IN_CHANNEL_STRIDE, in_lcl_off2 += MLO_IN_LCL_TILE_SZ)
            {
#if MLO_INPUTS_ALIGNED == 0
                vis &= (ic + i_c < MLO_N_INPUTS);
#endif

                uint elem_id      = lcl_id;
                uint lcl_p_stride = MLO_GRP_SZ0;
                uint lcl_base     = 0;
                uint lcl_y        = 0;
                uint lcl_x        = 0;
                uint gbl_base     = in_off2;

                readData(elem_id,
                         (MLO_IN_LCL_HEIGHT * MLO_IN_LCL_WIDTH),
                         lcl_p_stride,
                         &lcl_indata[in_lcl_off2],
                         lcl_base,
                         MLO_IN_LCL_HEIGHT,
                         MLO_IN_LCL_WIDTH,
                         MLO_IN_LCL_WIDTH,
                         lcl_y,
                         lcl_x,
                         &in[0],
                         gbl_base,
                         MLO_IN_HEIGHT,
                         MLO_IN_WIDTH,
                         MLO_IN_STRIDE,
                         y_in_grp,
                         x_in_grp,
                         vis,
                         false);
            }
        }
#else
        for(uint i = wave_id; i < MLO_N_IN_TILES_TOTAL; i += MLO_N_PROC_WAVES)
        {
#if MLO_N_IN_TILES_PERSTACK & (MLO_N_IN_TILES_PERSTACK - 1)
            uint i_b = iDiv_legacy(i, MLO_N_IN_TILES_PERSTACK);
            uint i_c = iMod(i, i_b, MLO_N_IN_TILES_PERSTACK);
#else
            uint i_b   = i / MLO_N_IN_TILES_PERSTACK;
            uint i_c   = i & (MLO_N_IN_TILES_PERSTACK - 1);
#endif

            bool vis = true;

#if MLO_BATCH_ALIGNED == 0
            vis &= (b_index + i_b < MLO_BATCH_SZ);
#endif

#if MLO_INPUTS_ALIGNED == 0
            vis &= (ic + i_c < MLO_N_INPUTS);
#endif
            uint in_off2     = in_off + i_b * MLO_IN_BATCH_STRIDE + i_c * MLO_IN_CHANNEL_STRIDE;
            uint in_lcl_off2 = i_b * MLO_IN_LCL_PERSTACK_SZ + i_c * MLO_IN_LCL_TILE_SZ;

            uint elem_id      = wave_lcl_id;
            uint lcl_p_stride = MLO_N_READ_PROCS;
            uint lcl_base     = 0;
#if MLO_DIR_FORWARD == 1
            uint lcl_y        = MLO_FILTER_PAD1;
            uint lcl_x        = MLO_FILTER_PAD0;
#else
            uint lcl_y = (MLO_FILTER_PAD1 / MLO_FILTER_STRIDE0);
            uint lcl_x = (MLO_FILTER_PAD0 / MLO_FILTER_STRIDE1);
#endif
            uint gbl_base     = in_off2;

            readData(elem_id,
                     (MLO_IN_HEIGHT * MLO_IN_WIDTH),
                     lcl_p_stride,
                     &lcl_indata[in_lcl_off2],
                     lcl_base,
                     MLO_IN_HEIGHT,
                     MLO_IN_WIDTH,
                     MLO_IN_LCL_WIDTH,
                     lcl_y,
                     lcl_x,
                     &in[0],
                     gbl_base,
                     MLO_IN_HEIGHT,
                     MLO_IN_WIDTH,
                     MLO_IN_STRIDE,
                     y_grp,
                     x_grp,
                     vis,
                     false);
        }
#endif

        // read inputs and weights
        // put weights into LDS

#if 1 // only weights

        for(uint i = lcl_id; i < MLO_WEIGHTS_SZ; i += MLO_GRP_SZ)
        {
#if MLO_DIR_FORWARD == 1
// here is [tops][bottoms]
#if(MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ) & ((MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ) - 1)
            uint lcl_o = iDiv_legacy(i, (MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ));
            uint gbl_i = iMod(i, lcl_o, (MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ));
#else
            uint lcl_o = i / (MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ);
            uint gbl_i = i & ((MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ) - 1);
#endif
            uint gbl_we_off   = wei_off + lcl_o * MLO_N_INPUTS * MLO_FILTER_SZ + gbl_i;
            bool within_range = gbl_we_off < (MLO_N_OUTPUTS * MLO_N_INPUTS * MLO_FILTER_SZ);

            gbl_we_off = (within_range) ? gbl_we_off : 0;
            _FLOAT wei = weights[gbl_we_off];
            wei        = (within_range) ? wei : 0;
            lcl_wei[i] = wei;
#else
// outputs are botoms(inputs))
// inputs are tops(outputs)
#if(MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ) & ((MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ) - 1)
            uint lcl_o = iDiv_legacy(i, (MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ));
            uint gbl_i = iMod(i, lcl_o, (MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ));
#else
            uint lcl_o = i / (MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ);
            uint gbl_i = i & ((MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ) - 1);
#endif
#if MLO_FILTER_SZ & (MLO_FILTER_SZ - 1)
            uint lcl_c = iDiv_legacy(gbl_i, MLO_FILTER_SZ);
            uint lcl_i = iMod(gbl_i, lcl_c, MLO_FILTER_SZ);
#else
            uint lcl_c = gbl_i / MLO_FILTER_SZ;
            uint lcl_i = gbl_i & (MLO_FILTER_SZ - 1);
#endif

            uint lcl_we_off = mad24(
                mad24(lcl_c, (uint)MLO_N_IN_TILES_PERSTACK, lcl_o), (uint)MLO_FILTER_SZ, lcl_i);
            uint gbl_we_off = mad24(
                mad24(lcl_o, (uint)MLO_N_OUTPUTS, lcl_c), (uint)MLO_FILTER_SZ, wei_off + lcl_i);
            bool within_range   = gbl_we_off < (MLO_N_OUTPUTS * MLO_N_INPUTS * MLO_FILTER_SZ);
            gbl_we_off          = (within_range) ? gbl_we_off : 0;
            _FLOAT wei          = weights[gbl_we_off];
            wei                 = (within_range) ? wei : 0;
            lcl_wei[lcl_we_off] = wei;

#endif
        }

#endif

        // over all batch stacks

#endif // all input

        barrier(CLK_LOCAL_MEM_FENCE);

        // convolution
        Conv(o_map_base, in_stg_off, pvt_in_stage, lcl_indata, pvt_wei_stage, lcl_wei, pvt_accum);

        //		barrier(CLK_LOCAL_MEM_FENCE);
    }
// write results out
#if MLO_DIR_FORWARD == 1
#if MLO_FILTER_STRIDE0 == 1
    uint x_out_grp = x_grp;
#else
    uint x_out_grp = x_tile_blk * MLO_IN_TILE0;
#endif
#if MLO_FILTER_STRIDE1 == 1
    uint y_out_grp = y_grp;
#else
    uint y_out_grp = y_tile_blk * MLO_IN_TILE1;
#endif
#else
    uint x_out_grp  = x_grp * MLO_FILTER_STRIDE0;
    uint y_out_grp  = y_grp * MLO_FILTER_STRIDE1;
#endif
    uint x_out_lcl = alu_tl0 * MLO_OUT_TILE0;
    uint y_out_lcl = alu_tl1 * MLO_OUT_TILE1;

    uint out_off = (b_index + stack) * MLO_OUT_BATCH_STRIDE + o_map * MLO_OUT_CHANNEL_STRIDE +
                   (y_out_grp + y_out_lcl) * MLO_OUT_STRIDE + x_out_grp + x_out_lcl;

    _FLOAT conv_res   = 0.;
    _FLOAT bn_res     = 0.;
    _FLOAT output_res = 0.;
#ifdef MIOPEN_YES_ACTIV
    _FLOAT actv_res;
#endif

// over all local stacks
#if MLO_BATCH_ALIGNED == 0
    if(b_index + stack < MLO_BATCH_SZ)
#endif
    {

        // over all local outputs
        uint out_off1 = out_off;
        for(uint o = 0; o < MLO_N_OUT_TILES; ++o, out_off1 += MLO_OUT_CHANNEL_STRIDE)
        {
#ifndef NO_BN
#ifdef SPATIAL_BN
            uint c_i            = o_map + o;
            _FLOAT pmean        = estimatedMean[c_i];
            _FLOAT pvar         = estimatedVariance[c_i];
            _FLOAT pscale       = scale[c_i];
            _FLOAT pbias        = bn_bias[c_i];
            _FLOAT pinvVariance = rsqrt(fabs(pvar + epsilon));
#endif
#endif

#if MLO_OUTPUTS_ALIGNED == 0
            if(o_map + o < MLO_N_OUTPUTS)
#endif
            {
                // over output tile
                uint out_off2 = out_off1;
#if MLO_OUT_TILE0 == 1
                for(uint j = 0; j < MLO_OUT_TILE1 && y_out_grp + y_out_lcl + j < MLO_OUT_HEIGHT;
                    ++j, out_off2 += MLO_OUT_STRIDE)
                {
                    for(uint i = 0;
                        i < MLO_OUT_TILE0 && x_out_grp + x_out_lcl + i < MLO_OUT_WIDTH &&
                        out_off2 + i < MLO_OUT_BATCH_STRIDE * MLO_BATCH_SZ;
                        ++i)
                    {
                        if(1)
                        {
#else
                for(uint j = 0; j < MLO_OUT_TILE1; ++j, out_off2 += MLO_OUT_STRIDE)
                {
                    if(y_out_grp + y_out_lcl + j < MLO_OUT_HEIGHT)
                        for(uint i = 0; i < MLO_OUT_TILE0; ++i)
                        {
                            if(x_out_grp + x_out_lcl + i < MLO_OUT_WIDTH &&
                               out_off2 + i < MLO_OUT_BATCH_STRIDE * MLO_BATCH_SZ)
                            {
#endif

#ifndef NO_BN
#ifdef PERACT_BN
                            uint chw_i = (out_off2 + i)
#if MLO_BATCH_SZ > 1
                                         % (MLO_OUT_BATCH_STRIDE)
#endif
                                ;
                            _FLOAT pmean        = estimatedMean[chw_i];
                            _FLOAT pvar         = estimatedVariance[chw_i];
                            _FLOAT pscale       = scale[chw_i];
                            _FLOAT pbias        = bn_bias[chw_i];
                            _FLOAT pinvVariance = rsqrt(fabs(pvar + epsilon));
#endif
#endif
                            conv_res =
                                (_FLOAT)pvt_accum[o * MLO_OUT_TILE_SZ + j * MLO_OUT_TILE0 + i]
#if MLO_CONV_BIAS
                                + conv_bias[o_map + o]
#endif
                                ;

#ifdef NO_BN
                            bn_res = conv_res;
#else
                                bn_res     = pscale * (conv_res - pmean) * pinvVariance + pbias;
#endif

#if(MIOPEN_NRN_OP_ID > 0)
#ifdef MIOPEN_YES_ACTIV
                            ActivationFunction(
                                1, &actv_res, (const _FLOAT*)&bn_res, gamma, beta, alpha);
                            output_res = actv_res;

#endif

#else
                                output_res = bn_res;
#endif
                            out[out_off2 + i] = output_res;
                        }
                    }
                }
            }
        }
    }
}
