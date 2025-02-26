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
#define INLINE

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

#define MLO_N_INPUTS_REMAINDER (MLO_N_INPUTS % (2 * MLO_N_IN_TILES_PERSTACK))

#include "math_ops.h"
#include "data_ops.h"

INLINE
void Conv(uint o_map_base,
          uint in_stg_off,
          __private _FLOAT2* __restrict pvt_in_stage,
          __local _FLOAT2* __restrict lcl_indata,
          __private _FLOAT2* __restrict pvt_wei_stage,
          __local _FLOAT2* __restrict lcl_wei,
          __private _FLOAT2* __restrict pvt_accum)
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
                    pvt_wei_stage[i] = lcl_wei[wei_stg_off + i];
                }

                // actual conv

                for(uint j = 0; j < MLO_OUT_TILE1; ++j)
                {
#if MLO_DIR_FORWARD != 1
                    if(((j + k + 1 - MLO_PADDING_SHIFT1 + (MLO_FILTER_SIZE1 % MLO_FILTER_STRIDE1)) %
                        MLO_FILTER_STRIDE1) == 0)
#endif
                        for(uint i = 0; i < MLO_OUT_TILE0; ++i)
                        {
                            _FLOAT2 sum = (_FLOAT2)(0);
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
                                sum += pvt_in_stage[j * MLO_PVT_IN_WIDTH * MLO_FILTER_STRIDE1 +
                                                    i * MLO_FILTER_STRIDE0 + l] *
                                       pvt_wei_stage[l_act];
#else
                            if(((i + l + 1 - MLO_PADDING_SHIFT0 +
                                 (MLO_FILTER_SIZE0 % MLO_FILTER_STRIDE0)) %
                                MLO_FILTER_STRIDE0) == 0)
                            {
                                sum += pvt_in_stage[(j / MLO_FILTER_STRIDE1) * MLO_PVT_IN_WIDTH +
                                                    (i + l) / MLO_FILTER_STRIDE0] *
                                       pvt_wei_stage[l_act];
                            }
#endif
                            }
                            pvt_accum[(o_c * MLO_OUT_TILE1 + j) * MLO_OUT_TILE0 + i] += sum;
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

INLINE
__kernel void MIOpenConvUni(const __global _FLOAT* __restrict in,
                            const __global _FLOAT* __restrict weights,
#if MLO_CONV_BIAS == 1
                            const __global _FLOAT* __restrict bias,
#endif
                            __global _FLOAT* __restrict out,
                            UNUSED _FLOAT padding_val)
{
    // Local and private arrays are defined as _FLOAT2
    __local _FLOAT2 lcl_indata[MLO_IN_LCL_SZ];
    __local _FLOAT2 lcl_wei[MLO_WEIGHTS_SZ];

    __private _FLOAT2 pvt_accum[MLO_PVT_ACCUM_DATA_SZ] = {MLO_PVT_ACCUM_DATA_SZ * ((_FLOAT2)(0))};
    __private _FLOAT2 pvt_in_stage[MLO_PVT_IN_HEIGHT * MLO_PVT_IN_WIDTH];
    __private _FLOAT2 pvt_wei_stage[MLO_FILTER_SIZE0];

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

    uint in_off    = b_index * MLO_IN_BATCH_STRIDE;
    uint2 in_offv2 = (uint2)(in_off, in_off + MLO_IN_CHANNEL_STRIDE * MLO_N_IN_TILES_PERSTACK);

#if MLO_DIR_FORWARD == 1
    uint wei_off    = mul24(o_map_plane, (uint)(MLO_N_INPUTS * MLO_FILTER_SZ));
    uint2 wei_offv2 = (uint2)(wei_off, wei_off + MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ);
#else
    uint wei_off    = mul24(o_map_plane, (uint)MLO_FILTER_SZ);
    uint2 wei_offv2 =
        (uint2)(wei_off, wei_off + MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ * MLO_N_OUTPUTS);
#endif

#if MLO_LARGE_MAP == 0
    for(uint i = lcl_id; i < MLO_IN_LCL_SZ; i += MLO_GRP_SZ)
    {
        lcl_indata[i] = (_FLOAT2)(0);
    }
#endif

    for(uint i = 0; i < MLO_PVT_ACCUM_DATA_SZ; ++i)
    {
        pvt_accum[i] = (_FLOAT2)(0);
    }

    // Two consecutive inputs are packed into _FLOAT2 vectors.
    for(uint ic = 0; ic < MLO_N_INPUTS; ic += 2 * MLO_N_IN_TILES_PERSTACK,
             in_offv2 += (uint2)(2 * MLO_IN_CHANNEL_STRIDE * MLO_N_IN_TILES_PERSTACK),
             wei_offv2 += (uint2)(2 * MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ
#if MLO_DIR_FORWARD != 1
                                  * MLO_N_OUTPUTS
#endif
                                  ))
    {
        barrier(CLK_LOCAL_MEM_FENCE);

#if MLO_N_INPUTS_REMAINDER <= MLO_N_IN_TILES_PERSTACK
        bool IsLast = (ic + MLO_N_IN_TILES_PERSTACK >= MLO_N_INPUTS);
#endif
        // small map has been read in full continiously into the lDS buffer within padded rect,
        // padding has been done on initilization.
        // large map calculates padding on the fly and fills it with 0.

#if 1 // all inputs

#if MLO_LARGE_MAP == 1
        uint in_lcl_off1 = 0;
        uint2 in_off1    = in_offv2;
        for(uint i_b = 0; i_b < MLO_N_STACKS;
            ++i_b, in_off1 += (uint2)(MLO_IN_BATCH_STRIDE), in_lcl_off1 += MLO_IN_LCL_PERSTACK_SZ)
        {
            bool visX = true;
            bool visY = true;
#if MLO_BATCH_ALIGNED == 0
            visX &= (b_index + i_b < MLO_BATCH_SZ);
            visY &= (b_index + i_b < MLO_BATCH_SZ);
#endif

            // over all inputs in stack
            uint2 in_off2    = in_off1;
            uint in_lcl_off2 = in_lcl_off1;
            for(uint i_c = 0; i_c < MLO_N_IN_TILES_PERSTACK;
                ++i_c, in_off2 += (uint2)(MLO_IN_CHANNEL_STRIDE), in_lcl_off2 += MLO_IN_LCL_TILE_SZ)
            {
#if MLO_INPUTS_ALIGNED == 0
                visX &= (ic + i_c < MLO_N_INPUTS);
                visY &= (ic + MLO_N_IN_TILES_PERSTACK + i_c < MLO_N_INPUTS);
#endif
                readDataVec2(lcl_id,
                             (MLO_IN_LCL_HEIGHT * MLO_IN_LCL_WIDTH),
                             MLO_GRP_SZ0,
                             &lcl_indata[in_lcl_off2],
                             0,
                             MLO_IN_LCL_HEIGHT,
                             MLO_IN_LCL_WIDTH,
                             &in[0],
                             in_off2,
                             MLO_IN_HEIGHT,
                             MLO_IN_WIDTH,
                             MLO_IN_STRIDE,
                             y_in_grp,
                             x_in_grp,
                             visX,
                             visY,
#if MLO_N_INPUTS_REMAINDER <= MLO_N_IN_TILES_PERSTACK
                             IsLast,
#endif
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
            uint i_b = i / MLO_N_IN_TILES_PERSTACK;
            uint i_c = i & (MLO_N_IN_TILES_PERSTACK - 1);
#endif

            bool visX = true;
            bool visY = true;

#if MLO_BATCH_ALIGNED == 0
            visX &= (b_index + i_b < MLO_BATCH_SZ);
            visY &= (b_index + i_b < MLO_BATCH_SZ);
#endif

#if MLO_INPUTS_ALIGNED == 0
            visX &= (ic + i_c < MLO_N_INPUTS);
            visY &= (ic + MLO_N_IN_TILES_PERSTACK + i_c < MLO_N_INPUTS);
#endif
            uint2 in_off2 =
                in_offv2 + (uint2)(i_b * MLO_IN_BATCH_STRIDE + i_c * MLO_IN_CHANNEL_STRIDE);
            uint in_lcl_off2 = i_b * MLO_IN_LCL_PERSTACK_SZ + i_c * MLO_IN_LCL_TILE_SZ;

            readDataVec2(wave_lcl_id,
                         (MLO_IN_HEIGHT * MLO_IN_WIDTH),
                         MLO_N_READ_PROCS,
                         &lcl_indata[in_lcl_off2],
                         0,
                         MLO_IN_HEIGHT,
                         MLO_IN_WIDTH,
                         MLO_IN_LCL_WIDTH,
#if MLO_DIR_FORWARD == 1
                         MLO_FILTER_PAD1,
                         MLO_FILTER_PAD0,
#else
                         (MLO_FILTER_PAD1 / MLO_FILTER_STRIDE0),
                         (MLO_FILTER_PAD0 / MLO_FILTER_STRIDE1),
#endif
                         &in[0],
                         in_off2,
                         MLO_IN_STRIDE,
                         y_grp,
                         x_grp,
                         visX,
                         visY,
#if MLO_N_INPUTS_REMAINDER <= MLO_N_IN_TILES_PERSTACK
                         IsLast,
#endif
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
            if((wei_offv2.x + lcl_o * MLO_N_INPUTS * MLO_FILTER_SZ + gbl_i) <
               (MLO_N_OUTPUTS * MLO_N_INPUTS * MLO_FILTER_SZ))
                lcl_wei[i].x = weights[wei_offv2.x + lcl_o * MLO_N_INPUTS * MLO_FILTER_SZ + gbl_i];
            else
                lcl_wei[i].x = weights[0];
            if((wei_offv2.y + lcl_o * MLO_N_INPUTS * MLO_FILTER_SZ + gbl_i) <
               (MLO_N_OUTPUTS * MLO_N_INPUTS * MLO_FILTER_SZ))
                lcl_wei[i].y = weights[wei_offv2.y + lcl_o * MLO_N_INPUTS * MLO_FILTER_SZ + gbl_i];
            else
                lcl_wei[i].y = weights[0];
#else
            // outputs are botoms(inputs))
            // inputs are tops(outputs)

#if(MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ) & ((MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ) - 1)
            uint lcl_o = iDiv_legacy(i, (MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ));
            uint gbl_i = iMod(i, lcl_o, (MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ));
#else
            uint lcl_o            = i / (MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ);
            uint gbl_i            = i & ((MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ) - 1);
#endif
#if MLO_FILTER_SZ & (MLO_FILTER_SZ - 1)
            uint lcl_c = iDiv_legacy(gbl_i, MLO_FILTER_SZ);
            uint lcl_i = iMod(gbl_i, lcl_c, MLO_FILTER_SZ);
#else
            uint lcl_c            = gbl_i / MLO_FILTER_SZ;
            uint lcl_i            = gbl_i & (MLO_FILTER_SZ - 1);
#endif

            uint lcl_we_off = mad24(
                mad24(lcl_c, (uint)MLO_N_IN_TILES_PERSTACK, lcl_o), (uint)MLO_FILTER_SZ, lcl_i);
            uint2 gbl_we_off =
                (uint2)(mad24(mad24((uint2)(lcl_o), (uint2)(MLO_N_OUTPUTS), (uint2)(lcl_c)),
                              (uint2)(MLO_FILTER_SZ),
                              wei_offv2 + (uint2)(lcl_i)));
#if 0
			bool within_rangeX = gbl_we_off.x < (MLO_N_OUTPUTS*MLO_N_INPUTS*MLO_FILTER_SZ);
			bool within_rangeY = gbl_we_off.y < (MLO_N_OUTPUTS*MLO_N_INPUTS*MLO_FILTER_SZ);
			gbl_we_off.x = (within_rangeX) ? gbl_we_off.x : 0u;
			gbl_we_off.y = (within_rangeY) ? gbl_we_off.y : 0u;
			_FLOAT2 wei = (_FLOAT2)(weights[gbl_we_off.x], weights[gbl_we_off.y]);
			wei.x = (within_rangeX) ? wei.x : (_FLOAT)0;
			wei.y = (within_rangeY) ? wei.y : (_FLOAT)0;
			lcl_wei[lcl_we_off] = wei;
#else
            lcl_wei[lcl_we_off].x = (gbl_we_off.x < (MLO_N_OUTPUTS * MLO_N_INPUTS * MLO_FILTER_SZ))
                                        ? weights[gbl_we_off.x]
                                        : (_FLOAT)0;
            lcl_wei[lcl_we_off].y = (gbl_we_off.y < (MLO_N_OUTPUTS * MLO_N_INPUTS * MLO_FILTER_SZ))
                                        ? weights[gbl_we_off.y]
                                        : (_FLOAT)0;
#endif

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
    uint x_out_grp = x_grp * MLO_FILTER_STRIDE0;
    uint y_out_grp = y_grp * MLO_FILTER_STRIDE1;
#endif
    uint x_out_lcl = alu_tl0 * MLO_OUT_TILE0;
    uint y_out_lcl = alu_tl1 * MLO_OUT_TILE1;

    uint out_off = (b_index + stack) * MLO_OUT_BATCH_STRIDE + o_map * MLO_OUT_CHANNEL_STRIDE +
                   (y_out_grp + y_out_lcl) * MLO_OUT_STRIDE + x_out_grp + x_out_lcl;
// over all local stacks
#if MLO_BATCH_ALIGNED == 0
    if(b_index + stack < MLO_BATCH_SZ)
#endif
    {

        // over all local outputs
        uint out_off1 = out_off;
        for(uint o = 0; o < MLO_N_OUT_TILES; ++o, out_off1 += MLO_OUT_CHANNEL_STRIDE)
        {
#if MLO_OUTPUTS_ALIGNED == 0
            if(o_map + o < MLO_N_OUTPUTS)
#endif
            {
// over output tile
#if MLO_CONV_BIAS == 1
                _FLOAT bias_val = bias[o_map + o];
#endif
                uint out_off2 = out_off1;
#if MLO_OUT_TILE0 == 1
                for(uint j = 0; j < MLO_OUT_TILE1 && y_out_grp + y_out_lcl + j < MLO_OUT_HEIGHT;
                    ++j, out_off2 += MLO_OUT_STRIDE)
                {
                    __global _FLOAT* out_p = &out[out_off2];
                    for(uint i = 0;
                        i < MLO_OUT_TILE0 && x_out_grp + x_out_lcl + i < MLO_OUT_WIDTH &&
                        out_off2 + i < MLO_OUT_BATCH_STRIDE * MLO_BATCH_SZ;
                        ++i)
                    {
#else
                for(uint j = 0; j < MLO_OUT_TILE1; ++j, out_off2 += MLO_OUT_STRIDE)
                {
                    __global _FLOAT* out_p = &out[out_off2];
                    if(y_out_grp + y_out_lcl + j < MLO_OUT_HEIGHT)
                        for(uint i = 0; i < MLO_OUT_TILE0; ++i)
                        {
                            if(x_out_grp + x_out_lcl + i < MLO_OUT_WIDTH &&
                               out_off2 + i < MLO_OUT_BATCH_STRIDE * MLO_BATCH_SZ)
#endif
#if MLO_N_INPUTS <= MLO_N_IN_TILES_PERSTACK
                        out_p[i] = pvt_accum[o * MLO_OUT_TILE_SZ + j * MLO_OUT_TILE0 + i].x
#else
                                out_p[i] =
                                    pvt_accum[o * MLO_OUT_TILE_SZ + j * MLO_OUT_TILE0 + i].x +
                                    pvt_accum[o * MLO_OUT_TILE_SZ + j * MLO_OUT_TILE0 + i].y
#endif
#if MLO_CONV_BIAS == 1
                                   + bias_val
#endif
                            ;
                    }
                }
            }
        }
    }
}
