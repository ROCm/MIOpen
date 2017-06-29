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

#define MLO_FILTER_SZ (MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0)

#define MLO_GRP_SZ0 (MLO_GRP_TILE0 * MLO_GRP_TILE1)
#define MLO_GRP_SZ1 1
#define MLO_GRP_SZ2 1
#define MLO_GRP_SZ (MLO_GRP_SZ0 * MLO_GRP_SZ1 * MLO_GRP_SZ2)
#define MLO_N_PROC_WAVES ((MLO_GRP_SZ + MLO_N_READ_PROCS - 1) / MLO_N_READ_PROCS)
#define MLO_OUT_TILE_SZ (MLO_OUT_PIX_TILE1 * MLO_OUT_PIX_TILE0)
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

#define MLO_ALUTILES_STACK_SZ (MLO_N_ALUTILES_PERSTACK * MLO_ALU_TILE_SZ)
#define MLO_N_IN_TILES_TOTAL (MLO_N_IN_TILES_PERSTACK * MLO_N_STACKS)

#define MLO_N_OUT_TILE_BLOCKS0 ((MLO_OUT_WIDTH + MLO_IN_TILE0 - 1) / MLO_IN_TILE0)
#define MLO_N_OUT_TILE_BLOCKS1 ((MLO_OUT_HEIGHT + MLO_IN_TILE1 - 1) / MLO_IN_TILE1)
#define MLO_N_IN_PACKS (MLO_N_INPUTS / MLO_N_IN_TILES_PERSTACK)

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

#define MLO_IN_LCL_WIDTH               \
    (MLO_IN_TILE0 + MLO_FILTER_SIZE0 - \
     1) // here we use kernel size. it's important when padding == 0
#define MLO_IN_LCL_HEIGHT (MLO_IN_TILE1 + MLO_FILTER_SIZE1 - 1)
#define MLO_IN_LCL_TILE_SZ (MLO_IN_LCL_WIDTH * MLO_IN_LCL_HEIGHT)
#define MLO_IN_LCL_PERSTACK_SZ (MLO_IN_LCL_TILE_SZ * MLO_N_IN_TILES_PERSTACK)
#define MLO_IN_LCL_SZ (MLO_IN_LCL_PERSTACK_SZ * MLO_N_STACKS)

#define MLO_WEIGHTS_SZ (MLO_N_OUT_TILES_PERSTACK * MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ)

#define MLO_PVT_ACCUM_DATA_SZ (MLO_N_OUT_TILES * MLO_OUT_TILE_SZ)
#define MLO_PVT_IN_WIDTH (MLO_FILTER_SIZE0 + MLO_OUT_PIX_TILE0 - 1)
#define MLO_PVT_IN_HEIGHT (MLO_OUT_PIX_TILE1)

#define MLO_LCL_WEIGHTS 1

#if defined(__AMDGCN__)
extern uint __llvm_amdgcn_readfirstlane(uint) __asm("llvm.amdgcn.readfirstlane");
#define uniform(x) __llvm_amdgcn_readfirstlane(x)
#else
#define uniform(x) (x)
#endif

static inline void calculateXYPos(uint linPos, uint width, uint* __restrict x, uint* __restrict y)
{

    (*y) = (uint)((float)linPos * (1.0f / (float)width) + 0.00001f);

    (*x) = linPos - (*y) * width;
}

static inline uint calculateOffset(uint stride, uint x, uint y)
{
    uint ret = y * stride + x;
    return (ret);
}

static inline void readDataElem(uint linPos,
                                __local _FLOAT* lcl_data,
                                uint lcl_base,
                                UNUSED uint lcl_height,
                                uint lcl_width,
                                uint lcl_stride,
                                uint lcl_y,
                                uint lcl_x,
                                const __global _FLOAT* gbl_data,
                                uint gbl_base,
                                uint gbl_height,
                                uint gbl_width,
                                uint gbl_stride,
                                int gbl_y,
                                int gbl_x,
                                bool vis,
                                UNUSED bool debug)
{
    uint x, y;
    calculateXYPos(linPos, lcl_width, &x, &y);
    int g_x       = x + gbl_x;
    int g_y       = y + gbl_y;
    uint gbl_off0 = calculateOffset(gbl_stride, g_x, g_y);
    uint gbl_off  = gbl_off0 + gbl_base;

#if MLO_LARGE_MAP == 1
    uint lcl_off = lcl_base + linPos;
    (void)lcl_stride;
    (void)lcl_x;
    (void)lcl_y;
#else
    uint l_x     = x + lcl_x;
    uint l_y     = y + lcl_y;
    uint lcl_off = lcl_base + mad24(l_y, lcl_stride, l_x);
#endif

#if MLO_LARGE_MAP == 1
    vis &= (g_x >= 0 && g_x < gbl_width && g_y >= 0 && g_y < gbl_height);
#else
    (void)gbl_width;
    (void)gbl_height;
#endif
    gbl_off        = (vis) ? gbl_off : 0;
    _FLOAT gbl_val = gbl_data[gbl_off];
    gbl_val        = (vis) ? gbl_val : 0;

    lcl_data[lcl_off] = gbl_val;
}

static inline void readData(uint lcl_id,
                            uint size,
                            uint lcl_p_stride,
                            __local _FLOAT* lcl_data,
                            uint lcl_base,
                            uint lcl_height,
                            uint lcl_width,
                            uint lcl_stride,
                            uint lcl_y,
                            uint lcl_x,
                            const __global _FLOAT* gbl_data,
                            uint gbl_base,
                            uint gbl_height,
                            uint gbl_width,
                            uint gbl_stride,
                            int gbl_y,
                            int gbl_x,
                            bool vis,
                            bool debug)
{

    for(uint i = lcl_id; i < size; i += lcl_p_stride)
    {
        readDataElem(i,
                     lcl_data,
                     lcl_base,
                     lcl_height,
                     lcl_width,
                     lcl_stride,
                     lcl_y,
                     lcl_x,
                     gbl_data,
                     gbl_base,
                     gbl_height,
                     gbl_width,
                     gbl_stride,
                     gbl_y,
                     gbl_x,
                     vis,
                     debug);
    }
}

static inline void loadData(uint lcl_id,
                            uint lcl_p_stride,
                            __local _FLOAT* lcl_data,
                            uint lcl_off,
                            uint lcl_size,
                            uint lcl_height,
                            uint lcl_width,
                            uint lcl_stride,
                            uint lcl_bot_y,
                            uint lcl_bot_x,
                            const __global _FLOAT* gbl_data,
                            uint gbl_off,
                            uint gbl_size,
                            uint gbl_height,
                            uint glb_width,
                            uint gbl_stride,
                            int gbl_bot_y,
                            int gbl_bot_x,
                            uint buf_block_ind,
                            uint max_n_bufs,
                            uint lcl_n_bufs,
                            bool debug)
{

    for(uint c = 0; c < lcl_n_bufs; ++c, lcl_off += lcl_size, gbl_off += gbl_size)
    {
        bool vis = (buf_block_ind + c < max_n_bufs);
        readData(lcl_id,
                 lcl_size,
                 lcl_p_stride,
                 lcl_data,
                 lcl_off,
                 lcl_height,
                 lcl_width,
                 lcl_stride,
                 lcl_bot_y,
                 lcl_bot_x,
                 gbl_data,
                 gbl_off,
                 gbl_height,
                 glb_width,
                 gbl_stride,
                 gbl_bot_y,
                 gbl_bot_x,
                 vis,
                 (debug));
    }
}

static inline void Conv(uint o_map_base,
                        uint in_stg_off,
                        _FLOAT* __restrict pvt_in_stage,
                        __local _FLOAT* __restrict lcl_indata,
                        _FLOAT* __restrict pvt_wei_stage,
                        __local _FLOAT* __restrict lcl_wei,
                        _FLOAT* __restrict pvt_accum)
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
        uint in_stg_off2 = in_stg_off1;
        for(uint j = 0; j < MLO_PVT_IN_HEIGHT - 1; ++j, in_stg_off2 += MLO_IN_LCL_WIDTH)
        {
            for(uint i = 0; i < MLO_PVT_IN_WIDTH; ++i)
            {
                pvt_in_stage[j * MLO_PVT_IN_WIDTH + i] = lcl_indata[in_stg_off2 + i];
            }
        }

// over filter rows
#ifdef __AMDGCN__
#if(MLO_FILTER_SZ > 9) || (MLO_IN_CHANNEL_STRIDE <= 196) || \
    (MLO_IN_CHANNEL_STRIDE > 784 && MLO_DIR_FORWARD != 1)
#pragma unroll
#else
#pragma unroll 2
#endif
#endif
        for(uint k = 0; k < MLO_FILTER_SIZE1; ++k, in_stg_off2 += MLO_IN_LCL_WIDTH)
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

                for(uint j = 0; j < MLO_OUT_PIX_TILE1; ++j)
                {
                    for(uint i = 0; i < MLO_OUT_PIX_TILE0; ++i)
                    {
                        for(uint l = 0; l < MLO_FILTER_SIZE0; ++l)
                        {

                            uint l_act = 0;
#if MLO_DIR_FORWARD == 1
                            l_act = l;

#else
                            // in reverse horizontal and vertical orders
                            l_act = MLO_FILTER_SIZE0 - 1 - l;

#endif

                            pvt_accum[(o_c * MLO_OUT_PIX_TILE1 + j) * MLO_OUT_PIX_TILE0 + i] +=
                                pvt_in_stage[j * MLO_PVT_IN_WIDTH + i + l] * pvt_wei_stage[l_act];
                        }
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

            //				mem_fence(CLK_LOCAL_MEM_FENCE);

        } // for(uint k = 0; k < MLO_FILER_SIZE1; ++k,in_stg_off2+=MLO_IN_LCL_WIDTH)

    } // for(uint i_c = 0; i_c < MLO_N_IN_TILES_PERSTACK; ++i_c, in_stg_off1 +=
      // MLO_IN_LCL_PERSTACK_SZ)
}

__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2))) __kernel void
MIOpenConvUniC(const __global _FLOAT* __restrict in,
               const __global _FLOAT* __restrict weights,
#if MLO_CONV_BIAS
               const __global _FLOAT* __restrict bias,
#endif
               __global _FLOAT* __restrict out,
               UNUSED _FLOAT padding_val)
{
    __local _FLOAT lcl_indata[MLO_IN_LCL_SZ];
    __local _FLOAT lcl_wei[MLO_WEIGHTS_SZ];
    __private _FLOAT pvt_accum[MLO_PVT_ACCUM_DATA_SZ];
    __private _FLOAT pvt_in_stage[MLO_PVT_IN_HEIGHT * MLO_PVT_IN_WIDTH];
    __private _FLOAT pvt_wei_stage[MLO_FILTER_SIZE0];

    uint grp_id0 = get_group_id(0);
#if MLO_OUT_WIDTH == MLO_IN_TILE0
    uint y_tile_blk = grp_id0;
    uint x_tile_blk = 0;
#else
#if MLO_N_OUT_TILE_BLOCKS0 & (MLO_N_OUT_TILE_BLOCKS0 - 1)
    uint y_tile_blk = (uint)((float)grp_id0 * (1.0f / (float)MLO_N_OUT_TILE_BLOCKS0) + 0.00001f);
    int x_tile_blk  = grp_id0 - mul24(y_tile_blk, (uint)MLO_N_OUT_TILE_BLOCKS0);
#else
    uint y_tile_blk = grp_id0 / MLO_N_OUT_TILE_BLOCKS0;
    int x_tile_blk  = grp_id0 & (MLO_N_OUT_TILE_BLOCKS0 - 1);
#endif
#endif
    uint o_pack = get_group_id(1); // block of outputs
    uint b_pack = get_group_id(2); // batch block

    uint lcl_id = get_local_id(0);
#if MLO_ALUTILES_STACK_SZ & (MLO_ALUTILES_STACK_SZ - 1)
    uint stack = (uint)((float)lcl_id * (1.0f / (float)MLO_ALUTILES_STACK_SZ) + 0.00001f); // stack
    uint alu_stack_id = lcl_id - mul24(stack, (uint)MLO_ALUTILES_STACK_SZ); // alu index in stack
#else
    uint stack      = lcl_id / MLO_ALUTILES_STACK_SZ; // stack
    uint alu_stack_id = lcl_id & (MLO_ALUTILES_STACK_SZ - 1); // alu index in stack
#if MLO_ALUTILES_STACK_SZ >= 64
    stack                 = uniform(stack);
#endif
#endif
// ALU plane inside stack
#if MLO_ALU_TILE_SZ & (MLO_ALU_TILE_SZ - 1)
    uint alu_out_plane_id = (uint)((float)alu_stack_id * (1.0f / (float)MLO_ALU_TILE_SZ) +
                                   0.00001f); // alu output plane index
    uint alu_out_id =
        alu_stack_id -
        mul24(alu_out_plane_id, (uint)MLO_ALU_TILE_SZ); // alu index inside an ALU output plane
#else
    uint alu_out_plane_id = alu_stack_id / MLO_ALU_TILE_SZ;       // alu output plane index
    uint alu_out_id       = alu_stack_id & (MLO_ALU_TILE_SZ - 1); // alu index inside an ALU output plane
#endif
// pos inside ALU tile
#if MLO_ALU_VTILE0 & (MLO_ALU_VTILE0 - 1)
    uint alu_tl1 = (uint)((float)alu_out_id * (1.0f / (float)MLO_ALU_VTILE0) + 0.00001f);
    uint alu_tl0 = alu_out_id - mul24(alu_tl1, (uint)MLO_ALU_VTILE0);
#else
    uint alu_tl1          = alu_out_id / MLO_ALU_VTILE0;
    uint alu_tl0          = alu_out_id & (MLO_ALU_VTILE0 - 1);
#endif

    uint o_map_plane =
        o_pack * MLO_N_OUT_TILES_PERSTACK; // first output maps index per full ALU plane stack
    uint o_map_base = alu_out_plane_id * MLO_N_OUT_TILES; // local output map offset
    uint o_map      = o_map_plane + o_map_base;           // output map index per ALU plane
    uint b_index    = b_pack * MLO_N_STACKS;

#if MLO_LARGE_MAP != 1
#if MLO_GRP_SZ <= MLO_N_READ_PROCS
    uint wave_id     = 0;
    uint wave_lcl_id = lcl_id;
#elif MLO_N_READ_PROCS & (MLO_N_READ_PROCS - 1)
    uint wave_id     = (uint)((float)lcl_id * (1.0f / (float)MLO_N_READ_PROCS) + 0.00001f);
    uint wave_lcl_id = lcl_id - mul24(wave_id, (uint)MLO_N_READ_PROCS);
#else
    uint wave_id     = (uint)((uint)lcl_id / MLO_N_READ_PROCS);
    uint wave_lcl_id = lcl_id & (MLO_N_READ_PROCS - 1);
#if MLO_N_READ_PROCS >= 64
    wave_id          = uniform(wave_id);
#endif
#endif
#endif

    int x_grp  = x_tile_blk * MLO_IN_TILE0;
    uint y_grp = y_tile_blk * MLO_IN_TILE1;

// TO DO: scale
#if MLO_LARGE_MAP == 1
    int x_in_grp = x_grp - MLO_FILTER_PAD0;
    int y_in_grp = y_grp - MLO_FILTER_PAD1;
#endif

    uint x_in_lcl = alu_tl0 * MLO_OUT_PIX_TILE0;
    uint y_in_lcl = alu_tl1 * MLO_OUT_PIX_TILE1;

    // base offset to read data from local input data
    uint in_stg_off = stack * MLO_IN_LCL_PERSTACK_SZ + (y_in_lcl)*MLO_IN_LCL_WIDTH + x_in_lcl;

    uint in_off = b_index * MLO_IN_BATCH_STRIDE;

#if MLO_DIR_FORWARD == 1
    uint wei_off = mul24(o_map_plane, (uint)(MLO_N_INPUTS * MLO_FILTER_SZ));
#else
    uint wei_off          = mul24(o_map_plane, (uint)MLO_FILTER_SZ);
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
                                        *
                                        MLO_N_OUTPUTS
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
                         true);
            }
        }
#else
#ifdef __AMDGCN__
#if(MLO_FILTER_SZ <= 9) && (MLO_IN_CHANNEL_STRIDE <= 784)
#pragma unroll
#endif
#endif
        for(uint i = wave_id; i < MLO_N_IN_TILES_TOTAL; i += MLO_N_PROC_WAVES)
        {
//(MLO_N_STACKS * MLO_N_OUT_TILES_PERSTACK)
#if MLO_N_IN_TILES_PERSTACK & (MLO_N_IN_TILES_PERSTACK - 1)
            uint i_b = (uint)((float)i * (1.0f / (float)MLO_N_IN_TILES_PERSTACK) + 0.00001f);
            uint i_c = i - mul24(i_b, (uint)MLO_N_IN_TILES_PERSTACK);
#else
            uint i_b = (uint)i / MLO_N_IN_TILES_PERSTACK;
            uint i_c = i & (MLO_N_IN_TILES_PERSTACK - 1);
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
            uint lcl_y        = MLO_FILTER_PAD1;
            uint lcl_x        = MLO_FILTER_PAD0;
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
                     true);
        }
#endif

// read inputs and weights
// put weights into LDS

#if 1 // only weights

#if(MLO_WEIGHTS_SZ >= MLO_GRP_SZ) && defined(__AMDGCN__)
#if MLO_WEIGHTS_SZ / MLO_GRP_SZ > 4
#pragma unroll
#else
#pragma unroll(MLO_WEIGHTS_SZ / MLO_GRP_SZ)
#endif
#endif
        for(uint i = lcl_id; i < MLO_WEIGHTS_SZ; i += MLO_GRP_SZ)
        {
#if MLO_DIR_FORWARD == 1
// here is [tops][bottoms]
#if(MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ) & ((MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ) - 1)
            uint lcl_o = (uint)(
                (float)i * (1.0f / (float)(MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ)) + 0.00001f);
            uint gbl_i = i - mul24(lcl_o, (uint)(MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ));
#else
            uint lcl_o = i / (MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ);
            uint gbl_i = i & ((MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ) - 1);
#endif
            if((wei_off + lcl_o * MLO_N_INPUTS * MLO_FILTER_SZ + gbl_i) <
               (MLO_N_OUTPUTS * MLO_N_INPUTS * MLO_FILTER_SZ))
                lcl_wei[i] = weights[wei_off + lcl_o * MLO_N_INPUTS * MLO_FILTER_SZ + gbl_i];
            else
                lcl_wei[i] = weights[0];
#else
// outputs are botoms(inputs))
// inputs are tops(outputs)

#if(MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ) & ((MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ) - 1)
            uint lcl_o = (uint)(
                (float)i * (1.0f / (float)(MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ)) + 0.00001f);
            uint gbl_i = i - mul24(lcl_o, (uint)(MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ));
#else
            uint lcl_o = i / (MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ);
            uint gbl_i = i & ((MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ) - 1);
#endif
#if MLO_FILTER_SZ & (MLO_FILTER_SZ - 1)
            uint lcl_c = (uint)((float)gbl_i * (1.0f / (float)MLO_FILTER_SZ) + 0.00001f);
            uint lcl_i = gbl_i - mul24(lcl_c, (uint)MLO_FILTER_SZ);
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
#if MLO_GRP_SZ > MLO_ACTIVE_ALUS
        if(lcl_id < MLO_ACTIVE_ALUS)
#endif
            Conv(o_map_base,
                 in_stg_off,
                 pvt_in_stage,
                 lcl_indata,
                 pvt_wei_stage,
                 lcl_wei,
                 pvt_accum);

        //		barrier(CLK_LOCAL_MEM_FENCE);
    }

#if MLO_GRP_SZ > MLO_ACTIVE_ALUS
    if(lcl_id >= MLO_ACTIVE_ALUS)
    {
        return;
    }
#endif
    // write results out
    uint x_out_grp = x_grp;
    uint y_out_grp = y_grp;
    uint x_out_lcl = alu_tl0 * MLO_OUT_PIX_TILE0;
    uint y_out_lcl = alu_tl1 * MLO_OUT_PIX_TILE1;

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
            // over output tile
            _FLOAT bias_val = 0;
#if MLO_CONV_BIAS
            bias_val = bias[o_map + o];
#endif
            uint out_off2 = out_off1;
            for(uint j = 0; j < MLO_OUT_PIX_TILE1; ++j, out_off2 += MLO_OUT_STRIDE)
            {
                __global _FLOAT* out_p = &out[out_off2];
                for(uint i = 0; i < MLO_OUT_PIX_TILE0; ++i)
                {
                    if(true
#if 1 // MLO_OUT_ALIGNED == 0
                       &&
                       y_out_lcl + j < MLO_OUT_TILE1 &&
                       y_out_grp + y_out_lcl + j < MLO_OUT_HEIGHT &&
                       x_out_lcl + i < MLO_OUT_TILE0 && x_out_grp + x_out_lcl + i < MLO_OUT_WIDTH
#endif
#if MLO_OUTPUTS_ALIGNED == 0
                       &&
                       o_map + o < MLO_N_OUTPUTS
#endif
                       )
                    {
                        out_p[i] =
                            pvt_accum[o * MLO_OUT_TILE_SZ + j * MLO_OUT_PIX_TILE0 + i] + bias_val;
#if 0
						if ( out_off2 + i == 1 /*y_out_grp + y_out_lcl + j == 2 && x_out_grp + x_out_lcl + i == 0*/)
						{
							printf("K:out: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d    %f %f %f\n",
								MLO_OUT_TILE1,
								MLO_OUT_TILE0,
								grp_id0,
								lcl_id,
								alu_out_plane_id,
								alu_out_id,
								b_index + stack,
								o,
								o_map,
								out_off,
								out_off1,
								out_off2,
								y_out_grp,
								y_out_lcl,
								x_out_grp,
								x_out_lcl,
								j,
								i,
								pvt_accum[o*MLO_OUT_TILE_SZ + j * MLO_OUT_PIX_TILE0 + i],
								 bias_val,
								out[out_off2 + i]
								);
						}
#endif
                    }
                }
            }
        }
    }
}
