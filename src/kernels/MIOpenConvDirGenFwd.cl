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
#include "math_ops.h"
#include "data_ops.h"

#ifndef MLO_OUT_ALIGNED
#define MLO_OUT_ALIGNED 0
#endif

// MLO_GRP_SZ0              group size in dim 0
// MLO_GRP_SZ1				group size in dim 1
// MLO_GRP_SZ2				group size in dim 2
// MLO_GRP_SZ               n of wk-item in the group
// MLO_N_IN_CHNLS			total number of input channels
// MLO_LCL_N_IN_CHNLS		n of localy kept input channels
// MLO_IN_WIDTH				input width in NCHW layout
// MLO_IN_HEIGHT			input height stride in NCHW layout
// MLO_IN_STRIDE			input stride in NCHW layout
// MLO_IN_CHNL_STRIDE       input channel stride in NCHW layout
// MLO_IN_BATCH_STRIDE      input batch stride in NCHW layout
// MLO_BATCH_SZ		        batch szie
// MLO_FLTR_SZ0             filter 0 dim size
// MLO_FLTR_PAD_SZ0				filter 0 dim pad
// MLO_FLTR_STRIDE0			filter 0 dim stride
// MLO_FLTR_SZ1             filter 1 dim size
// MLO_FLTR_PAD_SZ1				filter 1 dim pad
// MLO_FLTR_STRIDE1			filter 1 dim stride
// MLO_IN_CHNL_LOOP         main input channel loop
// MLO_OUT_WIDTH			output width in NCHW layout
// MLO_OUT_HEIGHT			output height stride in NCHW layout
// MLO_OUT_STRIDE			output stride in NCHW layout
// MLO_N_OUT_PIX_SZ0        n output pixel per wk item in 0 dim
// MLO_N_OUT_PIX_SZ1		n output pexels per wk item in 1 dim
// MLO_N_IN_PIX_SZ0        n input pixels per wk item in 0 dim
// MLO_N_IN_PIX_SZ1		n input pexels per wk item in 1 dim
// MLO_N_STACKS           n of separate data stacks
// MLO_N_PROCS1           n of processors per stack 1 dim
// MLO_N_PROCS0           n of processors per stack 0 dim
// MLO_IN_SZ0			horizontal read dim 0
// MLO_IN_SZ1			vertical read dim 1

// inputs are taken from different stacks of batches - to use the same filters
#define MLO_LCL_IMG_WIDTH (MLO_IN_SZ0 + MLO_FLTR_SZ0 - 1)
#define MLO_LCL_IMG_HEIGHT (MLO_IN_SZ1 + MLO_FLTR_SZ1 - 1)
#define MLO_LCL_IMG_SIZE (MLO_LCL_IMG_WIDTH * MLO_LCL_IMG_HEIGHT)
#define MLO_PVT_COL_STG_HEIGHT ((MLO_N_OUT_PIX_SZ1 - 1) * MLO_FLTR_STRIDE1 + 1)
#define MLO_PVT_COL_STG_WIDTH ((MLO_N_OUT_PIX_SZ0 - 1) * MLO_FLTR_STRIDE0 + MLO_FLTR_SZ0)
#define MLO_LCL_WEI_SIZE (MLO_LCL_N_OUT_CHNLS * MLO_FLTR_SZ0 * MLO_FLTR_SZ1)

#define MLO_PVT_OUT_DATA_HEIGHT MLO_N_OUT_PIX_SZ1* MLO_LCL_N_OUT_CHNLS
#define MLO_PVT_OUT_DATA_SZ MLO_PVT_OUT_DATA_HEIGHT* MLO_N_OUT_PIX_SZ0

__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2))) __kernel void
MIOpenCDFGen(const __global _FLOAT* __restrict bot,
             const __global _FLOAT* __restrict weights,
#if MLO_CONV_BIAS == 1
             const __global _FLOAT* __restrict bias,
#endif
             __global _FLOAT* __restrict top,
             _FLOAT padding_val
             //      int in_main_loop
)
{
    __local _FLOAT lcl_img[MLO_LCL_IMG_SIZE * MLO_LCL_N_IN_CHNLS];
    __local _FLOAT lcl_wei[MLO_LCL_WEI_SIZE];

    _FLOAT pvt_bot_dat[MLO_PVT_COL_STG_HEIGHT * MLO_PVT_COL_STG_WIDTH];
    _FLOAT_ACCUM pvt_top_dat[MLO_PVT_OUT_DATA_SZ];
    _FLOAT pvt_wei_dat[MLO_FLTR_SZ0];

    uint x_out_grp = get_group_id(0) * MLO_N_PROCS0 * MLO_N_OUT_PIX_SZ0;
    uint y_out_grp = get_group_id(1) * MLO_N_PROCS1 * MLO_N_OUT_PIX_SZ1;
    uint y_in_grp  = y_out_grp * MLO_FLTR_STRIDE1;
    uint x_in_grp  = x_out_grp * MLO_FLTR_STRIDE0;

    uint lcl_id   = mad24(get_local_id(1), (uint)MLO_GRP_SZ0, get_local_id(0));
    uint lcl_proc = lcl_id / (MLO_N_PROCS0 * MLO_N_PROCS1); // input id from diff stack
#if(MLO_N_PROCS0 * MLO_N_PROCS1) & (MLO_N_PROCS0 * MLO_N_PROCS1 - 1)
    uint lcl_in_proc_id = -mad24((int)lcl_proc,
                                 (MLO_N_PROCS0 * MLO_N_PROCS1),
                                 -(int)lcl_id); // wk item id for the input to make a coalesed read
#else
    uint lcl_in_proc_id = lcl_id & (MLO_N_PROCS0 * MLO_N_PROCS1 -
                                    1); // wk item id for the input to make a coalesed read
#endif
    uint lcl_proc_id1 = lcl_in_proc_id / MLO_N_PROCS0; //
#if MLO_N_PROCS0 & (MLO_N_PROCS0 - 1)
    uint lcl_proc_id0 = -mad24((int)lcl_proc_id1, MLO_N_PROCS0, -(int)lcl_in_proc_id); //
#else
    uint lcl_proc_id0   = lcl_in_proc_id & (MLO_N_PROCS0 - 1); //
#endif
    uint x_out_lcl = mul24(lcl_proc_id0, (uint)MLO_N_OUT_PIX_SZ0);
    uint y_out_lcl = mul24(lcl_proc_id1, (uint)MLO_N_OUT_PIX_SZ1);
    uint x_out     = x_out_grp + x_out_lcl;
    uint y_out     = y_out_grp + y_out_lcl;

    uint y_in_lcl = y_out * MLO_FLTR_STRIDE1 - y_in_grp;
    uint x_in_lcl = x_out * MLO_FLTR_STRIDE0 - x_in_grp;

    uint ob   = get_global_id(2);
    uint o_id = ob / MLO_N_STACKS; // block of outputs
#if MLO_N_STACKS & (MLO_N_STACKS - 1)
    uint b_id = -mad24((int)o_id, MLO_N_STACKS, -(int)ob); // block of batchs
#else
    uint b_id           = ob & (MLO_N_STACKS - 1);             // block of batchs
#endif
    // my batch
    uint b = b_id * MLO_LCL_N_IN_CHNLS + lcl_proc;

    uint in_off = b * MLO_IN_BATCH_STRIDE;
    uint wei_off =
        mul24(o_id, (uint)(MLO_LCL_N_OUT_CHNLS * MLO_N_IN_CHNLS * MLO_FLTR_SZ0 * MLO_FLTR_SZ1));
    uint lcl_off =
        mul24(lcl_proc, (uint)MLO_LCL_IMG_SIZE) + y_in_lcl * MLO_LCL_IMG_WIDTH + x_in_lcl;

#if MLO_BIG == 0
    for(uint i = lcl_id; i < MLO_LCL_IMG_SIZE * MLO_LCL_N_IN_CHNLS; i += MLO_GRP_SZ)
    {
        lcl_img[i] = 0;
    }
#endif
    for(uint i = 0; i < MLO_PVT_OUT_DATA_SZ; ++i)
    {
        pvt_top_dat[i] = (_FLOAT_ACCUM)0;
    }

    for(uint c = 0; c < MLO_N_IN_CHNLS;
        c++, in_off += MLO_IN_CHNL_STRIDE, wei_off += MLO_FLTR_SZ0 * MLO_FLTR_SZ1)
    {

        barrier(CLK_LOCAL_MEM_FENCE);
        // put weights for all our outputs for this input into LDS
        for(uint i = lcl_id; i < MLO_LCL_N_OUT_CHNLS * MLO_FLTR_SZ0 * MLO_FLTR_SZ1; i += MLO_GRP_SZ)
        {
#if(MLO_FLTR_SZ0 * MLO_FLTR_SZ1) & (MLO_FLTR_SZ0 * MLO_FLTR_SZ1 - 1)
            uint lcl_o   = (uint)((float)i / (MLO_FLTR_SZ0 * MLO_FLTR_SZ1) + 0.00001f);
            uint lcl_o_i = i - lcl_o * (MLO_FLTR_SZ0 * MLO_FLTR_SZ1);
#else
            uint lcl_o   = i / (MLO_FLTR_SZ0 * MLO_FLTR_SZ1);
            uint lcl_o_i = i & (MLO_FLTR_SZ0 * MLO_FLTR_SZ1 - 1);
#endif

            lcl_wei[i] =
                weights[wei_off + lcl_o * MLO_N_IN_CHNLS * MLO_FLTR_SZ0 * MLO_FLTR_SZ1 + lcl_o_i];
        }

        readDataTile(lcl_img,
                     bot,
                     y_in_grp,
                     x_in_grp,
                     MLO_IN_STRIDE,
                     (in_off + y_in_grp * MLO_IN_STRIDE + x_in_grp),
                     MLO_LCL_IMG_WIDTH,
                     0,
                     MLO_IN_HEIGHT,
                     MLO_IN_WIDTH,
                     MLO_LCL_IMG_HEIGHT,
                     MLO_LCL_IMG_WIDTH,
                     lcl_proc_id1,
                     lcl_proc_id0,
                     MLO_N_PROCS1,
                     MLO_N_PROCS0,
                     MLO_FLTR_PAD_SZ1,
                     MLO_FLTR_PAD_SZ0,
                     padding_val);

        barrier(CLK_LOCAL_MEM_FENCE);

        // get first MLO_N_OUT_PIX_SZ1 lines

        uint j = 0;

        uint lcl_off2 = lcl_off;

        for(; j < MLO_PVT_COL_STG_HEIGHT - 1; ++j, lcl_off2 += MLO_LCL_IMG_WIDTH)
        {

            // read input data

            for(uint i = 0; i < MLO_PVT_COL_STG_WIDTH; ++i)
            {
                pvt_bot_dat[j * MLO_PVT_COL_STG_WIDTH + i] = lcl_img[lcl_off2 + i];
            }
        }

        // convolve with the filter
        uint lcl_wei_off = 0;
        for(uint k = 0; k < MLO_FLTR_SZ1;
            ++k, ++j, lcl_off2 += MLO_LCL_IMG_WIDTH, lcl_wei_off += MLO_FLTR_SZ0)
        {

            // read input data
            for(uint i = 0; i < MLO_PVT_COL_STG_WIDTH; ++i)
            {
                pvt_bot_dat[(MLO_PVT_COL_STG_HEIGHT - 1) * MLO_PVT_COL_STG_WIDTH + i] =
                    lcl_img[lcl_off2 + i];
            }

            // convolve over all outputs
            uint lcl_wei_off2 = lcl_wei_off;

            for(uint o = 0; o < MLO_LCL_N_OUT_CHNLS;
                ++o, lcl_wei_off2 += MLO_FLTR_SZ1 * MLO_FLTR_SZ0)
            {
                // read weights
                for(uint w = 0; w < MLO_FLTR_SZ0; ++w)
                {
                    pvt_wei_dat[w] = lcl_wei[lcl_wei_off2 + w];
                }

                // convolve over the tile
                for(uint pj = 0; pj < MLO_N_OUT_PIX_SZ1; ++pj)
                {
                    for(uint pi = 0; pi < MLO_N_OUT_PIX_SZ0; ++pi)
                    {

                        _FLOAT_ACCUM sum = CVT_FLOAT2ACCUM(0);
                        for(uint m = 0; m < MLO_FLTR_SZ0; ++m)
                        {
                            sum += CVT_FLOAT2ACCUM(
                                       pvt_bot_dat[pj * MLO_FLTR_STRIDE1 * MLO_PVT_COL_STG_WIDTH +
                                                   pi * MLO_FLTR_STRIDE0 + m]) *
                                   CVT_FLOAT2ACCUM(pvt_wei_dat[m]);

#if 0
                                                                if (y_out + pj == 0 && x_out + pi == 14)
                                                                {

                                                                        printf("K:cnv: %f %f %f\n",
                                                                                pvt_top_dat[(o * MLO_N_OUT_PIX_SZ1 + pj) * MLO_N_OUT_PIX_SZ0 + pi],
                                                                                pvt_bot_dat[pj * MLO_FLTR_STRIDE1 * MLO_PVT_COL_STG_WIDTH + pi* MLO_FLTR_STRIDE0 + m],
                                                                                pvt_wei_dat[m]
                                                                                );
                                                                }

#endif
                        }
                        pvt_top_dat[(o * MLO_N_OUT_PIX_SZ1 + pj) * MLO_N_OUT_PIX_SZ0 + pi] += sum;
                    }
                }
            }

            // move up

            for(uint jj = 0; jj < MLO_PVT_COL_STG_HEIGHT - 1; ++jj)
            {

                for(uint ii = 0; ii < MLO_PVT_COL_STG_WIDTH; ++ii)
                {
                    pvt_bot_dat[jj * MLO_PVT_COL_STG_WIDTH + ii] =
                        pvt_bot_dat[(jj + 1) * MLO_PVT_COL_STG_WIDTH + ii];
                }
            }
        }
    }

    // write into all output feature maps
    uint top_off = b * MLO_OUT_BATCH_STRIDE + o_id * MLO_LCL_N_OUT_CHNLS * MLO_OUT_CHNL_STRIDE +
                   y_out * MLO_OUT_STRIDE + x_out;

    for(uint o = 0; o < MLO_LCL_N_OUT_CHNLS; ++o, top_off += MLO_OUT_CHNL_STRIDE)
    {

#if MLO_OUT_ALIGNED == 0
        if(o_id * MLO_LCL_N_OUT_CHNLS + o < MLO_N_OUT_CHNLS)
#endif
        {
#if MLO_CONV_BIAS == 1

            bias_val = bias[o_id * MLO_LCL_N_OUT_CHNLS + o];
#endif

            uint top_off2 = top_off;

            for(uint j = 0; j < MLO_N_OUT_PIX_SZ1; ++j, top_off2 += MLO_OUT_STRIDE)
            {
                if(y_out + j < MLO_OUT_HEIGHT)
                {
                    for(uint i = 0; i < MLO_N_OUT_PIX_SZ0; ++i)
                    {
#if MLO_ALIGNED == 0
                        if(x_out + i < MLO_OUT_WIDTH)
#endif

#if MLO_CONV_BIAS == 1
                            top[top_off2 + i] = CVT_ACCUM2FLOAT(
                                pvt_top_dat[(o * MLO_N_OUT_PIX_SZ1 + j) * MLO_N_OUT_PIX_SZ0 + i] +
                                CVT_FLOAT2ACCUM(bias_val));
#else
                        top[top_off2 + i] = CVT_ACCUM2FLOAT(
                            pvt_top_dat[(o * MLO_N_OUT_PIX_SZ1 + j) * MLO_N_OUT_PIX_SZ0 + i]);
#endif

#if 0
                                                if (y_out + j == 0 && x_out + i == 14)
                                                {

                                                        printf("K:out: %d %f %f\n",
                                                                top_off2 + i,
                                                                pvt_top_dat[(o * MLO_N_OUT_PIX_SZ1 + j) * MLO_N_OUT_PIX_SZ0 + i],
                                                                top[top_off2 + i]
                                                                );
                                                }

#endif
                    }
                }
            }
        }
    }
}

__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2))) __kernel void
MIOpenCDFGen4(const __global _FLOAT* __restrict bot,
              const __global _FLOAT* __restrict weights,
#if MLO_CONV_BIAS == 1
              const __global _FLOAT* __restrict bias,
#endif
              __global _FLOAT* __restrict top,
              _FLOAT padding_val
              //         int in_main_loop
)
{
    __local _FLOAT lcl_img[MLO_LCL_IMG_SIZE * MLO_LCL_N_IN_CHNLS * MLO_IN_STACKS];
    __local _FLOAT lcl_wei[MLO_LCL_WEI_SIZE * MLO_OUT_STACKS];

    _FLOAT pvt_bot_dat[MLO_PVT_COL_STG_HEIGHT * MLO_PVT_COL_STG_WIDTH];
    _FLOAT_ACCUM pvt_top_dat[MLO_PVT_OUT_DATA_SZ];
    _FLOAT pvt_wei_dat[MLO_FLTR_SZ0];

    uint x_out_grp = get_group_id(0) * MLO_N_PROCS0 * MLO_N_OUT_PIX_SZ0;
    uint y_out_grp = get_group_id(1) * MLO_N_PROCS1 * MLO_N_OUT_PIX_SZ1;
    uint y_in_grp  = y_out_grp * MLO_FLTR_STRIDE1;
    uint x_in_grp  = x_out_grp * MLO_FLTR_STRIDE0;

    uint lcl_id       = mad24(get_local_id(1), (uint)MLO_GRP_SZ0, get_local_id(0));
    uint proc_tile1   = (get_local_id(1) >> MLO_LG2N_PROC_TILE1);
    uint lcl_proc_id1 = get_local_id(1) - (proc_tile1 << MLO_LG2N_PROC_TILE1); //
    uint lcl_proc_id0 = get_local_id(0);                                       //
    uint x_out_lcl    = mul24(lcl_proc_id0, (uint)MLO_N_OUT_PIX_SZ0);
    uint y_out_lcl    = mul24(lcl_proc_id1, (uint)MLO_N_OUT_PIX_SZ1);
    uint x_out        = x_out_grp + x_out_lcl;
    uint y_out        = y_out_grp + y_out_lcl;

    uint y_in_lcl = y_out * MLO_FLTR_STRIDE1 - y_in_grp;
    uint x_in_lcl = x_out * MLO_FLTR_STRIDE0 - x_in_grp;

    uint ob   = get_global_id(2);
    uint o_id = ob / MLO_N_STACKS;                    // block of outputs
    uint b_id = ob - mul24(o_id, (uint)MLO_N_STACKS); // block of batchs
    // my batch
    uint b      = b_id * MLO_LCL_N_IN_CHNLS * MLO_IN_STACKS + proc_tile1;
    uint in_off = b * MLO_IN_BATCH_STRIDE;
    uint wei_off =
        mul24(o_id, (uint)(MLO_LCL_N_OUT_CHNLS * MLO_N_IN_CHNLS * MLO_FLTR_SZ0 * MLO_FLTR_SZ1));
    uint lcl_off = MLO_LCL_IMG_SIZE * MLO_LCL_N_IN_CHNLS * proc_tile1 +
                   y_in_lcl * MLO_LCL_IMG_WIDTH + x_in_lcl;

#if MLO_BIG == 0
    for(uint i = lcl_id; i < MLO_LCL_IMG_SIZE * MLO_LCL_N_IN_CHNLS; i += MLO_GRP_SZ)
    {
        lcl_img[i] = 0;
    }
#endif
    for(uint i = 0; i < MLO_PVT_OUT_DATA_SZ; ++i)
    {
        pvt_top_dat[i] = (_FLOAT_ACCUM)0;
    }

    for(uint c = 0; c < MLO_N_IN_CHNLS;
        c++, in_off += MLO_IN_CHNL_STRIDE, wei_off += MLO_FLTR_SZ0 * MLO_FLTR_SZ1)
    {

        barrier(CLK_LOCAL_MEM_FENCE);
        // put weights for all our outputs for this input into LDS
        for(uint i = lcl_id; i < MLO_LCL_N_OUT_CHNLS * MLO_FLTR_SZ0 * MLO_FLTR_SZ1; i += MLO_GRP_SZ)
        {
#if(MLO_FLTR_SZ0 * MLO_FLTR_SZ1) & (MLO_FLTR_SZ0 * MLO_FLTR_SZ1 - 1)
            uint lcl_o   = (uint)((float)i / (MLO_FLTR_SZ0 * MLO_FLTR_SZ1) + 0.00001f);
            uint lcl_o_i = i - lcl_o * (MLO_FLTR_SZ0 * MLO_FLTR_SZ1);
#else
            uint lcl_o   = i / (MLO_FLTR_SZ0 * MLO_FLTR_SZ1);
            uint lcl_o_i = i & (MLO_FLTR_SZ0 * MLO_FLTR_SZ1 - 1);
#endif

            lcl_wei[i] =
                weights[wei_off + lcl_o * MLO_N_IN_CHNLS * MLO_FLTR_SZ0 * MLO_FLTR_SZ1 + lcl_o_i];
        }

        readDataTile(&lcl_img[MLO_LCL_IMG_SIZE * MLO_LCL_N_IN_CHNLS * proc_tile1],
                     bot,
                     y_in_grp,
                     x_in_grp,
                     MLO_IN_STRIDE,
                     (in_off + y_in_grp * MLO_IN_STRIDE + x_in_grp),
                     MLO_LCL_IMG_WIDTH,
                     0,
                     MLO_IN_HEIGHT,
                     MLO_IN_WIDTH,
                     MLO_LCL_IMG_HEIGHT,
                     MLO_LCL_IMG_WIDTH,
                     lcl_proc_id1,
                     lcl_proc_id0,
                     MLO_N_PROCS1,
                     MLO_N_PROCS0,
                     MLO_FLTR_PAD_SZ1,
                     MLO_FLTR_PAD_SZ0,
                     padding_val);

        barrier(CLK_LOCAL_MEM_FENCE);

        // get first MLO_N_OUT_PIX_SZ1 lines

        uint j = 0;

        uint lcl_off2 = lcl_off;

        for(; j < MLO_PVT_COL_STG_HEIGHT - 1; ++j, lcl_off2 += MLO_LCL_IMG_WIDTH)
        {

            // read input data

            for(uint i = 0; i < MLO_PVT_COL_STG_WIDTH; ++i)
            {
                pvt_bot_dat[j * MLO_PVT_COL_STG_WIDTH + i] = lcl_img[lcl_off2 + i];
            }
        }

        // convolve with the filter
        uint lcl_wei_off = 0;
        for(uint k = 0; k < MLO_FLTR_SZ1;
            ++k, ++j, lcl_off2 += MLO_LCL_IMG_WIDTH, lcl_wei_off += MLO_FLTR_SZ0)
        {

            // read input data
            for(uint i = 0; i < MLO_PVT_COL_STG_WIDTH; ++i)
            {
                pvt_bot_dat[(MLO_PVT_COL_STG_HEIGHT - 1) * MLO_PVT_COL_STG_WIDTH + i] =
                    lcl_img[lcl_off2 + i];
            }

            // convolve over all outputs
            uint lcl_wei_off2 = lcl_wei_off;

            for(uint o = 0; o < MLO_LCL_N_OUT_CHNLS;
                ++o, lcl_wei_off2 += MLO_FLTR_SZ1 * MLO_FLTR_SZ0)
            {
                // read weights
                for(uint w = 0; w < MLO_FLTR_SZ0; ++w)
                {
                    pvt_wei_dat[w] = lcl_wei[lcl_wei_off2 + w];
                }

                // convolve over the tile
                for(uint pj = 0; pj < MLO_N_OUT_PIX_SZ1; ++pj)
                {
                    for(uint pi = 0; pi < MLO_N_OUT_PIX_SZ0; ++pi)
                    {

                        _FLOAT_ACCUM sum = CVT_FLOAT2ACCUM(0);
                        for(uint m = 0; m < MLO_FLTR_SZ0; ++m)
                        {
                            sum += CVT_FLOAT2ACCUM(
                                       pvt_bot_dat[pj * MLO_FLTR_STRIDE1 * MLO_PVT_COL_STG_WIDTH +
                                                   pi * MLO_FLTR_STRIDE0 + m]) *
                                   CVT_FLOAT2ACCUM(pvt_wei_dat[m]);

#if 0
                                                        if (get_group_id(0) == 0 && get_group_id(1) == 0 && get_group_id(2) == 0 && proc_tile1 == 0 && o == 0 && y_out + pj == 2 && x_out + pi == 0)
                                                        {

                                                                printf("K:cnv: %d %d %d %d %f %f %f\n",
                                                                        get_local_id(1),
                                                                        get_local_id(0),
                                                                        y_out,
                                                                        x_out,
                                                                        pvt_top_dat[(o * MLO_N_OUT_PIX_SZ1 + pj) * MLO_N_OUT_PIX_SZ0 + pi],
                                                                        pvt_bot_dat[pj * MLO_FLTR_STRIDE1 * MLO_PVT_COL_STG_WIDTH + pi* MLO_FLTR_STRIDE0 + m],
                                                                        pvt_wei_dat[m]
                                                                        );
                                                        }

#endif
                        }
                        pvt_top_dat[(o * MLO_N_OUT_PIX_SZ1 + pj) * MLO_N_OUT_PIX_SZ0 + pi] += sum;
                    }
                }
            }

            // move up

            for(uint jj = 0; jj < MLO_PVT_COL_STG_HEIGHT - 1; ++jj)
            {

                for(uint ii = 0; ii < MLO_PVT_COL_STG_WIDTH; ++ii)
                {
                    pvt_bot_dat[jj * MLO_PVT_COL_STG_WIDTH + ii] =
                        pvt_bot_dat[(jj + 1) * MLO_PVT_COL_STG_WIDTH + ii];
                }
            }
        }
    }

    // write into all output feature maps
    uint top_off = b * MLO_OUT_BATCH_STRIDE + o_id * MLO_LCL_N_OUT_CHNLS * MLO_OUT_CHNL_STRIDE +
                   y_out * MLO_OUT_STRIDE + x_out;

    for(uint o = 0; o < MLO_LCL_N_OUT_CHNLS; ++o, top_off += MLO_OUT_CHNL_STRIDE)
    {

#if MLO_OUT_ALIGNED == 0
        if(o_id * MLO_LCL_N_OUT_CHNLS + o < MLO_N_OUT_CHNLS)
#endif
        {
#if MLO_CONV_BIAS == 1

            bias_val = bias[o_id * MLO_LCL_N_OUT_CHNLS + o];
#endif

            uint top_off2 = top_off;

            for(uint j = 0; j < MLO_N_OUT_PIX_SZ1; ++j, top_off2 += MLO_OUT_STRIDE)
            {
                if(y_out + j < MLO_OUT_HEIGHT)
                {
                    for(uint i = 0; i < MLO_N_OUT_PIX_SZ0; ++i)
                    {
#if MLO_ALIGNED == 0
                        if(x_out + i < MLO_OUT_WIDTH)
#endif
#if MLO_CONV_BIAS == 1
                            top[top_off2 + i] = CVT_ACCUM2FLOAT(
                                pvt_top_dat[(o * MLO_N_OUT_PIX_SZ1 + j) * MLO_N_OUT_PIX_SZ0 + i] +
                                CVT_FLOAT2ACCUM(bias_val));
#else
                        top[top_off2 + i] = CVT_ACCUM2FLOAT(
                            pvt_top_dat[(o * MLO_N_OUT_PIX_SZ1 + j) * MLO_N_OUT_PIX_SZ0 + i]);
#endif

#if 0
                                                if (get_group_id(0) == 0 && get_group_id(1) == 0 && get_group_id(2) == 0 && proc_tile1 == 0 && o == 0 && y_out + j == 2 && x_out + i == 0)
                                                {

                                                        printf("K:out: %d %f %f\n",
                                                                top_off2 + i,
                                                                pvt_top_dat[(o * MLO_N_OUT_PIX_SZ1 + j) * MLO_N_OUT_PIX_SZ0 + i],
                                                                top[top_off2 + i]
                                                                );
                                                }

#endif
                    }
                }
            }
        }
    }
}

__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2))) __kernel void
aDNNConv_img2col(const __global _FLOAT* __restrict img,
                 __global _FLOAT* __restrict col,
                 _FLOAT padding_val)
{
    __local _FLOAT lcl_img[MLO_LCL_IMG_SIZE * MLO_LCL_N_IN_CHNLS];
    _FLOAT col_stg[MLO_PVT_COL_STG_HEIGHT * MLO_PVT_COL_STG_WIDTH];
    //        _FLOAT col_out[MLO_PVT_COL_OUT_HEIGHT * MLO_PVT_COL_OUT_WIDTH];

    uint x_out_grp = get_group_id(0) * MLO_N_PROCS0 * MLO_N_OUT_PIX_SZ0;
    uint y_out_grp = get_group_id(1) * MLO_N_PROCS1 * MLO_N_OUT_PIX_SZ1;
    uint y_in_grp  = y_out_grp * MLO_FLTR_STRIDE1;
    uint x_in_grp  = x_out_grp * MLO_FLTR_STRIDE0;

    uint lcl_id = mad24(get_local_id(1), (uint)MLO_GRP_SZ0, get_local_id(0));
#if(MLO_N_PROCS0 * MLO_N_PROCS1) & (MLO_N_PROCS0 * MLO_N_PROCS1 - 1)
    uint lcl_proc =
        (uint)((float)lcl_id / (MLO_N_PROCS0 * MLO_N_PROCS1)); // input id from diff stack
    uint lcl_in_proc_id = -mad24((int)lcl_proc,
                                 (MLO_N_PROCS0 * MLO_N_PROCS1),
                                 -(int)lcl_id); // wk item id for the input to make a coalesed read
#else
    uint lcl_proc       = lcl_id / (MLO_N_PROCS0 * MLO_N_PROCS1); // input id from diff stack
    uint lcl_in_proc_id = lcl_id & (MLO_N_PROCS0 * MLO_N_PROCS1 -
                                    1); // wk item id for the input to make a coalesed read
#endif
    uint lcl_proc_id1 = (uint)((float)lcl_in_proc_id / MLO_N_PROCS0); //
#if MLO_N_PROCS0 & (MLO_N_PROCS0 - 1)
    uint lcl_proc_id0 = -mad24((int)lcl_proc_id1, MLO_N_PROCS0, -(int)lcl_in_proc_id); //
#else
    uint lcl_proc_id0   = lcl_in_proc_id & (MLO_N_PROCS0 - 1); //
#endif
    uint x_out_lcl = mul24(lcl_proc_id0, (uint)MLO_N_OUT_PIX_SZ0);
    uint y_out_lcl = mul24(lcl_proc_id1, (uint)MLO_N_OUT_PIX_SZ1);

    uint cb = get_global_id(2);
#if MLO_N_IN_CHNLS & (MLO_N_IN_CHNLS - 1)
    uint b_id = (uint)((float)cb / (float)MLO_N_IN_CHNLS);   // batch block
    uint c    = -mad24((int)b_id, MLO_N_IN_CHNLS, -(int)cb); // channel
#else
    uint b_id           = cb / MLO_N_IN_CHNLS;                 // batch block
    uint c              = cb & (MLO_N_IN_CHNLS - 1);           // channel
#endif
    // my batch
    uint b = b_id * MLO_LCL_N_IN_CHNLS + lcl_proc;

    uint x_out = x_out_grp + x_out_lcl;
    uint y_out = y_out_grp + y_out_lcl;

    uint in_off = b * MLO_IN_BATCH_STRIDE + c * MLO_IN_CHNL_STRIDE;

    for(uint i = lcl_id; i < MLO_LCL_IMG_SIZE * MLO_LCL_N_IN_CHNLS; i += MLO_GRP_SZ)
    {
        lcl_img[i] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

#if MLO_BIG

    readDataTile(lcl_img,
                 img,
                 y_in_grp,
                 x_in_grp,
                 MLO_IN_STRIDE,
                 (in_off + y_in_grp * MLO_IN_STRIDE + x_in_grp),
                 MLO_LCL_IMG_WIDTH,
                 0,
                 MLO_IN_HEIGHT,
                 MLO_IN_WIDTH,
                 MLO_LCL_IMG_HEIGHT,
                 MLO_LCL_IMG_WIDTH,
                 lcl_proc_id1,
                 lcl_proc_id0,
                 MLO_N_PROCS1,
                 MLO_N_PROCS0,
                 MLO_FLTR_PAD_SZ1,
                 MLO_FLTR_PAD_SZ0,
                 padding_val);

#else
    uint lcl_base       = MLO_LCL_IMG_WIDTH * MLO_FLTR_PAD_SZ1 + MLO_FLTR_PAD_SZ0;
    readData(&lcl_img[lcl_proc * MLO_LCL_IMG_SIZE + lcl_base],
             img,
             lcl_in_proc_id,
             (MLO_N_PROCS0 * MLO_N_PROCS1),
             (MLO_IN_WIDTH * MLO_IN_HEIGHT),
             MLO_IN_WIDTH,
             MLO_IN_STRIDE,
             in_off,
             MLO_LCL_IMG_WIDTH,
             0);

#endif
    barrier(CLK_LOCAL_MEM_FENCE);

#if MLO_BATCH_ALIGNED == 0
    if(b >= MLO_BATCH_SZ)
    {
        return;
    }
#endif

    // concatinate
    uint col_off = c * MLO_OUT_STRIDE * MLO_FLTR_SZ1 * MLO_FLTR_SZ0 +
                   (b * MLO_OUT_HEIGHT + y_out) * MLO_OUT_WIDTH + x_out;

    // get first MLO_N_OUT_PIX_SZ1 lines

    uint j        = 0;
    uint y_in_lcl = y_out * MLO_FLTR_STRIDE1 - y_in_grp;
    uint x_in_lcl = x_out * MLO_FLTR_STRIDE0 - x_in_grp;

    uint lcl_off = lcl_proc * MLO_LCL_IMG_SIZE + y_in_lcl * MLO_LCL_IMG_WIDTH + x_in_lcl;

    for(; j < MLO_PVT_COL_STG_HEIGHT - 1; ++j, lcl_off += MLO_LCL_IMG_WIDTH)
    {

        // read input data

        for(uint i = 0; i < MLO_PVT_COL_STG_WIDTH; ++i)
        {
            col_stg[j * MLO_PVT_COL_STG_WIDTH + i] = lcl_img[lcl_off + i];
        }
    }

    for(uint k = 0; k < MLO_FLTR_SZ1; ++k, ++j, lcl_off += MLO_LCL_IMG_WIDTH)
    {

        // read input data
        for(uint i = 0; i < MLO_PVT_COL_STG_WIDTH; ++i)
        {
            col_stg[(MLO_PVT_COL_STG_HEIGHT - 1) * MLO_PVT_COL_STG_WIDTH + i] =
                lcl_img[lcl_off + i];

#if 0
                        if(b==0 && c==0 && x_out==0 && y_out==0)
                        {
                                printf("k: j=%d i=%d y_l=%d x_l=%d l_of=%d v00=%f v10=%f  v01=%f v11=%f\n",
                                j,
                                i,
                                y_in_lcl,
                                x_in_lcl,
                                lcl_off + y_in_lcl * MLO_LCL_IMG_WIDTH + x_in_lcl,
                                lcl_img[lcl_off + (y_in_lcl-1) * MLO_LCL_IMG_WIDTH + x_in_lcl - 1],
                                lcl_img[lcl_off + (y_in_lcl-1) * MLO_LCL_IMG_WIDTH + x_in_lcl],
                                lcl_img[lcl_off + y_in_lcl * MLO_LCL_IMG_WIDTH + x_in_lcl - 1],
                                lcl_img[lcl_off + y_in_lcl * MLO_LCL_IMG_WIDTH + x_in_lcl]
                                );
                        }
#endif
        }

        // transpose
        uint out_offsets[MLO_N_OUT_PIX_SZ1 * MLO_N_OUT_PIX_SZ0];
        _FLOAT out_vals[MLO_N_OUT_PIX_SZ1 * MLO_N_OUT_PIX_SZ0];

        for(uint l = 0; l < MLO_FLTR_SZ0; ++l, col_off += MLO_OUT_STRIDE)
        {
            for(uint ii = 0; ii < MLO_N_OUT_PIX_SZ0; ++ii)
            {

                for(uint jj = 0, col_off3 = 0; jj < MLO_N_OUT_PIX_SZ1;
                    ++jj, col_off3 += MLO_OUT_WIDTH)
                {
                    out_vals[jj * MLO_N_OUT_PIX_SZ0 + ii] =
                        col_stg[jj * MLO_FLTR_STRIDE1 * MLO_PVT_COL_STG_WIDTH +
                                ii * MLO_FLTR_STRIDE0 + l];
                    out_offsets[jj * MLO_N_OUT_PIX_SZ0 + ii] = col_off + col_off3 + ii;
#if MLO_ALIGNED == 0
                    if(y_out + jj >= MLO_OUT_HEIGHT || x_out + ii >= MLO_OUT_WIDTH)
                    {
                        out_vals[jj * MLO_N_OUT_PIX_SZ0 + ii]    = 0;
                        out_offsets[jj * MLO_N_OUT_PIX_SZ0 + ii] = 0;
                    }
#endif
                }
            }

            for(uint ii = 0; ii < MLO_N_OUT_PIX_SZ0; ++ii)
            {

                for(uint jj = 0, col_off3 = 0; jj < MLO_N_OUT_PIX_SZ1;
                    ++jj, col_off3 += MLO_OUT_WIDTH)
                {
                    col[out_offsets[jj * MLO_N_OUT_PIX_SZ0 + ii]] =
                        out_vals[jj * MLO_N_OUT_PIX_SZ0 + ii];
                }
            }
        }

        // move up

        for(uint jj = 0; jj < MLO_PVT_COL_STG_HEIGHT - 1; ++jj)
        {

            for(uint ii = 0; ii < MLO_PVT_COL_STG_WIDTH; ++ii)
            {
                col_stg[jj * MLO_PVT_COL_STG_WIDTH + ii] =
                    col_stg[(jj + 1) * MLO_PVT_COL_STG_WIDTH + ii];
            }
        }
    }
}
