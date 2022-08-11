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
#define THREE 3
#define FOUR 4
#define EIGHT 8

#if MIOPEN_USE_FP16 == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT half
#endif
#if MIOPEN_USE_FP32 == 1
#define _FLOAT float
#endif

#define _FLOAT2 PPCAT(_FLOAT, TWO)
#define _FLOAT3 PPCAT(_FLOAT, THREE)
#define _FLOAT4 PPCAT(_FLOAT, FOUR)
#define _FLOAT8 PPCAT(_FLOAT, EIGHT)

#define DBG_OUT 0

#define UNUSED __attribute__((__unused__))

#define MLO_LRN_GROUP_SZ2 1
#define MLO_LRN_STRIDE 1

#define MLO_LRN_LEFT_PAD0 (((MLO_LRN_PRE_PAD0 + MLO_READ_UNIT - 1) / MLO_READ_UNIT) * MLO_READ_UNIT)
#define MLO_LRN_RIGHT_SIDE                                                               \
    (((MLO_LRN_GROUP_SZ0 * MLO_LRN_N_HORIZ_OUT_PIX + MLO_LRN_PAD0 + MLO_READ_UNIT - 1) / \
      MLO_READ_UNIT) *                                                                   \
     MLO_READ_UNIT)
#define MLO_LRN_LCL_DATA_WIDTH (MLO_LRN_LEFT_PAD0 + MLO_LRN_RIGHT_SIDE)
#define MLO_LCL_READ4 (MLO_LRN_LCL_DATA_WIDTH / MLO_READ_UNIT)
#define MLO_LRN_LCL_DATA_HEIGHT (MLO_LRN_GROUP_SZ1 * MLO_LRN_N_VERT_OUT_PIX + MLO_LRN_KERNEL_SZ - 1)
#define MLO_LRN_GROUP_SZ (MLO_LRN_GROUP_SZ2 * MLO_LRN_GROUP_SZ1 * MLO_LRN_GROUP_SZ0)
//#define MLO_LRN_PREPAD_SZ (MLO_LRN_KERNEL_SZ - 1)/2

struct LRNForwardParam
{
    _FLOAT alphaoverarea;
    _FLOAT alpha;
    _FLOAT beta;
    _FLOAT K;
};

#include "math_ops.h"
__attribute__((reqd_work_group_size(MLO_LRN_GROUP_SZ0, MLO_LRN_GROUP_SZ1, MLO_LRN_GROUP_SZ2)))
__kernel void
MIOpenLRNWithinChannel_PS(const __global _FLOAT* bot,
                          __global _FLOAT* top,
#if MLO_LRN_DO_SCALE
                          __global _FLOAT* scale,
#endif
                          _FLOAT alphaoverarea,
                          UNUSED _FLOAT alpha,
                          _FLOAT beta,
                          _FLOAT K)
{
    // IT's taken from POOLING AVE with stride = 1'
    __local _FLOAT bot_data[MLO_LRN_LCL_DATA_WIDTH * MLO_LRN_LCL_DATA_HEIGHT];
    int x       = get_group_id(0) * MLO_LRN_GROUP_SZ0 * MLO_LRN_N_HORIZ_OUT_PIX;
    int y       = get_group_id(1) * MLO_LRN_GROUP_SZ1 * MLO_LRN_N_VERT_OUT_PIX;
    int lcl_id0 = get_local_id(0);
    int lcl_id1 = get_local_id(1);
    int ob      = get_global_id(2); // output * batch_sz
    int o       = iDiv_legacy(ob, MLO_LRN_BATCH_SZ);
    int b       = iMod(ob, o, MLO_LRN_BATCH_SZ);
    int bot_x   = x;
    int bot_y   = y;
    int bot_off = b * MLO_LRN_BOT_BATCH_STRIDE + o * MLO_LRN_BOT_CHANNEL_STRIDE;

    // load tile
    for(int b_j = lcl_id1; b_j < MLO_LRN_LCL_DATA_HEIGHT; b_j += MLO_LRN_GROUP_SZ1)
    {
        int bot_y_act = bot_y + b_j - MLO_LRN_PRE_PAD1;

        bool invisibleY = (bot_y_act < 0) || (bot_y_act >= MLO_LRN_BOT_HEIGHT);

        int bot_y_off = bot_y_act * MLO_LRN_BOT_STRIDE;

        int lcl_off_v = mul24(b_j, (int)MLO_LRN_LCL_DATA_WIDTH);

        for(int b_i = lcl_id0; b_i < MLO_LCL_READ4; b_i += MLO_LRN_GROUP_SZ0)
        {

            int bot_x_act = bot_x + (b_i * MLO_READ_UNIT) - MLO_LRN_LEFT_PAD0;

            bool invisibleX;
            for(int i = 0; i < MLO_READ_UNIT; ++i)
            {

                int bot_off_x = bot_off + bot_y_off + bot_x_act + i;

                invisibleX = (bot_x_act + i < 0) || (bot_x_act + i >= MLO_LRN_BOT_WIDTH);

                bot_off_x = (invisibleX || invisibleY) ? 0 : bot_off_x;

                _FLOAT bot_val = bot[bot_off_x];

                bot_val = (invisibleX || invisibleY) ? 0 : bot_val;

                bot_data[lcl_off_v + (b_i * MLO_READ_UNIT) + i] = bot_val;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
#if MLO_LRN_N_HORIZ_OUT_PIX > 1
    _FLOAT partial_sum_x[MLO_LRN_N_HORIZ_OUT_PIX - 1]; // horizontal partial sum
#endif
#if MLO_LRN_N_VERT_OUT_PIX > 1
    _FLOAT partial_sum_xy[MLO_LRN_N_VERT_OUT_PIX - 1]
                         [MLO_LRN_N_HORIZ_OUT_PIX]; // horizontal-vertical partial sums.
#endif
    _FLOAT accum[MLO_LRN_N_VERT_OUT_PIX][MLO_LRN_N_HORIZ_OUT_PIX]; // accumulator

    int top_y = mad24(lcl_id1, (int)MLO_LRN_N_VERT_OUT_PIX, y);
    int top_x = mad24(lcl_id0, (int)MLO_LRN_N_HORIZ_OUT_PIX, x);

    int lcl_y = mul24(lcl_id1, (int)MLO_LRN_N_VERT_OUT_PIX);
    int lcl_x =
        mad24(lcl_id0, (int)(MLO_LRN_N_HORIZ_OUT_PIX), (int)(MLO_LRN_LEFT_PAD0 - MLO_LRN_PRE_PAD0));
    int lcl_off = mad24(lcl_y, MLO_LRN_LCL_DATA_WIDTH, lcl_x);

    for(int j = 0; j < MLO_LRN_N_VERT_OUT_PIX; ++j)
    {
        for(int i = 0; i < MLO_LRN_N_HORIZ_OUT_PIX; ++i)
        {
            accum[j][i] = 0;
        }
    }
#if MLO_LRN_N_VERT_OUT_PIX > 1
    for(int j = 0; j < MLO_LRN_N_VERT_OUT_PIX - 1; ++j)
    {
        for(int i = 0; i < MLO_LRN_N_HORIZ_OUT_PIX; ++i)
        {
            partial_sum_xy[j][i] = 0;
        }
    }
#endif

    // running window  summation
    _FLOAT mov_accum;
    int jj = 0;
    int ii = 0;

    // first to get vertica partial sums

#if MLO_LRN_N_VERT_OUT_PIX > 1
    for(; jj < (int)(MLO_LRN_N_VERT_OUT_PIX - 1); ++jj)
    {
        for(ii = 0; ii < (int)(MLO_LRN_N_HORIZ_OUT_PIX - 1); ++ii)
        {

            _FLOAT bot_val = bot_data[lcl_off + jj * MLO_LRN_LCL_DATA_WIDTH + ii];

            _FLOAT accum_tmp = bot_val * bot_val;

#if MLO_LRN_N_HORIZ_OUT_PIX > 1
            // save horizontal partial sums
            partial_sum_x[ii] = accum_tmp;
#endif
            // accumulate in vert-horizontal(0)
            partial_sum_xy[jj][0] += accum_tmp;
        }

        for(; ii < (int)MLO_LRN_KERNEL_SZ0; ++ii)
        {

            _FLOAT bot_val   = bot_data[lcl_off + jj * MLO_LRN_LCL_DATA_WIDTH + ii];
            _FLOAT accum_tmp = bot_val * bot_val;
            // accumulate in vert horizontal(0)
            partial_sum_xy[jj][0] += accum_tmp;
        }

        // running horizontal window

        for(; ii < (int)(MLO_LRN_KERNEL_SZ0 + MLO_LRN_N_HORIZ_OUT_PIX - 1); ++ii)
        {

            _FLOAT bot_val   = bot_data[lcl_off + jj * MLO_LRN_LCL_DATA_WIDTH + ii];
            _FLOAT accum_tmp = bot_val * bot_val;
            // calculate all vertical-horizontal partial sums
            partial_sum_xy[jj][ii - MLO_LRN_KERNEL_SZ0 + 1] =
                partial_sum_xy[jj][ii - MLO_LRN_KERNEL_SZ0] +
                (accum_tmp
#if MLO_LRN_N_HORIZ_OUT_PIX > 1
                 - partial_sum_x[ii - MLO_LRN_KERNEL_SZ0]
#endif
                );
        }

        // put into accumulator[0][i]
        // whatever has been accumulated so far
        for(int i = 0; i < MLO_LRN_N_HORIZ_OUT_PIX; ++i)
        {
            accum[0][i] += partial_sum_xy[jj][i];
        }
    }
#endif

    // calculate row 0 accumulators
    for(; jj < (int)MLO_LRN_KERNEL_SZ1; ++jj)
    {
        mov_accum = 0;

        for(ii = 0; ii < (int)(MLO_LRN_N_HORIZ_OUT_PIX - 1); ++ii)
        {

            _FLOAT bot_val   = bot_data[lcl_off + jj * MLO_LRN_LCL_DATA_WIDTH + ii];
            _FLOAT accum_tmp = bot_val * bot_val;
#if MLO_LRN_N_HORIZ_OUT_PIX > 1
            partial_sum_x[ii] = accum_tmp;
#endif
            mov_accum += accum_tmp;
        }

        for(; ii < (int)MLO_LRN_KERNEL_SZ0; ++ii)
        {

            _FLOAT bot_val   = bot_data[lcl_off + jj * MLO_LRN_LCL_DATA_WIDTH + ii];
            _FLOAT accum_tmp = bot_val * bot_val;
            mov_accum += accum_tmp;
        }

        accum[0][0] += mov_accum;
        // running horizontal window

        for(; ii < (int)(MLO_LRN_KERNEL_SZ0 + MLO_LRN_N_HORIZ_OUT_PIX - 1); ++ii)
        {

            _FLOAT bot_val   = bot_data[lcl_off + jj * MLO_LRN_LCL_DATA_WIDTH + ii];
            _FLOAT accum_tmp = bot_val * bot_val;
            // running horizontal window
            mov_accum += (accum_tmp
#if MLO_LRN_N_HORIZ_OUT_PIX > 1
                          - partial_sum_x[ii - MLO_LRN_KERNEL_SZ0]
#endif
            );
            accum[0][ii - MLO_LRN_KERNEL_SZ0 + 1] += mov_accum;
        }
    }

    // accumulate all other rows besides 0
    for(; jj < (int)(MLO_LRN_KERNEL_SZ1 + MLO_LRN_N_VERT_OUT_PIX - 1); ++jj)
    {
        // first running horizontal winodw as before
        mov_accum = 0;
        for(ii = 0; ii < (int)(MLO_LRN_N_HORIZ_OUT_PIX - 1); ++ii)
        {

            _FLOAT bot_val   = bot_data[lcl_off + jj * MLO_LRN_LCL_DATA_WIDTH + ii];
            _FLOAT accum_tmp = bot_val * bot_val;
#if MLO_LRN_N_HORIZ_OUT_PIX > 1
            partial_sum_x[ii] = accum_tmp;
#endif
            accum[jj - MLO_LRN_KERNEL_SZ1 + 1][0] += accum_tmp;
        }
        for(; ii < (int)MLO_LRN_KERNEL_SZ0; ++ii)
        {

            _FLOAT bot_val   = bot_data[lcl_off + jj * MLO_LRN_LCL_DATA_WIDTH + ii];
            _FLOAT accum_tmp = bot_val * bot_val;
            accum[jj - MLO_LRN_KERNEL_SZ1 + 1][0] += accum_tmp;
        }
        // running horizontal window

        int ii1 = ii;
        for(; ii < (int)(MLO_LRN_KERNEL_SZ0 + MLO_LRN_N_HORIZ_OUT_PIX - 1); ++ii)
        {

            _FLOAT bot_val   = bot_data[lcl_off + jj * MLO_LRN_LCL_DATA_WIDTH + ii];
            _FLOAT accum_tmp = bot_val * bot_val;
            //
            accum[jj - MLO_LRN_KERNEL_SZ1 + 1][ii - MLO_LRN_KERNEL_SZ0 + 1] =
                accum[jj - MLO_LRN_KERNEL_SZ1 + 1][ii - MLO_LRN_KERNEL_SZ0] + accum_tmp;
#if MLO_LRN_N_HORIZ_OUT_PIX > 1
            accum[jj - MLO_LRN_KERNEL_SZ1 + 1][ii - MLO_LRN_KERNEL_SZ0 + 1] -=
                partial_sum_x[ii - MLO_LRN_KERNEL_SZ0];
#endif
        }

        // finally running vertical window

        for(ii = ii1; ii < (int)(MLO_LRN_KERNEL_SZ0 + MLO_LRN_N_HORIZ_OUT_PIX - 1); ++ii)
        {

            // finish horizontal summation
            // add/substarct vertical patial sum
            accum[jj - MLO_LRN_KERNEL_SZ1 + 1][ii - MLO_LRN_KERNEL_SZ0 + 1] +=
                accum[jj - MLO_LRN_KERNEL_SZ1][ii - MLO_LRN_KERNEL_SZ0 + 1];
#if MLO_LRN_N_VERT_OUT_PIX > 1
            accum[jj - MLO_LRN_KERNEL_SZ1 + 1][ii - MLO_LRN_KERNEL_SZ0 + 1] -=
                partial_sum_xy[jj - MLO_LRN_KERNEL_SZ1][ii - MLO_LRN_KERNEL_SZ0 + 1];
#endif
        }
#if MLO_LRN_N_VERT_OUT_PIX > 1
        accum[jj - MLO_LRN_KERNEL_SZ1 + 1][0] -= partial_sum_xy[jj - MLO_LRN_KERNEL_SZ1][0];
#endif
        accum[jj - MLO_LRN_KERNEL_SZ1 + 1][0] += accum[jj - MLO_LRN_KERNEL_SZ1][0];
    }

    // normalization
    _FLOAT prv_scale[MLO_LRN_N_VERT_OUT_PIX][MLO_LRN_N_HORIZ_OUT_PIX];
    _FLOAT adj_alphaoverarea = alphaoverarea;
    for(int k = 0; k < MLO_LRN_N_VERT_OUT_PIX; k++)
    {

        //			int hstart = y + lcl_id1 * MLO_LRN_N_VERT_OUT_PIX  + k -
        // MLO_LRN_PAD1;
        //			int hend = min(hstart + MLO_LRN_KERNEL_SZ, MLO_LRN_BOT_HEIGHT +
        // MLO_LRN_PAD1);

        for(int l = 0; l < MLO_LRN_N_HORIZ_OUT_PIX; l++)
        {

            //				int wstart = x + lcl_id0 * MLO_LRN_N_HORIZ_OUT_PIX + l -
            // MLO_LRN_PAD0;
            //				int wend = min(wstart + MLO_LRN_KERNEL_SZ, MLO_LRN_BOT_WIDTH
            //+
            // MLO_LRN_PAD0);
            //				int adj_area_size = (hend - hstart) * (wend - wstart);
            //				adj_alphaoverarea = alpha / adj_area_size;

            prv_scale[k][l] = K + accum[k][l] * adj_alphaoverarea;
        }
    }

    int top_off = b * MLO_LRN_TOP_BATCH_STRIDE + o * MLO_LRN_TOP_CHANNEL_STRIDE +
                  top_y * MLO_LRN_TOP_STRIDE + top_x;
#if MLO_LRN_DO_SCALE
    int scale_off = b * MLO_LRN_SCALE_BATCH_STRIDE + o * MLO_LRN_SCALE_CHANNEL_STRIDE +
                    top_y * MLO_LRN_SCALE_STRIDE + top_x;
#endif

    // final output

    for(int k = 0; k < MLO_LRN_N_VERT_OUT_PIX
#if MLO_OUT_VERT_ALIGNED == 0
                   && (top_y + k < MLO_LRN_TOP_HEIGHT)
#endif
            ;
        k++)
    {
        for(int l = 0; l < MLO_LRN_N_HORIZ_OUT_PIX
#if MLO_OUT_HORIZ_ALIGNED == 0
                       && (top_x + l < MLO_LRN_TOP_WIDTH)
#endif
                ;
            l++)
        {
            _FLOAT s;
            s = exp((_FLOAT)-beta * log(prv_scale[k][l]));
            //					s = pow(prv_scale[k][l], -beta);
            _FLOAT bot_val = bot_data[lcl_off + mad24((k + MLO_LRN_PRE_PAD1),
                                                      (int)MLO_LRN_LCL_DATA_WIDTH,
                                                      (l + MLO_LRN_PRE_PAD0))];
#if MLO_LRN_DO_SCALE
            scale[scale_off + k * MLO_LRN_SCALE_STRIDE + l] = prv_scale[k][l];
#endif
            top[top_off + k * MLO_LRN_TOP_STRIDE + l] = bot_val * s;
        }
    }
}

#if(MLO_LRN_N_INPUTS < MLO_LRN_KERNEL_SZ)
#define MLO_LOW_CHNL_COUNT 1
#else
#define MLO_LOW_CHNL_COUNT 0
#endif
__attribute__((reqd_work_group_size(MLO_LRN_GROUP_SZ0, MLO_LRN_GROUP_SZ1, MLO_LRN_GROUP_SZ2)))
__kernel void
MIOpenLRNAcrossChannels4(const __global _FLOAT* bottom,
                         __global _FLOAT* top,
#if MLO_LRN_DO_SCALE
                         __global _FLOAT* scale,
#endif
                         _FLOAT alphaoverarea,
                         UNUSED _FLOAT alpha,
                         _FLOAT beta,
                         _FLOAT K)
{
    int pix_id          = get_global_id(0); //
    int b               = get_global_id(2); // batch
    MLO_READ_TYPE accum = 0;
    MLO_READ_TYPE bot_in2[MLO_LRN_KERNEL_SZ];
    MLO_READ_TYPE bot_in[MLO_LRN_KERNEL_SZ];
    int c_i = 0, c_o = 0;
    for(int i = 0; i < MLO_LRN_KERNEL_SZ; ++i)
    {
        bot_in2[i] = 0;
        bot_in[i]  = 0;
    }

    int top_off = 0;
#if MLO_LRN_DO_SCALE
    int scale_off;
#endif

    for(c_i = 0; c_i < MLO_LRN_PAD; c_i++)
    {
        MLO_READ_TYPE prv_in;
        prv_in = 0;

#if MLO_LOW_CHNL_COUNT == 1
        if(c_i < MLO_LRN_N_INPUTS)
#endif
        {
#if MLO_C1x1_PIXLEFT > 0
            // if the last one
            if(pix_id == MLO_MAP_SZ4 - 1)
            {

                for(int j = 0; j < MLO_C1x1_PIXLEFT; ++j)
                {
                    ((_FLOAT*)&prv_in)[j] =
                        bottom[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE * c_i +
                               (pix_id * MLO_READ_UNIT) + j];
                }
            }
            else
#endif
            {
                prv_in = *(__global MLO_READ_TYPE*)&bottom[MLO_LRN_BOT_BATCH_STRIDE * b +
                                                           MLO_LRN_BOT_CHANNEL_STRIDE * c_i +
                                                           (pix_id * MLO_READ_UNIT)];
            }
        }

        bot_in2[c_i] = prv_in * prv_in;
        bot_in[c_i]  = prv_in;
        accum        = accum + bot_in2[c_i];
        //				fma(bot_in2[c_i + MLO_LRN_PAD], bot_in2[c_i + MLO_LRN_PAD],
        // accum);
    }

    for(; c_i < MLO_LRN_KERNEL_SZ; c_i++, c_o++)
    {
        MLO_READ_TYPE prv_in;
        prv_in = 0;

#if MLO_LOW_CHNL_COUNT == 1
        if(c_i < MLO_LRN_N_INPUTS)
#endif
        {

#if MLO_C1x1_PIXLEFT > 0
            // if the last one
            if(pix_id == MLO_MAP_SZ4 - 1)
            {

                for(int j = 0; j < MLO_C1x1_PIXLEFT; ++j)
                {
                    ((_FLOAT*)&prv_in)[j] =
                        bottom[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE * c_i +
                               (pix_id * MLO_READ_UNIT) + j];
                }
            }
            else
#endif
            {
                prv_in = *(__global MLO_READ_TYPE*)&bottom[MLO_LRN_BOT_BATCH_STRIDE * b +
                                                           MLO_LRN_BOT_CHANNEL_STRIDE * c_i +
                                                           (pix_id * MLO_READ_UNIT)];
            }
        }

        bot_in2[c_i] = prv_in * prv_in;
        bot_in[c_i]  = prv_in;
        accum        = accum + bot_in2[c_i];

        top_off = b * MLO_LRN_TOP_BATCH_STRIDE + c_o * MLO_LRN_TOP_CHANNEL_STRIDE +
                  (pix_id * MLO_READ_UNIT);
#if MLO_LRN_DO_SCALE
        scale_off = b * MLO_LRN_SCALE_BATCH_STRIDE + c_o * MLO_LRN_SCALE_CHANNEL_STRIDE +
                    (pix_id * MLO_READ_UNIT);
#endif
        MLO_READ_TYPE prv_scale = ((MLO_READ_TYPE)K + accum * (MLO_READ_TYPE)alphaoverarea);
        //				fma(accum,alphaoverarea, (_FLOAT)1.f);

        MLO_READ_TYPE exp_scale = exp((MLO_READ_TYPE)-beta * log(prv_scale));
        //				pow(prv_scale,-beta);
        // bug
        //	MLO_READ_TYPE prv_out = sqrt(bot_in2[c_o]);
        MLO_READ_TYPE prv_out = bot_in[c_o];
        MLO_READ_TYPE out_val = prv_out * exp_scale;
#if MLO_LOW_CHNL_COUNT == 1
        if(c_o < MLO_LRN_N_OUTPUTS)
#endif
        {

#if MLO_C1x1_PIXLEFT > 0

            // if the last one
            if(pix_id == MLO_MAP_SZ4 - 1)
            {
                for(int j = 0; j < MLO_C1x1_PIXLEFT; ++j)
                {
                    top[top_off + j] = ((_FLOAT*)&out_val)[j];
#if DBG_OUT
                    printf("K:o0: %d %f %f %f %f %f\n",
                           top_off + j,
                           top[top_off + j],
                           ((_FLOAT*)&prv_out)[j],
                           ((_FLOAT*)&exp_scale)[j],
                           ((_FLOAT*)&prv_scale)[j],
                           ((_FLOAT*)&accum)[j]);
#endif

#if MLO_LRN_DO_SCALE
                    scale[scale_off + j] = ((_FLOAT*)&prv_scale)[j];
#endif
                }
            }
            else
#endif
            {

                *((__global MLO_READ_TYPE*)&top[top_off]) = out_val;
#if MLO_LRN_DO_SCALE
                *((__global MLO_READ_TYPE*)&scale[scale_off]) = prv_scale;
#endif
            }
        }
    }

    for(; c_i < MLO_LRN_N_INPUTS; c_i++, c_o++)
    {

        MLO_READ_TYPE prv_in;
        prv_in = 0;

#if MLO_C1x1_PIXLEFT > 0
        // if the last one
        if(pix_id == MLO_MAP_SZ4 - 1)
        {

            for(int j = 0; j < MLO_C1x1_PIXLEFT; ++j)
            {
                ((_FLOAT*)&prv_in)[j] =
                    bottom[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE * c_i +
                           (pix_id * MLO_READ_UNIT) + j];
            }
        }
        else
#endif
        {
            prv_in = *(__global MLO_READ_TYPE*)&bottom[MLO_LRN_BOT_BATCH_STRIDE * b +
                                                       MLO_LRN_BOT_CHANNEL_STRIDE * c_i +
                                                       (pix_id * MLO_READ_UNIT)];
        }

        MLO_READ_TYPE prv_bot_in2 = prv_in * prv_in;
        accum                     = accum + prv_bot_in2;

        accum = accum - bot_in2[0];
        //				fma(-bot_in2[0], bot_in2[0], accum);

        for(int i = 0; i < MLO_LRN_KERNEL_SZ - 1; i++)
        {
            bot_in2[i] = bot_in2[i + 1];
            bot_in[i]  = bot_in[i + 1];
        }

        bot_in2[MLO_LRN_KERNEL_SZ - 1] = prv_bot_in2;
        bot_in[MLO_LRN_KERNEL_SZ - 1]  = prv_in;

        top_off = b * MLO_LRN_TOP_BATCH_STRIDE + c_o * MLO_LRN_TOP_CHANNEL_STRIDE +
                  (pix_id * MLO_READ_UNIT);
#if MLO_LRN_DO_SCALE
        scale_off = b * MLO_LRN_SCALE_BATCH_STRIDE + c_o * MLO_LRN_SCALE_CHANNEL_STRIDE +
                    (pix_id * MLO_READ_UNIT);
#endif
        MLO_READ_TYPE prv_scale = ((MLO_READ_TYPE)K + accum * (MLO_READ_TYPE)alphaoverarea);
        //				fma(accum,alphaoverarea, (_FLOAT)1.f);

        MLO_READ_TYPE exp_scale = exp((MLO_READ_TYPE)-beta * log(prv_scale));
        //				pow(prv_scale,-beta);
        // bug
        //			MLO_READ_TYPE prv_out = sqrt(bot_in2[MLO_LRN_PRE_PAD]);
        MLO_READ_TYPE prv_out = bot_in[MLO_LRN_PRE_PAD];
        MLO_READ_TYPE out_val = prv_out * exp_scale;

#if MLO_LOW_CHNL_COUNT == 1
        if(c_o < MLO_LRN_N_OUTPUTS)
#endif
        {

#if MLO_C1x1_PIXLEFT > 0

            // if the last one
            if(pix_id == MLO_MAP_SZ4 - 1)
            {
                for(int j = 0; j < MLO_C1x1_PIXLEFT; ++j)
                {
                    top[top_off + j] = ((_FLOAT*)&out_val)[j];
#if DBG_OUT
                    printf("K:o1: %d %f %f %f\n",
                           top_off + j,
                           top[top_off + j],
                           ((_FLOAT*)&prv_out)[j],
                           ((_FLOAT*)&exp_scale)[j]);
#endif

#if MLO_LRN_DO_SCALE
                    scale[scale_off + j] = ((_FLOAT*)&prv_scale)[j];
#endif
                }
            }
            else
#endif
            {

                *((__global MLO_READ_TYPE*)&top[top_off]) = out_val;
#if MLO_LRN_DO_SCALE
                *((__global MLO_READ_TYPE*)&scale[scale_off]) = prv_scale;
#endif
            }
        }
    }

    for(; c_i < MLO_LRN_N_INPUTS + MLO_LRN_PAD; c_i++, c_o++)
    {

        accum = accum - bot_in2[0];
        //				fma(-bot_in2[0], bot_in2[0], accum);

        for(int i = 0; i < MLO_LRN_KERNEL_SZ - 1; i++)
        {
            bot_in2[i] = bot_in2[i + 1];
            bot_in[i]  = bot_in[i + 1];
        }

        top_off = b * MLO_LRN_TOP_BATCH_STRIDE + c_o * MLO_LRN_TOP_CHANNEL_STRIDE +
                  (pix_id * MLO_READ_UNIT);
#if MLO_LRN_DO_SCALE
        scale_off = b * MLO_LRN_SCALE_BATCH_STRIDE + c_o * MLO_LRN_SCALE_CHANNEL_STRIDE +
                    (pix_id * MLO_READ_UNIT);
#endif
        MLO_READ_TYPE prv_scale = ((MLO_READ_TYPE)K + accum * (MLO_READ_TYPE)alphaoverarea);
        //				fma(accum,alphaoverarea, (_FLOAT)1.f);

        MLO_READ_TYPE exp_scale = exp((MLO_READ_TYPE)-beta * log(prv_scale));
        //				pow(prv_scale,-beta);
        // bug
        //			MLO_READ_TYPE prv_out = sqrt(bot_in2[MLO_LRN_PRE_PAD]);
        MLO_READ_TYPE prv_out = bot_in[MLO_LRN_PRE_PAD];

        MLO_READ_TYPE out_val = prv_out * exp_scale;
#if MLO_LOW_CHNL_COUNT == 1
        if(c_o < MLO_LRN_N_OUTPUTS)
#endif
        {

#if MLO_C1x1_PIXLEFT > 0

            // if the last one
            if(pix_id == MLO_MAP_SZ4 - 1)
            {
                for(int j = 0; j < MLO_C1x1_PIXLEFT; ++j)
                {
                    top[top_off + j] = ((_FLOAT*)&out_val)[j];
#if DBG_OUT
                    printf("K:o2: %d %f %f %f\n",
                           top_off + j,
                           top[top_off + j],
                           ((_FLOAT*)&prv_out)[j],
                           ((_FLOAT*)&exp_scale)[j]);
#endif

#if MLO_LRN_DO_SCALE
                    scale[scale_off + j] = ((_FLOAT*)&prv_scale)[j];
#endif
                }
            }
            else
#endif
            {

                *((__global MLO_READ_TYPE*)&top[top_off]) = out_val;
#if MLO_LRN_DO_SCALE
                *((__global MLO_READ_TYPE*)&scale[scale_off]) = prv_scale;
#endif
            }
        }
    }
}
