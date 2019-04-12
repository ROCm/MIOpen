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
#define _FLOAT_PREC half
#define EPSILON (_FLOAT)0.0001
#endif
#if MIOPEN_USE_FP32 == 1
#define _FLOAT float
#define _FLOAT_PREC float
#define EPSILON (_FLOAT)0.000001
#endif

#define _FLOAT2 PPCAT(_FLOAT, TWO)
#define _FLOAT4 PPCAT(_FLOAT, FOUR)
#define _FLOAT8 PPCAT(_FLOAT, EIGHT)

#define UNUSED __attribute__((__unused__))

#define MIOPEN_NRN_GROUP_SZ2 1

#include "activation_functions.h"

#ifdef LITE

/**********************************************************************************************
**********************************************************************************************/

// N - batch size
// C - # of maps
// H - map height
// W - map width
// TENS_LEN = (N*C*H*W);
// RD_BLCK = (TENS_LEN%4==0) ? 4 : (TENS_LEN%3==0)? 3 : (TENS_LEN%2==0)? 2 : 1;
// READ_TYPE = (RD_BLCK==4) ? "float4" : (RD_BLCK == 3) ? "float3" : (RD_BLC==2) ? "float2" :
// "float";
// local size = (256, 1, 1)
// global size = ((TENS_LEN/RD_BLCK), 1, 1)

__kernel void MIOpenActiveFwdLite(const __global _FLOAT* bot,
                                  __global _FLOAT* top,
                                  _FLOAT gamma,
                                  _FLOAT beta,
                                  _FLOAT alpha,
                                  const long bot_offset,
                                  const long top_offset)
{
    uint gid0 = get_global_id(0);

    uint index = gid0 * MIOPEN_READ_UNIT;

    _FLOAT data[MIOPEN_READ_UNIT];
    _FLOAT response[MIOPEN_READ_UNIT];

    *((MIOPEN_READ_TYPE*)data) = *((const __global MIOPEN_READ_TYPE*)(bot + bot_offset + index));

    ActivationFunction(MIOPEN_READ_UNIT, response, (const _FLOAT*)data, gamma, beta, alpha);

    *((__global MIOPEN_READ_TYPE*)(top + top_offset + index)) = *((MIOPEN_READ_TYPE*)response);
}

/**********************************************************************************************
**********************************************************************************************/

__kernel void MIOpenActiveFwd2DLite(const __global _FLOAT* bot,
                                    __global _FLOAT* top,
                                    _FLOAT gamma,
                                    _FLOAT beta,
                                    _FLOAT alpha,
                                    const long bot_offset,
                                    const long top_offset,
                                    const uint bot_stride,
                                    const uint top_stride)
{
    uint x_id = get_global_id(0);
    uint y    = get_global_id(1);

    uint bot_index = y * bot_stride + x_id * MIOPEN_READ_UNIT;
    uint top_index = y * top_stride + x_id * MIOPEN_READ_UNIT;

    _FLOAT data[MIOPEN_READ_UNIT];
    _FLOAT response[MIOPEN_READ_UNIT];

    *((MIOPEN_READ_TYPE*)data) =
        *((const __global MIOPEN_READ_TYPE*)(bot + bot_offset + bot_index));

    ActivationFunction(MIOPEN_READ_UNIT, response, (const _FLOAT*)data, gamma, beta, alpha);

    *((__global MIOPEN_READ_TYPE*)(top + top_offset + top_index)) = *((MIOPEN_READ_TYPE*)response);
}

/**********************************************************************************************
**********************************************************************************************/

__kernel void MIOpenActiveBwdLite(__global _FLOAT* bot_diff,
                                  __global const _FLOAT* top_diff,
                                  __global const _FLOAT* bot,
                                  __global const _FLOAT* top,
                                  _FLOAT diff_scale,
                                  _FLOAT gamma,
                                  _FLOAT beta,
                                  _FLOAT alpha,
                                  const long bot_diff_offset,
                                  const long top_diff_offset,
                                  const long bot_offset,
                                  const long top_offset)
{
    int gid0 = get_global_id(0);

    int index = gid0 * MIOPEN_READ_UNIT;

    _FLOAT bot_diff_dat[MIOPEN_READ_UNIT];
    _FLOAT top_diff_dat[MIOPEN_READ_UNIT];
    _FLOAT bot_dat[MIOPEN_READ_UNIT];
    _FLOAT top_dat[MIOPEN_READ_UNIT];

    *((MIOPEN_READ_TYPE*)top_diff_dat) =
        *((const __global MIOPEN_READ_TYPE*)(top_diff + top_diff_offset + index));
    *((MIOPEN_READ_TYPE*)bot_dat) = *((const __global MIOPEN_READ_TYPE*)(bot + bot_offset + index));
    *((MIOPEN_READ_TYPE*)top_dat) = *((const __global MIOPEN_READ_TYPE*)(top + top_offset + index));

    ActivationFunction_Diff(MIOPEN_READ_UNIT,
                            bot_diff_dat,
                            top_diff_dat,
                            bot_dat,
                            top_dat,
                            diff_scale,
                            gamma,
                            beta,
                            alpha);

    *((__global MIOPEN_READ_TYPE*)(bot_diff + bot_diff_offset + index)) =
        *((MIOPEN_READ_TYPE*)bot_diff_dat);
}

/**********************************************************************************************
**********************************************************************************************/

__kernel void MIOpenActiveBwd2DLite(__global _FLOAT* bot_diff,
                                    __global const _FLOAT* top_diff,
                                    __global const _FLOAT* bot,
                                    __global const _FLOAT* top,
                                    _FLOAT diff_scale,
                                    _FLOAT gamma,
                                    _FLOAT beta,
                                    _FLOAT alpha,
                                    const long bot_diff_offset,
                                    const long top_diff_offset,
                                    const long bot_offset,
                                    const long top_offset,
                                    const uint bot_diff_stride,
                                    const uint top_diff_stride,
                                    const uint bot_stride,
                                    const uint top_stride)
{
    uint x_id = get_global_id(0);
    uint y    = get_global_id(1);

    uint bot_diff_index = y * bot_diff_stride + x_id * MIOPEN_READ_UNIT;
    uint top_diff_index = y * top_diff_stride + x_id * MIOPEN_READ_UNIT;
    uint bot_index      = y * bot_stride + x_id * MIOPEN_READ_UNIT;
    uint top_index      = y * top_stride + x_id * MIOPEN_READ_UNIT;

    _FLOAT bot_diff_dat[MIOPEN_READ_UNIT];
    _FLOAT top_diff_dat[MIOPEN_READ_UNIT];
    _FLOAT bot_dat[MIOPEN_READ_UNIT];
    _FLOAT top_dat[MIOPEN_READ_UNIT];

    *((MIOPEN_READ_TYPE*)top_diff_dat) =
        *((const __global MIOPEN_READ_TYPE*)(top_diff + top_diff_offset + top_diff_index));
    *((MIOPEN_READ_TYPE*)bot_dat) =
        *((const __global MIOPEN_READ_TYPE*)(bot + bot_offset + bot_index));
    *((MIOPEN_READ_TYPE*)top_dat) =
        *((const __global MIOPEN_READ_TYPE*)(top + top_offset + top_index));

    ActivationFunction_Diff(MIOPEN_READ_UNIT,
                            bot_diff_dat,
                            top_diff_dat,
                            bot_dat,
                            top_dat,
                            diff_scale,
                            gamma,
                            beta,
                            alpha);

    *((__global MIOPEN_READ_TYPE*)(bot_diff + bot_diff_offset + bot_diff_index)) =
        *((MIOPEN_READ_TYPE*)bot_diff_dat);
}

/**************************************************************************************************************/

#else

/***************************************************************************************************************/
__attribute__((reqd_work_group_size(MIOPEN_NRN_GROUP_SZ0,
                                    MIOPEN_NRN_GROUP_SZ1,
                                    MIOPEN_NRN_GROUP_SZ2))) __kernel void
MIOpenNeuronFwd(const __global _FLOAT* bot,
                __global _FLOAT* top,
                _FLOAT gamma,
                _FLOAT beta,
                _FLOAT alpha,
                const long xOffset,
                const long yOffset)
{
    int x            = get_global_id(0); // channel x

#if MIOPEN_N_OUT_STRIDE > MIOPEN_OUT_BLOCK_SZ
    int n_out_stride = MIOPEN_N_OUT_STRIDE;
    int c_out        = MIOPEN_C_OUT;
    int h_out        = MIOPEN_H_OUT;
    int w_out        = MIOPEN_W_OUT;
#endif
#if MIOPEN_N_IN_STRIDE > MIOPEN_IN_BLOCK_SZ
    int n_in_stride  = MIOPEN_N_IN_STRIDE;
    int c_in         = MIOPEN_C_IN;
    int h_in         = MIOPEN_H_IN;
    int w_in         = MIOPEN_W_IN;
#endif

    _FLOAT data[MIOPEN_READ_UNIT];
    _FLOAT response[MIOPEN_READ_UNIT];
#if MIOPEN_N_PIXS_OFF > 0
    if(x == MIOPEN_MAP_SZ_ALIGNED - 1)
    {
        int i = 0;
        for(; i < MIOPEN_N_PIXS_OFF; ++i)
        {
#if MIOPEN_N_IN_STRIDE > MIOPEN_IN_BLOCK_SZ
            if(n_in_stride > c_in * h_in * w_in && c_in != 0 && h_in != 0 && w_in != 0)
            {
                int loc, n_loc, c_loc, h_loc, w_loc;
                loc   = x * MIOPEN_READ_UNIT + i;
                n_loc = loc / (MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN);
                c_loc =
                    (loc % (MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN)) / (MIOPEN_H_IN * MIOPEN_W_IN);
                h_loc = ((loc % (MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN)) %
                         (MIOPEN_H_IN * MIOPEN_W_IN)) /
                        MIOPEN_W_IN;
                w_loc = ((loc % (MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN)) %
                         (MIOPEN_H_IN * MIOPEN_W_IN)) %
                        MIOPEN_W_IN;

                data[i] = bot[xOffset + n_loc * MIOPEN_N_IN_STRIDE + c_loc * MIOPEN_C_IN_STRIDE +
                              h_loc * MIOPEN_H_IN_STRIDE + w_loc * MIOPEN_W_IN_STRIDE];
            }
            else
#endif
            {
                data[i] = bot[xOffset + x * MIOPEN_READ_UNIT + i];
            }
        }
        for(; i < MIOPEN_READ_UNIT; ++i)
        {
            data[i] = (_FLOAT)1.f;
        }
    }
    else
#endif
    {
        for(int i = 0; i < MIOPEN_READ_UNIT; ++i)
        {
#if MIOPEN_N_IN_STRIDE > MIOPEN_IN_BLOCK_SZ
            if(n_in_stride > c_in * h_in * w_in && c_in != 0 && h_in != 0 && w_in != 0)
            {
                int loc, n_loc, c_loc, h_loc, w_loc;
                loc   = x * MIOPEN_READ_UNIT + i;
                n_loc = loc / (MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN);
                c_loc =
                    (loc % (MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN)) / (MIOPEN_H_IN * MIOPEN_W_IN);
                h_loc = ((loc % (MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN)) %
                         (MIOPEN_H_IN * MIOPEN_W_IN)) /
                        MIOPEN_W_IN;
                w_loc = ((loc % (MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN)) %
                         (MIOPEN_H_IN * MIOPEN_W_IN)) %
                        MIOPEN_W_IN;

                data[i] = bot[xOffset + n_loc * MIOPEN_N_IN_STRIDE + c_loc * MIOPEN_C_IN_STRIDE +
                              h_loc * MIOPEN_H_IN_STRIDE + w_loc * MIOPEN_W_IN_STRIDE];
            }
            else
#endif
            {
                data[i] = bot[xOffset + x * MIOPEN_READ_UNIT + i];
            }
        }
    }
    ActivationFunction(MIOPEN_READ_UNIT, response, (const _FLOAT*)data, gamma, beta, alpha);

#if MIOPEN_N_PIXS_OFF > 0
    if(x == MIOPEN_MAP_SZ_ALIGNED - 1)
    {
        int i = 0;
        for(; i < MIOPEN_N_PIXS_OFF; ++i)
        {
#if MIOPEN_N_OUT_STRIDE > MIOPEN_OUT_BLOCK_SZ
            if(n_out_stride > c_out * h_out * w_out && c_out != 0 && h_out != 0 && w_out != 0)
            {
                int loc, n_loc, c_loc, h_loc, w_loc;
                loc   = x * MIOPEN_READ_UNIT + i;
                n_loc = loc / (MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT);
                c_loc = (loc % (MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT)) /
                        (MIOPEN_H_OUT * MIOPEN_W_OUT);
                h_loc = ((loc % (MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT)) %
                         (MIOPEN_H_OUT * MIOPEN_W_OUT)) /
                        MIOPEN_W_OUT;
                w_loc = ((loc % (MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT)) %
                         (MIOPEN_H_OUT * MIOPEN_W_OUT)) %
                        MIOPEN_W_OUT;

                top[yOffset + n_loc * MIOPEN_N_OUT_STRIDE + c_loc * MIOPEN_C_OUT_STRIDE +
                    h_loc * MIOPEN_H_OUT_STRIDE + w_loc * MIOPEN_W_OUT_STRIDE] = response[i];
            }
            else
#endif
            {
                top[yOffset + x * MIOPEN_READ_UNIT + i] = response[i];
            }
        }
    }
    else
#endif
    {
        for(int i = 0; i < MIOPEN_READ_UNIT; ++i)
        {
#if MIOPEN_N_OUT_STRIDE > MIOPEN_OUT_BLOCK_SZ
            if(n_out_stride > c_out * h_out * w_out && c_out != 0 && h_out != 0 && w_out != 0)
            {
                int loc, n_loc, c_loc, h_loc, w_loc;
                loc   = x * MIOPEN_READ_UNIT + i;
                n_loc = loc / (MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT);
                c_loc = (loc % (MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT)) /
                        (MIOPEN_H_OUT * MIOPEN_W_OUT);
                h_loc = ((loc % (MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT)) %
                         (MIOPEN_H_OUT * MIOPEN_W_OUT)) /
                        MIOPEN_W_OUT;
                w_loc = ((loc % (MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT)) %
                         (MIOPEN_H_OUT * MIOPEN_W_OUT)) %
                        MIOPEN_W_OUT;

                top[yOffset + n_loc * MIOPEN_N_OUT_STRIDE + c_loc * MIOPEN_C_OUT_STRIDE +
                    h_loc * MIOPEN_H_OUT_STRIDE + w_loc * MIOPEN_W_OUT_STRIDE] = response[i];
            }
            else
#endif
            {
                top[yOffset + x * MIOPEN_READ_UNIT + i] = response[i];
            }
        }
    }
}

__attribute__((reqd_work_group_size(MIOPEN_NRN_GROUP_SZ0,
                                    MIOPEN_NRN_GROUP_SZ1,
                                    MIOPEN_NRN_GROUP_SZ2))) __kernel void
MIOpenNeuronBwd(__global _FLOAT* bot_diff,
                __global const _FLOAT* top_diff,
                __global const _FLOAT* bot_data,
                __global const _FLOAT* top_data,
                _FLOAT diff_scale,
                _FLOAT gamma,
                _FLOAT beta,
                _FLOAT alpha,
                const long dxOffset,
                const long dyOffset,
                const long xOffset,
                const long yOffset)
{
    int x             = get_global_id(0); // channel x

#if MIOPEN_N_OUT_STRIDE > MIOPEN_OUT_BLOCK_SZ || MIOPEN_N_DOUT_STRIDE > MIOPEN_DOUT_BLOCK_SZ || \
    MIOPEN_N_IN_STRIDE > MIOPEN_IN_BLOCK_SZ
    int n_out_stride  = MIOPEN_N_OUT_STRIDE;
    int c_out         = MIOPEN_C_OUT;
    int h_out         = MIOPEN_H_OUT;
    int w_out         = MIOPEN_W_OUT;
    int n_dout_stride = MIOPEN_N_DOUT_STRIDE;
    int c_dout        = MIOPEN_C_DOUT;
    int h_dout        = MIOPEN_H_DOUT;
    int w_dout        = MIOPEN_W_DOUT;
    int n_in_stride   = MIOPEN_N_IN_STRIDE;
    int c_in          = MIOPEN_C_IN;
    int h_in          = MIOPEN_H_IN;
    int w_in          = MIOPEN_W_IN;
#endif

#if MIOPEN_N_DIN_STRIDE > MIOPEN_DIN_BLOCK_SZ
    int n_din_stride  = MIOPEN_N_DIN_STRIDE;
    int c_din         = MIOPEN_C_DIN;
    int h_din         = MIOPEN_H_DIN;
    int w_din         = MIOPEN_W_DIN;
#endif

    _FLOAT bot_diff_dat[MIOPEN_READ_UNIT];
    _FLOAT top_diff_dat[MIOPEN_READ_UNIT];
    _FLOAT bot_dat[MIOPEN_READ_UNIT];
    _FLOAT top_dat[MIOPEN_READ_UNIT];
#if MIOPEN_N_PIXS_OFF > 0
    if(x == MIOPEN_MAP_SZ_ALIGNED - 1)
    {
        int i = 0;
        for(; i < MIOPEN_N_PIXS_OFF; ++i)
        {
#if MIOPEN_N_OUT_STRIDE > MIOPEN_OUT_BLOCK_SZ || MIOPEN_N_DOUT_STRIDE > MIOPEN_DOUT_BLOCK_SZ || \
    MIOPEN_N_IN_STRIDE > MIOPEN_IN_BLOCK_SZ
            if((n_out_stride > c_out * h_out * w_out || n_dout_stride > c_dout * h_dout * w_dout ||
                n_in_stride > c_in * h_in * w_in) &&
               c_out != 0 && h_out != 0 && w_out != 0 && c_dout != 0 && h_dout != 0 &&
               w_dout != 0 && c_in != 0 && h_in != 0 && w_in != 0)
            {
                int loc, n_loc_top_diff, c_loc_top_diff, h_loc_top_diff, w_loc_top_diff, n_loc_top,
                    c_loc_top, h_loc_top, w_loc_top, n_loc_bot, c_loc_bot, h_loc_bot, w_loc_bot;
                loc = x * MIOPEN_READ_UNIT + i;

                n_loc_top_diff = loc / (MIOPEN_C_DOUT * MIOPEN_H_DOUT * MIOPEN_W_DOUT);
                c_loc_top_diff = (loc % (MIOPEN_C_DOUT * MIOPEN_H_DOUT * MIOPEN_W_DOUT)) /
                                 (MIOPEN_H_DOUT * MIOPEN_W_DOUT);
                h_loc_top_diff = ((loc % (MIOPEN_C_DOUT * MIOPEN_H_DOUT * MIOPEN_W_DOUT)) %
                                  (MIOPEN_H_DOUT * MIOPEN_W_DOUT)) /
                                 MIOPEN_W_DOUT;
                w_loc_top_diff = ((loc % (MIOPEN_C_DOUT * MIOPEN_H_DOUT * MIOPEN_W_DOUT)) %
                                  (MIOPEN_H_DOUT * MIOPEN_W_DOUT)) %
                                 MIOPEN_W_DOUT;

                n_loc_top = loc / (MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT);
                c_loc_top = (loc % (MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT)) /
                            (MIOPEN_H_OUT * MIOPEN_W_OUT);
                h_loc_top = ((loc % (MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT)) %
                             (MIOPEN_H_OUT * MIOPEN_W_OUT)) /
                            MIOPEN_W_OUT;
                w_loc_top = ((loc % (MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT)) %
                             (MIOPEN_H_OUT * MIOPEN_W_OUT)) %
                            MIOPEN_W_OUT;

                n_loc_bot = loc / (MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN);
                c_loc_bot =
                    (loc % (MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN)) / (MIOPEN_H_IN * MIOPEN_W_IN);
                h_loc_bot = ((loc % (MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN)) %
                             (MIOPEN_H_IN * MIOPEN_W_IN)) /
                            MIOPEN_W_IN;
                w_loc_bot = ((loc % (MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN)) %
                             (MIOPEN_H_IN * MIOPEN_W_IN)) %
                            MIOPEN_W_IN;

                top_diff_dat[i] = top_diff[dyOffset + n_loc_top_diff * MIOPEN_N_DOUT_STRIDE +
                                           c_loc_top_diff * MIOPEN_C_DOUT_STRIDE +
                                           h_loc_top_diff * MIOPEN_H_DOUT_STRIDE +
                                           w_loc_top_diff * MIOPEN_W_DOUT_STRIDE];
                bot_dat[i] =
                    bot_data[xOffset + n_loc_bot * MIOPEN_N_IN_STRIDE +
                             c_loc_bot * MIOPEN_C_IN_STRIDE + h_loc_bot * MIOPEN_H_IN_STRIDE +
                             w_loc_bot * MIOPEN_W_IN_STRIDE];
                top_dat[i] =
                    top_data[yOffset + n_loc_top * MIOPEN_N_OUT_STRIDE +
                             c_loc_top * MIOPEN_C_OUT_STRIDE + h_loc_top * MIOPEN_H_OUT_STRIDE +
                             w_loc_top * MIOPEN_W_OUT_STRIDE];
            }
            else
#endif
            {
                top_diff_dat[i] = top_diff[dyOffset + x * MIOPEN_READ_UNIT + i];
                bot_dat[i]      = bot_data[xOffset + x * MIOPEN_READ_UNIT + i];
                top_dat[i]      = top_data[yOffset + x * MIOPEN_READ_UNIT + i];
            }
        }
        for(; i < MIOPEN_READ_UNIT; ++i)
        {
            top_diff_dat[i] = (_FLOAT)1.f;
            bot_dat[i]      = (_FLOAT)1.f;
            top_dat[i]      = (_FLOAT)1.f;
        }
    }
    else
#endif
    {
        for(int i = 0; i < MIOPEN_READ_UNIT; ++i)
        {
#if MIOPEN_N_OUT_STRIDE > MIOPEN_OUT_BLOCK_SZ || MIOPEN_N_DOUT_STRIDE > MIOPEN_DOUT_BLOCK_SZ || \
    MIOPEN_N_IN_STRIDE > MIOPEN_IN_BLOCK_SZ
            if((n_out_stride > c_out * h_out * w_out || n_dout_stride > c_dout * h_dout * w_dout ||
                n_in_stride > c_in * h_in * w_in) &&
               c_out != 0 && h_out != 0 && w_out != 0 && c_dout != 0 && h_dout != 0 &&
               w_dout != 0 && c_in != 0 && h_in != 0 && w_in != 0)
            {
                int loc, n_loc_top_diff, c_loc_top_diff, h_loc_top_diff, w_loc_top_diff, n_loc_top,
                    c_loc_top, h_loc_top, w_loc_top, n_loc_bot, c_loc_bot, h_loc_bot, w_loc_bot;
                loc = x * MIOPEN_READ_UNIT + i;

                n_loc_top_diff = loc / (MIOPEN_C_DOUT * MIOPEN_H_DOUT * MIOPEN_W_DOUT);
                c_loc_top_diff = (loc % (MIOPEN_C_DOUT * MIOPEN_H_DOUT * MIOPEN_W_DOUT)) /
                                 (MIOPEN_H_DOUT * MIOPEN_W_DOUT);
                h_loc_top_diff = ((loc % (MIOPEN_C_DOUT * MIOPEN_H_DOUT * MIOPEN_W_DOUT)) %
                                  (MIOPEN_H_DOUT * MIOPEN_W_DOUT)) /
                                 MIOPEN_W_DOUT;
                w_loc_top_diff = ((loc % (MIOPEN_C_DOUT * MIOPEN_H_DOUT * MIOPEN_W_DOUT)) %
                                  (MIOPEN_H_DOUT * MIOPEN_W_DOUT)) %
                                 MIOPEN_W_DOUT;

                n_loc_top = loc / (MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT);
                c_loc_top = (loc % (MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT)) /
                            (MIOPEN_H_OUT * MIOPEN_W_OUT);
                h_loc_top = ((loc % (MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT)) %
                             (MIOPEN_H_OUT * MIOPEN_W_OUT)) /
                            MIOPEN_W_OUT;
                w_loc_top = ((loc % (MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT)) %
                             (MIOPEN_H_OUT * MIOPEN_W_OUT)) %
                            MIOPEN_W_OUT;

                n_loc_bot = loc / (MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN);
                c_loc_bot =
                    (loc % (MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN)) / (MIOPEN_H_IN * MIOPEN_W_IN);
                h_loc_bot = ((loc % (MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN)) %
                             (MIOPEN_H_IN * MIOPEN_W_IN)) /
                            MIOPEN_W_IN;
                w_loc_bot = ((loc % (MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN)) %
                             (MIOPEN_H_IN * MIOPEN_W_IN)) %
                            MIOPEN_W_IN;

                top_diff_dat[i] = top_diff[dyOffset + n_loc_top_diff * MIOPEN_N_DOUT_STRIDE +
                                           c_loc_top_diff * MIOPEN_C_DOUT_STRIDE +
                                           h_loc_top_diff * MIOPEN_H_DOUT_STRIDE +
                                           w_loc_top_diff * MIOPEN_W_DOUT_STRIDE];
                bot_dat[i] =
                    bot_data[xOffset + n_loc_bot * MIOPEN_N_IN_STRIDE +
                             c_loc_bot * MIOPEN_C_IN_STRIDE + h_loc_bot * MIOPEN_H_IN_STRIDE +
                             w_loc_bot * MIOPEN_W_IN_STRIDE];
                top_dat[i] =
                    top_data[yOffset + n_loc_top * MIOPEN_N_OUT_STRIDE +
                             c_loc_top * MIOPEN_C_OUT_STRIDE + h_loc_top * MIOPEN_H_OUT_STRIDE +
                             w_loc_top * MIOPEN_W_OUT_STRIDE];
            }
            else
#endif
            {
                top_diff_dat[i] = top_diff[dyOffset + x * MIOPEN_READ_UNIT + i];
                bot_dat[i]      = bot_data[xOffset + x * MIOPEN_READ_UNIT + i];
                top_dat[i]      = top_data[yOffset + x * MIOPEN_READ_UNIT + i];
            }
        }
    }

    ActivationFunction_Diff(MIOPEN_READ_UNIT,
                            bot_diff_dat,
                            top_diff_dat,
                            bot_dat,
                            top_dat,
                            diff_scale,
                            gamma,
                            beta,
                            alpha);

#if MIOPEN_N_PIXS_OFF > 0
    if(x == MIOPEN_MAP_SZ_ALIGNED - 1)
    {
        int i = 0;
        for(; i < MIOPEN_N_PIXS_OFF; ++i)
        {
#if MIOPEN_N_DIN_STRIDE > MIOPEN_DIN_BLOCK_SZ
            if(n_din_stride > c_din * h_din * w_din && c_din != 0 && h_din != 0 && w_din != 0)
            {
                int loc, n_loc_bot_diff, c_loc_bot_diff, h_loc_bot_diff, w_loc_bot_diff;
                loc = x * MIOPEN_READ_UNIT + i;

                n_loc_bot_diff = loc / (MIOPEN_C_DIN * MIOPEN_H_DIN * MIOPEN_W_DIN);
                c_loc_bot_diff = (loc % (MIOPEN_C_DIN * MIOPEN_H_DIN * MIOPEN_W_DIN)) /
                                 (MIOPEN_H_DIN * MIOPEN_W_DIN);
                h_loc_bot_diff = ((loc % (MIOPEN_C_DIN * MIOPEN_H_DIN * MIOPEN_W_DIN)) %
                                  (MIOPEN_H_DIN * MIOPEN_W_DIN)) /
                                 MIOPEN_W_DIN;
                w_loc_bot_diff = ((loc % (MIOPEN_C_DIN * MIOPEN_H_DIN * MIOPEN_W_DIN)) %
                                  (MIOPEN_H_DIN * MIOPEN_W_DIN)) %
                                 MIOPEN_W_DIN;

                bot_diff[dxOffset + n_loc_bot_diff * MIOPEN_N_DIN_STRIDE +
                         c_loc_bot_diff * MIOPEN_C_DIN_STRIDE +
                         h_loc_bot_diff * MIOPEN_H_DIN_STRIDE +
                         w_loc_bot_diff * MIOPEN_W_DIN_STRIDE] = bot_diff_dat[i];
            }
            else
#endif
            {
                bot_diff[dxOffset + x * MIOPEN_READ_UNIT + i] = bot_diff_dat[i];
            }
        }
    }
    else
#endif
    {
        for(int i = 0; i < MIOPEN_READ_UNIT; ++i)
        {
#if MIOPEN_N_DIN_STRIDE > MIOPEN_DIN_BLOCK_SZ
            if(n_din_stride > c_din * h_din * w_din && c_din != 0 && h_din != 0 && w_din != 0)
            {
                int loc, n_loc_bot_diff, c_loc_bot_diff, h_loc_bot_diff, w_loc_bot_diff;
                loc = x * MIOPEN_READ_UNIT + i;

                n_loc_bot_diff = loc / (MIOPEN_C_DIN * MIOPEN_H_DIN * MIOPEN_W_DIN);
                c_loc_bot_diff = (loc % (MIOPEN_C_DIN * MIOPEN_H_DIN * MIOPEN_W_DIN)) /
                                 (MIOPEN_H_DIN * MIOPEN_W_DIN);
                h_loc_bot_diff = ((loc % (MIOPEN_C_DIN * MIOPEN_H_DIN * MIOPEN_W_DIN)) %
                                  (MIOPEN_H_DIN * MIOPEN_W_DIN)) /
                                 MIOPEN_W_DIN;
                w_loc_bot_diff = ((loc % (MIOPEN_C_DIN * MIOPEN_H_DIN * MIOPEN_W_DIN)) %
                                  (MIOPEN_H_DIN * MIOPEN_W_DIN)) %
                                 MIOPEN_W_DIN;

                bot_diff[dxOffset + n_loc_bot_diff * MIOPEN_N_DIN_STRIDE +
                         c_loc_bot_diff * MIOPEN_C_DIN_STRIDE +
                         h_loc_bot_diff * MIOPEN_H_DIN_STRIDE +
                         w_loc_bot_diff * MIOPEN_W_DIN_STRIDE] = bot_diff_dat[i];
            }
            else
#endif
            {
                bot_diff[dxOffset + x * MIOPEN_READ_UNIT + i] = bot_diff_dat[i];
            }
        }
    }
}

#endif // #ifdef LITE
