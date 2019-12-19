/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#ifndef MIOPEN_USE_FP16
#define MIOPEN_USE_FP16 0
#endif
#ifndef MIOPEN_USE_FP32
#define MIOPEN_USE_FP32 0
#endif

#if MIOPEN_USE_FP16 == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT half
#define EPSILON (_FLOAT)0.0001
#ifndef HALF_MAX
#define MAX_VAL 65504 /* max value */
#else
#define MAX_VAL HALF_MAX
#endif
#endif
#if MIOPEN_USE_FP32 == 1
#define _FLOAT float
#define EPSILON (_FLOAT)0.000001
#ifndef FLT_MAX
#define MAX_VAL 3.402823466e+38F /* max value */
#else
#define MAX_VAL FLT_MAX
#endif
#endif

#define _FLOAT_PREC _FLOAT
#define UNUSED __attribute__((__unused__))

#include "activation_functions.h"

#ifndef LSTM_FWD_HID
#define LSTM_FWD_HID 0
#endif
#ifndef LSTM_BWD_HID
#define LSTM_BWD_HID 0
#endif

#if LSTM_FWD_HID
#ifndef INFERENCE_MODE
#define INFERENCE_MODE 0
#endif

__kernel void LSTMFwdHidUpdate(const global _FLOAT* cx,
                               global _FLOAT* reservespace,
                               const long cx_offset,
                               const long i_offset,
                               const long f_offset,
                               const long o_offset,
                               const long c_offset,
                               const long cell_offset,
                               const long cell_offset_pre,
#if INFERENCE_MODE
                               UNUSED
#endif
                               const long activ_cell_offset,
                               const long hidden_offset,
                               const char use_cx,
                               const char is_seq_begin,
                               const int direction,
                               const int cur_batch,
                               const int use_batch)
{
    int total_item     = cur_batch * HY_H / RD_BLCK;
    total_item         = max(total_item, 1);
    _FLOAT activ_param = 1;

    _FLOAT s_dat[RD_BLCK];

    _FLOAT i_dat[RD_BLCK];
    _FLOAT f_dat[RD_BLCK];
    _FLOAT o_dat[RD_BLCK];
    _FLOAT c_dat[RD_BLCK];

    _FLOAT cx_dat[RD_BLCK];

    for(int gid = get_global_id(0); gid < total_item; gid += get_global_size(0))
    {
        int b_idx   = gid * RD_BLCK / HY_H;
        int h_idx   = gid * RD_BLCK - b_idx * HY_H;
        int rsv_idx = b_idx * HY_STRIDE + h_idx;

        *((READ_TYPE*)s_dat) = *((const global READ_TYPE*)(reservespace + i_offset + rsv_idx));
        ActivationFunction_Sigmoid(
            RD_BLCK, i_dat, (const _FLOAT*)s_dat, activ_param, activ_param, activ_param);

        *((READ_TYPE*)s_dat) = *((const global READ_TYPE*)(reservespace + f_offset + rsv_idx));
        ActivationFunction_Sigmoid(
            RD_BLCK, f_dat, (const _FLOAT*)s_dat, activ_param, activ_param, activ_param);

        *((READ_TYPE*)s_dat) = *((const global READ_TYPE*)(reservespace + o_offset + rsv_idx));
        ActivationFunction_Sigmoid(
            RD_BLCK, o_dat, (const _FLOAT*)s_dat, activ_param, activ_param, activ_param);

        *((READ_TYPE*)s_dat) = *((const global READ_TYPE*)(reservespace + c_offset + rsv_idx));
        ActivationFunction_TanH(
            RD_BLCK, c_dat, (const _FLOAT*)s_dat, activ_param, activ_param, activ_param);

        if((bool)is_seq_begin)
        {
            if((bool)use_cx)
            {
                *((READ_TYPE*)cx_dat) =
                    *((const global READ_TYPE*)(cx + cx_offset + gid * RD_BLCK));
            }
            else
            {
                for(int i = 0; i < RD_BLCK; ++i)
                {
                    cx_dat[i] = (_FLOAT)0;
                }
            }
        }
        else
        {
            if(b_idx < use_batch)
            {
                *((READ_TYPE*)cx_dat) =
                    *((const global READ_TYPE*)(reservespace + cell_offset_pre + rsv_idx));
            }
            else
            {
                if(direction == 1 && (bool)use_cx)
                {
                    *((READ_TYPE*)cx_dat) =
                        *((const global READ_TYPE*)(cx + cx_offset + gid * RD_BLCK));
                }
                else
                {
                    for(int i = 0; i < RD_BLCK; ++i)
                    {
                        cx_dat[i] = (_FLOAT)0;
                    }
                }
            }
        }

        for(int i = 0; i < RD_BLCK; ++i)
        {
            s_dat[i] = i_dat[i] * c_dat[i] + f_dat[i] * cx_dat[i];
        }
        ActivationFunction_TanH(RD_BLCK, cx_dat, s_dat, activ_param, activ_param, activ_param);

        *((global READ_TYPE*)(reservespace + i_offset + rsv_idx)) = *((READ_TYPE*)i_dat);
        *((global READ_TYPE*)(reservespace + f_offset + rsv_idx)) = *((READ_TYPE*)f_dat);
        *((global READ_TYPE*)(reservespace + o_offset + rsv_idx)) = *((READ_TYPE*)o_dat);
        *((global READ_TYPE*)(reservespace + c_offset + rsv_idx)) = *((READ_TYPE*)c_dat);

        *((global READ_TYPE*)(reservespace + cell_offset + rsv_idx)) = *((READ_TYPE*)s_dat);
#if !INFERENCE_MODE
        *((global READ_TYPE*)(reservespace + activ_cell_offset + b_idx * HY_STRIDE / 6 + h_idx)) =
            *((READ_TYPE*)cx_dat);
#endif
        for(int i = 0; i < RD_BLCK; ++i)
        {
            s_dat[i] = o_dat[i] * cx_dat[i];
        }

        *((global READ_TYPE*)(reservespace + hidden_offset + rsv_idx)) = *((READ_TYPE*)s_dat);
    }
}
#endif

#if LSTM_BWD_HID
__kernel void LSTMBwdHidUpdate(const global _FLOAT* cx,
                               const global _FLOAT* dcy,
                               global _FLOAT* reservespace,
                               global _FLOAT* workspace,
                               const long cx_offset,
                               const long dcy_offset,
                               const long i_offset,
                               const long f_offset,
                               const long o_offset,
                               const long c_offset,
                               const long activ_cell_offset,
                               const long cell_offset_pre,
                               const long di_offset,
                               const long df_offset,
                               const long do_offset,
                               const long dc_offset,
                               const long dcell_offset,
                               const long dcell_offset_pre,
                               const long dhidden_offset,
                               const long f_offset_pre,
                               const char use_cx,
                               const char use_dcy,
                               const char is_seq_begin,
                               const char is_seq_end,
                               const int direction,
                               const int cur_batch,
                               const int use_batch,
                               const int use_batch2)
{
    int total_item     = cur_batch * HY_H / RD_BLCK;
    total_item         = max(total_item, 1);
    _FLOAT activ_param = 1;

    _FLOAT dh_dat[RD_BLCK];

    _FLOAT s_dat[RD_BLCK];

    _FLOAT i_dat[RD_BLCK];
    _FLOAT f_dat[RD_BLCK];
    _FLOAT o_dat[RD_BLCK];
    _FLOAT c_dat[RD_BLCK];

    _FLOAT di_dat[RD_BLCK];
    _FLOAT df_dat[RD_BLCK];
    _FLOAT do_dat[RD_BLCK];
    _FLOAT dc_dat[RD_BLCK];

    _FLOAT cx_dat[RD_BLCK];
    _FLOAT dcx_dat[RD_BLCK];

    for(int gid = get_global_id(0); gid < total_item; gid += get_global_size(0))
    {
        int b_idx   = gid * RD_BLCK / HY_H;
        int h_idx   = gid * RD_BLCK - b_idx * HY_H;
        int rsv_idx = b_idx * HY_STRIDE + h_idx;

        *((READ_TYPE*)dh_dat) = *((const global READ_TYPE*)(workspace + dhidden_offset + rsv_idx));
        *((READ_TYPE*)o_dat)  = *((const global READ_TYPE*)(reservespace + o_offset + rsv_idx));

        *((READ_TYPE*)i_dat) = *((const global READ_TYPE*)(reservespace + i_offset + rsv_idx));

        *((READ_TYPE*)c_dat) = *((const global READ_TYPE*)(reservespace + c_offset + rsv_idx));

        for(int i = 0; i < RD_BLCK; ++i)
        {
            s_dat[i] = dh_dat[i] * o_dat[i];
        }

        *((READ_TYPE*)cx_dat) = *((const global READ_TYPE*)(reservespace + activ_cell_offset +
                                                            b_idx * HY_STRIDE / 6 + h_idx));

        ActivationFunction_TanH_Diff(RD_BLCK,
                                     dcx_dat,
                                     s_dat,
                                     cx_dat,
                                     cx_dat,
                                     activ_param,
                                     activ_param,
                                     activ_param,
                                     activ_param);

        if((bool)is_seq_end)
        {
            if((bool)use_dcy)
            {
                *((READ_TYPE*)s_dat) =
                    *((const global READ_TYPE*)(dcy + dcy_offset + gid * RD_BLCK));

                for(int i = 0; i < RD_BLCK; ++i)
                {
                    dcx_dat[i] += s_dat[i];
                }
            }
        }
        else
        {
            if(b_idx < use_batch)
            {
                *((READ_TYPE*)s_dat) =
                    *((const global READ_TYPE*)(workspace + dcell_offset_pre + rsv_idx));
                *((READ_TYPE*)f_dat) =
                    *((const global READ_TYPE*)(reservespace + f_offset_pre + rsv_idx));

                for(int i = 0; i < RD_BLCK; ++i)
                {
                    dcx_dat[i] += s_dat[i] * f_dat[i];
                }
            }
            else
            {
                if(direction == 0 && (bool)use_dcy)
                {
                    *((READ_TYPE*)s_dat) =
                        *((const global READ_TYPE*)(dcy + dcy_offset + gid * RD_BLCK));

                    for(int i = 0; i < RD_BLCK; ++i)
                    {
                        dcx_dat[i] += s_dat[i];
                    }
                }
            }
        }

        if((bool)is_seq_begin)
        {
            if((bool)use_cx)
            {
                *((READ_TYPE*)df_dat) =
                    *((const global READ_TYPE*)(cx + cx_offset + gid * RD_BLCK));

                for(int i = 0; i < RD_BLCK; ++i)
                {
                    df_dat[i] *= dcx_dat[i];
                }
            }
            else
            {
                for(int i = 0; i < RD_BLCK; ++i)
                {
                    df_dat[i] = (_FLOAT)0;
                }
            }
        }
        else
        {
            if(b_idx < use_batch2)
            {
                *((READ_TYPE*)df_dat) =
                    *((const global READ_TYPE*)(reservespace + cell_offset_pre + rsv_idx));

                for(int i = 0; i < RD_BLCK; ++i)
                {
                    df_dat[i] *= dcx_dat[i];
                }
            }
            else
            {
                if(direction == 1 && (bool)use_cx)
                {
                    *((READ_TYPE*)df_dat) =
                        *((const global READ_TYPE*)(cx + cx_offset + gid * RD_BLCK));

                    for(int i = 0; i < RD_BLCK; ++i)
                    {
                        df_dat[i] *= dcx_dat[i];
                    }
                }
                else
                {
                    for(int i = 0; i < RD_BLCK; ++i)
                    {
                        df_dat[i] = (_FLOAT)0;
                    }
                }
            }
        }

        *((READ_TYPE*)f_dat) = *((const global READ_TYPE*)(reservespace + f_offset + rsv_idx));

        ActivationFunction_Sigmoid_Diff(RD_BLCK,
                                        s_dat,
                                        df_dat,
                                        f_dat,
                                        f_dat,
                                        activ_param,
                                        activ_param,
                                        activ_param,
                                        activ_param);

        *((global READ_TYPE*)(workspace + df_offset + rsv_idx)) = *((READ_TYPE*)s_dat);

        for(int i = 0; i < RD_BLCK; ++i)
        {
            di_dat[i] = c_dat[i] * dcx_dat[i];
        }

        ActivationFunction_Sigmoid_Diff(RD_BLCK,
                                        s_dat,
                                        di_dat,
                                        i_dat,
                                        i_dat,
                                        activ_param,
                                        activ_param,
                                        activ_param,
                                        activ_param);

        *((global READ_TYPE*)(workspace + di_offset + rsv_idx)) = *((READ_TYPE*)s_dat);

        for(int i = 0; i < RD_BLCK; ++i)
        {
            do_dat[i] = cx_dat[i] * dh_dat[i];
        }

        ActivationFunction_Sigmoid_Diff(RD_BLCK,
                                        s_dat,
                                        do_dat,
                                        o_dat,
                                        o_dat,
                                        activ_param,
                                        activ_param,
                                        activ_param,
                                        activ_param);

        *((global READ_TYPE*)(workspace + do_offset + rsv_idx)) = *((READ_TYPE*)s_dat);

        for(int i = 0; i < RD_BLCK; ++i)
        {
            dc_dat[i] = i_dat[i] * dcx_dat[i];
        }

        ActivationFunction_TanH_Diff(RD_BLCK,
                                     s_dat,
                                     dc_dat,
                                     c_dat,
                                     c_dat,
                                     activ_param,
                                     activ_param,
                                     activ_param,
                                     activ_param);

        *((global READ_TYPE*)(workspace + dc_offset + rsv_idx)) = *((READ_TYPE*)s_dat);

        *((global READ_TYPE*)(workspace + dcell_offset + rsv_idx))   = *((READ_TYPE*)dcx_dat);
        *((global READ_TYPE*)(workspace + dhidden_offset + rsv_idx)) = *((READ_TYPE*)dh_dat);
    }
}
#endif
