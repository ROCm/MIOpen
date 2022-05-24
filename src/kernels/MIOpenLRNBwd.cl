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

#define DBG_RANGE 0
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

#define UNUSED __attribute__((__unused__))

#define MLO_LRN_GROUP_SZ2 1
#define MLO_LRN_STRIDE 1

#define MLO_LRN_LCL_DATA_WIDTH (MLO_LRN_GROUP_SZ0 * MLO_LRN_N_HORIZ_OUT_PIX + MLO_LRN_KERNEL_SZ - 1)
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

struct LRNBackwardParam
{
    _FLOAT ratio;
    _FLOAT alpha;
    _FLOAT beta;
};

/*

This is a naive implementation.
The "sliding window" -based implementation is in MIOpenLRNFwd.cl file

*/

__attribute__((reqd_work_group_size(MLO_LRN_GROUP_SZ0, MLO_LRN_GROUP_SZ1, MLO_LRN_GROUP_SZ2)))
__kernel void
MIOpenLRNWithinChannelBwd(const __global _FLOAT* top,
                          const __global _FLOAT* bot,
                          const __global _FLOAT* top_df,
                          const __global _FLOAT* scale,
                          __global _FLOAT* bot_df,
                          UNUSED _FLOAT ratio,
                          _FLOAT alpha,
                          _FLOAT beta)
{
    __local _FLOAT top_df_data[MLO_LRN_LCL_DATA_WIDTH * MLO_LRN_LCL_DATA_HEIGHT];
    __local _FLOAT ratio_data[MLO_LRN_LCL_DATA_WIDTH * MLO_LRN_LCL_DATA_HEIGHT];
    int x          = get_group_id(0) * MLO_LRN_GROUP_SZ0 * MLO_LRN_N_HORIZ_OUT_PIX;
    int y          = get_group_id(1) * MLO_LRN_GROUP_SZ1 * MLO_LRN_N_VERT_OUT_PIX;
    int lcl_id0    = get_local_id(0);
    int lcl_id1    = get_local_id(1);
    int ob         = get_global_id(2); // output * batch_sz
    int o          = ob / MLO_LRN_BATCH_SZ;
    int b          = ob - o * MLO_LRN_BATCH_SZ;
    int top_x      = x;
    int top_y      = y;
    int top_df_off = b * MLO_LRN_TOPDF_BATCH_STRIDE + o * MLO_LRN_TOPDF_CHANNEL_STRIDE;
    int scale_off  = b * MLO_LRN_SCALE_BATCH_STRIDE + o * MLO_LRN_SCALE_CHANNEL_STRIDE;
    int bot_x      = x + lcl_id0 * MLO_LRN_N_HORIZ_OUT_PIX;
    int bot_y      = y + lcl_id1 * MLO_LRN_N_VERT_OUT_PIX;

    _FLOAT prv_exp_scale[MLO_LRN_N_VERT_OUT_PIX][MLO_LRN_N_HORIZ_OUT_PIX];
    //		_FLOAT prv_top_df[MLO_LRN_N_VERT_OUT_PIX][MLO_LRN_N_HORIZ_OUT_PIX];

    // load top_diff and scale tiles
    for(int b_j = lcl_id1; b_j < MLO_LRN_LCL_DATA_HEIGHT; b_j += MLO_LRN_GROUP_SZ1)
    {
        int top_y_act = top_y + b_j - MLO_LRN_PAD;

        bool invisibleY = (top_y_act < 0) || (top_y_act >= MLO_LRN_TOP_HEIGHT);

        top_y_act = (invisibleY) ? 0 : top_y_act;

        int top_df_y_off = top_y_act * MLO_LRN_TOPDF_STRIDE;
        int scale_y_off  = top_y_act * MLO_LRN_SCALE_STRIDE;

        int lcl_off_v = b_j * MLO_LRN_LCL_DATA_WIDTH;

        for(int b_i = lcl_id0; b_i < MLO_LRN_LCL_DATA_WIDTH; b_i += MLO_LRN_GROUP_SZ0)
        {

            int top_x_act = top_x + b_i - MLO_LRN_PAD;

            bool invisibleX = (top_x_act < 0) || (top_x_act >= MLO_LRN_TOP_WIDTH);

            top_x_act = (invisibleX) ? 0 : top_x_act;
#if DBG_RANGE
            if(top_df_off + top_df_y_off + top_x_act >=
               MLO_LRN_BATCH_SZ * MLO_LRN_TOPDF_BATCH_STRIDE)
            {
                printf("K:err:topdf-off_range\n");
            }
#endif
            _FLOAT top_df_val = top_df[top_df_off + top_df_y_off + top_x_act];
            _FLOAT scale_val  = scale[scale_off + scale_y_off + top_x_act];

            top_df_val = (invisibleX || invisibleY) ? 0 : top_df_val;
            scale_val  = (invisibleX || invisibleY) ? (_FLOAT)1.f : scale_val;

            top_df_data[lcl_off_v + b_i] = top_df_val;
            ratio_data[lcl_off_v + b_i]  = scale_val;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // actual top_diffs and scales
    for(int j = 0; j < MLO_LRN_N_VERT_OUT_PIX; ++j)
    {
        int lcl_off_v =
            (lcl_id1 * MLO_LRN_N_VERT_OUT_PIX + MLO_LRN_PAD + j) * MLO_LRN_LCL_DATA_WIDTH;
        for(int i = 0; i < MLO_LRN_N_HORIZ_OUT_PIX; i++)
        {
            _FLOAT scale_ratio =
                ratio_data[lcl_off_v + lcl_id0 * MLO_LRN_N_HORIZ_OUT_PIX + MLO_LRN_PAD + i];
            prv_exp_scale[j][i] = exp(-beta * log(scale_ratio));
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    // read top and load ratio tile
    int top_off = b * MLO_LRN_TOP_BATCH_STRIDE + o * MLO_LRN_TOP_CHANNEL_STRIDE;
    for(int b_j = lcl_id1; b_j < MLO_LRN_LCL_DATA_HEIGHT; b_j += MLO_LRN_GROUP_SZ1)
    {
        int top_y_act = top_y + b_j - MLO_LRN_PAD;

        bool invisibleY = (top_y_act < 0) || (top_y_act >= MLO_LRN_TOP_HEIGHT);

        top_y_act = (invisibleY) ? 0 : top_y_act;

        int top_y_off = top_y_act * MLO_LRN_TOP_STRIDE;

        int lcl_off_v = b_j * MLO_LRN_LCL_DATA_WIDTH;

        for(int b_i = lcl_id0; b_i < MLO_LRN_LCL_DATA_WIDTH; b_i += MLO_LRN_GROUP_SZ0)
        {

            int top_x_act = top_x + b_i - MLO_LRN_PAD;

            bool invisibleX = (top_x_act < 0) || (top_x_act >= MLO_LRN_TOP_WIDTH);

            top_x_act = (invisibleX) ? 0 : top_x_act;
#if DBG_RANGE

            if(top_off + top_y_off + top_x_act >= MLO_LRN_BATCH_SZ * MLO_LRN_TOP_BATCH_STRIDE)
            {
                printf("K:err:top-off_range\n");
            }
#endif

            _FLOAT top_val = top[top_off + top_y_off + top_x_act];

            top_val = (invisibleX || invisibleY) ? 0 : top_val;

            _FLOAT top_df_val = top_df_data[lcl_off_v + b_i];

            _FLOAT scale_val = ratio_data[lcl_off_v + b_i];

            // scale val is not 0
            _FLOAT ratio_dta = (top_df_val * top_val) / scale_val;
            // replacing scale with ratio
            ratio_data[lcl_off_v + b_i] = ratio_dta;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // caculate bot diff
    _FLOAT prv_bot_diff[MLO_LRN_N_VERT_OUT_PIX][MLO_LRN_N_HORIZ_OUT_PIX];

    for(int j = 0; j < MLO_LRN_N_VERT_OUT_PIX; ++j)
    {
        int v_off_v = (lcl_id1 * MLO_LRN_N_VERT_OUT_PIX + j);
        int hstart  = y + v_off_v - MLO_LRN_PAD;
        int hend    = min(hstart + MLO_LRN_KERNEL_SZ, MLO_LRN_TOP_HEIGHT + MLO_LRN_PRE_PAD);

        // accum offset, vertical
        //			int lcl_a_off_v = v_off_v *  MLO_LRN_LCL_DATA_WIDTH;
        // value offset, vertical
        int lcl_v_off_v = (v_off_v + MLO_LRN_PAD) * MLO_LRN_LCL_DATA_WIDTH;
        for(int i = 0; i < MLO_LRN_N_HORIZ_OUT_PIX; i++)
        {
            _FLOAT prv_ratio_accum = (_FLOAT)0;
            int v_off_h            = lcl_id0 * MLO_LRN_N_HORIZ_OUT_PIX + i;

            int wstart = x + v_off_h - MLO_LRN_PAD;
            int wend   = min(wstart + MLO_LRN_KERNEL_SZ, MLO_LRN_TOP_WIDTH + MLO_LRN_PRE_PAD);

            int adj_area_size = (hend - hstart) * (wend - wstart);

            // accum offset, horiz
            int lcl_a_off_h = v_off_h;
            //	value offset, horiz
            int lcl_v_off_h = lcl_a_off_h + MLO_LRN_PAD;

            for(int k = 0; k < MLO_LRN_KERNEL_SZ; k++)
            {
                for(int l = 0; l < MLO_LRN_KERNEL_SZ; l++)
                {
                    prv_ratio_accum +=
                        ratio_data[(v_off_v + k) * MLO_LRN_LCL_DATA_WIDTH + lcl_a_off_h + l];
                }
            }

            _FLOAT top_df_val = top_df_data[lcl_v_off_v + lcl_v_off_h];

            uint bot_off0 = MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE * o +
                            MLO_LRN_BOT_STRIDE * (y + v_off_v) + x + v_off_h;

            uint bot_off = (y + v_off_v < MLO_LRN_BOT_HEIGHT && x + v_off_h < MLO_LRN_BOT_WIDTH &&
                            b < MLO_LRN_BATCH_SZ && o < MLO_LRN_N_OUTPUTS)
                               ? bot_off0
                               : MLO_LRN_BATCH_SZ * MLO_LRN_BOT_BATCH_STRIDE - 1;
#if DBG_RANGE

            if(bot_off >= MLO_LRN_BATCH_SZ * MLO_LRN_BOT_BATCH_STRIDE)
            {
                printf("K:err:bot-off_range\n");
            }
#endif
            _FLOAT bot_dta = bot[bot_off];

            bot_dta = (y + v_off_v < MLO_LRN_BOT_HEIGHT && x + v_off_h < MLO_LRN_BOT_WIDTH &&
                       b < MLO_LRN_BATCH_SZ && o < MLO_LRN_N_OUTPUTS)
                          ? bot_dta
                          : 0;

            _FLOAT adj_ratio       = (_FLOAT)2.f * alpha * beta / adj_area_size;
            _FLOAT prv_accum_ratio = adj_ratio * bot_dta * prv_ratio_accum;
            prv_bot_diff[j][i]     = prv_exp_scale[j][i] * top_df_val - prv_accum_ratio;
        }
    }

    for(int j = 0; j < MLO_LRN_N_VERT_OUT_PIX; j++)
    {
        for(int i = 0; i < MLO_LRN_N_HORIZ_OUT_PIX; i++)
        {
            if(bot_y + j < MLO_LRN_BOT_HEIGHT && bot_x + i < MLO_LRN_BOT_WIDTH &&
               b < MLO_LRN_BATCH_SZ && o < MLO_LRN_N_OUTPUTS)
            {
#if DBG_RANGE

                if(MLO_LRN_BOTDF_BATCH_STRIDE * b + MLO_LRN_BOTDF_CHANNEL_STRIDE * o +
                       MLO_LRN_BOTDF_STRIDE * (bot_y + j) + bot_x + i >=
                   MLO_LRN_BATCH_SZ * MLO_LRN_BOTDF_BATCH_STRIDE)
                {
                    printf("K:err:botdf-off_range\n");
                }
#endif
                bot_df[MLO_LRN_BOTDF_BATCH_STRIDE * b + MLO_LRN_BOTDF_CHANNEL_STRIDE * o +
                       MLO_LRN_BOTDF_STRIDE * (bot_y + j) + bot_x + i] = prv_bot_diff[j][i];
            }
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
MIOpenLRNAcrossChannelsBwd1(const __global _FLOAT* top,
                            const __global _FLOAT* bot,
                            const __global _FLOAT* top_df,
                            const __global _FLOAT* scale,
                            __global _FLOAT* bot_df,
                            _FLOAT ratio,
                            UNUSED _FLOAT alpha,
                            _FLOAT beta)
{
    int x              = get_global_id(0); // channel x
    int y              = get_global_id(1); // channel y
    int b              = get_global_id(2); // batch
    _FLOAT accum_ratio = 0;
    _FLOAT top_df_in[MLO_LRN_KERNEL_SZ];
    _FLOAT scale_in[MLO_LRN_KERNEL_SZ];
    _FLOAT ratio_dta[MLO_LRN_KERNEL_SZ];
    int c_i = 0, c_o = 0;
    int bot_df_off = 0;

    for(c_i = 0; c_i < MLO_LRN_PRE_PAD; c_i++)
    {

        top_df_in[c_i] = top_df[MLO_LRN_TOPDF_BATCH_STRIDE * b +
                                MLO_LRN_TOPDF_CHANNEL_STRIDE * c_i + MLO_LRN_TOPDF_STRIDE * y + x];
        scale_in[c_i]  = scale[MLO_LRN_SCALE_BATCH_STRIDE * b + MLO_LRN_SCALE_CHANNEL_STRIDE * c_i +
                              MLO_LRN_SCALE_STRIDE * y + x];
        _FLOAT top_dta = top[MLO_LRN_TOP_BATCH_STRIDE * b + MLO_LRN_TOP_CHANNEL_STRIDE * c_i +
                             MLO_LRN_TOP_STRIDE * y + x];

        ratio_dta[c_i] = (top_df_in[c_i] * top_dta) / scale_in[c_i];

#if MLO_LOW_CHNL_COUNT == 1
        ratio_dta[c_i] = (c_i < MLO_LRN_N_OUTPUTS) ? ratio_dta[c_i] : 0;
#endif

        accum_ratio = accum_ratio + ratio_dta[c_i];
    }

    for(; c_i < MLO_LRN_KERNEL_SZ; c_i++, c_o++)
    {
        top_df_in[c_i] = top_df[MLO_LRN_TOPDF_BATCH_STRIDE * b +
                                MLO_LRN_TOPDF_CHANNEL_STRIDE * c_i + MLO_LRN_TOPDF_STRIDE * y + x];
        scale_in[c_i]  = scale[MLO_LRN_SCALE_BATCH_STRIDE * b + MLO_LRN_SCALE_CHANNEL_STRIDE * c_i +
                              MLO_LRN_SCALE_STRIDE * y + x];
        _FLOAT top_dta = top[MLO_LRN_TOP_BATCH_STRIDE * b + MLO_LRN_TOP_CHANNEL_STRIDE * c_i +
                             MLO_LRN_TOP_STRIDE * y + x];
        ratio_dta[c_i] = (top_df_in[c_i] * top_dta) / scale_in[c_i];
#if MLO_LOW_CHNL_COUNT == 1
        ratio_dta[c_i] = (c_i < MLO_LRN_N_OUTPUTS) ? ratio_dta[c_i] : 0;
#endif

        accum_ratio = accum_ratio + ratio_dta[c_i];
#if MLO_LOW_CHNL_COUNT == 1
        if(c_o < MLO_LRN_N_INPUTS)
#endif
        {
            _FLOAT bot_dta = bot[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE * c_o +
                                 MLO_LRN_BOT_STRIDE * y + x];

            _FLOAT prv_scale = scale_in[c_o];

            _FLOAT exp_scale = exp(-beta * log(prv_scale));
            //					pow(prv_scale, -beta);

            _FLOAT prv_accum_ratio = ratio * bot_dta * accum_ratio;

            _FLOAT out_val = top_df_in[c_o] * exp_scale - prv_accum_ratio;

            bot_df_off = MLO_LRN_BOTDF_BATCH_STRIDE * b + MLO_LRN_BOTDF_CHANNEL_STRIDE * c_o +
                         MLO_LRN_BOTDF_STRIDE * y + x;

            bot_df[bot_df_off] = out_val;
        }
    }

    for(; c_i < MLO_LRN_N_INPUTS; c_i++, c_o++)
    {

        _FLOAT prv_top_df_in =
            top_df[MLO_LRN_TOPDF_BATCH_STRIDE * b + MLO_LRN_TOPDF_CHANNEL_STRIDE * c_i +
                   MLO_LRN_TOPDF_STRIDE * y + x];
        _FLOAT prv_scale_in =
            scale[MLO_LRN_SCALE_BATCH_STRIDE * b + MLO_LRN_SCALE_CHANNEL_STRIDE * c_i +
                  MLO_LRN_SCALE_STRIDE * y + x];
        _FLOAT top_dta       = top[MLO_LRN_TOP_BATCH_STRIDE * b + MLO_LRN_TOP_CHANNEL_STRIDE * c_i +
                             MLO_LRN_TOP_STRIDE * y + x];
        _FLOAT prv_ratio_dta = prv_top_df_in * top_dta / prv_scale_in;
#if MLO_LOW_CHNL_COUNT == 1
        prv_ratio_dta = (c_i < MLO_LRN_N_OUTPUTS) ? prv_ratio_dta : 0;
#endif

        accum_ratio = accum_ratio + prv_ratio_dta;

        accum_ratio = accum_ratio - ratio_dta[0];

        for(int i = 0; i < MLO_LRN_KERNEL_SZ - 1; i++)
        {
            top_df_in[i] = top_df_in[i + 1];
            scale_in[i]  = scale_in[i + 1];
            ratio_dta[i] = ratio_dta[i + 1];
        }

        top_df_in[MLO_LRN_KERNEL_SZ - 1] = prv_top_df_in;
        scale_in[MLO_LRN_KERNEL_SZ - 1]  = prv_scale_in;
        ratio_dta[MLO_LRN_KERNEL_SZ - 1] = prv_ratio_dta;

#if MLO_LOW_CHNL_COUNT == 1
        if(c_o < MLO_LRN_N_INPUTS)
#endif
        {
            _FLOAT bot_dta = bot[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE * c_o +
                                 MLO_LRN_BOT_STRIDE * y + x];

            _FLOAT prv_scale = scale_in[MLO_LRN_PAD];

            _FLOAT exp_scale = exp(-beta * log(prv_scale));
            //				pow(prv_scale,-beta);

            _FLOAT prv_accum_ratio = ratio * bot_dta * accum_ratio;

            _FLOAT out_val = top_df_in[MLO_LRN_PAD] * exp_scale - prv_accum_ratio;

            bot_df_off = MLO_LRN_BOTDF_BATCH_STRIDE * b + MLO_LRN_BOTDF_CHANNEL_STRIDE * c_o +
                         MLO_LRN_BOTDF_STRIDE * y + x;

            bot_df[bot_df_off] = out_val;
        }
    }

    for(; c_i < MLO_LRN_N_INPUTS + MLO_LRN_PRE_PAD; c_i++, c_o++)
    {

        accum_ratio = accum_ratio - ratio_dta[0];

        for(int i = 0; i < MLO_LRN_KERNEL_SZ - 1; i++)
        {
            top_df_in[i] = top_df_in[i + 1];
            scale_in[i]  = scale_in[i + 1];
            ratio_dta[i] = ratio_dta[i + 1];
        }

#if MLO_LOW_CHNL_COUNT == 1
        if(c_o < MLO_LRN_N_INPUTS)
#endif
        {
            _FLOAT bot_dta = bot[MLO_LRN_BOT_BATCH_STRIDE * b + MLO_LRN_BOT_CHANNEL_STRIDE * c_o +
                                 MLO_LRN_BOT_STRIDE * y + x];

            _FLOAT prv_scale = scale_in[MLO_LRN_PAD];

            _FLOAT exp_scale = exp(-beta * log(prv_scale));
            //				pow(prv_scale,-beta);

            _FLOAT prv_accum_ratio = ratio * bot_dta * accum_ratio;

            _FLOAT out_val = top_df_in[MLO_LRN_PAD] * exp_scale - prv_accum_ratio;

            bot_df_off = MLO_LRN_BOTDF_BATCH_STRIDE * b + MLO_LRN_BOTDF_CHANNEL_STRIDE * c_o +
                         MLO_LRN_BOTDF_STRIDE * y + x;

            bot_df[bot_df_off] = out_val;
        }
    }
}
