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

#ifndef MLO_POOLING_INDEX_TYPE
#error "MLO_POOLING_INDEX_TYPE not defined"
#else
typedef MLO_POOLING_INDEX_TYPE index_t;
#endif

#ifndef MLO_POOLING_INDEX_MAX
#error "MLO_POOLING_INDEX_MAX not defined"
#endif

#define MLO_POOLING_OP_AVE 0
#define MLO_POOLING_OP_MAX 1
#define MLO_POOLING_OP_STC 2

#define MLO_POOLBWD_GROUP_SZ2 1

#define MLO_POOLBWD_LCL_DATA_WIDTH                                                   \
    ((MLO_POOLBWD_GROUP_SZ0 * MLO_POOLBWD_N_HORIZ_OUT_PIX + MLO_POOLING_KERNEL_SZ0 + \
      MLO_POOLING_STRIDE0 - 2) /                                                     \
     MLO_POOLING_STRIDE0)
#define MLO_POOLBWD_LCL_DATA_HEIGHT                                                 \
    ((MLO_POOLBWD_GROUP_SZ1 * MLO_POOLBWD_N_VERT_OUT_PIX + MLO_POOLING_KERNEL_SZ1 + \
      MLO_POOLING_STRIDE1 - 2) /                                                    \
     MLO_POOLING_STRIDE1)

__attribute__((reqd_work_group_size(MLO_POOLBWD_GROUP_SZ0,
                                    MLO_POOLBWD_GROUP_SZ1,
                                    MLO_POOLBWD_GROUP_SZ2))) __kernel void
mloPoolingAveBwd(const __global _FLOAT* top_diff, __global _FLOAT* bot_diff)
{
    __local _FLOAT lcl_top_diff[MLO_POOLBWD_LCL_DATA_WIDTH * MLO_POOLBWD_LCL_DATA_HEIGHT];

    int x       = get_group_id(0) * MLO_POOLBWD_GROUP_SZ0 * MLO_POOLBWD_N_HORIZ_OUT_PIX;
    int y       = get_group_id(1) * MLO_POOLBWD_GROUP_SZ1 * MLO_POOLBWD_N_VERT_OUT_PIX;
    int lcl_id0 = get_local_id(0);
    int lcl_id1 = get_local_id(1);
    //		int lcl_id = (lcl_id1 << MLO_POOLBWD_GROUP_LG2SZ1) + lcl_id0;
    int ob = get_global_id(2); // outputs * batch_sz
    int b  = ob / MLO_POOLING_N_OUTPUTS;
    int o  = ob - b * MLO_POOLING_N_OUTPUTS;

    int top_x = (x + MLO_POOLING_PAD0 - MLO_POOLING_KERNEL_SZ0) < 0
                    ? 0
                    : (x + MLO_POOLING_PAD0 - MLO_POOLING_KERNEL_SZ0) / MLO_POOLING_STRIDE0 + 1;
    int top_y = (y + MLO_POOLING_PAD1 - MLO_POOLING_KERNEL_SZ1) < 0
                    ? 0
                    : (y + MLO_POOLING_PAD1 - MLO_POOLING_KERNEL_SZ1) / MLO_POOLING_STRIDE1 + 1;
    int top_off = b * MLO_POOLBWD_TOPDF_BATCH_STRIDE + o * MLO_POOLBWD_TOPDF_CHANNEL_STRIDE;

    _FLOAT res[MLO_POOLBWD_N_VERT_OUT_PIX][MLO_POOLBWD_N_HORIZ_OUT_PIX];
    for(int k = 0; k < MLO_POOLBWD_N_VERT_OUT_PIX; k++)
    {
        for(int l = 0; l < MLO_POOLBWD_N_HORIZ_OUT_PIX; l++)
        {
            res[k][l] = 0;
        }
    }

    // load tile
    for(int tj = lcl_id1; tj < MLO_POOLBWD_LCL_DATA_HEIGHT; tj += MLO_POOLBWD_GROUP_SZ1)
    {
        int top_y_act = top_y + tj;
        int top_y_off = top_y_act * MLO_POOLBWD_TOPDF_STRIDE;

        int lcl_off_v = tj * MLO_POOLBWD_LCL_DATA_WIDTH;

        bool invisibleY = (top_y_act >= MLO_POOLBWD_TOP_HEIGHT);

        for(int ti = lcl_id0; ti < MLO_POOLBWD_LCL_DATA_WIDTH; ti += MLO_POOLBWD_GROUP_SZ0)
        {

            int top_x_act = top_x + ti;

            bool invisibleX = (top_x_act >= MLO_POOLBWD_TOP_WIDTH);

            int top_diff_off = (invisibleX || invisibleY) ? 0 : top_off + top_y_off + top_x_act;

            _FLOAT top_val = top_diff[top_diff_off];

            top_val = (invisibleX || invisibleY) ? 0 : top_val;

            lcl_top_diff[lcl_off_v + ti] = top_val;
#if 0
				if (lcl_id1==0&&o==0&&b==0)
				{
				  printf("K:in: %d %d %d   %f\n", top_off + top_y_off + top_x_act, top_y_act, top_x_act, top_val);
				}
#endif
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int bot_y = (y + lcl_id1 * MLO_POOLBWD_N_VERT_OUT_PIX);
    int bot_x = (x + lcl_id0 * MLO_POOLBWD_N_HORIZ_OUT_PIX);

    for(int k = 0; k < MLO_POOLBWD_N_VERT_OUT_PIX; k++)
    {

        int h          = bot_y + k + MLO_POOLING_PAD1;
        int top_hstart = (h < MLO_POOLING_KERNEL_SZ1)
                             ? 0
                             : (h - MLO_POOLING_KERNEL_SZ1) / MLO_POOLING_STRIDE1 + 1;
        int top_hend = min(h / MLO_POOLING_STRIDE1 + 1, MLO_POOLBWD_TOP_HEIGHT);

        for(int l = 0; l < MLO_POOLBWD_N_HORIZ_OUT_PIX; l++)
        {

            int w          = bot_x + l + MLO_POOLING_PAD0;
            int top_wstart = (w < MLO_POOLING_KERNEL_SZ0)
                                 ? 0
                                 : (w - MLO_POOLING_KERNEL_SZ0) / MLO_POOLING_STRIDE0 + 1;
            int top_wend = min(w / MLO_POOLING_STRIDE0 + 1, MLO_POOLBWD_TOP_WIDTH);

            for(int top_h = top_hstart; top_h < top_hend; ++top_h)
            {
                int hstart = top_h * MLO_POOLING_STRIDE1 - MLO_POOLING_PAD1;
                int hend   = min(hstart + MLO_POOLING_KERNEL_SZ1, MLO_POOLBWD_BOT_HEIGHT);
                hstart     = max(hstart, 0);

                for(int top_w = top_wstart; top_w < top_wend; ++top_w)
                {
                    // figure out the pooling size
                    int wstart = top_w * MLO_POOLING_STRIDE0 - MLO_POOLING_PAD0;
                    int wend   = min(wstart + MLO_POOLING_KERNEL_SZ0, MLO_POOLBWD_BOT_WIDTH);
                    wstart     = max(wstart, 0);
                    int pool_size =
#ifdef MLO_POOLING_OP_AVE_INCLUSIVE
                        MLO_POOLING_KERNEL_SZ0 * MLO_POOLING_KERNEL_SZ1;
                    (void)wend;
                    (void)hend;
#else
                        (hend - hstart) * (wend - wstart);
#endif
                    pool_size     = (pool_size == 0) ? 1 : pool_size;
                    int lcl_top_h = top_h - top_y;
                    int lcl_top_w = top_w - top_x;
                    _FLOAT add_val =
                        (lcl_top_diff[lcl_top_h * MLO_POOLBWD_LCL_DATA_WIDTH + lcl_top_w] /
                         (_FLOAT)pool_size);
                    res[k][l] += add_val;
#if 0
				if (bot_x+l==6&&bot_y+k==0&&o==3&&b==0)
				{
				  printf("K:com: %d %d %d %d %d %d   %10.8f %10.8f %10.8f %d\n", k,l,top_h, top_w, lcl_top_h, lcl_top_w, res[k][l], add_val, lcl_top_diff[lcl_top_h *  MLO_POOLBWD_LCL_DATA_WIDTH + lcl_top_w], pool_size);
				}
#endif
                }
            }
        }
    }

    int bot_off = b * MLO_POOLBWD_BOTDF_BATCH_STRIDE + o * MLO_POOLBWD_BOTDF_CHANNEL_STRIDE +
                  bot_y * MLO_POOLBWD_BOTDF_STRIDE + bot_x;
    for(int k = 0; k < MLO_POOLBWD_N_VERT_OUT_PIX; k++)
    {
        for(int l = 0; l < MLO_POOLBWD_N_HORIZ_OUT_PIX; l++)
        {
            if(bot_y + k < MLO_POOLBWD_BOT_HEIGHT && bot_x + l < MLO_POOLBWD_BOT_WIDTH)
            {
                bot_diff[bot_off + k * MLO_POOLBWD_BOTDF_STRIDE + l] = res[k][l];
#if 0
					if (lcl_id0==0&&lcl_id1==0&&o==0&&b==0)
					{
						printf("K:out: %d %d %d  %f\n", bot_off + k * MLO_POOLBWD_BOTDF_STRIDE +l, k, l, bot_diff[bot_off + k * MLO_POOLBWD_BOT_STRIDE +l]);
					}
#endif
            }
        }
    }
}

__attribute__((reqd_work_group_size(MLO_POOLBWD_GROUP_SZ0,
                                    MLO_POOLBWD_GROUP_SZ1,
                                    MLO_POOLBWD_GROUP_SZ2))) __kernel void
mloPoolingMaxBwd(const __global _FLOAT* top_df, __global _FLOAT* bot_df, __global index_t* mask)
{
    __local _FLOAT lcl_top_df[MLO_POOLBWD_LCL_DATA_WIDTH * MLO_POOLBWD_LCL_DATA_HEIGHT];
    __local index_t lcl_mask[MLO_POOLBWD_LCL_DATA_WIDTH * MLO_POOLBWD_LCL_DATA_HEIGHT];

    int gid0    = get_group_id(0);
    int gid1    = get_group_id(1);
    int x       = gid0 * MLO_POOLBWD_GROUP_SZ0 * MLO_POOLBWD_N_HORIZ_OUT_PIX;
    int y       = gid1 * MLO_POOLBWD_GROUP_SZ1 * MLO_POOLBWD_N_VERT_OUT_PIX;
    int lcl_id0 = get_local_id(0);
    int lcl_id1 = get_local_id(1);
    int ob      = get_global_id(2); // outputs * batch_sz
    int b       = ob / MLO_POOLING_N_OUTPUTS;
    int o       = ob - b * MLO_POOLING_N_OUTPUTS;

    int top_x = (x + MLO_POOLING_PAD0 - MLO_POOLING_KERNEL_SZ0) < 0
                    ? 0
                    : (x + MLO_POOLING_PAD0 - MLO_POOLING_KERNEL_SZ0) / MLO_POOLING_STRIDE0 + 1;
    int top_y = (y + MLO_POOLING_PAD1 - MLO_POOLING_KERNEL_SZ1) < 0
                    ? 0
                    : (y + MLO_POOLING_PAD1 - MLO_POOLING_KERNEL_SZ1) / MLO_POOLING_STRIDE1 + 1;
    int top_df_off = b * MLO_POOLBWD_TOPDF_BATCH_STRIDE + o * MLO_POOLBWD_TOPDF_CHANNEL_STRIDE;

    _FLOAT res[MLO_POOLBWD_N_VERT_OUT_PIX][MLO_POOLBWD_N_HORIZ_OUT_PIX];
    _FLOAT top_df_val;
    index_t mask_val;
    // load tiles
    // top df and top
    for(int tj = lcl_id1; tj < MLO_POOLBWD_LCL_DATA_HEIGHT; tj += MLO_POOLBWD_GROUP_SZ1)
    {
        int top_y_act    = top_y + tj;
        int top_df_y_off = top_y_act * MLO_POOLBWD_TOPDF_STRIDE;

        int lcl_off_v = tj * MLO_POOLBWD_LCL_DATA_WIDTH;

        bool visibleY = (top_y_act < MLO_POOLBWD_TOP_HEIGHT);

        for(int ti = lcl_id0; ti < MLO_POOLBWD_LCL_DATA_WIDTH; ti += MLO_POOLBWD_GROUP_SZ0)
        {
            mask_val      = MLO_POOLING_INDEX_MAX;
            int top_x_act = top_x + ti;
            int lcl_idx   = lcl_off_v + ti;

            bool visible = visibleY && (top_x_act < MLO_POOLBWD_TOP_WIDTH);
            if(visible)
            {
                int idx = top_df_off + top_df_y_off + top_x_act;

                top_df_val = top_df[idx];
                mask_val   = mask[idx];
                // top_df_val *= visible;

                lcl_top_df[lcl_idx] = top_df_val;
            }
            lcl_mask[lcl_idx] = mask_val;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    _FLOAT add_val;
    int bt_y  = (y + lcl_id1 * MLO_POOLBWD_N_VERT_OUT_PIX);
    int bt_x  = (x + lcl_id0 * MLO_POOLBWD_N_HORIZ_OUT_PIX);
    int b_idx = bt_y * MLO_POOLBWD_BOT_WIDTH + bt_x;

    for(int k = 0; k < MLO_POOLBWD_N_VERT_OUT_PIX; k++)
    {
        int b_y = bt_y + k;

        // top most top y that can be influenced by this bot y
        int tt_y = (b_y + MLO_POOLING_PAD1 - MLO_POOLING_KERNEL_SZ1 + MLO_POOLING_STRIDE1) /
                   MLO_POOLING_STRIDE1;
        tt_y = max(0, tt_y);

        for(int l = 0; l < MLO_POOLBWD_N_HORIZ_OUT_PIX; l++)
        {
            int b_x = bt_x + l;
            // left most top x that can be influenced by this bot x
            int lt_x = (b_x + MLO_POOLING_PAD0 - MLO_POOLING_KERNEL_SZ0 + MLO_POOLING_STRIDE0) /
                       MLO_POOLING_STRIDE0;
            lt_x = max(0, lt_x);

            // find and sum up all tops that have been influenced by particular bot
            res[k][l] = 0;

            for(int th = tt_y;
                th <
                tt_y + (MLO_POOLING_KERNEL_SZ1 + MLO_POOLING_STRIDE1 - 1) / MLO_POOLING_STRIDE1;
                ++th)
            {
                __attribute__((opencl_unroll_hint(2))) for(int tw = lt_x;
                                                           tw < lt_x +
                                                                    (MLO_POOLING_KERNEL_SZ0 +
                                                                     MLO_POOLING_STRIDE0 - 1) /
                                                                        MLO_POOLING_STRIDE0;
                                                           ++tw)
                {
                    int lcl_th = th - top_y;
                    int lcl_tw = tw - top_x;

                    int filter_x   = b_x - tw * MLO_POOLING_STRIDE0 + MLO_POOLING_PAD0;
                    int filter_y   = b_y - th * MLO_POOLING_STRIDE1 + MLO_POOLING_PAD1;
                    int filter_idx = filter_x + filter_y * MLO_POOLING_KERNEL_SZ0;

                    // note, that b_idx == b_y * MLO_POOLBWD_BOT_WIDTH + b_x
                    // computing b_idx instead of using (b_y * MLO_POOLBWD_BOT_WIDTH + b_x) saves
                    // VGPR
                    bool visible = (lcl_th < MLO_POOLBWD_LCL_DATA_HEIGHT) &&
                                   (lcl_tw < MLO_POOLBWD_LCL_DATA_WIDTH);
                    int lcl_idx = visible ? (lcl_th * MLO_POOLBWD_LCL_DATA_WIDTH + lcl_tw) : 0;

                    bool match = visible && (filter_idx == lcl_mask[lcl_idx]) && (filter_x >= 0) &&
                                 (filter_y >= 0);

                    //_FLOAT add_val = lcl_top_df[lcl_idx] * match;
                    //_FLOAT add_val = match ? lcl_top_df[lcl_idx] : (_FLOAT)0;
                    if(match)
                    {
                        add_val = lcl_top_df[lcl_idx];
                        res[k][l] += add_val;
                    }
                }
            }
            b_idx++;
        }
        b_idx += MLO_POOLBWD_BOT_WIDTH - MLO_POOLBWD_N_HORIZ_OUT_PIX;
    }

    int bot_df_off = b * MLO_POOLBWD_BOTDF_BATCH_STRIDE + o * MLO_POOLBWD_BOTDF_CHANNEL_STRIDE +
                     bt_y * MLO_POOLBWD_BOTDF_STRIDE + bt_x;
    for(int k = 0; k < MLO_POOLBWD_N_VERT_OUT_PIX; k++)
    {
        for(int l = 0; l < MLO_POOLBWD_N_HORIZ_OUT_PIX; l++)
        {
            if((bt_y + k) < MLO_POOLBWD_BOT_HEIGHT && (bt_x + l) < MLO_POOLBWD_BOT_WIDTH)
            {
                bot_df[bot_df_off + k * MLO_POOLBWD_BOTDF_STRIDE + l] = res[k][l];
            }
        }
    }
}
