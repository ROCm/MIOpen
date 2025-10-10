/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include <hip/hip_runtime.h>
#include "miopen_cstdint.hpp"
#include "miopen_limits.hpp"
#include "float_types.h"
#include "pooling_functions.h"

#ifdef USE_IMG_INDEX
#if !(USE_IMG_INDEX == 0 || USE_IMG_INDEX == 1)
#error "Bad value of USE_IMG_INDEX"
#endif
#else
#define USE_IMG_INDEX 1
#endif

#if(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE) || (MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE)
#define AVERAGE_OPS 1
#else
#define AVERAGE_OPS 0
#endif

constexpr auto MLO_POOLBWD_LCL_DATA_WIDTH = (MLO_POOLBWD_GROUP_SZ0 * MLO_POOLBWD_N_HORIZ_OUT_PIX +
                                             MLO_POOLING_KERNEL_SZ0 + MLO_POOLING_STRIDE0 - 2) /
                                            MLO_POOLING_STRIDE0;
constexpr auto MLO_POOLBWD_LCL_DATA_HEIGHT = (MLO_POOLBWD_GROUP_SZ1 * MLO_POOLBWD_N_VERT_OUT_PIX +
                                              MLO_POOLING_KERNEL_SZ1 + MLO_POOLING_STRIDE1 - 2) /
                                             MLO_POOLING_STRIDE1;

constexpr int block_size = MLO_POOLBWD_GROUP_SZ0 * MLO_POOLBWD_GROUP_SZ1;

#if AVERAGE_OPS

extern "C" __global__ __launch_bounds__(block_size) //
    void mloPoolingAveBwd(const FLOAT* top_diff,
                          FLOAT* bot_diff,
                          int mlo_pad1,
                          int mlo_pad0,
                          int mlo_n_outputs,
                          int mlo_bot_height,
                          int mlo_bot_width,
                          int mlo_top_height,
                          int mlo_top_width,
                          int mlo_botdf_batch_str,
                          int mlo_botdf_channel_str,
                          int mlo_botdf_str,
                          int mlo_topdf_batch_str,
                          int mlo_topdf_channel_str,
                          int mlo_topdf_str)
{
    __shared__ FLOAT lcl_top_diff[MLO_POOLBWD_LCL_DATA_WIDTH * MLO_POOLBWD_LCL_DATA_HEIGHT];

    int x       = blockIdx.x * MLO_POOLBWD_GROUP_SZ0 * MLO_POOLBWD_N_HORIZ_OUT_PIX;
    int y       = blockIdx.y * MLO_POOLBWD_GROUP_SZ1 * MLO_POOLBWD_N_VERT_OUT_PIX;
    int lcl_id0 = threadIdx.x;
    int lcl_id1 = threadIdx.y;
    //        int lcl_id = (lcl_id1 << MLO_POOLBWD_GROUP_LG2SZ1) + lcl_id0;
    int ob = blockIdx.z; // outputs * batch_sz
    int b  = ob / mlo_n_outputs;
    int o  = ob - b * mlo_n_outputs;

    int top_x   = (x + mlo_pad0 - MLO_POOLING_KERNEL_SZ0) < 0
                      ? 0
                      : (x + mlo_pad0 - MLO_POOLING_KERNEL_SZ0) / MLO_POOLING_STRIDE0 + 1;
    int top_y   = (y + mlo_pad1 - MLO_POOLING_KERNEL_SZ1) < 0
                      ? 0
                      : (y + mlo_pad1 - MLO_POOLING_KERNEL_SZ1) / MLO_POOLING_STRIDE1 + 1;
    int top_off = b * mlo_topdf_batch_str + o * mlo_topdf_channel_str;

    FLOAT_ACCUM res[MLO_POOLBWD_N_VERT_OUT_PIX][MLO_POOLBWD_N_HORIZ_OUT_PIX];
    for(int k = 0; k < MLO_POOLBWD_N_VERT_OUT_PIX; k++)
    {
        for(int l = 0; l < MLO_POOLBWD_N_HORIZ_OUT_PIX; l++)
        {
            res[k][l] = FLOAT_ACCUM{0};
        }
    }

// load tile
#pragma unroll
    for(int tj1 = 0; tj1 < MLO_POOLBWD_LCL_DATA_HEIGHT; tj1 += MLO_POOLBWD_GROUP_SZ1)
    {
        const auto tj = tj1 + lcl_id1;
        if(MLO_POOLBWD_LCL_DATA_HEIGHT % MLO_POOLBWD_GROUP_SZ1 == 0 ||
           tj < MLO_POOLBWD_LCL_DATA_HEIGHT)
        {
            int top_y_act = top_y + tj;
            int top_y_off = top_y_act * mlo_topdf_str;

            int lcl_off_v = tj * MLO_POOLBWD_LCL_DATA_WIDTH;

            bool invisibleY = (top_y_act >= mlo_top_height);

#pragma unroll
            for(int ti1 = 0; ti1 < MLO_POOLBWD_LCL_DATA_WIDTH; ti1 += MLO_POOLBWD_GROUP_SZ0)
            {
                const auto ti = ti1 + lcl_id0;
                if(MLO_POOLBWD_LCL_DATA_WIDTH % MLO_POOLBWD_GROUP_SZ0 == 0 ||
                   ti < MLO_POOLBWD_LCL_DATA_WIDTH)
                {
                    int top_x_act = top_x + ti;

                    bool invisibleX = (top_x_act >= mlo_top_width);

                    int top_diff_off =
                        (invisibleX || invisibleY) ? 0 : top_off + top_y_off + top_x_act;

                    FLOAT top_val = top_diff[top_diff_off];

                    top_val = (invisibleX || invisibleY) ? 0 : top_val;

                    lcl_top_diff[lcl_off_v + ti] = top_val;
                }
            }
        }
    }

    __syncthreads();

    int bot_y = (y + lcl_id1 * MLO_POOLBWD_N_VERT_OUT_PIX);
    int bot_x = (x + lcl_id0 * MLO_POOLBWD_N_HORIZ_OUT_PIX);

    for(int k = 0; k < MLO_POOLBWD_N_VERT_OUT_PIX; k++)
    {

        int h          = bot_y + k + mlo_pad1;
        int top_hstart = (h < MLO_POOLING_KERNEL_SZ1)
                             ? 0
                             : (h - MLO_POOLING_KERNEL_SZ1) / MLO_POOLING_STRIDE1 + 1;
        int top_hend   = min(h / MLO_POOLING_STRIDE1 + 1, mlo_top_height);

        for(int l = 0; l < MLO_POOLBWD_N_HORIZ_OUT_PIX; l++)
        {

            int w          = bot_x + l + mlo_pad0;
            int top_wstart = (w < MLO_POOLING_KERNEL_SZ0)
                                 ? 0
                                 : (w - MLO_POOLING_KERNEL_SZ0) / MLO_POOLING_STRIDE0 + 1;
            int top_wend   = min(w / MLO_POOLING_STRIDE0 + 1, mlo_top_width);

            for(int top_h = top_hstart; top_h < top_hend; ++top_h)
            {
                int hstart = top_h * MLO_POOLING_STRIDE1 - mlo_pad1;
                int hend   = min(hstart + MLO_POOLING_KERNEL_SZ1, mlo_bot_height);
                hstart     = max(hstart, 0);

                for(int top_w = top_wstart; top_w < top_wend; ++top_w)
                {
                    // figure out the pooling size
                    int wstart = top_w * MLO_POOLING_STRIDE0 - mlo_pad0;
                    int wend   = min(wstart + MLO_POOLING_KERNEL_SZ0, mlo_bot_width);
                    wstart     = max(wstart, 0);

                    auto inv_pool_size = FLOAT_ACCUM{0};
                    if constexpr(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE)
                    {
                        inv_pool_size = FLOAT_ACCUM{1} / CVT_INTEGRAL2ACCUM(MLO_POOLING_KERNEL_SZ0 *
                                                                            MLO_POOLING_KERNEL_SZ1);
                    }
                    else
                    {
                        int pool_size = (hend - hstart) * (wend - wstart);
                        pool_size     = (pool_size == 0) ? 1 : pool_size;
                        inv_pool_size = approxRcp(CVT_INTEGRAL2ACCUM(pool_size));
                    }

                    int lcl_top_h = top_h - top_y;
                    int lcl_top_w = top_w - top_x;
                    FLOAT_ACCUM add_val =
                        CVT_FLOAT2ACCUM(
                            lcl_top_diff[lcl_top_h * MLO_POOLBWD_LCL_DATA_WIDTH + lcl_top_w]) *
                        inv_pool_size;

                    res[k][l] += add_val;
                }
            }
        }
    }

    int bot_off =
        b * mlo_botdf_batch_str + o * mlo_botdf_channel_str + bot_y * mlo_botdf_str + bot_x;
    for(int k = 0; k < MLO_POOLBWD_N_VERT_OUT_PIX; k++)
    {
        for(int l = 0; l < MLO_POOLBWD_N_HORIZ_OUT_PIX; l++)
        {
            if(bot_y + k < mlo_bot_height && bot_x + l < mlo_bot_width)
            {
                bot_diff[bot_off + k * mlo_botdf_str + l] = CVT_ACCUM2FLOAT(res[k][l]);
            }
        }
    }
}

#endif // AVERAGE_OPS

#if MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX

extern "C" __global__ __launch_bounds__(block_size) //
    void mloPoolingMaxBwd(const FLOAT* top_df,
                          FLOAT* bot_df,
                          index_t* mask,
                          int mlo_pad1,
                          int mlo_pad0,
                          int mlo_n_outputs,
                          int mlo_bot_height,
                          int mlo_bot_width,
                          int mlo_top_height,
                          int mlo_top_width,
                          int mlo_botdf_batch_str,
                          int mlo_botdf_channel_str,
                          int mlo_botdf_str,
                          int mlo_topdf_batch_str,
                          int mlo_topdf_channel_str,
                          int mlo_topdf_str)
{
    __shared__ FLOAT lcl_top_df[MLO_POOLBWD_LCL_DATA_WIDTH * MLO_POOLBWD_LCL_DATA_HEIGHT];
    __shared__ index_t lcl_mask[MLO_POOLBWD_LCL_DATA_WIDTH * MLO_POOLBWD_LCL_DATA_HEIGHT];

    int gid0    = blockIdx.x;
    int gid1    = blockIdx.y;
    int x       = gid0 * MLO_POOLBWD_GROUP_SZ0 * MLO_POOLBWD_N_HORIZ_OUT_PIX;
    int y       = gid1 * MLO_POOLBWD_GROUP_SZ1 * MLO_POOLBWD_N_VERT_OUT_PIX;
    int lcl_id0 = threadIdx.x;
    int lcl_id1 = threadIdx.y;
    int ob      = blockIdx.z; // outputs * batch_sz
    int b       = ob / mlo_n_outputs;
    int o       = ob - b * mlo_n_outputs;

    int top_x      = (x + mlo_pad0 - MLO_POOLING_KERNEL_SZ0) < 0
                         ? 0
                         : (x + mlo_pad0 - MLO_POOLING_KERNEL_SZ0) / MLO_POOLING_STRIDE0 + 1;
    int top_y      = (y + mlo_pad1 - MLO_POOLING_KERNEL_SZ1) < 0
                         ? 0
                         : (y + mlo_pad1 - MLO_POOLING_KERNEL_SZ1) / MLO_POOLING_STRIDE1 + 1;
    int top_df_off = b * mlo_topdf_batch_str + o * mlo_topdf_channel_str;

    FLOAT res[MLO_POOLBWD_N_VERT_OUT_PIX][MLO_POOLBWD_N_HORIZ_OUT_PIX];
    FLOAT top_df_val;
    index_t mask_val;
    // load tiles
    // top df and top
    for(int tj = lcl_id1; tj < MLO_POOLBWD_LCL_DATA_HEIGHT; tj += MLO_POOLBWD_GROUP_SZ1)
    {
        int top_y_act    = top_y + tj;
        int top_df_y_off = top_y_act * mlo_topdf_str;

        int lcl_off_v = tj * MLO_POOLBWD_LCL_DATA_WIDTH;

        bool visibleY = (top_y_act < mlo_top_height);

        for(int ti = lcl_id0; ti < MLO_POOLBWD_LCL_DATA_WIDTH; ti += MLO_POOLBWD_GROUP_SZ0)
        {
            mask_val      = std::numeric_limits<MLO_POOLING_INDEX_TYPE>::max();
            int top_x_act = top_x + ti;
            int lcl_idx   = lcl_off_v + ti;

            bool visible = visibleY && (top_x_act < mlo_top_width);
            if(visible)
            {
                int idx = top_df_off + top_df_y_off + top_x_act;

                top_df_val = top_df[idx];
                mask_val   = mask[idx];

                lcl_top_df[lcl_idx] = top_df_val;
            }
            lcl_mask[lcl_idx] = mask_val;
        }
    }

    __syncthreads();
    int bt_y = (y + lcl_id1 * MLO_POOLBWD_N_VERT_OUT_PIX);
    int bt_x = (x + lcl_id0 * MLO_POOLBWD_N_HORIZ_OUT_PIX);

    for(int k = 0; k < MLO_POOLBWD_N_VERT_OUT_PIX; k++)
    {
        int b_y = bt_y + k;

        // top most top y that can be influenced by this bot y
        int tt_y1 =
            (b_y + mlo_pad1 - MLO_POOLING_KERNEL_SZ1 + MLO_POOLING_STRIDE1) / MLO_POOLING_STRIDE1;
        int tt_y = max(0, tt_y1);

        for(int l = 0; l < MLO_POOLBWD_N_HORIZ_OUT_PIX; l++)
        {
            int b_x = bt_x + l;
            // left most top x that can be influenced by this bot x
            int lt_x1 = (b_x + mlo_pad0 - MLO_POOLING_KERNEL_SZ0 + MLO_POOLING_STRIDE0) /
                        MLO_POOLING_STRIDE0;
            int lt_x = max(0, lt_x1);

            // find and sum up all tops that have been influenced by particular bot
            res[k][l] = 0;

            for(int th = tt_y; th < tt_y + (MLO_POOLING_KERNEL_SZ1 + MLO_POOLING_STRIDE1 - 1) /
                                               MLO_POOLING_STRIDE1;
                ++th)
            {
#pragma unroll 2
                for(int tw = lt_x; tw < lt_x + (MLO_POOLING_KERNEL_SZ0 + MLO_POOLING_STRIDE0 - 1) /
                                                   MLO_POOLING_STRIDE0;
                    ++tw)
                {
                    int lcl_th = th - top_y;
                    int lcl_tw = tw - top_x;

                    bool visible = (lcl_th < MLO_POOLBWD_LCL_DATA_HEIGHT) &&
                                   (lcl_tw < MLO_POOLBWD_LCL_DATA_WIDTH);
                    int lcl_idx = visible ? (lcl_th * MLO_POOLBWD_LCL_DATA_WIDTH + lcl_tw) : 0;

                    bool match = visible;
                    if constexpr(USE_IMG_INDEX)
                    {
                        index_t img_idx = b_x + b_y * mlo_bot_width;
                        match           = match && (img_idx == lcl_mask[lcl_idx]);
                    }
                    else
                    {
                        const int filter_x   = b_x - tw * MLO_POOLING_STRIDE0 + mlo_pad0;
                        const int filter_y   = b_y - th * MLO_POOLING_STRIDE1 + mlo_pad1;
                        const int filter_idx = filter_x + filter_y * MLO_POOLING_KERNEL_SZ0;
                        match = match && (filter_idx == lcl_mask[lcl_idx]) && (filter_x >= 0) &&
                                (filter_y >= 0);
                    }

                    if(match)
                    {
                        FLOAT add_val = lcl_top_df[lcl_idx];
                        res[k][l] += add_val;
                    }
                }
            }
        }
    }

    int bot_df_off =
        b * mlo_botdf_batch_str + o * mlo_botdf_channel_str + bt_y * mlo_botdf_str + bt_x;
    for(int k = 0; k < MLO_POOLBWD_N_VERT_OUT_PIX; k++)
    {
        for(int l = 0; l < MLO_POOLBWD_N_HORIZ_OUT_PIX; l++)
        {
            if((bt_y + k) < mlo_bot_height && (bt_x + l) < mlo_bot_width)
            {
                bot_df[bot_df_off + k * mlo_botdf_str + l] = res[k][l];
            }
        }
    }
}

#endif // MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
