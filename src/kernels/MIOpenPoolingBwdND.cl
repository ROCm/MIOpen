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

#include "pooling_functions.h"

#ifndef MLO_POOLING_INDEX_MAX
#error "MLO_POOLING_INDEX_MAX not defined"
#endif

__attribute__((reqd_work_group_size(MLO_POOLING_GROUP_SZ0, 1, 1))) __kernel void
mloPoolingNDMaxBwd(const __global _FLOAT* top_df,
                   __global _FLOAT* bot_df,
                   __global index_t* mask,
                   const uint pad_d,
                   const uint pad_h,
                   const uint pad_w,
                   const uint batch,
                   const uint chal,
                   const uint bot_d,
                   const uint bot_h,
                   const uint bot_w,
                   const uint top_d,
                   const uint top_h,
                   const uint top_w,
                   const uint bot_str_b,
                   const uint bot_str_c,
                   const uint bot_str_d,
                   const uint bot_str_h,
                   const uint top_str_b,
                   const uint top_str_c,
                   const uint top_str_d,
                   const uint top_str_h,
                   const uint total_work)
{

    int bot_blk_w = (bot_w + PIX_W_PER_WORK - 1) / PIX_W_PER_WORK;
    int bot_blk_h = (bot_h + PIX_H_PER_WORK - 1) / PIX_H_PER_WORK;
    int bot_blk_d = (bot_d + PIX_D_PER_WORK - 1) / PIX_D_PER_WORK;

    bot_blk_w = max(bot_blk_w, 1);
    bot_blk_h = max(bot_blk_h, 1);
    bot_blk_d = max(bot_blk_d, 1);

    for(uint gid = get_global_id(0); gid < total_work; gid += MAX_ACTIV_WORKITEM)
    {
        int b_id = gid / chal / bot_blk_w / bot_blk_h / bot_blk_d;
        int c_id = (gid / bot_blk_w / bot_blk_h / bot_blk_d) % chal;

        int bot_d_id = ((gid / bot_blk_w / bot_blk_h) % bot_blk_d) * PIX_D_PER_WORK;
        int bot_h_id = ((gid / bot_blk_w) % bot_blk_h) * PIX_H_PER_WORK;
        int bot_w_id = (gid % bot_blk_w) * PIX_W_PER_WORK;

        int top_d_start =
            bot_d_id + pad_d < KERNEL_SZ_D ? 0 : (bot_d_id + pad_d - KERNEL_SZ_D) / STRIDE_D + 1;
        int top_h_start =
            bot_h_id + pad_h < KERNEL_SZ_H ? 0 : (bot_h_id + pad_h - KERNEL_SZ_H) / STRIDE_H + 1;
        int top_w_start =
            bot_w_id + pad_w < KERNEL_SZ_W ? 0 : (bot_w_id + pad_w - KERNEL_SZ_W) / STRIDE_W + 1;

        int top_d_end = (bot_d_id + PIX_D_PER_WORK - 1 + pad_d) / STRIDE_D + 1;
        int top_h_end = (bot_h_id + PIX_H_PER_WORK - 1 + pad_h) / STRIDE_H + 1;
        int top_w_end = (bot_w_id + PIX_W_PER_WORK - 1 + pad_w) / STRIDE_W + 1;

        top_d_end = min(top_d_end, (int)top_d);
        top_h_end = min(top_h_end, (int)top_h);
        top_w_end = min(top_w_end, (int)top_w);

        _FLOAT bot_data[PIX_D_PER_WORK][PIX_H_PER_WORK][PIX_W_PER_WORK] = {0};

        for(int h = top_d_start; h < top_d_end; ++h)
        {
            for(int j = top_h_start; j < top_h_end; ++j)
            {
                for(int i = top_w_start; i < top_w_end; ++i)
                {
                    uint top_gbl_off =
                        b_id * top_str_b + c_id * top_str_c + h * top_str_d + j * top_str_h + i;

                    _FLOAT top_val   = b_id < batch ? top_df[top_gbl_off] : 0;
                    index_t mask_idx = b_id < batch ? mask[top_gbl_off] : MLO_POOLING_INDEX_MAX;

                    uint mask_d_id = mask_idx / bot_h / bot_w;
                    uint mask_h_id = (mask_idx / bot_w) % bot_h;
                    uint mask_w_id = mask_idx % bot_w;

                    if(mask_d_id >= bot_d_id && mask_h_id >= bot_h_id && mask_w_id >= bot_w_id &&
                       mask_d_id < bot_d_id + PIX_D_PER_WORK &&
                       mask_h_id < bot_h_id + PIX_H_PER_WORK &&
                       mask_w_id < bot_w_id + PIX_W_PER_WORK)
                    {
                        mask_d_id -= bot_d_id;
                        mask_h_id -= bot_h_id;
                        mask_w_id -= bot_w_id;

                        bot_data[mask_d_id][mask_h_id][mask_w_id] += top_val;
                    }
                }
            }
        }

        uint bot_off = b_id * bot_str_b + c_id * bot_str_c + bot_d_id * bot_str_d +
                       bot_h_id * bot_str_h + bot_w_id;

        for(uint m = 0; m < PIX_D_PER_WORK; m++)
        {
            for(uint k = 0; k < PIX_H_PER_WORK; k++)
            {
                for(uint l = 0; l < PIX_W_PER_WORK; l++)
                {

                    if(bot_d_id + m < bot_d && bot_h_id + k < bot_h && bot_w_id + l < bot_w &&
                       b_id < batch)
                    {
                        uint bot_idx = bot_off + m * bot_str_d + k * bot_str_h + l;

                        bot_df[bot_idx] = bot_data[m][k][l];
                    }
                }
            }
        }
    }
}

__attribute__((reqd_work_group_size(MLO_POOLING_GROUP_SZ0, 1, 1))) __kernel void
mloPoolingNDAveBwd(const __global _FLOAT* top_df,
                   __global _FLOAT* bot_df,
                   const uint pad_d,
                   const uint pad_h,
                   const uint pad_w,
                   const uint batch,
                   const uint chal,
                   const uint bot_d,
                   const uint bot_h,
                   const uint bot_w,
                   const uint top_d,
                   const uint top_h,
                   const uint top_w,
                   const uint bot_str_b,
                   const uint bot_str_c,
                   const uint bot_str_d,
                   const uint bot_str_h,
                   const uint top_str_b,
                   const uint top_str_c,
                   const uint top_str_d,
                   const uint top_str_h,
                   const uint total_work)
{

    int bot_blk_w = (bot_w + PIX_W_PER_WORK - 1) / PIX_W_PER_WORK;
    int bot_blk_h = (bot_h + PIX_H_PER_WORK - 1) / PIX_H_PER_WORK;
    int bot_blk_d = (bot_d + PIX_D_PER_WORK - 1) / PIX_D_PER_WORK;

    bot_blk_w = max(bot_blk_w, 1);
    bot_blk_h = max(bot_blk_h, 1);
    bot_blk_d = max(bot_blk_d, 1);

    for(uint gid = get_global_id(0); gid < total_work; gid += MAX_ACTIV_WORKITEM)
    {
        int b_id = gid / chal / bot_blk_w / bot_blk_h / bot_blk_d;
        int c_id = (gid / bot_blk_w / bot_blk_h / bot_blk_d) % chal;

        int bot_d_id = ((gid / bot_blk_w / bot_blk_h) % bot_blk_d) * PIX_D_PER_WORK;
        int bot_h_id = ((gid / bot_blk_w) % bot_blk_h) * PIX_H_PER_WORK;
        int bot_w_id = (gid % bot_blk_w) * PIX_W_PER_WORK;

        int top_d_start =
            bot_d_id + pad_d < KERNEL_SZ_D ? 0 : (bot_d_id + pad_d - KERNEL_SZ_D) / STRIDE_D + 1;
        int top_h_start =
            bot_h_id + pad_h < KERNEL_SZ_H ? 0 : (bot_h_id + pad_h - KERNEL_SZ_H) / STRIDE_H + 1;
        int top_w_start =
            bot_w_id + pad_w < KERNEL_SZ_W ? 0 : (bot_w_id + pad_w - KERNEL_SZ_W) / STRIDE_W + 1;

        int top_d_end = (bot_d_id + PIX_D_PER_WORK - 1 + pad_d) / STRIDE_D + 1;
        int top_h_end = (bot_h_id + PIX_H_PER_WORK - 1 + pad_h) / STRIDE_H + 1;
        int top_w_end = (bot_w_id + PIX_W_PER_WORK - 1 + pad_w) / STRIDE_W + 1;

        top_d_end = min(top_d_end, (int)top_d);
        top_h_end = min(top_h_end, (int)top_h);
        top_w_end = min(top_w_end, (int)top_w);

        _FLOAT bot_data[PIX_D_PER_WORK][PIX_H_PER_WORK][PIX_W_PER_WORK] = {0};

        for(int h = top_d_start; h < top_d_end; ++h)
        {
            int dstart = h * STRIDE_D - pad_d;
            int dend   = min((dstart + KERNEL_SZ_D), (int)bot_d);
            dstart     = max(dstart, 0);

            for(int j = top_h_start; j < top_h_end; ++j)
            {
                int hstart = j * STRIDE_H - pad_h;
                int hend   = min((hstart + KERNEL_SZ_H), (int)bot_h);
                hstart     = max(hstart, 0);

                for(int i = top_w_start; i < top_w_end; ++i)
                {
                    int wstart = i * STRIDE_W - pad_w;
                    int wend   = min((wstart + KERNEL_SZ_W), (int)bot_w);
                    wstart     = max(wstart, 0);

                    uint pool_size =
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE
                        KERNEL_SZ_W * KERNEL_SZ_H * KERNEL_SZ_D;
#else
                        (dend - dstart) * (hend - hstart) * (wend - wstart);
#endif
                    pool_size = (pool_size == 0) ? 1 : pool_size;

                    uint top_gbl_off =
                        b_id * top_str_b + c_id * top_str_c + h * top_str_d + j * top_str_h + i;
                    _FLOAT add_val = b_id < batch ? top_df[top_gbl_off] : 0;
                    add_val /= (_FLOAT)pool_size;

                    for(int m = dstart; m < dend; ++m)
                    {
                        for(int k = hstart; k < hend; ++k)
                        {
                            for(int l = wstart; l < wend; ++l)
                            {
                                if(m >= bot_d_id && m < PIX_D_PER_WORK + bot_d_id &&
                                   k >= bot_h_id && k < PIX_H_PER_WORK + bot_h_id &&
                                   l >= bot_w_id && l < PIX_W_PER_WORK + bot_w_id && b_id < batch)
                                {
                                    bot_data[m - bot_d_id][k - bot_h_id][l - bot_w_id] += add_val;
                                }
                            }
                        }
                    }
                }
            }
        }

        uint bot_off = b_id * bot_str_b + c_id * bot_str_c + bot_d_id * bot_str_d +
                       bot_h_id * bot_str_h + bot_w_id;

        for(uint m = 0; m < PIX_D_PER_WORK; m++)
        {
            for(uint k = 0; k < PIX_H_PER_WORK; k++)
            {
                for(uint l = 0; l < PIX_W_PER_WORK; l++)
                {

                    if(bot_d_id + m < bot_d && bot_h_id + k < bot_h && bot_w_id + l < bot_w &&
                       b_id < batch)
                    {
                        uint bot_idx = bot_off + m * bot_str_d + k * bot_str_h + l;

                        bot_df[bot_idx] = bot_data[m][k][l];
                    }
                }
            }
        }
    }
}
