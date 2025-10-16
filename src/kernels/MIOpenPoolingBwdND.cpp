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

#if MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX

extern "C" __global__ __launch_bounds__((MLO_POOLING_GROUP_SZ0)) //
    void mloPoolingNDMaxBwd(const FLOAT* top_df,
                            FLOAT* bot_df,
                            index_t* mask,
                            const unsigned int pad_d,
                            const unsigned int pad_h,
                            const unsigned int pad_w,
                            const unsigned int batch,
                            const unsigned int chal,
                            const unsigned int bot_d,
                            const unsigned int bot_h,
                            const unsigned int bot_w,
                            const unsigned int top_d,
                            const unsigned int top_h,
                            const unsigned int top_w,
                            const unsigned int bot_str_b,
                            const unsigned int bot_str_c,
                            const unsigned int bot_str_d,
                            const unsigned int bot_str_h,
                            const unsigned int top_str_b,
                            const unsigned int top_str_c,
                            const unsigned int top_str_d,
                            const unsigned int top_str_h,
                            const unsigned int total_work)
{

    int bot_blk_w = (bot_w + PIX_W_PER_WORK - 1) / PIX_W_PER_WORK;
    int bot_blk_h = (bot_h + PIX_H_PER_WORK - 1) / PIX_H_PER_WORK;
    int bot_blk_d = (bot_d + PIX_D_PER_WORK - 1) / PIX_D_PER_WORK;

    bot_blk_w = max(bot_blk_w, 1);
    bot_blk_h = max(bot_blk_h, 1);
    bot_blk_d = max(bot_blk_d, 1);

    for(unsigned int gid = blockIdx.x * MLO_POOLING_GROUP_SZ0 + threadIdx.x; gid < total_work;
        gid += MAX_ACTIV_WORKITEM)
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

        top_d_end = min(top_d_end, static_cast<int>(top_d));
        top_h_end = min(top_h_end, static_cast<int>(top_h));
        top_w_end = min(top_w_end, static_cast<int>(top_w));

        FLOAT bot_data[PIX_D_PER_WORK][PIX_H_PER_WORK][PIX_W_PER_WORK] = {FLOAT{0}};

        for(int h = top_d_start; h < top_d_end; ++h)
        {
            for(int j = top_h_start; j < top_h_end; ++j)
            {
                for(int i = top_w_start; i < top_w_end; ++i)
                {
                    unsigned int top_gbl_off =
                        b_id * top_str_b + c_id * top_str_c + h * top_str_d + j * top_str_h + i;

                    FLOAT top_val    = b_id < batch ? top_df[top_gbl_off] : FLOAT{0};
                    index_t mask_idx = b_id < batch
                                           ? mask[top_gbl_off]
                                           : std::numeric_limits<MLO_POOLING_INDEX_TYPE>::max();

                    unsigned int mask_d_id = mask_idx / bot_h / bot_w;
                    unsigned int mask_h_id = (mask_idx / bot_w) % bot_h;
                    unsigned int mask_w_id = mask_idx % bot_w;

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

        unsigned int bot_off = b_id * bot_str_b + c_id * bot_str_c + bot_d_id * bot_str_d +
                               bot_h_id * bot_str_h + bot_w_id;

        for(unsigned int m = 0; m < PIX_D_PER_WORK; m++)
        {
            for(unsigned int k = 0; k < PIX_H_PER_WORK; k++)
            {
                for(unsigned int l = 0; l < PIX_W_PER_WORK; l++)
                {

                    if(bot_d_id + m < bot_d && bot_h_id + k < bot_h && bot_w_id + l < bot_w &&
                       b_id < batch)
                    {
                        unsigned int bot_idx = bot_off + m * bot_str_d + k * bot_str_h + l;

                        bot_df[bot_idx] = bot_data[m][k][l];
                    }
                }
            }
        }
    }
}

#elif MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE || MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE

extern "C" __global__ __launch_bounds__(MLO_POOLING_GROUP_SZ0) //
    void mloPoolingNDAveBwd(const FLOAT* top_df,
                            FLOAT* bot_df,
                            const unsigned int pad_d,
                            const unsigned int pad_h,
                            const unsigned int pad_w,
                            const unsigned int batch,
                            const unsigned int chal,
                            const unsigned int bot_d,
                            const unsigned int bot_h,
                            const unsigned int bot_w,
                            const unsigned int top_d,
                            const unsigned int top_h,
                            const unsigned int top_w,
                            const unsigned int bot_str_b,
                            const unsigned int bot_str_c,
                            const unsigned int bot_str_d,
                            const unsigned int bot_str_h,
                            const unsigned int top_str_b,
                            const unsigned int top_str_c,
                            const unsigned int top_str_d,
                            const unsigned int top_str_h,
                            const unsigned int total_work)
{

    int bot_blk_w = (bot_w + PIX_W_PER_WORK - 1) / PIX_W_PER_WORK;
    int bot_blk_h = (bot_h + PIX_H_PER_WORK - 1) / PIX_H_PER_WORK;
    int bot_blk_d = (bot_d + PIX_D_PER_WORK - 1) / PIX_D_PER_WORK;

    bot_blk_w = max(bot_blk_w, 1);
    bot_blk_h = max(bot_blk_h, 1);
    bot_blk_d = max(bot_blk_d, 1);

    for(unsigned int gid = blockIdx.x * MLO_POOLING_GROUP_SZ0 + threadIdx.x; gid < total_work;
        gid += MAX_ACTIV_WORKITEM)
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

        top_d_end = min(top_d_end, static_cast<int>(top_d));
        top_h_end = min(top_h_end, static_cast<int>(top_h));
        top_w_end = min(top_w_end, static_cast<int>(top_w));

        FLOAT_ACCUM bot_data[PIX_D_PER_WORK][PIX_H_PER_WORK][PIX_W_PER_WORK] = {FLOAT_ACCUM{0}};

        for(int h = top_d_start; h < top_d_end; ++h)
        {
            int dstart = h * STRIDE_D - pad_d;
            int dend   = min((dstart + KERNEL_SZ_D), static_cast<int>(bot_d));
            dstart     = max(dstart, 0);

            for(int j = top_h_start; j < top_h_end; ++j)
            {
                int hstart = j * STRIDE_H - pad_h;
                int hend   = min((hstart + KERNEL_SZ_H), static_cast<int>(bot_h));
                hstart     = max(hstart, 0);

                for(int i = top_w_start; i < top_w_end; ++i)
                {
                    int wstart = i * STRIDE_W - pad_w;
                    int wend   = min((wstart + KERNEL_SZ_W), static_cast<int>(bot_w));
                    wstart     = max(wstart, 0);

                    auto inv_pool_size = FLOAT_ACCUM{0};
                    if constexpr(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE)
                    {
                        inv_pool_size = FLOAT_ACCUM{1} /
                                        CVT_INTEGRAL2ACCUM(KERNEL_SZ_W * KERNEL_SZ_H * KERNEL_SZ_D);
                    }
                    else
                    {
                        int pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
                        pool_size     = (pool_size == 0) ? 1 : pool_size;
                        inv_pool_size = approxRcp(CVT_INTEGRAL2ACCUM(pool_size));
                    }

                    unsigned int top_gbl_off =
                        b_id * top_str_b + c_id * top_str_c + h * top_str_d + j * top_str_h + i;
                    FLOAT_ACCUM add_val =
                        b_id < batch ? CVT_FLOAT2ACCUM(top_df[top_gbl_off]) : CVT_FP32_2ACCUM(0.0f);
                    add_val *= inv_pool_size;

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

        unsigned int bot_off = b_id * bot_str_b + c_id * bot_str_c + bot_d_id * bot_str_d +
                               bot_h_id * bot_str_h + bot_w_id;

        for(unsigned int m = 0; m < PIX_D_PER_WORK; m++)
        {
            for(unsigned int k = 0; k < PIX_H_PER_WORK; k++)
            {
                for(unsigned int l = 0; l < PIX_W_PER_WORK; l++)
                {

                    if(bot_d_id + m < bot_d && bot_h_id + k < bot_h && bot_w_id + l < bot_w &&
                       b_id < batch)
                    {
                        unsigned int bot_idx = bot_off + m * bot_str_d + k * bot_str_h + l;

                        bot_df[bot_idx] = CVT_ACCUM2FLOAT(bot_data[m][k][l]);
                    }
                }
            }
        }
    }
}

#endif
