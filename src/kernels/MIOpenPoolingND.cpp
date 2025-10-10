/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019-2025 Advanced Micro Devices, Inc.
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
#include "pooling_functions.h"

#ifndef USE_GLOBAL_INDEX
#define USE_GLOBAL_INDEX 1
#endif

#if defined(MLO_POOLING_SAVE_INDEX) && (MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX)
constexpr bool USE_MASK = true;
#else
constexpr bool USE_MASK = false;
#endif

#if(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE) || (MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE)
#define AVERAGE_OPS 1
#else
#define AVERAGE_OPS 0
#endif

// Let's use extended-precision accumulator only in FP16 pooling and only for averaging.
// For all other ops and datatypes, redefine macros used for accum-float conversion
// and accum types, so they do nothing, i.e. treate FLOAT_ACCUM as FLOAT.
#if !(AVERAGE_OPS && MIOPEN_USE_FP16)
#define MIOPEN_USE_NATIVE_DATATYPE_ACCUM 1
#endif

#include "float_types.h"

#if MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
#define MLO_POOLING_OP(A, B) (poolingMax((A), (B)))
#elif AVERAGE_OPS
#define MLO_POOLING_OP(A, B) ((A) + (B))
#endif

constexpr auto BOT_TILE_W = (TOP_W_PER_WORK - 1) * STRIDE_W + KERNEL_SZ_W;
constexpr auto BOT_TILE_H = (TOP_H_PER_WORK - 1) * STRIDE_H + KERNEL_SZ_H;
constexpr auto BOT_TILE_D = (TOP_D_PER_WORK - 1) * STRIDE_D + KERNEL_SZ_D;

extern "C" __global__ __launch_bounds__(MLO_POOLING_GROUP_SZ0) //
    void mloPoolingNDFwd(const FLOAT* bot,
                         FLOAT* top,
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

    int top_blk_w = (top_w + TOP_W_PER_WORK - 1) / TOP_W_PER_WORK;
    int top_blk_h = (top_h + TOP_H_PER_WORK - 1) / TOP_H_PER_WORK;
    int top_blk_d = (top_d + TOP_D_PER_WORK - 1) / TOP_D_PER_WORK;

    top_blk_w = max(top_blk_w, 1);
    top_blk_h = max(top_blk_h, 1);
    top_blk_d = max(top_blk_d, 1);

    for(unsigned int gid = blockIdx.x * MLO_POOLING_GROUP_SZ0 + threadIdx.x; gid < total_work;
        gid += MAX_ACTIV_WORKITEM)
    {
        int b_id = gid / chal / top_blk_w / top_blk_h / top_blk_d;
        int c_id = (gid / top_blk_w / top_blk_h / top_blk_d) % chal;

        int top_d_id = ((gid / top_blk_w / top_blk_h) % top_blk_d) * TOP_D_PER_WORK;
        int top_h_id = ((gid / top_blk_w) % top_blk_h) * TOP_H_PER_WORK;
        int top_w_id = (gid % top_blk_w) * TOP_W_PER_WORK;

        FLOAT bot_data[BOT_TILE_D][BOT_TILE_H][BOT_TILE_W];

        const bool vis_d = b_id < batch;

        for(unsigned int h = 0; h < BOT_TILE_D; ++h)
        {
            const int run_z  = top_d_id * STRIDE_D + h - pad_d;
            const auto vis_z = vis_d && run_z >= 0 && run_z < bot_d;

            for(unsigned int j = 0; j < BOT_TILE_H; ++j)
            {
                const int run_y  = top_h_id * STRIDE_H + j - pad_h;
                const auto vis_y = vis_z && run_y >= 0 && run_y < bot_h;

                for(unsigned int i = 0; i < BOT_TILE_W; ++i)
                {
                    const int run_x  = top_w_id * STRIDE_W + i - pad_w;
                    const auto vis_x = vis_y && run_x >= 0 && run_x < bot_w;

                    const int bot_gbl_off = b_id * bot_str_b + c_id * bot_str_c +
                                            run_z * bot_str_d + run_y * bot_str_h + run_x;

                    const bool vis = vis_x;

                    bot_data[h][j][i] = vis ? bot[bot_gbl_off]
                                        : MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
                                            ? /* MAX */ FLOAT{-MAX_VAL}
                                            : /* AVG */ FLOAT{0};
                }
            }
        }

#pragma unroll
        for(unsigned int m = 0; m < TOP_D_PER_WORK; m++)
        {
            int dstart = 0, dend = 0;
            if constexpr(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE)
            {
                dstart = (top_d_id + m) * STRIDE_D - pad_d;
                dend   = min((dstart + KERNEL_SZ_D), static_cast<int>(bot_d));
                dstart = max(dstart, 0);
            }

            for(unsigned int k = 0; k < TOP_H_PER_WORK; k++)
            {
                int hstart = 0, hend = 0;
                if constexpr(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE)
                {
                    hstart = (top_h_id + k) * STRIDE_H - pad_h;
                    hend   = min((hstart + KERNEL_SZ_H), static_cast<int>(bot_h));
                    hstart = max(hstart, 0);
                }

                for(unsigned int l = 0; l < TOP_W_PER_WORK; l++)
                {
                    int wstart = 0, wend = 0;
                    if constexpr(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE)
                    {
                        wstart = (top_w_id + l) * STRIDE_W - pad_w;
                        wend   = min((wstart + KERNEL_SZ_W), static_cast<int>(bot_w));
                        wstart = max(wstart, 0);
                    }

                    auto inv_pool_size = FLOAT_ACCUM{0};
                    if constexpr(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE)
                    {
                        int pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
                        pool_size     = (pool_size == 0) ? 1 : pool_size;
                        inv_pool_size = approxRcp(CVT_INTEGRAL2ACCUM(pool_size));
                    }
                    else if constexpr(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE)
                    {
                        inv_pool_size = FLOAT_ACCUM{1} /
                                        CVT_INTEGRAL2ACCUM(KERNEL_SZ_W * KERNEL_SZ_H * KERNEL_SZ_D);
                    }

                    FLOAT_ACCUM top_val = MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX //
                                              ? /* MAX */ FLOAT_ACCUM{-MAX_VAL_ACCUM}
                                              : /* AVG */ FLOAT_ACCUM{0};

                    index_t mask_idx = 0;

                    for(unsigned int h = 0; h < KERNEL_SZ_D; h++)
                    {
                        for(unsigned int j = 0; j < KERNEL_SZ_H; j++)
                        {
                            for(unsigned int i = 0; i < KERNEL_SZ_W; i++)
                            {
                                FLOAT_ACCUM bot_val = CVT_FLOAT2ACCUM(
                                    bot_data[h + m * STRIDE_D][j + k * STRIDE_H][i + l * STRIDE_W]);

                                if constexpr(USE_MASK)
                                {
                                    if(bot_val > top_val)
                                    {
                                        top_val = bot_val;

                                        if constexpr(USE_GLOBAL_INDEX)
                                        {
                                            mask_idx =
                                                ((top_w_id + l) * STRIDE_W + i - pad_w) +
                                                bot_w * ((top_h_id + k) * STRIDE_H + j - pad_h) +
                                                bot_w * bot_h *
                                                    ((top_d_id + m) * STRIDE_D + h - pad_d);
                                        }
                                        else
                                        {
                                            mask_idx = i + KERNEL_SZ_W * (j + KERNEL_SZ_H * h);
                                        }
                                    }
                                }
                                else
                                {
                                    top_val = MLO_POOLING_OP(top_val, bot_val);
                                }
                            }
                        }
                    }

                    if constexpr(AVERAGE_OPS)
                        top_val *= inv_pool_size;

                    if(top_d_id + m < top_d && top_h_id + k < top_h && top_w_id + l < top_w &&
                       b_id < batch)
                    {
                        unsigned int top_idx = b_id * top_str_b + c_id * top_str_c +
                                               (top_d_id + m) * top_str_d +
                                               (top_h_id + k) * top_str_h + top_w_id + l;

                        top[top_idx] = top_val;
                        if constexpr(USE_MASK)
                            mask[top_idx] = mask_idx;
                    }
                }
            }
        }
    }
}
