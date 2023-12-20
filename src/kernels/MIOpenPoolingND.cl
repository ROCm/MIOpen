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

#ifndef USE_GLOBAL_INDEX
#define USE_GLOBAL_INDEX 1
#endif

#if defined(MLO_POOLING_SAVE_INDEX) && (MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX)
#define USE_MASK 1
#else
#define USE_MASK 0
#endif

#ifndef MLO_POOLING_OP_ID
#define MLO_POOLING_OP_ID 0
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
#define MLO_POOLING_OP(A, B) (fmax((A), (B)))
#elif AVERAGE_OPS
#define MLO_POOLING_OP(A, B) ((A) + (B))
#endif

#define BOT_TILE_W ((TOP_W_PER_WORK - 1) * STRIDE_W + KERNEL_SZ_W)
#define BOT_TILE_H ((TOP_H_PER_WORK - 1) * STRIDE_H + KERNEL_SZ_H)
#define BOT_TILE_D ((TOP_D_PER_WORK - 1) * STRIDE_D + KERNEL_SZ_D)

__attribute__((reqd_work_group_size(MLO_POOLING_GROUP_SZ0, 1, 1))) __kernel void
mloPoolingNDFwd(const __global _FLOAT* bot,
                __global _FLOAT* top,
#if !USE_MASK
                UNUSED
#endif
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

    int top_blk_w = (top_w + TOP_W_PER_WORK - 1) / TOP_W_PER_WORK;
    int top_blk_h = (top_h + TOP_H_PER_WORK - 1) / TOP_H_PER_WORK;
    int top_blk_d = (top_d + TOP_D_PER_WORK - 1) / TOP_D_PER_WORK;

    top_blk_w = max(top_blk_w, 1);
    top_blk_h = max(top_blk_h, 1);
    top_blk_d = max(top_blk_d, 1);

    for(uint gid = get_global_id(0); gid < total_work; gid += MAX_ACTIV_WORKITEM)
    {
        int b_id = gid / chal / top_blk_w / top_blk_h / top_blk_d;
        int c_id = (gid / top_blk_w / top_blk_h / top_blk_d) % chal;

        int top_d_id = ((gid / top_blk_w / top_blk_h) % top_blk_d) * TOP_D_PER_WORK;
        int top_h_id = ((gid / top_blk_w) % top_blk_h) * TOP_H_PER_WORK;
        int top_w_id = (gid % top_blk_w) * TOP_W_PER_WORK;

        _FLOAT bot_data[BOT_TILE_D][BOT_TILE_H][BOT_TILE_W];

        for(uint h = 0; h < BOT_TILE_D; ++h)
        {
            int run_z = top_d_id * STRIDE_D + h - pad_d;
            for(uint j = 0; j < BOT_TILE_H; ++j)
            {
                int run_y = top_h_id * STRIDE_H + j - pad_h;
                for(uint i = 0; i < BOT_TILE_W; ++i)
                {
                    int run_x       = top_w_id * STRIDE_W + i - pad_w;
                    int bot_gbl_off = b_id * bot_str_b + c_id * bot_str_c + run_z * bot_str_d +
                                      run_y * bot_str_h + run_x;
                    bool vis = ((run_z >= 0 && run_z < bot_d) && (run_y >= 0 && run_y < bot_h) &&
                                (run_x >= 0 && run_x < bot_w)) &&
                               b_id < batch;

                    bot_data[h][j][i] = (vis) ? bot[bot_gbl_off] :
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
                                              (_FLOAT)(-MAX_VAL);
#elif AVERAGE_OPS
                                              (_FLOAT)(0);
#endif
                }
            }
        }

#pragma unroll
        for(uint m = 0; m < TOP_D_PER_WORK; m++)
        {
#if AVERAGE_OPS
            int dstart = (top_d_id + m) * STRIDE_D - pad_d;
            int dend   = min((dstart + KERNEL_SZ_D), (int)bot_d);
            dstart     = max(dstart, 0);
#endif
            for(uint k = 0; k < TOP_H_PER_WORK; k++)
            {
#if AVERAGE_OPS
                int hstart = (top_h_id + k) * STRIDE_H - pad_h;
                int hend   = min((hstart + KERNEL_SZ_H), (int)bot_h);
                hstart     = max(hstart, 0);
#endif
                for(uint l = 0; l < TOP_W_PER_WORK; l++)
                {

#if AVERAGE_OPS
                    int wstart = (top_w_id + l) * STRIDE_W - pad_w;
                    int wend   = min((wstart + KERNEL_SZ_W), (int)bot_w);
                    wstart     = max(wstart, 0);
                    uint pool_size =
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE
                        KERNEL_SZ_W * KERNEL_SZ_H * KERNEL_SZ_D;
                    (void)wend;
                    (void)hend;
                    (void)dend;
#else
                        (dend - dstart) * (hend - hstart) * (wend - wstart);
#endif
                    pool_size = (pool_size == 0) ? 1 : pool_size;
#endif

                    _FLOAT_ACCUM top_val =
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
                        (_FLOAT_ACCUM)(-MAX_VAL_ACCUM);
#elif AVERAGE_OPS
                        (_FLOAT_ACCUM)(0);
#endif

#if USE_MASK
                    index_t mask_idx = 0;
#endif

                    for(uint h = 0; h < KERNEL_SZ_D; h++)
                    {
                        for(uint j = 0; j < KERNEL_SZ_H; j++)
                        {
                            for(uint i = 0; i < KERNEL_SZ_W; i++)
                            {

                                _FLOAT_ACCUM bot_val = CVT_FLOAT2ACCUM(
                                    bot_data[h + m * STRIDE_D][j + k * STRIDE_H][i + l * STRIDE_W]);

#if USE_MASK
                                if(bot_val > top_val)
                                {
                                    top_val = bot_val;

#if USE_GLOBAL_INDEX
                                    mask_idx =
                                        ((top_w_id + l) * STRIDE_W + i - pad_w) +
                                        bot_w * ((top_h_id + k) * STRIDE_H + j - pad_h) +
                                        bot_w * bot_h * ((top_d_id + m) * STRIDE_D + h - pad_d);
#else
                                    mask_idx = i + KERNEL_SZ_W * (j + KERNEL_SZ_H * h);
#endif
                                }
#else
                                top_val = MLO_POOLING_OP(top_val, bot_val);
#endif
                            }
                        }
                    }

#if AVERAGE_OPS
                    top_val *= CVT_FP32_2ACCUM(1.f) / (_FLOAT_ACCUM)pool_size;
#endif

                    if(top_d_id + m < top_d && top_h_id + k < top_h && top_w_id + l < top_w &&
                       b_id < batch)
                    {
                        uint top_idx = b_id * top_str_b + c_id * top_str_c +
                                       (top_d_id + m) * top_str_d + (top_h_id + k) * top_str_h +
                                       top_w_id + l;

                        top[top_idx] = top_val;
#if USE_MASK
                        mask[top_idx] = mask_idx;
#endif
                    }
                }
            }
        }
    }
}
