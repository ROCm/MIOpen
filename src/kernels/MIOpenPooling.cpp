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
#include "pooling_functions.h"

#ifdef USE_IMG_INDEX
static_assert(USE_IMG_INDEX == 0 || USE_IMG_INDEX == 1, "Bad value of USE_IMG_INDEX");
#else
#define USE_IMG_INDEX 1
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

#if MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
#define MLO_POOLING_OP(A, B) (poolingMax((A), (B)))
#elif AVERAGE_OPS
#define MLO_POOLING_OP(A, B) ((A) + (B))
#endif

constexpr auto MLO_BOT_DATA_SZ0 =
    (MLO_POOLING_N_HORIZ_OUT_PIX - 1) * MLO_POOLING_STRIDE0 + MLO_POOLING_KERNEL_SZ0;

constexpr auto MLO_BOT_DATA_SZ1 =
    (MLO_POOLING_N_VERT_OUT_PIX - 1) * MLO_POOLING_STRIDE1 + MLO_POOLING_KERNEL_SZ1;

// Let's use extended-precision accumulator only in FP16 pooling and only for averaging.
// For all other ops and datatypes, redefine macros used for accum-float conversion
// and accum types, so they do nothing, i.e. treate FLOAT_ACCUM as FLOAT.
#if !(AVERAGE_OPS && MIOPEN_USE_FP16)
#define MIOPEN_USE_NATIVE_DATATYPE_ACCUM 1
#endif

#include "float_types.h"

constexpr int block_size = MLO_POOLING_GROUP_SZ0 * MLO_POOLING_GROUP_SZ1;

extern "C" __global__ __launch_bounds__(block_size) void mloPoolingG(const FLOAT* bot,
                                                                     FLOAT* top,
                                                                     index_t* mask,
                                                                     int mlo_pad1,
                                                                     int mlo_pad0,
                                                                     int mlo_n_outputs,
                                                                     int mlo_bot_height,
                                                                     int mlo_bot_width,
                                                                     int mlo_top_height,
                                                                     int mlo_top_width,
                                                                     int mlo_bot_batch_str,
                                                                     int mlo_bot_channel_str,
                                                                     int mlo_bot_str,
                                                                     int mlo_top_batch_str,
                                                                     int mlo_top_channel_str,
                                                                     int mlo_top_str)
{
    const unsigned int x       = blockIdx.x * MLO_POOLING_GROUP_SZ0 * MLO_POOLING_N_HORIZ_OUT_PIX;
    const unsigned int y       = blockIdx.y * MLO_POOLING_GROUP_SZ1 * MLO_POOLING_N_VERT_OUT_PIX;
    const unsigned int lcl_id0 = threadIdx.x;
    const unsigned int lcl_id1 = threadIdx.y;

    const unsigned int ob      = blockIdx.z;
    const unsigned int b       = ob / mlo_n_outputs;
    const unsigned int o       = ob - b * mlo_n_outputs;
    const unsigned int bot_x   = (x + lcl_id0 * MLO_POOLING_N_HORIZ_OUT_PIX) * MLO_POOLING_STRIDE0;
    const unsigned int bot_y   = (y + lcl_id1 * MLO_POOLING_N_VERT_OUT_PIX) * MLO_POOLING_STRIDE1;
    const unsigned int bot_off = b * mlo_bot_batch_str + o * mlo_bot_channel_str;

    FLOAT bot_data[MLO_BOT_DATA_SZ1][MLO_BOT_DATA_SZ0];
    FLOAT_ACCUM res[MLO_POOLING_N_VERT_OUT_PIX][MLO_POOLING_N_HORIZ_OUT_PIX];

    index_t mask_private[MLO_POOLING_N_VERT_OUT_PIX][MLO_POOLING_N_HORIZ_OUT_PIX];

    for(int k = 0; k < MLO_POOLING_N_VERT_OUT_PIX; k++)
    {
        for(int l = 0; l < MLO_POOLING_N_HORIZ_OUT_PIX; l++)
        {
            res[k][l] = MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
                            ? /* MAX */ FLOAT_ACCUM{-MAX_VAL_ACCUM}
                            : /* AVG */ FLOAT{0};
        }
    }

    for(unsigned int j = 0; j < MLO_BOT_DATA_SZ1; ++j)
    {
        int run_y  = static_cast<int>(bot_y) + j - mlo_pad1;
        bool vis_y = run_y >= 0 && run_y < mlo_bot_height;

        for(unsigned int i = 0; i < MLO_BOT_DATA_SZ0; ++i)
        {
            int run_x                = static_cast<int>(bot_x) + i - mlo_pad0;
            unsigned int bot_gbl_off = bot_off + static_cast<unsigned int>(run_y) * mlo_bot_str +
                                       static_cast<unsigned int>(run_x);
            bool vis_x = run_x >= 0 && run_x < mlo_bot_width;
            bool vis   = vis_x && vis_y;

            bot_data[j][i] = vis ? bot[bot_gbl_off]
                             : MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX //
                                 ? /* MAX */ FLOAT{-MAX_VAL}
                                 : /* AVG */ FLOAT{0};
        }
    }

#pragma unroll
    for(unsigned int k = 0; k < MLO_POOLING_N_VERT_OUT_PIX; k++)
    {
        const unsigned int y_dst = y + lcl_id1 * MLO_POOLING_N_VERT_OUT_PIX + k;

        int hstart1 = 0, hstart = 0, hend = 0;
        if constexpr(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE || (USE_MASK && USE_IMG_INDEX))
        {
            hstart1 = static_cast<int>(y_dst) * MLO_POOLING_STRIDE1 - mlo_pad1;
            hend    = min((hstart1 + MLO_POOLING_KERNEL_SZ1), static_cast<int>(mlo_bot_height));
            hstart  = max(hstart1, 0);
        }

        for(unsigned int l = 0; l < MLO_POOLING_N_HORIZ_OUT_PIX; l++)
        {
            const unsigned int x_dst = x + lcl_id0 * MLO_POOLING_N_HORIZ_OUT_PIX + l;

            int wstart1 = 0, wstart = 0, wend = 0;
            if constexpr(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE || (USE_MASK && USE_IMG_INDEX))
            {
                wstart1 = static_cast<int>(x_dst) * MLO_POOLING_STRIDE0 - mlo_pad0;
                wend    = min((wstart1 + MLO_POOLING_KERNEL_SZ0), static_cast<int>(mlo_bot_width));
                wstart  = max(wstart1, 0);
            }

            auto inv_pool_size = FLOAT_ACCUM{0};
            if constexpr(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE)
            {
                int pool_size = (hend - hstart) * (wend - wstart);
                pool_size     = (pool_size == 0) ? 1 : pool_size;
                inv_pool_size = approxRcp(CVT_INTEGRAL2ACCUM(pool_size));
            }
            else if constexpr(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE)
            {
                inv_pool_size = FLOAT_ACCUM{1} /
                                CVT_INTEGRAL2ACCUM(MLO_POOLING_KERNEL_SZ0 * MLO_POOLING_KERNEL_SZ1);
            }

            if constexpr(USE_MASK)
                mask_private[k][l] = 0;

            for(unsigned int j = 0; j < MLO_POOLING_KERNEL_SZ1; j++)
            {
                for(unsigned int i = 0; i < MLO_POOLING_KERNEL_SZ0; i++)
                {

                    FLOAT_ACCUM bot_val = CVT_FLOAT2ACCUM(
                        bot_data[j + k * MLO_POOLING_STRIDE1][i + l * MLO_POOLING_STRIDE0]);

                    if constexpr(USE_MASK)
                    {
                        if(bot_val > res[k][l])
                        {
                            res[k][l] = bot_val;

                            if constexpr(USE_IMG_INDEX)
                            {
                                mask_private[k][l] = (hstart1 + j) * mlo_bot_width + (wstart1 + i);
                            }
                            else
                            {
                                mask_private[k][l] = i + MLO_POOLING_KERNEL_SZ0 * j;
                            }
                        }
                    }
                    else
                    {
                        res[k][l] = MLO_POOLING_OP(res[k][l], bot_val);
                    }
                }
            }

            if constexpr(AVERAGE_OPS)
                res[k][l] *= inv_pool_size;
        }
    }

    const unsigned int top_y = (y + lcl_id1 * MLO_POOLING_N_VERT_OUT_PIX);
    const unsigned int top_x = (x + lcl_id0 * MLO_POOLING_N_HORIZ_OUT_PIX);
    const unsigned int top_off =
        b * mlo_top_batch_str + o * mlo_top_channel_str + top_y * mlo_top_str + top_x;

    for(unsigned int k = 0; k < MLO_POOLING_N_VERT_OUT_PIX; k++)
    {
        for(unsigned int l = 0; l < MLO_POOLING_N_HORIZ_OUT_PIX; l++)
        {
            if(top_y + k < mlo_top_height && top_x + l < mlo_top_width)
            {
                top[top_off + k * mlo_top_str + l] = CVT_ACCUM2FLOAT(res[k][l]);
                if constexpr(USE_MASK)
                    mask[top_off + k * mlo_top_str + l] = mask_private[k][l];
            }
        }
    }
}
