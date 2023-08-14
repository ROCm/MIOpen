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

#include "pooling_functions.h"

#ifdef USE_IMG_INDEX
#if !(USE_IMG_INDEX == 0 || USE_IMG_INDEX == 1)
#error "Bad value of USE_IMG_INDEX"
#endif
#else
#define USE_IMG_INDEX 1
#endif

#if defined(MLO_POOLING_SAVE_INDEX) && (MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX)
#define USE_MASK 1
#else
#define USE_MASK 0
#endif

#if(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE) || (MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE)
#define AVERAGE_OPS 1
#else
#define AVERAGE_OPS 0
#endif

#define MLO_POOLING_GROUP_SZ2 1

#if MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
#define MLO_POOLING_OP(A, B) (fmax((A), (B)))
#elif AVERAGE_OPS
#define MLO_POOLING_OP(A, B) ((A) + (B))
#endif

#define MLO_BOT_DATA_SZ0 \
    ((MLO_POOLING_N_HORIZ_OUT_PIX - 1) * MLO_POOLING_STRIDE0 + MLO_POOLING_KERNEL_SZ0)
#define MLO_BOT_DATA_SZ1 \
    ((MLO_POOLING_N_VERT_OUT_PIX - 1) * MLO_POOLING_STRIDE1 + MLO_POOLING_KERNEL_SZ1)

// Let's use extended-precision accumulator only in FP16 pooling and only for averaging.
// For all other ops and datatypes, redefine macros used for accum-float conversion
// and accum types, so they do nothing, i.e. treate FLOAT_ACCUM as FLOAT.
#if !(AVERAGE_OPS && MIOPEN_USE_FP16)
#define MIOPEN_USE_NATIVE_DATATYPE_ACCUM 1
#endif
#include "float_types.h"

__attribute__((reqd_work_group_size(MLO_POOLING_GROUP_SZ0,
                                    MLO_POOLING_GROUP_SZ1,
                                    MLO_POOLING_GROUP_SZ2))) __kernel void
mloPoolingG(const __global _FLOAT* bot,
            __global _FLOAT* top,
#if !USE_MASK
            UNUSED
#endif
                __global index_t* mask,
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

    uint x       = get_group_id(0) * MLO_POOLING_GROUP_SZ0 * MLO_POOLING_N_HORIZ_OUT_PIX;
    uint y       = get_group_id(1) * MLO_POOLING_GROUP_SZ1 * MLO_POOLING_N_VERT_OUT_PIX;
    uint lcl_id0 = get_local_id(0);
    uint lcl_id1 = get_local_id(1);
    //		int lcl_id = (lcl_id1 << MLO_POOLING_GROUP_LG2SZ0) + lcl_id0;
    uint ob      = get_global_id(2); // output * batch_sz
    uint b       = ob / mlo_n_outputs;
    uint o       = ob - b * mlo_n_outputs;
    uint bot_x   = (x + lcl_id0 * MLO_POOLING_N_HORIZ_OUT_PIX) * MLO_POOLING_STRIDE0;
    uint bot_y   = (y + lcl_id1 * MLO_POOLING_N_VERT_OUT_PIX) * MLO_POOLING_STRIDE1;
    uint bot_off = b * mlo_bot_batch_str + o * mlo_bot_channel_str;

    _FLOAT bot_data[MLO_BOT_DATA_SZ1][MLO_BOT_DATA_SZ0];
    _FLOAT_ACCUM res[MLO_POOLING_N_VERT_OUT_PIX][MLO_POOLING_N_HORIZ_OUT_PIX];
#if USE_MASK
    index_t mask_private[MLO_POOLING_N_VERT_OUT_PIX][MLO_POOLING_N_HORIZ_OUT_PIX];
#endif
    for(int k = 0; k < MLO_POOLING_N_VERT_OUT_PIX; k++)
    {
        for(int l = 0; l < MLO_POOLING_N_HORIZ_OUT_PIX; l++)
        {
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
            res[k][l] = (_FLOAT_ACCUM)(-MAX_VAL_ACCUM);
#elif AVERAGE_OPS
            res[k][l] = (_FLOAT_ACCUM)(0);
#endif
        }
    }

    for(uint j = 0; j < MLO_BOT_DATA_SZ1; ++j)
    {
        int run_y  = (int)bot_y + j - mlo_pad1;
        bool vis_y = run_y >= 0 && run_y < mlo_bot_height;

        for(uint i = 0; i < MLO_BOT_DATA_SZ0; ++i)
        {
            int run_x        = (int)bot_x + i - mlo_pad0;
            uint bot_gbl_off = bot_off + (uint)run_y * mlo_bot_str + (uint)run_x;
            bool vis_x       = run_x >= 0 && run_x < mlo_bot_width;
            bot_data[j][i]   = vis_y && vis_x ? bot[bot_gbl_off] :
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
                                            (_FLOAT)(-MAX_VAL);
#elif AVERAGE_OPS
                                            (_FLOAT)(0);
#endif
        }
    }

#pragma unroll
    for(uint k = 0; k < MLO_POOLING_N_VERT_OUT_PIX; k++)
    {
#if(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE) || (USE_MASK && USE_IMG_INDEX)
        uint y_dst  = y + lcl_id1 * MLO_POOLING_N_VERT_OUT_PIX + k;
        int hstart1 = (int)y_dst * MLO_POOLING_STRIDE1 - mlo_pad1;
#endif
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
        int hend   = min((hstart1 + MLO_POOLING_KERNEL_SZ1), (int)mlo_bot_height);
        int hstart = max(hstart1, 0);
#endif
        for(uint l = 0; l < MLO_POOLING_N_HORIZ_OUT_PIX; l++)
        {
#if(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE) || (USE_MASK && USE_IMG_INDEX)
            uint x_dst  = x + lcl_id0 * MLO_POOLING_N_HORIZ_OUT_PIX + l;
            int wstart1 = (int)x_dst * MLO_POOLING_STRIDE0 - mlo_pad0;
#endif
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
            int wend       = min((wstart1 + MLO_POOLING_KERNEL_SZ0), (int)mlo_bot_width);
            int wstart     = max(wstart1, 0);
            uint pool_size = (hend - hstart) * (wend - wstart);
            pool_size      = (pool_size == 0) ? 1 : pool_size;
#endif
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE
            uint pool_size = MLO_POOLING_KERNEL_SZ0 * MLO_POOLING_KERNEL_SZ1;
            pool_size      = (pool_size == 0) ? 1 : pool_size;
#endif
#if USE_MASK
            mask_private[k][l] = 0;
#endif

            for(uint j = 0; j < MLO_POOLING_KERNEL_SZ1; j++)
            {
                for(uint i = 0; i < MLO_POOLING_KERNEL_SZ0; i++)
                {

                    _FLOAT_ACCUM bot_val = CVT_FLOAT2ACCUM(
                        bot_data[j + k * MLO_POOLING_STRIDE1][i + l * MLO_POOLING_STRIDE0]);

#if USE_MASK
                    if(bot_val > res[k][l])
                    {
                        res[k][l] = bot_val;
                        mask_private[k][l] =
#if USE_IMG_INDEX
                            (hstart1 + j) * mlo_bot_width + (wstart1 + i);
#else
                            i + MLO_POOLING_KERNEL_SZ0 * j;
#endif
                    }
#else
                    res[k][l] = MLO_POOLING_OP(res[k][l], bot_val);
#endif
                }
            }

#if AVERAGE_OPS
            res[k][l] *= CVT_FP32_2ACCUM(1.f) / (_FLOAT_ACCUM)pool_size;
#endif
        }
    }

    uint top_y   = (y + lcl_id1 * MLO_POOLING_N_VERT_OUT_PIX);
    uint top_x   = (x + lcl_id0 * MLO_POOLING_N_HORIZ_OUT_PIX);
    uint top_off = b * mlo_top_batch_str + o * mlo_top_channel_str + top_y * mlo_top_str + top_x;
    for(uint k = 0; k < MLO_POOLING_N_VERT_OUT_PIX; k++)
    {
        for(uint l = 0; l < MLO_POOLING_N_HORIZ_OUT_PIX; l++)
        {
            if(top_y + k < mlo_top_height && top_x + l < mlo_top_width)
            {
                top[top_off + k * mlo_top_str + l] = CVT_ACCUM2FLOAT(res[k][l]);
#if USE_MASK
                mask[top_off + k * mlo_top_str + l] = mask_private[k][l];
#endif
            }
        }
    }
}
