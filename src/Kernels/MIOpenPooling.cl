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

#define _FLOAT float
#define _FLOAT2 float2
#define _FLOAT4 float4
#define _FLOAT8 float8
#define _INT_MASK_GLOBAL uchar
#define _INT_MASK_LOCAL uchar

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F /* max value */
#endif

#define UNUSED __attribute__((__unused__))

#define MLO_POOLING_OP_AVE 0
#define MLO_POOLING_OP_MAX 1
#define MLO_POOLING_OP_STC 2

#define MLO_POOLING_GROUP_SZ2 1

#ifndef MLO_POOLING_OP_ID
#define MLO_POOLING_OP_ID 0
#endif
// max
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
#define MLO_POOLING_OP(A, B) fmax(A, B);
#elif MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
#define MLO_POOLING_OP(A, B) (A + B);
#endif

#define MLO_POOLING_LCL_DATA_WIDTH                                               \
    (MLO_POOLING_GROUP_SZ0 * MLO_POOLING_N_HORIZ_OUT_PIX * MLO_POOLING_STRIDE0 + \
     MLO_POOLING_KERNEL_SZ0 - 1)
#define MLO_POOLING_LCL_DATA_HEIGHT                                             \
    (MLO_POOLING_GROUP_SZ1 * MLO_POOLING_N_VERT_OUT_PIX * MLO_POOLING_STRIDE1 + \
     MLO_POOLING_KERNEL_SZ1 - 1)

__attribute__((reqd_work_group_size(MLO_POOLING_GROUP_SZ0,
                                    MLO_POOLING_GROUP_SZ1,
                                    MLO_POOLING_GROUP_SZ2))) __kernel void
mloPooling(const __global _FLOAT* bot,
           __global _FLOAT* top,
#if !defined(MLO_POOLING_DO_BACKWARD) || MLO_POOLING_OP_ID != MLO_POOLING_OP_MAX
           UNUSED
#endif
               __global _INT_MASK_GLOBAL* mask)
{
    __local _FLOAT bot_data[MLO_POOLING_LCL_DATA_WIDTH * MLO_POOLING_LCL_DATA_HEIGHT];

    int x       = get_group_id(0) * MLO_POOLING_GROUP_SZ0 * MLO_POOLING_N_HORIZ_OUT_PIX;
    int y       = get_group_id(1) * MLO_POOLING_GROUP_SZ1 * MLO_POOLING_N_VERT_OUT_PIX;
    int lcl_id0 = get_local_id(0);
    int lcl_id1 = get_local_id(1);
    //		int lcl_id = (lcl_id1 << MLO_POOLING_GROUP_LG2SZ0) + lcl_id0;
    int ob      = get_global_id(2); // output * batch_sz
    int b       = ob / MLO_POOLING_N_OUTPUTS;
    int o       = ob - b * MLO_POOLING_N_OUTPUTS;
    int bot_x   = x * MLO_POOLING_STRIDE0;
    int bot_y   = y * MLO_POOLING_STRIDE1;
    int bot_off = b * MLO_POOLING_BOT_BATCH_STRIDE + o * MLO_POOLING_BOT_CHANNEL_STRIDE;

    _FLOAT res[MLO_POOLING_N_VERT_OUT_PIX][MLO_POOLING_N_HORIZ_OUT_PIX];
#if defined(MLO_POOLING_DO_BACKWARD) && MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
    _INT_MASK_LOCAL mask_private[MLO_POOLING_N_VERT_OUT_PIX][MLO_POOLING_N_HORIZ_OUT_PIX];
#endif
    for(int k = 0; k < MLO_POOLING_N_VERT_OUT_PIX; k++)
    {
        for(int l = 0; l < MLO_POOLING_N_HORIZ_OUT_PIX; l++)
        {
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
            res[k][l] = -FLT_MAX;
#elif MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
            res[k][l] = 0;
#endif
        }
    }

    // load tile
    for(int b_j = lcl_id1; b_j < MLO_POOLING_LCL_DATA_HEIGHT; b_j += MLO_POOLING_GROUP_SZ1)
    {
        int bot_y_act = bot_y + b_j - MLO_POOLING_PAD1;

        bool invisibleY = (bot_y_act < 0) || (bot_y_act >= MLO_POOLING_BOT_HEIGHT);

        int bot_y_off = bot_y_act * MLO_POOLING_BOT_STRIDE;

        int lcl_off_v = b_j * MLO_POOLING_LCL_DATA_WIDTH;

        for(int b_i = lcl_id0; b_i < MLO_POOLING_LCL_DATA_WIDTH; b_i += MLO_POOLING_GROUP_SZ0)
        {

            int bot_x_act = bot_x + b_i - MLO_POOLING_PAD0;

            bool invisibleX = (bot_x_act < 0) || (bot_x_act >= MLO_POOLING_BOT_WIDTH);
            bool invisible  = invisibleX || invisibleY;

            int bot_idx    = invisible ? 0 : (bot_off + bot_y_off + bot_x_act);
            _FLOAT bot_val = bot[bot_idx];

            bot_val = (invisible) ?

#if MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
                                  -FLT_MAX
#elif MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
                                  0
#endif
                                  : bot_val;
            bot_data[lcl_off_v + b_i] = bot_val;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int lcl_y   = lcl_id1 * MLO_POOLING_N_VERT_OUT_PIX * MLO_POOLING_STRIDE1;
    int lcl_x   = lcl_id0 * MLO_POOLING_N_HORIZ_OUT_PIX * MLO_POOLING_STRIDE0;
    int lcl_off = lcl_y * MLO_POOLING_LCL_DATA_WIDTH + lcl_x;

    for(int k = 0; k < MLO_POOLING_N_VERT_OUT_PIX; k++)
    {
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
        int y_dst  = y + lcl_id1 * MLO_POOLING_N_VERT_OUT_PIX + k;
        int hstart = y_dst * MLO_POOLING_STRIDE1 - MLO_POOLING_PAD1;
        int hend = min(hstart + MLO_POOLING_KERNEL_SZ1, MLO_POOLING_BOT_HEIGHT + MLO_POOLING_PAD1);
#endif
        for(int l = 0; l < MLO_POOLING_N_HORIZ_OUT_PIX; l++)
        {

#if MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
            int x_dst  = x + lcl_id0 * MLO_POOLING_N_HORIZ_OUT_PIX + l;
            int wstart = x_dst * MLO_POOLING_STRIDE0 - MLO_POOLING_PAD0;
            int wend =
                min(wstart + MLO_POOLING_KERNEL_SZ0, MLO_POOLING_BOT_WIDTH + MLO_POOLING_PAD0);
            int pool_size = (hend - hstart) * (wend - wstart);
#endif
#if defined(MLO_POOLING_DO_BACKWARD) && MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
            mask_private[k][l] = 0xFF;
#endif

            for(int j = 0; j < MLO_POOLING_KERNEL_SZ1; j++)
            {
                for(int i = 0; i < MLO_POOLING_KERNEL_SZ0; i++)
                {

                    _FLOAT bot_val =
                        bot_data[lcl_off +
                                 (k * MLO_POOLING_STRIDE1 + j) * MLO_POOLING_LCL_DATA_WIDTH +
                                 (l * MLO_POOLING_STRIDE0 + i)];
#if 0
						if (y_dst == 0 && x_dst == 6)
						{

							printf("k: %d %f %f\n",
								lcl_off + (k * MLO_POOLING_STRIDE1+j)*MLO_POOLING_LCL_DATA_WIDTH + (l * MLO_POOLING_STRIDE0+i),
								res[k][l],
								bot_val
								);
						}
#endif
#if defined(MLO_POOLING_DO_BACKWARD) && MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
                    if(bot_val > res[k][l])
                    {
                        res[k][l]          = bot_val;
                        mask_private[k][l] = i + MLO_POOLING_KERNEL_SZ0 * j;
                    }
#else
                    res[k][l] = MLO_POOLING_OP(res[k][l], bot_val);
#endif
                }
            }

#if MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
            res[k][l] *= (_FLOAT)1.f / (_FLOAT)pool_size;
#endif
        }
    }

    int top_y   = (y + lcl_id1 * MLO_POOLING_N_VERT_OUT_PIX);
    int top_x   = (x + lcl_id0 * MLO_POOLING_N_HORIZ_OUT_PIX);
    int top_off = b * MLO_POOLING_TOP_BATCH_STRIDE + o * MLO_POOLING_TOP_CHANNEL_STRIDE +
                  top_y * MLO_POOLING_TOP_STRIDE + top_x;
    for(int k = 0; k < MLO_POOLING_N_VERT_OUT_PIX; k++)
    {
        for(int l = 0; l < MLO_POOLING_N_HORIZ_OUT_PIX; l++)
        {
            if(top_y + k < MLO_POOLING_TOP_HEIGHT && top_x + l < MLO_POOLING_TOP_WIDTH)
            {
                top[top_off + k * MLO_POOLING_TOP_STRIDE + l] = res[k][l];
#if defined(MLO_POOLING_DO_BACKWARD) && MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
                mask[top_off + k * MLO_POOLING_TOP_STRIDE + l] = mask_private[k][l];
#endif
            }
        }
    }
}
