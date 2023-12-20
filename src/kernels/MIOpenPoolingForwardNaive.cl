/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#if(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE) || (MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE)
#define AVERAGE_OPS 1
#else
#define AVERAGE_OPS 0
#endif

// Let's use extended-precision accumulator only in FP16 pooling and only for averaging.
// For all other ops and datatypes, use native accumulator, i.e. treate FLOAT_ACCUM as FLOAT.
#if !(AVERAGE_OPS && MIOPEN_USE_FP16)
#define MIOPEN_USE_NATIVE_DATATYPE_ACCUM 1
#endif
#include "float_types.h"

#ifndef MLO_POOLING_IS2D_KERNEL
#error "MLO_POOLING_IS2D_KERNEL must be defined"
#endif

#if AVERAGE_OPS
#define ARG_UNUSED_FOR_AVERAGE __attribute__((__unused__))
#else
#define ARG_UNUSED_FOR_AVERAGE
#endif

#if MLO_POOLING_IS2D_KERNEL
#define ARG_UNUSED_FOR_2D __attribute__((__unused__))
#else
#define ARG_UNUSED_FOR_2D
#endif

typedef unsigned long arg_size_t;

__kernel void mloPoolingForwardNaive(const __global _FLOAT* bot_ptr,
                                     __global _FLOAT* top_ptr,
                                     ARG_UNUSED_FOR_AVERAGE __global index_t* mask_ptr,
                                     ARG_UNUSED_FOR_AVERAGE int save_index,
                                     ARG_UNUSED_FOR_AVERAGE int index_mode,
                                     uint filter_d,
                                     uint filter_h,
                                     uint filter_w,
                                     uint filter_d_stride,
                                     uint filter_h_stride,
                                     uint filter_w_stride,
                                     uint filter_d_pad,
                                     uint filter_h_pad,
                                     uint filter_w_pad,
                                     uint all_n,
                                     uint all_c,
                                     uint bot_d,
                                     uint bot_h,
                                     uint bot_w,
                                     arg_size_t bot_n_stride,
                                     arg_size_t bot_c_stride,
                                     uint bot_d_stride,
                                     uint bot_h_stride,
                                     uint bot_w_stride,
                                     ARG_UNUSED_FOR_2D uint top_d,
                                     uint top_h,
                                     uint top_w,
                                     arg_size_t top_n_stride,
                                     arg_size_t top_c_stride,
                                     uint top_d_stride,
                                     uint top_h_stride,
                                     uint top_w_stride,
                                     ARG_UNUSED_FOR_AVERAGE arg_size_t mask_n_stride,
                                     ARG_UNUSED_FOR_AVERAGE arg_size_t mask_c_stride,
                                     ARG_UNUSED_FOR_AVERAGE uint mask_d_stride,
                                     ARG_UNUSED_FOR_AVERAGE uint mask_h_stride,
                                     ARG_UNUSED_FOR_AVERAGE uint mask_w_stride)
{
    const uint b = get_global_id(0);
    if(!(b < all_n))
        return;

    const uint o = get_global_id(1);
    if(!(o < all_c))
        return;

#if MLO_POOLING_IS2D_KERNEL
    // When we want 2D kernel, run only inner loop.
    // Fix k to 0 and take current j from the grid.
    const uint k = 0; // top_d == 1
    const uint j = get_global_id(2);
    if(!(j < top_h))
        return;
#else
    const uint k = get_global_id(2);
    if(!(k < top_d))
        return;
    for(uint j = 0; j < top_h; ++j)
    {
#endif
    for(uint i = 0; i < top_w; ++i)
    {
        const int int_dstart = k * filter_d_stride - filter_d_pad;
        const int int_hstart = j * filter_h_stride - filter_h_pad;
        const int int_wstart = i * filter_w_stride - filter_w_pad;
        const uint dend      = (uint)min(int_dstart + (int)filter_d, (int)bot_d);
        const uint hend      = (uint)min(int_hstart + (int)filter_h, (int)bot_h);
        const uint wend      = (uint)min(int_wstart + (int)filter_w, (int)bot_w);
        const uint dstart    = (uint)max(int_dstart, 0);
        const uint hstart    = (uint)max(int_hstart, 0);
        const uint wstart    = (uint)max(int_wstart, 0);

#if MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
        uint pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
        pool_size      = (pool_size == 0) ? 1 : pool_size;
#elif MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE
            const uint pool_size = filter_w * filter_h * filter_d;
#endif

#if AVERAGE_OPS
        _FLOAT_ACCUM res = (_FLOAT_ACCUM)(0);
#else // MAX
            _FLOAT_ACCUM res     = (_FLOAT_ACCUM)(-MAX_VAL_ACCUM);
            bool found           = false; // May remain false if bot contains only NaNs/-INFs.
            uint d_save          = 0;
            uint h_save          = 0;
            uint w_save          = 0;
#endif
        for(uint d = dstart; d < dend; ++d)
        {
            for(uint h = hstart; h < hend; ++h)
            {
                for(uint w = wstart; w < wend; ++w)
                {
                    const size_t bot_index = b * bot_n_stride             //
                                             + o * bot_c_stride           //
                                             + (size_t)(d * bot_d_stride) //
                                             + (size_t)(h * bot_h_stride) //
                                             + (size_t)(w * bot_w_stride);
#if AVERAGE_OPS
                    res += bot_ptr[bot_index];
#else // MAX
                        if(bot_ptr[bot_index] > res)
                        {
                            res = bot_ptr[bot_index];
                            if(save_index)
                            {
                                found  = true;
                                d_save = d;
                                h_save = h;
                                w_save = w;
                            }
                        }
#endif
                }
            }
        }

#if AVERAGE_OPS
        res *= CVT_FP32_2ACCUM(1.f) / (_FLOAT_ACCUM)pool_size;
#else // MAX
            if(save_index)
            {
                index_t res_index = 0;

                /// Preventing overflow during computation of res_index:
                /// If Index is shorter than uint, then let's perform computation in 32-bit
                /// domain and then convert to narrower Index. That would reduce the probability of
                /// overflow. If Index is wider then 32 bits, then it seems like it is better to
                /// convert to Index type before multiplication. However this is not actually
                /// necessary, see \ref multiply_dims_overflow_assumption. Let's always compute in
                /// 32 bits and then convert.

                if(found)
                {
                    if(index_mode == 1)
                        res_index = (index_t)(d_save * bot_h * bot_w //
                                              + h_save * bot_w       //
                                              + w_save);
                    else
                        res_index = (index_t)(                                                    //
                            ((d_save - k * filter_d_stride + filter_d_pad) * filter_w * filter_h) //
                            + ((h_save - j * filter_h_stride + filter_h_pad) * filter_w)          //
                            + (w_save - i * filter_w_stride + filter_w_pad)                       //
                        );
                }

                const size_t mask_index = b * mask_n_stride             //
                                          + o * mask_c_stride           //
                                          + (size_t)(k * mask_d_stride) //
                                          + (size_t)(j * mask_h_stride) //
                                          + (size_t)(i * mask_w_stride);
                mask_ptr[mask_index] = res_index;
            }
#endif
        const size_t top_index = b * top_n_stride             //
                                 + o * top_c_stride           //
                                 + (size_t)(k * top_d_stride) //
                                 + (size_t)(j * top_h_stride) //
                                 + (size_t)(i * top_w_stride);

        top_ptr[top_index] = (_FLOAT)res;
    }
#if !MLO_POOLING_IS2D_KERNEL
}
#endif
}
