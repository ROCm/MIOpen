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

__kernel void mloPoolingForwardNaive(const __global _FLOAT* bot_ptr,
                                     __global _FLOAT* top_ptr,
                                     __global index_t* mask_ptr,
                                     int save_index,
                                     int index_mode,
                                     uint pad_d,
                                     uint pool_d_stride,
                                     uint filter_d,
                                     uint pad_h,
                                     uint pool_h_stride,
                                     uint filter_h,
                                     uint pad_w,
                                     uint pool_w_stride,
                                     uint filter_w,
                                     uint bot_d,
                                     uint bot_h,
                                     uint bot_w,
                                     uint bot_w_stride,
                                     uint bot_h_stride,
                                     uint bot_d_stride,
                                     size_t bot_c_stride,
                                     size_t bot_n_stride,
                                     uint top_d,
                                     uint top_h,
                                     uint top_w_stride,
                                     uint top_h_stride,
                                     uint top_d_stride,
                                     size_t top_c_stride,
                                     size_t top_n_stride,
                                     uint mask_w_stride,
                                     uint mask_h_stride,
                                     uint mask_d_stride,
                                     size_t mask_c_stride,
                                     size_t mask_n_stride)
{
    const uint b = get_global_id(0);
    const uint o = get_global_id(1);
    const uint i = get_global_id(2);

    for(uint j = 0; j < top_h; ++j)
    {
        for(uint k = 0; k < top_d; ++k)
        {
            _FLOAT_ACCUM res =
#if AVERAGE_OPS
                (_FLOAT_ACCUM)(0);
#else // MAX
                (_FLOAT_ACCUM)(-MAX_VAL_ACCUM);
#endif
            const int int_dstart = k * pool_d_stride - pad_d;
            const int int_hstart = j * pool_h_stride - pad_h;
            const int int_wstart = i * pool_w_stride - pad_w;
            const uint dend      = (uint)min(int_dstart + (int)filter_d, (int)bot_d);
            const uint hend      = (uint)min(int_hstart + (int)filter_h, (int)bot_h);
            const uint wend      = (uint)min(int_wstart + (int)filter_w, (int)bot_w);
            const uint dstart    = (uint)max(int_dstart, 0);
            const uint hstart    = (uint)max(int_hstart, 0);
            const uint wstart    = (uint)max(int_wstart, 0);

#if MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
            uint pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
            pool_size      = (pool_size == 0) ? 1 : pool_size;
#else // MAX or AVE_INCLUSIVE
            const uint pool_size = filter_w * filter_h * filter_d;
#endif
            bool found = false; // This may remain false if the input tensor
                                // contains only NaNs and -INFs.
            uint d_save = 0;
            uint h_save = 0;
            uint w_save = 0;
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
                        res_index = (index_t)(                                           //
                            ((d_save - k * pool_d_stride + pad_d) * filter_w * filter_h) //
                            + ((h_save - j * pool_h_stride + pad_h) * filter_w)          //
                            + (w_save - i * pool_w_stride + pad_w)                       //
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
    }
}
