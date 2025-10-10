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

using arg_size_t = unsigned long;

extern "C" __global__ void mloPoolingForwardNaive(const FLOAT* bot_ptr,
                                                  FLOAT* top_ptr,
                                                  index_t* mask_ptr,
                                                  int save_index,
                                                  int index_mode,
                                                  unsigned int filter_d,
                                                  unsigned int filter_h,
                                                  unsigned int filter_w,
                                                  unsigned int filter_d_stride,
                                                  unsigned int filter_h_stride,
                                                  unsigned int filter_w_stride,
                                                  unsigned int filter_d_pad,
                                                  unsigned int filter_h_pad,
                                                  unsigned int filter_w_pad,
                                                  unsigned int all_n,
                                                  unsigned int all_c,
                                                  unsigned int bot_d,
                                                  unsigned int bot_h,
                                                  unsigned int bot_w,
                                                  arg_size_t bot_n_stride,
                                                  arg_size_t bot_c_stride,
                                                  unsigned int bot_d_stride,
                                                  unsigned int bot_h_stride,
                                                  unsigned int bot_w_stride,
                                                  unsigned int top_d,
                                                  unsigned int top_h,
                                                  unsigned int top_w,
                                                  arg_size_t top_n_stride,
                                                  arg_size_t top_c_stride,
                                                  unsigned int top_d_stride,
                                                  unsigned int top_h_stride,
                                                  unsigned int top_w_stride,
                                                  arg_size_t mask_n_stride,
                                                  arg_size_t mask_c_stride,
                                                  unsigned int mask_d_stride,
                                                  unsigned int mask_h_stride,
                                                  unsigned int mask_w_stride)
{
    const unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if(!(b < all_n))
        return;

    const unsigned int o = blockIdx.y * blockDim.y + threadIdx.y;
    if(!(o < all_c))
        return;

    const auto inner_loop = [&](const unsigned int k, const unsigned int j) {
        for(unsigned int i = 0; i < top_w; ++i)
        {
            const int int_dstart = k * filter_d_stride - filter_d_pad;
            const int int_hstart = j * filter_h_stride - filter_h_pad;
            const int int_wstart = i * filter_w_stride - filter_w_pad;
            const unsigned int dend =
                static_cast<unsigned int>(min(int_dstart + (int)filter_d, (int)bot_d));
            const unsigned int hend =
                static_cast<unsigned int>(min(int_hstart + (int)filter_h, (int)bot_h));
            const unsigned int wend =
                static_cast<unsigned int>(min(int_wstart + (int)filter_w, (int)bot_w));
            const unsigned int dstart = static_cast<unsigned int>(max(int_dstart, 0));
            const unsigned int hstart = static_cast<unsigned int>(max(int_hstart, 0));
            const unsigned int wstart = static_cast<unsigned int>(max(int_wstart, 0));

            unsigned int pool_size = 0;

            if constexpr(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE)
            {
                pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
                pool_size = (pool_size == 0) ? 1 : pool_size;
            }
            else if constexpr(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE)
                pool_size = filter_w * filter_h * filter_d;

            FLOAT_ACCUM res = AVERAGE_OPS ? FLOAT_ACCUM{0} : /* MAX */ FLOAT_ACCUM{-MAX_VAL_ACCUM};
            bool found      = false; // May remain false if bot contains only NaNs/-INFs.
            unsigned int d_save = 0;
            unsigned int h_save = 0;
            unsigned int w_save = 0;

            for(unsigned int d = dstart; d < dend; ++d)
            {
                for(unsigned int h = hstart; h < hend; ++h)
                {
                    for(unsigned int w = wstart; w < wend; ++w)
                    {
                        const size_t bot_index = b * bot_n_stride                        //
                                                 + o * bot_c_stride                      //
                                                 + static_cast<size_t>(d * bot_d_stride) //
                                                 + static_cast<size_t>(h * bot_h_stride) //
                                                 + static_cast<size_t>(w * bot_w_stride);
                        if constexpr(AVERAGE_OPS)
                        {
                            res += bot_ptr[bot_index];
                        }
                        else // MAX
                        {
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
                        }
                    }
                }
            }

            if constexpr(AVERAGE_OPS)
            {
                res *= CVT_FP32_2ACCUM(1.f) / static_cast<FLOAT_ACCUM>(pool_size);
            }
            else // MAX
            {
                if(save_index)
                {
                    index_t res_index = 0;

                    /// Preventing overflow during computation of res_index:
                    /// If Index is shorter than unsigned int, then let's perform computation in
                    /// 32-bit domain and then convert to narrower Index. That would reduce the
                    /// probability of overflow. If Index is wider then 32 bits, then it seems like
                    /// it is better to convert to Index type before multiplication. However this is
                    /// not actually necessary, see \ref multiply_dims_overflow_assumption. Let's
                    /// always compute in 32 bits and then convert.

                    if(found)
                    {
                        if(index_mode == 1)
                            res_index = static_cast<index_t>(d_save * bot_h * bot_w //
                                                             + h_save * bot_w       //
                                                             + w_save);
                        else
                            res_index = static_cast<index_t>( //
                                ((d_save - k * filter_d_stride + filter_d_pad) * filter_w *
                                 filter_h)                                                   //
                                + ((h_save - j * filter_h_stride + filter_h_pad) * filter_w) //
                                + (w_save - i * filter_w_stride + filter_w_pad)              //
                            );
                    }

                    const size_t mask_index = b * mask_n_stride                        //
                                              + o * mask_c_stride                      //
                                              + static_cast<size_t>(k * mask_d_stride) //
                                              + static_cast<size_t>(j * mask_h_stride) //
                                              + static_cast<size_t>(i * mask_w_stride);
                    mask_ptr[mask_index] = res_index;
                }
            }

            const size_t top_index = b * top_n_stride                        //
                                     + o * top_c_stride                      //
                                     + static_cast<size_t>(k * top_d_stride) //
                                     + static_cast<size_t>(j * top_h_stride) //
                                     + static_cast<size_t>(i * top_w_stride);

            top_ptr[top_index] = static_cast<FLOAT>(res);
        }
    };

    if constexpr(MLO_POOLING_IS2D_KERNEL)
    {
        // When we want 2D kernel, run only inner loop.
        // Fix k to 0 and take current j from the grid.
        const unsigned int k = 0; // top_d == 1
        const unsigned int j = blockIdx.z * blockDim.z + threadIdx.z;
        if(!(j < top_h))
            return;

        inner_loop(k, j);
    }
    else
    {
        const unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;
        if(!(k < top_d))
            return;

        for(unsigned int j = 0; j < top_h; ++j)
        {
            inner_loop(k, j);
        }
    }
}
