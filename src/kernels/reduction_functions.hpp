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

#ifndef GUARD_REDUCTION_FUNCTIONS_HPP
#define GUARD_REDUCTION_FUNCTIONS_HPP

#include "configuration.hpp"
#include "static_unroll.hpp"

// NOTE: This header should be independent from batchnorm_functions.hpp
// Even is in OpenCL implementation, these functions are only enabled under
// certain condition. But now, these template will not be compiled before
// calling them.
// #include "batchnorm_functions.hpp"
namespace miopen {
namespace reduction {

template <typename FloatAccum, unsigned int SizeLclData>
__forceinline__ __device__ void lds_reduce2(FloatAccum& x,
                                            FloatAccum& y,
                                            FloatAccum scale,
                                            FloatAccum (&lcl_data_x)[SizeLclData],
                                            FloatAccum (&lcl_data_y)[SizeLclData],
                                            unsigned int lid)
{
    lcl_data_x[lid] = x;
    lcl_data_y[lid] = y;
    __syncthreads();
    for(unsigned int red = (miopen::batchnorm::config::lds_size >> 1); red > 0; red >>= 1)
    {
        if(lid < red)
        {
            lcl_data_x[lid] += lcl_data_x[lid + red];
            lcl_data_y[lid] += lcl_data_y[lid + red];
        }
        __syncthreads();
    }

    x = lcl_data_x[0] * scale;
    y = lcl_data_y[0] * scale;
}

template <typename FloatAccum>
__forceinline__ __device__ void dpp_interleaved_reduction(FloatAccum& temp_sum1,
                                                          FloatAccum& temp_sum2)
{
    __asm__ volatile("s_nop 4\n"
                     "v_add_f32 %0 %0 %0 row_shr:1 bound_ctrl:0\n"
                     "v_add_f32 %1 %1 %1 row_shr:1 bound_ctrl:0\n"
                     "s_nop 0\n"
                     "v_add_f32 %0 %0 %0 row_shr:2 bound_ctrl:0\n"
                     "v_add_f32 %1 %1 %1 row_shr:2 bound_ctrl:0\n"
                     "s_nop 0\n"
                     "v_add_f32 %0 %0 %0 row_shr:4 bank_mask:0xe\n"
                     "v_add_f32 %1 %1 %1 row_shr:4 bank_mask:0xe\n"
                     "s_nop 0\n"
                     "v_add_f32 %0 %0 %0 row_shr:8 bank_mask:0xc\n"
                     "v_add_f32 %1 %1 %1 row_shr:8 bank_mask:0xc\n"
                     "s_nop 0\n"
                     "v_add_f32 %0 %0 %0 row_bcast:15 row_mask:0xa\n"
                     "v_add_f32 %1 %1 %1 row_bcast:15 row_mask:0xa\n"
                     "s_nop 0\n"
                     "v_add_f32 %0 %0 %0 row_bcast:31 row_mask:0xc\n"
                     "v_add_f32 %1 %1 %1 row_bcast:31 row_mask:0xc\n"
                     "s_nop 0"
                     : "=v"(temp_sum1), "=v"(temp_sum2)
                     : "0"(temp_sum1), "1"(temp_sum2));
}

template <typename FloatAccum, unsigned int SizeLclData>
__forceinline__ __device__ void gcn_reduce2(FloatAccum& x,
                                            FloatAccum& y,
                                            FloatAccum scale,
                                            FloatAccum (&lcl_data_x)[SizeLclData],
                                            FloatAccum (&lcl_data_y)[SizeLclData],
                                            unsigned int lid)
{
    const unsigned int ldsidx = lid >> 6;
    dpp_interleaved_reduction(x, y);
    // Last thread
    if((lid % 64) == 63)
    {
        lcl_data_x[ldsidx] = x;
        lcl_data_y[ldsidx] = y;
    }

    __syncthreads();

    x = y = 0;

    // This could be changeed to clang loop unroll(full), because the size is small
    static_unroll_count<unsigned int, 0, miopen::batchnorm::config::lds_gcn_size, 1, 2>{
        [&](unsigned int i) {
            x += lcl_data_x[i];
            y += lcl_data_y[i];
        }};

    x *= scale;
    y *= scale;
}

} // namespace reduction
} // namespace miopen

#endif
