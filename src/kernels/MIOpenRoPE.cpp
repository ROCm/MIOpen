/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "tensor_view.hpp"

template <typename TI, typename TO>
__device__ void ropefwdcontiguous(const TI* __restrict__ x,
                                  const TI* __restrict__ scaled_freqs_cos,
                                  const TI* __restrict__ scaled_freqs_sin,
                                  TO* __restrict__ y,
                                  tensor_view_t<5> x_tv,
                                  tensor_view_t<3> scaled_freqs_cos_tv,
                                  tensor_view_t<3> scaled_freqs_sin_tv,
                                  tensor_view_t<5> y_tv,
                                  uint64_t output_numele)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= output_numel)
        return;

    tensor_layout_t<5> ncdhw(x_tv, gid);

    FLOAT_ACCUM input = CVT_FLOAT2ACCUM(x[x_tv.get_tensor_view_idx(ncdhw)]);
    FLOAT_ACCUM input_rotate_half =
        (ncdhw.layout[4] % 2 == 0)
            ? CVT_FLOAT2ACCUM(-x[x_tv.get_tensor_view_idx(ncdhw.add_tensor_layout_t(4, 1))])
            : CVT_FLOAT2ACCUM(x[x_tv.get_tensor_view_idx(ncdhw.sub_tensor_layout_t(4, 1))]);

    tensor_layout_t<3> ncw(ncdhw.layout[2], ncdhw.layout[3], ncdhw.layout[4]);

    FLOAT_ACCUM cos_val =
        CVT_FLOAT2ACCUM(scaled_freqs_cos[scaled_freqs_cos_tv.get_tensor_view_idx(ncw)]);
    FLOAT_ACCUM sin_val =
        CVT_FLOAT2ACCUM(scaled_freqs_sin[scaled_freqs_sin_tv.get_tensor_view_idx(ncw)]);

    FLOAT_ACCUM val = (input * cos_val) + (input_rotate_half * sin_val);

    y[y_tv.get_tensor_view_idx(ncdhw)] = CVT_ACCUM2FLOAT(val);
}

template <typename TI, typename TO>
__device__ void ropebwdcontiguous(const TI* __restrict__ dy,
                                  const TI* __restrict__ scaled_freqs_cos,
                                  const TI* __restrict__ scaled_freqs_sin,
                                  TO* __restrict__ dx,
                                  tensor_view_t<5> dy_tv,
                                  tensor_view_t<3> scaled_freqs_cos_tv,
                                  tensor_view_t<3> scaled_freqs_sin_tv,
                                  tensor_view_t<5> dx_tv,
                                  uint64_t output_numele)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= output_numel)
        return;

    tensor_layout_t<5> ncdhw(dy_tv, gid);

    FLOAT_ACCUM output_grad = CVT_FLOAT2ACCUM(dy[dy_tv.get_tensor_view_idx(ncdhw)]);
    FLOAT_ACCUM output_grad_rotate_half =
        (ncdhw.layout[4] % 2 == 0)
            ? CVT_FLOAT2ACCUM(dy[dy_tv.get_tensor_view_idx(ncdhw.add_tensor_layout_t(4, 1))])
            : CVT_FLOAT2ACCUM(-dy[dy_tv.get_tensor_view_idx(ncdhw.sub_tensor_layout_t(4, 1))]);

    tensor_layout_t<3> ncw(ncdhw.layout[2], ncdhw.layout[3], ncdhw.layout[4]);

    FLOAT_ACCUM cos_val =
        CVT_FLOAT2ACCUM(scaled_freqs_cos[scaled_freqs_cos_tv.get_tensor_view_idx(ncw)]);
    FLOAT_ACCUM sin_val =
        (ncw.layout[2] % 2 == 0)
            ? CVT_FLOAT2ACCUM(scaled_freqs_sin[scaled_freqs_sin_tv.get_tensor_view_idx(
                  ncw.add_tensor_layout_t(2, 1))])
            : CVT_FLOAT2ACCUM(scaled_freqs_sin[scaled_freqs_sin_tv.get_tensor_view_idx(
                  ncw.sub_tensor_layout_t(2, 1))]);

    FLOAT_ACCUM val = (output_grad * cos_val) + (output_grad_rotate_half * sin_val);

    dx[dx_tv.get_tensor_view_idx(ncdhw)] = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void RoPEFwdContiguous(const INPUT_TYPE* __restrict__ x,
                                             const INPUT_TYPE* __restrict__ scaled_freqs_cos,
                                             const INPUT_TYPE* __restrict__ scaled_freqs_sin,
                                             OUTPUT_TYPE* __restrict__ y,
                                             tensor_view_t<5> x_tv,
                                             tensor_view_t<3> scaled_freqs_cos_tv,
                                             tensor_view_t<3> scaled_freqs_sin_tv,
                                             tensor_view_t<5> y_tv,
                                             uint64_t output_numel)
{
    // instantiate the kernel
    ropefwdcontiguous<INPUT_TYPE, OUTPUT_TYPE>(x,
                                               scaled_freqs_cos,
                                               scaled_freqs_sin,
                                               y,
                                               x_tv,
                                               scaled_freqs_cos_tv,
                                               scaled_freqs_sin_tv,
                                               y_tv,
                                               output_numel);
}

extern "C" __global__ void RoPEBwdContiguous(const INPUT_TYPE* __restrict__ dy,
                                             const INPUT_TYPE* __restrict__ scaled_freqs_cos,
                                             const INPUT_TYPE* __restrict__ scaled_freqs_sin,
                                             OUTPUT_TYPE* __restrict__ dx,
                                             tensor_view_t<5> dy_tv,
                                             tensor_view_t<3> scaled_freqs_cos_tv,
                                             tensor_view_t<3> scaled_freqs_sin_tv,
                                             tensor_view_t<5> dx_tv,
                                             uint64_t output_numel)
{
    // instantiate the kernel
    ropebwdcontiguous<INPUT_TYPE, OUTPUT_TYPE>(dy,
                                               scaled_freqs_cos,
                                               scaled_freqs_sin,
                                               dx,
                                               dy_tv,
                                               scaled_freqs_cos_tv,
                                               scaled_freqs_sin_tv,
                                               dx_tv,
                                               output_numel);
}
