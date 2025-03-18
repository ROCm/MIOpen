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
#include "hip_atomic.hpp"
#include "tensor_view.hpp"

template <typename DTYPE>
__device__ void pad_reflection_1d_fwd(const DTYPE* __restrict__ input,
                                      DTYPE* __restrict__ output,
                                      int64_t padding_l,
                                      uint64_t output_size,
                                      tensor_view_t<3> input_tv,
                                      tensor_view_t<3> output_tv)
{
    const int64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= output_size)
        return;

    int64_t in_W = input_tv.size[2];

    auto i_tensor_layout = tensor_layout_t<3>(output_tv, gid);

    int64_t in_start_x  = max(0L, -padding_l);
    int64_t out_start_x = max(0L, padding_l);

    int64_t w = i_tensor_layout.layout[2];

    i_tensor_layout.layout[2] = (w < padding_l)          ? (2 * padding_l - w)
                                : (w < in_W + padding_l) ? w
                                                         : (2 * (in_W + padding_l - 1) - w);

    i_tensor_layout.layout[2] = i_tensor_layout.layout[2] - out_start_x + in_start_x;

    auto o_tensor_layout = tensor_layout_t<3>(output_tv, gid);
    output[output_tv.get_tensor_view_idx(o_tensor_layout)] =
        input[input_tv.get_tensor_view_idx(i_tensor_layout)];
}

extern "C" __global__ void PadReflection1dFwd(const IO_TYPE* __restrict__ input,
                                              IO_TYPE* __restrict__ output,
                                              int64_t padding_l,
                                              uint64_t output_size,
                                              tensor_view_t<3> input_tv,
                                              tensor_view_t<3> output_tv)
{
    pad_reflection_1d_fwd<IO_TYPE>(input, output, padding_l, output_size, input_tv, output_tv);
}

template <typename DTYPE>
__device__ void pad_reflection_1d_bwd(DTYPE* __restrict__ input_grad,
                                      const DTYPE* __restrict__ output_grad,
                                      long padding_l,
                                      uint64_t output_grad_size,
                                      tensor_view_t<3> input_grad_tv,
                                      tensor_view_t<3> output_grad_tv)
{
    const int64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= output_grad_size)
        return;

    int64_t in_W = input_grad_tv.size[2];

    auto o_tensor_layout = tensor_layout_t<3>(output_grad_tv, gid);
    auto i_tensor_layout = tensor_layout_t<3>(output_grad_tv, gid);

    int64_t in_start_x  = max(0L, -padding_l);
    int64_t out_start_x = max(0L, padding_l);

    int64_t w = i_tensor_layout.layout[2];

    w = (w < padding_l)          ? (2 * padding_l - w)
        : (w < in_W + padding_l) ? w
                                 : (2 * (in_W + padding_l - 1) - w);

    i_tensor_layout.layout[2] = w - out_start_x + in_start_x;

    atomic_add_g(&input_grad[input_grad_tv.get_tensor_view_idx(i_tensor_layout)],
                 output_grad[output_grad_tv.get_tensor_view_idx(o_tensor_layout)]);
}

extern "C" __global__ void PadReflection1dBwd(IO_TYPE* __restrict__ input_grad,
                                              const IO_TYPE* __restrict__ output_grad,
                                              long padding_l,
                                              uint64_t output_grad_size,
                                              tensor_view_t<3> input_grad_tv,
                                              tensor_view_t<3> output_grad_tv)
{
    pad_reflection_1d_bwd<IO_TYPE>(
        input_grad, output_grad, padding_l, output_grad_size, input_grad_tv, output_grad_tv);
}
