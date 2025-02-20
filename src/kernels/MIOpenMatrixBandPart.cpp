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

#ifndef N_TYPE
#define N_TYPE int32_t
#endif

template <typename T, typename Tn>
__device__ void matrixBandPart(const T* __restrict__ input,
                               T* __restrict__ output,
                               const Tn* __restrict__ num_lower,
                               const Tn* __restrict__ num_upper,
                               uint64_t numel,
                               uint64_t num_dim,
                               tensor_view_t<5> input_tv,
                               tensor_view_t<5> output_tv)
{
    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= numel)
    {
        return;
    }

    int64_t w = gid % input_tv.size[num_dim - 1];
    int64_t h = (gid / input_tv.size[num_dim - 1]) % input_tv.size[num_dim - 2];

    int64_t num_lower_val = static_cast<int64_t>(num_lower[0]);
    int64_t num_upper_val = static_cast<int64_t>(num_upper[0]);
    int64_t diff          = h - w;

    bool in_band = (num_lower_val < 0 || diff <= num_lower_val) &&
                   (num_upper_val < 0 || (-diff) <= num_upper_val);

    tensor_layout_t<5> layout(input_tv, gid);

    output[output_tv.get_tensor_view_idx(layout)] =
        in_band ? input[input_tv.get_tensor_view_idx(layout)] : static_cast<T>(0);
}

extern "C" __global__ void MatrixBandPart(const D_TYPE* __restrict__ input,
                                          D_TYPE* __restrict__ output,
                                          const N_TYPE* __restrict__ num_lower,
                                          const N_TYPE* __restrict__ num_upper,
                                          uint64_t numel,
                                          uint64_t num_dim,
                                          tensor_view_t<5> input_tv,
                                          tensor_view_t<5> output_tv)
{
    matrixBandPart<D_TYPE, N_TYPE>(
        input, output, num_lower, num_upper, numel, num_dim, input_tv, output_tv);
}

template <typename T, typename Tn>
__device__ void matrixBandPartContiguous(const T* __restrict__ input,
                                         T* __restrict__ output,
                                         const Tn* __restrict__ num_lower,
                                         const Tn* __restrict__ num_upper,
                                         uint64_t numel,
                                         int64_t W,
                                         int64_t H)
{
    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= numel)
    {
        return;
    }

    int64_t w = gid % W;
    int64_t h = (gid / W) % H;

    int64_t num_lower_val = static_cast<int64_t>(num_lower[0]);
    int64_t num_upper_val = static_cast<int64_t>(num_upper[0]);
    int64_t diff          = h - w;

    bool in_band = (num_lower_val < 0 || diff <= num_lower_val) &&
                   (num_upper_val < 0 || (-diff) <= num_upper_val);

    output[gid] = in_band ? input[gid] : static_cast<T>(0);
}

extern "C" __global__ void MatrixBandPartContiguous(const D_TYPE* __restrict__ input,
                                                    D_TYPE* __restrict__ output,
                                                    const N_TYPE* __restrict__ num_lower,
                                                    const N_TYPE* __restrict__ num_upper,
                                                    uint64_t numel,
                                                    int64_t W,
                                                    int64_t H)
{
    matrixBandPartContiguous<D_TYPE, N_TYPE>(input, output, num_lower, num_upper, numel, W, H);
}
