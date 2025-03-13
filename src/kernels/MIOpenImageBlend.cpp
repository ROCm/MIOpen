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

template <typename TIO>
__device__ TIO clamp(TIO val, TIO min, TIO max)
{
    val = val < min ? min : val;
    val = val > max ? max : val;
    return val;
}

template <typename TIO>
__device__ void DeviceRgbToGrayscale(
    const TIO* img, TIO* gray, size_t N, tensor_view_t<4> img_tv, tensor_view_t<4> gray_tv)
{
    size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid >= N)
        return;

    tensor_layout_t<4> gray_layout(gray_tv, gid);
    auto n = gray_layout.layout[0];
    auto h = gray_layout.layout[2];
    auto w = gray_layout.layout[3];

    FLOAT_ACCUM r = CVT_FLOAT2ACCUM(img[img_tv.get_tensor_view_idx({n, 0, h, w})]);
    FLOAT_ACCUM g = CVT_FLOAT2ACCUM(img[img_tv.get_tensor_view_idx({n, 1, h, w})]);
    FLOAT_ACCUM b = CVT_FLOAT2ACCUM(img[img_tv.get_tensor_view_idx({n, 2, h, w})]);

    TIO value = CVT_ACCUM2FLOAT(0.2989f * r + 0.587f * g + 0.114f * b);

    gray[gid] = value;
}

template <typename TIO>
__device__ void DeviceBlendContiguous(const TIO* img1,
                                      const TIO* img2,
                                      TIO* output,
                                      size_t n_stride,
                                      size_t c_stride,
                                      size_t N,
                                      float ratio,
                                      float bound)
{
    size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid >= N)
        return;

    const size_t n       = gid / n_stride;
    const size_t img2_id = n * c_stride + gid % c_stride;

    FLOAT_ACCUM img1_v = CVT_FLOAT2ACCUM(img1[gid]);
    FLOAT_ACCUM img2_v = CVT_FLOAT2ACCUM(img2[img2_id]);

    TIO value = CVT_ACCUM2FLOAT(clamp((ratio * img1_v + (1.0f - ratio) * img2_v), 0.0f, bound));

    output[gid] = value;
}

extern "C" __global__ void RGBToGrayscale(
    const DTYPE* img, DTYPE* gray, size_t N, tensor_view_t<4> img_tv, tensor_view_t<4> gray_tv)
{
    DeviceRgbToGrayscale(img, gray, N, img_tv, gray_tv);
}

extern "C" __global__ void BlendContiguous(const DTYPE* img1,
                                           const DTYPE* img2,
                                           DTYPE* output,
                                           size_t n_stride,
                                           size_t c_stride,
                                           size_t N,
                                           float ratio,
                                           float bound)
{
    DeviceBlendContiguous(img1, img2, output, n_stride, c_stride, N, ratio, bound);
}
