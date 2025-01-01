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

// RoIAlign Forward
template <typename DTYPE>
__device__ FLOAT_ACCUM bilinear_interpolate(const DTYPE* input,
                                            const long roi_batch_index,
                                            const long c,
                                            long height,
                                            long width,
                                            FLOAT_ACCUM y,
                                            FLOAT_ACCUM x,
                                            tensor_view_t<4> input_tv)
{
    long y_low;
    long x_low;
    long y_high, x_high;
    FLOAT_ACCUM ly, lx, hy, hx;

    FLOAT_ACCUM v1, v2, v3, v4;
    FLOAT_ACCUM w1, w2, w3, w4;
    FLOAT_ACCUM val;

    if(y < -1.0f || y > height || x < -1.0f || x > width)
    {
        return 0;
    }

    if(y <= 0)
    {
        y = 0;
    }
    if(x <= 0)
    {
        x = 0;
    }

    y_low = (long)y;
    x_low = (long)x;

    if(y_low >= height - 1)
    {
        y_high = y_low = height - 1;
        y              = (FLOAT_ACCUM)y_low;
    }
    else
    {
        y_high = y_low + 1;
    }

    if(x_low >= width - 1)
    {
        x_high = x_low = width - 1;
        x              = (FLOAT_ACCUM)x_low;
    }
    else
    {
        x_high = x_low + 1;
    }

    if((y_low < 0) || (y_high > height - 1) || (x_low < 0) || (x_high > width - 1) ||
       (roi_batch_index < 0) || (roi_batch_index > input_tv.size[0] - 1))
    {
        return 0;
    }

    ly = y - y_low;
    lx = x - x_low;
    hy = 1.0f - ly;
    hx = 1.0f - lx;

    v1 = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({roi_batch_index, c, y_low, x_low})]);
    v2 = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({roi_batch_index, c, y_low, x_high})]);
    v3 = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({roi_batch_index, c, y_high, x_low})]);
    v4 = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({roi_batch_index, c, y_high, x_high})]);

    w1 = hy * hx;
    w2 = hy * lx;
    w3 = ly * hx;
    w4 = ly * lx;

    val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

    return val;
}

template <typename DTYPE>
__device__ void roialign_fwd(const DTYPE* input,
                             const DTYPE* rois,
                             DTYPE* output,
                             int output_h,
                             int output_w,
                             float spatial_scale,
                             int sampling_ratio,
                             char aligned,
                             tensor_view_t<4> input_tv,
                             tensor_view_t<2> rois_tv,
                             tensor_view_t<4> output_tv)
{
    /*
     * input : input, (N, C, H, W)
     * rois : input, (K, 5)
     * output : output, (K, C, OH, OW)
     * gws = {(floor(K*C*output_h*output_w/LOCAL_SIZE)+1)*}
     * lws = {LOCAL_SIZE}
     */

    long gid = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: Pass those as arguments to avoid recomputation
    long N = input_tv.size[0];
    long C = input_tv.size[1];
    long H = input_tv.size[2];
    long W = input_tv.size[3];
    long K = rois_tv.size[0];

    if(gid > K * C * output_h * output_w - 1)
    {
        return;
    }

    long kch = gid / output_w;
    long kc  = kch / output_h;

    long pw = gid % output_w;
    long ph = kch % output_h;
    long c  = kc % C;
    long k  = kc / C;

    long roi_batch_index = CVT_FLOAT2ACCUM(rois[rois_tv.get_tensor_view_idx({k, 0})]);

    if(roi_batch_index < 0 || roi_batch_index >= N)
    {
        output[output_tv.get_tensor_view_idx({k, c, ph, pw})] = 0;

        return;
    }

    FLOAT_ACCUM roi_offset = aligned ? 0.5f : 0;

    FLOAT_ACCUM roi_start_w =
        CVT_FLOAT2ACCUM(rois[rois_tv.get_tensor_view_idx({k, 1})]) * spatial_scale - roi_offset;
    FLOAT_ACCUM roi_start_h =
        CVT_FLOAT2ACCUM(rois[rois_tv.get_tensor_view_idx({k, 2})]) * spatial_scale - roi_offset;
    FLOAT_ACCUM roi_end_w =
        CVT_FLOAT2ACCUM(rois[rois_tv.get_tensor_view_idx({k, 3})]) * spatial_scale - roi_offset;
    FLOAT_ACCUM roi_end_h =
        CVT_FLOAT2ACCUM(rois[rois_tv.get_tensor_view_idx({k, 4})]) * spatial_scale - roi_offset;

    FLOAT_ACCUM roi_width  = roi_end_w - roi_start_w;
    FLOAT_ACCUM roi_height = roi_end_h - roi_start_h;

    FLOAT_ACCUM bin_size_h;
    FLOAT_ACCUM bin_size_w;

    long roi_bin_grid_h;
    long roi_bin_grid_w;

    FLOAT_ACCUM count;

    FLOAT_ACCUM output_val = 0.0f;

    long iy, ix;

    if(!aligned)
    {
        roi_width  = fmax((FLOAT_ACCUM)roi_width, (FLOAT_ACCUM)1.0f);
        roi_height = fmax((FLOAT_ACCUM)roi_height, (FLOAT_ACCUM)1.0f);
    }
    bin_size_h = roi_height / output_h;
    bin_size_w = roi_width / output_w;

    roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / output_h);
    roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / output_w);

    roi_bin_grid_h = (isnan(roi_height) || (roi_bin_grid_h > H)) ? H : roi_bin_grid_h;
    roi_bin_grid_w = (isnan(roi_width) || (roi_bin_grid_w > W)) ? W : roi_bin_grid_w;
    bin_size_h     = (isnan(roi_height) || (bin_size_h > H)) ? H : bin_size_h;
    bin_size_w     = (isnan(roi_width) || (bin_size_w > W)) ? W : bin_size_w;

    count = roi_bin_grid_h * roi_bin_grid_w;

    for(iy = 0; iy < roi_bin_grid_h; ++iy)
    {
        const FLOAT_ACCUM y =
            roi_start_h + ph * bin_size_h + (iy + 0.5f) * bin_size_h / roi_bin_grid_h;
        for(ix = 0; ix < roi_bin_grid_w; ++ix)
        {
            const FLOAT_ACCUM x =
                roi_start_w + pw * bin_size_w + (ix + 0.5f) * bin_size_w / roi_bin_grid_w;
            FLOAT_ACCUM val =
                bilinear_interpolate<DTYPE>(input, roi_batch_index, c, H, W, y, x, input_tv);
            output_val += val;
        }
    }

    output_val /= count;
    output[output_tv.get_tensor_view_idx({k, c, ph, pw})] = CVT_ACCUM2FLOAT(output_val);
}

extern "C" __global__ void RoIAlignForward(const IO_TYPE* __restrict__ input,
                                           const IO_TYPE* __restrict__ rois,
                                           IO_TYPE* __restrict__ output,
                                           int output_h,
                                           int output_w,
                                           float spatial_scale,
                                           int sampling_ratio,
                                           bool aligned,
                                           tensor_view_t<4> input_tv,
                                           tensor_view_t<2> rois_tv,
                                           tensor_view_t<4> output_tv)
{
    roialign_fwd<IO_TYPE>(input,
                          rois,
                          output,
                          output_h,
                          output_w,
                          spatial_scale,
                          sampling_ratio,
                          aligned,
                          input_tv,
                          rois_tv,
                          output_tv);
}

// RoIAlign Backward
template <typename DTYPE>
__device__ void roialign_backward(const DTYPE* output_grad,
                                  const DTYPE* rois,
                                  DTYPE* input_grad,
                                  const uint64_t N,
                                  const uint64_t C,
                                  const uint64_t H,
                                  const uint64_t W,
                                  const uint64_t K,
                                  const uint64_t OH,
                                  const uint64_t OW,
                                  const float spatial_scale,
                                  const int64_t sampling_ratio,
                                  const bool aligned,
                                  tensor_view_t<4> output_grad_tv,
                                  tensor_view_t<2> rois_tv,
                                  tensor_view_t<4> input_grad_tv)
{
    /*
     * output_grad : input, (K, C, OH, OW)
     * rois : input, (K, 5)
     * input_grad : output, (N, C, H, W)
     * gws = {ceil(C * H * W, LOCAL_SIZE), N}
     * lws = {LOCAL_SIZE, 1}
     */

    uint64_t chw = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n   = blockIdx.y * blockDim.y + threadIdx.y;

    if(n >= N)
        return;

    uint64_t ch = chw / W, w = chw % W;
    uint64_t c = ch / H, h = ch % H;

    if(c >= C)
        return;

    // ATOMIC FREE!
    FLOAT_ACCUM p_input_grad = 0;
    for(auto k = 0; k < K; ++k)
    {
        // Check k-th roi box belongs to n-th image inside mini-batch
        // if(GET_2D_VAL_AT(rois, k, 0) != n)
        if(CVT_FLOAT2ACCUM(rois[rois_tv.get_tensor_view_idx({k, 0})]) != n)
            continue;

        // roi box
        FLOAT_ACCUM offset = aligned ? 0.5f : 0;

        FLOAT_ACCUM x1 =
            CVT_FLOAT2ACCUM(rois[rois_tv.get_tensor_view_idx({k, 1})]) * spatial_scale - offset;
        FLOAT_ACCUM y1 =
            CVT_FLOAT2ACCUM(rois[rois_tv.get_tensor_view_idx({k, 2})]) * spatial_scale - offset;
        FLOAT_ACCUM x2 =
            CVT_FLOAT2ACCUM(rois[rois_tv.get_tensor_view_idx({k, 3})]) * spatial_scale - offset;
        FLOAT_ACCUM y2 =
            CVT_FLOAT2ACCUM(rois[rois_tv.get_tensor_view_idx({k, 4})]) * spatial_scale - offset;

        FLOAT_ACCUM roi_h = y2 - y1;
        FLOAT_ACCUM roi_w = x2 - x1;

        if(!aligned)
        {
            // Force ROI to be at least 1x1
            roi_h = fmax(roi_h, (FLOAT_ACCUM)1);
            roi_w = fmax(roi_w, (FLOAT_ACCUM)1);
        }

        // bin is OH * OW cells inside ROI
        FLOAT_ACCUM bin_h = roi_h / OH;
        FLOAT_ACCUM bin_w = roi_w / OW;

        // grid is sampling_ratio_h * sampling_ratio_w cells inside bin
        // Each center of grid is sampled and avgpooled into bin
        int64_t sampling_ratio_h = sampling_ratio > 0 ? sampling_ratio : ceil(roi_h / OH);
        int64_t sampling_ratio_w = sampling_ratio > 0 ? sampling_ratio : ceil(roi_w / OW);

        for(long oh = 0; oh < OH; ++oh)
        {
            for(long ow = 0; ow < OW; ++ow)
            {
                FLOAT_ACCUM weight = 0;
                for(long r = 0; r < sampling_ratio_h; ++r)
                {
                    FLOAT_ACCUM sy = y1 + bin_h * oh + bin_h / sampling_ratio_h * (r + 0.5f);
                    if(sy < 0 || sy > H)
                        continue;

                    if(sy > H - 1)
                    {
                        sy = (FLOAT_ACCUM)(H - 1);
                    }

                    for(long s = 0; s < sampling_ratio_w; ++s)
                    {
                        FLOAT_ACCUM sx = x1 + bin_w * ow + bin_w / sampling_ratio_w * (s + 0.5f);
                        if(sx < 0 || sx > W)
                            continue;
                        if(sx > W - 1)
                        {
                            sx = (FLOAT_ACCUM)(W - 1);
                        }

                        weight += fmax((FLOAT_ACCUM)(1 - fabs(sy - h)), (FLOAT_ACCUM)0) *
                                  fmax((FLOAT_ACCUM)(1 - fabs(sx - w)), (FLOAT_ACCUM)0);
                    }
                }
                if(weight != 0)
                {
                    p_input_grad +=
                        CVT_FLOAT2ACCUM(
                            output_grad[output_grad_tv.get_tensor_view_idx({k, c, oh, ow})]) *
                        weight / (sampling_ratio_h * sampling_ratio_w);
                }
            }
        }
    }

    input_grad[input_grad_tv.get_tensor_view_idx({n, c, h, w})] = CVT_ACCUM2FLOAT(p_input_grad);
}

extern "C" __global__ void RoIAlignBackward(const IO_TYPE* output_grad,
                                            const IO_TYPE* rois,
                                            IO_TYPE* input_grad,
                                            const uint64_t N,
                                            const uint64_t C,
                                            const uint64_t H,
                                            const uint64_t W,
                                            const uint64_t K,
                                            const uint64_t OH,
                                            const uint64_t OW,
                                            const float spatial_scale,
                                            const int64_t sampling_ratio,
                                            const bool aligned,
                                            tensor_view_t<4> output_grad_tv,
                                            tensor_view_t<2> rois_tv,
                                            tensor_view_t<4> input_grad_tv)
{
    roialign_backward<IO_TYPE>(output_grad,
                               rois,
                               input_grad,
                               N,
                               C,
                               H,
                               W,
                               K,
                               OH,
                               OW,
                               spatial_scale,
                               sampling_ratio,
                               aligned,
                               output_grad_tv,
                               rois_tv,
                               input_grad_tv);
}

template <typename DTYPE>
__device__ void roialign_backward_atomic(const DTYPE* output_grad,
                                         const DTYPE* rois,
                                         DTYPE* input_grad,
                                         const uint64_t N,
                                         const uint64_t C,
                                         const uint64_t H,
                                         const uint64_t W,
                                         const uint64_t K,
                                         const uint64_t OH,
                                         const uint64_t OW,
                                         const float spatial_scale,
                                         const int64_t sampling_ratio,
                                         const bool aligned,
                                         tensor_view_t<4> output_grad_tv,
                                         tensor_view_t<2> rois_tv,
                                         tensor_view_t<4> input_grad_tv)
{
    /*
     * output_grad : input, (K, C, OH, OW)
     * rois : input, (K, 5)
     * input_grad : output, (N, C, H, W)
     * gws = {ceil(K * C * OH * OW, LOCAL_SIZE), 1}
     * lws = {LOCAL_SIZE, 1}
     */

    int64_t kchw = blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t ow = kchw % OW;
    uint64_t oh = (kchw / OW) % OH;
    uint64_t c  = (kchw / (OW * OH)) % C;
    uint64_t k  = (kchw / (C * OW * OH));
    if(k >= K)
        return;

    // Check k-th roi box belongs to n-th image inside mini-batch
    int64_t n = CVT_FLOAT2ACCUM(rois[rois_tv.get_tensor_view_idx({k, 0})]);

    if(n < 0 || n >= N)
        return;

    // roi box
    FLOAT_ACCUM offset = aligned ? 0.5f : 0;

    FLOAT_ACCUM x1 =
        CVT_FLOAT2ACCUM(rois[rois_tv.get_tensor_view_idx({k, 1})]) * spatial_scale - offset;
    FLOAT_ACCUM y1 =
        CVT_FLOAT2ACCUM(rois[rois_tv.get_tensor_view_idx({k, 2})]) * spatial_scale - offset;
    FLOAT_ACCUM x2 =
        CVT_FLOAT2ACCUM(rois[rois_tv.get_tensor_view_idx({k, 3})]) * spatial_scale - offset;
    FLOAT_ACCUM y2 =
        CVT_FLOAT2ACCUM(rois[rois_tv.get_tensor_view_idx({k, 4})]) * spatial_scale - offset;

    FLOAT_ACCUM roi_h = y2 - y1;
    FLOAT_ACCUM roi_w = x2 - x1;
    if(!aligned)
    {
        // Force ROI to be at least 1x1
        roi_h = fmax(roi_h, (FLOAT_ACCUM)1);
        roi_w = fmax(roi_w, (FLOAT_ACCUM)1);
    }

    // bin is OH * OW cells inside ROI
    FLOAT_ACCUM bin_h = roi_h / OH;
    FLOAT_ACCUM bin_w = roi_w / OW;

    // grid is sampling_ratio_h * sampling_ratio_w cells inside bin
    // Each center of grid is sampled and avgpooled into bin
    uint64_t sampling_ratio_h = sampling_ratio > 0 ? sampling_ratio : ceil(roi_h / OH);
    uint64_t sampling_ratio_w = sampling_ratio > 0 ? sampling_ratio : ceil(roi_w / OW);

    const uint64_t count = sampling_ratio_h * sampling_ratio_w;

    int64_t x_low, x_high, y_low, y_high;

    FLOAT_ACCUM ograd =
        CVT_FLOAT2ACCUM(output_grad[output_grad_tv.get_tensor_view_idx({k, c, oh, ow})]);

    for(auto r = 0; r < sampling_ratio_h; r++)
    {
        FLOAT_ACCUM y = y1 + bin_h * oh + bin_h / sampling_ratio_h * (r + 0.5f);
        if(y < 0 || y > H)
            continue;
        y_low = (int64_t)y;
        if(y_low >= H - 1)
        {
            y_high = y_low = H - 1;
            y              = (FLOAT_ACCUM)y_low;
        }
        else
        {
            y_high = y_low + 1;
        }
        for(auto s = 0; s < sampling_ratio_w; ++s)
        {
            FLOAT_ACCUM x = x1 + bin_w * ow + bin_w / sampling_ratio_w * (s + 0.5f);
            if(x < 0 || x > W)
                continue;
            x_low = (long)x;
            if(x_low >= W - 1)
            {
                x_high = x_low = W - 1;
                x              = (FLOAT_ACCUM)x_low;
            }
            else
            {
                x_high = x_low + 1;
            }

            FLOAT_ACCUM ly = y - y_low;
            FLOAT_ACCUM lx = x - x_low;
            FLOAT_ACCUM hy = 1.0 - ly;
            FLOAT_ACCUM hx = 1.0 - lx;

            FLOAT_ACCUM w1 = hy * hx;
            FLOAT_ACCUM w2 = hy * lx;
            FLOAT_ACCUM w3 = ly * hx;
            FLOAT_ACCUM w4 = ly * lx;

            FLOAT_ACCUM g1 = ograd * w1 / count;
            FLOAT_ACCUM g2 = ograd * w2 / count;
            FLOAT_ACCUM g3 = ograd * w3 / count;
            FLOAT_ACCUM g4 = ograd * w4 / count;

            if(x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0)
            {
                atomic_add_g(input_grad + input_grad_tv.get_tensor_view_idx({n, c, y_low, x_low}),
                             g1);
                atomic_add_g(input_grad + input_grad_tv.get_tensor_view_idx({n, c, y_low, x_high}),
                             g2);
                atomic_add_g(input_grad + input_grad_tv.get_tensor_view_idx({n, c, y_high, x_low}),
                             g3);
                atomic_add_g(input_grad + input_grad_tv.get_tensor_view_idx({n, c, y_high, x_high}),
                             g4);
            }
        }
    }
}

extern "C" __global__ void RoIAlignBackwardAtomic(const IO_TYPE* output_grad,
                                                  const IO_TYPE* rois,
                                                  IO_TYPE* input_grad,
                                                  const uint64_t N,
                                                  const uint64_t C,
                                                  const uint64_t H,
                                                  const uint64_t W,
                                                  const uint64_t K,
                                                  const uint64_t OH,
                                                  const uint64_t OW,
                                                  const float spatial_scale,
                                                  const int64_t sampling_ratio,
                                                  const bool aligned,
                                                  tensor_view_t<4> output_grad_tv,
                                                  tensor_view_t<2> rois_tv,
                                                  tensor_view_t<4> input_grad_tv)
{
    roialign_backward_atomic<IO_TYPE>(output_grad,
                                      rois,
                                      input_grad,
                                      N,
                                      C,
                                      H,
                                      W,
                                      K,
                                      OH,
                                      OW,
                                      spatial_scale,
                                      sampling_ratio,
                                      aligned,
                                      output_grad_tv,
                                      rois_tv,
                                      input_grad_tv);
}
