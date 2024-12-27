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
#include "hip_atomic.hpp"

template <typename DTYPE>
__device__ DTYPE bilinear_interpolate(const DTYPE* input,
                                      const long roi_batch_index,
                                      const long c,
                                      long height,
                                      long width,
                                      DTYPE y,
                                      DTYPE x,
                                      tensor_view_t<4> input_tv)
{
    long y_low;
    long x_low;
    long y_high, x_high;
    DTYPE ly, lx, hy, hx;

    DTYPE v1, v2, v3, v4;
    DTYPE w1, w2, w3, w4;
    DTYPE val;

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
        y              = (DTYPE)y_low;
    }
    else
    {
        y_high = y_low + 1;
    }

    if(x_low >= width - 1)
    {
        x_high = x_low = width - 1;
        x              = (DTYPE)x_low;
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

    // v1 = GET_4D_VAL_AT(input, roi_batch_index, c, y_low, x_low);
    // v2 = GET_4D_VAL_AT(input, roi_batch_index, c, y_low, x_high);
    // v3 = GET_4D_VAL_AT(input, roi_batch_index, c, y_high, x_low);
    // v4 = GET_4D_VAL_AT(input, roi_batch_index, c, y_high, x_high);

    v1 = input[input_tv.get_tensor_view_idx({roi_batch_index, c, y_low, x_low})];
    v2 = input[input_tv.get_tensor_view_idx({roi_batch_index, c, y_low, x_high})];
    v3 = input[input_tv.get_tensor_view_idx({roi_batch_index, c, y_high, x_low})];
    v4 = input[input_tv.get_tensor_view_idx({roi_batch_index, c, y_high, x_high})];

    w1 = hy * hx;
    w2 = hy * lx;
    w3 = ly * hx;
    w4 = ly * lx;

    val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

    return val;
}

template <typename DTYPE>
__device__ void DeviceRoIAlignForward(const DTYPE* __restrict__ input,
                                      const DTYPE* __restrict__ rois,
                                      DTYPE* __restrict__ output,
                                      int output_h,
                                      int output_w,
                                      float spatial_scale,
                                      int sampling_ratio,
                                      bool aligned,
                                      int roi_batch_base_idx,
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

    // long gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    long gid = blockIdx.x * blockDim.x + threadIdx.x;

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

    // long roi_batch_index = (long)(GET_2D_VAL_AT(rois, k, 0));
    DTYPE roi_batch_index = rois[rois_tv.get_tensor_view_idx({k, 0})];
    roi_batch_index -= roi_batch_base_idx;

    if(roi_batch_index < 0 || roi_batch_index >= N)
    {
        // SET_4D_VAL_AT(output, k, c, ph, pw, 0);
        output[output_tv.get_tensor_view_idx({k, c, ph, pw})] = 0;
        return;
    }

    DTYPE roi_offset = aligned ? 0.5f : 0;
    // DTYPE roi_start_w = GET_2D_VAL_AT(rois, k, 1) * spatial_scale - roi_offset;
    // DTYPE roi_start_h = GET_2D_VAL_AT(rois, k, 2) * spatial_scale - roi_offset;
    // DTYPE roi_end_w   = GET_2D_VAL_AT(rois, k, 3) * spatial_scale - roi_offset;
    // DTYPE roi_end_h   = GET_2D_VAL_AT(rois, k, 4) * spatial_scale - roi_offset;

    // tensor_layout_t<2> rois_layout = tensor_layout_t<2>(rois_tv, k * C + 1);
    DTYPE roi_start_w = rois[rois_tv.get_tensor_view_idx({k, 1})] * spatial_scale - roi_offset;
    DTYPE roi_start_h = rois[rois_tv.get_tensor_view_idx({k, 2})] * spatial_scale - roi_offset;
    DTYPE roi_end_w   = rois[rois_tv.get_tensor_view_idx({k, 3})] * spatial_scale - roi_offset;
    DTYPE roi_end_h   = rois[rois_tv.get_tensor_view_idx({k, 4})] * spatial_scale - roi_offset;

    DTYPE roi_width  = roi_end_w - roi_start_w;
    DTYPE roi_height = roi_end_h - roi_start_h;

    DTYPE bin_size_h;
    DTYPE bin_size_w;

    long roi_bin_grid_h;
    long roi_bin_grid_w;

    DTYPE count;

    DTYPE output_val = 0.0f;

    long iy, ix;

    if(!aligned)
    {
        roi_width  = fmax((DTYPE)roi_width, (DTYPE)1.0f);
        roi_height = fmax((DTYPE)roi_height, (DTYPE)1.0f);
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
        const DTYPE y = roi_start_h + ph * bin_size_h + (iy + 0.5f) * bin_size_h / roi_bin_grid_h;
        for(ix = 0; ix < roi_bin_grid_w; ++ix)
        {
            const DTYPE x =
                roi_start_w + pw * bin_size_w + (ix + 0.5f) * bin_size_w / roi_bin_grid_w;
            DTYPE val =
                bilinear_interpolate<DTYPE>(input, roi_batch_index, c, H, W, y, x, input_tv);
            output_val += val;
        }
    }
    output_val /= count;
    // SET_4D_VAL_AT(output, k, c, ph, pw, output_val);
    output[output_tv.get_tensor_view_idx({k, c, ph, pw})] = output_val;
}

template <typename DTYPE>
__device__ void RoIAlignBackward(DTYPE* output_grad,
                                 DTYPE* rois,
                                 DTYPE* input_grad,
                                 long N,
                                 long C,
                                 long H,
                                 long W,
                                 long K,
                                 int OH,
                                 int OW,
                                 DTYPE spatial_scale,
                                 int sampling_ratio,
                                 char aligned,
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
    // long chw = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x,
    //      n   = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    long chw = blockIdx.x * blockDim.x + threadIdx.x;
    long n   = blockIdx.y * blockDim.y + threadIdx.y;

    long ch = chw / W, w = chw % W;
    long c = ch / H, h = ch % H;
    if(c >= C)
        return;

    // ATOMIC FREE!
    DTYPE p_input_grad = 0;
    for(long k = 0; k < K; ++k)
    {
        // Check k-th roi box belongs to n-th image inside mini-batch
        // if(GET_2D_VAL_AT(rois, k, 0) != n)
        if(rois[rois_tv.get_tensor_view_idx({k, 0})] != n)
            continue;
        // roi box
        DTYPE offset = aligned ? 0.5f : 0;
        // DTYPE y1     = GET_2D_VAL_AT(rois, k, 1) * spatial_scale - offset;
        // DTYPE x1     = GET_2D_VAL_AT(rois, k, 2) * spatial_scale - offset;
        // DTYPE y2     = GET_2D_VAL_AT(rois, k, 3) * spatial_scale - offset;
        // DTYPE x2     = GET_2D_VAL_AT(rois, k, 4) * spatial_scale - offset;

        DTYPE y1 = rois[rois_tv.get_tensor_view_idx({k, 1})] * spatial_scale - offset;
        DTYPE x1 = rois[rois_tv.get_tensor_view_idx({k, 2})] * spatial_scale - offset;
        DTYPE y2 = rois[rois_tv.get_tensor_view_idx({k, 3})] * spatial_scale - offset;
        DTYPE x2 = rois[rois_tv.get_tensor_view_idx({k, 4})] * spatial_scale - offset;

        DTYPE roi_h = x2 - x1;
        DTYPE roi_w = y2 - y1;
        if(!aligned)
        {
            // Force ROI to be at least 1x1
            // heehoon: I don't know why PyTorch do this; it seems unnecessary. I'll
            // just follow their behavior.
            roi_h = fmax(roi_h, (DTYPE)1);
            roi_w = fmax(roi_w, (DTYPE)1);
        }

        // bin is OH * OW cells inside ROI
        DTYPE bin_h = roi_h / OH;
        DTYPE bin_w = roi_w / OW;

        // grid is sampling_ratio_h * sampling_ratio_w cells inside bin
        // Each center of grid is sampled and avgpooled into bin
        long sampling_ratio_h = sampling_ratio > 0 ? sampling_ratio : ceil(roi_h / OH);
        long sampling_ratio_w = sampling_ratio > 0 ? sampling_ratio : ceil(roi_w / OW);

        for(long oh = 0; oh < OH; ++oh)
        {
            for(long ow = 0; ow < OW; ++ow)
            {
                DTYPE weight = 0;
                for(long r = 0; r < sampling_ratio_h; ++r)
                {
                    DTYPE sx = x1 + bin_h * oh + bin_h / sampling_ratio_h * (r + 0.5f);
                    sx       = fmin(fmax(sx, (DTYPE)0), (DTYPE)(H - 1));
                    for(long s = 0; s < sampling_ratio_w; ++s)
                    {
                        DTYPE sy = y1 + bin_w * ow + bin_w / sampling_ratio_w * (s + 0.5f);
                        sy       = fmin(fmax(sy, (DTYPE)0), (DTYPE)(W - 1));
                        weight += fmax((DTYPE)(1 - fabs(sx - h)), (DTYPE)0) *
                                  fmax((DTYPE)(1 - fabs(sy - w)), (DTYPE)0);
                    }
                }
                if(weight != 0)
                {
                    // p_input_grad += GET_4D_VAL_AT(output_grad, k, c, oh, ow) * weight /
                    //                 (sampling_ratio_h * sampling_ratio_w);
                    p_input_grad +=
                        output_grad[output_grad_tv.get_tensor_view_idx({k, c, oh, ow})] * weight /
                        (sampling_ratio_h * sampling_ratio_w);
                }
            }
        }
    }
    // SET_4D_VAL_AT(input_grad, n, c, h, w, p_input_grad);
    input_grad[input_grad_tv.get_tensor_view_idx({n, c, h, w})] = p_input_grad;
}

template <typename DTYPE>
__device__ void roialign_backward_atomic(DTYPE* output_grad,
                                         DTYPE* rois,
                                         DTYPE* input_grad,
                                         long N,
                                         long C,
                                         long H,
                                         long W,
                                         long K,
                                         int OH,
                                         int OW,
                                         DTYPE spatial_scale,
                                         int sampling_ratio,
                                         char aligned,
                                         int roi_batch_base_idx,
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
    // long kchw = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    long kchw = blockIdx.x * blockDim.x + threadIdx.x;
    long ow   = kchw % OW;
    long oh   = (kchw / OW) % OH;
    long c    = (kchw / (OW * OH)) % C;
    long k    = (kchw / (C * OW * OH));
    if(k >= K)
        return;

    // Check k-th roi box belongs to n-th image inside mini-batch
    // long n = GET_2D_VAL_AT(rois, k, 0);
    long n = rois[rois_tv.get_tensor_view_idx({k, 0})];
    n -= roi_batch_base_idx;
    if(n < 0 || n >= N)
    {
        return;
    }

    // roi box
    DTYPE offset = aligned ? 0.5f : 0;
    // DTYPE x1     = GET_2D_VAL_AT(rois, k, 1) * spatial_scale - offset;
    // DTYPE y1     = GET_2D_VAL_AT(rois, k, 2) * spatial_scale - offset;
    // DTYPE x2     = GET_2D_VAL_AT(rois, k, 3) * spatial_scale - offset;
    // DTYPE y2     = GET_2D_VAL_AT(rois, k, 4) * spatial_scale - offset;

    DTYPE x1 = rois[rois_tv.get_tensor_view_idx({k, 1})] * spatial_scale - offset;
    DTYPE y1 = rois[rois_tv.get_tensor_view_idx({k, 2})] * spatial_scale - offset;
    DTYPE x2 = rois[rois_tv.get_tensor_view_idx({k, 3})] * spatial_scale - offset;
    DTYPE y2 = rois[rois_tv.get_tensor_view_idx({k, 4})] * spatial_scale - offset;

    DTYPE roi_h = y2 - y1;
    DTYPE roi_w = x2 - x1;
    if(!aligned)
    {
        // Force ROI to be at least 1x1
        // heehoon: I don't know why PyTorch do this; it seems unnecessary. I'll
        // just follow their behavior.
        roi_h = fmax(roi_h, (DTYPE)1);
        roi_w = fmax(roi_w, (DTYPE)1);
    }

    // bin is OH * OW cells inside ROI
    DTYPE bin_h = roi_h / OH;
    DTYPE bin_w = roi_w / OW;

    // grid is sampling_ratio_h * sampling_ratio_w cells inside bin
    // Each center of grid is sampled and avgpooled into bin
    long sampling_ratio_h = sampling_ratio > 0 ? sampling_ratio : ceil(roi_h / OH);
    long sampling_ratio_w = sampling_ratio > 0 ? sampling_ratio : ceil(roi_w / OW);

    long count = sampling_ratio_h * sampling_ratio_w;

    long x_low, x_high, y_low, y_high;

    // DTYPE ograd = GET_4D_VAL_AT(output_grad, k, c, oh, ow);
    DTYPE ograd = output_grad[output_grad_tv.get_tensor_view_idx({k, c, oh, ow})];

    for(long r = 0; r < sampling_ratio_h; r++)
    {
        DTYPE y = y1 + bin_h * oh + bin_h / sampling_ratio_h * (r + 0.5f);
        if(y < 0 || y > H)
            continue;
        y_low = (long)y;
        if(y_low >= H - 1)
        {
            y_high = y_low = H - 1;
            y              = (DTYPE)y_low;
        }
        else
        {
            y_high = y_low + 1;
        }
        for(long s = 0; s < sampling_ratio_w; ++s)
        {
            DTYPE x = x1 + bin_w * ow + bin_w / sampling_ratio_w * (s + 0.5f);
            if(x < 0 || x > W)
                continue;
            x_low = (long)x;
            if(x_low >= W - 1)
            {
                x_high = x_low = W - 1;
                x              = (DTYPE)x_low;
            }
            else
            {
                x_high = x_low + 1;
            }

            DTYPE ly = y - y_low;
            DTYPE lx = x - x_low;
            DTYPE hy = 1.0 - ly;
            DTYPE hx = 1.0 - lx;

            DTYPE w1 = hy * hx;
            DTYPE w2 = hy * lx;
            DTYPE w3 = ly * hx;
            DTYPE w4 = ly * lx;

            DTYPE g1 = ograd * w1 / count;
            DTYPE g2 = ograd * w2 / count;
            DTYPE g3 = ograd * w3 / count;
            DTYPE g4 = ograd * w4 / count;

            if(x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0)
            {
                // atomic_add_g(&(TV_4D_AT(input_grad, n, c, y_low, x_low)), g1);
                // atomic_add_g(&(TV_4D_AT(input_grad, n, c, y_low, x_high)), g2);
                // atomic_add_g(&(TV_4D_AT(input_grad, n, c, y_high, x_low)), g3);
                // atomic_add_g(&(TV_4D_AT(input_grad, n, c, y_high, x_high)), g4);
                atomic_add_g(input_grad[input_grad_tv.get_tensor_view_idx({n, c, y_low, x_low})],
                             g1);
                atomic_add_g(input_grad[input_grad_tv.get_tensor_view_idx({n, c, y_low, x_high})],
                             g2);
                atomic_add_g(input_grad[input_grad_tv.get_tensor_view_idx({n, c, y_high, x_low})],
                             g3);
                atomic_add_g(input_grad[input_grad_tv.get_tensor_view_idx({n, c, y_high, x_high})],
                             g4);
            }
        }
    }
}

extern "C" __global__ void RoIAlignForward(const FLOAT* __restrict__ input,
                                           const FLOAT* __restrict__ rois,
                                           FLOAT* __restrict__ output,
                                           int output_h,
                                           int output_w,
                                           float spatial_scale,
                                           int sampling_ratio,
                                           bool aligned,
                                           int roi_batch_base_idx,
                                           tensor_view_t<4> input_tv,
                                           tensor_view_t<2> rois_tv,
                                           tensor_view_t<4> output_tv)
{
    // printf("Global kernel called\n");
    DeviceRoIAlignForward<FLOAT>(input,
                                 rois,
                                 output,
                                 output_h,
                                 output_w,
                                 spatial_scale,
                                 sampling_ratio,
                                 aligned,
                                 roi_batch_base_idx,
                                 input_tv,
                                 rois_tv,
                                 output_tv);
}

extern "C" __global__ void RoIAlignBackwardAtomic(const FLOAT* output_grad,
                                                  const FLOAT* rois,
                                                  FLOAT* input_grad,
                                                  const long N,
                                                  const long C,
                                                  const long H,
                                                  const long W,
                                                  const const long K,
                                                  const int32_t OH,
                                                  const int32_t OW,
                                                  FLOAT spatial_scale,
                                                  int32_t sampling_ratio,
                                                  char aligned,
                                                  int32_t roi_batch_base_idx,
                                                  tensor_view_t<4> output_grad_tv,
                                                  tensor_view_t<2> rois_tv,
                                                  tensor_view_t<4> input_grad_tv)
{
    roialign_backward_atomic<FLOAT>(output_grad,
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
                                    roi_batch_base_idx,
                                    output_grad_tv,
                                    rois_tv,
                                    input_grad_tv);
}
