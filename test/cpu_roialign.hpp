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

#pragma once

#include <miopen/tensor_view_utils.hpp>

#include "tensor_holder.hpp"

template <class T>
void cpu_roialign_forward(const tensor<T> input,
                          const tensor<T> rois,
                          tensor<T>& output,
                          const uint64_t output_h,
                          const uint64_t output_w,
                          const float spatial_scale,
                          const int64_t sampling_ratio,
                          const bool aligned)
{
    auto input_tv  = miopen::get_inner_expanded_tv<4>(input.desc);
    auto rois_tv   = miopen::get_inner_expanded_tv<2>(rois.desc);
    auto output_tv = miopen::get_inner_expanded_tv<4>(output.desc);

    const auto input_lengths = input.desc.GetLengths();
    const auto C             = input_lengths[1];
    const auto H             = input_lengths[2];
    const auto W             = input_lengths[3];

    const auto K = rois.desc.GetLengths()[0];

    const float roi_offset = aligned ? 0.5f : 0.0f;

    for(auto k = 0; k < K; ++k)
    {
        int64_t roi_batch_idx = static_cast<int64_t>(rois[rois_tv.get_tensor_view_idx({k, 0})]);

        const float roi_w1 =
            static_cast<float>(rois[rois_tv.get_tensor_view_idx({k, 1})]) * spatial_scale -
            roi_offset;
        const float roi_h1 =
            static_cast<float>(rois[rois_tv.get_tensor_view_idx({k, 2})]) * spatial_scale -
            roi_offset;
        const float roi_w2 =
            static_cast<float>(rois[rois_tv.get_tensor_view_idx({k, 3})]) * spatial_scale -
            roi_offset;
        const float roi_h2 =
            static_cast<float>(rois[rois_tv.get_tensor_view_idx({k, 4})]) * spatial_scale -
            roi_offset;

        float roi_w = roi_w2 - roi_w1;
        float roi_h = roi_h2 - roi_h1;

        if(!aligned)
        {
            roi_w = std::fmax(roi_w, 1);
            roi_h = std::fmax(roi_h, 1);
        }
        const float bin_size_h = roi_h / static_cast<float>(output_h);
        const float bin_size_w = roi_w / static_cast<float>(output_w);

        const int64_t bin_grid_h = (sampling_ratio > 0)
                                       ? sampling_ratio
                                       : static_cast<int64_t>(std::ceil(roi_h / output_h));
        const int64_t bin_grid_w = (sampling_ratio > 0)
                                       ? sampling_ratio
                                       : static_cast<int64_t>(std::ceil(roi_w / output_w));

        const float scale = 1.0f / static_cast<float>(bin_grid_h * bin_grid_w);

        for(auto c = 0; c < C; ++c)
        {
            for(auto ph = 0; ph < output_h; ++ph)
            {
                for(auto pw = 0; pw < output_w; ++pw)
                {
                    float sum = 0.0f;
                    for(auto iy = 0; iy < bin_grid_h; ++iy)
                    {
                        const float yy = roi_h1 + static_cast<float>(ph) * bin_size_h +
                                         (static_cast<float>(iy) + 0.5f) *
                                             (bin_size_h / static_cast<float>(bin_grid_h));
                        if(yy < -1.0 || yy > static_cast<float>(H))
                        {
                            continue;
                        }
                        for(int ix = 0; ix < bin_grid_w; ++ix)
                        {
                            const float xx = roi_w1 + pw * bin_size_w +
                                             (static_cast<float>(ix) + 0.5f) *
                                                 (bin_size_w / static_cast<float>(bin_grid_w));
                            if(xx < -1.f || xx > static_cast<float>(W))
                            {
                                continue;
                            }

                            const float y  = std::min(std::max(yy, 0.f), static_cast<float>(H - 1));
                            const float x  = std::min(std::max(xx, 0.f), static_cast<float>(W - 1));
                            const int yl   = static_cast<int>(std::floor(y));
                            const int xl   = static_cast<int>(std::floor(x));
                            const int yh   = std::min(yl + 1, static_cast<int>(H - 1));
                            const int xh   = std::min(xl + 1, static_cast<int>(W - 1));
                            const float py = y - static_cast<float>(yl);
                            const float px = x - static_cast<float>(xl);
                            const float qy = 1.f - py;
                            const float qx = 1.f - px;

                            float w1 = qy * qx;
                            float w2 = qy * px;
                            float w3 = py * qx;
                            float w4 = py * px;

                            sum += w1 * static_cast<float>(input[input_tv.get_tensor_view_idx(
                                            {roi_batch_idx, c, yl, xl})]) +
                                   w2 * static_cast<float>(input[input_tv.get_tensor_view_idx(
                                            {roi_batch_idx, c, yl, xh})]) +
                                   w3 * static_cast<float>(input[input_tv.get_tensor_view_idx(
                                            {roi_batch_idx, c, yh, xl})]) +
                                   w4 * static_cast<float>(input[input_tv.get_tensor_view_idx(
                                            {roi_batch_idx, c, yh, xh})]);
                        }
                    }
                    float output_val = sum * scale;
                    output[output_tv.get_tensor_view_idx({k, c, ph, pw})] =
                        static_cast<T>(output_val);
                }
            }
        }
    }
}

template <class T>
void cpu_roialign_backward(const tensor<T> output_grad,
                           const tensor<T> rois,
                           tensor<T>& input_grad,
                           const int64_t OH,
                           const int64_t OW,
                           const float spatial_scale,
                           const int64_t sampling_ratio,
                           const bool aligned)
{
    std::fill(input_grad.data.begin(), input_grad.data.end(), static_cast<T>(0));

    // Calculate input_grad on float and then convert to T
    // to preserve precision
    tensor<float> float_input_grad(input_grad.desc.GetLengths(), input_grad.desc.GetStrides());

    auto input_grad_tv  = miopen::get_inner_expanded_tv<4>(input_grad.desc);
    auto rois_tv        = miopen::get_inner_expanded_tv<2>(rois.desc);
    auto output_grad_tv = miopen::get_inner_expanded_tv<4>(output_grad.desc);

    const auto input_grad_lengths = input_grad.desc.GetLengths();
    const auto N                  = input_grad_lengths[0];
    const auto C                  = input_grad_lengths[1];
    const auto H                  = input_grad_lengths[2];
    const auto W                  = input_grad_lengths[3];

    const auto output_grad_numel = output_grad.desc.GetElementSize();

    for(auto i = 0; i < output_grad_numel; i++)
    {
        uint64_t ow = i % OW;
        uint64_t oh = (i / OW) % OH;
        uint64_t c  = (i / (OW * OH)) % C;
        uint64_t k  = (i / (C * OW * OH));

        // Check k-th roi box belongs to n-th image inside mini-batch
        int64_t n = rois[rois_tv.get_tensor_view_idx({k, 0})];

        if(n < 0 || n >= N)
            return;

        // roi box
        float offset = aligned ? 0.5f : 0;

        float x1 =
            static_cast<float>(rois[rois_tv.get_tensor_view_idx({k, 1})]) * spatial_scale - offset;
        float y1 =
            static_cast<float>(rois[rois_tv.get_tensor_view_idx({k, 2})]) * spatial_scale - offset;
        float x2 =
            static_cast<float>(rois[rois_tv.get_tensor_view_idx({k, 3})]) * spatial_scale - offset;
        float y2 =
            static_cast<float>(rois[rois_tv.get_tensor_view_idx({k, 4})]) * spatial_scale - offset;

        float roi_h = y2 - y1;
        float roi_w = x2 - x1;
        if(!aligned)
        {
            // Force ROI to be at least 1x1
            roi_h = std::fmax(roi_h, 1);
            roi_w = std::fmax(roi_w, 1);
        }

        // bin is OH * OW cells inside ROI
        float bin_h = roi_h / OH;
        float bin_w = roi_w / OW;

        // grid is sampling_ratio_h * sampling_ratio_w cells inside bin
        // Each center of grid is sampled and avgpooled into bin
        uint64_t sampling_ratio_h = sampling_ratio > 0 ? sampling_ratio : std::ceil(roi_h / OH);
        uint64_t sampling_ratio_w = sampling_ratio > 0 ? sampling_ratio : std::ceil(roi_w / OW);

        const uint64_t count = sampling_ratio_h * sampling_ratio_w;

        int64_t x_low, x_high, y_low, y_high;

        float ograd =
            static_cast<float>(output_grad[output_grad_tv.get_tensor_view_idx({k, c, oh, ow})]);

        for(auto r = 0; r < sampling_ratio_h; r++)
        {
            float y = y1 + bin_h * oh + bin_h / sampling_ratio_h * (r + 0.5f);
            if(y < 0 || y > H)
                continue;
            y_low = static_cast<int64_t>(y);
            if(y_low >= H - 1)
            {
                y_high = y_low = H - 1;
                y              = static_cast<float>(y_low);
            }
            else
            {
                y_high = y_low + 1;
            }
            for(auto s = 0; s < sampling_ratio_w; ++s)
            {
                float x = x1 + bin_w * ow + bin_w / sampling_ratio_w * (s + 0.5f);
                if(x < 0 || x > W)
                    continue;

                x_low = static_cast<int64_t>(x);
                if(x_low >= W - 1)
                {
                    x_high = x_low = W - 1;
                    x              = static_cast<float>(x_low);
                }
                else
                {
                    x_high = x_low + 1;
                }

                float ly = y - y_low;
                float lx = x - x_low;
                float hy = 1.0 - ly;
                float hx = 1.0 - lx;

                float w1 = hy * hx;
                float w2 = hy * lx;
                float w3 = ly * hx;
                float w4 = ly * lx;

                float g1 = ograd * w1 / count;
                float g2 = ograd * w2 / count;
                float g3 = ograd * w3 / count;
                float g4 = ograd * w4 / count;

                if(x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0)
                {
                    float_input_grad[input_grad_tv.get_tensor_view_idx({n, c, y_low, x_low})] += g1;
                    float_input_grad[input_grad_tv.get_tensor_view_idx({n, c, y_low, x_high})] +=
                        g2;
                    float_input_grad[input_grad_tv.get_tensor_view_idx({n, c, y_high, x_low})] +=
                        g3;
                    float_input_grad[input_grad_tv.get_tensor_view_idx({n, c, y_high, x_high})] +=
                        g4;
                }
            }
        }
    }

    // Assign float_input_grad to input_grad
    for(auto n = 0; n < N; n++)
    {
        for(auto c = 0; c < C; c++)
        {
            for(auto h = 0; h < H; h++)
            {
                for(auto w = 0; w < W; w++)
                {
                    input_grad[input_grad_tv.get_tensor_view_idx({n, c, h, w})] = static_cast<T>(
                        float_input_grad[input_grad_tv.get_tensor_view_idx({n, c, h, w})]);
                }
            }
        }
    }
}
