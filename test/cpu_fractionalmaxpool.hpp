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

#include "tensor_holder.hpp"
#include <miopen/tensor_view_utils.hpp>

inline int64_t get_interval(
    double sample, int64_t index, int64_t input_size, int64_t output_size, int64_t pool_size)
{
    if(index == output_size - 1)
    {
        return input_size - pool_size;
    }
    else
    {
        double alpha =
            static_cast<double>(input_size - pool_size) / static_cast<double>(output_size - 1);
        return static_cast<int64_t>((index + sample) * alpha) -
               static_cast<int64_t>(sample * alpha);
    }
}

template <class T, class Ti>
void cpu_fractionalmaxpool2d_forward(const tensor<T> input,
                                     tensor<T>& output,
                                     tensor<Ti>& indices,
                                     const tensor<T> random_sample,
                                     const int64_t KH,
                                     const int64_t KW)
{
    auto input_tv         = miopen::get_inner_expanded_tv<4>(input.desc);
    auto output_tv        = miopen::get_inner_expanded_tv<4>(output.desc);
    auto indices_tv       = miopen::get_inner_expanded_tv<4>(indices.desc);
    auto random_sample_tv = miopen::get_inner_expanded_tv<3>(random_sample.desc);

    par_ford(output.desc.GetElementSize())([&](auto gid) {
        tensor_layout_t<4> layout(output_tv, gid);
        int64_t n, c, oh, ow;
        n  = layout.layout[0];
        c  = layout.layout[1];
        oh = layout.layout[2];
        ow = layout.layout[3];

        if(n >= output_tv.size[0])
            return;

        int64_t pool_h = get_interval(
            static_cast<double>(random_sample[random_sample_tv.get_tensor_view_idx({n, c, 0})]),
            oh,
            input_tv.size[2],
            output_tv.size[2],
            KH);
        int64_t pool_w = get_interval(
            static_cast<double>(random_sample[random_sample_tv.get_tensor_view_idx({n, c, 1})]),
            ow,
            input_tv.size[3],
            output_tv.size[3],
            KW);

        double m = log(0);

        int64_t h_end = pool_h + KH;
        int64_t w_end = pool_w + KW;

        if(indices.desc.GetElementSize() == 1)
        {
            for(int64_t h = pool_h; h < h_end; ++h)
            {
                for(int64_t w = pool_w; w < w_end; ++w)
                {
                    double val =
                        static_cast<double>(input[input_tv.get_tensor_view_idx({n, c, h, w})]);
                    if(val > m || std::isnan(val))
                    {
                        m = val;
                    }
                }
            }
            output[output_tv.get_tensor_view_idx(layout)] = static_cast<T>(m);
        }
        else
        {
            int64_t mi = pool_h * input_tv.size[3] + pool_w;
            for(int64_t h = pool_h; h < h_end; ++h)
            {
                for(int64_t w = pool_w; w < w_end; ++w)
                {
                    double val = (input[input_tv.get_tensor_view_idx({n, c, h, w})]);
                    if(val > m || std::isnan(val))
                    {
                        m  = val;
                        mi = h * input_tv.size[3] + w;
                    }
                }
            }
            output[output_tv.get_tensor_view_idx(layout)]   = static_cast<T>(m);
            indices[indices_tv.get_tensor_view_idx(layout)] = static_cast<Ti>(mi);
        }
    });
}

template <class T, class Ti>
void cpu_fractionalmaxpool2d_backward(const tensor<Ti> indices,
                                      const tensor<T> output_grad,
                                      tensor<T>& input_grad)
{
    auto indices_tv     = miopen::get_inner_expanded_tv<4>(indices.desc);
    auto output_grad_tv = miopen::get_inner_expanded_tv<4>(output_grad.desc);
    auto input_grad_tv  = miopen::get_inner_expanded_tv<4>(input_grad.desc);

    for(uint64_t gid = 0; gid < output_grad.desc.GetElementSize(); ++gid)
    {
        tensor_layout_t<4> layout(output_grad_tv, gid);
        uint64_t index = static_cast<uint64_t>(indices[indices_tv.get_tensor_view_idx(layout)]);
        uint64_t h     = index / input_grad_tv.size[3];
        uint64_t w     = index % input_grad_tv.size[3];

        double val = static_cast<double>(input_grad[input_grad_tv.get_tensor_view_idx(
                         {layout.layout[0], layout.layout[1], h, w})]) +
                     static_cast<double>(output_grad[output_grad_tv.get_tensor_view_idx(layout)]);
        input_grad[input_grad_tv.get_tensor_view_idx({layout.layout[0], layout.layout[1], h, w})] =
            static_cast<T>(val);
    }
}

template <class T, class Ti>
void cpu_fractionalmaxpool3d_forward(const tensor<T> input,
                                     tensor<T>& output,
                                     tensor<Ti>& indices,
                                     const tensor<T> random_sample,
                                     const int64_t KD,
                                     const int64_t KH,
                                     const int64_t KW)
{
    auto input_tv         = miopen::get_inner_expanded_tv<5>(input.desc);
    auto output_tv        = miopen::get_inner_expanded_tv<5>(output.desc);
    auto indices_tv       = miopen::get_inner_expanded_tv<5>(indices.desc);
    auto random_sample_tv = miopen::get_inner_expanded_tv<3>(random_sample.desc);

    par_ford(output.desc.GetElementSize())([&](auto gid) {
        tensor_layout_t<5> layout(output_tv, gid);
        int64_t n, c, od, oh, ow;
        n  = layout.layout[0];
        c  = layout.layout[1];
        od = layout.layout[2];
        oh = layout.layout[3];
        ow = layout.layout[4];

        if(n >= output_tv.size[0])
            return;

        int64_t pool_d = get_interval(
            static_cast<double>(random_sample[random_sample_tv.get_tensor_view_idx({n, c, 0})]),
            od,
            input_tv.size[2],
            output_tv.size[2],
            KD);
        int64_t pool_h = get_interval(
            static_cast<double>(random_sample[random_sample_tv.get_tensor_view_idx({n, c, 1})]),
            oh,
            input_tv.size[3],
            output_tv.size[3],
            KH);
        int64_t pool_w = get_interval(
            static_cast<double>(random_sample[random_sample_tv.get_tensor_view_idx({n, c, 2})]),
            ow,
            input_tv.size[4],
            output_tv.size[4],
            KW);

        double m = log(0);

        int64_t d_end = pool_d + KD;
        int64_t h_end = pool_h + KH;
        int64_t w_end = pool_w + KW;

        if(indices.desc.GetElementSize() == 1)
        {
            for(int64_t d = pool_d; d < d_end; ++d)
            {
                for(int64_t h = pool_h; h < h_end; ++h)
                {
                    for(int64_t w = pool_w; w < w_end; ++w)
                    {
                        double val = static_cast<double>(
                            input[input_tv.get_tensor_view_idx({n, c, d, h, w})]);
                        if(val > m || std::isnan(val))
                        {
                            m = val;
                        }
                    }
                }
            }
            output[output_tv.get_tensor_view_idx(layout)] = static_cast<T>(m);
        }
        else
        {
            int64_t mi =
                pool_d * input_tv.size[3] * input_tv.size[4] + pool_h * input_tv.size[4] + pool_w;
            for(int64_t d = pool_d; d < d_end; ++d)
            {
                for(int64_t h = pool_h; h < h_end; ++h)
                {
                    for(int64_t w = pool_w; w < w_end; ++w)
                    {
                        double val = (input[input_tv.get_tensor_view_idx({n, c, d, h, w})]);
                        if(val > m || std::isnan(val))
                        {
                            m  = val;
                            mi = d * input_tv.size[3] * input_tv.size[4] + h * input_tv.size[4] + w;
                        }
                    }
                }
            }
            output[output_tv.get_tensor_view_idx(layout)]   = static_cast<T>(m);
            indices[indices_tv.get_tensor_view_idx(layout)] = static_cast<Ti>(mi);
        }
    });
}

template <class T, class Ti>
void cpu_fractionalmaxpool3d_backward(const tensor<Ti> indices,
                                      const tensor<T> output_grad,
                                      tensor<T>& input_grad)
{
    auto indices_tv     = miopen::get_inner_expanded_tv<5>(indices.desc);
    auto output_grad_tv = miopen::get_inner_expanded_tv<5>(output_grad.desc);
    auto input_grad_tv  = miopen::get_inner_expanded_tv<5>(input_grad.desc);

    for(uint64_t gid = 0; gid < output_grad.desc.GetElementSize(); ++gid)
    {
        tensor_layout_t<5> layout(output_grad_tv, gid);
        uint64_t index = static_cast<uint64_t>(indices[indices_tv.get_tensor_view_idx(layout)]);
        uint64_t d     = index / (input_grad_tv.size[4] * input_grad_tv.size[3]);
        uint64_t h     = (index / input_grad_tv.size[4]) % input_grad_tv.size[3];
        uint64_t w     = index % input_grad_tv.size[4];

        double val = static_cast<double>(input_grad[input_grad_tv.get_tensor_view_idx(
                         {layout.layout[0], layout.layout[1], d, h, w})]) +
                     static_cast<double>(output_grad[output_grad_tv.get_tensor_view_idx(layout)]);
        input_grad[input_grad_tv.get_tensor_view_idx(
            {layout.layout[0], layout.layout[1], d, h, w})] = static_cast<T>(val);
    }
}
