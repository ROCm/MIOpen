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
#pragma once

#include "tensor_holder.hpp"
#include <miopen/tensor_view_utils.hpp>
#include "ford.hpp"

template <class T>
void cpu_adaptivemaxpool_forward_1d(
    tensor<T> input, tensor<T>& output, tensor<int64_t>& indices, size_t C, size_t H, size_t OH)
{
    auto dims  = input.desc.GetLengths();
    auto numel = output.desc.GetElementSize();

    auto input_tv   = miopen::get_inner_expanded_tv<3>(input.desc);
    auto output_tv  = miopen::get_inner_expanded_tv<3>(output.desc);
    auto indices_tv = miopen::get_inner_expanded_tv<3>(indices.desc);

    par_ford(numel)([&](size_t gid) {
        size_t nc = gid / OH, oh = gid % OH;
        size_t n = nc / C, c = nc % C;

        size_t h  = oh * H / OH;
        size_t kh = ((oh + 1) * H + OH - 1) / OH;

        float m = -std::numeric_limits<float>::max();
        if(indices.desc.GetElementSize() == 1)
        {
            for(size_t ih = h; ih < kh; ++ih)
            {
                m = std::max(m,
                             static_cast<float>(input[input_tv.get_tensor_view_idx({n, c, ih})]));
            }
            output[output_tv.get_tensor_view_idx({n, c, oh})] = static_cast<T>(m);
        }
        else
        {
            size_t mi = 0;
            for(size_t ih = h; ih < kh; ++ih)
            {
                if(static_cast<float>(input[input_tv.get_tensor_view_idx({n, c, ih})]) > m)
                {
                    m  = static_cast<float>(input[input_tv.get_tensor_view_idx({n, c, ih})]);
                    mi = ih;
                }
            }
            output[output_tv.get_tensor_view_idx({n, c, oh})]   = static_cast<T>(m);
            indices[indices_tv.get_tensor_view_idx({n, c, oh})] = mi;
        }
    });
}

template <class T>
void cpu_adaptivemaxpool_forward_2d(tensor<T> input,
                                    tensor<T>& output,
                                    tensor<int64_t>& indices,
                                    size_t C,
                                    size_t H,
                                    size_t W,
                                    size_t OH,
                                    size_t OW)
{
    auto dims  = input.desc.GetLengths();
    auto numel = output.desc.GetElementSize();

    auto input_tv   = miopen::get_inner_expanded_tv<4>(input.desc);
    auto output_tv  = miopen::get_inner_expanded_tv<4>(output.desc);
    auto indices_tv = miopen::get_inner_expanded_tv<4>(indices.desc);

    par_ford(numel)([&](size_t gid) {
        size_t ncoh = gid / OW, ow = gid % OW;
        size_t nc = ncoh / OH, oh = ncoh % OH;
        size_t n = nc / C, c = nc % C;

        size_t h  = (oh * H) / OH;
        size_t kh = ((oh + 1) * H + OH - 1) / OH;

        size_t w  = (ow * W) / OW;
        size_t kw = ((ow + 1) * W + OW - 1) / OW;

        float m = -std::numeric_limits<float>::max();
        if(indices.desc.GetElementSize() == 1)
        {
            for(size_t ih = h; ih < kh; ++ih)
            {
                for(size_t iw = w; iw < kw; ++iw)
                {
                    m = std::max(
                        m, static_cast<float>(input[input_tv.get_tensor_view_idx({n, c, ih, iw})]));
                }
            }
            output[output_tv.get_tensor_view_idx({n, c, oh, ow})] = static_cast<T>(m);
        }
        else
        {
            size_t mi = 0;
            for(size_t ih = h; ih < kh; ++ih)
            {
                for(size_t iw = w; iw < kw; ++iw)
                {
                    float input_val =
                        static_cast<float>(input[input_tv.get_tensor_view_idx({n, c, ih, iw})]);
                    if(m < input_val)
                    {
                        m  = input_val;
                        mi = ih * W + iw;
                    }
                }
            }
            output[output_tv.get_tensor_view_idx({n, c, oh, ow})]   = static_cast<T>(m);
            indices[indices_tv.get_tensor_view_idx({n, c, oh, ow})] = mi;
        }
    });
}

template <class T>
void cpu_adaptivemaxpool_forward_3d(tensor<T> input,
                                    tensor<T>& output,
                                    tensor<int64_t>& indices,
                                    size_t C,
                                    size_t D,
                                    size_t H,
                                    size_t W,
                                    size_t OD,
                                    size_t OH,
                                    size_t OW)
{
    auto dims  = input.desc.GetLengths();
    auto numel = output.desc.GetElementSize();

    auto input_tv   = miopen::get_inner_expanded_tv<5>(input.desc);
    auto output_tv  = miopen::get_inner_expanded_tv<5>(output.desc);
    auto indices_tv = miopen::get_inner_expanded_tv<5>(indices.desc);

    par_ford(numel)([&](size_t gid) {
        size_t ncodoh = gid / OW, ow = gid % OW;
        size_t ncod = ncodoh / OH, oh = ncodoh % OH;
        size_t nc = ncod / OD, od = ncod % OD;
        size_t n = nc / C, c = nc % C;

        size_t d  = (od * D) / OD;
        size_t kd = ((od + 1) * D + OD - 1) / OD;

        size_t h  = (oh * H) / OH;
        size_t kh = ((oh + 1) * H + OH - 1) / OH;

        size_t w  = (ow * W) / OW;
        size_t kw = ((ow + 1) * W + OW - 1) / OW;

        float m = -std::numeric_limits<float>::max();
        if(indices.desc.GetElementSize() == 1)
        {
            for(size_t id = d; id < kd; ++id)
            {
                for(size_t ih = h; ih < kh; ++ih)
                {
                    for(size_t iw = w; iw < kw; ++iw)
                    {
                        m = std::max(m,
                                     static_cast<float>(
                                         input[input_tv.get_tensor_view_idx({n, c, id, ih, iw})]));
                    }
                }
            }
            output[output_tv.get_tensor_view_idx({n, c, od, oh, ow})] = static_cast<T>(m);
        }
        else
        {
            size_t mi = 0;
            for(size_t id = d; id < kd; ++id)
            {
                for(size_t ih = h; ih < kh; ++ih)
                {
                    for(size_t iw = w; iw < kw; ++iw)
                    {
                        float input_val = static_cast<float>(
                            input[input_tv.get_tensor_view_idx({n, c, id, ih, iw})]);
                        if(m < input_val)
                        {
                            m  = input_val;
                            mi = (id * H + ih) * W + iw;
                        }
                    }
                }
            }
            output[output_tv.get_tensor_view_idx({n, c, od, oh, ow})]   = static_cast<T>(m);
            indices[indices_tv.get_tensor_view_idx({n, c, od, oh, ow})] = mi;
        }
    });
}

template <class T>
void cpu_adaptivemaxpool_backward_1d(tensor<int64_t> indices,
                                     tensor<T> output_grad,
                                     tensor<T>& input_grad,
                                     size_t C,
                                     size_t H,
                                     size_t OH)
{
    auto dims  = input_grad.desc.GetLengths();
    auto numel = input_grad.desc.GetElementSize();

    auto output_grad_tv = miopen::get_inner_expanded_tv<3>(output_grad.desc);
    auto input_grad_tv  = miopen::get_inner_expanded_tv<3>(input_grad.desc);
    auto indices_tv     = miopen::get_inner_expanded_tv<3>(indices.desc);

    par_ford(numel)([&](size_t gid) {
        size_t nc = gid / H, h = gid % H;
        size_t n = nc / C, c = nc % C;

        size_t oh  = (h * OH) / H;
        size_t koh = (((h + 1) * OH + H - 1) / H) - oh;

        float grad = 0;
        for(size_t ih = oh; ih < (oh + koh); ++ih)
        {
            size_t idx = indices[indices_tv.get_tensor_view_idx({n, c, ih})];
            if(idx == h)
            {
                grad +=
                    static_cast<float>(output_grad[output_grad_tv.get_tensor_view_idx({n, c, ih})]);
            }
        }
        input_grad[input_grad_tv.get_tensor_view_idx({n, c, h})] = static_cast<T>(grad);
    });
}

template <class T>
void cpu_adaptivemaxpool_backward_2d(tensor<int64_t> indices,
                                     tensor<T> output_grad,
                                     tensor<T>& input_grad,
                                     size_t C,
                                     size_t H,
                                     size_t W,
                                     size_t OH,
                                     size_t OW)
{
    auto dims  = input_grad.desc.GetLengths();
    auto numel = input_grad.desc.GetElementSize();

    auto output_grad_tv = miopen::get_inner_expanded_tv<4>(output_grad.desc);
    auto input_grad_tv  = miopen::get_inner_expanded_tv<4>(input_grad.desc);
    auto indices_tv     = miopen::get_inner_expanded_tv<4>(indices.desc);

    par_ford(numel)([&](size_t gid) {
        size_t nch = gid / W, w = gid % W;
        size_t nc = nch / H, h = nch % H;
        size_t n = nc / C, c = nc % C;

        size_t oh  = (h * OH) / H;
        size_t koh = ((h + 1) * OH + H - 1) / H - oh;

        size_t ow  = (w * OW) / W;
        size_t kow = ((w + 1) * OW + W - 1) / W - ow;

        float grad = 0;
        for(size_t ih = oh; ih < (oh + koh); ++ih)
        {
            for(size_t iw = ow; iw < (ow + kow); ++iw)
            {
                if(indices[indices_tv.get_tensor_view_idx({n, c, ih, iw})] == h * W + w)
                {
                    grad += static_cast<float>(
                        output_grad[output_grad_tv.get_tensor_view_idx({n, c, ih, iw})]);
                }
            }
        }

        input_grad[input_grad_tv.get_tensor_view_idx({n, c, h, w})] = static_cast<T>(grad);
    });
}

template <class T>
void cpu_adaptivemaxpool_backward_3d(tensor<int64_t> indices,
                                     tensor<T> output_grad,
                                     tensor<T>& input_grad,
                                     size_t C,
                                     size_t D,
                                     size_t H,
                                     size_t W,
                                     size_t OD,
                                     size_t OH,
                                     size_t OW)
{
    auto dims  = input_grad.desc.GetLengths();
    auto numel = input_grad.desc.GetElementSize();

    auto output_grad_tv = miopen::get_inner_expanded_tv<5>(output_grad.desc);
    auto input_grad_tv  = miopen::get_inner_expanded_tv<5>(input_grad.desc);
    auto indices_tv     = miopen::get_inner_expanded_tv<5>(indices.desc);

    par_ford(numel)([&](size_t gid) {
        size_t ncdh = gid / W, w = gid % W;
        size_t ncd = ncdh / H, h = ncdh % H;
        size_t nc = ncd / D, d = ncd % D;
        size_t n = nc / C, c = nc % C;

        size_t od  = (d * OD) / D;
        size_t kod = ((d + 1) * OD + D - 1) / D - od;

        size_t oh  = (h * OH) / H;
        size_t koh = ((h + 1) * OH + H - 1) / H - oh;

        size_t ow  = (w * OW) / W;
        size_t kow = ((w + 1) * OW + W - 1) / W - ow;

        float grad = 0;
        for(size_t id = od; id < (od + kod); ++id)
        {
            for(size_t ih = oh; ih < (oh + koh); ++ih)
            {
                for(size_t iw = ow; iw < (ow + kow); ++iw)
                {
                    if(indices[indices_tv.get_tensor_view_idx({n, c, id, ih, iw})] ==
                       (d * H + h) * W + w)
                    {
                        grad += static_cast<float>(
                            output_grad[output_grad_tv.get_tensor_view_idx({n, c, id, ih, iw})]);
                    }
                }
            }
        }

        input_grad[input_grad_tv.get_tensor_view_idx({n, c, d, h, w})] = static_cast<T>(grad);
    });
}
