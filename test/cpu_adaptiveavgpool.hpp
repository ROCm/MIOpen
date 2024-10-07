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
#include "ford.hpp"

template <class T>
void cpu_adaptiveavgpool_forward_1d(
    tensor<T> input, tensor<T>& output, size_t N, size_t C, size_t H, size_t OH)
{
    auto dims  = input.desc.GetLengths();
    auto numel = output.desc.GetElementSize();

    auto input_tv  = miopen::get_inner_expanded_tv<3>(input.desc);
    auto output_tv = miopen::get_inner_expanded_tv<3>(output.desc);

    par_ford(numel)([&](size_t gid) {
        size_t nc = gid / OH, oh = gid % OH;
        size_t n = nc / C, c = nc % C;

        size_t h  = static_cast<size_t>(std::floor(static_cast<float>(oh * H) / OH));
        size_t kh = static_cast<size_t>(std::ceil(static_cast<float>((oh + 1) * H) / OH)) - h;

        float sum = 0;
        for(size_t ih = h; ih < (h + kh); ++ih)
        {
            sum += static_cast<float>(input[input_tv.get_tensor_view_idx({n, c, ih})]);
        }

        output[output_tv.get_tensor_view_idx({n, c, oh})] = static_cast<T>(sum / kh);
    });
}

template <class T>
void cpu_adaptiveavgpool_forward_2d(tensor<T> input,
                                    tensor<T>& output,
                                    size_t N,
                                    size_t C,
                                    size_t H,
                                    size_t W,
                                    size_t OH,
                                    size_t OW)
{
    auto dims  = input.desc.GetLengths();
    auto numel = output.desc.GetElementSize();

    auto input_tv  = miopen::get_inner_expanded_tv<4>(input.desc);
    auto output_tv = miopen::get_inner_expanded_tv<4>(output.desc);

    par_ford(numel)([&](size_t gid) {
        size_t ncoh = gid / OW, ow = gid % OW;
        size_t nc = ncoh / OH, oh = ncoh % OH;
        size_t n = nc / C, c = nc % C;

        size_t h  = static_cast<size_t>(std::floor(static_cast<float>(oh * H) / OH));
        size_t kh = static_cast<size_t>(std::ceil(static_cast<float>((oh + 1) * H) / OH)) - h;

        size_t w  = static_cast<size_t>(std::floor(static_cast<float>(ow * W) / OW));
        size_t kw = static_cast<size_t>(std::ceil(static_cast<float>((ow + 1) * W) / OW)) - w;

        float divider = static_cast<float>(kh * kw);
        float sum     = 0;
        for(size_t ih = h; ih < (h + kh); ++ih)
        {
            for(size_t iw = w; iw < (w + kw); ++iw)
            {
                sum += static_cast<float>(input[input_tv.get_tensor_view_idx({n, c, ih, iw})]);
            }
        }

        output[output_tv.get_tensor_view_idx({n, c, oh, ow})] = static_cast<T>(sum / divider);
    });
}

template <class T>
void cpu_adaptiveavgpool_forward_3d(tensor<T> input,
                                    tensor<T>& output,
                                    size_t N,
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

    auto input_tv  = miopen::get_inner_expanded_tv<5>(input.desc);
    auto output_tv = miopen::get_inner_expanded_tv<5>(output.desc);

    par_ford(numel)([&](size_t gid) {
        size_t ncodoh = gid / OW, ow = gid % OW;
        size_t ncod = ncodoh / OH, oh = ncodoh % OH;
        size_t nc = ncod / OD, od = ncod % OD;
        size_t n = nc / C, c = nc % C;

        size_t d  = static_cast<size_t>(std::floor(static_cast<float>(od * D) / OD));
        size_t kd = static_cast<size_t>(std::ceil(static_cast<float>((od + 1) * D) / OD)) - d;

        size_t h  = static_cast<size_t>(std::floor(static_cast<float>(oh * H) / OH));
        size_t kh = static_cast<size_t>(std::ceil(static_cast<float>((oh + 1) * H) / OH)) - h;

        size_t w  = static_cast<size_t>(std::floor(static_cast<float>(ow * W) / OW));
        size_t kw = static_cast<size_t>(std::ceil(static_cast<float>((ow + 1) * W) / OW)) - w;

        float sum = 0;
        for(size_t id = d; id < (d + kd); ++id)
        {
            for(size_t ih = h; ih < (h + kh); ++ih)
            {
                for(size_t iw = w; iw < (w + kw); ++iw)
                {
                    sum +=
                        static_cast<float>(input[input_tv.get_tensor_view_idx({n, c, id, ih, iw})]);
                }
            }
        }

        output[output_tv.get_tensor_view_idx({n, c, od, oh, ow})] =
            static_cast<T>(sum / (kd * kh * kw));
    });
}

template <class T>
void cpu_adaptiveavgpool_backward_1d(
    tensor<T> output_grad, tensor<T>& input_grad, size_t N, size_t C, size_t H, size_t OH)
{
    auto dims  = input_grad.desc.GetLengths();
    auto numel = input_grad.desc.GetElementSize();

    auto output_grad_tv = miopen::get_inner_expanded_tv<3>(output_grad.desc);
    auto input_grad_tv  = miopen::get_inner_expanded_tv<3>(input_grad.desc);

    par_ford(numel)([&](size_t gid) {
        size_t nc = gid / H, h = gid % H;
        size_t n = nc / C, c = nc % C;

        size_t oh  = static_cast<size_t>(std::floor(static_cast<float>(h * OH) / H));
        size_t koh = static_cast<size_t>(std::ceil(static_cast<float>((h + 1) * OH) / H)) - oh;

        float grad = 0;
        for(size_t ih = oh; ih < (oh + koh); ++ih)
        {
            size_t kh = static_cast<size_t>(std::ceil(static_cast<float>((ih + 1) * H) / OH)) -
                        static_cast<size_t>(std::floor(static_cast<float>(ih * H) / OH));
            grad +=
                static_cast<float>(output_grad[output_grad_tv.get_tensor_view_idx({n, c, ih})]) /
                kh;
        }

        input_grad[input_grad_tv.get_tensor_view_idx({n, c, h})] = static_cast<T>(grad);
    });
}

template <class T>
void cpu_adaptiveavgpool_backward_2d(tensor<T> output_grad,
                                     tensor<T>& input_grad,
                                     size_t N,
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

    par_ford(numel)([&](size_t gid) {
        size_t nch = gid / W, w = gid % W;
        size_t nc = nch / H, h = nch % H;
        size_t n = nc / C, c = nc % C;

        size_t oh  = static_cast<size_t>(std::floor(static_cast<float>(h * OH) / H));
        size_t koh = static_cast<size_t>(std::ceil(static_cast<float>((h + 1) * OH) / H)) - oh;

        size_t ow  = static_cast<size_t>(std::floor(static_cast<float>(w * OW) / W));
        size_t kow = static_cast<size_t>(std::ceil(static_cast<float>((w + 1) * OW) / W)) - ow;

        float grad = 0;
        for(size_t ih = oh; ih < (oh + koh); ++ih)
        {
            size_t kh = static_cast<size_t>(std::ceil(static_cast<float>((ih + 1) * H) / OH)) -
                        static_cast<size_t>(std::floor(static_cast<float>(ih * H) / OH));
            for(size_t iw = ow; iw < (ow + kow); ++iw)
            {
                size_t kw = static_cast<size_t>(std::ceil(static_cast<float>((iw + 1) * W) / OW)) -
                            static_cast<size_t>(std::floor(static_cast<float>(iw * W) / OW));
                grad += static_cast<float>(
                            output_grad[output_grad_tv.get_tensor_view_idx({n, c, ih, iw})]) /
                        (kh * kw);
            }
        }

        input_grad[input_grad_tv.get_tensor_view_idx({n, c, h, w})] = static_cast<T>(grad);
    });
}

template <class T>
void cpu_adaptiveavgpool_backward_3d(tensor<T> output_grad,
                                     tensor<T>& input_grad,
                                     size_t N,
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

    par_ford(numel)([&](size_t gid) {
        size_t ncdh = gid / W, w = gid % W;
        size_t ncd = ncdh / H, h = ncdh % H;
        size_t nc = ncd / D, d = ncd % D;
        size_t n = nc / C, c = nc % C;

        size_t od  = static_cast<size_t>(std::floor(static_cast<float>(d * OD) / D));
        size_t kod = static_cast<size_t>(std::ceil(static_cast<float>((d + 1) * OD) / D)) - od;

        size_t oh  = static_cast<size_t>(std::floor(static_cast<float>(h * OH) / H));
        size_t koh = static_cast<size_t>(std::ceil(static_cast<float>((h + 1) * OH) / H)) - oh;

        size_t ow  = static_cast<size_t>(std::floor(static_cast<float>(w * OW) / W));
        size_t kow = static_cast<size_t>(std::ceil(static_cast<float>((w + 1) * OW) / W)) - ow;

        float grad = 0;
        for(size_t id = od; id < (od + kod); ++id)
        {
            size_t kd = static_cast<size_t>(std::ceil(static_cast<float>((id + 1) * D) / OD)) -
                        static_cast<size_t>(std::floor(static_cast<float>(id * D) / OD));
            for(size_t ih = oh; ih < (oh + koh); ++ih)
            {
                size_t kh = static_cast<size_t>(std::ceil(static_cast<float>((ih + 1) * H) / OH)) -
                            static_cast<size_t>(std::floor(static_cast<float>(ih * H) / OH));
                for(size_t iw = ow; iw < (ow + kow); ++iw)
                {
                    size_t kw =
                        static_cast<size_t>(std::ceil(static_cast<float>((iw + 1) * W) / OW)) -
                        static_cast<size_t>(std::floor(static_cast<float>(iw * W) / OW));
                    grad +=
                        static_cast<float>(
                            output_grad[output_grad_tv.get_tensor_view_idx({n, c, id, ih, iw})]) /
                        (kd * kh * kw);
                }
            }
        }

        input_grad[input_grad_tv.get_tensor_view_idx({n, c, d, h, w})] = static_cast<T>(grad);
    });
}
