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

#include <cmath>
#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <../test/ford.hpp>

template <typename Tgpu, typename Tcheck>
int32_t mloAdaptiveAvgPoolForward1dRunHost(const miopenTensorDescriptor_t inputDesc,
                                           const miopenTensorDescriptor_t outputDesc,
                                           const Tgpu* input,
                                           Tcheck* output,
                                           const size_t C,
                                           const size_t H,
                                           const size_t OH)
{
    auto numel = miopen::deref(outputDesc).GetElementSize();

    auto input_tv  = miopen::get_inner_expanded_tv<3>(miopen::deref(inputDesc));
    auto output_tv = miopen::get_inner_expanded_tv<3>(miopen::deref(outputDesc));

    par_ford(numel)([&](size_t gid) {
        size_t nc = gid / OH, oh = gid % OH;
        size_t n = nc / C, c = nc % C;

        size_t h  = oh * H / OH;
        size_t kh = (((oh + 1) * H + OH - 1) / OH) - h;

        float sum = 0;
        for(size_t ih = h; ih < (h + kh); ++ih)
        {
            sum += static_cast<float>(input[input_tv.get_tensor_view_idx({n, c, ih})]);
        }

        output[output_tv.get_tensor_view_idx({n, c, oh})] = static_cast<Tcheck>(sum / kh);
    });
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int32_t mloAdaptiveAvgPoolForward2dRunHost(const miopenTensorDescriptor_t inputDesc,
                                           const miopenTensorDescriptor_t outputDesc,
                                           const Tgpu* input,
                                           Tcheck* output,
                                           const size_t C,
                                           const size_t H,
                                           const size_t W,
                                           const size_t OH,
                                           const size_t OW)
{
    auto numel = miopen::deref(outputDesc).GetElementSize();

    auto input_tv  = miopen::get_inner_expanded_tv<4>(miopen::deref(inputDesc));
    auto output_tv = miopen::get_inner_expanded_tv<4>(miopen::deref(outputDesc));

    par_ford(numel)([&](size_t gid) {
        size_t ncoh = gid / OW, ow = gid % OW;
        size_t nc = ncoh / OH, oh = ncoh % OH;
        size_t n = nc / C, c = nc % C;

        size_t h  = (oh * H) / OH;
        size_t kh = (((oh + 1) * H + OH - 1) / OH) - h;

        size_t w  = (ow * W) / OW;
        size_t kw = (((ow + 1) * W + OW - 1) / OW) - w;

        float divider = static_cast<float>(kh * kw);
        float sum     = 0;
        for(size_t ih = h; ih < (h + kh); ++ih)
        {
            for(size_t iw = w; iw < (w + kw); ++iw)
            {
                sum += static_cast<float>(input[input_tv.get_tensor_view_idx({n, c, ih, iw})]);
            }
        }

        output[output_tv.get_tensor_view_idx({n, c, oh, ow})] = static_cast<Tcheck>(sum / divider);
    });
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int32_t mloAdaptiveAvgPoolForward3dRunHost(const miopenTensorDescriptor_t inputDesc,
                                           const miopenTensorDescriptor_t outputDesc,
                                           const Tgpu* input,
                                           Tcheck* output,
                                           const size_t C,
                                           const size_t D,
                                           const size_t H,
                                           const size_t W,
                                           const size_t OD,
                                           const size_t OH,
                                           const size_t OW)
{
    auto numel = miopen::deref(outputDesc).GetElementSize();

    auto input_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto output_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(outputDesc));

    par_ford(numel)([&](size_t gid) {
        size_t ncodoh = gid / OW, ow = gid % OW;
        size_t ncod = ncodoh / OH, oh = ncodoh % OH;
        size_t nc = ncod / OD, od = ncod % OD;
        size_t n = nc / C, c = nc % C;

        size_t d  = (od * D) / OD;
        size_t kd = ((od + 1) * D + OD - 1) / OD - d;

        size_t h  = (oh * H) / OH;
        size_t kh = ((oh + 1) * H + OH - 1) / OH - h;

        size_t w  = (ow * W) / OW;
        size_t kw = ((ow + 1) * W + OW - 1) / OW - w;

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
            static_cast<Tcheck>(sum / (kd * kh * kw));
    });
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int32_t mloAdaptiveAvgPoolBackward1dRunHost(const miopenTensorDescriptor_t outputGradDesc,
                                            const miopenTensorDescriptor_t inputGradDesc,
                                            const Tgpu* output_grad,
                                            Tcheck* input_grad,
                                            const size_t C,
                                            const size_t H,
                                            const size_t OH)
{
    auto numel = miopen::deref(inputGradDesc).GetElementSize();

    auto output_grad_tv = miopen::get_inner_expanded_tv<3>(miopen::deref(outputGradDesc));
    auto input_grad_tv  = miopen::get_inner_expanded_tv<3>(miopen::deref(inputGradDesc));

    par_ford(numel)([&](size_t gid) {
        size_t nc = gid / H, h = gid % H;
        size_t n = nc / C, c = nc % C;

        size_t oh  = (h * OH) / H;
        size_t koh = (((h + 1) * OH + H - 1) / H) - oh;

        float grad = 0;
        for(size_t ih = oh; ih < (oh + koh); ++ih)
        {
            size_t kh = ((ih + 1) * H + OH - 1) / OH - (ih * H) / OH;
            grad +=
                static_cast<float>(output_grad[output_grad_tv.get_tensor_view_idx({n, c, ih})]) /
                kh;
        }
        input_grad[input_grad_tv.get_tensor_view_idx({n, c, h})] = static_cast<Tcheck>(grad);
    });
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int32_t mloAdaptiveAvgPoolBackward2dRunHost(const miopenTensorDescriptor_t outputGradDesc,
                                            const miopenTensorDescriptor_t inputGradDesc,
                                            const Tgpu* output_grad,
                                            Tcheck* input_grad,
                                            const size_t C,
                                            const size_t H,
                                            const size_t W,
                                            const size_t OH,
                                            const size_t OW)
{
    auto numel = miopen::deref(inputGradDesc).GetElementSize();

    auto output_grad_tv = miopen::get_inner_expanded_tv<4>(miopen::deref(outputGradDesc));
    auto input_grad_tv  = miopen::get_inner_expanded_tv<4>(miopen::deref(inputGradDesc));

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
            size_t kh = ((ih + 1) * H + OH - 1) / OH - (ih * H) / OH;
            for(size_t iw = ow; iw < (ow + kow); ++iw)
            {
                size_t kw = ((iw + 1) * W + OW - 1) / OW - (iw * W) / OW;
                grad += static_cast<float>(
                            output_grad[output_grad_tv.get_tensor_view_idx({n, c, ih, iw})]) /
                        (kh * kw);
            }
        }

        input_grad[input_grad_tv.get_tensor_view_idx({n, c, h, w})] = static_cast<Tcheck>(grad);
    });
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int32_t mloAdaptiveAvgPoolBackward3dRunHost(const miopenTensorDescriptor_t outputGradDesc,
                                            const miopenTensorDescriptor_t inputGradDesc,
                                            const Tgpu* output_grad,
                                            Tcheck* input_grad,
                                            const size_t C,
                                            const size_t D,
                                            const size_t H,
                                            const size_t W,
                                            const size_t OD,
                                            const size_t OH,
                                            const size_t OW)
{
    auto numel = miopen::deref(inputGradDesc).GetElementSize();

    auto output_grad_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(outputGradDesc));
    auto input_grad_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(inputGradDesc));

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
            size_t kd = ((id + 1) * D + OD - 1) / OD - (id * D) / OD;
            for(size_t ih = oh; ih < (oh + koh); ++ih)
            {
                size_t kh = ((ih + 1) * H + OH - 1) / OH - (ih * H) / OH;
                for(size_t iw = ow; iw < (ow + kow); ++iw)
                {
                    size_t kw = ((iw + 1) * W + OW - 1) / OW - (iw * W) / OW;
                    grad +=
                        static_cast<float>(
                            output_grad[output_grad_tv.get_tensor_view_idx({n, c, id, ih, iw})]) /
                        (kd * kh * kw);
                }
            }
        }

        input_grad[input_grad_tv.get_tensor_view_idx({n, c, d, h, w})] = static_cast<Tcheck>(grad);
    });
    return miopenStatusSuccess;
}
