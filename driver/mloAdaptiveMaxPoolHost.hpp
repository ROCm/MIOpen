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

#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <../test/ford.hpp>

template <typename Tgpu, typename Tcheck>
int32_t mloAdaptiveMaxPoolForward1dRunHost(const miopenTensorDescriptor_t inputDesc,
                                           const miopenTensorDescriptor_t outputDesc,
                                           const miopenTensorDescriptor_t indicesDesc,
                                           const Tgpu* input,
                                           Tcheck* output,
                                           int64_t* indices,
                                           uint64_t N,
                                           uint64_t C,
                                           uint64_t H,
                                           uint64_t OH,
                                           bool use_indices)
{
    auto dims  = miopen::deref(inputDesc).GetLengths();
    auto numel = miopen::deref(outputDesc).GetElementSize();

    auto input_tv  = miopen::get_inner_expanded_tv<3>(miopen::deref(inputDesc));
    auto output_tv = miopen::get_inner_expanded_tv<3>(miopen::deref(outputDesc));
    tensor_view_t<3> indices_tv;
    if(use_indices)
        indices_tv = miopen::get_inner_expanded_tv<3>(miopen::deref(indicesDesc));

    par_ford(numel)([&](uint64_t gid) {
        uint64_t nc = gid / OH, oh = gid % OH;
        uint64_t n = nc / C, c = nc % C;

        uint64_t h  = oh * H / OH;
        uint64_t kh = ((oh + 1) * H + OH - 1) / OH;

        float m = -std::numeric_limits<float>::max();
        if(!use_indices)
        {
            for(uint64_t ih = h; ih < kh; ++ih)
            {
                m = std::max(m,
                             static_cast<float>(input[input_tv.get_tensor_view_idx({n, c, ih})]));
            }
            output[output_tv.get_tensor_view_idx({n, c, oh})] = static_cast<Tcheck>(m);
        }
        else
        {
            uint64_t mi = 0;
            for(uint64_t ih = h; ih < kh; ++ih)
            {
                if(static_cast<float>(input[input_tv.get_tensor_view_idx({n, c, ih})]) > m)
                {
                    m  = static_cast<float>(input[input_tv.get_tensor_view_idx({n, c, ih})]);
                    mi = ih;
                }
            }
            output[output_tv.get_tensor_view_idx({n, c, oh})]   = static_cast<Tcheck>(m);
            indices[indices_tv.get_tensor_view_idx({n, c, oh})] = mi;
        }
    });
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int32_t mloAdaptiveMaxPoolForward2dRunHost(const miopenTensorDescriptor_t inputDesc,
                                           const miopenTensorDescriptor_t outputDesc,
                                           const miopenTensorDescriptor_t indicesDesc,
                                           const Tgpu* input,
                                           Tcheck* output,
                                           int64_t* indices,
                                           uint64_t N,
                                           uint64_t C,
                                           uint64_t H,
                                           uint64_t W,
                                           uint64_t OH,
                                           uint64_t OW,
                                           bool use_indices)
{
    auto dims  = miopen::deref(inputDesc).GetLengths();
    auto numel = miopen::deref(outputDesc).GetElementSize();

    auto input_tv  = miopen::get_inner_expanded_tv<4>(miopen::deref(inputDesc));
    auto output_tv = miopen::get_inner_expanded_tv<4>(miopen::deref(outputDesc));
    tensor_view_t<4> indices_tv;
    if(use_indices)
        indices_tv = miopen::get_inner_expanded_tv<4>(miopen::deref(indicesDesc));

    par_ford(numel)([&](uint64_t gid) {
        uint64_t ncoh = gid / OW, ow = gid % OW;
        uint64_t nc = ncoh / OH, oh = ncoh % OH;
        uint64_t n = nc / C, c = nc % C;

        uint64_t h  = (oh * H) / OH;
        uint64_t kh = ((oh + 1) * H + OH - 1) / OH;

        uint64_t w  = (ow * W) / OW;
        uint64_t kw = ((ow + 1) * W + OW - 1) / OW;

        float m = -std::numeric_limits<float>::max();
        if(!use_indices)
        {
            for(uint64_t ih = h; ih < kh; ++ih)
            {
                for(uint64_t iw = w; iw < kw; ++iw)
                {
                    m = std::max(
                        m, static_cast<float>(input[input_tv.get_tensor_view_idx({n, c, ih, iw})]));
                }
            }
            output[output_tv.get_tensor_view_idx({n, c, oh, ow})] = static_cast<Tcheck>(m);
        }
        else
        {
            uint64_t mi = 0;
            for(uint64_t ih = h; ih < kh; ++ih)
            {
                for(uint64_t iw = w; iw < kw; ++iw)
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
            output[output_tv.get_tensor_view_idx({n, c, oh, ow})]   = static_cast<Tcheck>(m);
            indices[indices_tv.get_tensor_view_idx({n, c, oh, ow})] = mi;
        }
    });
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int32_t mloAdaptiveMaxPoolForward3dRunHost(const miopenTensorDescriptor_t inputDesc,
                                           const miopenTensorDescriptor_t outputDesc,
                                           const miopenTensorDescriptor_t indicesDesc,
                                           const Tgpu* input,
                                           Tcheck* output,
                                           int64_t* indices,
                                           uint64_t N,
                                           uint64_t C,
                                           uint64_t D,
                                           uint64_t H,
                                           uint64_t W,
                                           uint64_t OD,
                                           uint64_t OH,
                                           uint64_t OW,
                                           bool use_indices)
{
    auto dims  = miopen::deref(inputDesc).GetLengths();
    auto numel = miopen::deref(outputDesc).GetElementSize();

    auto input_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto output_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(outputDesc));
    tensor_view_t<5> indices_tv;
    if(use_indices)
        indices_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(indicesDesc));

    par_ford(numel)([&](uint64_t gid) {
        uint64_t ncodoh = gid / OW, ow = gid % OW;
        uint64_t ncod = ncodoh / OH, oh = ncodoh % OH;
        uint64_t nc = ncod / OD, od = ncod % OD;
        uint64_t n = nc / C, c = nc % C;

        uint64_t d  = (od * D) / OD;
        uint64_t kd = ((od + 1) * D + OD - 1) / OD;

        uint64_t h  = (oh * H) / OH;
        uint64_t kh = ((oh + 1) * H + OH - 1) / OH;

        uint64_t w  = (ow * W) / OW;
        uint64_t kw = ((ow + 1) * W + OW - 1) / OW;

        float m = -std::numeric_limits<float>::max();
        if(!use_indices)
        {
            for(uint64_t id = d; id < kd; ++id)
            {
                for(uint64_t ih = h; ih < kh; ++ih)
                {
                    for(uint64_t iw = w; iw < kw; ++iw)
                    {
                        m = std::max(m,
                                     static_cast<float>(
                                         input[input_tv.get_tensor_view_idx({n, c, id, ih, iw})]));
                    }
                }
            }
            output[output_tv.get_tensor_view_idx({n, c, od, oh, ow})] = static_cast<Tcheck>(m);
        }
        else
        {
            uint64_t mi = 0;
            for(uint64_t id = d; id < kd; ++id)
            {
                for(uint64_t ih = h; ih < kh; ++ih)
                {
                    for(uint64_t iw = w; iw < kw; ++iw)
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
            output[output_tv.get_tensor_view_idx({n, c, od, oh, ow})]   = static_cast<Tcheck>(m);
            indices[indices_tv.get_tensor_view_idx({n, c, od, oh, ow})] = mi;
        }
    });
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int32_t mloAdaptiveMaxPoolBackward1dRunHost(const miopenTensorDescriptor_t indicesDesc,
                                            const miopenTensorDescriptor_t outputGradDesc,
                                            const miopenTensorDescriptor_t inputGradDesc,
                                            const int64_t* indices,
                                            const Tgpu* output_grad,
                                            Tcheck* input_grad,
                                            uint64_t N,
                                            uint64_t C,
                                            uint64_t H,
                                            uint64_t OH)
{
    auto dims  = miopen::deref(inputGradDesc).GetLengths();
    auto numel = miopen::deref(inputGradDesc).GetElementSize();

    auto indices_tv     = miopen::get_inner_expanded_tv<3>(miopen::deref(indicesDesc));
    auto output_grad_tv = miopen::get_inner_expanded_tv<3>(miopen::deref(outputGradDesc));
    auto input_grad_tv  = miopen::get_inner_expanded_tv<3>(miopen::deref(inputGradDesc));

    par_ford(numel)([&](uint64_t gid) {
        uint64_t nc = gid / H, h = gid % H;
        uint64_t n = nc / C, c = nc % C;

        uint64_t oh  = (h * OH) / H;
        uint64_t koh = (((h + 1) * OH + H - 1) / H) - oh;

        float grad = 0;
        for(uint64_t ih = oh; ih < (oh + koh); ++ih)
        {
            uint64_t idx = indices[indices_tv.get_tensor_view_idx({n, c, ih})];
            if(idx == h)
            {
                grad +=
                    static_cast<float>(output_grad[output_grad_tv.get_tensor_view_idx({n, c, ih})]);
            }
        }
        input_grad[input_grad_tv.get_tensor_view_idx({n, c, h})] = static_cast<Tcheck>(grad);
    });
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int32_t mloAdaptiveMaxPoolBackward2dRunHost(const miopenTensorDescriptor_t indicesDesc,
                                            const miopenTensorDescriptor_t outputGradDesc,
                                            const miopenTensorDescriptor_t inputGradDesc,
                                            const int64_t* indices,
                                            const Tgpu* output_grad,
                                            Tcheck* input_grad,
                                            uint64_t N,
                                            uint64_t C,
                                            uint64_t H,
                                            uint64_t W,
                                            uint64_t OH,
                                            uint64_t OW)
{
    auto dims  = miopen::deref(inputGradDesc).GetLengths();
    auto numel = miopen::deref(inputGradDesc).GetElementSize();

    auto indices_tv     = miopen::get_inner_expanded_tv<4>(miopen::deref(indicesDesc));
    auto output_grad_tv = miopen::get_inner_expanded_tv<4>(miopen::deref(outputGradDesc));
    auto input_grad_tv  = miopen::get_inner_expanded_tv<4>(miopen::deref(inputGradDesc));

    par_ford(numel)([&](uint64_t gid) {
        uint64_t nch = gid / W, w = gid % W;
        uint64_t nc = nch / H, h = nch % H;
        uint64_t n = nc / C, c = nc % C;

        uint64_t oh  = (h * OH) / H;
        uint64_t koh = ((h + 1) * OH + H - 1) / H - oh;

        uint64_t ow  = (w * OW) / W;
        uint64_t kow = ((w + 1) * OW + W - 1) / W - ow;

        float grad = 0;
        for(uint64_t ih = oh; ih < (oh + koh); ++ih)
        {
            for(uint64_t iw = ow; iw < (ow + kow); ++iw)
            {
                if(indices[indices_tv.get_tensor_view_idx({n, c, ih, iw})] == h * W + w)
                {
                    grad += static_cast<float>(
                        output_grad[output_grad_tv.get_tensor_view_idx({n, c, ih, iw})]);
                }
            }
        }

        input_grad[input_grad_tv.get_tensor_view_idx({n, c, h, w})] = static_cast<Tcheck>(grad);
    });
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int32_t mloAdaptiveMaxPoolBackward3dRunHost(const miopenTensorDescriptor_t indicesDesc,
                                            const miopenTensorDescriptor_t outputGradDesc,
                                            const miopenTensorDescriptor_t inputGradDesc,
                                            const int64_t* indices,
                                            const Tgpu* output_grad,
                                            Tcheck* input_grad,
                                            uint64_t N,
                                            uint64_t C,
                                            uint64_t D,
                                            uint64_t H,
                                            uint64_t W,
                                            uint64_t OD,
                                            uint64_t OH,
                                            uint64_t OW)
{
    auto dims  = miopen::deref(inputGradDesc).GetLengths();
    auto numel = miopen::deref(inputGradDesc).GetElementSize();

    auto indices_tv     = miopen::get_inner_expanded_tv<5>(miopen::deref(indicesDesc));
    auto output_grad_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(outputGradDesc));
    auto input_grad_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(inputGradDesc));

    par_ford(numel)([&](uint64_t gid) {
        uint64_t ncdh = gid / W, w = gid % W;
        uint64_t ncd = ncdh / H, h = ncdh % H;
        uint64_t nc = ncd / D, d = ncd % D;
        uint64_t n = nc / C, c = nc % C;

        uint64_t od  = (d * OD) / D;
        uint64_t kod = ((d + 1) * OD + D - 1) / D - od;

        uint64_t oh  = (h * OH) / H;
        uint64_t koh = ((h + 1) * OH + H - 1) / H - oh;

        uint64_t ow  = (w * OW) / W;
        uint64_t kow = ((w + 1) * OW + W - 1) / W - ow;

        float grad = 0;
        for(uint64_t id = od; id < (od + kod); ++id)
        {
            for(uint64_t ih = oh; ih < (oh + koh); ++ih)
            {
                for(uint64_t iw = ow; iw < (ow + kow); ++iw)
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

        input_grad[input_grad_tv.get_tensor_view_idx({n, c, d, h, w})] = static_cast<Tcheck>(grad);
    });
    return miopenStatusSuccess;
}
