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
int32_t mloAvgPoolForward2dRunHost(const miopenTensorDescriptor_t inputDesc,
                                   const miopenTensorDescriptor_t outputDesc,
                                   const Tgpu* input,
                                   Tcheck* output,
                                   int64_t N,
                                   int64_t C,
                                   int64_t H,
                                   int64_t W,
                                   int64_t OH,
                                   int64_t OW,
                                   const int64_t* ksize,
                                   const int64_t* stride,
                                   const int64_t* padding,
                                   bool count_include_pad,
                                   int64_t divisor_override)
{
    auto dims  = miopen::deref(inputDesc).GetLengths();
    auto numel = miopen::deref(outputDesc).GetElementSize();

    auto input_tv  = miopen::get_inner_expanded_tv<4>(miopen::deref(inputDesc));
    auto output_tv = miopen::get_inner_expanded_tv<4>(miopen::deref(outputDesc));

    par_ford(numel)([&](int64_t gid) {
        int64_t ncoh = gid / OW, ow = gid % OW;
        int64_t nc = ncoh / OH, oh = ncoh % OH;
        int64_t n = nc / C, c = nc % C;
        int64_t R  = ksize[0];
        int64_t S  = ksize[1];
        int64_t sh = stride[0];
        int64_t sw = stride[1];
        int64_t ph = padding[0];
        int64_t pw = padding[1];

        float m = 0;
        for(int64_t r = 0; r < R; ++r)
        {
            for(int64_t s = 0; s < S; ++s)
            {
                // input idx : (n, c, h, w)
                int64_t h = oh * sh - ph + r;
                if(h < 0 || h >= H)
                    continue;
                int64_t w = ow * sw - pw + s;
                if(w < 0 || w >= W)
                    continue;
                // int64_t input_idx = ((n * C + c) * H + h) * W + w;
                m += static_cast<float>(input[input_tv.get_tensor_view_idx({n, c, h, w})]);
            }
        }

        int64_t hstart = oh * sh - ph;
        int64_t wstart = ow * sw - pw;
        int64_t hend   = min(hstart + R, H + ph);
        int64_t wend   = min(wstart + S, W + pw);

        const int64_t pool_size = (hend - hstart) * (wend - wstart);

        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend   = min(hend, H);
        wend   = min(wend, W);

        int64_t divide_factor;
        if(divisor_override != 0)
        {
            divide_factor = divisor_override;
        }
        else
        {
            if(count_include_pad)
            {
                divide_factor = pool_size;
            }
            else
            {
                divide_factor = (hend - hstart) * (wend - wstart);
            }
        }
        float val = m / divide_factor;

        output[output_tv.get_tensor_view_idx({n, c, oh, ow})] = static_cast<Tcheck>(val);
    });
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int32_t mloAvgPoolForward3dRunHost(const miopenTensorDescriptor_t inputDesc,
                                   const miopenTensorDescriptor_t outputDesc,
                                   const Tgpu* input,
                                   Tcheck* output,
                                   int64_t N,
                                   int64_t C,
                                   int64_t D,
                                   int64_t H,
                                   int64_t W,
                                   int64_t OD,
                                   int64_t OH,
                                   int64_t OW,
                                   const int64_t* ksize,
                                   const int64_t* stride,
                                   const int64_t* padding,
                                   bool count_include_pad,
                                   int64_t divisor_override)
{
    auto dims  = miopen::deref(inputDesc).GetLengths();
    auto numel = miopen::deref(outputDesc).GetElementSize();

    auto input_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto output_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(outputDesc));

    par_ford(numel)([&](int64_t gid) {
        int64_t ncodoh = gid / OW, ow = gid % OW;
        int64_t ncod = ncodoh / OH, oh = ncodoh % OH;
        int64_t nc = ncod / OD, od = ncod % OD;
        int64_t n = nc / C, c = nc % C;
        int64_t KD = ksize[0];
        int64_t R  = ksize[1];
        int64_t S  = ksize[2];
        int64_t sd = stride[0];
        int64_t sh = stride[1];
        int64_t sw = stride[2];
        int64_t pd = padding[0];
        int64_t ph = padding[1];
        int64_t pw = padding[2];

        float sum = 0;
        for(int64_t kd = 0; kd < KD; ++kd)
        {
            for(int64_t r = 0; r < R; ++r)
            {
                for(int64_t s = 0; s < S; ++s)
                {
                    // input idx : (n, c, d, h, w)
                    int64_t d = od * sd - pd + kd;
                    if(d < 0 || d >= D)
                        continue;
                    int64_t h = oh * sh - ph + r;
                    if(h < 0 || h >= H)
                        continue;
                    int64_t w = ow * sw - pw + s;
                    if(w < 0 || w >= W)
                        continue;
                    // int64_t input_idx = ((n * C + c) * H + h) * W + w;
                    sum += static_cast<float>(input[input_tv.get_tensor_view_idx({n, c, d, h, w})]);
                }
            }
        }
        int64_t dstart = od * sd - pd;
        int64_t hstart = oh * sh - ph;
        int64_t wstart = ow * sw - pw;
        int64_t dend   = min(dstart + KD, D + pd);
        int64_t hend   = min(hstart + R, H + ph);
        int64_t wend   = min(wstart + S, W + pw);

        const int64_t pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
        dstart                  = max(dstart, 0);
        hstart                  = max(hstart, 0);
        wstart                  = max(wstart, 0);
        dend                    = min(dend, D);
        hend                    = min(hend, H);
        wend                    = min(wend, W);

        int64_t divide_factor;
        if(divisor_override != 0)
        {
            divide_factor = divisor_override;
        }
        else
        {
            if(count_include_pad)
            {
                divide_factor = pool_size;
            }
            else
            {
                divide_factor = (dend - dstart) * (hend - hstart) * (wend - wstart);
            }
        }
        float val                                                 = sum / divide_factor;
        output[output_tv.get_tensor_view_idx({n, c, od, oh, ow})] = static_cast<Tcheck>(val);
    });
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int32_t mloAvgPoolBackward2dRunHost(const miopenTensorDescriptor_t outputGradDesc,
                                    const miopenTensorDescriptor_t inputGradDesc,
                                    Tgpu* output_grad,
                                    Tcheck* input_grad,
                                    int64_t N,
                                    int64_t C,
                                    int64_t H,
                                    int64_t W,
                                    int64_t OH,
                                    int64_t OW,
                                    const int64_t* ksize,
                                    const int64_t* stride,
                                    const int64_t* padding,
                                    bool count_include_pad,
                                    int64_t divisor_override)
{
    auto dims  = miopen::deref(inputGradDesc).GetLengths();
    auto numel = miopen::deref(inputGradDesc).GetElementSize();

    auto output_grad_tv = miopen::get_inner_expanded_tv<4>(miopen::deref(outputGradDesc));
    auto input_grad_tv  = miopen::get_inner_expanded_tv<4>(miopen::deref(inputGradDesc));

    par_ford(numel)([&](int64_t gid) {
        int64_t nch = gid / W, w = gid % W;
        int64_t nc = nch / H, h = nch % H;
        int64_t n = nc / C, c = nc % C;
        int64_t R  = ksize[0];
        int64_t S  = ksize[1];
        int64_t sh = stride[0];
        int64_t sw = stride[1];
        int64_t ph = padding[0];
        int64_t pw = padding[1];

        float grad = 0;
        for(int64_t r = 0; r < R; ++r)
        {
            for(int64_t s = 0; s < S; ++s)
            {
                int64_t ohsh = h + ph - r;
                if(ohsh % sh != 0)
                    continue;
                int64_t oh = ohsh / sh;
                if(oh < 0 || oh >= OH)
                    continue;
                int64_t owsw = w + pw - s;
                if(owsw % sw != 0)
                    continue;
                int64_t ow = owsw / sw;
                if(ow < 0 || ow >= OW)
                    continue;

                int64_t hstart = oh * sh - ph;
                int64_t wstart = ow * sw - pw;
                int64_t hend   = min(hstart + R, H + ph);
                int64_t wend   = min(wstart + S, W + pw);

                const int64_t pool_size = (hend - hstart) * (wend - wstart);

                hstart = max(hstart, 0);
                wstart = max(wstart, 0);
                hend   = min(hend, H);
                wend   = min(wend, W);

                int64_t divide_factor;
                if(divisor_override != 0)
                {
                    divide_factor = divisor_override;
                }
                else
                {
                    if(count_include_pad)
                    {
                        divide_factor = pool_size;
                    }
                    else
                    {
                        divide_factor = (hend - hstart) * (wend - wstart);
                    }
                }

                grad += static_cast<float>(
                            output_grad[output_grad_tv.get_tensor_view_idx({n, c, oh, ow})]) /
                        divide_factor;
            }
        }
        input_grad[input_grad_tv.get_tensor_view_idx({n, c, h, w})] = static_cast<Tcheck>(grad);
    });
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int32_t mloAvgPoolBackward3dRunHost(const miopenTensorDescriptor_t outputGradDesc,
                                    const miopenTensorDescriptor_t inputGradDesc,
                                    Tgpu* output_grad,
                                    Tcheck* input_grad,
                                    int64_t N,
                                    int64_t C,
                                    int64_t D,
                                    int64_t H,
                                    int64_t W,
                                    int64_t OD,
                                    int64_t OH,
                                    int64_t OW,
                                    const int64_t* ksize,
                                    const int64_t* stride,
                                    const int64_t* padding,
                                    bool count_include_pad,
                                    int64_t divisor_override)
{
    auto dims  = miopen::deref(inputGradDesc).GetLengths();
    auto numel = miopen::deref(inputGradDesc).GetElementSize();

    auto output_grad_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(outputGradDesc));
    auto input_grad_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(inputGradDesc));

    par_ford(numel)([&](int64_t gid) {
        int64_t ncdh = gid / W, w = gid % W;
        int64_t ncd = ncdh / H, h = ncdh % H;
        int64_t nc = ncd / D, d = ncd % D;
        int64_t n = nc / C, c = nc % C;
        int64_t KD = ksize[0];
        int64_t R  = ksize[1];
        int64_t S  = ksize[2];
        int64_t sd = stride[0];
        int64_t sh = stride[1];
        int64_t sw = stride[2];
        int64_t pd = padding[0];
        int64_t ph = padding[1];
        int64_t pw = padding[2];

        float grad = 0;
        for(int64_t kd = 0; kd < KD; ++kd)
        {
            for(int64_t r = 0; r < R; ++r)
            {
                for(int64_t s = 0; s < S; ++s)
                {
                    int64_t odsd = d + pd - kd;
                    if(odsd % sd != 0)
                        continue;
                    int64_t od = odsd / sd;
                    if(od < 0 || od >= OD)
                        continue;

                    int64_t ohsh = h + ph - r;
                    if(ohsh % sh != 0)
                        continue;
                    int64_t oh = ohsh / sh;
                    if(oh < 0 || oh >= OH)
                        continue;

                    int64_t owsw = w + pw - s;
                    if(owsw % sw != 0)
                        continue;
                    int64_t ow = owsw / sw;
                    if(ow < 0 || ow >= OW)
                        continue;

                    int64_t dstart = od * sd - pd;
                    int64_t hstart = oh * sh - ph;
                    int64_t wstart = ow * sw - pw;
                    int64_t dend   = min(dstart + KD, D + pd);
                    int64_t hend   = min(hstart + R, H + ph);
                    int64_t wend   = min(wstart + S, W + pw);

                    const int64_t pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
                    dstart                  = max(dstart, 0);
                    hstart                  = max(hstart, 0);
                    wstart                  = max(wstart, 0);
                    dend                    = min(dend, D);
                    hend                    = min(hend, H);
                    wend                    = min(wend, W);
                    int64_t divide_factor;
                    if(divisor_override != 0)
                    {
                        divide_factor = divisor_override;
                    }
                    else
                    {
                        if(count_include_pad)
                        {
                            divide_factor = pool_size;
                        }
                        else
                        {
                            divide_factor = (dend - dstart) * (hend - hstart) * (wend - wstart);
                        }
                    }
                    grad +=
                        static_cast<float>(
                            output_grad[output_grad_tv.get_tensor_view_idx({n, c, od, oh, ow})]) /
                        divide_factor;
                }
            }
        }
        input_grad[input_grad_tv.get_tensor_view_idx({n, c, d, h, w})] = static_cast<Tcheck>(grad);
    });
    return miopenStatusSuccess;
}
