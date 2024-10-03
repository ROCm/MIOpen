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
#ifndef GUARD_CPU_AVGPOOL_HPP
#define GUARD_CPU_AVGPOOL_HPP

#include "tensor_holder.hpp"
#include <miopen/tensor_view_utils.hpp>

template <class T>
void cpu_avgpool_forward_2d(tensor<T> input,
                            tensor<T>& output,
                            long N,
                            long C,
                            long H,
                            long W,
                            long OH,
                            long OW,
                            tensor<long> ksize,
                            tensor<long> stride,
                            tensor<long> padding,
                            bool count_include_pad,
                            long divisor_override)
{
    auto dims  = input.desc.GetLengths();
    auto numel = output.desc.GetElementSize();

    auto input_tv  = miopen::get_inner_expanded_tv<4>(input.desc);
    auto output_tv = miopen::get_inner_expanded_tv<4>(output.desc);

    for(long gid = 0; gid < numel; gid++)
    {
        long ncoh = gid / OW, ow = gid % OW;
        long nc = ncoh / OH, oh = ncoh % OH;
        long n = nc / C, c = nc % C;
        long R  = ksize[0];
        long S  = ksize[1];
        long sh = stride[0];
        long sw = stride[1];
        long ph = padding[0];
        long pw = padding[1];

        if(n >= N)
            return;

        float m = 0;
        for(long r = 0; r < R; ++r)
        {
            for(long s = 0; s < S; ++s)
            {
                // input idx : (n, c, h, w)
                long h = oh * sh - ph + r;
                if(h < 0 || h >= H)
                    continue;
                long w = ow * sw - pw + s;
                if(w < 0 || w >= W)
                    continue;
                // long input_idx = ((n * C + c) * H + h) * W + w;
                m += static_cast<float>(input[input_tv.get_tensor_view_idx({n, c, h, w})]);
            }
        }

        long hstart = oh * sh - ph;
        long wstart = ow * sw - pw;
        long hend   = min(hstart + R, H + ph);
        long wend   = min(wstart + S, W + pw);

        const long pool_size = (hend - hstart) * (wend - wstart);

        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend   = min(hend, H);
        wend   = min(wend, W);

        long divide_factor;
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

        output[output_tv.get_tensor_view_idx({n, c, oh, ow})] = static_cast<T>(val);
    }
}

template <class T>
void cpu_avgpool_forward_3d(tensor<T> input,
                            tensor<T>& output,
                            long N,
                            long C,
                            long D,
                            long H,
                            long W,
                            long OD,
                            long OH,
                            long OW,
                            tensor<long> ksize,
                            tensor<long> stride,
                            tensor<long> padding,
                            bool count_include_pad,
                            long divisor_override)
{
    auto dims  = input.desc.GetLengths();
    auto numel = output.desc.GetElementSize();

    auto input_tv  = miopen::get_inner_expanded_tv<5>(input.desc);
    auto output_tv = miopen::get_inner_expanded_tv<5>(output.desc);

    for(long gid = 0; gid < numel; gid++)
    {
        long ncodoh = gid / OW, ow = gid % OW;
        long ncod = ncodoh / OH, oh = ncodoh % OH;
        long nc = ncod / OD, od = ncod % OD;
        long n = nc / C, c = nc % C;
        long KD = ksize[0];
        long R  = ksize[1];
        long S  = ksize[2];
        long sd = stride[0];
        long sh = stride[1];
        long sw = stride[2];
        long pd = padding[0];
        long ph = padding[1];
        long pw = padding[2];

        if(n >= N)
            return;
        float sum = 0;
        for(long kd = 0; kd < KD; ++kd)
        {
            for(long r = 0; r < R; ++r)
            {
                for(long s = 0; s < S; ++s)
                {
                    // input idx : (n, c, d, h, w)
                    long d = od * sd - pd + kd;
                    if(d < 0 || d >= D)
                        continue;
                    long h = oh * sh - ph + r;
                    if(h < 0 || h >= H)
                        continue;
                    long w = ow * sw - pw + s;
                    if(w < 0 || w >= W)
                        continue;
                    // long input_idx = ((n * C + c) * H + h) * W + w;
                    sum += static_cast<float>(input[input_tv.get_tensor_view_idx({n, c, d, h, w})]);
                }
            }
        }
        long dstart = od * sd - pd;
        long hstart = oh * sh - ph;
        long wstart = ow * sw - pw;
        long dend   = min(dstart + KD, D + pd);
        long hend   = min(hstart + R, H + ph);
        long wend   = min(wstart + S, W + pw);

        const long pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
        dstart               = max(dstart, 0);
        hstart               = max(hstart, 0);
        wstart               = max(wstart, 0);
        dend                 = min(dend, D);
        hend                 = min(hend, H);
        wend                 = min(wend, W);

        long divide_factor;
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
        output[output_tv.get_tensor_view_idx({n, c, od, oh, ow})] = static_cast<T>(val);
    }
}

template <class T>
void cpu_avgpool_backward_2d(tensor<T> output_grad,
                             tensor<T>& input_grad,
                             long N,
                             long C,
                             long H,
                             long W,
                             long OH,
                             long OW,
                             tensor<long> ksize,
                             tensor<long> stride,
                             tensor<long> padding,
                             bool count_include_pad,
                             long divisor_override)
{
    auto dims  = input_grad.desc.GetLengths();
    auto numel = input_grad.desc.GetElementSize();

    auto output_grad_tv = miopen::get_inner_expanded_tv<4>(output_grad.desc);
    auto input_grad_tv  = miopen::get_inner_expanded_tv<4>(input_grad.desc);

    for(long gid = 0; gid < numel; gid++)
    {
        long nch = gid / W, w = gid % W;
        long nc = nch / H, h = nch % H;
        long n = nc / C, c = nc % C;
        long R  = ksize[0];
        long S  = ksize[1];
        long sh = stride[0];
        long sw = stride[1];
        long ph = padding[0];
        long pw = padding[1];

        if(n >= N)
            return;

        float grad = 0;
        for(long r = 0; r < R; ++r)
        {
            for(long s = 0; s < S; ++s)
            {
                long ohsh = h + ph - r;
                if(ohsh % sh != 0)
                    continue;
                long oh = ohsh / sh;
                if(oh < 0 || oh >= OH)
                    continue;
                long owsw = w + pw - s;
                if(owsw % sw != 0)
                    continue;
                long ow = owsw / sw;
                if(ow < 0 || ow >= OW)
                    continue;

                long hstart = oh * sh - ph;
                long wstart = ow * sw - pw;
                long hend   = min(hstart + R, H + ph);
                long wend   = min(wstart + S, W + pw);

                const long pool_size = (hend - hstart) * (wend - wstart);

                hstart = max(hstart, 0);
                wstart = max(wstart, 0);
                hend   = min(hend, H);
                wend   = min(wend, W);

                long divide_factor;
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
        input_grad[input_grad_tv.get_tensor_view_idx({n, c, h, w})] = static_cast<T>(grad);
    }
}

template <class T>
void cpu_avgpool_backward_3d(tensor<T> output_grad,
                             tensor<T>& input_grad,
                             long N,
                             long C,
                             long D,
                             long H,
                             long W,
                             long OD,
                             long OH,
                             long OW,
                             tensor<long> ksize,
                             tensor<long> stride,
                             tensor<long> padding,
                             bool count_include_pad,
                             long divisor_override)
{
    auto dims  = input_grad.desc.GetLengths();
    auto numel = input_grad.desc.GetElementSize();

    auto output_grad_tv = miopen::get_inner_expanded_tv<5>(output_grad.desc);
    auto input_grad_tv  = miopen::get_inner_expanded_tv<5>(input_grad.desc);

    for(long gid = 0; gid < numel; gid++)
    {
        long ncdh = gid / W, w = gid % W;
        long ncd = ncdh / H, h = ncdh % H;
        long nc = ncd / D, d = ncd % D;
        long n = nc / C, c = nc % C;
        long KD = ksize[0];
        long R  = ksize[1];
        long S  = ksize[2];
        long sd = stride[0];
        long sh = stride[1];
        long sw = stride[2];
        long pd = padding[0];
        long ph = padding[1];
        long pw = padding[2];

        if(n >= N)
            return;

        float grad = 0;
        for(long kd = 0; kd < KD; ++kd)
        {
            for(long r = 0; r < R; ++r)
            {
                for(long s = 0; s < S; ++s)
                {
                    long odsd = d + pd - kd;
                    if(odsd % sd != 0)
                        continue;
                    long od = odsd / sd;
                    if(od < 0 || od >= OD)
                        continue;

                    long ohsh = h + ph - r;
                    if(ohsh % sh != 0)
                        continue;
                    long oh = ohsh / sh;
                    if(oh < 0 || oh >= OH)
                        continue;

                    long owsw = w + pw - s;
                    if(owsw % sw != 0)
                        continue;
                    long ow = owsw / sw;
                    if(ow < 0 || ow >= OW)
                        continue;

                    long dstart = od * sd - pd;
                    long hstart = oh * sh - ph;
                    long wstart = ow * sw - pw;
                    long dend   = min(dstart + KD, D + pd);
                    long hend   = min(hstart + R, H + ph);
                    long wend   = min(wstart + S, W + pw);

                    const long pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
                    dstart               = max(dstart, 0);
                    hstart               = max(hstart, 0);
                    wstart               = max(wstart, 0);
                    dend                 = min(dend, D);
                    hend                 = min(hend, H);
                    wend                 = min(wend, W);
                    long divide_factor;
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
        input_grad[input_grad_tv.get_tensor_view_idx({n, c, d, h, w})] = static_cast<T>(grad);
    }
}

#endif
