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
#ifndef GUARD_CPU_INTERPOLATE_HPP
#define GUARD_CPU_INTERPOLATE_HPP

#include "miopen/miopen.h"
#include "tensor_holder.hpp"
#include <miopen/interpolate/utils.hpp>

inline float compute_linear_scale_factor(float scale_factor,
                                         int64_t input_size,
                                         int64_t output_size,
                                         bool align_corners)
{
    if(align_corners)
    {
        if(input_size == 1)
        {
            return static_cast<float>(output_size);
        }
        return static_cast<float>(output_size - 1) / (input_size - 1);
    }
    else if(scale_factor == 0)
    {
        return static_cast<float>(output_size) / input_size;
    }
    else
    {
        return static_cast<float>(scale_factor);
    }
}

inline float get_src_index(int64_t dest_index, float scale_factor, bool align_corners)
{
    if(align_corners)
    {
        return dest_index / scale_factor;
    }
    else
    {
        return (dest_index + 0.5f) / scale_factor - 0.5f;
    }
}

inline int64_t linear_back_index(int64_t src, float scale_factor, bool align_corners)
{
    return static_cast<int64_t>(std::ceil(get_src_index(src, 1.f / scale_factor, align_corners)));
}

inline void compute_linear_back_index_from_to(int64_t src,
                                              int64_t input_isze,
                                              int64_t output_size,
                                              float scale_factor,
                                              bool align_corners,
                                              int64_t* from,
                                              int64_t* to)
{
    if(src - 1 < 1)
    {
        *from = 0;
    }
    else
    {
        *from = linear_back_index(src - 1, scale_factor, align_corners);
    }
    if(src + 1 > input_isze)
    {
        *to = output_size;
    }
    else
    {
        *to = std::min(output_size, linear_back_index(src + 1, scale_factor, align_corners));
    }
}

inline void compute_source_index_and_lambda(int64_t h,
                                            float scale_factor,
                                            int64_t Hin,
                                            int64_t Hout,
                                            bool align_corners,
                                            int64_t* hin_index0,
                                            int64_t* hin_index1,
                                            float* lambda0,
                                            float* lambda1)
{
    float hin_index_actual = static_cast<float>(
        std::max(static_cast<float>(0.), get_src_index(h, scale_factor, align_corners)));
    *hin_index0 = static_cast<int64_t>(hin_index_actual);
    *hin_index1 = std::min(*hin_index0 + 1, Hin - 1);
    *lambda1    = hin_index_actual - *hin_index0;
    *lambda0    = 1.f - *lambda1;
}

inline float get_back_lambda(int64_t src, int64_t src0, int64_t src1, float lambda0, float lambda1)
{
    if(src == src0)
    {
        if(src0 == src1)
        {
            return 1; // lambda0 + lambda1 = 1
        }
        return lambda0;
    }
    if(src == src1)
    {
        return lambda1;
    }
    // This case can happen due to floating point mutiplification.
    // ex> 7 * (105/9) = 87 or 86.99999995
    return 0;
}

inline float compute_back_lambda(
    int64_t dest, int64_t src, float scale_factor, int64_t Hin, int64_t Hout, bool align_corners)
{
    if(Hin == Hout)
    {
        return 1;
    }
    int64_t index0;
    int64_t index1;
    float lambda0;
    float lambda1;
    compute_source_index_and_lambda(
        dest, scale_factor, Hin, Hout, align_corners, &index0, &index1, &lambda0, &lambda1);
    return get_back_lambda(src, index0, index1, lambda0, lambda1);
}

template <class T>
void cpu_interpolate_linear_forward(const tensor<T> input,
                                    tensor<T>& output,
                                    const size_t nelems,
                                    const tensor<float> scale_factors,
                                    const bool align_corners)
{
    auto input_tv  = miopen::solver::interpolate::get_inner_expanded_tv<3>(input.desc);
    auto output_tv = miopen::solver::interpolate::get_inner_expanded_tv<3>(output.desc);

    for(int64_t gid = 0; gid < nelems; ++gid)
    {
        auto tensor_layout = tensor_layout_t<3>(output_tv, gid);
        int64_t n          = tensor_layout.layout[0];
        int64_t c          = tensor_layout.layout[1];
        int64_t h          = tensor_layout.layout[2];

        int64_t Hin  = input_tv.size[2];
        int64_t Hout = output_tv.size[2];
        if(Hin == Hout || Hout == 1)
        {
            output[output_tv.get_tensor_view_idx(tensor_layout)] =
                input[input_tv.get_tensor_view_idx(tensor_layout)];
            continue;
        }

        float scale_factor_h = scale_factors[0];
        scale_factor_h = compute_linear_scale_factor(scale_factor_h, Hin, Hout, align_corners);

        int64_t hin_index0;
        int64_t hin_index1;
        float lambda1;
        float lambda0;
        compute_source_index_and_lambda(h,
                                        scale_factor_h,
                                        Hin,
                                        Hout,
                                        align_corners,
                                        &hin_index0,
                                        &hin_index1,
                                        &lambda0,
                                        &lambda1);

        tensor_layout_t<3> input_layout0;
        input_layout0.layout[0] = n;
        input_layout0.layout[1] = c;
        input_layout0.layout[2] = hin_index0;

        tensor_layout_t<3> input_layout1;
        input_layout1.layout[0] = n;
        input_layout1.layout[1] = c;
        input_layout1.layout[2] = hin_index1;

        float input0 = input[input_tv.get_tensor_view_idx(input_layout0)];
        float input1 = input[input_tv.get_tensor_view_idx(input_layout1)];

        output[output_tv.get_tensor_view_idx(tensor_layout)] =
            static_cast<T>(input0 * lambda0 + input1 * lambda1);
    }
}

template <class T>
void cpu_interpolate_linear_backward(tensor<T>& input_grad,
                                     const tensor<T> output_grad,
                                     const size_t nelems,
                                     const tensor<float> scale_factors,
                                     const bool align_corners)
{
    auto output_grad_tv = miopen::solver::interpolate::get_inner_expanded_tv<3>(output_grad.desc);
    auto input_grad_tv  = miopen::solver::interpolate::get_inner_expanded_tv<3>(input_grad.desc);

    for(int64_t gid = 0; gid < nelems; ++gid)
    {
        auto tensor_layout = tensor_layout_t<3>(input_grad_tv, gid);
        int64_t n          = tensor_layout.layout[0];
        int64_t c          = tensor_layout.layout[1];
        int64_t h          = tensor_layout.layout[2];

        int64_t Hin  = input_grad_tv.size[2];
        int64_t Hout = output_grad_tv.size[2];

        if(Hin == Hout)
        {
            input_grad[input_grad_tv.get_tensor_view_idx(tensor_layout)] =
                output_grad[output_grad_tv.get_tensor_view_idx(tensor_layout)];
            continue;
        }

        float scale_factor_h = scale_factors[0];
        float scale_factor = compute_linear_scale_factor(scale_factor_h, Hin, Hout, align_corners);

        int64_t from, to;
        compute_linear_back_index_from_to(h, Hin, Hout, scale_factor, align_corners, &from, &to);

        float output = 0;
        for(int64_t i = from; i < to; i++)
        {
            tensor_layout_t<3> output_layout;
            output_layout.layout[0] = n;
            output_layout.layout[1] = c;
            output_layout.layout[2] = i;
            output +=
                static_cast<float>(output_grad[output_grad_tv.get_tensor_view_idx(output_layout)]) *
                compute_back_lambda(i, h, scale_factor, Hin, Hout, align_corners);
        }
        input_grad[input_grad_tv.get_tensor_view_idx(tensor_layout)] = static_cast<T>(output);
    }
}

template <class T>
void cpu_interpolate_bilinear_forward(const tensor<T> input,
                                      tensor<T>& output,
                                      const size_t nelems,
                                      const tensor<float> scale_factors,
                                      const bool align_corners)
{
    auto input_tv  = miopen::solver::interpolate::get_inner_expanded_tv<4>(input.desc);
    auto output_tv = miopen::solver::interpolate::get_inner_expanded_tv<4>(output.desc);

    for(int64_t gid = 0; gid < nelems; ++gid)
    {
        auto tensor_layout = tensor_layout_t<4>(output_tv, gid);
        int64_t n          = tensor_layout.layout[0];
        int64_t c          = tensor_layout.layout[1];
        int64_t h          = tensor_layout.layout[2];
        int64_t w          = tensor_layout.layout[3];

        int64_t Hin  = input_tv.size[2];
        int64_t Hout = output_tv.size[2];
        int64_t Win  = input_tv.size[3];
        int64_t Wout = output_tv.size[3];

        if(Hin == Hout && Win == Wout)
        {
            output[output_tv.get_tensor_view_idx(tensor_layout)] =
                input[input_tv.get_tensor_view_idx(tensor_layout)];
            continue;
        }

        int64_t hin_index0 = h;
        int64_t hin_index1 = h;
        float hlambda0     = 1;
        float hlambda1     = 0;
        if(Hin != Hout && Hout != 1)
        {
            float scale_factor_h = scale_factors[0];
            float scale_factor_h_ =
                compute_linear_scale_factor(scale_factor_h, Hin, Hout, align_corners);
            compute_source_index_and_lambda(h,
                                            scale_factor_h_,
                                            Hin,
                                            Hout,
                                            align_corners,
                                            &hin_index0,
                                            &hin_index1,
                                            &hlambda0,
                                            &hlambda1);
        }

        int64_t win_index0 = w;
        int64_t win_index1 = w;
        float wlambda0     = 1;
        float wlambda1     = 0;
        if(Win != Wout && Wout != 1)
        {
            float scale_factor_w = scale_factors[1];
            float scale_factor_w_ =
                compute_linear_scale_factor(scale_factor_w, Win, Wout, align_corners);
            compute_source_index_and_lambda(w,
                                            scale_factor_w_,
                                            Win,
                                            Wout,
                                            align_corners,
                                            &win_index0,
                                            &win_index1,
                                            &wlambda0,
                                            &wlambda1);
        }

        tensor_layout_t<4> input_layout00;
        input_layout00.layout[0] = n;
        input_layout00.layout[1] = c;
        input_layout00.layout[2] = hin_index0;
        input_layout00.layout[3] = win_index0;

        tensor_layout_t<4> input_layout01;
        input_layout01.layout[0] = n;
        input_layout01.layout[1] = c;
        input_layout01.layout[2] = hin_index0;
        input_layout01.layout[3] = win_index1;

        tensor_layout_t<4> input_layout10;
        input_layout10.layout[0] = n;
        input_layout10.layout[1] = c;
        input_layout10.layout[2] = hin_index1;
        input_layout10.layout[3] = win_index0;

        tensor_layout_t<4> input_layout11;
        input_layout11.layout[0] = n;
        input_layout11.layout[1] = c;
        input_layout11.layout[2] = hin_index1;
        input_layout11.layout[3] = win_index1;

        output[output_tv.get_tensor_view_idx(tensor_layout)] = static_cast<T>(
            (static_cast<float>(input[input_tv.get_tensor_view_idx(input_layout00)]) * wlambda0 +
             static_cast<float>(input[input_tv.get_tensor_view_idx(input_layout01)]) * wlambda1) *
                hlambda0 +
            (static_cast<float>(input[input_tv.get_tensor_view_idx(input_layout10)]) * wlambda0 +
             static_cast<float>(input[input_tv.get_tensor_view_idx(input_layout11)]) * wlambda1) *
                hlambda1);
    }
}

template <class T>
void cpu_interpolate_bilinear_backward(tensor<T>& input_grad,
                                       const tensor<T> output_grad,
                                       const size_t nelems,
                                       const tensor<float> scale_factors,
                                       const bool align_corners)
{
    auto output_grad_tv = miopen::solver::interpolate::get_inner_expanded_tv<4>(output_grad.desc);
    auto input_grad_tv  = miopen::solver::interpolate::get_inner_expanded_tv<4>(input_grad.desc);

    for(int64_t gid = 0; gid < nelems; ++gid)
    {
        auto tensor_layout = tensor_layout_t<4>(input_grad_tv, gid);
        int64_t n          = tensor_layout.layout[0];
        int64_t c          = tensor_layout.layout[1];
        int64_t h          = tensor_layout.layout[2];
        int64_t w          = tensor_layout.layout[3];

        int64_t Hin  = input_grad_tv.size[2];
        int64_t Hout = output_grad_tv.size[2];
        int64_t Win  = input_grad_tv.size[3];
        int64_t Wout = output_grad_tv.size[3];

        float scale_factor_h = scale_factors[0];
        float scale_factor_h_ =
            compute_linear_scale_factor(scale_factor_h, Hin, Hout, align_corners);

        float scale_factor_w = scale_factors[1];
        float scale_factor_w_ =
            compute_linear_scale_factor(scale_factor_w, Win, Wout, align_corners);

        int64_t h_from, h_to;
        if(Hin == Hout)
        {
            h_from = h;
            h_to   = h + 1;
        }
        else
        {
            compute_linear_back_index_from_to(
                h, Hin, Hout, scale_factor_h_, align_corners, &h_from, &h_to);
        }
        int64_t w_from, w_to;
        if(Win == Wout)
        {
            w_from = w;
            w_to   = w + 1;
        }
        else
        {
            compute_linear_back_index_from_to(
                w, Win, Wout, scale_factor_w_, align_corners, &w_from, &w_to);
        }

        float output = 0;
        for(int64_t i = h_from; i < h_to; i++)
        {
            float h_lambda = compute_back_lambda(i, h, scale_factor_h_, Hin, Hout, align_corners);
            if(h_lambda == 0.)
                continue;
            for(int64_t j = w_from; j < w_to; j++)
            {
                float w_lambda =
                    compute_back_lambda(j, w, scale_factor_w_, Win, Wout, align_corners);

                tensor_layout_t<4> output_layout;
                output_layout.layout[0] = n;
                output_layout.layout[1] = c;
                output_layout.layout[2] = i;
                output_layout.layout[3] = j;

                output += static_cast<float>(
                              output_grad[output_grad_tv.get_tensor_view_idx(output_layout)]) *
                          h_lambda * w_lambda;
            }
        }
        input_grad[input_grad_tv.get_tensor_view_idx(tensor_layout)] = static_cast<T>(output);
    }
}

template <class T>
void cpu_interpolate_trilinear_forward(const tensor<T> input,
                                       tensor<T>& output,
                                       const size_t nelems,
                                       const tensor<float> scale_factors,
                                       const bool align_corners)
{
    auto input_tv  = miopen::solver::interpolate::get_inner_expanded_tv<5>(input.desc);
    auto output_tv = miopen::solver::interpolate::get_inner_expanded_tv<5>(output.desc);

    for(int64_t gid = 0; gid < nelems; ++gid)
    {
        auto tensor_layout = tensor_layout_t<5>(output_tv, gid);
        int64_t n          = tensor_layout.layout[0];
        int64_t c          = tensor_layout.layout[1];
        int64_t d          = tensor_layout.layout[2];
        int64_t h          = tensor_layout.layout[3];
        int64_t w          = tensor_layout.layout[4];

        int64_t Din  = input_tv.size[2];
        int64_t Dout = output_tv.size[2];
        int64_t Hin  = input_tv.size[3];
        int64_t Hout = output_tv.size[3];
        int64_t Win  = input_tv.size[4];
        int64_t Wout = output_tv.size[4];

        if(Hin == Hout && Win == Wout && Din == Dout)
        {
            output[output_tv.get_tensor_view_idx(tensor_layout)] =
                input[input_tv.get_tensor_view_idx(tensor_layout)];
            continue;
        }

        int64_t din_index0 = d;
        int64_t din_index1 = d;
        float dlambda0     = 1;
        float dlambda1     = 0;
        if(Din != Dout && Dout != 1)
        {
            float scale_factor_d = scale_factors[0];
            float scale_factor_d_ =
                compute_linear_scale_factor(scale_factor_d, Din, Dout, align_corners);
            compute_source_index_and_lambda(d,
                                            scale_factor_d_,
                                            Din,
                                            Dout,
                                            align_corners,
                                            &din_index0,
                                            &din_index1,
                                            &dlambda0,
                                            &dlambda1);
        }

        int64_t hin_index0 = h;
        int64_t hin_index1 = h;
        float hlambda0     = 1;
        float hlambda1     = 0;
        if(Hin != Hout && Hout != 1)
        {
            float scale_factor_h = scale_factors[1];
            float scale_factor_h_ =
                compute_linear_scale_factor(scale_factor_h, Hin, Hout, align_corners);
            compute_source_index_and_lambda(h,
                                            scale_factor_h_,
                                            Hin,
                                            Hout,
                                            align_corners,
                                            &hin_index0,
                                            &hin_index1,
                                            &hlambda0,
                                            &hlambda1);
        }

        int64_t win_index0 = w;
        int64_t win_index1 = w;
        float wlambda0     = 1;
        float wlambda1     = 0;
        if(Win != Wout && Wout != 1)
        {
            float scale_factor_w = scale_factors[2];
            float scale_factor_w_ =
                compute_linear_scale_factor(scale_factor_w, Win, Wout, align_corners);
            compute_source_index_and_lambda(w,
                                            scale_factor_w_,
                                            Win,
                                            Wout,
                                            align_corners,
                                            &win_index0,
                                            &win_index1,
                                            &wlambda0,
                                            &wlambda1);
        }

        tensor_layout_t<5> input_layout000;
        input_layout000.layout[0] = n;
        input_layout000.layout[1] = c;
        input_layout000.layout[2] = din_index0;
        input_layout000.layout[3] = hin_index0;
        input_layout000.layout[4] = win_index0;

        tensor_layout_t<5> input_layout001;
        input_layout001.layout[0] = n;
        input_layout001.layout[1] = c;
        input_layout001.layout[2] = din_index0;
        input_layout001.layout[3] = hin_index0;
        input_layout001.layout[4] = win_index1;

        tensor_layout_t<5> input_layout010;
        input_layout010.layout[0] = n;
        input_layout010.layout[1] = c;
        input_layout010.layout[2] = din_index0;
        input_layout010.layout[3] = hin_index1;
        input_layout010.layout[4] = win_index0;

        tensor_layout_t<5> input_layout011;
        input_layout011.layout[0] = n;
        input_layout011.layout[1] = c;
        input_layout011.layout[2] = din_index0;
        input_layout011.layout[3] = hin_index1;
        input_layout011.layout[4] = win_index1;

        tensor_layout_t<5> input_layout100;
        input_layout100.layout[0] = n;
        input_layout100.layout[1] = c;
        input_layout100.layout[2] = din_index1;
        input_layout100.layout[3] = hin_index0;
        input_layout100.layout[4] = win_index0;

        tensor_layout_t<5> input_layout101;
        input_layout101.layout[0] = n;
        input_layout101.layout[1] = c;
        input_layout101.layout[2] = din_index1;
        input_layout101.layout[3] = hin_index0;
        input_layout101.layout[4] = win_index1;

        tensor_layout_t<5> input_layout110;
        input_layout110.layout[0] = n;
        input_layout110.layout[1] = c;
        input_layout110.layout[2] = din_index1;
        input_layout110.layout[3] = hin_index1;
        input_layout110.layout[4] = win_index0;

        tensor_layout_t<5> input_layout111;
        input_layout111.layout[0] = n;
        input_layout111.layout[1] = c;
        input_layout111.layout[2] = din_index1;
        input_layout111.layout[3] = hin_index1;
        input_layout111.layout[4] = win_index1;

        output[output_tv.get_tensor_view_idx(tensor_layout)] = static_cast<T>(
            (static_cast<float>(input[input_tv.get_tensor_view_idx(input_layout000)]) * wlambda0 +
             static_cast<float>(input[input_tv.get_tensor_view_idx(input_layout001)]) * wlambda1) *
                hlambda0 +
            (static_cast<float>(input[input_tv.get_tensor_view_idx(input_layout010)]) * wlambda0 +
             static_cast<float>(input[input_tv.get_tensor_view_idx(input_layout011)]) * wlambda1) *
                hlambda1 +
            (static_cast<float>(input[input_tv.get_tensor_view_idx(input_layout100)]) * wlambda0 +
             static_cast<float>(input[input_tv.get_tensor_view_idx(input_layout101)]) * wlambda1) *
                dlambda0 +
            (static_cast<float>(input[input_tv.get_tensor_view_idx(input_layout110)]) * wlambda0 +
             static_cast<float>(input[input_tv.get_tensor_view_idx(input_layout111)]) * wlambda1) *
                dlambda1);
    }
}
template <class T>
void cpu_interpolate_trilinear_backward(tensor<T>& input_grad,
                                        const tensor<T> output_grad,
                                        const size_t nelems,
                                        const tensor<float> scale_factors,
                                        const bool align_corners)
{
    auto output_grad_tv = miopen::solver::interpolate::get_inner_expanded_tv<5>(output_grad.desc);
    auto input_grad_tv  = miopen::solver::interpolate::get_inner_expanded_tv<5>(input_grad.desc);

    for(int64_t gid = 0; gid < nelems; ++gid)
    {
        auto tensor_layout = tensor_layout_t<5>(input_grad_tv, gid);
        int64_t n          = tensor_layout.layout[0];
        int64_t c          = tensor_layout.layout[1];
        int64_t d          = tensor_layout.layout[2];
        int64_t h          = tensor_layout.layout[3];
        int64_t w          = tensor_layout.layout[4];

        int64_t Din  = input_grad_tv.size[2];
        int64_t Dout = output_grad_tv.size[2];
        int64_t Hin  = input_grad_tv.size[3];
        int64_t Hout = output_grad_tv.size[3];
        int64_t Win  = input_grad_tv.size[4];
        int64_t Wout = output_grad_tv.size[4];

        float scale_factor_d = scale_factors[0];
        float scale_factor_d_ =
            compute_linear_scale_factor(scale_factor_d, Din, Dout, align_corners);

        float scale_factor_h = scale_factors[1];
        float scale_factor_h_ =
            compute_linear_scale_factor(scale_factor_h, Hin, Hout, align_corners);

        float scale_factor_w = scale_factors[2];
        float scale_factor_w_ =
            compute_linear_scale_factor(scale_factor_w, Win, Wout, align_corners);

        int64_t d_from, d_to, h_from, h_to, w_from, w_to;
        compute_linear_back_index_from_to(
            d, Din, Dout, scale_factor_d_, align_corners, &d_from, &d_to);
        compute_linear_back_index_from_to(
            h, Hin, Hout, scale_factor_h_, align_corners, &h_from, &h_to);
        compute_linear_back_index_from_to(
            w, Win, Wout, scale_factor_w_, align_corners, &w_from, &w_to);

        float output = 0;
        for(int64_t i = d_from; i < d_to; i++)
        {
            float d_lambda = compute_back_lambda(i, d, scale_factor_d_, Din, Dout, align_corners);
            for(int64_t j = h_from; j < h_to; j++)
            {
                float h_lambda =
                    compute_back_lambda(j, h, scale_factor_h_, Hin, Hout, align_corners);
                for(int64_t k = w_from; k < w_to; k++)
                {
                    float w_lambda =
                        compute_back_lambda(k, w, scale_factor_w_, Win, Wout, align_corners);
                    tensor_layout_t<5> output_layout;
                    output_layout.layout[0] = n;
                    output_layout.layout[1] = c;
                    output_layout.layout[2] = i;
                    output_layout.layout[3] = j;
                    output_layout.layout[4] = k;

                    output += output_grad[output_grad_tv.get_tensor_view_idx(output_layout)] *
                              d_lambda * h_lambda * w_lambda;
                }
            }
        }
        input_grad[input_grad_tv.get_tensor_view_idx(tensor_layout)] = output;
    }
}

inline float compute_scales_value(float scale, int64_t input_size, int64_t output_size)
{
    return (scale == 0.f) ? (static_cast<float>(input_size) / output_size) : (1.0f / scale);
}

inline int64_t
nearest_idx(int64_t output_index, int64_t input_size, int64_t output_size, float scales)
{
    if(output_size == input_size)
    {
        return output_index;
    }
    else if(output_size == 2 * input_size)
    {
        return output_index / 2;
    }
    else
    {
        float scale = compute_scales_value(scales, input_size, output_size);
        return std::min(static_cast<int64_t>((output_index * scale)), input_size);
    }
}

template <class T>
void cpu_nearest_forward(const tensor<T> input,
                         tensor<T>& output,
                         const size_t nelems,
                         const tensor<float> scale_factors)
{
    auto input_tv  = miopen::solver::interpolate::get_inner_expanded_tv<5>(input.desc);
    auto output_tv = miopen::solver::interpolate::get_inner_expanded_tv<5>(output.desc);

    for(int64_t gid = 0; gid < nelems; ++gid)
    {
        auto tensor_layout = tensor_layout_t<5>(output_tv, gid);
        int64_t n          = tensor_layout.layout[0];
        int64_t c          = tensor_layout.layout[1];
        int64_t d          = tensor_layout.layout[2];
        int64_t h          = tensor_layout.layout[3];
        int64_t w          = tensor_layout.layout[4];

        int64_t Dout = output_tv.size[2];
        int64_t Hout = output_tv.size[3];
        int64_t Wout = output_tv.size[4];
        int64_t Din  = input_tv.size[2];
        int64_t Hin  = input_tv.size[3];
        int64_t Win  = input_tv.size[4];

        int64_t x = nearest_idx(d, Din, Dout, scale_factors[0]);
        int64_t y = nearest_idx(h, Hin, Hout, scale_factors[1]);
        int64_t z = nearest_idx(w, Win, Wout, scale_factors[2]);

        tensor_layout_t<5> input_layout;
        input_layout.layout[0] = n;
        input_layout.layout[1] = c;
        input_layout.layout[2] = x;
        input_layout.layout[3] = y;
        input_layout.layout[4] = z;

        output[output_tv.get_tensor_view_idx(tensor_layout)] =
            input[input_tv.get_tensor_view_idx(input_layout)];
    }
}

inline int64_t
nearest_idx_back(int64_t input_index, int64_t input_size, int64_t output_size, float scales)
{
    if(output_size == input_size)
    {
        return input_index;
    }
    else if(output_size == 2 * input_size)
    {
        return input_index * 2;
    }
    else
    {
        float scale = compute_scales_value(scales, input_size, output_size);
        return std::min(static_cast<int64_t>(std::ceil(input_index / scale)), output_size);
    }
}

template <class T>
void cpu_nearest_backward(tensor<T>& input_grad,
                          const tensor<T> output_grad,
                          const size_t nelems,
                          const tensor<float> scale_factors)
{
    auto input_grad_tv  = miopen::solver::interpolate::get_inner_expanded_tv<5>(input_grad.desc);
    auto output_grad_tv = miopen::solver::interpolate::get_inner_expanded_tv<5>(output_grad.desc);

    for(int64_t gid = 0; gid < nelems; ++gid)
    {
        auto tensor_layout = tensor_layout_t<5>(input_grad_tv, gid);
        int64_t n          = tensor_layout.layout[0];
        int64_t c          = tensor_layout.layout[1];
        int64_t x          = tensor_layout.layout[2];
        int64_t y          = tensor_layout.layout[3];
        int64_t z          = tensor_layout.layout[4];

        int64_t Dout = output_grad_tv.size[2];
        int64_t Hout = output_grad_tv.size[3];
        int64_t Wout = output_grad_tv.size[4];
        int64_t Din  = input_grad_tv.size[2];
        int64_t Hin  = input_grad_tv.size[3];
        int64_t Win  = input_grad_tv.size[4];

        float scale_factor_d = scale_factors[0];
        float scale_factor_h = scale_factors[1];
        float scale_factor_w = scale_factors[2];

        int64_t dstart = nearest_idx_back(x, Din, Dout, scale_factor_d);
        int64_t dlimit = nearest_idx_back(x + 1, Din, Dout, scale_factor_d);
        int64_t hstart = nearest_idx_back(y, Hin, Hout, scale_factor_h);
        int64_t hlimit = nearest_idx_back(y + 1, Hin, Hout, scale_factor_h);
        int64_t wstart = nearest_idx_back(z, Win, Wout, scale_factor_w);
        int64_t wlimit = nearest_idx_back(z + 1, Win, Wout, scale_factor_w);

        float grad = 0.f;
        for(int64_t d = dstart; d < dlimit; d++)
        {
            for(int64_t h = hstart; h < hlimit; h++)
            {
                for(int64_t w = wstart; w < wlimit; w++)
                {
                    tensor_layout_t<5> output_grad_layout;
                    output_grad_layout.layout[0] = n;
                    output_grad_layout.layout[1] = c;
                    output_grad_layout.layout[2] = d;
                    output_grad_layout.layout[3] = h;
                    output_grad_layout.layout[4] = w;

                    grad += static_cast<float>(
                        output_grad[output_grad_tv.get_tensor_view_idx(output_grad_layout)]);
                }
            }
        }
        input_grad[input_grad_tv.get_tensor_view_idx(tensor_layout)] = static_cast<T>(grad);
    }
}

inline float
bicubic_idx(int64_t output_index, int64_t output_size, float scale_factor, bool align_corners)
{
    if(output_size == 1)
    {
        if(align_corners)
        {
            return 0;
        }
        return -0.5f;
    }
    return get_src_index(output_index, scale_factor, align_corners);
}

inline float cubic_convolution1(float x, float A) { return ((A + 2) * x - (A + 3)) * x * x + 1; }

inline float cubic_convolution2(float x, float A)
{
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

inline void get_cubic_upsampling_coefficients(float coeffs[4], float t)
{
    float A = -0.75f;

    float x1  = t;
    coeffs[0] = cubic_convolution2(x1 + 1.0f, A);
    coeffs[1] = cubic_convolution1(x1, A);

    float x2  = 1.0f - t;
    coeffs[2] = cubic_convolution1(x2, A);
    coeffs[3] = cubic_convolution2(x2 + 1.0f, A);
}

inline float cubic_interp1d(float x0, float x1, float x2, float x3, float t)
{
    float coeffs[4];
    get_cubic_upsampling_coefficients(coeffs, t);

    return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

inline int64_t bound(int64_t p, int64_t max_size)
{
    return std::max(std::min(p, max_size - 1), 0L);
}

template <class T>
void cpu_bicubic_forward(const tensor<T> input,
                         tensor<T>& output,
                         const size_t nelems,
                         const tensor<float> scale_factors,
                         const bool align_corners)
{
    auto input_tv  = miopen::solver::interpolate::get_inner_expanded_tv<4>(input.desc);
    auto output_tv = miopen::solver::interpolate::get_inner_expanded_tv<4>(output.desc);

    for(int64_t gid = 0; gid < nelems; ++gid)
    {
        auto tensor_layout = tensor_layout_t<4>(output_tv, gid);
        int64_t n          = tensor_layout.layout[0];
        int64_t c          = tensor_layout.layout[1];
        int64_t h          = tensor_layout.layout[2];
        int64_t w          = tensor_layout.layout[3];

        int64_t Hin  = input_tv.size[2];
        int64_t Win  = input_tv.size[3];
        int64_t Hout = output_tv.size[2];
        int64_t Wout = output_tv.size[3];
        if(Hin == Hout && Win == Wout)
        {
            output[output_tv.get_tensor_view_idx(tensor_layout)] =
                input[input_tv.get_tensor_view_idx(tensor_layout)];
            continue;
        }

        float scale_factor_h = scale_factors[0];
        float scale_factor_h_ =
            compute_linear_scale_factor(scale_factor_h, Hin, Hout, align_corners);
        float real_y = bicubic_idx(h, Hout, scale_factor_h_, align_corners);
        int64_t in_y = static_cast<int64_t>(std::floor(real_y));
        float t_y    = real_y - in_y;

        float scale_factor_w = scale_factors[1];
        float scale_factor_w_ =
            compute_linear_scale_factor(scale_factor_w, Win, Wout, align_corners);
        float real_x = bicubic_idx(w, Wout, scale_factor_w_, align_corners);
        int64_t in_x = static_cast<int64_t>(std::floor(real_x));
        float t_x    = real_x - in_x;

        float coefficients[4];
#pragma unroll
        for(int k = 0; k < 4; k++)
        {
            int64_t y = bound(in_y - 1 + k, Hin);
            tensor_layout_t<4> input_layout0;
            input_layout0.layout[0] = n;
            input_layout0.layout[1] = c;
            input_layout0.layout[2] = y;
            input_layout0.layout[3] = bound(in_x - 1, Win);

            tensor_layout_t<4> input_layout1;
            input_layout1.layout[0] = n;
            input_layout1.layout[1] = c;
            input_layout1.layout[2] = y;
            input_layout1.layout[3] = bound(in_x - 0, Win);

            tensor_layout_t<4> input_layout2;
            input_layout2.layout[0] = n;
            input_layout2.layout[1] = c;
            input_layout2.layout[2] = y;
            input_layout2.layout[3] = bound(in_x + 1, Win);

            tensor_layout_t<4> input_layout3;
            input_layout3.layout[0] = n;
            input_layout3.layout[1] = c;
            input_layout3.layout[2] = y;
            input_layout3.layout[3] = bound(in_x + 2, Win);

            coefficients[k] = cubic_interp1d(
                static_cast<float>(input[input_tv.get_tensor_view_idx(input_layout0)]),
                static_cast<float>(input[input_tv.get_tensor_view_idx(input_layout1)]),
                static_cast<float>(input[input_tv.get_tensor_view_idx(input_layout2)]),
                static_cast<float>(input[input_tv.get_tensor_view_idx(input_layout3)]),
                t_x);
        }
        output[output_tv.get_tensor_view_idx(tensor_layout)] = static_cast<T>(cubic_interp1d(
            coefficients[0], coefficients[1], coefficients[2], coefficients[3], t_y));
    }
}

template <class T>
void cpu_bicubic_backward(tensor<T>& input_grad,
                          const tensor<T> output_grad,
                          const size_t nelems,
                          const tensor<float> scale_factors,
                          const bool align_corners)
{
    auto input_grad_tv  = miopen::solver::interpolate::get_inner_expanded_tv<4>(input_grad.desc);
    auto output_grad_tv = miopen::solver::interpolate::get_inner_expanded_tv<4>(output_grad.desc);

    std::vector<float> workspace;
    workspace.resize(nelems, 0.f);

    int64_t Hin  = input_grad_tv.size[2];
    int64_t Hout = output_grad_tv.size[2];
    int64_t Win  = input_grad_tv.size[3];
    int64_t Wout = output_grad_tv.size[3];

    size_t out_elems = output_grad.desc.GetElementSize();

    for(int64_t gid = 0; gid < out_elems; ++gid)
    {
        auto tensor_layout = tensor_layout_t<4>(output_grad_tv, gid);
        int64_t n          = tensor_layout.layout[0];
        int64_t c          = tensor_layout.layout[1];
        int64_t h          = tensor_layout.layout[2];
        int64_t w          = tensor_layout.layout[3];

        if(Hin == Hout && Win == Wout)
        {
            input_grad[input_grad_tv.get_tensor_view_idx(tensor_layout)] =
                output_grad[output_grad_tv.get_tensor_view_idx(tensor_layout)];
            continue;
        }

        float scale_factor_h = scale_factors[0];
        float scale_factor_h_ =
            compute_linear_scale_factor(scale_factor_h, Hin, Hout, align_corners);
        float real_y = bicubic_idx(h, Hout, scale_factor_h_, align_corners);
        int64_t in_y = static_cast<int64_t>(std::floor(real_y));
        float t_y    = real_y - static_cast<float>(in_y);

        float scale_factor_w = scale_factors[1];
        float scale_factor_w_ =
            compute_linear_scale_factor(scale_factor_w, Win, Wout, align_corners);
        float real_x = bicubic_idx(w, Wout, scale_factor_w_, align_corners);
        int64_t in_x = static_cast<int64_t>(std::floor(real_x));
        float t_x    = real_x - static_cast<float>(in_x);

        float y_coeffs[4];
        float x_coeffs[4];
        get_cubic_upsampling_coefficients(y_coeffs, t_y);
        get_cubic_upsampling_coefficients(x_coeffs, t_x);

        float out_value =
            static_cast<float>(output_grad[output_grad_tv.get_tensor_view_idx(tensor_layout)]);

#pragma unroll
        for(int i = 0; i < 4; i++)
        {
            int64_t input_h = bound(in_y - 1 + i, Hin);
#pragma unroll
            for(int j = 0; j < 4; j++)
            {
                int64_t input_w = bound(in_x - 1 + j, Win);
                tensor_layout_t<4> in_grad_layout;
                in_grad_layout.layout[0] = n;
                in_grad_layout.layout[1] = c;
                in_grad_layout.layout[2] = input_h;
                in_grad_layout.layout[3] = input_w;

                workspace[input_grad_tv.get_tensor_view_idx(in_grad_layout)] +=
                    out_value * y_coeffs[i] * x_coeffs[j];
            }
        }
    }

    if(!(Hin == Hout && Win == Wout))
    {
        for(int64_t gid = 0; gid < nelems; ++gid)
        {
            auto tensor_layout = tensor_layout_t<4>(input_grad_tv, gid);
            input_grad[input_grad_tv.get_tensor_view_idx(tensor_layout)] =
                static_cast<T>(workspace[input_grad_tv.get_tensor_view_idx(tensor_layout)]);
        }
    }
}

template <class T>
void cpu_interpolate_forward(const tensor<T> input,
                             tensor<T>& output,
                             const size_t nelems,
                             const tensor<float> scale_factors,
                             const bool align_corners,
                             const miopenInterpolateMode_t mode)
{
    if(mode == MIOPEN_INTERPOLATE_MODE_NEAREST)
    {
        cpu_nearest_forward(input, output, nelems, scale_factors);
    }
    else if(mode == MIOPEN_INTERPOLATE_MODE_LINEAR)
    {
        cpu_interpolate_linear_forward(input, output, nelems, scale_factors, align_corners);
    }
    else if(mode == MIOPEN_INTERPOLATE_MODE_BILINEAR)
    {
        cpu_interpolate_bilinear_forward(input, output, nelems, scale_factors, align_corners);
    }
    else if(mode == MIOPEN_INTERPOLATE_MODE_TRILINEAR)
    {
        cpu_interpolate_trilinear_forward(input, output, nelems, scale_factors, align_corners);
    }
    else if(mode == MIOPEN_INTERPOLATE_MODE_BICUBIC)
    {
        cpu_bicubic_forward(input, output, nelems, scale_factors, align_corners);
    }
}

template <class T>
void cpu_interpolate_backward(tensor<T>& input_grad,
                              const tensor<T> output_grad,
                              const size_t nelems,
                              const tensor<float> scale_factors,
                              const bool align_corners,
                              const miopenInterpolateMode_t mode)
{
    if(mode == MIOPEN_INTERPOLATE_MODE_NEAREST)
    {
        cpu_nearest_backward(input_grad, output_grad, nelems, scale_factors);
    }
    else if(mode == MIOPEN_INTERPOLATE_MODE_LINEAR)
    {
        cpu_interpolate_linear_backward(
            input_grad, output_grad, nelems, scale_factors, align_corners);
    }
    else if(mode == MIOPEN_INTERPOLATE_MODE_BILINEAR)
    {
        cpu_interpolate_bilinear_backward(
            input_grad, output_grad, nelems, scale_factors, align_corners);
    }
    else if(mode == MIOPEN_INTERPOLATE_MODE_TRILINEAR)
    {
        cpu_interpolate_trilinear_backward(
            input_grad, output_grad, nelems, scale_factors, align_corners);
    }
    else if(mode == MIOPEN_INTERPOLATE_MODE_BICUBIC)
    {
        cpu_bicubic_backward(input_grad, output_grad, nelems, scale_factors, align_corners);
    }
}

#endif // GUARD_CPU_INTERPOLATE_HPP
