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

#include "tensor_holder.hpp"
#include <miopen/interpolate/utils.hpp>

inline float compute_linear_scale_factor(float scale_factor,
                                         long input_size,
                                         long output_size,
                                         bool align_corners)
{
    if(align_corners)
    {
        if(input_size == 1)
        {
            return (float)output_size;
        }
        return (float)(output_size - 1) / (input_size - 1);
    }
    else if(scale_factor == 0)
    {
        return (float)output_size / input_size;
    }
    else
    {
        return (float)scale_factor;
    }
}

inline float get_src_index(long dest_index, float scale_factor, bool align_corners)
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

inline long linear_back_index(long src, float scale_factor, bool align_corners)
{
    return (long)ceil(get_src_index(src, 1.f / scale_factor, align_corners));
}

inline void compute_linear_back_index_from_to(long src,
                                              long input_isze,
                                              long output_size,
                                              float scale_factor,
                                              bool align_corners,
                                              long* from,
                                              long* to)
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
        *to = min(output_size, linear_back_index(src + 1, scale_factor, align_corners));
    }
}

inline void compute_source_index_and_lambda(long h,
                                            float scale_factor,
                                            long Hin,
                                            long Hout,
                                            bool align_corners,
                                            long* hin_index0,
                                            long* hin_index1,
                                            float* lambda0,
                                            float* lambda1)
{
    float hin_index_actual = (float)max((float)0., get_src_index(h, scale_factor, align_corners));
    *hin_index0            = (long)hin_index_actual;
    *hin_index1            = min(*hin_index0 + 1, Hin - 1);
    *lambda1               = hin_index_actual - *hin_index0;
    *lambda0               = 1.f - *lambda1;
}

inline float get_back_lambda(long src, long src0, long src1, float lambda0, float lambda1)
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
    long dest, long src, float scale_factor, long Hin, long Hout, bool align_corners)
{
    if(Hin == Hout)
    {
        return 1;
    }
    long index0;
    long index1;
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
                                    const float* scale_factors,
                                    const bool align_corners)
{
    auto input_tv  = get_inner_expanded_tv<3>(input.desc);
    auto output_tv = get_inner_expanded_tv<3>(output.desc);

    for(unsigned long gid = 0; gid < nelems; ++gid)
    {
        auto tensor_layout = tensor_layout_t<3>(output_tv, gid);
        long n             = tensor_layout.layout[0];
        long c             = tensor_layout.layout[1];
        long h             = tensor_layout.layout[2];

        long Hin  = input_tv.size[2];
        long Hout = output_tv.size[2];
        if(Hin == Hout || Hout == 1)
        {
            output[output_tv.get_tensor_view_idx(tensor_layout)] =
                input[input_tv.get_tensor_view_idx(tensor_layout)];
            continue;
        }

        float scale_factor_h = scale_factors[0];
        scale_factor_h = compute_linear_scale_factor(scale_factor_h, Hin, Hout, align_corners);

        long hin_index0;
        long hin_index1;
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
                                     tensor<T> output_grad,
                                     const size_t nelems,
                                     const float* scale_factors,
                                     const bool align_corners)
{
    auto output_grad_tv = get_inner_expanded_tv<3>(output_grad.desc);
    auto input_grad_tv  = get_inner_expanded_tv<3>(input_grad.desc);

    for(unsigned long gid = 0; gid < nelems; ++gid)
    {
        auto tensor_layout = tensor_layout_t<3>(input_grad_tv, gid);
        long n             = tensor_layout.layout[0];
        long c             = tensor_layout.layout[1];
        long h             = tensor_layout.layout[2];

        long Hin  = input_grad_tv.size[2];
        long Hout = output_grad_tv.size[2];

        if(Hin == Hout)
        {
            input_grad[input_grad_tv.get_tensor_view_idx(tensor_layout)] =
                output_grad[output_grad_tv.get_tensor_view_idx(tensor_layout)];
            continue;
        }

        float scale_factor_h = scale_factors[0];
        float scale_factor = compute_linear_scale_factor(scale_factor_h, Hin, Hout, align_corners);

        long from, to;
        compute_linear_back_index_from_to(h, Hin, Hout, scale_factor, align_corners, &from, &to);

        float output = 0;
        for(long i = from; i < to; i++)
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
                                      const float* scale_factors,
                                      const bool align_corners)
{
    auto input_tv  = get_inner_expanded_tv<4>(input.desc);
    auto output_tv = get_inner_expanded_tv<4>(output.desc);

    for(unsigned long gid = 0; gid < nelems; ++gid)
    {
        auto tensor_layout = tensor_layout_t<4>(output_tv, gid);
        long n             = tensor_layout.layout[0];
        long c             = tensor_layout.layout[1];
        long h             = tensor_layout.layout[2];
        long w             = tensor_layout.layout[3];

        long Hin  = input_tv.size[2];
        long Hout = output_tv.size[2];
        long Win  = input_tv.size[3];
        long Wout = output_tv.size[3];

        if(Hin == Hout && Win == Wout)
        {
            output[output_tv.get_tensor_view_idx(tensor_layout)] =
                input[input_tv.get_tensor_view_idx(tensor_layout)];
            continue;
        }

        long hin_index0 = h;
        long hin_index1 = h;
        float hlambda0  = 1;
        float hlambda1  = 0;
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

        long win_index0 = w;
        long win_index1 = w;
        float wlambda0  = 1;
        float wlambda1  = 0;
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
                                       tensor<T> output_grad,
                                       const size_t nelems,
                                       const float* scale_factors,
                                       const bool align_corners)
{
    auto output_grad_tv = get_inner_expanded_tv<4>(output_grad.desc);
    auto input_grad_tv  = get_inner_expanded_tv<4>(input_grad.desc);

    for(unsigned long gid = 0; gid < nelems; ++gid)
    {
        auto tensor_layout = tensor_layout_t<4>(input_grad_tv, gid);
        long n             = tensor_layout.layout[0];
        long c             = tensor_layout.layout[1];
        long h             = tensor_layout.layout[2];
        long w             = tensor_layout.layout[3];

        long Hin  = input_grad_tv.size[2];
        long Hout = output_grad_tv.size[2];
        long Win  = input_grad_tv.size[3];
        long Wout = output_grad_tv.size[3];

        float scale_factor_h = scale_factors[0];
        float scale_factor_h_ =
            compute_linear_scale_factor(scale_factor_h, Hin, Hout, align_corners);

        float scale_factor_w = scale_factors[1];
        float scale_factor_w_ =
            compute_linear_scale_factor(scale_factor_w, Win, Wout, align_corners);

        long h_from, h_to;
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
        long w_from, w_to;
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
        for(long i = h_from; i < h_to; i++)
        {
            float h_lambda = compute_back_lambda(i, h, scale_factor_h_, Hin, Hout, align_corners);
            if(h_lambda == 0.)
                continue;
            for(long j = w_from; j < w_to; j++)
            {
                float w_lambda =
                    compute_back_lambda(j, w, scale_factor_w_, Win, Wout, align_corners);

                tensor_layout_t<4> output_layout;
                output_layout.layout[0] = n;
                output_layout.layout[1] = c;
                output_layout.layout[2] = i;
                output_layout.layout[3] = j; // Corrected index from 4 to 3

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
                                       const float* scale_factors,
                                       const bool align_corners)
{
    auto input_tv  = get_inner_expanded_tv<5>(input.desc);
    auto output_tv = get_inner_expanded_tv<5>(output.desc);
}
template <class T>
void cpu_interpolate_trilinear_backward(tensor<T>& input_grad,
                                        tensor<T> output_grad,
                                        const size_t nelems,
                                        const float* scale_factors,
                                        const bool align_corners)
{
    auto output_grad_tv = get_inner_expanded_tv<5>(output_grad.desc);
    auto input_grad_tv  = get_inner_expanded_tv<5>(input_grad.desc);

    for(unsigned long gid = 0; gid < nelems; ++gid)
    {
        auto tensor_layout = tensor_layout_t<5>(input_grad_tv, gid);
        long n             = tensor_layout.layout[0];
        long c             = tensor_layout.layout[1];
        long d             = tensor_layout.layout[2];
        long h             = tensor_layout.layout[3];
        long w             = tensor_layout.layout[4];

        long Din  = input_grad_tv.size[2];
        long Dout = output_grad_tv.size[2];
        long Hin  = input_grad_tv.size[3];
        long Hout = output_grad_tv.size[3];
        long Win  = input_grad_tv.size[4];
        long Wout = output_grad_tv.size[4];

        float scale_factor_d = scale_factors[0];
        float scale_factor_d_ =
            compute_linear_scale_factor(scale_factor_d, Din, Dout, align_corners);

        float scale_factor_h = scale_factors[1];
        float scale_factor_h_ =
            compute_linear_scale_factor(scale_factor_h, Hin, Hout, align_corners);

        float scale_factor_w = scale_factors[2];
        float scale_factor_w_ =
            compute_linear_scale_factor(scale_factor_w, Win, Wout, align_corners);

        long d_from, d_to, h_from, h_to, w_from, w_to;
        compute_linear_back_index_from_to(
            d, Din, Dout, scale_factor_d_, align_corners, &d_from, &d_to);
        compute_linear_back_index_from_to(
            h, Hin, Hout, scale_factor_h_, align_corners, &h_from, &h_to);
        compute_linear_back_index_from_to(
            w, Win, Wout, scale_factor_w_, align_corners, &w_from, &w_to);

        float output = 0;
        for(long i = d_from; i < d_to; i++)
        {
            float d_lambda = compute_back_lambda(i, d, scale_factor_d_, Din, Dout, align_corners);
            for(long j = h_from; j < h_to; j++)
            {
                float h_lambda =
                    compute_back_lambda(j, h, scale_factor_h_, Hin, Hout, align_corners);
                for(long k = w_from; k < w_to; k++)
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
#endif // GUARD_CPU_INTERPOLATE_HPP