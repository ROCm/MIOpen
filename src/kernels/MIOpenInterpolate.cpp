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
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "tensor_view.hpp"

__device__ inline FLOAT_ACCUM compute_linear_scale_factor(FLOAT_ACCUM scale_factor,
                                                          long input_size,
                                                          long output_size,
                                                          bool align_corners)
{
    if(align_corners)
    {
        if(input_size == 1)
        {
            return (FLOAT_ACCUM)output_size;
        }
        return (FLOAT_ACCUM)(output_size - 1) / (input_size - 1);
    }
    else if(scale_factor == 0)
    {
        return (FLOAT_ACCUM)output_size / input_size;
    }
    else
    {
        return (FLOAT_ACCUM)scale_factor;
    }
}

__device__ inline FLOAT_ACCUM
get_src_index(long dest_index, FLOAT_ACCUM scale_factor, bool align_corners)
{
    if(align_corners)
    {
        return dest_index / scale_factor;
    }
    else
    {
        // Follow Opencv resize logic.
        return (dest_index + 0.5f) / scale_factor - 0.5f;
    }
}

__device__ inline long linear_back_index(long src, FLOAT_ACCUM scale_factor, bool align_corners)
{
    return (long)ceil(get_src_index(src, 1.f / scale_factor, align_corners));
}

__device__ inline void compute_linear_back_index_from_to(long src,
                                                         long input_isze,
                                                         long output_size,
                                                         FLOAT_ACCUM scale_factor,
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

__device__ inline void compute_source_index_and_lambda(long h,
                                                       FLOAT_ACCUM scale_factor,
                                                       long Hin,
                                                       long Hout,
                                                       bool align_corners,
                                                       long* hin_index0,
                                                       long* hin_index1,
                                                       FLOAT_ACCUM* lambda0,
                                                       FLOAT_ACCUM* lambda1)
{
    FLOAT_ACCUM hin_index_actual =
        (FLOAT_ACCUM)max((FLOAT_ACCUM)0., get_src_index(h, scale_factor, align_corners));
    *hin_index0 = (long)hin_index_actual;
    *hin_index1 = min(*hin_index0 + 1, Hin - 1);
    *lambda1    = hin_index_actual - *hin_index0;
    *lambda0    = 1.f - *lambda1;
}

__device__ inline FLOAT_ACCUM
get_back_lambda(long src, long src0, long src1, FLOAT_ACCUM lambda0, FLOAT_ACCUM lambda1)
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

__device__ inline FLOAT_ACCUM compute_back_lambda(
    long dest, long src, FLOAT_ACCUM scale_factor, long Hin, long Hout, bool align_corners)
{
    if(Hin == Hout)
    {
        return 1;
    }
    long index0;
    long index1;
    FLOAT_ACCUM lambda0;
    FLOAT_ACCUM lambda1;
    compute_source_index_and_lambda(
        dest, scale_factor, Hin, Hout, align_corners, &index0, &index1, &lambda0, &lambda1);
    return get_back_lambda(src, index0, index1, lambda0, lambda1);
}

template <typename TI, typename TO>
__device__ inline void interpolateLinearForward(const TI* input,
                                                TO* output,
                                                const tensor_view_t<3> input_tv,
                                                const tensor_view_t<3> output_tv,
                                                const size_t nelems,
                                                const float* scale_factors,
                                                const bool align_corners)
{
    unsigned long gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= nelems)
        return;

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
        return;
    }

    FLOAT_ACCUM scale_factor_h = CVT_FP32_2ACCUM(scale_factors[0]);
    scale_factor_h = compute_linear_scale_factor(scale_factor_h, Hin, Hout, align_corners);

    long hin_index0;
    long hin_index1;
    FLOAT_ACCUM lambda1;
    FLOAT_ACCUM lambda0;
    compute_source_index_and_lambda(
        h, scale_factor_h, Hin, Hout, align_corners, &hin_index0, &hin_index1, &lambda0, &lambda1);

    tensor_layout_t<3> input_layout0;
    input_layout0.layout[0] = n;
    input_layout0.layout[1] = c;
    input_layout0.layout[2] = hin_index0;

    tensor_layout_t<3> input_layout1;
    input_layout1.layout[0] = n;
    input_layout1.layout[1] = c;
    input_layout1.layout[2] = hin_index1;

    FLOAT_ACCUM input0 = input[input_tv.get_tensor_view_idx(input_layout0)];
    FLOAT_ACCUM input1 = input[input_tv.get_tensor_view_idx(input_layout1)];

    output[output_tv.get_tensor_view_idx(tensor_layout)] =
        CVT_ACCUM2FLOAT(input0 * lambda0 + input1 * lambda1);
}

extern "C" __global__ void InterpolateLinearForward(const INPUT_TYPE* __restrict__ input,
                                                    OUTPUT_TYPE* __restrict__ output,
                                                    const tensor_view_t<3> input_tv,
                                                    const tensor_view_t<3> output_tv,
                                                    const size_t nelems,
                                                    const float* scale_factors,
                                                    const bool align_corners)
{
    interpolateLinearForward<INPUT_TYPE, OUTPUT_TYPE>(
        input, output, input_tv, output_tv, nelems, scale_factors, align_corners);
}

template <typename TI, typename TO>
__device__ inline void interpolateLinearBackward(TO* input_grad,
                                                 const TI* output_grad,
                                                 const tensor_view_t<3> input_grad_tv,
                                                 const tensor_view_t<3> output_grad_tv,
                                                 const size_t nelems,
                                                 const float* scale_factors,
                                                 const bool align_corners)
{
    unsigned long gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= nelems)
        return;

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
        return;
    }

    FLOAT_ACCUM scale_factor_h = CVT_FP32_2ACCUM(scale_factors[0]);
    FLOAT_ACCUM scale_factor =
        compute_linear_scale_factor(scale_factor_h, Hin, Hout, align_corners);

    long from, to;
    compute_linear_back_index_from_to(h, Hin, Hout, scale_factor, align_corners, &from, &to);

    FLOAT_ACCUM output = 0;
    for(long i = from; i < to; i++)
    {
        tensor_layout_t<3> output_layout;
        output_layout.layout[0] = n;
        output_layout.layout[1] = c;
        output_layout.layout[2] = i;
        output += CVT_FLOAT2ACCUM(output_grad[output_grad_tv.get_tensor_view_idx(output_layout)]) *
                  compute_back_lambda(i, h, scale_factor, Hin, Hout, align_corners);
    }
    input_grad[input_grad_tv.get_tensor_view_idx(tensor_layout)] = CVT_ACCUM2FLOAT(output);
}

extern "C" __global__ void InterpolateLinearBackward(OUTPUT_TYPE* __restrict__ input_grad,
                                                     const INPUT_TYPE* __restrict__ output_grad,
                                                     const tensor_view_t<3> input_grad_tv,
                                                     const tensor_view_t<3> output_grad_tv,
                                                     const size_t nelems,
                                                     const float* scale_factors,
                                                     const bool align_corners)
{
    interpolateLinearBackward<INPUT_TYPE, OUTPUT_TYPE>(input_grad,
                                                       output_grad,
                                                       input_grad_tv,
                                                       output_grad_tv,
                                                       nelems,
                                                       scale_factors,
                                                       align_corners);
}

template <typename TI, typename TO>
__device__ inline void interpolateBilinearForward(const TI* input,
                                                  TO* output,
                                                  const tensor_view_t<4> input_tv,
                                                  const tensor_view_t<4> output_tv,
                                                  const size_t nelems,
                                                  const float* scale_factors,
                                                  const bool align_corners)
{
    unsigned long gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= nelems)
        return;

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
        return;
    }

    long hin_index0      = h;
    long hin_index1      = h;
    FLOAT_ACCUM hlambda0 = 1;
    FLOAT_ACCUM hlambda1 = 0;
    if(Hin != Hout && Hout != 1)
    {
        FLOAT_ACCUM scale_factor_h = CVT_FP32_2ACCUM(scale_factors[0]);
        FLOAT_ACCUM scale_factor_h_ =
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

    long win_index0      = w;
    long win_index1      = w;
    FLOAT_ACCUM wlambda0 = 1;
    FLOAT_ACCUM wlambda1 = 0;
    if(Win != Wout && Wout != 1)
    {
        FLOAT_ACCUM scale_factor_w = CVT_FP32_2ACCUM(scale_factors[1]);
        FLOAT_ACCUM scale_factor_w_ =
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

    output[output_tv.get_tensor_view_idx(tensor_layout)] = CVT_ACCUM2FLOAT(
        (CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(input_layout00)]) * wlambda0 +
         CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(input_layout01)]) * wlambda1) *
            hlambda0 +
        (CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(input_layout10)]) * wlambda0 +
         CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(input_layout11)]) * wlambda1) *
            hlambda1);
}

extern "C" __global__ void InterpolateBilinearForward(const INPUT_TYPE* __restrict__ input,
                                                      OUTPUT_TYPE* __restrict__ output,
                                                      const tensor_view_t<4> input_tv,
                                                      const tensor_view_t<4> output_tv,
                                                      const size_t nelems,
                                                      const float* scale_factors,
                                                      const bool align_corners)
{
    interpolateBilinearForward<INPUT_TYPE, OUTPUT_TYPE>(
        input, output, input_tv, output_tv, nelems, scale_factors, align_corners);
}

template <typename TI, typename TO>
__device__ inline void interpolateBilinearBackward(TO* input_grad,
                                                   const TI* output_grad,
                                                   const tensor_view_t<4> input_grad_tv,
                                                   const tensor_view_t<4> output_grad_tv,
                                                   const size_t nelems,
                                                   const float* scale_factors,
                                                   const bool align_corners)
{
    unsigned long gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= nelems)
        return;

    auto tensor_layout = tensor_layout_t<4>(input_grad_tv, gid);
    long n             = tensor_layout.layout[0];
    long c             = tensor_layout.layout[1];
    long h             = tensor_layout.layout[2];
    long w             = tensor_layout.layout[3];

    long Hin  = input_grad_tv.size[2];
    long Hout = output_grad_tv.size[2];
    long Win  = input_grad_tv.size[3];
    long Wout = output_grad_tv.size[3];

    FLOAT_ACCUM scale_factor_h = CVT_FP32_2ACCUM(scale_factors[0]);
    FLOAT_ACCUM scale_factor_h_ =
        compute_linear_scale_factor(scale_factor_h, Hin, Hout, align_corners);

    FLOAT_ACCUM scale_factor_w = CVT_FP32_2ACCUM(scale_factors[1]);
    FLOAT_ACCUM scale_factor_w_ =
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

    FLOAT_ACCUM output = 0;
    for(long i = h_from; i < h_to; i++)
    {
        FLOAT_ACCUM h_lambda = compute_back_lambda(i, h, scale_factor_h_, Hin, Hout, align_corners);
        if(h_lambda == 0.)
            continue;
        for(long j = w_from; j < w_to; j++)
        {
            FLOAT_ACCUM w_lambda =
                compute_back_lambda(j, w, scale_factor_w_, Win, Wout, align_corners);

            tensor_layout_t<4> output_layout;
            output_layout.layout[0] = n;
            output_layout.layout[1] = c;
            output_layout.layout[2] = i;
            output_layout.layout[4] = j;

            output +=
                CVT_FLOAT2ACCUM(output_grad[output_grad_tv.get_tensor_view_idx(output_layout)]) *
                h_lambda * w_lambda;
        }
    }
    input_grad[input_grad_tv.get_tensor_view_idx(tensor_layout)] = CVT_ACCUM2FLOAT(output);
}

extern "C" __global__ void InterpolateBilinearBackward(OUTPUT_TYPE* __restrict__ input_grad,
                                                       const INPUT_TYPE* __restrict__ output_grad,
                                                       const tensor_view_t<4> input_grad_tv,
                                                       const tensor_view_t<4> output_grad_tv,
                                                       const size_t nelems,
                                                       const float* scale_factors,
                                                       const bool align_corners)
{
    interpolateBilinearBackward<INPUT_TYPE, OUTPUT_TYPE>(input_grad,
                                                         output_grad,
                                                         input_grad_tv,
                                                         output_grad_tv,
                                                         nelems,
                                                         scale_factors,
                                                         align_corners);
}

template <typename TI, typename TO>
__device__ inline void interpolateTrilinearForward(const TI* input,
                                                   TO* output,
                                                   const tensor_view_t<5> input_tv,
                                                   const tensor_view_t<5> output_tv,
                                                   const size_t nelems,
                                                   const float* scale_factors,
                                                   const bool align_corners)
{
    unsigned long gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= nelems)
        return;

    auto tensor_layout = tensor_layout_t<5>(output_tv, gid);
    long n             = tensor_layout.layout[0];
    long c             = tensor_layout.layout[1];
    long d             = tensor_layout.layout[2];
    long h             = tensor_layout.layout[3];
    long w             = tensor_layout.layout[4];

    long Din  = input_tv.size[2];
    long Dout = output_tv.size[2];
    long Hin  = input_tv.size[3];
    long Hout = output_tv.size[3];
    long Win  = input_tv.size[4];
    long Wout = output_tv.size[4];

    if(Hin == Hout && Win == Wout && Din == Dout)
    {
        output[output_tv.get_tensor_view_idx(tensor_layout)] =
            input[input_tv.get_tensor_view_idx(tensor_layout)];
        return;
    }

    long din_index0      = d;
    long din_index1      = d;
    FLOAT_ACCUM dlambda0 = 1;
    FLOAT_ACCUM dlambda1 = 0;
    if(Din != Dout && Dout != 1)
    {
        FLOAT_ACCUM scale_factor_d = CVT_FP32_2ACCUM(scale_factors[0]);
        FLOAT_ACCUM scale_factor_d_ =
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

    long hin_index0      = h;
    long hin_index1      = h;
    FLOAT_ACCUM hlambda0 = 1;
    FLOAT_ACCUM hlambda1 = 0;
    if(Hin != Hout && Hout != 1)
    {
        FLOAT_ACCUM scale_factor_h = CVT_FP32_2ACCUM(scale_factors[1]);
        FLOAT_ACCUM scale_factor_h_ =
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

    long win_index0      = w;
    long win_index1      = w;
    FLOAT_ACCUM wlambda0 = 1;
    FLOAT_ACCUM wlambda1 = 0;
    if(Win != Wout && Wout != 1)
    {
        FLOAT_ACCUM scale_factor_w = CVT_FP32_2ACCUM(scale_factors[2]);
        FLOAT_ACCUM scale_factor_w_ =
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

    output[output_tv.get_tensor_view_idx(tensor_layout)] = CVT_ACCUM2FLOAT(
        (CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(input_layout000)]) * wlambda0 +
         CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(input_layout001)]) * wlambda1) *
            hlambda0 +
        (CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(input_layout010)]) * wlambda0 +
         CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(input_layout011)]) * wlambda1) *
            hlambda1 +
        (CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(input_layout100)]) * wlambda0 +
         CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(input_layout101)]) * wlambda1) *
            dlambda0 +
        (CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(input_layout110)]) * wlambda0 +
         CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(input_layout111)]) * wlambda1) *
            dlambda1);
}

extern "C" __global__ void InterpolateTrilinearForward(const INPUT_TYPE* __restrict__ input,
                                                       OUTPUT_TYPE* __restrict__ output,
                                                       const tensor_view_t<5> input_tv,
                                                       const tensor_view_t<5> output_tv,
                                                       const size_t nelems,
                                                       const float* scale_factors,
                                                       const bool align_corners)
{
    interpolateTrilinearForward<INPUT_TYPE, OUTPUT_TYPE>(
        input, output, input_tv, output_tv, nelems, scale_factors, align_corners);
}

template <typename TI, typename TO>
__device__ inline void interpolateTrilinearBackward(TO* input_grad,
                                                    const TI* output_grad,
                                                    const tensor_view_t<5> input_grad_tv,
                                                    const tensor_view_t<5> output_grad_tv,
                                                    const size_t nelems,
                                                    const float* scale_factors,
                                                    const bool align_corners)
{
    unsigned long gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= nelems)
        return;

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

    FLOAT_ACCUM scale_factor_d = CVT_FP32_2ACCUM(scale_factors[0]);
    FLOAT_ACCUM scale_factor_d_ =
        compute_linear_scale_factor(scale_factor_d, Din, Dout, align_corners);

    FLOAT_ACCUM scale_factor_h = CVT_FP32_2ACCUM(scale_factors[1]);
    FLOAT_ACCUM scale_factor_h_ =
        compute_linear_scale_factor(scale_factor_h, Hin, Hout, align_corners);

    FLOAT_ACCUM scale_factor_w = CVT_FP32_2ACCUM(scale_factors[2]);
    FLOAT_ACCUM scale_factor_w_ =
        compute_linear_scale_factor(scale_factor_w, Win, Wout, align_corners);

    long d_from, d_to;
    if(Din == Dout)
    {
        d_from = d;
        d_to   = d + 1;
    }
    else
    {
        compute_linear_back_index_from_to(
            d, Din, Dout, scale_factor_d_, align_corners, &d_from, &d_to);
    }
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

    FLOAT_ACCUM output = 0;
    for(long i = d_from; i < d_to; i++)
    {
        FLOAT_ACCUM d_lambda = compute_back_lambda(i, d, scale_factor_d_, Din, Dout, align_corners);
        if(d_lambda == 0.f)
            continue;
        for(long j = h_from; j < h_to; j++)
        {
            FLOAT_ACCUM h_lambda =
                compute_back_lambda(j, h, scale_factor_h_, Hin, Hout, align_corners);
            if(h_lambda == 0.f)
                continue;
            for(long k = w_from; k < w_to; k++)
            {
                FLOAT_ACCUM w_lambda =
                    compute_back_lambda(k, w, scale_factor_w_, Win, Wout, align_corners);
                tensor_layout_t<5> output_layout;
                output_layout.layout[0] = n;
                output_layout.layout[1] = c;
                output_layout.layout[2] = i;
                output_layout.layout[3] = j;
                output_layout.layout[4] = k;

                output += CVT_FLOAT2ACCUM(
                              output_grad[output_grad_tv.get_tensor_view_idx(output_layout)]) *
                          d_lambda * h_lambda * w_lambda;
            }
        }
    }
    input_grad[input_grad_tv.get_tensor_view_idx(tensor_layout)] = CVT_ACCUM2FLOAT(output);
}

extern "C" __global__ void InterpolateTrilinearBackward(OUTPUT_TYPE* __restrict__ input_grad,
                                                        const INPUT_TYPE* __restrict__ output_grad,
                                                        const tensor_view_t<5> input_grad_tv,
                                                        const tensor_view_t<5> output_grad_tv,
                                                        const size_t nelems,
                                                        const float* scale_factors,
                                                        const bool align_corners)
{
    interpolateTrilinearBackward<INPUT_TYPE, OUTPUT_TYPE>(input_grad,
                                                          output_grad,
                                                          input_grad_tv,
                                                          output_grad_tv,
                                                          nelems,
                                                          scale_factors,
                                                          align_corners);
}

__device__ inline FLOAT_ACCUM
compute_scales_value(FLOAT_ACCUM scale, long input_size, long output_size)
{
    return (scale == 0.f) ? ((FLOAT_ACCUM)input_size / output_size) : (1.0f / scale);
}

__device__ inline long
nearest_idx(long output_index, long input_size, long output_size, FLOAT_ACCUM scales)
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
        FLOAT_ACCUM scale = compute_scales_value(scales, input_size, output_size);
        return min((long)(output_index * scale), input_size);
    }
}

template <typename TI, typename TO>
__device__ inline void interpolateNearestForward(const TI* input,
                                                 TO* output,
                                                 const tensor_view_t<5> input_tv,
                                                 const tensor_view_t<5> output_tv,
                                                 const size_t nelems,
                                                 const float* scale_factors)
{
    unsigned long gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= nelems)
        return;

    auto tensor_layout = tensor_layout_t<5>(output_tv, gid);
    long n             = tensor_layout.layout[0];
    long c             = tensor_layout.layout[1];
    long d             = tensor_layout.layout[2];
    long h             = tensor_layout.layout[3];
    long w             = tensor_layout.layout[4];

    long Dout = output_tv.size[2];
    long Hout = output_tv.size[3];
    long Wout = output_tv.size[4];
    long Din  = input_tv.size[2];
    long Hin  = input_tv.size[3];
    long Win  = input_tv.size[4];

    long x = nearest_idx(d, Din, Dout, scale_factor_d);
    long y = nearest_idx(h, Hin, Hout, scale_factor_h);
    long z = nearest_idx(w, Win, Wout, scale_factor_w);

    tensor_layout_t<5> input_layout;
    input_layout.layout[0] = n;
    input_layout.layout[1] = c;
    input_layout.layout[2] = x;
    input_layout.layout[3] = y;
    input_layout.layout[4] = z;

    output[output_tv.get_tensor_view_idx(tensor_layout)] =
        input[input_tv.get_tensor_view_idx(input_layout)];
}

extern "C" __global__ void InterpolateNearestForward(const INPUT_TYPE* __restrict__ input,
                                                     OUTPUT_TYPE* __restrict__ output,
                                                     const tensor_view_t<5> input_tv,
                                                     const tensor_view_t<5> output_tv,
                                                     const size_t nelems,
                                                     const float* scale_factors)
{
    interpolateNearestForward<INPUT_TYPE, OUTPUT_TYPE>(
        input, output, input_tv, output_tv, nelems, scale_factors);
}

__device__ inline long
nearest_idx_back(long input_index, long input_size, long output_size, FLOAT_ACCUM scales)
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
        FLOAT_ACCUM scale = compute_scales_value(scales, input_size, output_size);
        return min((long)ceil(input_index / scale), output_size);
    }
}

template <typename TI, typename TO>
__device__ inline void interpolateNearestBackward(TO* input_grad,
                                                  const TI* output_grad,
                                                  const tensor_view_t<5> input_tv,
                                                  const tensor_view_t<5> output_tv,
                                                  const size_t nelems,
                                                  const float* scale_factors)
{
    unsigned long gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= nelems)
        return;

    auto tensor_layout = tensor_layout_t<5>(input_grad_tv, gid);
    long n             = tensor_layout.layout[0];
    long c             = tensor_layout.layout[1];
    long x             = tensor_layout.layout[2];
    long y             = tensor_layout.layout[3];
    long z             = tensor_layout.layout[4];

    long Dout = output_grad_tv.size[2];
    long Hout = output_grad_tv.size[3];
    long Wout = output_grad_tv.size[4];
    long Din  = input_grad_tv.size[2];
    long Hin  = input_grad_tv.size[3];
    long Win  = input_grad_tv.size[4];

    long dstart = nearest_idx_back(x, Din, Dout, scale_factor_d);
    long dlimit = nearest_idx_back(x + 1, Din, Dout, scale_factor_d);
    long hstart = nearest_idx_back(y, Hin, Hout, scale_factor_h);
    long hlimit = nearest_idx_back(y + 1, Hin, Hout, scale_factor_h);
    long wstart = nearest_idx_back(z, Win, Wout, scale_factor_w);
    long wlimit = nearest_idx_back(z + 1, Win, Wout, scale_factor_w);

    FLOAT_ACCUM grad = 0.f;
    for(long d = dstart; d < dlimit; d++)
    {
        for(long h = hstart; h < hlimit; h++)
        {
            for(long w = wstart; w < wlimit; w++)
            {
                tensor_layout_t<5> output_grad_layout;
                output_grad_layout.layout[0] = n;
                output_grad_layout.layout[1] = c;
                output_grad_layout.layout[2] = d;
                output_grad_layout.layout[3] = h;
                output_grad_layout.layout[4] = w;

                grad += CVT_FLOAT2ACCUM(
                    output_grad[output_grad_tv.get_tensor_view_idx(output_grad_layout)]);
            }
        }
    }
    input_grad[input_grad_tv.get_tensor_view_idx(tensor_layout)] = CVT_ACCUM2FLOAT(grad);
}

extern "C" __global__ void InterpolateNearestBackward(const OUTPUT_TYPE* __restrict__ input_grad,
                                                      const INPUT_TYPE* __restrict__ output_grad,
                                                      const tensor_view_t<5> input_grad_tv,
                                                      const tensor_view_t<5> output_grad_tv,
                                                      const size_t nelems,
                                                      const float* scale_factors)
{
    interpolateNearestBackward<INPUT_TYPE, OUTPUT_TYPE>(
        input_grad, output_grad, input_grad_tv, output_grad_tv, nelems, scale_factors);
}

__device__ inline FLOAT_ACCUM cubic_convolution1(FLOAT_ACCUM x, FLOAT_ACCUM A)
{
    return ((A + 2) * x - (A + 3)) * x * x + 1;
}

__device__ inline FLOAT_ACCUM cubic_convolution2(FLOAT_ACCUM x, FLOAT_ACCUM A)
{
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

__device__ inline void get_cubic_upsampling_coefficients(FLOAT_ACCUM coeffs[4], FLOAT_ACCUM t)
{
    FLOAT_ACCUM A = -0.75f;

    FLOAT_ACCUM x1 = t;
    coeffs[0]      = cubic_convolution2(x1 + 1.0f, A);
    coeffs[1]      = cubic_convolution1(x1, A);

    FLOAT_ACCUM x2 = 1.0f - t;
    coeffs[2]      = cubic_convolution1(x2, A);
    coeffs[3]      = cubic_convolution2(x2 + 1.0f, A);
}

__device__ inline FLOAT_ACCUM
cubic_interp1d(FLOAT_ACCUM x0, FLOAT_ACCUM x1, FLOAT_ACCUM x2, FLOAT_ACCUM x3, FLOAT_ACCUM t)
{
    FLOAT_ACCUM coeffs[4];
    get_cubic_upsampling_coefficients(coeffs, t);

    return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

__device__ inline long bound(long p, long max_size) { return max(min(p, max_size - 1), 0l); }

template <typename TI, typename TO>
__device__ inline void interpolateBicubicForward(const TI* input,
                                                 TO* output,
                                                 const tensor_view_t<4> input_tv,
                                                 const tensor_view_t<4> output_tv,
                                                 const size_t nelems,
                                                 const float* scale_factors,
                                                 const bool align_corners)
{
    unsigned long gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= nelems)
        return;

    auto tensor_layout = tensor_layout_t<4>(output_tv, gid);
    long n             = tensor_layout.layout[0];
    long c             = tensor_layout.layout[1];
    long h             = tensor_layout.layout[2];
    long w             = tensor_layout.layout[3];

    long Hin  = input_tv.size[2];
    long Win  = input_tv.size[3];
    long Hout = output_tv.size[2];
    long Wout = output_tv.size[3];
    if(Hin == Hout && Win == Wout)
    {
        output[output_tv.get_tensor_view_idx(tensor_layout)] =
            input[input_tv.get_tensor_view_idx(tensor_layout)];
        return;
    }

    FLOAT_ACCUM scale_factor_h = CVT_FP32_2ACCUM(scale_factors[0]);
    FLOAT_ACCUM scale_factor_h_ =
        compute_linear_scale_factor(scale_factor_h, Hin, Hout, align_corners);
    FLOAT_ACCUM real_y = bicubic_idx(h, Hout, scale_factor_h_, align_corners);
    long in_y          = (long)floor(real_y);
    FLOAT_ACCUM t_y    = real_y - in_y;

    FLOAT_ACCUM scale_factor_w = CVT_FP32_2ACCUM(scale_factors[1]);
    FLOAT_ACCUM scale_factor_w_ =
        compute_linear_scale_factor(scale_factor_w, Win, Wout, align_corners);
    FLOAT_ACCUM real_x = bicubic_idx(w, Wout, scale_factor_w_, align_corners);
    long in_x          = (long)floor(real_x);
    FLOAT_ACCUM t_x    = real_x - in_x;

    FLOAT_ACCUM coefficients[4];
#pragma unroll
    for(int k = 0; k < 4; k++)
    {
        long y = bound(in_y - 1 + k, Hin);
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

        coefficients[k] =
            cubic_interp1d(CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(input_layout0)]),
                           CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(input_layout1)]),
                           CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(input_layout2)]),
                           CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(input_layout3)]),
                           t_x);
    }

    output[output_tv.get_tensor_view_idx(tensor_layout)] = CVT_ACCUM2FLOAT(
        cubic_interp1d(coefficients[0], coefficients[1], coefficients[2], coefficients[3], t_y));
}

extern "C" __global__ void InterpolateBicubicForward(const INPUT_TYPE* __restrict__ input,
                                                     OUTPUT_TYPE* __restrict__ output,
                                                     const tensor_view_t<4> input_tv,
                                                     const tensor_view_t<4> output_tv,
                                                     const size_t nelems,
                                                     const float* scale_factors,
                                                     const bool align_corners)
{
    interpolateBicubicForward<INPUT_TYPE, OUTPUT_TYPE>(
        input, output, input_tv, output_tv, nelems, scale_factors, align_corners);
}

template <typename TI, typename TO>
__device__ inline void interpolateBicubicBackward(TO* input_grad,
                                                  const TI* output_grad,
                                                  const tensor_view_t<4> input_grad_tv,
                                                  const tensor_view_t<4> output_grad_tv,
                                                  const size_t nelems,
                                                  const float* scale_factors,
                                                  const bool align_corners)
{
    unsigned long gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= nelems)
        return;

    //     auto tensor_layout = tensor_layout_t<4>(output_grad_tv, gid);
    //     long n             = tensor_layout.layout[0];
    //     long c             = tensor_layout.layout[1];
    //     long h             = tensor_layout.layout[2];
    //     long w             = tensor_layout.layout[3];

    //     long Hin  = input_grad_tv.size[2];
    //     long Hout = output_grad_tv.size[2];
    //     long Win  = input_grad_tv.size[3];
    //     long Wout = output_grad_tv.size[3];

    //     if(Hin == Hout && Win == Wout)
    //     {
    //         input_grad[input_grad_tv.get_tensor_view_idx(tensor_layout)] =
    //             output_grad[output_grad_tv.get_tensor_view_idx(tensor_layout)];
    //         return;
    //     }

    //     FLOAT_ACCUM scale_factor_h = CVT_FP32_2ACCUM(scale_factors[0]);
    //     FLOAT_ACCUM scale_factor_h_ =
    //         compute_linear_scale_factor(scale_factor_h, Hin, Hout, align_corners);
    //     FLOAT_ACCUM real_y = bicubic_idx(h, Hout, scale_factor_h_, align_corners);
    //     long in_y          = (long)floor(real_y);
    //     FLOAT_ACCUM t_y    = real_y - in_y;

    //     FLOAT_ACCUM scale_factor_w = CVT_FP32_2ACCUM(scale_factors[1]);
    //     FLOAT_ACCUM scale_factor_w_ =
    //         compute_linear_scale_factor(scale_factor_w, Win, Wout, align_corners);
    //     FLOAT_ACCUM real_x = bicubic_idx(w, Wout, scale_factor_w_, align_corners);
    //     long in_x          = (long)floor(real_x);
    //     FLOAT_ACCUM t_x    = real_x - in_x;

    //     FLOAT_ACCUM y_coeffs[4];
    //     FLOAT_ACCUM x_coeffs[4];
    //     get_cubic_upsampling_coefficients(y_coeffs, t_y);
    //     get_cubic_upsampling_coefficients(x_coeffs, t_x);
    //     FLOAT_ACCUM out_value =
    //         CVT_FLOAT2ACCUM(output_grad[output_grad_tv.get_tensor_view_idx(tensor_layout)]);
    // #pragma unroll
    //     for(int i = 0; i < 4; i++)
    //     {
    //         long input_h = bound(in_y - 1 + i, Hin);
    // #pragma unroll
    //         for(int j = 0; j < 4; j++)
    //         {
    //             long input_w = bound(in_x - 1 + j, Win);
    //             atomic_add_g(input_grad + TV4D_IDX(input_grad_tv, n, c, input_h, input_w),
    //                          out_value * y_coeffs[i] * x_coeffs[j]);
    //         }
    //     }
}

extern "C" __global__ void InterpolateBicubicBackward(OUTPUT_TYPE* __restrict__ input_grad,
                                                      const INPUT_TYPE* __restrict__ output_grad,
                                                      const tensor_view_t<4> input_grad_tv,
                                                      const tensor_view_t<4> output_grad_tv,
                                                      const size_t nelems,
                                                      const float* scale_factors,
                                                      const bool align_corners)
{
    interpolateBicubicBackward<INPUT_TYPE, OUTPUT_TYPE>(input_grad,
                                                        output_grad,
                                                        input_grad_tv,
                                                        output_grad_tv,
                                                        nelems,
                                                        scale_factors,
                                                        align_corners);
}
