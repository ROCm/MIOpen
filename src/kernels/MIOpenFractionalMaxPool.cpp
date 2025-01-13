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
#include "hip_atomic.hpp"
#include "tensor_view.hpp"

__device__ inline int64_t get_interval(
    FLOAT_ACCUM sample, int64_t index, int64_t input_size, int64_t output_size, int64_t pool_size)
{
    if(index == output_size - 1)
    {
        return input_size - pool_size;
    }
    else
    {
        FLOAT_ACCUM alpha = static_cast<FLOAT_ACCUM>(input_size - pool_size) /
                            static_cast<FLOAT_ACCUM>(output_size - 1);
        return static_cast<int64_t>((index + sample) * alpha) -
               static_cast<int64_t>(sample * alpha);
    }
}

template <typename T, typename Ti>
__device__ void fractionalMaxPool2dForward(const T* input,
                                           T* output,
                                           Ti* indices,
                                           T* random_sample,
                                           int64_t KH,
                                           int64_t KW,
                                           tensor_view_t<4> input_tv,
                                           tensor_view_t<4> output_tv,
                                           tensor_view_t<4> indices_tv,
                                           tensor_view_t<3> random_sample_tv)
{
    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    tensor_layout_t<4> layout(output_tv, gid);
    int64_t n, c, oh, ow;
    n  = layout.layout[0];
    c  = layout.layout[1];
    oh = layout.layout[2];
    ow = layout.layout[3];

    if(n >= output_tv.size[0])
        return;

    int64_t pool_h = get_interval(
        CVT_FLOAT2ACCUM(random_sample[random_sample_tv.get_tensor_view_idx({n, c, 0})]),
        oh,
        input_tv.size[2],
        output_tv.size[2],
        KH);
    int64_t pool_w = get_interval(
        CVT_FLOAT2ACCUM(random_sample[random_sample_tv.get_tensor_view_idx({n, c, 1})]),
        ow,
        input_tv.size[3],
        output_tv.size[3],
        KW);

    FLOAT_ACCUM m = log(0);

    int64_t h_end = pool_h + KH;
    int64_t w_end = pool_w + KW;

    if(!indices)
    {
        for(int64_t h = pool_h; h < h_end; ++h)
        {
            for(int64_t w = pool_w; w < w_end; ++w)
            {
                FLOAT_ACCUM val =
                    CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({n, c, h, w})]);
                if(val > m || isnan(val))
                {
                    m = val;
                }
            }
        }
        output[output_tv.get_tensor_view_idx(layout)] = CVT_ACCUM2FLOAT(m);
    }
    else
    {
        int64_t mi = pool_h * input_tv.size[3] + pool_w;
        for(int64_t h = pool_h; h < h_end; ++h)
        {
            for(int64_t w = pool_w; w < w_end; ++w)
            {
                FLOAT_ACCUM val =
                    CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({n, c, h, w})]);
                if(val > m || isnan(val))
                {
                    m  = val;
                    mi = h * input_tv.size[3] + w;
                }
            }
        }

        output[output_tv.get_tensor_view_idx(layout)]   = CVT_ACCUM2FLOAT(m);
        indices[indices_tv.get_tensor_view_idx(layout)] = static_cast<Ti>(mi);
    }
}

extern "C" __global__ void FractionalMaxPool2dForward(const D_TYPE* input,
                                                      D_TYPE* output,
                                                      I_TYPE* indices,
                                                      D_TYPE* random_sample,
                                                      int64_t KH,
                                                      int64_t KW,
                                                      tensor_view_t<4> input_tv,
                                                      tensor_view_t<4> output_tv,
                                                      tensor_view_t<4> indices_tv,
                                                      tensor_view_t<3> random_sample_tv)
{
    fractionalMaxPool2dForward<D_TYPE, I_TYPE>(input,
                                               output,
                                               indices,
                                               random_sample,
                                               KH,
                                               KW,
                                               input_tv,
                                               output_tv,
                                               indices_tv,
                                               random_sample_tv);
}

template <typename T, typename Ti>
__device__ void fractionalMaxPool2dBackward(const Ti* indices,
                                            const T* output_grad,
                                            T* input_grad,
                                            uint64_t numel,
                                            tensor_view_t<4> indices_tv,
                                            tensor_view_t<4> output_grad_tv,
                                            tensor_view_t<4> input_grad_tv)
{
    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if(gid >= numel)
        return;

    tensor_layout_t<4> layout(output_grad_tv, gid);
    uint64_t index = static_cast<uint64_t>(indices[indices_tv.get_tensor_view_idx(layout)]);
    uint64_t h     = index / input_grad_tv.size[3];
    uint64_t w     = index % input_grad_tv.size[3];

    atomic_add_g(
        &input_grad[input_grad_tv.get_tensor_view_idx({layout.layout[0], layout.layout[1], h, w})],
        CVT_FLOAT2ACCUM(output_grad[output_grad_tv.get_tensor_view_idx(layout)]));
}

extern "C" __global__ void FractionalMaxPool2dBackward(const I_TYPE* indices,
                                                       const D_TYPE* output_grad,
                                                       D_TYPE* input_grad,
                                                       uint64_t numel,
                                                       tensor_view_t<4> indices_tv,
                                                       tensor_view_t<4> output_grad_tv,
                                                       tensor_view_t<4> input_grad_tv)
{
    fractionalMaxPool2dBackward<D_TYPE, I_TYPE>(
        indices, output_grad, input_grad, numel, indices_tv, output_grad_tv, input_grad_tv);
}

template <typename T, typename Ti>
__device__ void fractionalMaxPool3dForward(const T* input,
                                           T* output,
                                           Ti* indices,
                                           T* random_sample,
                                           int64_t KD,
                                           int64_t KH,
                                           int64_t KW,
                                           tensor_view_t<5> input_tv,
                                           tensor_view_t<5> output_tv,
                                           tensor_view_t<5> indices_tv,
                                           tensor_view_t<3> random_sample_tv)
{
    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;

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
        CVT_FLOAT2ACCUM(random_sample[random_sample_tv.get_tensor_view_idx({n, c, 0})]),
        od,
        input_tv.size[2],
        output_tv.size[2],
        KD);
    int64_t pool_h = get_interval(
        CVT_FLOAT2ACCUM(random_sample[random_sample_tv.get_tensor_view_idx({n, c, 1})]),
        oh,
        input_tv.size[3],
        output_tv.size[3],
        KH);
    int64_t pool_w = get_interval(
        CVT_FLOAT2ACCUM(random_sample[random_sample_tv.get_tensor_view_idx({n, c, 2})]),
        ow,
        input_tv.size[4],
        output_tv.size[4],
        KW);

    FLOAT_ACCUM m = log(0);

    int64_t d_end = pool_d + KD;
    int64_t h_end = pool_h + KH;
    int64_t w_end = pool_w + KW;

    if(!indices)
    {
        for(int64_t d = pool_d; d < d_end; ++d)
        {
            for(int64_t h = pool_h; h < h_end; ++h)
            {
                for(int64_t w = pool_w; w < w_end; ++w)
                {
                    FLOAT_ACCUM val =
                        CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({n, c, d, h, w})]);
                    if(val > m || isnan(val))
                    {
                        m = val;
                    }
                }
            }
        }
        output[output_tv.get_tensor_view_idx(layout)] = CVT_ACCUM2FLOAT(m);
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
                    FLOAT_ACCUM val =
                        CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx({n, c, d, h, w})]);
                    if(val > m || isnan(val))
                    {
                        m  = val;
                        mi = d * input_tv.size[3] * input_tv.size[4] + h * input_tv.size[4] + w;
                    }
                }
            }
        }
        output[output_tv.get_tensor_view_idx(layout)]   = CVT_ACCUM2FLOAT(m);
        indices[indices_tv.get_tensor_view_idx(layout)] = static_cast<Ti>(mi);
    }
}

extern "C" __global__ void FractionalMaxPool3dForward(const D_TYPE* input,
                                                      D_TYPE* output,
                                                      I_TYPE* indices,
                                                      D_TYPE* random_sample,
                                                      int64_t KD,
                                                      int64_t KH,
                                                      int64_t KW,
                                                      tensor_view_t<5> input_tv,
                                                      tensor_view_t<5> output_tv,
                                                      tensor_view_t<5> indices_tv,
                                                      tensor_view_t<3> random_sample_tv)
{
    fractionalMaxPool3dForward<D_TYPE, I_TYPE>(input,
                                               output,
                                               indices,
                                               random_sample,
                                               KD,
                                               KH,
                                               KW,
                                               input_tv,
                                               output_tv,
                                               indices_tv,
                                               random_sample_tv);
}

template <typename T, typename Ti>
__device__ void fractionalMaxPool3dBackward(const Ti* indices,
                                            const T* output_grad,
                                            T* input_grad,
                                            uint64_t numel,
                                            tensor_view_t<5> indices_tv,
                                            tensor_view_t<5> output_grad_tv,
                                            tensor_view_t<5> input_grad_tv)
{
    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if(gid >= numel)
        return;

    tensor_layout_t<5> layout(output_grad_tv, gid);
    uint64_t index = static_cast<uint64_t>(indices[indices_tv.get_tensor_view_idx(layout)]);
    uint64_t d     = index / (input_grad_tv.size[4] * input_grad_tv.size[3]);
    uint64_t h     = (index / input_grad_tv.size[4]) % input_grad_tv.size[3];
    uint64_t w     = index % input_grad_tv.size[4];

    atomic_add_g(&input_grad[input_grad_tv.get_tensor_view_idx(
                     {layout.layout[0], layout.layout[1], d, h, w})],
                 CVT_FLOAT2ACCUM(output_grad[output_grad_tv.get_tensor_view_idx(layout)]));
}

extern "C" __global__ void FractionalMaxPool3dBackward(const I_TYPE* indices,
                                                       const D_TYPE* output_grad,
                                                       D_TYPE* input_grad,
                                                       uint64_t numel,
                                                       tensor_view_t<5> indices_tv,
                                                       tensor_view_t<5> output_grad_tv,
                                                       tensor_view_t<5> input_grad_tv)
{
    fractionalMaxPool3dBackward<D_TYPE, I_TYPE>(
        indices, output_grad, input_grad, numel, indices_tv, output_grad_tv, input_grad_tv);
}