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

template <typename T, typename Ti>
__device__ void fractionalMaxPool2dForward(const T* input,
                                           T* output,
                                           Ti* indices,
                                           T* random_sample,
                                           int64_t KD,
                                           int64_t KH,
                                           tensor_view_t<4> input_tv,
                                           tensor_view_t<4> output_tv,
                                           tensor_view_t<4> indices_tv,
                                           tensor_view_t<3> random_sample_tv)
{
    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    int n, c, oh, ow;
    GET_NCHW(n, c, oh, ow, gid, output);
    if(n >= output_tv.size[0])
        return;

    int pool_w = get_interval<DTYPE>(
        TV_3D_AT(random_sample, n, c, 0), ow, input_tv.size[3], output_tv.size[3], kw);
    int pool_h = get_interval<DTYPE>(
        TV_3D_AT(random_sample, n, c, 1), oh, input_tv.size[2], output_tv.size[2], kh);

    DTYPE m = DTYPE_LOWEST;

    int h_end = pool_h + kh;
    int w_end = pool_w + kw;

    if(!indices)
    {
        for(int h = pool_h; h < h_end; ++h)
        {
            for(int w = pool_w; w < w_end; ++w)
            {
                DTYPE val = TV_4D_AT(input, n, c, h, w);
                if(val > m || isnan(val))
                {
                    m = val;
                }
            }
        }
        SET_4D_VAL(output, gid, m);
    }
    else
    {
        long mi = pool_h * input_tv.size[3] + pool_w;

        for(int h = pool_h; h < h_end; ++h)
        {
            for(int w = pool_w; w < w_end; ++w)
            {
                DTYPE val = TV_4D_AT(input, n, c, h, w);
                if(val > m || isnan(val))
                {
                    m  = val;
                    mi = h * input_tv.size[3] + w;
                }
            }
        }
        SET_4D_VAL(output, gid, m);
        SET_4D_VAL(indices, gid, mi);
    }
}

extern "C" __global__ void FractionalMaxPool2dForward(const D_TYPE* input,
                                                      D_TYPE* output,
                                                      I_TYPE* indices,
                                                      D_TYPE* random_sample,
                                                      int64_t KD,
                                                      int64_t KH,
                                                      tensor_view_t<4> input_tv,
                                                      tensor_view_t<4> output_tv,
                                                      tensor_view_t<4> indices_tv,
                                                      tensor_view_t<3> random_sample_tv)
{
    fractionalMaxPool2dForward<D_TYPE, I_TYPE>(input,
                                               output,
                                               indices,
                                               random_sample,
                                               KD,
                                               KH,
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
    uint64_t index = indices[indices_tv.get_tensor_view_idx(layout)];
    uint64_t h     = index / input_grad_tv.size[3];
    uint64_t w     = index % input_grad_tv.size[3];

    atomic_add_g(
        &input_grad[input_grad_tv.get_tensor_view_idx({layout.layout[0], layout.layout[1], h, w})],
        output_grad[output_grad_tv.get_tensor_view_idx(layout)]);
}

extern "C" __global__ void FractionalMaxPool2dBackward(const I_TYPE* indices,
                                                       const D_TYPE* output_grad,
                                                       D_TYPE* input_grad,
                                                       tensor_view_t<4> indices_tv,
                                                       tensor_view_t<4> output_grad_tv,
                                                       tensor_view_t<4> input_grad_tv)
{
    fractionalMaxPool2dBackward<D_TYPE, I_TYPE>(
        indices, output_grad, input_grad, indices_tv, output_grad_tv, input_grad_tv);
}
