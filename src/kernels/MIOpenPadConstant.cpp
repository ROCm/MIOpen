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
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#endif

#include "float_types.h"
#include "tensor_view.hpp"

#include "MIOpenPadConstant.hpp"

template <typename DTYPE>
__device__ void padconstant_fwd(const DTYPE* __restrict__ input,
                                DTYPE* __restrict__ output,
                                const padding_5d_t padding,
                                const size_t output_size,
                                FLOAT_ACCUM value,
                                tensor_view_t<5> input_tv,
                                tensor_view_t<5> output_tv)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= output_size)
        return;

    auto o_tensor_layout = tensor_layout_t<5>(output_tv, gid);
    auto i_tensor_layout = tensor_layout_t<5>({0, 0, 0, 0, 0});

    bool flag = true;

    for(uint64_t i = 0; i < 5; ++i)
    {
        int64_t idx = o_tensor_layout.layout[i] - padding.val[2 * i];
        if(idx < 0 || idx >= input_tv.size[i])
        {
            flag = false;
            break;
        }

        i_tensor_layout.layout[i] = idx;
    }

    // DTYPE val;
    if(flag)
    {
        output[output_tv.get_tensor_view_idx(o_tensor_layout)] =
            input[input_tv.get_tensor_view_idx(i_tensor_layout)];
    }
    else
    {
        output[output_tv.get_tensor_view_idx(o_tensor_layout)] = CVT_ACCUM2FLOAT(value);
    }

    // output[output_tv.get_tensor_view_idx(o_tensor_layout)] = val;
}

template <typename DTYPE>
__device__ void padconstant_bwd(DTYPE* __restrict__ input_grad,
                                const DTYPE* __restrict__ output_grad,
                                const padding_5d_t padding,
                                const uint64_t input_grad_size,
                                tensor_view_t<5> input_grad_tv,
                                tensor_view_t<5> output_grad_tv)

{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= input_grad_size)
        return;

    auto ig_tensor_layout = tensor_layout_t<5>(input_grad_tv, gid);
    auto og_tensor_layout = tensor_layout_t<5>({0, 0, 0, 0, 0});

    bool flag = true;

    for(uint64_t i = 0; i < 5; ++i)
    {
        int64_t idx = ig_tensor_layout.layout[i] + padding.val[2 * i];
        if(idx < 0 || idx >= output_grad_tv.size[i])
        {
            flag = false;
            break;
        }

        og_tensor_layout.layout[i] = idx;
    }

    // DTYPE val = flag ? output_grad[output_grad_tv.get_tensor_view_idx(og_tensor_layout)]
    //                  : static_cast<DTYPE>(0);
    // input_grad[input_grad_tv.get_tensor_view_idx(ig_tensor_layout)] = val;
    input_grad[input_grad_tv.get_tensor_view_idx(ig_tensor_layout)] =
        flag ? output_grad[output_grad_tv.get_tensor_view_idx(og_tensor_layout)]
             : static_cast<DTYPE>(0);
}

extern "C" __global__ void PadConstantFwd(const IO_TYPE* __restrict__ input,
                                          IO_TYPE* __restrict__ output,
                                          const padding_5d_t padding,
                                          const size_t output_size,
                                          FLOAT_ACCUM value,
                                          tensor_view_t<5> input_tv,
                                          tensor_view_t<5> output_tv)
{
    // padconstantfwd<INPUT_TYPE, OUTPUT_TYPE>(x, y, x_tv, y_tv, padding, output_size, value);
    padconstant_fwd<IO_TYPE>(input, output, padding, output_size, value, input_tv, output_tv);
}

extern "C" __global__ void PadConstantBwd(IO_TYPE* __restrict__ input_grad,
                                          const IO_TYPE* __restrict__ output_grad,
                                          // const tensor_view_5d_t dx_tv,
                                          // const tensor_view_5d_t y_grad_tv,
                                          const padding_5d_t padding,
                                          const uint64_t input_grad_size,
                                          tensor_view_t<5> input_grad_tv,
                                          tensor_view_t<5> output_grad_tv)
{
    padconstant_bwd<IO_TYPE>(
        input_grad, output_grad, padding, input_grad_size, input_grad_tv, output_grad_tv);
}
