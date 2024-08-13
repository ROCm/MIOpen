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

template <typename TI, typename TO, uint32_t NDIMS, bool SingleWeight>
__device__ void PReLUBackward(const TI* __restrict__ input,
                              const TI* __restrict__ weight,
                              const TO* __restrict__ output_grad,
                              TI* __restrict__ input_grad,
                              FLOAT_ACCUM* __restrict__ weight_grad_collector,
                              uint64_t N,
                              tensor_view_t<NDIMS> input_tv,
                              tensor_view_t<1> weight_tv,
                              tensor_view_t<NDIMS> output_grad_tv,
                              tensor_view_t<NDIMS> input_grad_tv)
{
    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= N)
        return;

    auto tensor_layout  = tensor_layout_t<NDIMS>(input_tv, gid);
    FLOAT_ACCUM input_v = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(tensor_layout)]);
    FLOAT_ACCUM grad_v =
        CVT_FLOAT2ACCUM(output_grad[output_grad_tv.get_tensor_view_idx(tensor_layout)]);

    if(input_grad != nullptr)
    {
        FLOAT_ACCUM weight_v = CVT_FLOAT2ACCUM(
            weight[SingleWeight ? 0 : weight_tv.get_tensor_view_idx({tensor_layout.layout[1]})]);
        FLOAT_ACCUM input_grad_v = input_v > 0 ? grad_v : weight_v * grad_v;
        input_grad[input_grad_tv.get_tensor_view_idx(tensor_layout)] =
            CVT_ACCUM2FLOAT(input_grad_v);
    }
    if(weight_grad_collector != nullptr)
    {
        weight_grad_collector[gid] = input_v > 0 ? 0 : input_v * grad_v;
    }
}

extern "C" __global__ void PReLUSWBackward(const INPUT_TYPE* __restrict__ input,
                                           const INPUT_TYPE* __restrict__ weight,
                                           const OUTPUT_TYPE* __restrict__ output_grad,
                                           INPUT_TYPE* __restrict__ input_grad,
                                           FLOAT_ACCUM* __restrict__ weight_grad_collector,
                                           uint64_t N,
                                           tensor_view_t<VIEW_DIMS> input_tv,
                                           tensor_view_t<1> weight_tv,
                                           tensor_view_t<VIEW_DIMS> output_grad_tv,
                                           tensor_view_t<VIEW_DIMS> input_grad_tv)
{
    // instantiate the kernel
    PReLUBackward<INPUT_TYPE, OUTPUT_TYPE, VIEW_DIMS, true>(input,
                                                            weight,
                                                            output_grad,
                                                            input_grad,
                                                            weight_grad_collector,
                                                            N,
                                                            input_tv,
                                                            weight_tv,
                                                            output_grad_tv,
                                                            input_grad_tv);
}

extern "C" __global__ void PReLUMWBackward(const INPUT_TYPE* __restrict__ input,
                                           const INPUT_TYPE* __restrict__ weight,
                                           const OUTPUT_TYPE* __restrict__ output_grad,
                                           INPUT_TYPE* __restrict__ input_grad,
                                           FLOAT_ACCUM* __restrict__ weight_grad_collector,
                                           uint64_t N,
                                           tensor_view_t<VIEW_DIMS> input_tv,
                                           tensor_view_t<1> weight_tv,
                                           tensor_view_t<VIEW_DIMS> output_grad_tv,
                                           tensor_view_t<VIEW_DIMS> input_grad_tv)
{
    // instantiate the kernel
    PReLUBackward<INPUT_TYPE, OUTPUT_TYPE, VIEW_DIMS, false>(input,
                                                             weight,
                                                             output_grad,
                                                             input_grad,
                                                             weight_grad_collector,
                                                             N,
                                                             input_tv,
                                                             weight_tv,
                                                             output_grad_tv,
                                                             input_grad_tv);
}
