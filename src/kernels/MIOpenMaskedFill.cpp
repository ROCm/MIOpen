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

using int8_t = __hip_internal::int8_t;

template <typename TIO>
__device__ void MaskedFillForwardImpl(const TIO* input,
                                      const int8_t* mask,
                                      TIO* output,
                                      tensor_view_t<5> input_tv,
                                      tensor_view_t<5> mask_tv,
                                      tensor_view_t<5> output_tv,
                                      float value,
                                      uint64_t numel)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= numel)
        return;

    tensor_layout_t<5> output_layout{output_tv, gid};
    tensor_layout_t<5> input_layout{input_tv, gid};
    tensor_layout_t<5> mask_layout{mask_tv, gid};

    output[output_tv.get_tensor_view_idx(output_layout)] =
        mask[mask_tv.get_tensor_view_idx(mask_layout)]
            ? value
            : input[input_tv.get_tensor_view_idx(input_layout)];
}

extern "C" __global__ void MaskedFillForward(const IO_TYPE* input,
                                             const int8_t* mask,
                                             IO_TYPE* output,
                                             tensor_view_t<5> input_tv,
                                             tensor_view_t<5> mask_tv,
                                             tensor_view_t<5> output_tv,
                                             float value,
                                             uint64_t numel)
{
    MaskedFillForwardImpl<IO_TYPE>(input, mask, output, input_tv, mask_tv, output_tv, value, numel);
}

template <typename TIO>
__device__ void MaskedFillBackwardImpl(const TIO* output_grad,
                                       const int8_t* mask,
                                       TIO* input_grad,
                                       tensor_view_t<5> output_grad_tv,
                                       tensor_view_t<5> mask_tv,
                                       tensor_view_t<5> input_grad_tv,
                                       uint64_t numel)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= numel)
        return;

    tensor_layout_t<5> input_grad_layout{input_grad_tv, gid};
    tensor_layout_t<5> output_grad_layout{output_grad_tv, gid};
    tensor_layout_t<5> mask_layout{mask_tv, gid};
    input_grad[input_grad_tv.get_tensor_view_idx(input_grad_layout)] =
        mask[mask_tv.get_tensor_view_idx(mask_layout)]
            ? static_cast<TIO>(0)
            : output_grad[output_grad_tv.get_tensor_view_idx(output_grad_layout)];
}
extern "C" __global__ void MaskedFillBackward(const IO_TYPE* output_grad,
                                              const int8_t* mask,
                                              IO_TYPE* input_grad,
                                              tensor_view_t<5> output_grad_tv,
                                              tensor_view_t<5> mask_tv,
                                              tensor_view_t<5> input_grad_tv,
                                              uint64_t numel)
{
    MaskedFillBackwardImpl<IO_TYPE>(
        output_grad, mask, input_grad, output_grad_tv, mask_tv, input_grad_tv, numel);
}
