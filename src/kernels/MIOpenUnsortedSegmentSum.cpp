/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <cstdint>
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "hip_atomic.hpp"
#include "float_types.h"
#include "tensor_view.hpp"

template <typename TIO>
__device__ void unsortedsegmentsumFwd(const TIO* __restrict__ input,
                                      TIO* __restrict__ output,
                                      const uint64_t* __restrict__ segment_ids,
                                      uint64_t num_segments,
                                      tensor_view_t<4> input_tv,
                                      tensor_view_t<4> output_tv,
                                      tensor_view_t<4> segment_ids_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= N)
        return;

    uint64_t input_segment_index = gid / inner_dim_size;
    uint64_t segment_offset      = gid % inner_dim_size;

    uint64_t output_segment_index = R_GET_VAL(segment_ids, input_segment_index);

    if(output_segment_index < 0 || output_segment_index >= output_outer_dim_size)
    {
        return;
    }

    uint64_t output_index = output_segment_index * inner_dim_size + segment_offset + output_off;
    atomic_add_g(output + output_index, GET_VAL(input, gid));
}

extern "C" __global__ void UnsortedSegmentSumFwd(const D_TYPE* __restrict__ input,
                                                 D_TYPE* __restrict__ output,
                                                 const uint64_t* __restrict__ segment_ids,
                                                 uint64_t num_segments,
                                                 tensor_view_t<4> input_tv,
                                                 tensor_view_t<4> output_tv,
                                                 tensor_view_t<4> segment_ids_tv)
{
    unsortedsegmentsumFwd<D_TYPE>(
        input, output, segment_ids, num_segments, input_tv, output_tv, segment_ids_tv);
}

template <typename TIO>
__device__ void unsortedsegmentsumBwd(const TIO* __restrict__ output_grad,
                                      TIO* __restrict__ input_grad,
                                      const uint64_t* __restrict__ segment_ids,
                                      uint64_t num_segments,
                                      tensor_view_t<4> output_grad_tv,
                                      tensor_view_t<4> input_grad_tv,
                                      tensor_view_t<4> segment_ids_tv)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= N)
        return;

    uint64_t input_segment_index = gid / inner_dim_size;
    uint64_t segment_offset      = gid % inner_dim_size;

    uint64_t output_segment_index = R_GET_VAL(segment_ids, input_segment_index);

    if(output_segment_index < 0 || output_segment_index >= output_outer_dim_size)
    {
        return;
    }

    uint64_t output_index = output_segment_index * inner_dim_size + segment_offset;
    SET_VAL(input_grad, gid, GET_VAL(output_grad, output_index));
}

extern "C" __global__ void UnsortedSegmentSumBwd(const D_TYPE* __restrict__ output_grad,
                                                 D_TYPE* __restrict__ input_grad,
                                                 const uint64_t* __restrict__ segment_ids,
                                                 uint64_t num_segments,
                                                 tensor_view_t<4> output_grad_tv,
                                                 tensor_view_t<4> input_grad_tv,
                                                 tensor_view_t<4> segment_ids_tv)
{
    unsortedsegmentsumFwdContiguous<D_TYPE>(output_grad,
                                            input_grad,
                                            segment_ids,
                                            num_segments,
                                            output_grad_tv,
                                            input_grad_tv,
                                            segment_ids_tv);
}
