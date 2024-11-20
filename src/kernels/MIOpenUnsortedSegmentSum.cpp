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
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "hip_atomic.hpp"
#include "float_types.h"

template <typename TIO, typename TS>
__device__ void unsortedSegmentSumFwd(const TIO* __restrict__ input,
                                      TIO* __restrict__ output,
                                      const TS* __restrict__ segment_ids,
                                      const uint64_t N,
                                      const uint64_t inner_dim_size,
                                      const uint64_t num_segments)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= N)
        return;

    uint64_t input_segment_index  = gid / inner_dim_size;
    uint64_t segment_offset       = gid % inner_dim_size;
    uint64_t output_segment_index = static_cast<uint64_t>(segment_ids[input_segment_index]);

    if(output_segment_index >= num_segments)
        return;

    uint64_t output_index = output_segment_index * inner_dim_size + segment_offset;
    atomic_add_g(output + output_index, CVT_FLOAT2ACCUM(input[gid]));
}

extern "C" __global__ void UnsortedSegmentSumFwd(const D_TYPE* __restrict__ input,
                                                 D_TYPE* __restrict__ output,
                                                 const SEG_TYPE* __restrict__ segment_ids,
                                                 const uint64_t N,
                                                 const uint64_t inner_dim_size,
                                                 const uint64_t num_segments)
{
    unsortedSegmentSumFwd<D_TYPE, SEG_TYPE>(
        input, output, segment_ids, N, inner_dim_size, num_segments);
}

template <typename TIO, typename TS>
__device__ void unsortedSegmentSumBwd(const TIO* __restrict__ output_grad,
                                      TIO* __restrict__ input_grad,
                                      const TS* __restrict__ segment_ids,
                                      const uint64_t N,
                                      const uint64_t inner_dim_size,
                                      const uint64_t num_segments)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= N)
        return;

    uint64_t input_segment_index  = gid / inner_dim_size;
    uint64_t segment_offset       = gid % inner_dim_size;
    uint64_t output_segment_index = static_cast<uint64_t>(segment_ids[input_segment_index]);

    if(output_segment_index >= num_segments)
        return;

    uint64_t output_index = output_segment_index * inner_dim_size + segment_offset;
    input_grad[gid]       = output_grad[output_index];
}

extern "C" __global__ void UnsortedSegmentSumBwd(const D_TYPE* __restrict__ output_grad,
                                                 D_TYPE* __restrict__ input_grad,
                                                 const SEG_TYPE* __restrict__ segment_ids,
                                                 const uint64_t N,
                                                 const uint64_t inner_dim_size,
                                                 const uint64_t num_segments)
{
    unsortedSegmentSumBwd<D_TYPE, SEG_TYPE>(
        output_grad, input_grad, segment_ids, N, inner_dim_size, num_segments);
}
