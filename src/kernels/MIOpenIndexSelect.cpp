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

template <typename TIO>
__device__ void IndexSelectForwardImpl(const TIO* input,
                                       const size_t* indices,
                                       TIO* output,
                                       size_t dim,
                                       tensor_view_t<5> input_tv,
                                       tensor_view_t<1> indices_tv,
                                       tensor_view_t<5> output_tv)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    tensor_layout_t<5> output_layout{output_tv, gid};

    if(output_layout.layout[0] >= output_tv.size[0])
        return;

    auto max_idx                    = input_tv.size[dim];
    tensor_layout_t<5> input_layout = output_layout;
    tensor_layout_t<1> indices_layout{output_layout.layout[dim]};
    input_layout.layout[dim] = indices[indices_tv.get_tensor_view_idx(indices_layout)];

    if(input_layout.layout[dim] < max_idx)
    {
        output[output_tv.get_tensor_view_idx(output_layout)] =
            input[input_tv.get_tensor_view_idx(input_layout)];
    }
    else
    {
        output[output_tv.get_tensor_view_idx(output_layout)] = 0;
    }
}

extern "C" __global__ void IndexSelectForward(const IO_TYPE* input,
                                              const size_t* indices,
                                              IO_TYPE* output,
                                              size_t dim,
                                              tensor_view_t<5> input_tv,
                                              tensor_view_t<1> indices_tv,
                                              tensor_view_t<5> output_tv)
{
    IndexSelectForwardImpl<IO_TYPE>(input, indices, output, dim, input_tv, indices_tv, output_tv);
}

template <typename TIO>
__device__ void IndexSelectContiguousForwardImpl(const IO_TYPE* input,
                                                 const size_t* indices,
                                                 IO_TYPE* output,
                                                 size_t dim,
                                                 tensor_view_t<5> input_tv,
                                                 tensor_view_t<5> output_tv)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    tensor_layout_t<5> output_layout{output_tv, gid};

    if(output_layout.layout[0] >= output_tv.size[0])
        return;

    auto max_idx                    = input_tv.size[dim];
    tensor_layout_t<5> input_layout = output_layout;
    input_layout.layout[dim]        = indices[output_layout.layout[dim]];

    if(input_layout.layout[dim] < max_idx)
    {
        output[gid] = input[input_tv.get_tensor_view_idx(input_layout)];
    }
    else
    {
        output[gid] = 0;
    }
}

extern "C" __global__ void IndexSelectContiguousForward(const IO_TYPE* input,
                                                        const size_t* indices,
                                                        IO_TYPE* output,
                                                        size_t dim,
                                                        tensor_view_t<5> input_tv,
                                                        tensor_view_t<5> output_tv)
{
    IndexSelectContiguousForwardImpl<IO_TYPE>(input, indices, output, dim, input_tv, output_tv);
}

template <typename TIO>
__device__ void IndexSelectBackwardImpl(const TIO* outGrad,
                                        const size_t* indices,
                                        TIO* inGrad,
                                        size_t dim,
                                        size_t output_numel,
                                        tensor_view_t<5> outGrad_tv,
                                        tensor_view_t<1> indices_tv,
                                        tensor_view_t<5> inGrad_tv)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= output_numel)
        return;

    auto max_idx = inGrad_tv.size[dim];
    tensor_layout_t<5> outGrad_layout{outGrad_tv, gid};
    tensor_layout_t<1> indices_layout{outGrad_layout.layout[dim]};
    auto idx = indices[indices_tv.get_tensor_view_idx(indices_layout)];
    if(idx < max_idx)
    {
        tensor_layout_t<5> inGrad_layout = outGrad_layout;
        inGrad_layout.layout[dim]        = idx;
        FLOAT_ACCUM val = CVT_FLOAT2ACCUM(outGrad[outGrad_tv.get_tensor_view_idx(outGrad_layout)]);
        atomic_add_g(&inGrad[inGrad_tv.get_tensor_view_idx(inGrad_layout)], val);
    }
}

extern "C" __global__ void IndexSelectBackward(const IO_TYPE* outGrad,
                                               const size_t* indices,
                                               IO_TYPE* inGrad,
                                               size_t dim,
                                               size_t output_numel,
                                               tensor_view_t<5> outGrad_tv,
                                               tensor_view_t<1> indices_tv,
                                               tensor_view_t<5> inGrad_tv)
{
    IndexSelectBackwardImpl<IO_TYPE>(
        outGrad, indices, inGrad, dim, output_numel, outGrad_tv, indices_tv, inGrad_tv);
}
